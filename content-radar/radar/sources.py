"""Source fetchers: RSS/Atom, Hacker News (Algolia), arXiv.

Every source here is free and keyless, and every one is a published feed or a
public read-only API — there is no HTML scraping of article bodies and no
paywall circumvention. One dead source must never fail a run, so each fetcher
catches its own errors and reports them.

Note on Reddit: unauthenticated ``.json`` endpoints were deprecated in May 2026
and now return 403, and OAuth registration is approval-gated. Reddit is
deliberately not a source here. See docs/project_spec.md.
"""

from __future__ import annotations

import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import feedparser
import httpx

from .models import Item
from .normalise import canonical_url

FEEDS_FILE = Path(__file__).parent / "feeds.toml"

USER_AGENT = (
    "content-radar/0.1 (+https://brandenmillward.github.io; weekly personal content brief)"
)
TIMEOUT = 10.0
MAX_WORKERS = 8

# A run that loses more than this fraction of its sources is treated as failed
# rather than thin — a broken fetcher and a quiet news week must not look alike.
SOURCE_FAILURE_THRESHOLD = 0.30


@dataclass(slots=True)
class SourceSpec:
    name: str
    kind: str  # "rss" | "hn" | "arxiv"
    lane: str
    weight: float = 0.5
    url: str = ""
    query: str = ""       # HN search term
    category: str = ""    # arXiv category, e.g. cs.AI
    min_points: int = 25  # HN score floor


@dataclass(slots=True)
class FetchReport:
    """What happened during a fetch, so the brief can be honest about coverage."""

    items: list[Item]
    ok: list[str]
    failed: dict[str, str]

    @property
    def total_sources(self) -> int:
        return len(self.ok) + len(self.failed)

    @property
    def failure_rate(self) -> float:
        if self.total_sources == 0:
            return 1.0
        return len(self.failed) / self.total_sources

    @property
    def degraded(self) -> bool:
        return self.failure_rate > SOURCE_FAILURE_THRESHOLD


def load_sources(path: Path = FEEDS_FILE) -> list[SourceSpec]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    specs = []
    for entry in data.get("source", []):
        specs.append(
            SourceSpec(
                name=entry["name"],
                kind=entry.get("kind", "rss"),
                lane=entry["lane"],
                weight=float(entry.get("weight", 0.5)),
                url=entry.get("url", ""),
                query=entry.get("query", ""),
                category=entry.get("category", ""),
                min_points=int(entry.get("min_points", 25)),
            )
        )
    return specs


def source_weights(specs: list[SourceSpec]) -> dict[str, float]:
    return {s.name: s.weight for s in specs}


def _iso(value) -> str | None:
    """feedparser hands back a time.struct_time; normalise it to ISO-8601 UTC."""
    if not value:
        return None
    try:
        return datetime(*value[:6], tzinfo=UTC).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def fetch_rss(spec: SourceSpec, client: httpx.Client) -> list[Item]:
    response = client.get(spec.url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    parsed = feedparser.parse(response.content)

    items = []
    for entry in parsed.entries:
        link = entry.get("link", "")
        title = (entry.get("title") or "").strip()
        if not link or not title:
            continue
        summary = (entry.get("summary") or entry.get("description") or "")[:600]
        items.append(
            Item(
                url=link,
                canonical_url=canonical_url(link),
                title=title,
                source=spec.name,
                lane=spec.lane,
                published_at=_iso(entry.get("published_parsed") or entry.get("updated_parsed")),
                summary=summary,
            )
        )
    return items


def fetch_hn(spec: SourceSpec, client: httpx.Client, since_days: int) -> list[Item]:
    """Hacker News via the Algolia index — free, no key, no auth."""
    cutoff = int((datetime.now(UTC) - timedelta(days=since_days)).timestamp())
    response = client.get(
        "https://hn.algolia.com/api/v1/search_by_date",
        params={
            "tags": "story",
            "query": spec.query,
            "numericFilters": f"created_at_i>{cutoff},points>{spec.min_points}",
            "hitsPerPage": 50,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    items = []
    for hit in response.json().get("hits", []):
        link = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        items.append(
            Item(
                url=link,
                canonical_url=canonical_url(link),
                title=title,
                source=spec.name,
                lane=spec.lane,
                published_at=hit.get("created_at"),
                points=int(hit.get("points") or 0),
                summary=(hit.get("story_text") or "")[:600],
            )
        )
    return items


def fetch_arxiv(spec: SourceSpec, client: httpx.Client) -> list[Item]:
    """arXiv Atom API — free, no key. Newest submissions in a category."""
    response = client.get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": f"cat:{spec.category}",
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": 40,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    parsed = feedparser.parse(response.content)

    items = []
    for entry in parsed.entries:
        link = entry.get("link", "")
        title = " ".join((entry.get("title") or "").split())
        if not link or not title:
            continue
        items.append(
            Item(
                url=link,
                canonical_url=canonical_url(link),
                title=title,
                source=spec.name,
                lane=spec.lane,
                published_at=_iso(entry.get("published_parsed")),
                summary=" ".join((entry.get("summary") or "").split())[:600],
            )
        )
    return items


def _fetch_one(spec: SourceSpec, since_days: int) -> list[Item]:
    with httpx.Client(follow_redirects=True) as client:
        if spec.kind == "hn":
            return fetch_hn(spec, client, since_days)
        if spec.kind == "arxiv":
            return fetch_arxiv(spec, client)
        return fetch_rss(spec, client)


def fetch_all(specs: list[SourceSpec], since_days: int = 7) -> FetchReport:
    """Fetch every source concurrently. A failing source is recorded, not raised."""
    items: list[Item] = []
    ok: list[str] = []
    failed: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, spec, since_days): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            try:
                fetched = future.result()
            except Exception as exc:  # noqa: BLE001 — one bad source must not end the run
                failed[spec.name] = f"{type(exc).__name__}: {exc}"[:200]
                continue
            items.extend(fetched)
            ok.append(spec.name)

    return FetchReport(items=items, ok=sorted(ok), failed=failed)
