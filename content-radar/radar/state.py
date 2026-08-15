"""Append-only JSONL state.

State is the product: what was suggested, picked, shipped and killed is the
thing that stops week five re-pitching what was rejected in week two. It lives
in git rather than a database precisely so it is diffable, greppable and
readable on GitHub from a phone.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .models import Item, utcnow
from .normalise import url_hash

STATE_DIR = Path("state")
ITEMS = STATE_DIR / "items.jsonl"
SEEN = STATE_DIR / "seen.jsonl"
DECISIONS = STATE_DIR / "decisions.jsonl"

SEEN_RETENTION_DAYS = 180
KILLED_SUPPRESSION_DAYS = 90
# Raw items age out fast — scoring only looks at a recent window, and this file
# is committed to git.
ITEM_RETENTION_DAYS = 45


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A truncated final line (interrupted write) must not break a run.
            continue
    return rows


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_items(items: list[Item], path: Path = ITEMS) -> int:
    _append_jsonl(path, [i.to_dict() for i in items])
    return len(items)


def read_items(path: Path = ITEMS, since_days: int | None = None) -> list[Item]:
    rows = _read_jsonl(path)
    items = [Item.from_dict(r) for r in rows]
    if since_days is None:
        return items

    cutoff = datetime.now(UTC) - timedelta(days=since_days)
    kept = []
    for item in items:
        stamp = item.published_at or item.fetched_at
        try:
            parsed = datetime.fromisoformat(stamp)
        except (ValueError, TypeError):
            kept.append(item)  # undated items are kept; scoring handles them
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        if parsed >= cutoff:
            kept.append(item)
    return kept


def prune_items(path: Path = ITEMS, days: int = ITEM_RETENTION_DAYS) -> int:
    """Drop items older than *days* from the raw item log.

    Without this the file grows forever, and it is committed to git — a weekly
    job would add roughly 20k items a year to a file nobody reads directly.
    Scoring only ever looks at a recent window, so old rows earn nothing. The
    seen-set is what remembers across time, and it is pruned separately.
    """
    rows = _read_jsonl(path)
    if not rows:
        return 0

    cutoff = datetime.now(UTC) - timedelta(days=days)
    kept = []
    for row in rows:
        stamp = row.get("published_at") or row.get("fetched_at") or ""
        try:
            at = datetime.fromisoformat(stamp)
        except (ValueError, TypeError):
            kept.append(row)  # undated rows are kept rather than silently lost
            continue
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        if at >= cutoff:
            kept.append(row)

    removed = len(rows) - len(kept)
    if removed:
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8"
        )
    return removed


def seen_hashes(path: Path = SEEN) -> set[str]:
    """URL hashes already surfaced in a previous brief."""
    return {r["hash"] for r in _read_jsonl(path) if "hash" in r}


def mark_seen(urls: list[str], path: Path = SEEN) -> int:
    """Record URLs as surfaced. Called after a brief is written, not after a
    fetch — an item is only 'seen' once it has actually been shown."""
    existing = seen_hashes(path)
    rows = []
    for url in urls:
        h = url_hash(url)
        if h in existing:
            continue
        existing.add(h)
        rows.append({"hash": h, "url": url, "at": utcnow()})
    _append_jsonl(path, rows)
    return len(rows)


def prune_seen(path: Path = SEEN, days: int = SEEN_RETENTION_DAYS) -> int:
    """Rewrite the seen-set without entries older than *days*. Keeps the file
    small enough to stay comfortable in git."""
    rows = _read_jsonl(path)
    if not rows:
        return 0

    cutoff = datetime.now(UTC) - timedelta(days=days)
    kept = []
    for row in rows:
        try:
            at = datetime.fromisoformat(row.get("at", ""))
        except ValueError:
            kept.append(row)
            continue
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        if at >= cutoff:
            kept.append(row)

    removed = len(rows) - len(kept)
    if removed:
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8"
        )
    return removed


def record_decision(
    cluster_id: str, title: str, state: str, reason: str = "", path: Path = DECISIONS
) -> None:
    """Record a decision. *state* is one of suggested / picked / shipped / killed."""
    valid = {"suggested", "picked", "shipped", "killed"}
    if state not in valid:
        raise ValueError(f"state must be one of {sorted(valid)}, got {state!r}")
    _append_jsonl(
        path,
        [
            {
                "at": utcnow(),
                "cluster_id": cluster_id,
                "title": title,
                "state": state,
                "reason": reason,
            }
        ],
    )


def suppressed_titles(path: Path = DECISIONS) -> dict[str, str]:
    """Titles that should not be re-suggested, mapped to why.

    Shipped topics are suppressed permanently; killed topics for 90 days, so a
    rejected idea can come back if it genuinely becomes newsworthy later.
    """
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=KILLED_SUPPRESSION_DAYS)
    out: dict[str, str] = {}

    for row in _read_jsonl(path):
        state = row.get("state")
        title = row.get("title", "")
        if not title:
            continue
        if state == "shipped":
            out[title] = "already published"
        elif state == "killed":
            try:
                at = datetime.fromisoformat(row.get("at", ""))
            except ValueError:
                continue
            if at.tzinfo is None:
                at = at.replace(tzinfo=UTC)
            if at >= cutoff:
                reason = row.get("reason") or "previously rejected"
                out[title] = f"killed {at.date()} — {reason}"
    return out
