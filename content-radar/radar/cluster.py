"""Collapse items covering the same story into clusters.

Two passes: exact canonical-URL match, then fuzzy title similarity. Cluster size
becomes a corroboration signal in scoring — eight outlets covering one
announcement is a stronger candidate than one blog covering it alone.
"""

from __future__ import annotations

import hashlib

from rapidfuzz import fuzz

from .models import Cluster, Item
from .normalise import normalise_title

# token_set_ratio ignores word order and duplicated words, so "OpenAI ships X"
# and "X shipped by OpenAI" score high. 85 was chosen to sit above routine
# headline variation and below genuinely different stories on one topic.
TITLE_THRESHOLD = 85


def _cluster_id(title: str, url: str) -> str:
    seed = f"{normalise_title(title)}|{url}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def cluster_items(items: list[Item], threshold: int = TITLE_THRESHOLD) -> list[Cluster]:
    """Group *items* into clusters, most-corroborated first.

    Ordering is deterministic: items are processed newest-first so the earliest
    strong headline becomes the cluster's representative title.
    """
    ordered = sorted(items, key=lambda i: (i.published_at or "", i.title), reverse=True)

    by_url: dict[str, Cluster] = {}
    clusters: list[Cluster] = []

    for item in ordered:
        # Pass 1 — same canonical URL is the same story, no question.
        existing = by_url.get(item.canonical_url)
        if existing is not None:
            _absorb(existing, item)
            continue

        # Pass 2 — near-identical headline.
        norm = normalise_title(item.title)
        match = None
        if norm:
            for candidate in clusters:
                if candidate.lane != item.lane:
                    continue
                if fuzz.token_set_ratio(norm, normalise_title(candidate.title)) >= threshold:
                    match = candidate
                    break

        if match is not None:
            _absorb(match, item)
            by_url[item.canonical_url] = match
            continue

        fresh = Cluster(
            cluster_id=_cluster_id(item.title, item.canonical_url),
            title=item.title,
            lane=item.lane,
        )
        _absorb(fresh, item)
        clusters.append(fresh)
        by_url[item.canonical_url] = fresh

    return sorted(clusters, key=lambda c: c.size, reverse=True)


def _absorb(cluster: Cluster, item: Item) -> None:
    cluster.items.append(item)
    if item.canonical_url not in cluster.urls:
        cluster.urls.append(item.canonical_url)
    if item.source not in cluster.sources:
        cluster.sources.append(item.source)
