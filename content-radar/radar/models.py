"""Shared record types.

Plain dataclasses with explicit ``to_dict`` / ``from_dict`` so every artefact on
disk is readable JSONL — diffable in a PR and greppable from a phone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

Lane = str  # "ai" | "regulated" | "genfinance"


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class Item:
    """One article, post or paper as fetched from a single source."""

    url: str
    canonical_url: str
    title: str
    source: str
    lane: Lane
    published_at: str | None = None
    points: int = 0
    summary: str = ""
    fetched_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Item:
        known = {f for f in cls.__slots__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(slots=True)
class Cluster:
    """A group of items telling the same story."""

    cluster_id: str
    title: str
    lane: Lane
    urls: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    first_seen: str = field(default_factory=utcnow)

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def max_points(self) -> int:
        return max((i.points for i in self.items), default=0)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "title": self.title,
            "lane": self.lane,
            "urls": self.urls,
            "sources": self.sources,
            "size": self.size,
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "first_seen": self.first_seen,
            "summary": next((i.summary for i in self.items if i.summary), ""),
        }
