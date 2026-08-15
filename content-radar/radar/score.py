"""Deterministic heuristic scoring.

This is arithmetic, not judgement. It exists to get from ~400 raw items to a
shortlist of ~30 worth a human-quality editorial pass. The editorial layer runs
in Claude (see .claude/commands/radar.md) and is where taste is applied — this
module only decides what is worth looking at.

Keeping it deterministic means it is unit-testable and its output is stable for
the same input, which matters when debugging "why did that topic surface?".
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime

from .models import Cluster

# Relative weight of each component in the final score. They sum to 1.0 so the
# result is always in [0, 1] and directly comparable across runs.
WEIGHTS = {
    "recency": 0.22,
    "source": 0.18,
    "engagement": 0.15,
    "authority": 0.30,
    "corroboration": 0.15,
}

RECENCY_HALF_LIFE_HOURS = 48.0
ENGAGEMENT_SATURATION = 500.0  # HN points at which engagement scores ~1.0
CORROBORATION_SATURATION = 8.0  # sources covering a story at which it scores ~1.0

# Penalty multiplier applied to a story already recorded in the seen-set. Not
# zero: a genuinely developing story should be able to resurface if it has
# gained real corroboration since.
SEEN_PENALTY = 0.15

# Authority tuning. Two title hits is a full score — a headline naming two of
# Branden's terms is unambiguously his territory. Body text needs eight, and is
# capped, so a long abstract stuffed with jargon cannot buy its way to the top.
TITLE_HITS_FOR_FULL = 2.0
BODY_HITS_FOR_FULL = 8.0
AUTHORITY_BODY_CHARS = 400

# The terms that mark a story as something Branden has standing to write about,
# rather than generic AI news. This is the single highest-weighted component,
# because "is this newsworthy" is abundant and "can he uniquely speak to it" is
# the scarce signal.
#
# Terms are matched as prefixes anchored to a word boundary, so "regulat" covers
# regulation and regulatory. Keep only the SHORTEST useful prefix of a family —
# listing both "eval" and "evaluation" makes one word score twice and inflates
# everything containing it.
AUTHORITY_TERMS = {
    "ai": [
        "multi-agent", "multi agent", "agent network", "orchestration", "agentic",
        "guardrail", "eval", "prompt", "context window", "context engineer",
        "coding agent", "tool use", "rag", "retrieval", "fine-tun",
        "inference cost", "model context protocol", "mcp", "agent skill",
        "upskilling", "adoption", "enterprise ai",
    ],
    "regulated": [
        "explainab", "interpretab", "xai", "model risk", "governance", "compliance",
        "regulat", "audit", "fca", "ico", "eu ai act", "gdpr", "bias", "fairness",
        "transparency", "accountab", "assurance", "supervis", "financial services",
        "banking", "insurance",
    ],
    "genfinance": [
        "student loan", "student finance", "pension", "auto-enrol", "auto enrol",
        "isa", "lisa", "mortgage", "first-time buyer", "cost of living",
        "interest rate", "inflation", "salary", "graduate", "workplace pension",
        "retirement", "savings", "tax threshold",
    ],
}


def _recency(cluster: Cluster, now: datetime) -> float:
    """Exponential decay with a 48-hour half-life, from the newest item."""
    stamps = []
    for item in cluster.items:
        if not item.published_at:
            continue
        try:
            parsed = datetime.fromisoformat(item.published_at)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        stamps.append(parsed)

    if not stamps:
        return 0.4  # unknown date: middling, neither promoted nor buried

    age_hours = max((now - max(stamps)).total_seconds() / 3600.0, 0.0)
    return 0.5 ** (age_hours / RECENCY_HALF_LIFE_HOURS)


def _source(cluster: Cluster, weights: dict[str, float]) -> float:
    """Mean trust weight of the sources covering the story."""
    if not cluster.sources:
        return 0.5
    values = [weights.get(s, 0.5) for s in cluster.sources]
    return sum(values) / len(values)


def _engagement(cluster: Cluster) -> float:
    """Log-scaled HN points. Most items have none, which is fine — this is one
    component of five, not a gate."""
    points = cluster.max_points
    if points <= 0:
        return 0.0
    return min(math.log1p(points) / math.log1p(ENGAGEMENT_SATURATION), 1.0)


def _term_in(text: str, term: str) -> bool:
    """Prefix match anchored to a word boundary.

    Anchoring matters: without it "rag" matches "storage" and "mcp" matches
    nothing useful, while "eval" would also fire inside "evaluation" for a
    second time if both were listed.
    """
    return re.search(r"\b" + re.escape(term), text) is not None


def _authority(cluster: Cluster) -> float:
    """How strongly this story sits in Branden's territory.

    A term in the *headline* means the story is about that thing. The same term
    buried in an abstract means the author happened to use the word — an arXiv
    abstract is long and dense with exactly this vocabulary, so counting body
    text equally let every paper saturate the component regardless of whether
    the topic was remotely relevant. Title hits are therefore worth four times a
    body hit, and body text is capped so length alone cannot buy a score.
    """
    terms = AUTHORITY_TERMS.get(cluster.lane, [])
    if not terms:
        return 0.0

    title = cluster.title.lower()
    body = " ".join(i.summary for i in cluster.items).lower()[:AUTHORITY_BODY_CHARS]

    title_hits = {t for t in terms if _term_in(title, t)}
    # Don't pay twice for a term that already scored in the title.
    body_hits = {t for t in terms if _term_in(body, t)} - title_hits

    score = len(title_hits) / TITLE_HITS_FOR_FULL + len(body_hits) / BODY_HITS_FOR_FULL
    return min(score, 1.0)


def _corroboration(cluster: Cluster) -> float:
    if cluster.size <= 1:
        return 0.0
    return min(math.log1p(cluster.size - 1) / math.log1p(CORROBORATION_SATURATION), 1.0)


def score_cluster(
    cluster: Cluster,
    *,
    source_weights: dict[str, float],
    seen_hashes: set[str],
    now: datetime | None = None,
) -> Cluster:
    """Attach a score and its components to *cluster*, and return it."""
    now = now or datetime.now(UTC)

    components = {
        "recency": _recency(cluster, now),
        "source": _source(cluster, source_weights),
        "engagement": _engagement(cluster),
        "authority": _authority(cluster),
        "corroboration": _corroboration(cluster),
    }

    total = sum(WEIGHTS[name] * value for name, value in components.items())

    from .normalise import url_hash

    if any(url_hash(u) in seen_hashes for u in cluster.urls):
        total *= SEEN_PENALTY
        components["seen_penalty"] = SEEN_PENALTY

    cluster.score = total
    cluster.components = components
    return cluster


def score_all(
    clusters: list[Cluster],
    *,
    source_weights: dict[str, float],
    seen_hashes: set[str],
    now: datetime | None = None,
) -> list[Cluster]:
    """Score every cluster and return them highest-first."""
    now = now or datetime.now(UTC)
    scored = [
        score_cluster(c, source_weights=source_weights, seen_hashes=seen_hashes, now=now)
        for c in clusters
    ]
    return sorted(scored, key=lambda c: c.score, reverse=True)
