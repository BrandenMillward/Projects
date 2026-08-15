from datetime import UTC, datetime, timedelta

from radar.cluster import cluster_items
from radar.models import Item
from radar.normalise import canonical_url, url_hash
from radar.score import WEIGHTS, score_all, score_cluster

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
WEIGHTS_MAP = {"A": 0.9, "B": 0.5, "C": 0.2}


def make(title, url, source, *, lane="ai", points=0, age_hours=1.0, summary=""):
    return Item(
        url=url,
        canonical_url=canonical_url(url),
        title=title,
        source=source,
        lane=lane,
        published_at=(NOW - timedelta(hours=age_hours)).isoformat(),
        points=points,
        summary=summary,
    )


def one(items):
    return score_cluster(
        cluster_items(items)[0], source_weights=WEIGHTS_MAP, seen_hashes=set(), now=NOW
    )


class TestWeights:
    def test_weights_sum_to_one_so_scores_are_comparable(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_score_is_bounded(self):
        c = one([make("Multi-agent orchestration guardrails", "https://a.com/1", "A", points=900)])
        assert 0.0 <= c.score <= 1.0


class TestMonotonicity:
    def test_fresher_scores_higher(self):
        fresh = one([make("Same story", "https://a.com/1", "A", age_hours=1)])
        stale = one([make("Same story", "https://a.com/1", "A", age_hours=240)])
        assert fresh.score > stale.score

    def test_trusted_source_scores_higher(self):
        good = one([make("Same story", "https://a.com/1", "A")])
        weak = one([make("Same story", "https://a.com/1", "C")])
        assert good.score > weak.score

    def test_more_engagement_scores_higher(self):
        hot = one([make("Same story", "https://a.com/1", "B", points=800)])
        cold = one([make("Same story", "https://a.com/1", "B", points=0)])
        assert hot.score > cold.score

    def test_more_corroboration_scores_higher(self):
        many = one(
            [
                make("Agent framework released", "https://a.com/1", "B"),
                make("Agent framework released today", "https://b.com/2", "B"),
                make("Agent framework is released", "https://c.com/3", "B"),
            ]
        )
        single = one([make("Agent framework released", "https://a.com/1", "B")])
        assert many.score > single.score


class TestAuthority:
    def test_authority_terms_beat_generic_news(self):
        expert = one(
            [
                make(
                    "Guardrails for multi-agent orchestration in production",
                    "https://a.com/1",
                    "B",
                    summary="evaluation and tool use",
                )
            ]
        )
        generic = one([make("A robot danced at a trade show", "https://a.com/2", "B")])
        assert expert.score > generic.score

    def test_authority_is_lane_specific(self):
        # Pension terms score in genfinance, not in the AI lane.
        fin = one([make("Workplace pension auto-enrolment reform", "https://a.com/1", "B",
                        lane="genfinance")])
        same_in_ai = one([make("Workplace pension auto-enrolment reform", "https://a.com/1", "B",
                               lane="ai")])
        assert fin.components["authority"] > same_in_ai.components["authority"]


class TestSeenPenalty:
    def test_already_seen_is_heavily_demoted(self):
        item = make("Multi-agent guardrails shipped", "https://a.com/1", "A", points=500)
        unseen = one([item])
        seen = score_cluster(
            cluster_items([item])[0],
            source_weights=WEIGHTS_MAP,
            seen_hashes={url_hash(item.url)},
            now=NOW,
        )
        assert seen.score < unseen.score
        assert "seen_penalty" in seen.components

    def test_seen_story_ranks_below_a_weaker_fresh_one(self):
        strong_seen = make("Multi-agent guardrails shipped", "https://a.com/1", "A", points=500)
        weak_fresh = make("Minor AI evaluation note", "https://b.com/2", "C")
        ranked = score_all(
            cluster_items([strong_seen, weak_fresh]),
            source_weights=WEIGHTS_MAP,
            seen_hashes={url_hash(strong_seen.url)},
            now=NOW,
        )
        assert ranked[0].title == weak_fresh.title


class TestOrdering:
    def test_score_all_returns_highest_first(self):
        ranked = score_all(
            cluster_items(
                [
                    make("Trivial gadget review", "https://c.com/1", "C", age_hours=200),
                    make("Guardrails for agent orchestration", "https://a.com/2", "A", points=400),
                ]
            ),
            source_weights=WEIGHTS_MAP,
            seen_hashes=set(),
            now=NOW,
        )
        assert [c.score for c in ranked] == sorted([c.score for c in ranked], reverse=True)

    def test_undated_items_do_not_crash(self):
        item = make("No date", "https://a.com/1", "A")
        item.published_at = None
        assert one([item]).score >= 0.0
