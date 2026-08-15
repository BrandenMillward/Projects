"""Regression tests built from the first live run (2026-08-14).

The first real shortlist put three arXiv papers in the top four — clinical AI
reasoning, humanoid motion tracking, and RAG over Italian parliamentary
proceedings — each scoring authority 1.0, the maximum on the most heavily
weighted component. None is remotely something Branden would write about.

The cause was that `_authority` counted term hits across title *and* a 600-char
summary with a saturation point of three. Academic abstracts are long and dense
with exactly that vocabulary, so every paper maxed the component. It was
measuring vocabulary density, not relevance.

Titles and summaries below are the real ones from that run.
"""

from datetime import UTC, datetime, timedelta

from radar.cluster import cluster_items
from radar.models import Item
from radar.normalise import canonical_url
from radar.score import score_all, score_cluster

NOW = datetime(2026, 8, 14, 17, 17, tzinfo=UTC)
WEIGHTS_MAP = {"arXiv cs.AI": 0.35, "arXiv cs.CL": 0.3, "InfoQ AI": 0.6, "FCA News": 0.95}

# --- real records from the 2026-08-14 run -----------------------------------

HUMANTRACKER = (
    "HumanTracker: Towards Comprehensive and Human-Aligned Motion Tracking Benchmark",
    "Humanoid motion tracking is central to teleoperation and whole-body imitation, yet "
    "evaluation often disagrees with what people perceive in videos. Kinematic errors "
    "average per-frame pose differences but miss the physical artifacts that matter most, "
    "particularly unstable support and incorrect contacts such as foot skating and mistimed "
    "touch-downs. Meanwhile, widely used test suites are small and lack the diversity needed "
    "to stress contact-rich, long-horizon behaviors. We introduce HumanTracker to make "
    "humanoid tracking evaluation both perceptually aligned and scalable.",
)

ITALIAN_RAG = (
    "Who Speaks Matters: Authority-Aware Multi-View RAG over Italian Parliamentary Proceedings",
    "Parliamentary proceedings are a primary record of democratic deliberation, yet their "
    "volume and fragmentation make multi-perspective access difficult for citizens, "
    "journalists, and researchers. Applying Retrieval-Augmented Generation (RAG) to "
    "parliamentary transcripts introduces three specific risks: dominance of the most "
    "frequent speakers, inability to weight speakers according to topical expertise, and "
    "citation misattribution in politically sensitive text.",
)

CONTEXT_ENGINEERING = (
    "The Right 300 Tokens Beat 100k Noisy Ones: The Architecture of Context Engineering",
    "Baruch Sadogursky and Patrick Debois discuss why coding agents fail due to bloated "
    "context windows and stuffed prompts. They explain practical context engineering fixes, "
    "including lazy-loaded skills, versioned context artifacts, externalized memory banks, "
    "and LLM-as-a-judge evals.",
)


def item(title, summary, source, lane="ai", hours=6):
    url = f"https://example.com/{abs(hash(title))}"
    return Item(
        url=url,
        canonical_url=canonical_url(url),
        title=title,
        source=source,
        lane=lane,
        published_at=(NOW - timedelta(hours=hours)).isoformat(),
        summary=summary,
    )


def score_one(title, summary, source, lane="ai", hours=6):
    it = item(title, summary, source, lane, hours)
    return score_cluster(
        cluster_items([it])[0], source_weights=WEIGHTS_MAP, seen_hashes=set(), now=NOW
    )


class TestAbstractsNoLongerSaturate:
    def test_irrelevant_paper_does_not_max_authority(self):
        """HumanTracker is about humanoid robots. It previously scored 1.0."""
        c = score_one(*HUMANTRACKER, "arXiv cs.AI")
        assert c.components["authority"] < 0.5, c.components

    def test_tangential_paper_does_not_max_authority(self):
        """Italian parliamentary RAG uses the vocabulary but isn't his subject."""
        c = score_one(*ITALIAN_RAG, "arXiv cs.AI")
        assert c.components["authority"] < 1.0, c.components

    def test_a_genuinely_relevant_headline_still_scores_well(self):
        """The fix must not simply suppress everything — context engineering for
        coding agents is squarely his territory, and it says so in the title."""
        c = score_one(*CONTEXT_ENGINEERING, "InfoQ AI")
        assert c.components["authority"] >= 0.5, c.components


class TestRankingAgainstRealData:
    def test_relevant_item_outranks_the_irrelevant_paper(self):
        """The headline test for this whole component: on the real run,
        HumanTracker outranked nothing useful — it just shouldn't be near the
        top. Context engineering should beat it outright."""
        ranked = score_all(
            cluster_items(
                [
                    item(*HUMANTRACKER, "arXiv cs.AI"),
                    item(*ITALIAN_RAG, "arXiv cs.AI"),
                    item(*CONTEXT_ENGINEERING, "InfoQ AI"),
                ]
            ),
            source_weights=WEIGHTS_MAP,
            seen_hashes=set(),
            now=NOW,
        )
        assert ranked[0].title == CONTEXT_ENGINEERING[0], [
            (round(c.score, 3), c.title[:50]) for c in ranked
        ]


class TestTermMatching:
    def test_eval_is_not_counted_twice(self):
        """'eval' and 'evaluation' were both in the term list, so one word
        scored twice and inflated anything containing it."""
        c = score_one("Evaluation of a thing", "evaluation evaluation", "arXiv cs.AI")
        # One distinct term family, in the title only.
        assert c.components["authority"] == 0.5, c.components

    def test_terms_anchor_to_a_word_boundary(self):
        """Unanchored, 'rag' matches 'storage' and 'average'."""
        c = score_one("Average storage costs rise", "average storage", "arXiv cs.AI")
        assert c.components["authority"] == 0.0, c.components

    def test_prefix_families_still_match_their_variants(self):
        """'regulat' must still cover regulation and regulatory."""
        c = score_one("New regulatory guidance", "", "FCA News", lane="regulated")
        assert c.components["authority"] > 0.0, c.components

    def test_body_length_alone_cannot_buy_a_score(self):
        """A long body full of terms must still score below a title match."""
        stuffed = " ".join(["orchestration guardrail prompt retrieval agentic"] * 40)
        body_only = score_one("An unrelated headline about cats", stuffed, "arXiv cs.AI")
        titled = score_one("Guardrails for agent orchestration", "", "arXiv cs.AI")
        assert titled.components["authority"] > body_only.components["authority"], (
            titled.components,
            body_only.components,
        )
