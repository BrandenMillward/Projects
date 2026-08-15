from radar.cluster import cluster_items
from radar.models import Item
from radar.normalise import canonical_url


def make(title: str, url: str, source: str, lane: str = "ai", points: int = 0) -> Item:
    return Item(
        url=url,
        canonical_url=canonical_url(url),
        title=title,
        source=source,
        lane=lane,
        published_at="2026-08-07T10:00:00+00:00",
        points=points,
    )


class TestClustering:
    def test_identical_urls_collapse(self):
        items = [
            make("EU AI Act guidance published", "https://e.com/a?utm_source=x", "A"),
            make("EU AI Act guidance published", "https://www.e.com/a", "B"),
        ]
        clusters = cluster_items(items)
        assert len(clusters) == 1
        assert clusters[0].size == 2

    def test_near_identical_titles_collapse_across_outlets(self):
        items = [
            make("OpenAI ships new agent framework", "https://a.com/1", "A"),
            make("OpenAI ships new agent framework for developers", "https://b.com/2", "B"),
            make("New agent framework shipped by OpenAI", "https://c.com/3", "C"),
        ]
        clusters = cluster_items(items)
        assert len(clusters) == 1, [c.title for c in clusters]
        assert clusters[0].size == 3
        assert set(clusters[0].sources) == {"A", "B", "C"}

    def test_genuinely_different_stories_stay_apart(self):
        items = [
            make("EU AI Act enforcement begins", "https://a.com/1", "A"),
            make("Pension auto-enrolment thresholds frozen", "https://b.com/2", "B", "genfinance"),
            make("Nvidia reports record datacentre revenue", "https://c.com/3", "C"),
        ]
        assert len(cluster_items(items)) == 3

    def test_lanes_never_merge(self):
        # Same headline, different lanes — must not be treated as one story.
        items = [
            make("Interest rates and AI", "https://a.com/1", "A", lane="ai"),
            make("Interest rates and AI", "https://b.com/2", "B", lane="genfinance"),
        ]
        assert len(cluster_items(items)) == 2

    def test_clusters_are_ordered_by_corroboration(self):
        items = [
            make("Lonely story", "https://x.com/1", "X"),
            make("Big story everyone covered", "https://a.com/1", "A"),
            make("Big story everyone covered today", "https://b.com/2", "B"),
            make("Big story that everyone covered", "https://c.com/3", "C"),
        ]
        clusters = cluster_items(items)
        assert clusters[0].size > clusters[-1].size

    def test_empty_input(self):
        assert cluster_items([]) == []

    def test_cluster_records_max_points(self):
        items = [
            make("Same story", "https://a.com/1", "A", points=10),
            make("Same story", "https://a.com/1", "B", points=340),
        ]
        assert cluster_items(items)[0].max_points == 340
