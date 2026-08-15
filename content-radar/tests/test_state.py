import json
from datetime import UTC, datetime, timedelta

from radar.models import Item
from radar.normalise import canonical_url, url_hash
from radar.state import (
    mark_seen,
    prune_items,
    prune_seen,
    read_items,
    record_decision,
    seen_hashes,
    suppressed_titles,
    write_items,
)


def item(title, url, age_days=0):
    return Item(
        url=url,
        canonical_url=canonical_url(url),
        title=title,
        source="S",
        lane="ai",
        published_at=(datetime.now(UTC) - timedelta(days=age_days)).isoformat(timespec="seconds"),
    )


class TestItems:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "items.jsonl"
        write_items([item("A", "https://a.com/1"), item("B", "https://b.com/2")], path)
        assert [i.title for i in read_items(path)] == ["A", "B"]

    def test_since_days_filters(self, tmp_path):
        path = tmp_path / "items.jsonl"
        write_items([item("fresh", "https://a.com/1", 1), item("old", "https://b.com/2", 40)], path)
        assert [i.title for i in read_items(path, since_days=7)] == ["fresh"]

    def test_a_truncated_final_line_does_not_break_a_run(self, tmp_path):
        path = tmp_path / "items.jsonl"
        write_items([item("good", "https://a.com/1")], path)
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"url": "https://b.com/2", "titl')  # interrupted write
        assert [i.title for i in read_items(path)] == ["good"]


class TestPruneItems:
    def test_old_items_are_dropped(self, tmp_path):
        path = tmp_path / "items.jsonl"
        write_items(
            [item("fresh", "https://a.com/1", 2), item("ancient", "https://b.com/2", 200)], path
        )
        assert prune_items(path, days=45) == 1
        assert [i.title for i in read_items(path)] == ["fresh"]

    def test_undated_rows_are_kept_rather_than_lost(self, tmp_path):
        path = tmp_path / "items.jsonl"
        path.write_text(json.dumps({"url": "https://a.com/1", "title": "no dates"}) + "\n")
        assert prune_items(path, days=1) == 0

    def test_pruning_an_absent_file_is_safe(self, tmp_path):
        assert prune_items(tmp_path / "nope.jsonl") == 0


class TestSeen:
    def test_marking_is_idempotent(self, tmp_path):
        path = tmp_path / "seen.jsonl"
        urls = ["https://a.com/1?utm_source=x", "https://www.a.com/1"]
        # Both canonicalise to the same URL, so only one entry is written.
        assert mark_seen(urls, path) == 1
        assert mark_seen(urls, path) == 0
        assert seen_hashes(path) == {url_hash("https://a.com/1")}

    def test_expired_entries_are_pruned(self, tmp_path):
        path = tmp_path / "seen.jsonl"
        old = (datetime.now(UTC) - timedelta(days=400)).isoformat(timespec="seconds")
        path.write_text(json.dumps({"hash": "deadbeef", "url": "u", "at": old}) + "\n")
        assert prune_seen(path, days=180) == 1
        assert seen_hashes(path) == set()


class TestDecisions:
    def test_shipped_is_suppressed_permanently(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        record_decision("c1", "Published thing", "shipped", path=path)
        assert suppressed_titles(path)["Published thing"] == "already published"

    def test_killed_is_suppressed_with_its_reason(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        record_decision("c2", "Rejected thing", "killed", "no angle", path=path)
        assert "no angle" in suppressed_titles(path)["Rejected thing"]

    def test_an_old_kill_stops_suppressing_so_it_can_return(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        old = (datetime.now(UTC) - timedelta(days=200)).isoformat(timespec="seconds")
        row = {
            "at": old,
            "cluster_id": "c3",
            "title": "Old kill",
            "state": "killed",
            "reason": "",
        }
        path.write_text(json.dumps(row) + "\n")
        assert "Old kill" not in suppressed_titles(path)

    def test_merely_suggested_does_not_suppress(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        record_decision("c4", "Suggested thing", "suggested", path=path)
        assert suppressed_titles(path) == {}

    def test_an_invalid_state_is_rejected(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        try:
            record_decision("c5", "T", "maybe", path=path)
        except ValueError as exc:
            assert "state must be one of" in str(exc)
        else:
            raise AssertionError("expected ValueError")
