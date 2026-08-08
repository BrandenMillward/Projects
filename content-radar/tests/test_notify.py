from pathlib import Path

from radar.notify import Channel, Payload, build_payload, deliver, parse_brief

TOPICS = [
    {"title": "Guardrails are the product", "pitch": "Why safety scaffolding is the deliverable.",
     "why_you": "You build these at Intent HQ.", "lane": "blog"},
    {"title": "The EU AI Act's Article 6 problem", "pitch": "Reclassification is the real risk.",
     "why_you": "Regulated-industry consulting background.", "lane": "blog"},
    {"title": "Student loan Plan 5 maths", "pitch": "What the threshold freeze costs you.",
     "why_you": "Speaks to your own cohort.", "lane": "video"},
]


class Recording(Channel):
    def __init__(self, name, ok=True, avail=True, raises=False):
        self.name, self._ok, self._avail, self._raises = name, ok, avail, raises
        self.calls = 0

    def available(self):
        return self._avail

    def send(self, payload):
        self.calls += 1
        if self._raises:
            raise RuntimeError("channel exploded")
        return self._ok


class TestBuildPayload:
    def test_push_is_within_the_mobile_limit(self):
        long_topics = [{"title": "A" * 400, "pitch": "", "why_you": "", "lane": "blog"}]
        payload = build_payload(Path("briefs/x.md"), long_topics)
        assert len(payload.push) <= 200

    def test_push_leads_with_count_and_lead_topic(self):
        payload = build_payload(Path("briefs/x.md"), TOPICS)
        assert "3 topics" in payload.push
        assert "Guardrails are the product" in payload.push

    def test_body_carries_every_topic_so_you_can_decide_without_opening_it(self):
        body = build_payload(Path("briefs/x.md"), TOPICS).body
        for topic in TOPICS:
            assert topic["title"] in body
            assert topic["pitch"] in body
            assert topic["why_you"] in body

    def test_failure_payload_is_flagged_and_explains_itself(self):
        payload = build_payload(Path("briefs/x.md"), [], failure="12 of 29 sources failed")
        assert payload.is_failure
        assert "12 of 29 sources failed" in payload.body
        assert payload.topic_count == 0


class TestDeliver:
    def test_one_channel_failing_does_not_sink_the_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        good, bad = Recording("good"), Recording("bad", ok=False)
        results = deliver(build_payload(Path("b.md"), TOPICS), [good, bad])
        assert results == {"good": True, "bad": False}
        assert any(results.values())

    def test_a_raising_channel_is_caught(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        results = deliver(build_payload(Path("b.md"), TOPICS), [Recording("boom", raises=True)])
        assert results == {"boom": False}

    def test_unavailable_channels_are_skipped_not_failed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        results = deliver(build_payload(Path("b.md"), TOPICS), [Recording("off", avail=False)])
        assert results == {}

    def test_payload_is_written_for_the_session_channels_to_relay(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        deliver(build_payload(Path("b.md"), TOPICS), [Recording("x")])
        written = (tmp_path / "state" / "last_notification.json").read_text()
        assert "Guardrails are the product" in written


class TestParseBrief:
    def test_round_trips_the_brief_format(self, tmp_path):
        brief = tmp_path / "brief.md"
        brief.write_text(
            "# Content Radar — 8 August 2026\n\n"
            "## 1. Guardrails are the product\n"
            "**Pitch:** Why safety scaffolding is the deliverable.\n"
            "**Why you:** You build these at Intent HQ.\n"
            "**Lane:** blog\n\n"
            "## 2. Student loan Plan 5 maths\n"
            "**Pitch:** What the threshold freeze costs you.\n"
            "**Why you:** Speaks to your own cohort.\n"
            "**Lane:** video\n",
            encoding="utf-8",
        )
        topics = parse_brief(brief)
        assert [t["title"] for t in topics] == [
            "Guardrails are the product",
            "Student loan Plan 5 maths",
        ]
        assert topics[1]["lane"] == "video"

    def test_missing_brief_is_not_an_error(self, tmp_path):
        assert parse_brief(tmp_path / "nope.md") == []


class TestPayloadShape:
    def test_payload_serialises(self):
        payload = Payload("s", "p", "b", "briefs/x.md", 3)
        assert payload.to_dict()["topic_count"] == 3
