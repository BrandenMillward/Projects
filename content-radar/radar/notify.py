"""Delivery.

The brief is pushed, not parked. Committing markdown is the archive; it is not
the delivery. A weekly brief nobody is told about is a brief nobody reads.

Two kinds of channel live here:

* **Python channels** — Telegram (v1). These send directly from this process.
* **Session channels** — phone push, email, and surfacing the brief file. These
  are sent by Claude in the run session (``PushNotification`` / ``SendUserFile``)
  and by the Routine's own completion notification. Python cannot send them, so
  it builds the payload and writes ``state/last_notification.json`` for the
  slash command to relay.

Splitting it this way means the message that reaches the phone is generated from
the committed brief, so what you read and what is in the repo cannot drift.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import httpx

from .models import utcnow

PAYLOAD_PATH = Path("state/last_notification.json")
PUSH_LIMIT = 200  # mobile OSes truncate beyond roughly this


@dataclass(slots=True)
class Payload:
    """What gets delivered, in the three lengths the channels need."""

    subject: str
    push: str  # one line, < 200 chars, for a lock screen
    body: str  # the full topic list, for email and Telegram
    brief_path: str
    topic_count: int
    is_failure: bool = False

    def to_dict(self) -> dict:
        return {
            "at": utcnow(),
            "subject": self.subject,
            "push": self.push,
            "body": self.body,
            "brief_path": self.brief_path,
            "topic_count": self.topic_count,
            "is_failure": self.is_failure,
        }


class Channel(ABC):
    """A delivery channel. ``send`` returns True on confirmed delivery."""

    name: str = "channel"

    @abstractmethod
    def available(self) -> bool:
        """Whether this channel is configured well enough to try."""

    @abstractmethod
    def send(self, payload: Payload) -> bool:
        """Deliver. Must not raise — return False instead."""


class ConsoleChannel(Channel):
    """Always available. Makes a local run visible without any configuration."""

    name = "console"

    def available(self) -> bool:
        return True

    def send(self, payload: Payload) -> bool:
        print(f"\n=== {payload.subject} ===\n{payload.body}\n")
        return True


class TelegramChannel(Channel):
    """Telegram Bot API — free at this volume, and the only two-way channel.

    Needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment. Absent
    those, the channel reports itself unavailable and is skipped silently.
    """

    name = "telegram"

    def __init__(self, token: str | None = None, chat_id: str | None = None) -> None:
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    def available(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, payload: Payload) -> bool:
        if not self.available():
            return False
        text = f"*{_escape_md(payload.subject)}*\n\n{_escape_md(payload.body)}"
        try:
            response = httpx.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text[:4000],  # Telegram caps a message at 4096
                    "parse_mode": "MarkdownV2",
                    "disable_web_page_preview": True,
                },
                timeout=15.0,
            )
            return response.status_code == 200
        except Exception:  # noqa: BLE001 — a channel failure is not a run failure
            return False


_MD_SPECIALS = r"_*[]()~`>#+-=|{}.!"


def _escape_md(text: str) -> str:
    return "".join("\\" + c if c in _MD_SPECIALS else c for c in text)


def build_payload(brief_path: Path, topics: list[dict], *, failure: str = "") -> Payload:
    """Build the delivery payload from the committed brief.

    *topics* is a list of ``{"title", "pitch", "why_you", "lane"}`` dicts.
    Passing *failure* produces a failure notification instead.
    """
    if failure:
        return Payload(
            subject="Content Radar — run failed",
            push=f"Radar failed: {failure}"[:PUSH_LIMIT],
            body=(
                f"The weekly run did not complete.\n\nReason: {failure}\n\n"
                "No brief was produced. Run `/radar` manually to retry."
            ),
            brief_path=str(brief_path),
            topic_count=0,
            is_failure=True,
        )

    count = len(topics)
    lead = topics[0]["title"] if topics else "no topics"

    push = f"{count} topic{'s' if count != 1 else ''} this week — lead: {lead}"
    if len(push) > PUSH_LIMIT:
        push = push[: PUSH_LIMIT - 1].rstrip() + "…"

    lines = [f"{count} topics for the week — decide from here, no need to open anything.\n"]
    for n, topic in enumerate(topics, start=1):
        lines.append(f"{n}. {topic['title']}  [{topic.get('lane', 'blog')}]")
        if topic.get("pitch"):
            lines.append(f"   {topic['pitch']}")
        if topic.get("why_you"):
            lines.append(f"   Why you: {topic['why_you']}")
        lines.append("")
    lines.append(f"Full brief: {brief_path}")
    lines.append("Pick one with:  /radar pick <n>")

    return Payload(
        subject=f"Content Radar — {count} topics",
        push=push,
        body="\n".join(lines),
        brief_path=str(brief_path),
        topic_count=count,
    )


def default_channels() -> list[Channel]:
    """Channels this process can send on its own."""
    return [ConsoleChannel(), TelegramChannel()]


def deliver(payload: Payload, channels: list[Channel] | None = None) -> dict[str, bool]:
    """Send on every available channel and write the payload for the session
    channels to relay.

    Returns a per-channel result map. A single channel failing is tolerated; the
    caller decides what to do when every one of them fails.
    """
    channels = channels if channels is not None else default_channels()

    PAYLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAYLOAD_PATH.write_text(
        json.dumps(payload.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results: dict[str, bool] = {}
    for channel in channels:
        if not channel.available():
            continue
        try:
            results[channel.name] = channel.send(payload)
        except Exception:  # noqa: BLE001 — defensive; channels should not raise
            results[channel.name] = False
    return results


def parse_brief(path: Path) -> list[dict]:
    """Extract topics from a written brief so the message and the file agree.

    Expects the shape written by ``.claude/commands/radar.md``:

        ## 1. <title>
        **Pitch:** ...
        **Why you:** ...
        **Lane:** ...
    """
    if not path.exists():
        return []

    topics: list[dict] = []
    current: dict | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^##\s+\d+\.\s+(.+?)\s*$", line)
        if heading:
            if current:
                topics.append(current)
            current = {"title": heading.group(1), "pitch": "", "why_you": "", "lane": ""}
            continue
        if current is None:
            continue
        for label, key in (("Pitch", "pitch"), ("Why you", "why_you"), ("Lane", "lane")):
            found = re.match(rf"^\*\*{label}:\*\*\s*(.+?)\s*$", line, re.IGNORECASE)
            if found:
                current[key] = found.group(1)

    if current:
        topics.append(current)
    return topics
