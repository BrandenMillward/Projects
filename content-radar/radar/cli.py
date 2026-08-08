"""Command line interface.

The Python side is deterministic: fetch, cluster, score, deliver. The editorial
judgement — which of the top-scoring clusters are actually worth writing, and
what the angle is — happens in Claude via ``/radar``. Keeping the split here
means this half is unit-testable and its output is reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import notify as notify_mod
from . import sources as sources_mod
from . import state as state_mod
from .cluster import cluster_items
from .score import score_all

CANDIDATES = Path("state/candidates.json")
MIN_TOPICS = 3  # below this a brief is treated as a thin-week failure


def _cmd_fetch(args: argparse.Namespace) -> int:
    specs = sources_mod.load_sources()
    report = sources_mod.fetch_all(specs, since_days=args.since)
    written = state_mod.write_items(report.items)

    print(f"fetched {written} items from {len(report.ok)}/{report.total_sources} sources")
    if report.failed:
        print(f"\n{len(report.failed)} source(s) failed:")
        for name, error in sorted(report.failed.items()):
            print(f"  - {name}: {error}")

    if report.degraded:
        print(
            f"\nDEGRADED: {report.failure_rate:.0%} of sources failed "
            f"(threshold {sources_mod.SOURCE_FAILURE_THRESHOLD:.0%}). "
            "This is a failure, not a quiet news week.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    specs = sources_mod.load_sources()
    items = state_mod.read_items(since_days=args.since)
    if not items:
        print("no items — run `radar fetch` first", file=sys.stderr)
        return 1

    clusters = cluster_items(items)
    scored = score_all(
        clusters,
        source_weights=sources_mod.source_weights(specs),
        seen_hashes=state_mod.seen_hashes(),
    )
    top = scored[: args.top]

    payload = {
        "generated_at": state_mod.utcnow(),
        "item_count": len(items),
        "cluster_count": len(clusters),
        "suppressed": state_mod.suppressed_titles(),
        "candidates": [c.to_dict() for c in top],
    }
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATES.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"{len(items)} items -> {len(clusters)} clusters -> "
        f"top {len(top)} written to {CANDIDATES}"
    )
    for n, cluster in enumerate(top[:10], start=1):
        print(f"  {n:2}. [{cluster.score:.3f}] ({cluster.size}x) {cluster.title[:80]}")
    return 0


def _cmd_notify(args: argparse.Namespace) -> int:
    brief = Path(args.brief)

    if args.failure:
        payload = notify_mod.build_payload(brief, [], failure=args.failure)
    else:
        topics = notify_mod.parse_brief(brief)
        if len(topics) < MIN_TOPICS:
            payload = notify_mod.build_payload(
                brief,
                topics,
                failure=f"only {len(topics)} topic(s) found, expected at least {MIN_TOPICS}",
            )
        else:
            payload = notify_mod.build_payload(brief, topics)

    results = notify_mod.deliver(payload)
    sent = [name for name, ok in results.items() if ok]
    failed = [name for name, ok in results.items() if not ok]

    print(f"payload written to {notify_mod.PAYLOAD_PATH}")
    if sent:
        print(f"delivered via: {', '.join(sorted(sent))}")
    if failed:
        print(f"failed on: {', '.join(sorted(failed))}", file=sys.stderr)

    if not sent:
        print("DELIVERY FAILED: no channel confirmed delivery", file=sys.stderr)
        return 1
    return 1 if payload.is_failure else 0


def _cmd_seen(args: argparse.Namespace) -> int:
    data = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    urls = [u for c in data.get("candidates", []) for u in c.get("urls", [])]
    added = state_mod.mark_seen(urls)
    pruned = state_mod.prune_seen()
    print(f"marked {added} new URLs as seen; pruned {pruned} expired entries")
    return 0


def _cmd_decide(args: argparse.Namespace) -> int:
    state_mod.record_decision(args.cluster_id, args.title, args.state, args.reason)
    print(f"recorded: {args.title!r} -> {args.state}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="radar", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="poll all sources and append to state/items.jsonl")
    fetch.add_argument("--since", type=int, default=7, help="days of history to request")
    fetch.set_defaults(func=_cmd_fetch)

    score = sub.add_parser("score", help="cluster and score items into state/candidates.json")
    score.add_argument("--since", type=int, default=7)
    score.add_argument("--top", type=int, default=30)
    score.set_defaults(func=_cmd_score)

    notify = sub.add_parser("notify", help="deliver the brief")
    notify.add_argument("--brief", required=True, help="path to the written brief")
    notify.add_argument("--failure", default="", help="send a failure notification instead")
    notify.set_defaults(func=_cmd_notify)

    seen = sub.add_parser("seen", help="mark surfaced candidates as seen, and prune")
    seen.set_defaults(func=_cmd_seen)

    decide = sub.add_parser("decide", help="record a decision about a topic")
    decide.add_argument("cluster_id")
    decide.add_argument("title")
    decide.add_argument("state", choices=["suggested", "picked", "shipped", "killed"])
    decide.add_argument("--reason", default="")
    decide.set_defaults(func=_cmd_decide)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
