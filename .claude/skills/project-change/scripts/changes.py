#!/usr/bin/env python3
"""Manage change records for a project.

Allocates IDs, writes records from the skill's template, maintains the register,
and validates that a record is complete enough to close.

The record is the artifact; the register is derived. `register` rebuilds the index
from the records on disk, so the index is never state anyone has to maintain.

Usage:
    changes.py new "Replace MongoDB with Supabase" --type migration
    changes.py list --status open
    changes.py show CHG-0001
    changes.py set CHG-0001 --status approved --decided-by branden
    changes.py validate CHG-0001
    changes.py register
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = SKILL_DIR / "assets" / "templates"

CHANGES_DIR = "docs/changes"
REGISTER = "docs/CHANGE_REGISTER.md"

TYPES = ["feature", "change", "deprecation", "migration", "dependency", "security", "fix"]
STATUSES = ["open", "approved", "in-progress", "shipped", "reverted", "deferred", "rejected"]
OPEN_STATUSES = {"open", "approved", "in-progress"}
REVERSIBILITY = ["cheap", "annoying", "one-way-door", "unrated"]


def today() -> str:
    return _dt.date.today().isoformat()


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)[:48] or "change"


def substitute(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


class Record:
    """A change record on disk: front matter plus body."""

    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text(encoding="utf-8")

    @property
    def front(self) -> dict[str, str]:
        m = re.match(r"^---\n(.*?)\n---\n", self.text, re.S)
        if not m:
            return {}
        fields = {}
        for line in m.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()
        return fields

    def get(self, key: str, default: str = "") -> str:
        return self.front.get(key, default) or default

    def set_fields(self, updates: dict[str, str]) -> None:
        text = self.text
        for key, value in updates.items():
            pattern = re.compile(rf"^({re.escape(key)}:).*$", re.M)
            if pattern.search(text):
                text = pattern.sub(rf"\1 {value}", text, count=1)
            else:
                text = re.sub(r"^---\n", f"---\n{key}: {value}\n", text, count=1)
        self.text = text
        self.path.write_text(text, encoding="utf-8")

    @property
    def body(self) -> str:
        return re.sub(r"^---\n.*?\n---\n", "", self.text, flags=re.S)


def records(root: Path) -> list[Record]:
    directory = root / CHANGES_DIR
    if not directory.is_dir():
        return []
    return [Record(p) for p in sorted(directory.glob("CHG-*.md"))]


def next_id(root: Path) -> str:
    highest = 0
    for record in records(root):
        m = re.match(r"CHG-(\d+)", record.path.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return f"CHG-{highest + 1:04d}"


def find(root: Path, change_id: str) -> Record:
    change_id = change_id.upper()
    if not change_id.startswith("CHG-"):
        change_id = f"CHG-{int(change_id):04d}"
    for record in records(root):
        if record.get("id").upper() == change_id:
            return record
    sys.exit(f"error: no record found for {change_id}")


def cmd_new(args: argparse.Namespace, root: Path) -> int:
    template_path = TEMPLATES / "change-record.md"
    if not template_path.is_file():
        sys.exit(f"error: template missing at {template_path}")

    change_id = next_id(root)
    path = root / CHANGES_DIR / f"{change_id}-{slugify(args.title)}.md"

    if path.exists() and not args.force:
        sys.exit(f"error: {path} already exists (use --force to overwrite)")

    content = substitute(
        template_path.read_text(encoding="utf-8"),
        {
            "ID": change_id,
            "TITLE": args.title,
            "TYPE": args.type,
            "STATUS": "open",
            "REVERSIBILITY": args.reversibility,
            "DATE": today(),
        },
    )

    if args.dry_run:
        print(f"would create {path}")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"created {path.relative_to(root)}")
    print(f"  id     {change_id}")
    print(f"  type   {args.type}")
    print("\nFill the Request section before assessing impact.")
    return 0


def cmd_list(args: argparse.Namespace, root: Path) -> int:
    found = records(root)
    if args.status and args.status != "all":
        wanted = OPEN_STATUSES if args.status == "open" else {args.status}
        found = [r for r in found if r.get("status") in wanted]
    if not found:
        print("no change records")
        return 0
    for record in found:
        print(
            f"{record.get('id'):<10} {record.get('status'):<12} "
            f"{record.get('type'):<12} {record.get('reversibility', 'unrated'):<13} "
            f"{record.get('title')}"
        )
    return 0


def cmd_show(args: argparse.Namespace, root: Path) -> int:
    print(find(root, args.id).text)
    return 0


def cmd_set(args: argparse.Namespace, root: Path) -> int:
    record = find(root, args.id)
    updates: dict[str, str] = {}
    if args.status:
        updates["status"] = args.status
        if args.status == "shipped":
            updates["shipped"] = today()
    if args.reversibility:
        updates["reversibility"] = args.reversibility
    if args.decided_by:
        updates["decided_by"] = args.decided_by
        updates["decided_on"] = today()
    if args.retroactive:
        updates["retroactive"] = "true"
    if not updates:
        sys.exit("error: nothing to set")

    if args.dry_run:
        print(f"would set on {record.get('id')}: {updates}")
        return 0

    record.set_fields(updates)
    print(f"{record.get('id')}: " + ", ".join(f"{k}={v}" for k, v in updates.items()))
    return 0


PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")

REQUIRED_SECTIONS = ["## Request", "## Impact", "## Reversibility", "## Decision", "## Verification"]

# What must actually be filled in, by status. A record is allowed to be incomplete
# while it is still moving; it is not allowed to close that way.
NEEDS_FILLED = {
    "open": ["## Request"],
    "approved": ["## Request", "## Impact", "## Decision"],
    "in-progress": ["## Request", "## Impact", "## Decision"],
    "shipped": REQUIRED_SECTIONS,
    "reverted": REQUIRED_SECTIONS,
    "deferred": ["## Request", "## Decision"],
    "rejected": ["## Request", "## Decision"],
}

CLOSED = {"shipped", "reverted"}


def section_text(text: str, heading: str) -> str:
    """The prose under a heading, with template scaffolding stripped out."""
    m = re.search(rf"^{re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not m:
        return ""
    body = re.sub(r"<!--.*?-->", "", m.group(1), flags=re.S)   # guidance comments
    body = re.sub(r"^\s*\|[\s|:-]*\|\s*$", "", body, flags=re.M)  # empty table rows
    body = re.sub(r"^\s*\*\*[^*]+:\*\*\s*$", "", body, flags=re.M)  # bare bold labels
    body = re.sub(r"^\s*[-*]\s*(\[[ x]\])?\s*$", "", body, flags=re.M)  # empty bullets
    body = re.sub(r"^\s*```.*?```\s*$", "", body, flags=re.M | re.S)  # untouched code fences
    return body.strip()


def cmd_validate(args: argparse.Namespace, root: Path) -> int:
    targets = [find(root, args.id)] if args.id else records(root)
    if not targets:
        print("no change records to validate")
        return 0

    failed = False
    for record in targets:
        problems = []
        status = record.get("status")

        if PLACEHOLDER.search(record.text):
            problems.append("unfilled {{PLACEHOLDER}} tokens remain")

        if status not in STATUSES:
            problems.append(f"status '{status}' is not one of {', '.join(STATUSES)}")

        for heading in REQUIRED_SECTIONS:
            if heading not in record.text:
                problems.append(f"missing section {heading}")

        for heading in NEEDS_FILLED.get(status, ["## Request"]):
            if heading in record.text and len(section_text(record.text, heading)) < 30:
                problems.append(f"{heading} is empty — required at status '{status}'")

        if status in CLOSED:
            if not record.get("decided_by"):
                problems.append("no decided_by — this is the field most standards actually ask for")
            if not re.search(r"- \[x\]", record.body, re.I):
                problems.append("no ticked verification item — unverified is not shipped")
            if record.get("reversibility", "unrated") == "unrated":
                problems.append("never rated for reversibility")

        if problems:
            failed = True
            print(f"{record.get('id')} — {len(problems)} problem(s)")
            for problem in problems:
                print(f"  - {problem}")
        elif not args.quiet:
            print(f"{record.get('id')} — ok")

    return 1 if failed else 0


def cmd_register(args: argparse.Namespace, root: Path) -> int:
    template_path = TEMPLATES / "CHANGE_REGISTER.md"
    if not template_path.is_file():
        sys.exit(f"error: template missing at {template_path}")

    open_rows, closed_rows = [], []
    for record in records(root):
        status = record.get("status")
        if status in OPEN_STATUSES:
            open_rows.append(
                f"| {record.get('id')} | {record.get('title')} | {record.get('type')} "
                f"| {record.get('reversibility', 'unrated')} | {record.get('opened')} |"
            )
        else:
            closed_rows.append(
                f"| {record.get('id')} | {record.get('title')} | {record.get('type')} "
                f"| {status} | {record.get('shipped', '—')} |"
            )

    content = substitute(
        template_path.read_text(encoding="utf-8"),
        {
            "DATE": today(),
            "OPEN_ROWS": "\n".join(open_rows) or "| — | none open | | | |",
            "CLOSED_ROWS": "\n".join(closed_rows) or "| — | none closed | | | |",
        },
    )

    path = root / REGISTER
    if args.dry_run:
        print(f"would write {path} ({len(open_rows)} open, {len(closed_rows)} closed)")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path.relative_to(root)} — {len(open_rows)} open, {len(closed_rows)} closed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage project change records.")
    parser.add_argument("--dir", default=".", help="Project root (default: cwd)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen, write nothing")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="Open a change record")
    p_new.add_argument("title")
    p_new.add_argument("--type", choices=TYPES, default="change")
    p_new.add_argument("--reversibility", choices=REVERSIBILITY, default="unrated")
    p_new.add_argument("--force", action="store_true")
    p_new.set_defaults(func=cmd_new)

    p_list = sub.add_parser("list", help="List change records")
    p_list.add_argument("--status", default="all", choices=["all", "open", *STATUSES])
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Print a change record")
    p_show.add_argument("id")
    p_show.set_defaults(func=cmd_show)

    p_set = sub.add_parser("set", help="Update fields on a record")
    p_set.add_argument("id")
    p_set.add_argument("--status", choices=STATUSES)
    p_set.add_argument("--reversibility", choices=REVERSIBILITY)
    p_set.add_argument("--decided-by", dest="decided_by")
    p_set.add_argument("--retroactive", action="store_true")
    p_set.set_defaults(func=cmd_set)

    p_val = sub.add_parser("validate", help="Check records are complete")
    p_val.add_argument("id", nargs="?")
    p_val.add_argument("--quiet", action="store_true", help="Only report problems")
    p_val.set_defaults(func=cmd_validate)

    p_reg = sub.add_parser("register", help="Rebuild the register from the records")
    p_reg.set_defaults(func=cmd_register)

    args = parser.parse_args()
    root = Path(args.dir).resolve()
    if not root.is_dir():
        sys.exit(f"error: {root} is not a directory")
    return args.func(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
