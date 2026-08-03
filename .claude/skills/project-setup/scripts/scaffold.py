#!/usr/bin/env python3
"""Lay down the file skeleton for a new project.

Copies the skill's templates into the target directory, substitutes placeholders,
and creates the .claude/ command and agent files that were selected.

This handles the mechanical part only. The templates arrive with HTML-comment
guidance and `{{...}}` placeholders still in them where real content is needed —
the calling agent fills those in from the signed-off project spec. Run this first
so no time is spent typing boilerplate, then edit substance in.

Existing files are never overwritten unless --force is passed; the script reports
what it skipped so nothing is silently clobbered.

Usage:
    scaffold.py --name "Thumbnail Studio" --dir ./thumbnail-studio
    scaffold.py --name "X" --dir . --stack next-supabase --commands commit,update-docs-and-commit
    scaffold.py --name "X" --dir . --agents changelog-writer,retro --hooks --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATES = SKILL_DIR / "assets" / "templates"
COMMANDS = SKILL_DIR / "assets" / "commands"
AGENTS = SKILL_DIR / "assets" / "agents"
HOOKS = SKILL_DIR / "assets" / "hooks"

# template filename -> destination, relative to the project root
DOC_LAYOUT = {
    "project_spec.md": "docs/project_spec.md",
    "ARCHITECTURE.md": "docs/ARCHITECTURE.md",
    "CHANGELOG.md": "docs/CHANGELOG.md",
    "PROJECT_STATUS.md": "docs/PROJECT_STATUS.md",
    "CLAUDE.md": "CLAUDE.md",
    "env.example": ".env.example",
}

# Stack-specific .gitignore additions on top of the common base.
STACK_IGNORES = {
    "node": ["node_modules/", ".next/", "dist/", "build/", ".turbo/", "*.tsbuildinfo"],
    "python": [
        "__pycache__/",
        "*.py[cod]",
        ".venv/",
        "venv/",
        ".pytest_cache/",
        ".ruff_cache/",
        "*.egg-info/",
    ],
    "swift": [".build/", "*.xcuserstate", "xcuserdata/", "DerivedData/"],
}

# Which ignore sets a named stack pulls in. Unknown stacks get every set, which is
# harmless — an ignore line for a language you aren't using costs nothing.
STACK_ALIASES = {
    "next": ["node"],
    "next-supabase": ["node"],
    "next-mongo": ["node"],
    "vue": ["node"],
    "angular": ["node"],
    "express": ["node"],
    "node": ["node"],
    "fastapi": ["python"],
    "django": ["python"],
    "python": ["python"],
    "swift": ["swift"],
    "fullstack": ["node", "python"],
}

BASE_IGNORE = [
    "# Secrets — never commit these",
    ".env",
    ".env.local",
    ".env.*.local",
    "*.pem",
    "",
    "# OS / editor",
    ".DS_Store",
    "Thumbs.db",
    ".idea/",
    ".vscode/*",
    "!.vscode/settings.json",
    "",
    "# Logs",
    "*.log",
    "npm-debug.log*",
    "",
    "# Claude Code local state",
    ".claude/settings.local.json",
    ".claude/transcripts/",
    ".claude/session.log",
]


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return re.sub(r"-{2,}", "-", s) or "project"


def substitute(text: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        text = text.replace("{{%s}}" % key, value)
    return text


class Scaffolder:
    def __init__(self, root: Path, values: dict[str, str], force: bool, dry_run: bool):
        self.root = root
        self.values = values
        self.force = force
        self.dry_run = dry_run
        self.created: list[str] = []
        self.skipped: list[str] = []

    def write(self, rel: str, content: str) -> None:
        dest = self.root / rel
        if dest.exists() and not self.force:
            self.skipped.append(rel)
            return
        if not self.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
        self.created.append(rel)

    def copy_template(self, template_name: str, rel: str) -> None:
        src = TEMPLATES / template_name
        if not src.exists():
            print(f"  ! template missing: {src}", file=sys.stderr)
            return
        self.write(rel, substitute(src.read_text(encoding="utf-8"), self.values))


def build_gitignore(stack: str | None) -> str:
    lines = list(BASE_IGNORE)
    keys = STACK_ALIASES.get((stack or "").lower(), list(STACK_IGNORES))
    for key in keys:
        lines += ["", f"# {key}"] + STACK_IGNORES[key]
    return "\n".join(lines) + "\n"


def copy_selected(sc: Scaffolder, src_dir: Path, names: list[str], dest_dir: str) -> None:
    """Copy named .md files from a skill asset dir into the project's .claude dir."""
    for name in names:
        stem = name.strip()
        if not stem:
            continue
        src = src_dir / f"{stem}.md"
        if not src.exists():
            available = sorted(p.stem for p in src_dir.glob("*.md"))
            print(
                f"  ! no such asset '{stem}' in {src_dir.name}/ (have: {', '.join(available)})",
                file=sys.stderr,
            )
            continue
        sc.write(f"{dest_dir}/{stem}.md", src.read_text(encoding="utf-8"))


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    if value.strip().lower() == "all":
        return ["__ALL__"]
    return [v for v in (p.strip() for p in value.split(",")) if v]


def expand_all(names: list[str], src_dir: Path) -> list[str]:
    if names == ["__ALL__"]:
        return sorted(p.stem for p in src_dir.glob("*.md"))
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description="Scaffold a new project's files.")
    ap.add_argument("--name", required=True, help="Human-readable project name")
    ap.add_argument("--dir", required=True, help="Target project directory")
    ap.add_argument("--stack", help="Stack slug, e.g. next-supabase, fastapi, swift")
    ap.add_argument("--one-liner", default="", help="One-sentence description for CLAUDE.md")
    ap.add_argument("--milestone", default="MVP", help="Current milestone name")
    ap.add_argument(
        "--commands",
        help="Comma-separated slash commands to install, or 'all'. "
        "Default: commit,commit-push-pr,update-docs-and-commit,create-issues",
    )
    ap.add_argument("--agents", help="Comma-separated subagents to install, or 'all'")
    ap.add_argument("--hooks", action="store_true", help="Copy the hooks example into .claude/")
    ap.add_argument("--no-docs", action="store_true", help="Skip the docs/ + CLAUDE.md set")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    ap.add_argument("--dry-run", action="store_true", help="Report what would happen, write nothing")
    args = ap.parse_args()

    root = Path(args.dir).expanduser().resolve()
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.exists():
        print(f"(dry run) would create directory {root}")

    values = {
        "PROJECT_NAME": args.name,
        "PROJECT_SLUG": slugify(args.name),
        "DATE": _dt.date.today().isoformat(),
        "ONE_LINER": args.one_liner or f"<one-sentence description of {args.name}>",
        "CURRENT_MILESTONE": args.milestone,
        "FEATURE_NAME": "<Feature name>",
    }

    sc = Scaffolder(root, values, args.force, args.dry_run)

    if not args.no_docs:
        for template_name, rel in DOC_LAYOUT.items():
            sc.copy_template(template_name, rel)
        # reference/ starts with the template itself, so the shape is obvious
        # when the first feature needs documenting.
        sc.copy_template("reference-doc.md", "docs/reference/_template.md")

    sc.write(".gitignore", build_gitignore(args.stack))

    cmd_names = parse_list(args.commands) or [
        "commit",
        "commit-push-pr",
        "update-docs-and-commit",
        "create-issues",
    ]
    copy_selected(sc, COMMANDS, expand_all(cmd_names, COMMANDS), ".claude/commands")

    agent_names = expand_all(parse_list(args.agents), AGENTS)
    copy_selected(sc, AGENTS, agent_names, ".claude/agents")

    if args.hooks:
        src = HOOKS / "settings.example.json"
        if src.exists():
            sc.write(".claude/settings.example.json", src.read_text(encoding="utf-8"))

    prefix = "(dry run) " if args.dry_run else ""
    print(f"\n{prefix}Scaffolded {args.name} in {root}\n")
    if sc.created:
        print("Created:")
        for rel in sc.created:
            print(f"  + {rel}")
    if sc.skipped:
        print("\nSkipped (already exist — pass --force to overwrite):")
        for rel in sc.skipped:
            print(f"  = {rel}")

    print(
        "\nNext: fill in the templates from the signed-off spec. Nothing should ship\n"
        "with {{...}} placeholders or guidance comments left in it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
