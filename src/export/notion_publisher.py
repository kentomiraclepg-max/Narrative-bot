"""
notion_publisher.py — Push narrative report to a Notion database page.

Required env vars:
  NOTION_TOKEN         — Integration token (secret_...)
  NOTION_DATABASE_ID   — Target database ID (32-char hex or URL-form)

Optional env vars (used as page properties if the database has them):
  n/a — publisher auto-detects available properties via API

Database minimum requirement:
  - One "Name" title property (every Notion DB has this by default)

Recommended optional properties to add in Notion for filtering/sorting:
  - Date       (type: Date)
  - Articles   (type: Number)
  - BTC        (type: Number)
  - Gold       (type: Number)
  - US_Stocks  (type: Number)
  - IHSG       (type: Number)
  - DXY        (type: Number)
"""
from __future__ import annotations

import os
import re
from datetime import datetime


# ── Notion block constructors ─────────────────────────────────────────────────

def _rt(text: str) -> list[dict]:
    """Build rich_text array, splitting into 2000-char chunks if needed."""
    text = text[:2000]
    return [{"type": "text", "text": {"content": text}}]


def _heading(text: str, level: int = 2) -> dict:
    t = {1: "heading_1", 2: "heading_2", 3: "heading_3"}.get(level, "heading_2")
    return {"object": "block", "type": t, t: {"rich_text": _rt(text)}}


def _paragraph(text: str) -> dict:
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": _rt(text)}}


def _bullet(text: str) -> dict:
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": _rt(text)}}


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _code(text: str, lang: str = "plain text") -> dict:
    return {"object": "block", "type": "code",
            "code": {"rich_text": _rt(text[:2000]), "language": lang}}


def _callout(text: str, emoji: str = "📊") -> dict:
    return {
        "object": "block", "type": "callout",
        "callout": {
            "rich_text": _rt(text),
            "icon": {"type": "emoji", "emoji": emoji},
        },
    }


# ── Markdown → Notion blocks ──────────────────────────────────────────────────

def _strip_markdown_inline(text: str) -> str:
    """Remove simple inline markdown (* ` _) so Notion renders clean text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*",     r"\1", text)
    text = re.sub(r"`(.+?)`",       r"\1", text)
    text = re.sub(r"_(.+?)_",       r"\1", text)
    return text.strip()


def _parse_table(table_lines: list[str]) -> list[list[str]]:
    """Parse markdown table lines into list-of-rows (excluding separator rows)."""
    rows = []
    for line in table_lines:
        if re.match(r"^\|[-:\s|]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _table_to_notion(rows: list[list[str]]) -> dict:
    """Convert parsed rows to a Notion table block (first row = header)."""
    if not rows:
        return _paragraph("")
    width = max(len(r) for r in rows)
    children = []
    for i, row in enumerate(rows):
        # Pad short rows
        cells = row + [""] * (width - len(row))
        children.append({
            "type": "table_row",
            "table_row": {
                "cells": [[{"type": "text", "text": {"content": c[:500]}}] for c in cells]
            },
        })
    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": width,
            "has_column_header": True,
            "has_row_header": False,
            "children": children,
        },
    }


def markdown_to_blocks(markdown: str) -> list[dict]:
    """Convert report markdown to a flat list of Notion blocks."""
    blocks: list[dict] = []
    lines = markdown.splitlines()
    i = 0

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if not line:
            i += 1
            continue

        # Headings
        if line.startswith("### "):
            blocks.append(_heading(_strip_markdown_inline(line[4:]), 3))
            i += 1

        elif line.startswith("## "):
            blocks.append(_heading(_strip_markdown_inline(line[3:]), 2))
            i += 1

        elif line.startswith("# "):
            # Skip — we use the title as page title instead
            i += 1

        # Horizontal rule → divider
        elif line == "---":
            blocks.append(_divider())
            i += 1

        # Table — collect all consecutive | lines
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = _parse_table(table_lines)
            if rows:
                blocks.append(_table_to_notion(rows))

        # Italic/metadata line (starts with _)
        elif line.startswith("_") and line.endswith("_"):
            blocks.append(_callout(_strip_markdown_inline(line), "📰"))
            i += 1

        # Nested bullet (starts with spaces + -)
        elif re.match(r"^\s{2,}-\s", raw):
            blocks.append(_bullet(_strip_markdown_inline(raw.lstrip(" -"))))
            i += 1

        # Top-level bullet
        elif line.startswith("- "):
            blocks.append(_bullet(_strip_markdown_inline(line[2:])))
            i += 1

        # Bold-only line (**...**)
        elif line.startswith("**") and line.endswith("**"):
            blocks.append(_paragraph(_strip_markdown_inline(line)))
            i += 1

        # Regular paragraph
        else:
            blocks.append(_paragraph(_strip_markdown_inline(line)))
            i += 1

    return blocks


# ── Property builder ──────────────────────────────────────────────────────────

def _build_properties(title: str, ts: datetime,
                      scored: dict[str, dict],
                      db_props: dict) -> dict:
    """Build page property dict, only including props that exist in the database."""
    props: dict = {
        "Name": {"title": [{"text": {"content": title}}]},
    }

    def _has(name: str, ptype: str) -> bool:
        p = db_props.get(name, {})
        return p.get("type") == ptype

    if _has("Date", "date"):
        props["Date"] = {"date": {"start": ts.strftime("%Y-%m-%d")}}

    if _has("Articles", "number") and scored:
        n = sum(v.get("n_signals", 0) for v in scored.values())
        props["Articles"] = {"number": n}

    asset_props = {
        "BTC":      "BTC",
        "XAUUSD":   "Gold",
        "US_STOCKS": "US_Stocks",
        "IDX":      "IHSG",
        "DXY":      "DXY",
    }
    for asset, prop_name in asset_props.items():
        if _has(prop_name, "number") and asset in scored:
            props[prop_name] = {"number": round(scored[asset].get("score", 0.0), 4)}

    return props


# ── Main publisher ────────────────────────────────────────────────────────────

def _append_blocks(notion, page_id: str, blocks: list[dict]) -> None:
    """Append blocks to an existing page in batches of 99."""
    BATCH = 99
    for start in range(0, len(blocks), BATCH):
        notion.blocks.children.append(
            block_id=page_id,
            children=blocks[start:start + BATCH],
        )


def publish_to_notion(report_path: str,
                      scored: dict[str, dict],
                      ts: datetime | None = None) -> str | None:
    """
    Push the report to Notion. Supports both a Database ID and a Page ID as
    NOTION_DATABASE_ID — auto-detects which one it is.

    Returns the new page URL on success, None otherwise.
    Silently skips if NOTION_TOKEN / NOTION_DATABASE_ID are not set.
    """
    token   = os.getenv("NOTION_TOKEN", "").strip()
    parent_id = os.getenv("NOTION_DATABASE_ID", "").strip()

    if not token or not parent_id:
        print("[Notion] NOTION_TOKEN or NOTION_DATABASE_ID not set — skipping.")
        return None

    try:
        from notion_client import Client           # type: ignore
        from notion_client.errors import APIResponseError  # type: ignore
    except ImportError:
        print("[Notion] notion-client not installed — run: pip install notion-client")
        return None

    ts = ts or datetime.utcnow()
    from datetime import timedelta
    ts_wib = ts + timedelta(hours=7)
    title = f"Narrative Report — {ts_wib.strftime('%Y-%m-%d %H:%M')} WIB"

    # Read report markdown
    try:
        with open(report_path, encoding="utf-8") as f:
            markdown = f.read()
    except Exception as e:
        print(f"[Notion] Cannot read report file: {e}")
        return None

    notion = Client(auth=token)
    blocks = markdown_to_blocks(markdown)
    BATCH  = 99

    # ── Try as database first ────────────────────────────────────────────────
    try:
        db_meta    = notion.databases.retrieve(database_id=parent_id)
        db_props   = db_meta.get("properties", {})
        properties = _build_properties(title, ts, scored, db_props)

        response = notion.pages.create(
            parent={"database_id": parent_id},
            properties=properties,
            children=blocks[:BATCH],
        )
        page_id  = response["id"]
        page_url = response.get("url", "")
        if len(blocks) > BATCH:
            _append_blocks(notion, page_id, blocks[BATCH:])
        print(f"[Notion] Published (database) → {page_url}")
        return page_url

    except APIResponseError as e:
        is_page = "is a page, not a database" in str(e)
        if not is_page:
            print(f"[Notion] Database error: {e}")
            return None
        # Fall through to page-parent mode

    # ── Fall back: parent is a plain page ────────────────────────────────────
    try:
        properties = {
            "title": [{"text": {"content": title}}],
        }
        response = notion.pages.create(
            parent={"page_id": parent_id},
            properties=properties,
            children=blocks[:BATCH],
        )
        page_id  = response["id"]
        page_url = response.get("url", "")
        if len(blocks) > BATCH:
            _append_blocks(notion, page_id, blocks[BATCH:])
        print(f"[Notion] Published (subpage) → {page_url}")
        return page_url

    except APIResponseError as e:
        print(f"[Notion] API error: {e}")
        return None
    except Exception as e:
        print(f"[Notion] Unexpected error: {e}")
        return None
