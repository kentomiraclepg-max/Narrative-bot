"""
reporter.py — Render per-asset impact analysis as a markdown report.

Output filename pattern:
    reports/narrative_impact_YYYY-MM-DD_HHMM.md

The report is intended to be checked into the `reports/` folder (or pushed
via Telegram) as the canonical human-readable artifact of one scan cycle.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from glob import glob

import config
from src.impact.asset_map import ASSET_LABELS, ASSETS
from src.impact.scorer    import aggregate_themes, summarize_drivers


# ── Delta helpers ────────────────────────────────────────────────────────────

def _scores_path(directory: str, ts: datetime) -> str:
    return os.path.join(directory, f"narrative_scores_{ts.strftime('%Y-%m-%d_%H%M')}.json")


def _load_previous_scores(directory: str, current_ts: datetime) -> dict[str, dict] | None:
    """Find and load the most recent scores JSON older than current_ts."""
    pattern = os.path.join(directory, "narrative_scores_*.json")
    candidates = sorted(glob(pattern), reverse=True)
    current_fname = f"narrative_scores_{current_ts.strftime('%Y-%m-%d_%H%M')}.json"
    for path in candidates:
        if os.path.basename(path) != current_fname:
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def _render_delta_section(scored: dict[str, dict],
                           previous: dict[str, dict]) -> list[str]:
    """Return markdown lines showing score changes since last report."""
    lines = ["## Changes since last report", ""]
    lines.append("| Asset | Prev | Now | Δ Score | Direction |")
    lines.append("|---|---:|---:|---:|---|")

    any_change = False
    for asset in ASSETS:
        curr = scored.get(asset)
        prev = previous.get(asset)
        if not curr or not prev:
            continue
        delta = curr["score"] - prev["score"]
        if abs(delta) < 0.01:
            continue
        any_change = True
        arrow = "▲" if delta > 0 else "▼"
        prev_label = prev.get("label", "?")
        curr_label = curr["label"]
        changed = f"{prev_label} → {curr_label}" if prev_label != curr_label else curr_label
        lines.append(
            f"| {ASSET_LABELS.get(asset, asset)} "
            f"| {prev['score']:+.3f} "
            f"| {curr['score']:+.3f} "
            f"| {delta:+.3f} "
            f"| {arrow} {changed} |"
        )

    if not any_change:
        lines.append("_No significant score changes (Δ < 0.01)._")
    lines.append("")
    return lines


# ── Helpers ──────────────────────────────────────────────────────────────────
_BAR_W = 20

def _bar(score: float, width: int = _BAR_W) -> str:
    """ASCII bar like  [────●────] — center is 0.0, ends are ±1.0."""
    score = max(-1.0, min(1.0, score))
    half  = width // 2
    pos   = half + int(round(score * half))
    cells = ["─"] * width
    cells[max(0, min(width - 1, pos))] = "●"
    return "[" + "".join(cells) + "]"


def _emoji_for(label: str) -> str:
    return {"BULLISH": "▲", "BEARISH": "▼", "NEUTRAL": "■"}.get(label, "■")


# ── Renderer ─────────────────────────────────────────────────────────────────
def render_report(scored: dict[str, dict],
                  articles: list[dict],
                  ts: datetime | None = None,
                  previous_scores: dict[str, dict] | None = None) -> str:
    """Render markdown report string from scorer output + raw article list."""
    ts = ts or datetime.utcnow()
    n_articles = len(articles)
    themes     = aggregate_themes(articles)

    lines: list[str] = []
    lines.append(f"# Narrative Impact Report — {ts.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append(f"_Corpus: **{n_articles}** articles · "
                 f"themes detected: **{len(themes)}** · "
                 f"assets tracked: **{len(scored)}**_")
    lines.append("")

    # ── Top-line snapshot table ──────────────────────────────────────────────
    lines.append("## At-a-glance")
    lines.append("")
    lines.append("| Asset | Bias | Strength | Score | Bar | Signals |")
    lines.append("|---|---|---|---:|---|---:|")
    for asset in ASSETS:
        r = scored.get(asset)
        if not r:
            continue
        label_emoji = _emoji_for(r["label"])
        lines.append(
            f"| {ASSET_LABELS.get(asset, asset)} "
            f"| {label_emoji} {r['label']} "
            f"| {r['strength']} "
            f"| {r['score']:+.3f} "
            f"| `{_bar(r['score'])}` "
            f"| {r['n_signals']} |"
        )
    lines.append("")

    # ── Dominant narratives ──────────────────────────────────────────────────
    if themes:
        lines.append("## Dominant macro themes (by article volume)")
        lines.append("")
        lines.append("| Theme | Articles | Avg sentiment |")
        lines.append("|---|---:|---:|")
        for t in themes[:10]:
            lines.append(f"| `{t['theme']}` | {t['n_articles']} | {t['avg_sentiment']:+.2f} |")
        lines.append("")

    # ── Delta vs previous report ─────────────────────────────────────────────
    if previous_scores:
        lines.extend(_render_delta_section(scored, previous_scores))

    # ── Per-asset deep-dive ──────────────────────────────────────────────────
    lines.append("## Per-asset analysis")
    lines.append("")
    for asset in ASSETS:
        r = scored.get(asset)
        if not r:
            continue
        lines.append(f"### {_emoji_for(r['label'])} {ASSET_LABELS.get(asset, asset)}")
        lines.append("")
        lines.append(f"- **Bias:** {r['label']} ({r['strength']})")
        lines.append(f"- **Score:** `{r['score']:+.3f}` `{_bar(r['score'])}`")
        lines.append(f"- **Signals:** {r['n_signals']}")
        drivers = summarize_drivers(r, top=5)
        if drivers:
            lines.append("- **Top drivers:**")
            for d in drivers:
                sign     = "▲" if d["contribution"] >= 0 else "▼"
                contrib  = abs(d["contribution"])
                w        = d.get("weight", 0.0)
                intens   = d.get("intensity", 0.0)
                avg_sent = d.get("avg_sentiment", 0.0)
                n_art    = d.get("n_articles", 0)
                vf       = d.get("volume_factor", 1.0)
                title    = (d.get("top_title") or "").replace("|", "·").strip()
                url      = d.get("top_url", "")
                src      = d.get("top_source", "")
                head     = f"[{title}]({url})" if url and title else (title or "—")
                lines.append(
                    f"  - `{d['theme']}` {sign} `{contrib:.3f}` "
                    f"· weight {w:+.2f} · intensity {intens:.2f} "
                    f"· avg_sent {avg_sent:+.2f} · {n_art} articles (vol={vf:.2f})"
                    f"\n    → {head} _({src})_"
                )
        lines.append("")

    # ── Methodology footnote ─────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(
        "**Methodology.** Articles are grouped by macro theme. "
        "For **event themes** (rate_hike, inflation, geopolitical, recession, …) "
        "intensity = |avg FinBERT sentiment| — the direction is fixed by the "
        "asset_map weight regardless of article framing. "
        "For **topical themes** (ihsg, rupiah, earnings, …) "
        "signed sentiment is used directly. "
        "Volume weight (log-scaled, saturates at 15 articles) reduces noise from "
        "single-source themes. "
        "Per-asset score = tanh(Σ weight × intensity × confidence × volume). "
        "|score| ≥ 0.45 → STRONG, ≥ 0.20 → MODERATE, ≥ 0.08 → MILD, else FLAT."
    )
    lines.append("")
    return "\n".join(lines)


def save_report(scored: dict[str, dict],
                articles: list[dict],
                ts: datetime | None = None,
                directory: str = "reports") -> str:
    """Render and write the report.  Returns the file path."""
    ts = ts or datetime.utcnow()
    os.makedirs(directory, exist_ok=True)

    previous = _load_previous_scores(directory, ts)
    body = render_report(scored, articles, ts=ts, previous_scores=previous)

    fname = f"narrative_impact_{ts.strftime('%Y-%m-%d_%H%M')}.md"
    path  = os.path.join(directory, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"[Report] wrote {path}")

    # Save companion scores JSON for next-run delta comparison
    scores_path = _scores_path(directory, ts)
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2)

    return path
