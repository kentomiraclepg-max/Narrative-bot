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


# ── Narrative summary generator ───────────────────────────────────────────────

_THEME_LABELS: dict[str, str] = {
    "rate_hike":      "Fed rate hike",
    "rate_cut":       "Fed rate cut",
    "inflation":      "inflation pressure",
    "recession":      "recession fears",
    "earnings":       "earnings season",
    "geopolitical":   "geopolitical tensions",
    "liquidity":      "liquidity/credit risk",
    "supply_chain":   "supply chain disruption",
    "currency":       "currency moves",
    "bi_rate":        "BI rate policy",
    "ihsg":           "IHSG movement",
    "rupiah":         "Rupiah movement",
    "commodity":      "commodity prices",
    "ai_tech":        "AI/tech developments",
    "oil_price":      "oil prices",
    "ojk_regulation": "OJK regulation",
    "bbm_harga":      "fuel price (BBM)",
    "us_debt":        "US debt/fiscal",
    "etf_flows":      "crypto ETF flows",
    "regulation":     "crypto regulation",
    "hack_exploit":   "hack/exploit incident",
    "halving":        "Bitcoin halving",
    "stablecoin":     "stablecoin volatility",
}


def _narrative_summary(asset: str, result: dict, drivers: list[dict]) -> str:
    """Generate a human-readable 2-3 sentence narrative for an asset."""
    if not drivers:
        return "_No significant news signals detected._"

    label    = result["label"]
    score    = result["score"]
    n_themes = len(drivers)

    bullish = [d for d in drivers if d["contribution"] > 0]
    bearish = [d for d in drivers if d["contribution"] < 0]

    main_side  = bullish if label == "BULLISH" else bearish if label == "BEARISH" else drivers
    other_side = bearish if label == "BULLISH" else bullish if label == "BEARISH" else []

    def _driver_desc(d: dict) -> str:
        theme   = _THEME_LABELS.get(d["theme"], d["theme"].replace("_", " "))
        n       = d["n_articles"]
        title   = (d.get("top_title") or "").strip()
        source  = d.get("top_source", "")
        snippet = f' — "{title[:65]}..."' if title else ""
        src_ref = f" ({source})" if source else ""
        art_str = f"{n} article{'s' if n != 1 else ''}{snippet}{src_ref}"
        return f"**{theme}** [{art_str}]"

    sentences: list[str] = []

    # Sentence 1 — main direction
    if main_side:
        top = main_side[0]
        direction_word = (
            "Primary bullish driver" if label == "BULLISH" else
            "Primary bearish driver" if label == "BEARISH" else
            "Strongest signal"
        )
        sentences.append(f"{direction_word}: {_driver_desc(top)}.")

    # Sentence 2 — secondary main drivers
    if len(main_side) >= 2:
        secondary = [_driver_desc(d) for d in main_side[1:3]]
        sentences.append(f"Reinforced by {' and '.join(secondary)}.")

    # Sentence 3 — counter narrative
    if other_side:
        top_other = other_side[0]
        counter_word = "Headwind" if label == "BULLISH" else "Partial support"
        sentences.append(
            f"{counter_word}: {_driver_desc(top_other)} "
            f"({'dampening' if label == 'BULLISH' else 'supporting'} "
            f"{'bullish' if label == 'BULLISH' else 'bearish'} bias)."
        )

    # If pure neutral with mixed signals
    if label == "NEUTRAL" and bullish and bearish:
        sentences = [
            f"Conflicting signals — {_driver_desc(bullish[0])} "
            f"offset by {_driver_desc(bearish[0])}.",
            f"Balance of {n_themes} themes yields neutral bias (score {score:+.3f}).",
        ]

    return " ".join(sentences)


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
        drivers = summarize_drivers(r, top=5)
        # Narrative summary — human-readable interpretation of drivers
        lines.append(f"> {_narrative_summary(asset, r, r.get('drivers', []))}")
        lines.append("")
        lines.append(f"- **Bias:** {r['label']} ({r['strength']})")
        lines.append(f"- **Score:** `{r['score']:+.3f}` `{_bar(r['score'])}`")
        lines.append(f"- **Signals:** {r['n_signals']}")
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
