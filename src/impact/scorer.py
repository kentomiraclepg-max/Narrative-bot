"""
scorer.py — Aggregate NLP-enriched articles into per-asset directional bias.

Core design changes vs the original:
  - Theme-level aggregation first (eliminates within-theme self-cancellation).
  - EVENT_THEMES use |avg_sentiment| as intensity — the directional impact is
    fixed by the asset_map weight regardless of article framing.
    e.g. "Gold climbs on geopolitical risk" (FinBERT +0.90) and
         "Iran conflict losses" (FinBERT -0.86) both correctly signal
         geopolitical risk is present → gold bullish, stocks bearish.
  - TOPICAL_THEMES use signed sentiment (positive article = positive outcome
    for the asset per the weight convention).
  - Volume-weighted: more articles on a theme = stronger signal, with
    diminishing returns above VOLUME_REFERENCE articles.
  - Final score = tanh(sum of theme contributions) → stays in [-1, +1],
    no longer collapses to zero when equal-and-opposite drivers exist.
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any

from src.impact.asset_map import ASSETS, THEME_IMPACT

# ── Theme classification ──────────────────────────────────────────────────────
# EVENT_THEMES: directional impact is fixed by the weight in asset_map.
# The article may frame it positively ("gold climbs because of geopolitics")
# or negatively ("conflict causes losses") — both indicate the event is present.
# We use |avg_sentiment| as intensity so framing doesn't flip the signal.
EVENT_THEMES: frozenset[str] = frozenset({
    "rate_hike", "rate_cut", "inflation", "recession",
    "liquidity", "supply_chain", "geopolitical",
    "regulation", "hack_exploit", "halving",
})

# TOPICAL_THEMES: signed sentiment reflects the direction directly.
# Positive article about earnings → bullish stocks. Positive rupiah news →
# rupiah strengthens (USDIDR down). etc.
TOPICAL_THEMES: frozenset[str] = frozenset({
    "earnings", "currency", "bi_rate", "ihsg",
    "rupiah", "commodity", "etf_flows", "stablecoin",
})

# Volume reference: how many articles before a theme is considered
# "fully covered". Fewer articles → signal is discounted.
_VOLUME_REFERENCE = 15


def _tier_boost(tier: int | None) -> float:
    return {1: 1.0, 2: 0.85, 3: 0.7}.get(tier or 3, 0.7)


def _volume_factor(n: int) -> float:
    """Logarithmic volume weight: 1 article → ~0.25, 15 → 1.0, 30 → capped."""
    return min(1.0, math.log2(1 + n) / math.log2(1 + _VOLUME_REFERENCE))


def _classify_strength(abs_score: float) -> str:
    if abs_score >= 0.45:
        return "STRONG"
    if abs_score >= 0.20:
        return "MODERATE"
    if abs_score >= 0.08:
        return "MILD"
    return "FLAT"


def _classify_label(score: float) -> str:
    if score >= 0.08:
        return "BULLISH"
    if score <= -0.08:
        return "BEARISH"
    return "NEUTRAL"


def _top_article(arts: list[dict]) -> dict:
    """Return the highest-credibility article as the theme representative."""
    return max(arts, key=lambda a: float(a.get("credibility") or 0.0),
               default={})


def score_assets(articles: list[dict],
                 assets: list[str] | None = None) -> dict[str, dict]:
    """
    Score each asset against the NLP-enriched article corpus.

    Each article should have:
      macro_themes:    list[str]
      sentiment_score: float in [-1, 1]
      credibility:     float in [0, 1]
      source_tier:     int 1..3
    """
    targets = assets or ASSETS

    # ── Step 1: Group articles by theme ──────────────────────────────────────
    theme_buckets: dict[str, list[dict]] = defaultdict(list)
    for art in articles:
        for theme in (art.get("macro_themes") or []):
            if theme in THEME_IMPACT:
                theme_buckets[theme].append(art)

    # ── Step 2: Per-theme aggregate signal ───────────────────────────────────
    theme_signals: dict[str, dict] = {}
    for theme, arts in theme_buckets.items():
        sentiments  = [float(a.get("sentiment_score") or 0.0) for a in arts]
        confidences = [
            float(a.get("credibility") or 0.5) * _tier_boost(a.get("source_tier"))
            for a in arts
        ]
        avg_sent = mean(sentiments)
        avg_conf = mean(confidences)
        n        = len(arts)
        vf       = _volume_factor(n)

        # Event themes: magnitude of discussion signals strength regardless
        # of how the article frames it. Weight determines direction.
        # Topical themes: signed sentiment (positive = the "good" outcome).
        if theme in EVENT_THEMES:
            intensity = abs(avg_sent)
        else:
            intensity = avg_sent

        theme_signals[theme] = {
            "intensity":     intensity,
            "avg_conf":      avg_conf,
            "avg_sentiment": avg_sent,
            "volume_factor": vf,
            "n_articles":    n,
            "articles":      arts,
        }

    # ── Step 3: Per-asset contributions ──────────────────────────────────────
    per_asset_drivers: dict[str, list[dict]] = {a: [] for a in targets}

    for theme, sig in theme_signals.items():
        weights = THEME_IMPACT.get(theme, {})
        for asset in targets:
            w = weights.get(asset, 0.0)
            if w == 0.0:
                continue
            contrib = w * sig["intensity"] * sig["avg_conf"] * sig["volume_factor"]
            top     = _top_article(sig["articles"])
            per_asset_drivers[asset].append({
                "theme":        theme,
                "contribution": round(contrib, 4),
                "weight":       w,
                "intensity":    round(sig["intensity"], 3),
                "avg_sentiment": round(sig["avg_sentiment"], 3),
                "confidence":   round(sig["avg_conf"], 3),
                "volume_factor": round(sig["volume_factor"], 2),
                "n_articles":   sig["n_articles"],
                "top_title":    top.get("title", ""),
                "top_url":      top.get("url", ""),
                "top_source":   top.get("source", ""),
            })

    # ── Step 4: Aggregate and classify ───────────────────────────────────────
    out: dict[str, dict] = {}
    for asset, drivers in per_asset_drivers.items():
        if not drivers:
            out[asset] = {
                "score":     0.0,
                "label":     "NEUTRAL",
                "strength":  "FLAT",
                "n_signals": 0,
                "drivers":   [],
            }
            continue

        raw_sum = sum(d["contribution"] for d in drivers)
        # tanh(raw_sum × 2) maps the sum to [-1,+1] while preserving sign.
        # Scale factor 2 spreads the signal; typical raw_sum ∈ [-0.5, +0.5].
        score = round(math.tanh(raw_sum * 2), 4)
        out[asset] = {
            "score":     score,
            "label":     _classify_label(score),
            "strength":  _classify_strength(abs(score)),
            "n_signals": sum(d["n_articles"] for d in drivers),
            "raw_sum":   round(raw_sum, 4),
            "drivers":   sorted(drivers, key=lambda d: abs(d["contribution"]),
                                reverse=True),
        }

    return out


def summarize_drivers(asset_result: dict, top: int = 5) -> list[dict]:
    return asset_result.get("drivers", [])[:top]


def aggregate_themes(articles: list[dict]) -> list[dict]:
    """Roll up dominant themes by article count with avg sentiment."""
    bucket: dict[str, list[float]] = {}
    for art in articles:
        for t in art.get("macro_themes") or []:
            bucket.setdefault(t, []).append(float(art.get("sentiment_score") or 0.0))
    rolled = [
        {
            "theme":       t,
            "n_articles":  len(scores),
            "avg_sentiment": round(mean(scores), 3) if scores else 0.0,
            "is_event":    t in EVENT_THEMES,
        }
        for t, scores in bucket.items()
    ]
    rolled.sort(key=lambda r: r["n_articles"], reverse=True)
    return rolled
