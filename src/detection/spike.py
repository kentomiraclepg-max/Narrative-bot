"""
Anomaly & narrative spike detection.
- Sentiment spike: sudden shift in avg sentiment within rolling window
- Volume spike: article count surge on a topic
- Sentiment divergence: conflicting signals across sources
"""
from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import config
from src.storage import db


def _now() -> datetime:
    return datetime.utcnow()


# ── Volume spike detection ────────────────────────────────────────────────────

def detect_volume_spike(articles: list[dict], window_minutes: int | None = None) -> list[dict]:
    """
    Detect topics where article volume is anomalously high in the rolling window.
    """
    window = window_minutes or config.SPIKE_WINDOW_MINUTES
    cutoff = _now() - timedelta(minutes=window)

    # Count articles per topic (macro theme)
    topic_counts: dict[str, list[dict]] = defaultdict(list)
    for a in articles:
        themes = a.get("macro_themes", [])
        pub    = a.get("published", "")
        try:
            pub_dt = datetime.fromisoformat(pub)
        except Exception:
            pub_dt = _now()

        if pub_dt >= cutoff:
            for theme in themes:
                topic_counts[theme].append(a)

    spikes = []
    for topic, topic_articles in topic_counts.items():
        count = len(topic_articles)
        if count >= config.MIN_ARTICLES_FOR_SPIKE:
            scores = [a.get("sentiment_score", 0.0) for a in topic_articles
                      if a.get("sentiment_score") is not None]
            avg_sent = statistics.mean(scores) if scores else 0.0
            spikes.append({
                "type":          "volume_spike",
                "topic":         topic,
                "article_count": count,
                "window_minutes": window,
                "avg_sentiment": round(avg_sent, 4),
                "severity":      "HIGH" if count >= config.MIN_ARTICLES_FOR_SPIKE * 2 else "MEDIUM",
                "articles":      [a["id"] for a in topic_articles],
                "detected_at":   _now().isoformat(),
            })

    return spikes


# ── Sentiment spike detection ─────────────────────────────────────────────────

def detect_sentiment_spike(topic: str, hours: int = 24) -> dict | None:
    """
    Compare recent vs baseline sentiment for a topic.
    Uses Z-score: spike if |z| > SPIKE_Z_THRESHOLD.
    """
    history = db.get_sentiment_history(topic, hours=hours)
    if len(history) < 3:
        return None

    scores = [h["score"] for h in history]
    if len(scores) < 3:
        return None

    # Baseline = older 80%, recent = newest 20%
    split  = max(1, len(scores) * 8 // 10)
    base   = scores[:split]
    recent = scores[split:]

    if len(base) < 2:
        return None

    mean_base = statistics.mean(base)
    std_base  = statistics.stdev(base) or 0.001
    mean_recent = statistics.mean(recent)

    z_score = (mean_recent - mean_base) / std_base

    if abs(z_score) < config.SPIKE_Z_THRESHOLD:
        return None

    direction = "bullish_surge" if z_score > 0 else "bearish_surge"
    severity  = "HIGH" if abs(z_score) > config.SPIKE_Z_THRESHOLD * 1.5 else "MEDIUM"

    return {
        "type":         "sentiment_spike",
        "topic":        topic,
        "direction":    direction,
        "z_score":      round(z_score, 3),
        "mean_base":    round(mean_base, 4),
        "mean_recent":  round(mean_recent, 4),
        "severity":     severity,
        "detected_at":  _now().isoformat(),
    }


# ── Sentiment divergence detection ───────────────────────────────────────────

def detect_sentiment_divergence(articles: list[dict]) -> list[dict]:
    """
    Detect when different source tiers report conflicting sentiment on same topic.
    Example: Tier 1 (Reuters) bearish while Reddit bullish → divergence.
    """
    topic_by_tier: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for a in articles:
        score  = a.get("sentiment_score")
        tier   = a.get("source_tier", 3)
        themes = a.get("macro_themes", [])
        if score is None:
            continue
        for theme in themes:
            topic_by_tier[theme][tier].append(score)

    divergences = []
    for topic, tier_scores in topic_by_tier.items():
        tier_avgs = {tier: statistics.mean(scores)
                     for tier, scores in tier_scores.items()
                     if scores}

        if len(tier_avgs) < 2:
            continue

        avgs  = list(tier_avgs.values())
        spread = max(avgs) - min(avgs)

        if spread >= config.DIVERGENCE_THRESHOLD:
            divergences.append({
                "type":       "sentiment_divergence",
                "topic":      topic,
                "tier_avgs":  {str(k): round(v, 4) for k, v in tier_avgs.items()},
                "spread":     round(spread, 4),
                "severity":   "HIGH" if spread > config.DIVERGENCE_THRESHOLD * 1.5 else "MEDIUM",
                "detected_at": _now().isoformat(),
                "interpretation": _interpret_divergence(tier_avgs),
            })

    return divergences


def _interpret_divergence(tier_avgs: dict[int, float]) -> str:
    t1 = tier_avgs.get(1)
    t3 = tier_avgs.get(3)
    if t1 is None or t3 is None:
        return "mixed signals"
    if t1 < 0 and t3 > 0:
        return "mainstream bearish, retail/social bullish — potential fear vs euphoria"
    if t1 > 0 and t3 < 0:
        return "mainstream bullish, retail/social bearish — potential smart money vs crowd divergence"
    return "mixed signals across tiers"


# ── Narrative momentum ────────────────────────────────────────────────────────

def compute_narrative_momentum(topic: str, hours: int = 24) -> float:
    """
    Momentum = rate of change of sentiment × volume.
    Positive = narrative accelerating bullish.
    Negative = narrative accelerating bearish.
    """
    history = db.get_sentiment_history(topic, hours=hours)
    if len(history) < 4:
        return 0.0

    # Split into two halves
    mid = len(history) // 2
    h1  = [h["score"] for h in history[:mid]]
    h2  = [h["score"] for h in history[mid:]]

    if not h1 or not h2:
        return 0.0

    delta = statistics.mean(h2) - statistics.mean(h1)
    vol_weight = min(len(history) / 20, 1.0)   # more data = higher confidence
    return round(delta * vol_weight, 4)


# ── Run all detections ────────────────────────────────────────────────────────

def run_all(articles: list[dict]) -> list[dict]:
    """Run all detectors on a batch of processed articles. Returns anomaly list."""
    anomalies = []
    anomalies.extend(detect_volume_spike(articles))
    anomalies.extend(detect_sentiment_divergence(articles))

    # Sentiment spikes for known themes
    from src.nlp.pipeline import MACRO_KEYWORDS, IDX_KEYWORDS
    for topic in list(MACRO_KEYWORDS.keys()) + list(IDX_KEYWORDS.keys()):
        spike = detect_sentiment_spike(topic)
        if spike:
            anomalies.append(spike)

    return anomalies
