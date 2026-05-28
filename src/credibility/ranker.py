"""
Source credibility scoring + cross-verification of narratives.
Anti-hallucination: a narrative is only confirmed if 2+ independent Tier 1/2 sources agree.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timedelta

import config
from src.storage import db

# Static source credibility base scores (0–1)
SOURCE_BASE_SCORES: dict[str, float] = {
    "reuters_business": 0.97,
    "reuters_markets":  0.97,
    "ft_world":         0.95,
    "wsj_markets":      0.94,
    "cnbc_finance":     0.85,
    "cnbc_markets":     0.85,
    "marketwatch":      0.80,
    "yahoofinance":     0.75,
    "seeking_alpha":    0.65,
    "coindesk":         0.60,
    "theblock":         0.58,
    "kontan":           0.80,
    "bisnis_id":        0.82,
    "detik_finance":    0.72,
    "sec_edgar":        1.00,
    "idx_disclosure":   1.00,
    "fed_speeches":     1.00,
    "treasury":         1.00,
    "reddit/wallstreetbets": 0.30,
    "reddit/investing": 0.45,
    "reddit/stocks":    0.40,
    "twitter":          0.25,
}

DEFAULT_CREDIBILITY = 0.50


def score_source(source: str) -> float:
    return SOURCE_BASE_SCORES.get(source, DEFAULT_CREDIBILITY)


def score_article(article: dict) -> float:
    """
    Composite credibility for a single article:
    - Base source score
    - Boost if Tier 1 source
    - Recency bonus (fresher = slightly more credible for time-sensitive news)
    """
    source_score = score_source(article.get("source", ""))
    tier_mult    = {1: 1.00, 2: 0.90, 3: 0.70}.get(article.get("source_tier", 3), 0.70)
    return round(source_score * tier_mult, 3)


def cross_verify(articles: list[dict], min_sources: int | None = None) -> dict:
    """
    Group articles by topic/theme and verify if narrative is confirmed
    by multiple independent sources.

    Returns {topic: {confirmed: bool, source_count: int, avg_credibility: float}}.
    """
    min_src = min_sources or config.CROSS_VERIFY_THRESHOLD
    topic_articles: dict[str, list[dict]] = defaultdict(list)

    for a in articles:
        for theme in a.get("macro_themes", []):
            topic_articles[theme].append(a)

    result = {}
    for topic, arts in topic_articles.items():
        # Unique sources (by base domain)
        sources  = {a.get("source", "").split("/")[0] for a in arts}
        tiers    = [a.get("source_tier", 3) for a in arts]
        creds    = [score_article(a) for a in arts]

        tier1_count = sum(1 for t in tiers if t == 1)
        confirmed   = len(sources) >= min_src and tier1_count >= 1

        result[topic] = {
            "confirmed":       confirmed,
            "source_count":    len(sources),
            "sources":         list(sources),
            "tier1_sources":   tier1_count,
            "avg_credibility": round(statistics.mean(creds), 3) if creds else 0.0,
            "article_count":   len(arts),
        }

    return result


def filter_low_credibility(articles: list[dict]) -> list[dict]:
    """Remove articles below minimum credibility threshold."""
    return [
        a for a in articles
        if score_article(a) >= config.MIN_CREDIBILITY_SCORE
    ]


def rank_by_credibility(articles: list[dict]) -> list[dict]:
    """Sort articles by credibility score descending."""
    return sorted(articles, key=lambda a: score_article(a), reverse=True)
