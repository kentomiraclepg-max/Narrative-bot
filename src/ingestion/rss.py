"""
RSS feed scraper: Reuters, CNBC, WSJ, FT, MarketWatch, Yahoo Finance,
Indonesian media (Kontan, Bisnis.com, Detik Finance).
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser
import requests

import config
from src.storage import db


def _article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _parse_date(entry) -> str:
    for field in ("published_parsed", "updated_parsed"):
        val = getattr(entry, field, None)
        if val:
            try:
                return datetime(*val[:6]).isoformat()
            except Exception:
                pass
    return datetime.utcnow().isoformat()


def _detect_market(source: str, title: str, summary: str) -> str:
    text = (title + " " + summary).lower()
    if any(k in source for k in ("kontan", "bisnis_id", "detik")):
        return "IDX"
    if any(k in text for k in ("indonesia", "ihsg", "rupiah", "bi rate", "ojk", "idx")):
        return "IDX"
    if any(k in text for k in ("bitcoin", "ethereum", "crypto", "defi")):
        return "CRYPTO"
    return "GLOBAL"


def _detect_lang(source: str) -> str:
    if any(k in source for k in ("kontan", "bisnis_id", "detik")):
        return "id"
    return "en"


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Per-source title prefixes to strip (wire services prepend their own name)
_TITLE_PREFIXES: dict[str, str] = {
    "financialjuice": "FinancialJuice: ",
}


def _fetch_raw(url: str, retries: int = 2) -> str | None:
    """Pre-fetch URL with browser UA. Retries once on 429 with backoff."""
    import time
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": _BROWSER_UA},
                             timeout=15, allow_redirects=True)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429 and attempt < retries:
                time.sleep(5 * (attempt + 1))
                continue
            return None
        except Exception:
            return None
    return None


def fetch_feed(name: str, url: str) -> list[dict]:
    """Fetch and parse a single RSS feed. Returns new articles."""
    try:
        raw = _fetch_raw(url)
        if raw is None:
            return []
        feed = feedparser.parse(raw)
    except Exception as e:
        print(f"[RSS] {name}: parse error {e}")
        return []

    tier = config.SOURCE_TIERS.get(name, 3)
    # Wire services (tier 1) publish many short items — fetch more of them
    max_entries = 100 if tier == 1 else 30
    articles = []

    for entry in feed.entries[:max_entries]:
        link = getattr(entry, "link", "") or ""
        if not link:
            continue

        title   = getattr(entry, "title",   "") or ""
        prefix  = _TITLE_PREFIXES.get(name, "")
        if prefix and title.startswith(prefix):
            title = title[len(prefix):]
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        raw_text = title + " " + summary

        article = {
            "id":          _article_id(link),
            "source":      name,
            "source_tier": tier,
            "title":       title[:500],
            "summary":     summary[:2000],
            "url":         link,
            "published":   _parse_date(entry),
            "market":      _detect_market(name, title, summary),
            "lang":        _detect_lang(name),
            "raw_text":    raw_text[:3000],
            "credibility": 1.0 - (tier - 1) * 0.2,   # tier 1→1.0, tier 2→0.8, tier 3→0.6
        }
        is_new = db.upsert_article(article)
        if is_new:
            articles.append(article)

    return articles


def fetch_all() -> list[dict]:
    """Fetch all configured RSS feeds. Returns all new articles."""
    all_new = []
    for name, url in config.RSS_FEEDS.items():
        try:
            new = fetch_feed(name, url)
            if new:
                print(f"[RSS] {name}: +{len(new)} new articles")
            all_new.extend(new)
        except Exception as e:
            print(f"[RSS] {name}: error {e}")
    return all_new


def fetch_by_market(market: str) -> list[dict]:
    """Fetch only feeds relevant to a given market."""
    market_sources = {
        "IDX":    ["kontan", "bisnis_id", "detik_finance"],
        "CRYPTO": ["coindesk", "theblock"],
        "GLOBAL": list(config.RSS_FEEDS.keys()),
    }
    sources = market_sources.get(market, list(config.RSS_FEEDS.keys()))
    all_new = []
    for name in sources:
        url = config.RSS_FEEDS.get(name)
        if url:
            all_new.extend(fetch_feed(name, url))
    return all_new
