"""
web_ingestor.py — Real-time narrative pulls via Google News RSS search.

Why this exists:
  The base `src/ingestion/rss.py` polls a fixed list of homepage feeds
  (Reuters Business, CNBC Markets, etc.).  For *narrative impact on a
  specific asset class* we need topic-targeted, freshly-updated headlines —
  e.g. "Federal Reserve rate decision", "IHSG hari ini", "bitcoin ETF flows".

  Google News RSS exposes a search-based feed at
      https://news.google.com/rss/search?q={query}&hl={lang}&gl={country}
  that aggregates the latest articles across all major outlets.  This module
  defines per-asset query bundles, pulls them, and returns article dicts in
  the same shape as `rss.fetch_feed()` so the rest of the pipeline
  (credibility filter → NLP → scorer → reporter) is unchanged.
"""
from __future__ import annotations

import hashlib
import re
import urllib.parse
from datetime import datetime
from typing import Iterable

import feedparser

from src.storage import db


# ── Per-asset narrative queries ──────────────────────────────────────────────
ASSET_QUERIES: dict[str, list[tuple[str, str]]] = {
    # asset → list of (query, lang_country) tuples
    "BTC": [
        ("bitcoin price today",                   "en-US"),
        ("bitcoin ETF flows institutional",       "en-US"),
        ("crypto regulation SEC CFTC",            "en-US"),
        ("bitcoin halving on-chain",              "en-US"),
    ],
    "US_STOCKS": [
        ("S&P 500 today",                         "en-US"),
        ("Federal Reserve FOMC rate decision",    "en-US"),
        ("US inflation CPI report",               "en-US"),
        ("US earnings season guidance",           "en-US"),
        ("Nasdaq tech stocks",                    "en-US"),
    ],
    "IDX": [
        ("IHSG hari ini",                         "id-ID"),
        ("Bank Indonesia BI rate suku bunga",     "id-ID"),
        ("rupiah dollar kurs",                    "id-ID"),
        ("komoditas nikel batu bara CPO",         "id-ID"),
        ("OJK bursa efek Indonesia",              "id-ID"),
    ],
    "XAUUSD": [
        ("gold price today XAUUSD",               "en-US"),
        ("central bank gold buying",              "en-US"),
        ("safe haven demand geopolitical",        "en-US"),
    ],
    "FX": [
        ("US dollar index DXY",                   "en-US"),
        ("EUR USD ECB policy",                    "en-US"),
        ("yen Bank of Japan intervention",        "en-US"),
        ("emerging market currencies",            "en-US"),
    ],
}

# Light source-credibility heuristic by domain
DOMAIN_TIER: dict[str, int] = {
    "reuters.com": 1, "ft.com": 1, "wsj.com": 1, "bloomberg.com": 1,
    "apnews.com": 1, "ap.org": 1, "economist.com": 1,
    "cnbc.com": 2, "marketwatch.com": 2, "yahoo.com": 2, "barrons.com": 2,
    "businessinsider.com": 2, "investing.com": 2, "forbes.com": 2,
    "coindesk.com": 3, "theblock.co": 3, "cointelegraph.com": 3,
    "kontan.co.id": 2, "bisnis.com": 2, "detik.com": 2, "cnbcindonesia.com": 2,
    "tempo.co": 2, "kompas.com": 2,
}


def _article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _domain(url: str) -> str:
    try:
        netloc = urllib.parse.urlparse(url).netloc.lower()
        return netloc.lstrip("www.")
    except Exception:
        return ""


def _domain_tier(url: str) -> int:
    d = _domain(url)
    for key, tier in DOMAIN_TIER.items():
        if key in d:
            return tier
    return 3


def _build_url(query: str, lang_country: str) -> str:
    lang, country = lang_country.split("-")
    q  = urllib.parse.quote_plus(query)
    return (f"https://news.google.com/rss/search?q={q}"
            f"&hl={lang}-{country}&gl={country}&ceid={country}:{lang}")


def _market_for_asset(asset: str) -> str:
    return {"IDX": "IDX", "BTC": "CRYPTO"}.get(asset, "GLOBAL")


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _fetch_raw(url: str) -> str | None:
    try:
        import requests as _req
        r = _req.get(url, headers={"User-Agent": _BROWSER_UA},
                     timeout=15, allow_redirects=True)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def fetch_query(query: str, lang_country: str = "en-US",
                asset_tag: str | None = None, limit: int = 15) -> list[dict]:
    """
    Run a single Google News RSS query and return new article dicts.
    Persists each into the canonical `articles` table; only *newly-inserted*
    rows are returned (to avoid reprocessing).
    """
    url = _build_url(query, lang_country)
    try:
        raw = _fetch_raw(url)
        if raw is None:
            return []
        feed = feedparser.parse(raw)
    except Exception as e:
        print(f"[WebIngest] '{query}' parse error: {e}")
        return []

    articles: list[dict] = []
    for entry in feed.entries[:limit]:
        link = getattr(entry, "link", "") or ""
        if not link:
            continue
        title    = getattr(entry, "title",   "") or ""
        summary  = _strip_html(getattr(entry, "summary", "")
                               or getattr(entry, "description", ""))
        raw_text = (title + " " + summary)[:3000]

        tier = _domain_tier(link)
        article = {
            "id":          _article_id(link),
            "source":      f"gnews:{_domain(link) or 'unknown'}",
            "source_tier": tier,
            "title":       title[:500],
            "summary":     summary[:2000],
            "url":         link,
            "published":   _parse_date(entry),
            "market":      _market_for_asset(asset_tag) if asset_tag else "GLOBAL",
            "lang":        lang_country.split("-")[0],
            "raw_text":    raw_text,
            "credibility": 1.0 - (tier - 1) * 0.2,
            "asset_tag":   asset_tag,     # carried in-memory, not persisted
            "query":       query,
        }
        is_new = db.upsert_article(article)
        if is_new:
            articles.append(article)
    return articles


def _parse_date(entry) -> str:
    for field in ("published_parsed", "updated_parsed"):
        val = getattr(entry, field, None)
        if val:
            try:
                return datetime(*val[:6]).isoformat()
            except Exception:
                pass
    return datetime.utcnow().isoformat()


def fetch_for_asset(asset: str, limit_per_query: int = 10) -> list[dict]:
    """Run every query bundled for `asset` and return combined new articles."""
    queries = ASSET_QUERIES.get(asset, [])
    out: list[dict] = []
    for q, lc in queries:
        try:
            new = fetch_query(q, lc, asset_tag=asset, limit=limit_per_query)
            if new:
                print(f"[WebIngest] {asset} | '{q}' (+{len(new)})")
            out.extend(new)
        except Exception as e:
            print(f"[WebIngest] {asset} '{q}' error: {e}")
    return out


def fetch_all_assets(assets: Iterable[str] | None = None,
                     limit_per_query: int = 10) -> list[dict]:
    """Run all queries for the given assets (defaults to every asset bundle)."""
    targets = list(assets) if assets is not None else list(ASSET_QUERIES.keys())
    all_new: list[dict] = []
    for a in targets:
        all_new.extend(fetch_for_asset(a, limit_per_query=limit_per_query))
    print(f"[WebIngest] total new: {len(all_new)} (assets={targets})")
    return all_new


# ── Extra theme classifiers (beyond MACRO_KEYWORDS in nlp/pipeline.py) ───────
_EXTRA_THEME_PATTERNS: dict[str, list[str]] = {
    "etf_flows":    ["etf inflow", "etf outflow", "spot etf", "blackrock", "fidelity"],
    "regulation":   ["sec", "cftc", "regulator", "lawsuit", "settlement", "fines"],
    "hack_exploit": ["hack", "exploit", "stolen", "breach", "drained"],
    "halving":      ["halving", "block reward"],
    "stablecoin":   ["stablecoin", "tether", "usdt", "usdc", "depeg"],
}


def classify_extra_themes(text: str) -> list[str]:
    """Theme classifier for narratives missed by the core MACRO_KEYWORDS list."""
    t = (text or "").lower()
    found: list[str] = []
    for theme, kws in _EXTRA_THEME_PATTERNS.items():
        if any(kw in t for kw in kws):
            found.append(theme)
    return found
