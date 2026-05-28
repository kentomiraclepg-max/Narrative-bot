"""
Regulatory disclosures: SEC EDGAR (8-K filings), IDX announcements,
Fed/FOMC speeches, Treasury press releases.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta

import feedparser
import requests
from bs4 import BeautifulSoup

import config
from src.storage import db


def _art_id(source: str, key: str) -> str:
    return hashlib.md5(f"{source}_{key}".encode()).hexdigest()


# ── SEC EDGAR ─────────────────────────────────────────────────────────────────

def fetch_sec_8k(days_back: int = 1) -> list[dict]:
    """Fetch recent 8-K filings from SEC EDGAR full-text search."""
    end   = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": "material event",
        "dateRange": "custom",
        "startdt": start,
        "enddt":   end,
        "forms":   "8-K",
    }
    headers = {"User-Agent": "NarrativeBot/1.0 contact@example.com"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[SEC] 8-K fetch error: {e}")
        return []

    articles = []
    for hit in data.get("hits", {}).get("hits", [])[:20]:
        src  = hit.get("_source", {})
        link = f"https://www.sec.gov/Archives/edgar/full-index/{src.get('file_date','').replace('-','/')}/company.idx"
        article = {
            "id":          _art_id("sec_8k", src.get("entity_id", hit["_id"])),
            "source":      "sec_edgar",
            "source_tier": 1,
            "title":       f"SEC 8-K: {src.get('display_names', [''])[0] if src.get('display_names') else ''}",
            "summary":     src.get("period_of_report", ""),
            "url":         f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={src.get('entity_id','')}&type=8-K",
            "published":   src.get("file_date", datetime.utcnow().isoformat()),
            "market":      "US",
            "lang":        "en",
            "raw_text":    src.get("period_of_report", ""),
            "credibility": 1.0,
        }
        if db.upsert_article(article):
            articles.append(article)

    return articles


# ── IDX Announcements ─────────────────────────────────────────────────────────

def fetch_idx_announcements(limit: int = 30) -> list[dict]:
    """Fetch IDX corporate announcements (keterbukaan informasi)."""
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.idx.co.id/",
    }
    params = {
        "language": "id",
        "indexFrom": 0,
        "pageSize": limit,
        "category": 1,
    }
    fallback_urls = [
        config.IDX_ANNOUNCE_URL,
        "https://www.idx.co.id/umum/GetAnnouncement",
        "https://www.idx.co.id/api/RINB/GetAnnouncement",
    ]
    r = None
    for url in fallback_urls:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code != 403:
                break
        except Exception:
            continue
    try:
        if r is None or r.status_code == 403:
            raise Exception(f"All IDX endpoints returned 403")
        r.raise_for_status()
        data = r.json()
        items = data.get("results", data if isinstance(data, list) else [])
    except Exception as e:
        print(f"[IDX] Announcement fetch: {e}")
        return []

    articles = []
    for item in items:
        title = item.get("Judul") or item.get("Title") or item.get("title", "")
        code  = item.get("KodeEmiten") or item.get("StockCode", "")
        url   = item.get("Attachment") or item.get("attachment", "")
        pub   = item.get("Date") or item.get("date", datetime.utcnow().isoformat())

        article = {
            "id":          _art_id("idx", title + pub),
            "source":      "idx_disclosure",
            "source_tier": 1,
            "title":       f"[{code}] {title}"[:500],
            "summary":     item.get("Ringkasan", "")[:2000],
            "url":         url or f"https://www.idx.co.id",
            "published":   pub,
            "market":      "IDX",
            "lang":        "id",
            "raw_text":    title,
            "credibility": 1.0,
        }
        if db.upsert_article(article):
            articles.append(article)

    return articles


# ── Fed / FOMC ────────────────────────────────────────────────────────────────

def fetch_fed_speeches() -> list[dict]:
    """Fetch Federal Reserve speeches via RSS."""
    try:
        feed = feedparser.parse(config.FED_SPEECHES_URL,
                                agent="NarrativeBot/1.0")
    except Exception as e:
        print(f"[Fed] Speeches: {e}")
        return []

    articles = []
    for entry in feed.entries[:10]:
        link  = getattr(entry, "link", "") or ""
        title = getattr(entry, "title", "") or ""
        desc  = getattr(entry, "summary", "") or ""

        article = {
            "id":          _art_id("fed", link or title),
            "source":      "fed_speeches",
            "source_tier": 1,
            "title":       title[:500],
            "summary":     desc[:2000],
            "url":         link,
            "published":   _parse_date(entry),
            "market":      "GLOBAL",
            "lang":        "en",
            "raw_text":    title + " " + desc,
            "credibility": 1.0,
        }
        if db.upsert_article(article):
            articles.append(article)

    return articles


def fetch_treasury() -> list[dict]:
    """Fetch US Treasury press releases."""
    try:
        feed = feedparser.parse(config.TREASURY_RSS_URL, agent="NarrativeBot/1.0")
    except Exception as e:
        print(f"[Treasury]: {e}")
        return []

    articles = []
    for entry in feed.entries[:10]:
        link  = getattr(entry, "link", "") or ""
        title = getattr(entry, "title", "") or ""
        desc  = getattr(entry, "summary", "") or ""

        article = {
            "id":          _art_id("treasury", link or title),
            "source":      "treasury",
            "source_tier": 1,
            "title":       title[:500],
            "summary":     desc[:2000],
            "url":         link,
            "published":   _parse_date(entry),
            "market":      "GLOBAL",
            "lang":        "en",
            "raw_text":    title + " " + desc,
            "credibility": 1.0,
        }
        if db.upsert_article(article):
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


def fetch_all_regulatory() -> list[dict]:
    results = []
    results.extend(fetch_sec_8k())
    results.extend(fetch_idx_announcements())
    results.extend(fetch_fed_speeches())
    results.extend(fetch_treasury())
    return results
