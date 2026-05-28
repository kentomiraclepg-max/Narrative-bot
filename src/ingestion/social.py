"""
Social media ingestion: Reddit (PRAW) + X/Twitter (bearer token, optional).
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime

import requests

import config
from src.storage import db

_reddit = None


def _get_reddit():
    global _reddit
    if _reddit is None:
        import praw
        if not config.REDDIT_CLIENT_ID:
            raise RuntimeError("REDDIT_CLIENT_ID not set in .env")
        _reddit = praw.Reddit(
            client_id=config.REDDIT_CLIENT_ID,
            client_secret=config.REDDIT_CLIENT_SECRET,
            user_agent=config.REDDIT_USER_AGENT,
            read_only=True,
        )
    return _reddit


def _post_id(source: str, post_id: str) -> str:
    return hashlib.md5(f"{source}_{post_id}".encode()).hexdigest()


# ── Reddit ────────────────────────────────────────────────────────────────────

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _fetch_reddit_public(sub_name: str, mode: str = "hot",
                          limit: int = 25) -> list[dict]:
    """Fetch subreddit posts via Reddit public JSON API (no credentials needed)."""
    url = f"https://www.reddit.com/r/{sub_name}/{mode}.json?limit={limit}"
    try:
        r = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=15)
        if r.status_code != 200:
            print(f"[Reddit] r/{sub_name}: HTTP {r.status_code}")
            return []
        data = r.json()
    except Exception as e:
        print(f"[Reddit] r/{sub_name}: {e}")
        return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        post_id = post.get("id", "")
        title   = post.get("title", "")
        body    = post.get("selftext", "") or ""
        text    = (title + " " + body).strip()
        score   = post.get("score", 0)

        article = {
            "id":          _post_id("reddit", post_id),
            "source":      f"reddit/{sub_name}",
            "source_tier": 3,
            "title":       title[:500],
            "summary":     body[:2000],
            "url":         f"https://reddit.com{post.get('permalink', '')}",
            "published":   datetime.utcfromtimestamp(
                               post.get("created_utc", time.time())).isoformat(),
            "market":      _detect_reddit_market(sub_name, text),
            "lang":        "id" if sub_name in ("indonesia", "investasi") else "en",
            "raw_text":    text[:3000],
            "credibility": 0.4 + min(score, 1000) / 5000,
        }
        if db.upsert_article(article):
            posts.append(article)

    return posts


def fetch_reddit(subreddits: list[str] | None = None,
                 limit: int = 25, mode: str = "hot") -> list[dict]:
    """Fetch posts from subreddits. Tries PRAW first, falls back to public API."""
    subs  = subreddits or config.REDDIT_SUBS
    posts = []

    # Try PRAW if credentials are configured
    if config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_ID != "your_client_id":
        try:
            reddit = _get_reddit()
            for sub_name in subs:
                try:
                    sub    = reddit.subreddit(sub_name)
                    stream = getattr(sub, mode)(limit=limit)
                    for post in stream:
                        text = (post.title or "") + " " + (post.selftext or "")
                        article = {
                            "id":          _post_id("reddit", post.id),
                            "source":      f"reddit/{sub_name}",
                            "source_tier": 3,
                            "title":       post.title[:500],
                            "summary":     post.selftext[:2000] if post.selftext else "",
                            "url":         f"https://reddit.com{post.permalink}",
                            "published":   datetime.utcfromtimestamp(
                                               post.created_utc).isoformat(),
                            "market":      _detect_reddit_market(sub_name, text),
                            "lang":        "id" if sub_name in ("indonesia", "investasi") else "en",
                            "raw_text":    text[:3000],
                            "credibility": 0.4 + min(post.score, 1000) / 5000,
                        }
                        if db.upsert_article(article):
                            posts.append(article)
                except Exception as e:
                    print(f"[Reddit] r/{sub_name}: {e}")
            return posts
        except Exception:
            pass  # fall through to public API

    # Fallback: public JSON API (no credentials required)
    for sub_name in subs:
        new = _fetch_reddit_public(sub_name, mode=mode, limit=limit)
        posts.extend(new)
        time.sleep(1)  # respect rate limit between subreddits

    return posts


def _detect_reddit_market(sub: str, text: str) -> str:
    text_lower = text.lower()
    if sub in ("indonesia", "investasi"):
        return "IDX"
    if any(k in text_lower for k in ("bitcoin", "ethereum", "crypto", "defi", "nft")):
        return "CRYPTO"
    return "GLOBAL"


# ── Twitter/X ─────────────────────────────────────────────────────────────────

def fetch_twitter(query: str, max_results: int = 20) -> list[dict]:
    """
    Fetch tweets matching a query.
    Requires TWITTER_BEARER_TOKEN (paid API tier).
    """
    if not config.TWITTER_BEARER_TOKEN:
        return []

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {config.TWITTER_BEARER_TOKEN}"}
    params = {
        "query":       f"{query} -is:retweet lang:en",
        "max_results": min(max_results, 100),
        "tweet.fields": "created_at,author_id,public_metrics,context_annotations",
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[Twitter] {query}: {e}")
        return []

    tweets = []
    for tweet in data.get("data", []):
        metrics = tweet.get("public_metrics", {})
        # Credibility based on engagement
        engagement = metrics.get("retweet_count", 0) + metrics.get("like_count", 0)
        cred = min(0.3 + engagement / 1000, 0.9)

        article = {
            "id":          _post_id("twitter", tweet["id"]),
            "source":      "twitter",
            "source_tier": 3,
            "title":       tweet["text"][:280],
            "summary":     "",
            "url":         f"https://twitter.com/i/web/status/{tweet['id']}",
            "published":   tweet.get("created_at", datetime.utcnow().isoformat()),
            "market":      "GLOBAL",
            "lang":        "en",
            "raw_text":    tweet["text"],
            "credibility": cred,
        }
        is_new = db.upsert_article(article)
        if is_new:
            tweets.append(article)

    return tweets


def fetch_all_social() -> list[dict]:
    posts = fetch_reddit()
    return posts
