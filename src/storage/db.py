"""
SQLite storage for articles, narratives, entities, and alerts.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any

import config

_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _init_schema(_conn)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS articles (
        id          TEXT PRIMARY KEY,
        source      TEXT,
        source_tier INTEGER DEFAULT 3,
        title       TEXT,
        summary     TEXT,
        url         TEXT UNIQUE,
        published   TEXT,
        fetched_at  TEXT,
        market      TEXT DEFAULT 'GLOBAL',
        lang        TEXT DEFAULT 'en',
        raw_text    TEXT,
        sentiment_label TEXT,
        sentiment_score REAL,
        tone        TEXT,
        entities    TEXT,
        credibility REAL DEFAULT 0.5,
        processed   INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS narratives (
        id          TEXT PRIMARY KEY,
        topic       TEXT,
        first_seen  TEXT,
        last_seen   TEXT,
        article_ids TEXT,
        sentiment_avg REAL,
        sentiment_std REAL,
        momentum    REAL,
        entities    TEXT,
        market_impact TEXT,
        alert_sent  INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id          TEXT PRIMARY KEY,
        type        TEXT,
        topic       TEXT,
        severity    TEXT,
        message     TEXT,
        data        TEXT,
        created_at  TEXT,
        sent        INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS sentiment_history (
        ts          TEXT,
        source      TEXT,
        topic       TEXT,
        score       REAL,
        volume      INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published);
    CREATE INDEX IF NOT EXISTS idx_articles_processed ON articles(processed);
    CREATE INDEX IF NOT EXISTS idx_articles_source    ON articles(source);
    CREATE INDEX IF NOT EXISTS idx_sentiment_ts       ON sentiment_history(ts);
    """)
    conn.commit()


def upsert_article(article: dict) -> bool:
    conn = _get_conn()
    try:
        conn.execute("""
        INSERT OR IGNORE INTO articles
          (id, source, source_tier, title, summary, url, published,
           fetched_at, market, lang, raw_text, credibility)
        VALUES (:id,:source,:source_tier,:title,:summary,:url,:published,
                :fetched_at,:market,:lang,:raw_text,:credibility)
        """, {
            "id":          article["id"],
            "source":      article.get("source", ""),
            "source_tier": article.get("source_tier", 3),
            "title":       article.get("title", ""),
            "summary":     article.get("summary", ""),
            "url":         article.get("url", ""),
            "published":   article.get("published", ""),
            "fetched_at":  datetime.utcnow().isoformat(),
            "market":      article.get("market", "GLOBAL"),
            "lang":        article.get("lang", "en"),
            "raw_text":    article.get("raw_text", ""),
            "credibility": article.get("credibility", 0.5),
        })
        conn.commit()
        return conn.execute("SELECT changes()").fetchone()[0] > 0
    except sqlite3.IntegrityError:
        return False


def update_article_nlp(article_id: str, sentiment_label: str,
                        sentiment_score: float, tone: str,
                        entities: list) -> None:
    conn = _get_conn()
    conn.execute("""
    UPDATE articles SET
        sentiment_label = ?, sentiment_score = ?, tone = ?,
        entities = ?, processed = 1
    WHERE id = ?
    """, (sentiment_label, sentiment_score, tone,
          json.dumps(entities), article_id))
    conn.commit()


def get_unprocessed(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM articles WHERE processed=0 ORDER BY fetched_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_recent_articles(hours: int = 24, source: str | None = None,
                         min_credibility: float = 0.0) -> list[dict]:
    conn = _get_conn()
    q = """SELECT * FROM articles
           WHERE fetched_at > datetime('now', ?)
             AND credibility >= ?"""
    params: list[Any] = [f"-{hours} hours", min_credibility]
    if source:
        q += " AND source = ?"
        params.append(source)
    q += " ORDER BY published DESC"
    return [dict(r) for r in conn.execute(q, params).fetchall()]


def upsert_narrative(narrative: dict) -> None:
    conn = _get_conn()
    conn.execute("""
    INSERT INTO narratives
      (id, topic, first_seen, last_seen, article_ids, sentiment_avg,
       sentiment_std, momentum, entities, market_impact)
    VALUES (:id,:topic,:first_seen,:last_seen,:article_ids,:sentiment_avg,
            :sentiment_std,:momentum,:entities,:market_impact)
    ON CONFLICT(id) DO UPDATE SET
      last_seen      = excluded.last_seen,
      article_ids    = excluded.article_ids,
      sentiment_avg  = excluded.sentiment_avg,
      sentiment_std  = excluded.sentiment_std,
      momentum       = excluded.momentum,
      entities       = excluded.entities,
      market_impact  = excluded.market_impact
    """, {
        "id":           narrative["id"],
        "topic":        narrative.get("topic", ""),
        "first_seen":   narrative.get("first_seen", datetime.utcnow().isoformat()),
        "last_seen":    narrative.get("last_seen", datetime.utcnow().isoformat()),
        "article_ids":  json.dumps(narrative.get("article_ids", [])),
        "sentiment_avg": narrative.get("sentiment_avg", 0.0),
        "sentiment_std": narrative.get("sentiment_std", 0.0),
        "momentum":     narrative.get("momentum", 0.0),
        "entities":     json.dumps(narrative.get("entities", [])),
        "market_impact": narrative.get("market_impact", ""),
    })
    conn.commit()


def save_alert(alert: dict) -> None:
    conn = _get_conn()
    conn.execute("""
    INSERT OR IGNORE INTO alerts
      (id, type, topic, severity, message, data, created_at)
    VALUES (:id,:type,:topic,:severity,:message,:data,:created_at)
    """, {
        "id":         alert["id"],
        "type":       alert.get("type", ""),
        "topic":      alert.get("topic", ""),
        "severity":   alert.get("severity", "LOW"),
        "message":    alert.get("message", ""),
        "data":       json.dumps(alert.get("data", {})),
        "created_at": datetime.utcnow().isoformat(),
    })
    conn.commit()


def log_sentiment(source: str, topic: str, score: float, volume: int) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sentiment_history (ts, source, topic, score, volume) VALUES (?,?,?,?,?)",
        (datetime.utcnow().isoformat(), source, topic, score, volume)
    )
    conn.commit()


def get_sentiment_history(topic: str, hours: int = 24) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM sentiment_history
           WHERE topic=? AND ts > datetime('now', ?)
           ORDER BY ts ASC""",
        (topic, f"-{hours} hours")
    ).fetchall()
    return [dict(r) for r in rows]


def stats() -> dict:
    conn = _get_conn()
    return {
        "total_articles":     conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
        "processed_articles": conn.execute("SELECT COUNT(*) FROM articles WHERE processed=1").fetchone()[0],
        "total_narratives":   conn.execute("SELECT COUNT(*) FROM narratives").fetchone()[0],
        "total_alerts":       conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0],
        "articles_24h":       conn.execute(
            "SELECT COUNT(*) FROM articles WHERE fetched_at > datetime('now','-24 hours')"
        ).fetchone()[0],
    }
