"""
Alert generation and dispatch (terminal + optional Telegram).
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime

import requests

import config
from src.storage import db

_last_alert: dict[str, float] = {}


def _alert_id(alert_type: str, topic: str) -> str:
    ts_bucket = int(time.time() // 3600)   # bucket by hour
    return hashlib.md5(f"{alert_type}_{topic}_{ts_bucket}".encode()).hexdigest()


def _cooldown_ok(topic: str, alert_type: str) -> bool:
    key = f"{alert_type}_{topic}"
    last = _last_alert.get(key, 0)
    return (time.time() - last) >= config.ALERT_COOLDOWN_SECONDS


def create_alert(anomaly: dict) -> dict | None:
    topic      = anomaly.get("topic", "unknown")
    alert_type = anomaly.get("type", "unknown")
    severity   = anomaly.get("severity", "LOW")

    if not _cooldown_ok(topic, alert_type):
        return None

    msg = _format_message(anomaly)
    alert = {
        "id":       _alert_id(alert_type, topic),
        "type":     alert_type,
        "topic":    topic,
        "severity": severity,
        "message":  msg,
        "data":     anomaly,
    }
    db.save_alert(alert)
    _last_alert[f"{alert_type}_{topic}"] = time.time()
    return alert


def _format_message(anomaly: dict) -> str:
    atype = anomaly.get("type", "")
    topic = anomaly.get("topic", "").upper().replace("_", " ")
    sev   = anomaly.get("severity", "")

    if atype == "volume_spike":
        return (f"🔺 VOLUME SPIKE [{sev}] — {topic}\n"
                f"   {anomaly.get('article_count')} articles in {anomaly.get('window_minutes')}min\n"
                f"   Avg sentiment: {anomaly.get('avg_sentiment', 0):+.3f}")

    if atype == "sentiment_spike":
        return (f"⚡ SENTIMENT SPIKE [{sev}] — {topic}\n"
                f"   Direction: {anomaly.get('direction','').upper()}\n"
                f"   Z-score: {anomaly.get('z_score', 0):+.2f}  "
                f"Base: {anomaly.get('mean_base', 0):+.3f} → "
                f"Now: {anomaly.get('mean_recent', 0):+.3f}")

    if atype == "sentiment_divergence":
        tiers = anomaly.get("tier_avgs", {})
        tier_str = "  ".join(f"T{k}:{v:+.3f}" for k, v in tiers.items())
        return (f"⚠️  DIVERGENCE [{sev}] — {topic}\n"
                f"   {tier_str}\n"
                f"   Spread: {anomaly.get('spread', 0):.3f}\n"
                f"   {anomaly.get('interpretation', '')}")

    return f"[{atype.upper()}] {topic} — {sev}"


def send_telegram(message: str) -> bool:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id":    config.TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "Markdown",
        }, timeout=5)
        return True
    except Exception:
        return False


def dispatch(alert: dict, send_tg: bool = True) -> None:
    """Print to terminal and optionally send to Telegram."""
    print(f"\n{'='*60}")
    print(alert["message"])
    print(f"{'='*60}\n")
    if send_tg:
        send_telegram(alert["message"])


def process_anomalies(anomalies: list[dict]) -> list[dict]:
    """Convert anomalies to alerts and dispatch."""
    dispatched = []
    for anomaly in anomalies:
        alert = create_alert(anomaly)
        if alert:
            dispatch(alert)
            dispatched.append(alert)
    return dispatched
