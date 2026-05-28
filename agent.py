"""
agent.py — Agentic narrative screening bot.

Loop every POLL_INTERVAL_SECONDS:
  1. Ingest  → RSS + Social + Regulatory
  2. Filter  → Credibility filter
  3. NLP     → FinBERT sentiment + spaCy entities + macro themes
  4. Persist → Update DB + sentiment history
  5. Graph   → Update knowledge graph
  6. Detect  → Spike + divergence detection
  7. Alert   → Dispatch alerts

Usage:
  python agent.py                  # run continuous loop
  python agent.py --once           # single scan
  python agent.py --dashboard      # show live dashboard after each scan
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import config
from src.ingestion.stream    import ingest_all
from src.impact.web_ingestor import fetch_all_assets, classify_extra_themes
from src.impact.scorer       import score_assets
from src.impact.reporter     import save_report
from src.nlp.pipeline        import process_batch
from src.detection.spike     import run_all as detect_all
from src.detection.alerts    import process_anomalies
from src.graph.builder       import update_from_articles
from src.credibility.ranker  import filter_low_credibility, cross_verify
from src.storage.db          import (get_unprocessed, update_article_nlp,
                                     log_sentiment, stats)


def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def run_scan(show_dashboard: bool = False) -> dict:
    print(f"\n[{_ts()}] ── SCAN START ──────────────────────────────")

    # 1. Ingest
    ingested = ingest_all()
    new_articles = list(ingested["all"])

    # 1b. Real-time per-asset narrative pulls (Google News RSS)
    try:
        asset_articles = fetch_all_assets()
        new_articles.extend(asset_articles)
    except Exception as e:
        print(f"[WebIngest] failed: {e}")

    # 2. Credibility filter
    new_articles = filter_low_credibility(new_articles)

    # 3. NLP (process unprocessed from DB too)
    unprocessed = get_unprocessed(limit=100)
    to_process  = {a["id"]: a for a in new_articles + unprocessed}

    if to_process:
        print(f"[NLP] Processing {len(to_process)} articles…")
        processed = process_batch(list(to_process.values()))

        # 4. Persist NLP results + augment with extra (crypto/etf/regulation) themes
        for article in processed:
            extras = classify_extra_themes(article.get("raw_text") or article.get("title", ""))
            if extras:
                article["macro_themes"] = list(set(article.get("macro_themes", []) + extras))

            update_article_nlp(
                article_id       = article["id"],
                sentiment_label  = article.get("sentiment_label", "neutral"),
                sentiment_score  = article.get("sentiment_score", 0.0),
                tone             = article.get("tone", "neutral"),
                entities         = article.get("entities", []),
            )
            # Log sentiment history per theme
            for theme in article.get("macro_themes", []):
                log_sentiment(
                    source  = article.get("source", ""),
                    topic   = theme,
                    score   = article.get("sentiment_score", 0.0),
                    volume  = 1,
                )
    else:
        processed = []
        print("[NLP] No new articles to process.")

    # 5. Knowledge graph
    if processed:
        update_from_articles(processed)

    # 6. Cross-verify narratives
    verification = cross_verify(processed)
    confirmed = [t for t, v in verification.items() if v["confirmed"]]
    if confirmed:
        print(f"[Verify] Confirmed narratives: {', '.join(confirmed)}")

    # 7. Anomaly detection
    all_recent = get_unprocessed(limit=0)  # empty → we'll use processed list
    anomalies  = detect_all(processed)
    if anomalies:
        print(f"[Detect] {len(anomalies)} anomalies found")

    # 8. Alerts
    alerts = process_anomalies(anomalies)

    # 9. Per-asset impact scoring + markdown report
    report_path = None
    if processed:
        try:
            scored = score_assets(processed)
            report_path = save_report(scored, processed)
        except Exception as e:
            print(f"[Impact] scoring/report failed: {e}")
            scored = {}

    # 10. Push report to Notion (skips if env vars not set)
    if report_path:
        try:
            from src.export.notion_publisher import publish_to_notion
            publish_to_notion(report_path, scored)
        except Exception as e:
            print(f"[Notion] {e}")
    else:
        scored = {}

    db_stats = stats()
    print(f"[{_ts()}] ── SCAN DONE  "
          f"new={len(new_articles)} processed={len(processed)} "
          f"anomalies={len(anomalies)} alerts={len(alerts)} "
          f"db_total={db_stats['total_articles']}")

    result = {
        "new_articles":   len(new_articles),
        "processed":      len(processed),
        "anomalies":      anomalies,
        "alerts":         alerts,
        "confirmed":      confirmed,
        "verification":   verification,
        "db_stats":       db_stats,
        "impact":         scored,
        "report_path":    report_path,
    }

    if show_dashboard:
        from dashboard import render
        render(result)

    return result


def run_loop(show_dashboard: bool = False) -> None:
    print("╔══════════════════════════════════════════╗")
    print("║  NARRATIVE SCREENING AGENT — STARTED     ║")
    print(f"║  Poll interval: {config.POLL_INTERVAL_SECONDS}s                      ║")
    print("╚══════════════════════════════════════════╝\n")

    while True:
        try:
            run_scan(show_dashboard=show_dashboard)
        except KeyboardInterrupt:
            print("\n[STOP] Agent stopped.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")

        print(f"  Next scan in {config.POLL_INTERVAL_SECONDS}s…")
        time.sleep(config.POLL_INTERVAL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Narrative Screening Agent")
    parser.add_argument("--once",      action="store_true", help="Run single scan and exit")
    parser.add_argument("--dashboard", action="store_true", help="Show rich dashboard after scan")
    args = parser.parse_args()

    if args.once:
        run_scan(show_dashboard=args.dashboard)
    else:
        run_loop(show_dashboard=args.dashboard)


if __name__ == "__main__":
    main()
