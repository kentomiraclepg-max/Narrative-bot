"""
Stream coordinator: runs all ingestion sources in one call.
"""
from __future__ import annotations

from src.ingestion import rss, social, regulatory


def ingest_all() -> dict[str, list]:
    """Run all ingestion sources and return combined new articles by source."""
    rss_new  = rss.fetch_all()
    soc_new  = social.fetch_all_social()
    reg_new  = regulatory.fetch_all_regulatory()

    total = len(rss_new) + len(soc_new) + len(reg_new)
    print(f"[Ingest] +{total} new articles  "
          f"(RSS:{len(rss_new)} Social:{len(soc_new)} Regulatory:{len(reg_new)})")

    return {
        "rss":        rss_new,
        "social":     soc_new,
        "regulatory": reg_new,
        "all":        rss_new + soc_new + reg_new,
    }
