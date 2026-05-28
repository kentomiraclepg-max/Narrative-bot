"""
Impact module — maps macro/news narratives to asset-class directional bias.

Pipeline:
  articles (with NLP) → scorer → per-asset bias → reporter (markdown)
"""
from src.impact.asset_map  import ASSETS, THEME_IMPACT, asset_bias_for_theme
from src.impact.scorer     import score_assets, summarize_drivers
from src.impact.reporter   import render_report, save_report

__all__ = [
    "ASSETS", "THEME_IMPACT", "asset_bias_for_theme",
    "score_assets", "summarize_drivers",
    "render_report", "save_report",
]
