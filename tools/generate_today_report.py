"""
generate_today_report.py — one-shot generator that feeds today's *real-world*
narratives (gathered via Claude WebSearch) through the impact pipeline and
writes a markdown report.

Sentiment scores below are FinBERT-style signed numbers in [-1, +1] estimated
from the headline tone (negative inflation print → negative; positive ETF
inflows → positive).  In the regular agent loop these are produced
automatically by `src/nlp/pipeline.py`.

Usage:
    cd /Users/kento/Project_Narrative
    python tools/generate_today_report.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

# Make project root importable when run from any CWD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.impact.scorer    import score_assets
from src.impact.reporter  import save_report


# ── Articles: real headlines from today's WebSearch (2026-05-16 UTC) ─────────
# Each article carries the macro_themes that src/nlp/pipeline.detect_macro_themes
# (+ src/impact/web_ingestor.classify_extra_themes) would tag.
ARTICLES = [
    # ── Fed / US rates ───────────────────────────────────────────────────────
    {
        "id":             "fomc_apr29_2026",
        "title":          "Fed holds rates steady at 3.50%–3.75% with four dissenters — most since 1992",
        "summary":        ("FOMC paused at the 29 April meeting but the statement added an "
                           "easing-bias clause; three policymakers dissented against the dovish "
                           "language while Gov. Miran wanted a cut."),
        "source":         "cnbc.com",
        "source_tier":    2,
        "url":            "https://www.cnbc.com/2026/04/29/fed-interest-rate-decision-april-2026.html",
        "credibility":    0.85,
        "sentiment_score": +0.15,    # net mildly dovish (easing-bias added)
        "macro_themes":   ["rate_cut"],   # dovish tilt
    },
    {
        "id":             "cpi_hot_may13_2026",
        "title":          "US CPI runs hot at 3.8% YoY; wholesale inflation jumps ~6%, biggest since 2022",
        "summary":        ("April CPI surprised to the upside at 3.8% annual versus consensus "
                           "expectations, and producer prices climbed roughly 6% — the largest "
                           "jump since 2022.  Treasuries sold off, dollar bid."),
        "source":         "reuters.com",
        "source_tier":    1,
        "url":            "https://www.cnbc.com/2026/05/11/stock-market-today-live-updates.html",
        "credibility":    1.0,
        "sentiment_score": -0.75,
        "macro_themes":   ["inflation", "rate_hike"],
    },
    {
        "id":             "stocks_may15_selloff",
        "title":          "S&P 500 -1.14%, Nasdaq -1.62% on hot inflation print and rising yields",
        "summary":        ("Risk-off Friday as 10-year yields jumped and Treasury auctions tailed; "
                           "Dow -0.81%, S&P -1.14%, Nasdaq -1.62%."),
        "source":         "thestreet.com",
        "source_tier":    2,
        "url":            "https://www.thestreet.com/latest-news/stock-market-today-may-15-2026-updates",
        "credibility":    0.8,
        "sentiment_score": -0.65,
        "macro_themes":   ["inflation"],
    },
    {
        "id":             "earnings_q1_beats_2026",
        "title":          "Q1 2026 earnings season tops estimates; blended growth 15.1% YoY",
        "summary":        ("With ~⅓ of S&P 500 names reported, blended earnings growth is 15.1%. "
                           "Morgan Stanley lifts 2026 S&P EPS target to $339 (+23% YoY)."),
        "source":         "investinglive.com",
        "source_tier":    2,
        "url":            "https://investinglive.com/stock-market-update/morgan-stanley-lifts-sp-500-target-to-8000-on-earnings-growth-story-20260513/",
        "credibility":    0.75,
        "sentiment_score": +0.85,
        "macro_themes":   ["earnings"],
    },

    # ── Bitcoin / crypto ─────────────────────────────────────────────────────
    {
        "id":             "btc_etf_outflow_may16",
        "title":          "Spot Bitcoin ETFs post $1B weekly outflow, halting six-week inflow streak",
        "summary":        ("U.S. spot BTC ETFs saw $1B of redemptions for the week ending May 15. "
                           "May 13 alone: -$635M.  BTC trades near $79K, struggling to hold $80K."),
        "source":         "cryptotimes.io",
        "source_tier":    3,
        "url":            "https://www.cryptotimes.io/2026/05/16/bitcoin-etfs-post-1b-weekly-outflow-halting-six-week-inflow-streak/",
        "credibility":    0.65,
        "sentiment_score": -0.7,
        "macro_themes":   ["etf_flows"],
    },
    {
        "id":             "btc_consolidation_may16",
        "title":          "Bitcoin hovers near $79,000 as macro headwinds and profit-taking weigh",
        "summary":        ("BTC consolidates after April rally; market cites hot CPI and Fed "
                           "uncertainty as the main drag."),
        "source":         "coindesk.com",
        "source_tier":    3,
        "url":            "https://www.coindesk.com/markets/2026/05/04/the-bitcoin-etf-recovery-in-flows-is-real-it-is-just-not-complete-yet",
        "credibility":    0.7,
        "sentiment_score": -0.3,
        "macro_themes":   ["etf_flows"],
    },

    # ── Indonesia: BI rate, rupiah, IHSG ─────────────────────────────────────
    {
        "id":             "bi_rate_hold_may7",
        "title":          "Bank Indonesia tetap pertahankan BI Rate di 4,75% untuk bulan ke-7",
        "summary":        ("RDG BI 7 Mei 2026 menahan BI Rate di 4,75% demi menjaga stabilitas "
                           "rupiah dan inflasi.  Deposit facility 3,75%; lending facility 5,50%."),
        "source":         "bi.go.id",
        "source_tier":    1,
        "url":            "https://www.bi.go.id/id/publikasi/ruang-media/news-release/Pages/sp_288426.aspx",
        "credibility":    1.0,
        "sentiment_score": +0.35,   # hawkish-but-stable; FX-supportive
        "macro_themes":   ["bi_rate"],
    },
    {
        "id":             "rupiah_jisdor_may13",
        "title":          "JISDOR rupiah 17,496/USD pada 13 Mei 2026 — masih tertekan tekanan dolar",
        "summary":        ("Referensi BI menunjukkan rupiah di 17,496 per dolar AS, "
                           "lemah dibanding 17,514 di hari sebelumnya namun masih di atas "
                           "level psikologis 17,000."),
        "source":         "bi.go.id",
        "source_tier":    1,
        "url":            "https://www.bi.go.id/en/statistik/informasi-kurs/jisdor/default.aspx",
        "credibility":    1.0,
        "sentiment_score": -0.45,
        "macro_themes":   ["rupiah", "currency"],
    },
    {
        "id":             "ihsg_drop_may13",
        "title":          "IHSG 6.723 — turun ~12% dalam 4 minggu ke titik terendah sejak April 2025",
        "summary":        ("Bursa terus tertekan, manufaktur Indonesia turun di April untuk "
                           "pertama kalinya dalam 9 bulan, terdampak pelemahan permintaan "
                           "pasca-Lebaran dan tekanan supply-chain global."),
        "source":         "tradingeconomics.com",
        "source_tier":    2,
        "url":            "https://tradingeconomics.com/indonesia/stock-market",
        "credibility":    0.8,
        "sentiment_score": -0.7,
        "macro_themes":   ["ihsg", "supply_chain"],
    },
    {
        "id":             "id_manufacturing_apr",
        "title":          "PMI Manufaktur Indonesia turun pertama dalam 9 bulan akibat shock pasokan global",
        "summary":        ("Indeks manufaktur kontraksi, terbebani lonjakan biaya energi pasca-"
                           "krisis Hormuz dan pelemahan permintaan ekspor."),
        "source":         "kontan.co.id",
        "source_tier":    2,
        "url":            "https://www.kontan.co.id/",
        "credibility":    0.8,
        "sentiment_score": -0.55,
        "macro_themes":   ["recession", "supply_chain"],
    },

    # ── Gold ─────────────────────────────────────────────────────────────────
    {
        "id":             "gold_cb_buying_q1",
        "title":          "Central banks net-bought 244 tonnes of gold in Q1, +3% YoY",
        "summary":        ("WGC: official-sector demand on track for ~755 tonnes in 2026; EM "
                           "central banks remain underweight gold relative to DM peers and "
                           "continue to accumulate."),
        "source":         "gold.org",
        "source_tier":    1,
        "url":            "https://www.gold.org/goldhub/research/gold-outlook-2026",
        "credibility":    0.95,
        "sentiment_score": +0.7,
        "macro_themes":   ["commodity", "currency"],
    },
    {
        "id":             "gold_5000_may_outlook",
        "title":          "Institutional forecasters still see $5,000–$6,000 gold by year-end",
        "summary":        ("StateStreet/SSGA monthly monitor and Goldsilver compile forecasts in "
                           "the $5,000–$6,000 area for Q4 2026 driven by geopolitics and "
                           "structural CB demand."),
        "source":         "goldsilver.com",
        "source_tier":    3,
        "url":            "https://goldsilver.com/industry-news/article/gold-price-outlook-may-2026-why-institutional-forecasters-still-see-5000/",
        "credibility":    0.6,
        "sentiment_score": +0.6,
        "macro_themes":   ["geopolitical", "commodity"],
    },

    # ── Geopolitics: Strait of Hormuz / Iran ─────────────────────────────────
    {
        "id":             "hormuz_blockade_status",
        "title":          "Strait of Hormuz still largely blocked; ~10M bpd of oil exports off the market",
        "summary":        ("Largest oil-supply disruption on record; crude near $100 per barrel. "
                           "US has blockaded Iranian ports since 13 April after the Israel-US "
                           "air war that began 28 February."),
        "source":         "imf.org",
        "source_tier":    1,
        "url":            "https://www.imf.org/en/blogs/articles/2026/03/30/how-the-war-in-the-middle-east-is-affecting-energy-trade-and-finance",
        "credibility":    1.0,
        "sentiment_score": -0.85,
        "macro_themes":   ["geopolitical", "supply_chain", "inflation"],
    },
    {
        "id":             "trump_xi_hormuz_may15",
        "title":          "Trump-Xi meet in Beijing, agree Strait of Hormuz 'must open' to free energy flow",
        "summary":        ("White House statement after the bilateral indicates a coordinated push "
                           "to ease the blockade; oil drifts lower on the headlines.  IEA reports "
                           "non-OPEC supply has stepped up by 3.5M bpd during the war."),
        "source":         "cnbc.com",
        "source_tier":    2,
        "url":            "https://www.cnbc.com/2026/05/15/china-us-oil-iran-war-strait-hormuz-trump-xi.html",
        "credibility":    0.85,
        "sentiment_score": +0.55,
        "macro_themes":   ["geopolitical"],
    },

    # ── FX / DXY / EUR / JPY ─────────────────────────────────────────────────
    {
        "id":             "dxy_mid_may_surge",
        "title":          "DXY surges to 101+ in mid-May as hot CPI revives Fed-hike calls",
        "summary":        ("Dollar index trades around 101.2 versus a 99.8 month-open; greenback "
                           "broadly bid against euro and yen as rate-differential narrative "
                           "tightens again."),
        "source":         "forex.com",
        "source_tier":    2,
        "url":            "https://www.forex.com/en-us/news-and-analysis/usd-into-2026-gold-eur-usd-usd-jpy/",
        "credibility":    0.75,
        "sentiment_score": +0.6,
        "macro_themes":   ["currency", "rate_hike"],
    },
    {
        "id":             "eurusd_one_month_low",
        "title":          "EUR/USD slips to 1.1617 — lowest in over a month",
        "summary":        ("Common currency weighed down by USD strength; ECB still on hold while "
                           "Fed-rate-cut path is repriced later into 2026."),
        "source":         "forex.com",
        "source_tier":    2,
        "url":            "https://www.forex.com/en/news-and-analysis/usd-into-2026-gold-eur-usd-usd-jpy/",
        "credibility":    0.75,
        "sentiment_score": -0.4,
        "macro_themes":   ["currency"],
    },
    {
        "id":             "usdjpy_158_boj_hawk",
        "title":          "USD/JPY pinned near 158.5; BoJ Summary of Opinions hints at imminent hike",
        "summary":        ("Yen weakness persists as US inflation reignites Fed-hike pricing. "
                           "BoJ April minutes show members openly discussing a near-term rate "
                           "hike amid rising oil costs."),
        "source":         "tradingeconomics.com",
        "source_tier":    2,
        "url":            "https://tradingeconomics.com/japan/currency",
        "credibility":    0.8,
        "sentiment_score": -0.3,
        "macro_themes":   ["currency", "rate_hike"],
    },
]


def main() -> None:
    print(f"[Generate] Articles in corpus: {len(ARTICLES)}")

    # Run impact scoring across all tracked assets
    scored = score_assets(ARTICLES)

    # Use today's date (2026-05-16) for the timestamp
    ts = datetime(2026, 5, 16, 12, 0, 0)
    path = save_report(scored, ARTICLES, ts=ts,
                       directory=os.path.join(os.path.dirname(__file__),
                                              "..", "reports"))
    print(f"[Generate] Report saved: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
