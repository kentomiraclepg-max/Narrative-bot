"""
asset_map.py — Static mapping of macro/news narrative themes to asset-class
directional impact weights.

Each (theme, asset) weight ∈ [-1.0, +1.0]:
   +1.0  → asset is strongly BULLISH when the theme/event becomes more probable
   -1.0  → asset is strongly BEARISH when the theme/event becomes more probable
    0.0  → no direct mechanical impact

How "more probable" is read off an article depends on the theme's natural
tone (see `BEARISH_TONED_THEMES`).  For "bearish-toned" event themes such as
`rate_hike`, `inflation`, `geopolitical` — a FinBERT-negative article means the
event is happening / intensifying; a positive article means it is fading.
For "topical" themes such as `earnings`, `etf_flows`, `rupiah` — the
sentiment sign is taken at face value.  The scorer in `scorer.py` handles
that flip so weights here can always be read as the asset's direction *when
the event is occurring*.

References for the directional logic:
  - Fed hawkishness → USD up, gold down, risk-off (Bernanke/Reinhart, BIS).
  - Geopolitical shocks → flight-to-safety bid for gold and USD; risk-off
    for equities and crypto (Caldara & Iacoviello, GPR index).
  - Commodity prices → tail wind for IDX (commodity-export-heavy), depending
    on coal/nickel/CPO mix (Bank Indonesia monetary report).
"""
from __future__ import annotations

# ── Tracked assets ───────────────────────────────────────────────────────────
ASSETS = [
    "BTC",         # Bitcoin
    "ETH",         # Ethereum
    "SOL",         # Solana
    "US_STOCKS",   # S&P 500 / Nasdaq / Dow composite
    "IDX",         # IHSG (Indonesia composite)
    "XAUUSD",      # Spot gold in USD
    "XTIUSD",      # WTI crude oil in USD
    "DXY",         # US dollar index
    "USDIDR",      # USD/IDR (higher = rupiah weaker)
    "EURUSD",      # EUR/USD
    "USDJPY",      # USD/JPY (higher = yen weaker)
]

# Pretty labels for reports
ASSET_LABELS = {
    "BTC":       "Bitcoin (BTC)",
    "ETH":       "Ethereum (ETH)",
    "SOL":       "Solana (SOL)",
    "US_STOCKS": "US Equities (S&P/Nasdaq)",
    "IDX":       "Indonesia Stocks (IHSG)",
    "XAUUSD":    "Gold (XAUUSD)",
    "XTIUSD":    "Crude Oil WTI (XTIUSD)",
    "DXY":       "US Dollar Index (DXY)",
    "USDIDR":    "Rupiah (USDIDR)",
    "EURUSD":    "Euro (EURUSD)",
    "USDJPY":    "Yen (USDJPY)",
}

# ── Theme → asset directional weight ─────────────────────────────────────────
# Keys match `MACRO_KEYWORDS` and `IDX_KEYWORDS` in src/nlp/pipeline.py.
# Value: dict[asset → weight].  A missing asset = 0.0.
#
# Convention: weight is the asset's direction when the **theme itself is
# bullish-flavored**.  Inflation is treated as a "thing" — its sentiment from
# FinBERT is typically negative (bad news), which combined with the weights
# below yields the correct sign for each asset.
THEME_IMPACT: dict[str, dict[str, float]] = {
    # Fed hawkish → USD bid, gold off, risk-off
    "rate_hike": {
        "BTC":       -0.7,
        "ETH":       -0.6,
        "SOL":       -0.7,
        "US_STOCKS": -0.6,
        "IDX":       -0.5,
        "XAUUSD":    -0.6,
        "XTIUSD":    -0.4,
        "DXY":       +0.8,
        "USDIDR":    +0.6,
        "EURUSD":    -0.6,
        "USDJPY":    +0.7,
    },
    # Fed dovish → USD off, gold bid, risk-on
    "rate_cut": {
        "BTC":       +0.7,
        "ETH":       +0.6,
        "SOL":       +0.7,
        "US_STOCKS": +0.7,
        "IDX":       +0.6,
        "XAUUSD":    +0.7,
        "XTIUSD":    +0.3,
        "DXY":       -0.8,
        "USDIDR":    -0.5,
        "EURUSD":    +0.6,
        "USDJPY":    -0.6,
    },
    # Inflation prints / surprises
    "inflation": {
        "BTC":       +0.2,
        "ETH":       +0.1,
        "SOL":       +0.1,
        "US_STOCKS": -0.5,
        "IDX":       -0.3,
        "XAUUSD":    +0.7,
        "XTIUSD":    +0.5,   # oil is a major inflation driver
        "DXY":       +0.3,
        "USDIDR":    +0.3,
        "EURUSD":    -0.2,
        "USDJPY":    +0.2,
    },
    # Recession / growth fears
    "recession": {
        "BTC":       -0.5,
        "ETH":       -0.5,
        "SOL":       -0.6,
        "US_STOCKS": -0.8,
        "IDX":       -0.7,
        "XAUUSD":    +0.5,
        "XTIUSD":    -0.7,   # demand destruction
        "DXY":       +0.2,
        "USDIDR":    +0.4,
        "EURUSD":    -0.3,
        "USDJPY":    -0.4,
    },
    # Corporate earnings (US/global)
    "earnings": {
        "BTC":       +0.2,
        "ETH":       +0.2,
        "SOL":       +0.2,
        "US_STOCKS": +0.8,
        "IDX":       +0.4,
        "XAUUSD":    -0.2,
        "XTIUSD":    +0.2,
        "DXY":       +0.1,
        "USDIDR":    -0.1,
        "EURUSD":    -0.1,
        "USDJPY":    +0.1,
    },
    # Geopolitical shock (war, sanctions, tariffs)
    "geopolitical": {
        "BTC":       -0.1,   # updated: less risk-off, more "digital gold" narrative
        "ETH":       -0.2,
        "SOL":       -0.3,
        "US_STOCKS": -0.7,
        "IDX":       -0.6,
        "XAUUSD":    +0.8,
        "XTIUSD":    +0.8,   # supply disruption risk
        "DXY":       +0.5,
        "USDIDR":    +0.5,
        "EURUSD":    -0.4,
        "USDJPY":    -0.3,
    },
    # Liquidity stress, credit events
    "liquidity": {
        "BTC":       -0.6,
        "ETH":       -0.5,
        "SOL":       -0.6,
        "US_STOCKS": -0.7,
        "IDX":       -0.6,
        "XAUUSD":    +0.5,
        "XTIUSD":    -0.3,
        "DXY":       +0.4,
        "USDIDR":    +0.5,
        "EURUSD":    -0.3,
        "USDJPY":    -0.3,
    },
    # Supply-chain disruptions
    "supply_chain": {
        "BTC":       -0.1,
        "ETH":       -0.1,
        "SOL":       -0.1,
        "US_STOCKS": -0.4,
        "IDX":       -0.2,
        "XAUUSD":    +0.3,
        "XTIUSD":    +0.4,   # shipping disruptions → oil demand
        "DXY":       +0.1,
        "USDIDR":    +0.2,
        "EURUSD":    -0.1,
        "USDJPY":    +0.1,
    },
    # Generic FX / dollar narrative
    "currency": {
        "BTC":       +0.2,
        "ETH":       +0.2,
        "SOL":       +0.2,
        "US_STOCKS": +0.1,
        "IDX":       +0.2,
        "XAUUSD":    +0.3,
        "XTIUSD":    -0.3,   # stronger USD → lower oil
        "DXY":        0.0,
        "USDIDR":    -0.1,
        "EURUSD":     0.0,
        "USDJPY":     0.0,
    },

    # ── Indonesia-specific themes ────────────────────────────────────────────
    "bi_rate": {
        "BTC":        0.0,
        "ETH":        0.0,
        "SOL":        0.0,
        "US_STOCKS":  0.0,
        "IDX":       +0.5,
        "XAUUSD":     0.0,
        "XTIUSD":     0.0,
        "DXY":       -0.1,
        "USDIDR":    -0.7,
        "EURUSD":     0.0,
        "USDJPY":     0.0,
    },
    "ihsg": {
        "BTC":        0.0,
        "ETH":        0.0,
        "SOL":        0.0,
        "US_STOCKS": +0.1,
        "IDX":       +0.9,
        "XAUUSD":     0.0,
        "XTIUSD":     0.0,
        "DXY":        0.0,
        "USDIDR":    -0.2,
        "EURUSD":     0.0,
        "USDJPY":     0.0,
    },
    "rupiah": {
        "BTC":        0.0,
        "ETH":        0.0,
        "SOL":        0.0,
        "US_STOCKS":  0.0,
        "IDX":       +0.3,
        "XAUUSD":     0.0,
        "XTIUSD":     0.0,
        "DXY":        0.0,
        "USDIDR":    -0.9,
        "EURUSD":     0.0,
        "USDJPY":     0.0,
    },
    "commodity": {
        "BTC":       +0.1,
        "ETH":       +0.1,
        "SOL":       +0.1,
        "US_STOCKS": +0.1,
        "IDX":       +0.6,
        "XAUUSD":    +0.4,
        "XTIUSD":    +0.5,
        "DXY":       -0.2,
        "USDIDR":    -0.3,
        "EURUSD":    +0.1,
        "USDJPY":    -0.2,
    },

    # ── New themes ───────────────────────────────────────────────────────────
    "ai_tech": {
        "BTC":       +0.2,
        "ETH":       +0.3,   # smart contracts / DeFi benefit from tech hype
        "SOL":       +0.3,
        "US_STOCKS": +0.5,   # tech sector leads
        "IDX":       +0.1,
        "XAUUSD":    -0.1,
        "XTIUSD":    -0.2,   # energy-efficiency narrative
        "DXY":       +0.1,
        "USDIDR":    -0.1,
        "EURUSD":    +0.1,
        "USDJPY":    +0.1,
    },
    "oil_price": {
        "BTC":       -0.1,
        "ETH":       -0.1,
        "SOL":       -0.1,
        "US_STOCKS": -0.3,   # higher energy costs compress margins
        "IDX":       +0.3,   # Indonesia is oil/energy exporter
        "XAUUSD":    +0.2,
        "XTIUSD":    +0.9,   # direct
        "DXY":       -0.3,   # oil priced in USD; stronger demand for non-USD oil buyers
        "USDIDR":    -0.2,   # higher oil = more USD inflows for Indonesian exporters
        "EURUSD":    +0.1,
        "USDJPY":    -0.2,   # Japan imports oil → yen weakens
    },
    "ojk_regulation": {
        "BTC":       -0.3,
        "ETH":       -0.3,
        "SOL":       -0.3,
        "US_STOCKS":  0.0,
        "IDX":       -0.2,
        "XAUUSD":     0.0,
        "XTIUSD":     0.0,
        "DXY":        0.0,
        "USDIDR":    +0.1,
        "EURUSD":     0.0,
        "USDJPY":     0.0,
    },
    "bbm_harga": {
        "BTC":        0.0,
        "ETH":        0.0,
        "SOL":        0.0,
        "US_STOCKS":  0.0,
        "IDX":       -0.3,   # higher fuel → inflation → BI tightening → IDX bearish
        "XAUUSD":    +0.1,
        "XTIUSD":    +0.5,
        "DXY":        0.0,
        "USDIDR":    +0.2,   # inflationary pressure on rupiah
        "EURUSD":     0.0,
        "USDJPY":     0.0,
    },
    "us_debt": {
        "BTC":       +0.4,   # dollar debasement / store-of-value narrative
        "ETH":       +0.3,
        "SOL":       +0.2,
        "US_STOCKS": -0.3,
        "IDX":       -0.2,
        "XAUUSD":    +0.6,
        "XTIUSD":    +0.2,
        "DXY":       -0.5,
        "USDIDR":    +0.3,
        "EURUSD":    +0.4,
        "USDJPY":    -0.2,
    },
}


# ── Crypto-specific theme overlays ───────────────────────────────────────────
# Themes not yet in MACRO_KEYWORDS but produced by web_ingestor.classify_extra
CRYPTO_THEME_IMPACT: dict[str, dict[str, float]] = {
    "etf_flows":     {"BTC": +0.8, "ETH": +0.5, "SOL": +0.3, "US_STOCKS": +0.1, "XAUUSD": -0.2},
    "regulation":    {"BTC": -0.6, "ETH": -0.6, "SOL": -0.5, "US_STOCKS": -0.1},
    "hack_exploit":  {"BTC": -0.7, "ETH": -0.8, "SOL": -0.9},
    "halving":       {"BTC": +0.5, "ETH": +0.2, "SOL": +0.1},
    "stablecoin":    {"BTC": -0.3, "ETH": -0.4, "SOL": -0.3, "DXY": +0.1},
}

# Merge crypto overlays into the canonical map
for _theme, _weights in CRYPTO_THEME_IMPACT.items():
    THEME_IMPACT.setdefault(_theme, {}).update(_weights)


# ── Theme polarity convention ────────────────────────────────────────────────
# Event themes whose articles are typically FinBERT-negative when the event
# *happens* (so the scorer must flip the sentiment sign to read it as
# "event intensity").  Everything not in this set is treated as a *topical*
# theme — sentiment sign is taken at face value.
BEARISH_TONED_THEMES: set[str] = {
    "rate_hike",
    "inflation",
    "recession",
    "liquidity",
    "geopolitical",
    "supply_chain",
    "regulation",
    "hack_exploit",
    "stablecoin",     # depeg / crisis stories
    "ojk_regulation", # regulator enforcement stories tend to be negative
    "bbm_harga",      # fuel price hike articles are typically negative
    "us_debt",        # debt ceiling / downgrade stories are negative events
}


def asset_bias_for_theme(theme: str, sentiment_score: float) -> dict[str, float]:
    """
    Compute per-asset directional bias contribution from a single (theme,
    sentiment) signal.

    Returns dict[asset → signed contribution in [-1, +1]].
    """
    weights = THEME_IMPACT.get(theme, {})
    return {asset: round(w * sentiment_score, 4) for asset, w in weights.items()}


def known_themes() -> list[str]:
    return list(THEME_IMPACT.keys())
