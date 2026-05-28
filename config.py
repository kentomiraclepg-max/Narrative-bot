import os
from dotenv import load_dotenv

load_dotenv()

# ── Reddit ────────────────────────────────────────────────────────────────────
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "NarrativeBot/1.0")

# ── Twitter/X ─────────────────────────────────────────────────────────────────
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# ── Telegram alerts ───────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── RSS feeds ─────────────────────────────────────────────────────────────────
RSS_FEEDS = {
    # Tier 2 — global finance
    "marketwatch":       "https://feeds.marketwatch.com/marketwatch/topstories",
    "yahoofinance":      "https://finance.yahoo.com/rss/topstories",
    "seeking_alpha":     "https://seekingalpha.com/feed.xml",
    "investing_com":     "https://www.investing.com/rss/news.rss",
    "cna_business":      "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511",
    # Tier 3 — crypto
    "coindesk":          "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "theblock":          "https://www.theblock.co/rss.xml",
    # IDX-specific
    "kontan":            "https://www.kontan.co.id/rss/industri",
    "bisnis_id":         "https://www.bisnis.com/rss",
    "detik_finance":     "https://finance.detik.com/rss",
    "antara_finance":    "https://www.antaranews.com/rss/bisnis.xml",
    # Tier 1 — real-time wire
    "financialjuice":    "https://www.financialjuice.com/feed.ashx?xy=rss",
}

SOURCE_TIERS = {
    "marketwatch": 2, "yahoofinance": 2, "seeking_alpha": 2, "investing_com": 2,
    "cna_business": 2,
    "coindesk": 3, "theblock": 3,
    "kontan": 2, "bisnis_id": 2, "detik_finance": 2, "antara_finance": 2,
    "financialjuice": 1,
}

# ── Reddit subreddits ─────────────────────────────────────────────────────────
REDDIT_SUBS = [
    "wallstreetbets", "investing", "stocks", "SecurityAnalysis",
    "Economics", "MacroEconomics", "indonesia", "investasi",
]

# ── Regulatory sources ────────────────────────────────────────────────────────
SEC_BASE_URL = "https://efts.sec.gov/LATEST/search-index?q=%22market%22&dateRange=custom&startdt={start}&enddt={end}&forms=8-K"
FED_CALENDAR_URL  = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FED_SPEECHES_URL  = "https://www.federalreserve.gov/feeds/speeches.xml"
TREASURY_RSS_URL  = "https://home.treasury.gov/system/files/rss/press-releases.xml"
IDX_ANNOUNCE_URL  = "https://www.idx.co.id/primary/NewsAnnouncement/ResultAnnounce"

# ── NLP ───────────────────────────────────────────────────────────────────────
FINBERT_MODEL      = "ProsusAI/finbert"
FINBERT_TONE_MODEL = "yiyanghkust/finbert-tone"
SPACY_MODEL        = "en_core_web_lg"          # upgraded from sm: better NER accuracy
SEMANTIC_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_THEME_THRESHOLD = 0.35
NLP_BATCH_SIZE     = 16
MAX_TEXT_LENGTH    = 512

# ── Detection thresholds ──────────────────────────────────────────────────────
SPIKE_WINDOW_MINUTES    = 30    # rolling window for spike detection
SPIKE_Z_THRESHOLD       = 2.5  # z-score above this = spike
DIVERGENCE_THRESHOLD    = 0.4  # sentiment score diff to flag divergence
MIN_ARTICLES_FOR_SPIKE  = 3    # need at least this many articles in window

# ── Credibility ───────────────────────────────────────────────────────────────
MIN_CREDIBILITY_SCORE  = 0.3   # below this = filtered out
CROSS_VERIFY_THRESHOLD = 2     # need 2+ sources to confirm a narrative

# ── Storage ───────────────────────────────────────────────────────────────────
DB_PATH         = "data/narrative.db"
GRAPH_PATH      = "data/graphs/knowledge_graph.gpickle"
CACHE_DIR       = "data/cache"
CACHE_TTL_HOURS = 6

# ── Agent loop ────────────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS = 300    # full scan every 5 min
ALERT_COOLDOWN_SECONDS = 600   # min 10 min between same-topic alerts
