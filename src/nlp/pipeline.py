"""
NLP pipeline — v2 upgrades:
  1. Semantic theme detection  : sentence-transformers cosine similarity (ensemble with keywords)
  2. Ensemble sentiment        : FinBERT (60%) + FinBERT-Tone (40%) combined score
  3. spaCy en_core_web_lg      : better NER accuracy without GPU requirement
"""
from __future__ import annotations

import re
from typing import Any

import config

# ── Lazy-loaded model globals ─────────────────────────────────────────────────
_finbert          = None
_finbert_tone     = None
_tokenizer        = None
_tone_tokenizer   = None
_tone_failed      = False
_nlp              = None
_embedder         = None   # sentence-transformers
_theme_embeddings = None   # pre-computed per-theme anchor embeddings


# ── Theme anchor sentences (opsi 1) ──────────────────────────────────────────
# Each value is a rich description that semantically covers the theme.
# The embedder encodes these once at startup and compares against article embeddings.

_THEME_ANCHORS: dict[str, str] = {
    "rate_hike":    "Federal Reserve raises interest rates hawkish monetary tightening FOMC hike restrictive higher for longer 25bp 50bp",
    "rate_cut":     "Federal Reserve cuts interest rates dovish easing monetary pivot accommodative rate reduction FOMC cut",
    "inflation":    "inflation consumer prices CPI PPI rising cost of living price surge price pressure hot inflation core inflation",
    "recession":    "recession economic contraction GDP decline slowdown hard landing negative growth economic downturn job losses",
    "earnings":     "corporate earnings quarterly results profit beat miss guidance revenue EPS earnings season financial results",
    "geopolitical": "war military conflict sanctions tariffs trade war geopolitical tension invasion ceasefire escalation risk-off safe haven",
    "liquidity":    "liquidity crisis credit crunch bank run contagion financial stress systemic risk interbank default bank failure",
    "supply_chain": "supply chain shortage inventory disruption logistics shipping cost port congestion freight bottleneck",
    "currency":     "dollar US dollar forex exchange rate DXY currency devaluation strength weakness FX",
    "bi_rate":      "Bank Indonesia interest rate monetary policy BI rate suku bunga bank sentral kebijakan moneter",
    "ihsg":         "IHSG Jakarta composite index Indonesian stock market bursa efek indeks saham",
    "rupiah":       "rupiah IDR Indonesian currency exchange rate kurs depreciation appreciation rupiah melemah menguat",
    "commodity":    "coal nickel palm oil crude oil copper commodity prices mineral resources commodity market",
    "ai_tech":      "artificial intelligence AI machine learning technology breakthrough semiconductor chip GPU Nvidia tech stocks innovation",
    "oil_price":    "crude oil WTI Brent oil price OPEC energy prices petroleum barrel oil supply demand",
    "ojk_regulation": "OJK Otoritas Jasa Keuangan regulasi keuangan perbankan Indonesia financial regulator banking regulation",
    "bbm_harga":    "BBM harga bensin pertamina solar pertalite bahan bakar minyak subsidi fuel price Indonesia energy subsidy",
    "us_debt":      "US debt ceiling fiscal deficit US Treasury bond government shutdown debt limit credit rating downgrade",
}

_THEME_NAMES = list(_THEME_ANCHORS.keys())


# ── Model loaders ─────────────────────────────────────────────────────────────

def _get_finbert():
    global _finbert, _tokenizer
    if _finbert is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("[NLP] Loading FinBERT…")
        _tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL)
        _finbert   = AutoModelForSequenceClassification.from_pretrained(config.FINBERT_MODEL)
        _finbert.eval()
        print("[NLP] FinBERT ready.")
    return _finbert, _tokenizer


def _get_finbert_tone():
    global _finbert_tone, _tone_tokenizer, _tone_failed
    if _tone_failed:
        return None, None
    if _finbert_tone is None:
        from transformers import BertTokenizer, BertForSequenceClassification
        print("[NLP] Loading FinBERT-Tone…")
        try:
            _tone_tokenizer = BertTokenizer.from_pretrained(config.FINBERT_TONE_MODEL)
            _finbert_tone   = BertForSequenceClassification.from_pretrained(config.FINBERT_TONE_MODEL)
            _finbert_tone.eval()
            print("[NLP] FinBERT-Tone ready.")
        except Exception as e:
            print(f"[NLP] FinBERT-Tone load failed: {e} — tone analysis disabled.")
            _tone_failed = True
            return None, None
    return _finbert_tone, _tone_tokenizer


def _get_spacy():
    global _nlp
    if _nlp is None:
        import spacy
        print(f"[NLP] Loading spaCy ({config.SPACY_MODEL})…")
        _nlp = spacy.load(config.SPACY_MODEL)
        print("[NLP] spaCy ready.")
    return _nlp


def _get_embedder():
    """Opsi 1: lazy-load sentence-transformers + pre-compute theme anchor embeddings."""
    global _embedder, _theme_embeddings
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        print(f"[NLP] Loading semantic embedder ({config.SEMANTIC_MODEL})…")
        _embedder = SentenceTransformer(config.SEMANTIC_MODEL)
        anchors = list(_THEME_ANCHORS.values())
        _theme_embeddings = _embedder.encode(anchors, normalize_embeddings=True)
        print("[NLP] Semantic embedder ready.")
    return _embedder, _theme_embeddings


# ── Sentiment (opsi 2: ensemble) ──────────────────────────────────────────────

def get_sentiment(text: str) -> dict:
    """
    Ensemble sentiment: FinBERT (60%) + FinBERT-Tone (40%).
    Returns {label, score, confidence} where score ∈ [-1, +1].
    """
    import torch
    import torch.nn.functional as F

    text = _clean(text)
    if not text:
        return {"label": "neutral", "score": 0.0, "confidence": 0.0}

    # Primary: FinBERT — labels: 0=positive, 1=negative, 2=neutral
    model, tokenizer = _get_finbert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=config.MAX_TEXT_LENGTH, padding=True)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]

    finbert_score = float(probs[0] - probs[1])
    finbert_conf  = float(probs.max())

    # Secondary: FinBERT-Tone — signed score or None if unavailable
    tone_score = _tone_signed_score(text)

    # Weighted ensemble
    if tone_score is not None:
        ensemble_score = 0.6 * finbert_score + 0.4 * tone_score
    else:
        ensemble_score = finbert_score

    # Derive label from ensemble score with small dead-zone
    if ensemble_score > 0.05:
        label = "positive"
    elif ensemble_score < -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label":      label,
        "score":      round(ensemble_score, 4),
        "confidence": round(finbert_conf, 4),
    }


def _tone_signed_score(text: str) -> float | None:
    """Returns a signed [-1, +1] score from FinBERT-Tone, or None on failure."""
    import torch
    import torch.nn.functional as F
    try:
        model, tokenizer = _get_finbert_tone()
        if model is None or tokenizer is None:
            return None
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=config.MAX_TEXT_LENGTH, padding=True)
        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=-1)[0]
        score = 0.0
        for idx, lbl in model.config.id2label.items():
            p = float(probs[idx])
            if "positive" in lbl.lower():
                score += p
            elif "negative" in lbl.lower():
                score -= p
        return score
    except Exception:
        return None


def get_tone(text: str) -> dict:
    """Returns {tone: 'bullish'|'bearish'|'neutral', score: float}."""
    tone_score = _tone_signed_score(_clean(text))
    if tone_score is None:
        return {"tone": "neutral", "score": 0.0}

    if tone_score > 0.05:
        tone = "bullish"
    elif tone_score < -0.05:
        tone = "bearish"
    else:
        tone = "neutral"

    return {"tone": tone, "score": round(abs(tone_score), 4)}


# ── Entity extraction (opsi 3: en_core_web_lg) ────────────────────────────────

def extract_entities(text: str) -> list[dict]:
    """Extract named entities using spaCy en_core_web_lg."""
    nlp = _get_spacy()
    doc = nlp(_clean(text)[:5000])

    seen = set()
    entities = []
    for ent in doc.ents:
        key = (ent.text.strip(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        category = _map_entity_category(ent.label_)
        if category:
            entities.append({
                "text":     ent.text.strip(),
                "label":    ent.label_,
                "category": category,
            })
    return entities


def _map_entity_category(label: str) -> str | None:
    return {
        "ORG":    "COMPANY",
        "PERSON": "PERSON",
        "GPE":    "LOCATION",
        "LOC":    "LOCATION",
        "MONEY":  "MONEY",
        "PERCENT":"PERCENT",
        "DATE":   "DATE",
        "EVENT":  "EVENT",
        "PRODUCT":"PRODUCT",
        "LAW":    "POLICY",
        "NORP":   "GROUP",
    }.get(label)


# ── Theme detection (opsi 1: semantic ensemble) ───────────────────────────────

MACRO_KEYWORDS = {
    "rate_hike":    [
        "rate hike", "raise rates", "hawkish", "tightening", "fed hike",
        "rate hike odds", "hike rates", "fed on guard", "higher for longer",
        "restrictive policy", "fomc hike", "50bp", "25bp hike",
    ],
    "rate_cut":     [
        "rate cut", "lower rates", "dovish", "easing", "pivot",
        "rate cut odds", "cut rates", "fed cut", "fomc cut", "rate reduction",
        "accommodative", "50bp cut",
    ],
    "inflation":    [
        "inflation", "cpi", "ppi", "price pressure", "core inflation",
        "inflation rate", "consumer prices", "producer prices", "price index",
        "inflation data", "inflation report", "price surge", "cost of living",
        "hot cpi", "inflation surge",
    ],
    "recession":    [
        "recession", "contraction", "gdp decline", "slowdown",
        "economic downturn", "negative growth", "hard landing", "soft landing",
        "gdp shrink", "economic contraction",
    ],
    "earnings":     [
        "earnings beat", "earnings miss", "revenue growth", "guidance",
        "quarterly results", "profit", "eps beat", "eps miss", "revenue beat",
        "earnings season", "q1 results", "q2 results", "q3 results", "q4 results",
    ],
    "geopolitical": [
        "war", "sanctions", "tariff", "trade war", "conflict",
        "geopolitical", "tension", "safe-haven", "safe haven", "risk-off",
        "military", "invasion", "ceasefire", "escalation", "iran", "russia",
        "ukraine", "middle east", "north korea", "taiwan strait",
    ],
    "liquidity":    [
        "liquidity", "credit crunch", "bank run", "contagion",
        "financial stress", "systemic risk", "interbank", "credit event",
        "default risk", "bank failure",
    ],
    "supply_chain": [
        "supply chain", "shortage", "inventory", "logistics",
        "port congestion", "shipping cost", "freight", "disruption",
    ],
    "currency":     [
        "dollar", "rupiah", "yuan", "devaluation", "forex",
        "dxy", "dollar index", "usd", "currency", "fx", "exchange rate",
        "dollar strength", "dollar weakness",
    ],
    "ai_tech":      [
        "artificial intelligence", "ai model", "machine learning", "chatgpt",
        "nvidia", "semiconductor", "chip shortage", "gpu", "large language model",
        "llm", "ai chip", "tech rally", "tech stocks", "openai", "deepseek",
    ],
    "oil_price":    [
        "crude oil", "wti", "brent", "oil price", "opec", "oil supply",
        "energy price", "barrel", "petroleum", "oil demand", "oil cut",
        "oil production", "shale", "oil inventory",
    ],
    "us_debt":      [
        "debt ceiling", "fiscal deficit", "us debt", "treasury bond",
        "government shutdown", "debt limit", "credit rating", "fitch downgrade",
        "moody's downgrade", "us default", "fiscal cliff", "national debt",
    ],
}

IDX_KEYWORDS = {
    "bi_rate":   [
        "bi rate", "suku bunga bi", "bank indonesia", "rdk",
        "bank sentral", "bi7drr", "kebijakan moneter",
    ],
    "ihsg":      [
        "ihsg", "composite index", "bursa efek",
        "indeks harga saham", "idx composite", "jakarta composite",
    ],
    "rupiah":    [
        "rupiah", "idr", "kurs", "nilai tukar", "rupiah melemah",
        "rupiah menguat", "rupiah depreciat", "rupiah appreciat",
    ],
    "commodity": [
        "nikel", "batu bara", "sawit", "minyak", "coal", "nickel", "cpo",
        "palm oil", "crude palm", "tin", "copper", "tembaga", "timah",
        "komoditas", "commodity price",
    ],
    "ojk_regulation": [
        "ojk", "otoritas jasa keuangan", "regulasi ojk", "izin ojk",
        "sanksi ojk", "peraturan ojk", "ojk cabut izin",
    ],
    "bbm_harga": [
        "bbm", "harga bbm", "bensin naik", "pertalite", "pertamina",
        "solar subsidi", "bahan bakar", "harga bensin", "bbm naik",
        "subsidi bbm", "harga minyak subsidi",
    ],
}


def detect_macro_themes(text: str) -> list[str]:
    """Ensemble: keyword match (fast) ∪ semantic similarity (recall)."""
    return list(set(_detect_by_keywords(text)) | set(_detect_by_semantics(text)))


def _detect_by_keywords(text: str) -> list[str]:
    text_lower = text.lower()
    return [
        theme
        for theme, keywords in {**MACRO_KEYWORDS, **IDX_KEYWORDS}.items()
        if any(kw in text_lower for kw in keywords)
    ]


def _detect_by_semantics(text: str) -> list[str]:
    """Semantic fallback: catches paraphrases that keywords miss."""
    try:
        import numpy as np
        embedder, theme_embs = _get_embedder()
        text_emb = embedder.encode([_clean(text)[:512]], normalize_embeddings=True)
        sims = (text_emb @ theme_embs.T)[0]
        return [_THEME_NAMES[i] for i, s in enumerate(sims)
                if s >= config.SEMANTIC_THEME_THRESHOLD]
    except Exception as e:
        print(f"[NLP] Semantic theme error: {e}")
        return []


# ── Combined pipeline ─────────────────────────────────────────────────────────

def process(article: dict) -> dict:
    """Run full NLP pipeline on an article."""
    text = article.get("raw_text") or article.get("title", "")

    sentiment = get_sentiment(text)
    tone      = get_tone(text)
    entities  = extract_entities(text)
    themes    = detect_macro_themes(text)

    return {
        **article,
        "sentiment_label": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "sentiment_conf":  sentiment["confidence"],
        "tone":            tone["tone"],
        "tone_score":      tone["score"],
        "entities":        entities,
        "macro_themes":    themes,
    }


def process_batch(articles: list[dict], max_workers: int = 4) -> list[dict]:
    if len(articles) <= 1:
        return [process(a) for a in articles]
    # Warm up models before spawning threads (avoids race on lazy load)
    _get_finbert()
    _get_spacy()
    _get_embedder()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(process, articles))


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
