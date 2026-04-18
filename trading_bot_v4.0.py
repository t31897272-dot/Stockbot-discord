#!/usr/bin/env python3
"""
Trading Bot v4.0 Ultra Pro
ML: RF + GB + SVM + XGBoost + LightGBM
Analyse Technique: RSI, MACD, Bollinger, EMA, Chandelier
Fondamentaux renforcés: Piotroski, FCF, ROE, D/E, Marge nette
Sentiment: VADER + TextBlob
Order Book, Macro, Fear&Greed, Twitter
"""

import os, asyncio, logging, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import discord
from discord.ext import commands, tasks
from datetime import datetime
import pytz, requests, nltk

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════
# CONFIGURATION
# ═══════════════════════════════

DISCORD_TOKEN      = os.getenv("DISCORD_TOKEN", "")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
NEWSAPI_KEY        = os.getenv("NEWSAPI_KEY", "")
TWITTER_API_KEY    = os.getenv("TWITTER_API_KEY", "")
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET     = os.getenv("BINANCE_SECRET", "")

WATCHLIST_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
WATCHLIST_STOCKS = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]
WATCHLIST        = WATCHLIST_CRYPTO + WATCHLIST_STOCKS

SIGNAL_THRESHOLD       = 0.70
SCAN_INTERVAL_MINUTES  = 60

vader = SentimentIntensityAnalyzer()

# ═══════════════════════════════
# MODULE 1 — INDICATEURS TECHNIQUES
# ═══════════════════════════════

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def compute_macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast).mean()
    es = series.ewm(span=slow).mean()
    ml = ef - es
    sl = ml.ewm(span=signal).mean()
    return ml, sl, ml - sl

def compute_bollinger(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + 2*std, sma, sma - 2*std

def detect_candle_pattern(df):
    """Détecte les patterns de chandeliers japonais."""
    o, h, l, c = df["Open"].iloc[-1], df["High"].iloc[-1], df["Low"].iloc[-1], df["Close"].iloc[-1]
    o2, c2 = df["Open"].iloc[-2], df["Close"].iloc[-2]
    body   = abs(c - o)
    range_ = h - l + 1e-9

    if body / range_ < 0.1:
        return "Doji ⚖️", "neutral"
    if c > o and c2 < o2 and c > o2 and o < c2:
        return "Bullish Engulfing 🟢", "bullish"
    if c < o and c2 > o2 and c < o2 and o > c2:
        return "Bearish Engulfing 🔴", "bearish"
    if c > o and (o - l) > 2 * body:
        return "Hammer 🔨", "bullish"
    if c < o and (h - o) > 2 * body:
        return "Shooting Star ⭐", "bearish"
    if c > o:
        return "Bougie Haussière 🟢", "bullish"
    return "Bougie Baissière 🔴", "bearish"

def add_features(df):
    df = df.copy()
    c, v = df["Close"], df["Volume"]
    df["rsi"]                   = compute_rsi(c)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(c)
    df["bb_upper"], df["bb_mid"], df["bb_lower"]   = compute_bollinger(c)
    df["ema9"]    = c.ewm(span=9).mean()
    df["ema21"]   = c.ewm(span=21).mean()
    df["ema50"]   = c.ewm(span=50).mean()
    df["ema200"]  = c.ewm(span=200).mean()
    df["sma200"]  = c.rolling(200).mean()
    df["vol_ma20"]    = v.rolling(20).mean()
    df["vol_ratio"]   = v / (df["vol_ma20"] + 1e-9)
    df["returns"]     = c.pct_change()
    df["volatility"]  = df["returns"].rolling(14).std()
    df["momentum"]    = c / c.shift(10) - 1
    df["price_vs_ema21"]  = (c - df["ema21"]) / (df["ema21"] + 1e-9)
    df["price_vs_ema200"] = (c - df["ema200"]) / (df["ema200"] + 1e-9)
    df["bb_position"]     = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["ema9_cross"]      = (df["ema9"]  > df["ema21"]).astype(int)
    df["ema21_cross"]     = (df["ema21"] > df["ema50"]).astype(int)
    df["golden_cross"]    = (df["ema50"] > df["sma200"]).astype(int)
    df["rsi_oversold"]    = (df["rsi"] < 30).astype(int)
    df["rsi_overbought"]  = (df["rsi"] > 70).astype(int)
    df["macd_bullish"]    = (df["macd"] > df["macd_signal"]).astype(int)
    df["vol_surge"]       = (df["vol_ratio"] > 1.5).astype(int)
    df["target"]          = (c.shift(-1) > c).astype(int)
    df.dropna(inplace=True)
    return df

FEATURES = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_position",
    "ema9", "ema21", "ema50", "vol_ratio",
    "returns", "volatility", "momentum", "price_vs_ema21",
    "price_vs_ema200", "ema9_cross", "ema21_cross", "golden_cross",
    "rsi_oversold", "rsi_overbought", "macd_bullish", "vol_surge"
]

# ═══════════════════════════════
# MODULE 2 — ENSEMBLE 5 MODÈLES
# ═══════════════════════════════

def train_ensemble(df):
    X  = df[FEATURES].values
    y  = df["target"].values
    sc = MinMaxScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, shuffle=False)

    rf    = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42).fit(Xtr, ytr)
    gb    = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42).fit(Xtr, ytr)
    svm   = SVC(probability=True, kernel="rbf", C=1.0, random_state=42).fit(Xtr, ytr)
    xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                use_label_encoder=False, eval_metric="logloss",
                                random_state=42, verbosity=0).fit(Xtr, ytr)
    lgb_m = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                random_state=42, verbose=-1).fit(Xtr, ytr)

    return {"scaler": sc, "rf": rf, "gb": gb, "svm": svm, "xgb": xgb_m, "lgb": lgb_m}

def predict_ensemble(models, df):
    X  = models["scaler"].transform(df[FEATURES].values)
    Xf = X[-1].reshape(1, -1)
    votes = {
        "Random Forest":      models["rf"].predict_proba(Xf)[0][1],
        "Gradient Boosting":  models["gb"].predict_proba(Xf)[0][1],
        "SVM":                models["svm"].predict_proba(Xf)[0][1],
        "XGBoost":            models["xgb"].predict_proba(Xf)[0][1],
        "LightGBM":           models["lgb"].predict_proba(Xf)[0][1],
    }
    return np.mean(list(votes.values())), votes

# ═══════════════════════════════
# MODULE 3 — ANALYSE TECHNIQUE DÉTAILLÉE
# ═══════════════════════════════

def get_technical_analysis(df):
    """Résumé complet de l'analyse technique."""
    last  = df.iloc[-1]
    price = last["Close"]

    rsi   = round(last["rsi"], 1)
    if rsi < 30:   rsi_label, rsi_em = "Survente", "🟢"
    elif rsi > 70: rsi_label, rsi_em = "Surachat", "🔴"
    else:          rsi_label, rsi_em = "Neutre",   "🟡"

    macd_bull = last["macd"] > last["macd_signal"]
    macd_label, macd_em = ("Haussier", "🟢") if macd_bull else ("Baissier", "🔴")

    bb_pos = last["bb_position"]
    if bb_pos > 0.8:   bb_label, bb_em = "Surachat (haut)", "🔴"
    elif bb_pos < 0.2: bb_label, bb_em = "Survente (bas)",  "🟢"
    else:              bb_label, bb_em = "Zone médiane",     "🟡"

    above_ema20  = price > last["ema21"]
    above_ema50  = price > last["ema50"]
    above_ema200 = price > last["ema200"]

    if above_ema20 and above_ema50 and above_ema200:
        trend_label, trend_em = "Tendance haussière forte", "🟢"
    elif above_ema20 and above_ema50:
        trend_label, trend_em = "Tendance haussière", "🟢"
    elif not above_ema20 and not above_ema50:
        trend_label, trend_em = "Tendance baissière forte", "🔴"
    else:
        trend_label, trend_em = "Tendance mixte", "🟡"

    golden = last["golden_cross"] == 1
    golden_label = "Golden Cross ✨" if golden else "Death Cross ⚠️"

    candle_name, candle_type = detect_candle_pattern(df)

    bullish_signals = sum([
        rsi < 30,
        macd_bull,
        bb_pos < 0.2,
        above_ema20,
        above_ema50,
        above_ema200,
        golden,
        candle_type == "bullish"
    ])

    return {
        "rsi": rsi, "rsi_label": rsi_label, "rsi_em": rsi_em,
        "macd_label": macd_label, "macd_em": macd_em,
        "macd_val": round(last["macd"], 4),
        "bb_label": bb_label, "bb_em": bb_em, "bb_pos": round(bb_pos * 100, 1),
        "trend_label": trend_label, "trend_em": trend_em,
        "above_ema20": above_ema20, "above_ema50": above_ema50, "above_ema200": above_ema200,
        "golden": golden, "golden_label": golden_label,
        "candle": candle_name, "candle_type": candle_type,
        "bullish_signals": bullish_signals,
        "vol_surge": last["vol_surge"] == 1,
    }

# ═══════════════════════════════
# MODULE 4 — FEAR & GREED
# ═══════════════════════════════

def get_fear_greed():
    try:
        d = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()["data"][0]
        v = int(d["value"])
        e = "😱" if v<=25 else ("😟" if v<=45 else ("😐" if v<=55 else ("😊" if v<=75 else "🤑")))
        return {"value": v, "label": d["value_classification"], "emoji": e}
    except:
        return None

# ═══════════════════════════════
# MODULE 5 — SENTIMENT NEWS
# ═══════════════════════════════

def analyze_sentiment(text):
    vs    = vader.polarity_scores(text)["compound"]
    tb    = TextBlob(text).sentiment.polarity
    score = round((vs + tb) / 2, 2)
    if score > 0.05:   return "positive", score
    elif score < -0.05: return "negative", score
    else:               return "neutral",  score

def get_news_sentiment(symbol):
    if not NEWSAPI_KEY: return None
    q = symbol.replace("-USD", "").replace("-", " ")
    try:
        arts = requests.get(
            f"https://newsapi.org/v2/everything?q={q}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWSAPI_KEY}",
            timeout=8).json().get("articles", [])
        res = []
        for a in arts[:3]:
            t = a.get("title", "")
            if t:
                s, sc = analyze_sentiment(t)
                res.append({"title": t[:80], "sentiment": s, "score": sc})
        if not res: return None
        return {"articles": res, "avg_score": round(np.mean([r["score"] for r in res]), 2)}
    except:
        return None

# ═══════════════════════════════
# MODULE 6 — TWITTER (OPTIONNEL)
# ═══════════════════════════════

def get_twitter_sentiment(symbol):
    if not TWITTER_API_KEY: return None
    q = symbol.replace("-USD", "").replace("-", " ")
    try:
        tweets = requests.get(
            f"https://api.twitter.com/2/tweets/search/recent?query={q}%20lang%3Aen&max_results=10",
            headers={"Authorization": f"Bearer {TWITTER_API_KEY}"}, timeout=8
        ).json().get("data", [])
        if not tweets: return None
        scores = [analyze_sentiment(t["text"])[1] for t in tweets[:5]]
        avg    = round(np.mean(scores), 2)
        return {
            "score": avg,
            "label": "Positif" if avg > 0.05 else ("Négatif" if avg < -0.05 else "Neutre"),
            "emoji": "🟢"      if avg > 0.05 else ("🔴"      if avg < -0.05 else "🟡")
        }
    except:
        return None

# ═══════════════════════════════
# MODULE 7 — ORDER BOOK
# ═══════════════════════════════

def get_order_book(symbol):
    if not BINANCE_API_KEY: return None
    sym = symbol.replace("-USD", "USDT").replace("-", "")
    try:
        d    = requests.get(f"https://api.binance.com/api/v3/depth?symbol={sym}&limit=20", timeout=5).json()
        bids = [(float(p), float(q)) for p, q in d.get("bids", [])[:5]]
        asks = [(float(p), float(q)) for p, q in d.get("asks", [])[:5]]
        bv   = sum(q for _, q in bids)
        sv   = sum(q for _, q in asks)
        tot  = bv + sv + 1e-9
        try:
            liq = requests.get(f"https://fapi.binance.com/fapi/v1/allForceOrders?symbol={sym}&limit=10", timeout=5).json()
            lm  = round(sum(float(o.get("origQty", 0)) * float(o.get("price", 0)) for o in liq if isinstance(liq, list)) / 1e6, 1)
        except:
            lm = 0
        return {
            "buy_ratio":       round(bv / tot * 100, 1),
            "sell_ratio":      round(sv / tot * 100, 1),
            "support":         max(p for p, _ in bids) if bids else None,
            "resistance":      min(p for p, _ in asks) if asks else None,
            "liquidations_m":  lm
        }
    except:
        return None

# ═══════════════════════════════
# MODULE 8 — MACRO
# ═══════════════════════════════

def get_macro():
    try:
        def pct(t):
            df = yf.Ticker(t).history(period="2d")
            return round((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100, 2) if len(df) >= 2 else 0.0
        return {"spx": pct("^GSPC"), "ndq": pct("^IXIC"), "dxy": pct("DX-Y.NYB")}
    except:
        return None

# ═══════════════════════════════
# MODULE 9 — FONDAMENTAUX RENFORCÉS
# ═══════════════════════════════

def compute_piotroski(info):
    """Score de Piotroski simplifié (0-9) — santé financière."""
    score = 0
    try:
        # Rentabilité (3 points)
        if info.get("returnOnAssets", 0) > 0:            score += 1
        if info.get("operatingCashflow", 0) > 0:         score += 1
        if info.get("revenueGrowth", 0) > 0:             score += 1
        # Levier & liquidité (3 points)
        de = info.get("debtToEquity", None)
        if de is not None and de < 1.0:                  score += 1
        cr = info.get("currentRatio", None)
        if cr is not None and cr > 1.0:                  score += 1
        shares = info.get("sharesOutstanding", 0)
        if shares > 0:                                   score += 1
        # Efficacité opérationnelle (3 points)
        if info.get("profitMargins", 0) > 0.1:           score += 1
        if info.get("revenueGrowth", 0) > 0.05:          score += 1
        if info.get("returnOnEquity", 0) > 0.1:          score += 1
    except:
        pass
    return score

def get_fundamentals(symbol):
    if symbol in WATCHLIST_CRYPTO: return None
    try:
        tk   = yf.Ticker(symbol)
        info = tk.info
        cal  = tk.calendar
        ne   = None
        if cal is not None and not cal.empty:
            try: ne = str(cal.iloc[0, 0])[:10]
            except: pass

        pe     = info.get("trailingPE")
        eps    = info.get("trailingEps")
        rg     = info.get("revenueGrowth")
        mg     = info.get("profitMargins")
        roe    = info.get("returnOnEquity")
        de     = info.get("debtToEquity")
        fcf    = info.get("freeCashflow")
        net_mg = info.get("netMargins")
        beta   = info.get("beta")
        pio    = compute_piotroski(info)

        if pio >= 7:   pio_label, pio_em = "Solide", "🟢"
        elif pio >= 5: pio_label, pio_em = "Correct", "🟡"
        else:          pio_label, pio_em = "Fragile", "🔴"

        return {
            "pe":        round(pe, 1)       if pe    else None,
            "eps":       round(eps, 2)      if eps   else None,
            "rev_growth":round(rg * 100, 1) if rg    else None,
            "margin":    round(mg * 100, 1) if mg    else None,
            "roe":       round(roe * 100, 1)if roe   else None,
            "de":        round(de, 2)       if de    else None,
            "fcf":       round(fcf / 1e9, 1)if fcf   else None,
            "net_margin":round(net_mg*100,1)if net_mg else None,
            "beta":      round(beta, 2)     if beta  else None,
            "piotroski": pio, "pio_label": pio_label, "pio_em": pio_em,
            "sector":    info.get("sector", "N/A"),
            "next_earn": ne
        }
    except:
        return None

# ═══════════════════════════════
# MODULE 10 — CONSTRUCTION SIGNAL
# ═══════════════════════════════

def build_signal(symbol, confidence, votes, df):
    price     = df["Close"].iloc[-1]
    change24  = round((price / df["Close"].iloc[-2] - 1) * 100, 2)
    is_crypto = symbol in WATCHLIST_CRYPTO

    if confidence >= 0.65:   action, cem = "BUY",  "🟢"
    elif confidence <= 0.45: action, cem = "SELL", "🔴"
    else:                    action, cem = "HOLD", "🟡"

    fg   = get_fear_greed()         if is_crypto else None
    news = get_news_sentiment(symbol)
    tw   = get_twitter_sentiment(symbol) if is_crypto else None
    ob   = get_order_book(symbol)   if is_crypto else None
    mc   = get_macro()
    fd   = get_fundamentals(symbol)
    ta   = get_technical_analysis(df)

    extra = [confidence]
    if fg:   extra.append(fg["value"] / 100 if action == "BUY" else 1 - fg["value"] / 100)
    if news: extra.append((news["avg_score"] + 1) / 2)
    if tw:   extra.append((tw["score"] + 1) / 2)
    ta_score = ta["bullish_signals"] / 8
    extra.append(ta_score)

    return {
        "symbol": symbol, "action": action, "cem": cem,
        "price": price, "change24": change24,
        "confidence": round(np.mean(extra), 2),
        "votes": votes,
        "tp": round(price * 1.05, 2), "sl": round(price * 0.975, 2),
        "fg": fg, "news": news, "twitter": tw, "ob": ob,
        "macro": mc, "fd": fd, "ta": ta, "is_crypto": is_crypto
    }

# ═══════════════════════════════
# MODULE 11 — FORMAT DISCORD
# ═══════════════════════════════

def fmt(s):
    cp  = int(s["confidence"] * 100)
    bar = "█" * (cp // 10) + "░" * (10 - cp // 10)
    cem = "📈" if s["change24"] >= 0 else "📉"
    now = datetime.now(pytz.timezone("Asia/Kuala_Lumpur")).strftime("%d/%m/%Y à %H:%M")
    nv  = sum(1 for v in s["votes"].values() if v >= 0.5)

    L = [
        "╔══════════════════════════════════════════╗",
        f"{s['cem']}  SIGNAL {s['action']} — {s['symbol']}",
        "╚══════════════════════════════════════════╝", "",
        f"💰 Prix actuel   : ${s['price']:,.2f}",
        f"{cem} Variation 24h : {s['change24']:+.2f}%", "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🧠  CONSENSUS ML ({nv}/5 modèles)",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]
    for mn, prob in s["votes"].items():
        L.append(f"  {'🔵' if prob>=0.5 else '🔴'} {mn:<20} → {'BUY  ✅' if prob>=0.5 else 'HOLD ⏸️'}")

    # Analyse Technique
    ta = s["ta"]
    L += [
        "", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📈  ANALYSE TECHNIQUE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  {ta['rsi_em']} RSI (14)       : {ta['rsi']} — {ta['rsi_label']}",
        f"  {ta['macd_em']} MACD           : {ta['macd_label']} ({ta['macd_val']:+.4f})",
        f"  {ta['bb_em']} Bollinger      : {ta['bb_label']} ({ta['bb_pos']}%)",
        f"  {ta['trend_em']} EMA 20/50/200 : {ta['trend_label']}",
        f"  ✨ {ta['golden_label']}",
        f"  🕯️  Chandelier   : {ta['candle']}",
    ]
    if ta["vol_surge"]:
        L.append("  🔥 Volume       : Surge détecté !")
    bull = ta["bullish_signals"]
    bull_bar = "🟢" * bull + "⚪" * (8 - bull)
    L.append(f"  📊 Score TA     : {bull}/8  {bull_bar}")

    if s["fg"] or s["twitter"]:
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "📡  SIGNAUX DE MARCHÉ",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        if s["fg"]:
            L.append(f"  {s['fg']['emoji']} Fear & Greed  : {s['fg']['value']} — {s['fg']['label']}")
        if s["twitter"]:
            tw = s["twitter"]
            L.append(f"  🐦 Twitter      : {tw['label']} ({tw['score']:+.2f}) {tw['emoji']}")

    if s["news"]:
        ns = s["news"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "📰  ANALYSE NEWS",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        for a in ns["articles"][:3]:
            em = "🟢" if a["sentiment"] == "positive" else ("🔴" if a["sentiment"] == "negative" else "🟡")
            L += [f"  {em} '{a['title']}'",
                  f"      → {a['sentiment'].capitalize()} ({a['score']:+.2f})"]
        avg_em = "🟢" if ns["avg_score"] > 0.05 else ("🔴" if ns["avg_score"] < -0.05 else "🟡")
        L.append(f"  📰 Score global : {ns['avg_score']:+.2f} {avg_em}")

    if s["ob"]:
        ob = s["ob"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "📖  ORDER BOOK",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        if ob["support"]:    L.append(f"  🟢 Support      : ${ob['support']:,.0f}")
        if ob["resistance"]: L.append(f"  🔴 Résistance   : ${ob['resistance']:,.0f}")
        if ob["liquidations_m"] > 0:
            L.append(f"  ⚡ Liquidations : ${ob['liquidations_m']}M")
        L.append(f"  📊 Buy/Sell     : {ob['buy_ratio']}% / {ob['sell_ratio']}%")

    if s["macro"]:
        mc = s["macro"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "🌍  MACRO-ÉCONOMIE",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              f"  📈 S&P 500  : {mc['spx']:+.2f}% {'🟢' if mc['spx']>=0 else '🔴'}",
              f"  📈 NASDAQ   : {mc['ndq']:+.2f}% {'🟢' if mc['ndq']>=0 else '🔴'}",
              f"  💵 DXY      : {mc['dxy']:+.2f}% {'🟢' if mc['dxy']<=0 else '🔴'}"]

    if s["fd"]:
        fd = s["fd"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "🏦  FONDAMENTAUX",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              f"  {fd['pio_em']} Piotroski    : {fd['piotroski']}/9 — {fd['pio_label']}"]
        if fd["pe"]:         L.append(f"  💹 P/E Ratio   : {fd['pe']}")
        if fd["eps"]:        L.append(f"  💰 EPS         : ${fd['eps']}")
        if fd["roe"]:
            em = "🟢" if fd["roe"] > 15 else ("🟡" if fd["roe"] > 5 else "🔴")
            L.append(f"  {em} ROE          : {fd['roe']}%")
        if fd["de"] is not None:
            em = "🟢" if fd["de"] < 0.5 else ("🟡" if fd["de"] < 1.5 else "🔴")
            L.append(f"  {em} Dette/Cap.   : {fd['de']}")
        if fd["fcf"]:
            em = "🟢" if fd["fcf"] > 0 else "🔴"
            L.append(f"  {em} Free CF      : ${fd['fcf']}B")
        if fd["net_margin"]:
            em = "🟢" if fd["net_margin"] > 15 else ("🟡" if fd["net_margin"] > 5 else "🔴")
            L.append(f"  {em} Marge nette  : {fd['net_margin']}%")
        if fd["rev_growth"]:
            em = "🟢" if fd["rev_growth"] > 0 else "🔴"
            L.append(f"  {em} Rev Growth   : {fd['rev_growth']:+.1f}%")
        if fd["beta"]:       L.append(f"  ⚡ Bêta         : {fd['beta']}")
        if fd["sector"]:     L.append(f"  🏭 Secteur      : {fd['sector']}")
        if fd["next_earn"]:
            L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                  "📅  EARNINGS",
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                  f"  ⚠️  Prochain earnings : {fd['next_earn']}"]

    L += [
        "", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🎯  CONFIANCE : {cp}%  {bar}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "",
        f"🎯 Take Profit : ${s['tp']:,.2f}  (+5.0%)",
        f"🛑 Stop Loss   : ${s['sl']:,.2f}  (-2.5%)",
        f"⚖️  Risk/Reward : 1 : 2.0"
    ]

    warns = []
    if s["fd"] and s["fd"]["next_earn"]:
        warns.append(f"→ Earnings le {s['fd']['next_earn']} — volatilité possible")
    if s["fd"] and s["fd"]["de"] and s["fd"]["de"] > 2:
        warns.append(f"→ Endettement élevé (D/E={s['fd']['de']})")
    if s["fd"] and s["fd"]["piotroski"] and s["fd"]["piotroski"] < 4:
        warns.append("→ Score Piotroski faible — santé financière fragile")
    if s["ob"] and s["ob"]["resistance"]:
        warns.append(f"→ Résistance forte à ${s['ob']['resistance']:,.0f}")
    if s["fg"] and s["fg"]["value"] >= 75:
        warns.append("→ Extreme Greed — surveiller une correction")
    if s["macro"] and s["macro"]["dxy"] > 0.5:
        warns.append("→ Dollar en hausse — pression baissière")
    if s["ta"]["rsi"] > 70:
        warns.append(f"→ RSI en surachat ({s['ta']['rsi']}) — risque de correction")
    if warns:
        L += ["", "⚠️  POINTS DE VIGILANCE :"] + [f"  {w}" for w in warns]

    L += ["", f"⏱️  Analysé le : {now}", f"🔁  Prochain scan : {SCAN_INTERVAL_MINUTES} min"]
    return "```\n" + "\n".join(L) + "\n```"

# ═══════════════════════════════
# MODULE 12 — DISCORD BOT
# ═══════════════════════════════

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
trained_models = {}

async def analyze_and_send(symbol, channel):
    try:
        logger.info(f"Analyse {symbol}...")
        df_raw = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if df_raw.empty or len(df_raw) < 50: return
        df = add_features(df_raw)
        if symbol not in trained_models or datetime.now().weekday() == 0:
            logger.info(f"Entraînement {symbol}...")
            trained_models[symbol] = train_ensemble(df)
        confidence, votes = predict_ensemble(trained_models[symbol], df)
        if SIGNAL_THRESHOLD > confidence > (1 - SIGNAL_THRESHOLD):
            logger.info(f"{symbol}: HOLD ignoré ({confidence:.2f})"); return
        signal  = build_signal(symbol, confidence, votes, df)
        message = fmt(signal)
        for chunk in [message[i:i+1900] for i in range(0, len(message), 1900)]:
            await channel.send(chunk)
        logger.info(f"✅ {signal['action']} — {symbol} ({int(confidence*100)}%)")
    except Exception as e:
        logger.error(f"Erreur {symbol}: {e}")

@tasks.loop(minutes=SCAN_INTERVAL_MINUTES)
async def scheduled_scan():
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if not channel: return
    logger.info("=== Scan automatique ===")
    for symbol in WATCHLIST:
        await analyze_and_send(symbol, channel)
        await asyncio.sleep(5)

@bot.event
async def on_ready():
    logger.info(f"✅ Bot connecté : {bot.user}")
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        tw = "✅" if TWITTER_API_KEY else "⚠️ optionnel"
        bn = "✅" if BINANCE_API_KEY else "⚠️ optionnel"
        ns = "✅" if NEWSAPI_KEY    else "❌ manquant"
        await channel.send(
            f"```\n🤖 Trading Bot v4.0 Ultra Pro démarré !\n"
            f"🧠 ML      : RF + GB + SVM + XGBoost + LightGBM\n"
            f"📈 Tech    : RSI + MACD + Bollinger + EMA + Chandelier\n"
            f"🏦 Fondamentaux : Piotroski + FCF + ROE + D/E\n"
            f"📰 News    : VADER + TextBlob {ns}\n"
            f"🐦 Twitter : {tw} | 📖 Order Book : {bn}\n"
            f"👁️  Watchlist : {len(WATCHLIST)} actifs surveillés\n"
            f"🔁 Scan toutes les {SCAN_INTERVAL_MINUTES} min\n```")
        scheduled_scan.start()

@bot.command(name="analyse")
async def cmd_analyse(ctx, symbol: str = "BTC-USD"):
    await ctx.send(f"⏳ Analyse **{symbol.upper()}** en cours...")
    await analyze_and_send(symbol.upper(), ctx.channel)

@bot.command(name="top5")
async def cmd_top5(ctx):
    await ctx.send("⏳ Top 5 en cours...")
    for sym in WATCHLIST[:5]:
        await analyze_and_send(sym, ctx.channel)
        await asyncio.sleep(3)

@bot.command(name="watchlist")
async def cmd_watchlist(ctx):
    await ctx.send(
        f"```\n📋 WATCHLIST\n\n"
        f"🪙 Crypto  : {chr(44).join(WATCHLIST_CRYPTO)}\n"
        f"📈 Actions : {chr(44).join(WATCHLIST_STOCKS)}\n```")

@bot.command(name="status")
async def cmd_status(ctx):
    ns = SCAN_INTERVAL_MINUTES - (datetime.now().minute % SCAN_INTERVAL_MINUTES)
    await ctx.send(
        f"```\n🤖 TRADING BOT v4.0 Ultra Pro\n\n"
        f"🧠 ML Ensemble    : ✅ (RF+GB+SVM+XGB+LGBM)\n"
        f"📈 Analyse Tech   : ✅ (RSI+MACD+BB+EMA+Chandelier)\n"
        f"🏦 Fondamentaux   : ✅ (Piotroski+FCF+ROE+D/E)\n"
        f"📰 News VADER     : {'✅' if NEWSAPI_KEY else '❌'}\n"
        f"🐦 Twitter        : {'✅' if TWITTER_API_KEY else '⚠️ optionnel'}\n"
        f"📖 Order Book     : {'✅' if BINANCE_API_KEY else '⚠️ optionnel'}\n"
        f"🌍 Macro SPX/DXY  : ✅\n"
        f"🔁 Prochain scan  : dans ~{ns} min\n```")

@bot.command(name="aide")
async def cmd_aide(ctx):
    await ctx.send(
        "```\n📖 COMMANDES\n\n"
        "!analyse [SYMBOL] → Signal complet immédiat (ex: !analyse NVDA)\n"
        "!top5             → Analyse les 5 premiers actifs\n"
        "!watchlist        → Liste des actifs surveillés\n"
        "!status           → Statut du bot\n"
        "!aide             → Cette aide\n```")

if __name__ == "__main__":
    if not DISCORD_TOKEN:      logger.error("DISCORD_TOKEN manquant !");      exit(1)
    if not DISCORD_CHANNEL_ID: logger.error("DISCORD_CHANNEL_ID manquant !"); exit(1)
    logger.info("Démarrage Trading Bot v4.0 Ultra Pro...")
    bot.run(DISCORD_TOKEN)
