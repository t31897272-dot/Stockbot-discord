#!/usr/bin/env python3
"""
Trading Bot v3.2 - Discord Signal Bot
Modules: ML Ensemble (Transformer + LSTM + RF + GB + SVM)
         News NLP (FinBERT), Order Book, Macro, Fundamentals, Earnings
"""

import os
import asyncio
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import discord
from discord.ext import commands, tasks
from datetime import datetime, timedelta
import pytz
import requests

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

DISCORD_TOKEN      = os.getenv("DISCORD_TOKEN", "")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))
NEWSAPI_KEY        = os.getenv("NEWSAPI_KEY", "")
TWITTER_API_KEY    = os.getenv("TWITTER_API_KEY", "")
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET     = os.getenv("BINANCE_SECRET", "")

WATCHLIST_CRYPTO  = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
WATCHLIST_STOCKS  = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]
WATCHLIST         = WATCHLIST_CRYPTO + WATCHLIST_STOCKS

SIGNAL_THRESHOLD       = 0.70
SCAN_INTERVAL_MINUTES  = 60
LOOKBACK_DAYS          = 60
SEQUENCE_LENGTH        = 20

# ═══════════════════════════════════════════════════════
# MODULE 1 — TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast).mean()
    ema_slow    = series.ewm(span=slow).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series, period=20):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, sma, lower

def add_features(df):
    df    = df.copy()
    close = df["Close"]
    volume = df["Volume"]
    df["rsi"]          = compute_rsi(close)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(close)
    df["bb_upper"], df["bb_mid"], df["bb_lower"]   = compute_bollinger(close)
    df["ema9"]         = close.ewm(span=9).mean()
    df["ema21"]        = close.ewm(span=21).mean()
    df["ema50"]        = close.ewm(span=50).mean()
    df["sma200"]       = close.rolling(200).mean()
    df["vol_ma20"]     = volume.rolling(20).mean()
    df["vol_ratio"]    = volume / (df["vol_ma20"] + 1e-9)
    df["returns"]      = close.pct_change()
    df["volatility"]   = df["returns"].rolling(14).std()
    df["momentum"]     = close / close.shift(10) - 1
    df["price_vs_ema21"] = (close - df["ema21"]) / (df["ema21"] + 1e-9)
    df["bb_position"]  = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["target"]       = (close.shift(-1) > close).astype(int)
    df.dropna(inplace=True)
    return df

FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_position",
    "ema9", "ema21", "ema50", "vol_ratio",
    "returns", "volatility", "momentum", "price_vs_ema21"
]

# ═══════════════════════════════════════════════════════
# MODULE 2 — DEEP LEARNING MODELS
# ═══════════════════════════════════════════════════════

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                       dim_feedforward=128, dropout=dropout,
                                                       batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.classifier(x[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def train_deep_model(model, X_seq, y_seq, epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_t = torch.FloatTensor(X_seq)
    y_t = torch.LongTensor(y_seq)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model

# ═══════════════════════════════════════════════════════
# MODULE 3 — ML ENSEMBLE (5 MODELES)
# ═══════════════════════════════════════════════════════

def train_ensemble(df):
    X        = df[FEATURE_COLS].values
    y        = df["target"].values
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svm = SVC(probability=True, random_state=42)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    X_seq, y_seq = build_sequences(X_scaled, y, SEQUENCE_LENGTH)
    transformer  = train_deep_model(TransformerModel(input_dim=len(FEATURE_COLS)), X_seq, y_seq)
    lstm         = train_deep_model(LSTMModel(input_dim=len(FEATURE_COLS)), X_seq, y_seq)

    return {"scaler": scaler, "rf": rf, "gb": gb, "svm": svm,
            "transformer": transformer, "lstm": lstm}

def predict_ensemble(models, df):
    X      = df[FEATURE_COLS].values[-SEQUENCE_LENGTH-1:]
    X_sc   = models["scaler"].transform(X)
    X_flat = X_sc[-1].reshape(1, -1)

    p_rf  = models["rf"].predict_proba(X_flat)[0][1]
    p_gb  = models["gb"].predict_proba(X_flat)[0][1]
    p_svm = models["svm"].predict_proba(X_flat)[0][1]

    X_seq = torch.FloatTensor(X_sc[-SEQUENCE_LENGTH:]).unsqueeze(0)
    with torch.no_grad():
        p_tr = torch.softmax(models["transformer"](X_seq), dim=1)[0][1].item()
        p_ls = torch.softmax(models["lstm"](X_seq), dim=1)[0][1].item()

    votes = {"Transformer": p_tr, "LSTM": p_ls,
             "Random Forest": p_rf, "Gradient Boosting": p_gb, "SVM": p_svm}
    return np.mean(list(votes.values())), votes

# ═══════════════════════════════════════════════════════
# MODULE 4 — FEAR & GREED INDEX
# ═══════════════════════════════════════════════════════

def get_fear_greed():
    try:
        r     = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data  = r.json()["data"][0]
        value = int(data["value"])
        label = data["value_classification"]
        emoji = "😱" if value <= 25 else ("😟" if value <= 45 else
                ("😐" if value <= 55 else ("😊" if value <= 75 else "🤑")))
        return {"value": value, "label": label, "emoji": emoji}
    except Exception as e:
        logger.warning(f"Fear & Greed: {e}")
        return None

# ═══════════════════════════════════════════════════════
# MODULE 5 — NEWS NLP (FinBERT)
# ═══════════════════════════════════════════════════════

_finbert_tokenizer = None
_finbert_model     = None

def load_finbert():
    global _finbert_tokenizer, _finbert_model
    if _finbert_tokenizer is None:
        logger.info("Chargement FinBERT...")
        _finbert_tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_model     = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        _finbert_model.eval()

def analyze_sentiment_finbert(text):
    load_finbert()
    inputs = _finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = _finbert_model(**inputs)
    probs  = torch.softmax(outputs.logits, dim=1)[0]
    labels = ["positive", "negative", "neutral"]
    score  = probs[0].item() - probs[1].item()
    return labels[probs.argmax().item()], round(score, 2)

def get_news_sentiment(symbol):
    if not NEWSAPI_KEY:
        return None
    query = symbol.replace("-USD", "").replace("-", " ")
    try:
        url     = (f"https://newsapi.org/v2/everything?q={query}"
                   f"&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWSAPI_KEY}")
        arts    = requests.get(url, timeout=8).json().get("articles", [])
        results = []
        for a in arts[:3]:
            title = a.get("title", "")
            if title:
                sentiment, score = analyze_sentiment_finbert(title)
                results.append({"title": title[:80], "sentiment": sentiment, "score": score})
        if not results:
            return None
        avg_score = round(np.mean([r["score"] for r in results]), 2)
        return {"articles": results, "avg_score": avg_score}
    except Exception as e:
        logger.warning(f"News: {e}")
        return None

# ═══════════════════════════════════════════════════════
# MODULE 6 — TWITTER/X SENTIMENT (OPTIONNEL)
# ═══════════════════════════════════════════════════════

def get_twitter_sentiment(symbol):
    if not TWITTER_API_KEY:
        return None
    query   = symbol.replace("-USD", "").replace("-", " ")
    headers = {"Authorization": f"Bearer {TWITTER_API_KEY}"}
    try:
        url    = (f"https://api.twitter.com/2/tweets/search/recent"
                  f"?query={query}%20lang%3Aen&max_results=10")
        tweets = requests.get(url, headers=headers, timeout=8).json().get("data", [])
        if not tweets:
            return None
        scores = [analyze_sentiment_finbert(t["text"])[1] for t in tweets[:5]]
        avg    = round(np.mean(scores), 2)
        emoji  = "🟢" if avg > 0.1 else ("🔴" if avg < -0.1 else "🟡")
        label  = "Positif" if avg > 0.1 else ("Négatif" if avg < -0.1 else "Neutre")
        return {"score": avg, "label": label, "emoji": emoji}
    except Exception as e:
        logger.warning(f"Twitter: {e}")
        return None

# ═══════════════════════════════════════════════════════
# MODULE 7 — ORDER BOOK BINANCE (OPTIONNEL, CRYPTO ONLY)
# ═══════════════════════════════════════════════════════

def get_order_book(symbol):
    if not BINANCE_API_KEY:
        return None
    binance_sym = symbol.replace("-USD", "USDT").replace("-", "")
    try:
        depth    = requests.get(f"https://api.binance.com/api/v3/depth?symbol={binance_sym}&limit=20",
                                timeout=5).json()
        bids     = [(float(p), float(q)) for p, q in depth.get("bids", [])[:5]]
        asks     = [(float(p), float(q)) for p, q in depth.get("asks", [])[:5]]
        buy_vol  = sum(q for _, q in bids)
        sell_vol = sum(q for _, q in asks)
        total    = buy_vol + sell_vol + 1e-9

        try:
            liq_data  = requests.get(
                f"https://fapi.binance.com/fapi/v1/allForceOrders?symbol={binance_sym}&limit=10",
                timeout=5).json()
            liq_m = round(sum(float(o.get("origQty",0)) * float(o.get("price",0))
                              for o in liq_data if isinstance(liq_data, list)) / 1e6, 1)
        except:
            liq_m = 0

        return {
            "buy_ratio":      round(buy_vol / total * 100, 1),
            "sell_ratio":     round(sell_vol / total * 100, 1),
            "support":        max(p for p, _ in bids) if bids else None,
            "resistance":     min(p for p, _ in asks) if asks else None,
            "liquidations_m": liq_m
        }
    except Exception as e:
        logger.warning(f"Order book: {e}")
        return None

# ═══════════════════════════════════════════════════════
# MODULE 8 — MACRO DATA
# ═══════════════════════════════════════════════════════

def get_macro_data():
    try:
        def pct(ticker):
            df = yf.Ticker(ticker).history(period="2d")
            if len(df) >= 2:
                return round((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100, 2)
            return 0.0
        return {"spx_change": pct("^GSPC"),
                "ndq_change": pct("^IXIC"),
                "dxy_change": pct("DX-Y.NYB")}
    except Exception as e:
        logger.warning(f"Macro: {e}")
        return None

# ═══════════════════════════════════════════════════════
# MODULE 9 — FONDAMENTAUX & EARNINGS (ACTIONS ONLY)
# ═══════════════════════════════════════════════════════

def get_fundamentals(symbol):
    if symbol in WATCHLIST_CRYPTO:
        return None
    try:
        tk   = yf.Ticker(symbol)
        info = tk.info
        cal  = tk.calendar

        next_earn = None
        if cal is not None and not cal.empty:
            try:
                next_earn = str(cal.iloc[0, 0])[:10]
            except:
                pass

        pe       = info.get("trailingPE")
        eps      = info.get("trailingEps")
        rev_grow = info.get("revenueGrowth")
        margin   = info.get("profitMargins")

        return {
            "pe":         round(pe, 1) if pe else None,
            "eps":        round(eps, 2) if eps else None,
            "rev_growth": round(rev_grow * 100, 1) if rev_grow else None,
            "margin":     round(margin * 100, 1) if margin else None,
            "sector":     info.get("sector", "N/A"),
            "next_earn":  next_earn
        }
    except Exception as e:
        logger.warning(f"Fundamentals {symbol}: {e}")
        return None

# ═══════════════════════════════════════════════════════
# MODULE 10 — SIGNAL BUILDER & FORMAT DISCORD
# ═══════════════════════════════════════════════════════

def build_signal(symbol, confidence, votes, df):
    price    = df["Close"].iloc[-1]
    prev     = df["Close"].iloc[-2]
    change24 = round((price / prev - 1) * 100, 2)
    is_crypto = symbol in WATCHLIST_CRYPTO

    if confidence >= 0.65:
        action, color_emoji = "BUY",  "🟢"
    elif confidence <= 0.45:
        action, color_emoji = "SELL", "🔴"
    else:
        action, color_emoji = "HOLD", "🟡"

    fg         = get_fear_greed() if is_crypto else None
    news       = get_news_sentiment(symbol)
    twitter    = get_twitter_sentiment(symbol) if is_crypto else None
    order_book = get_order_book(symbol) if is_crypto else None
    macro      = get_macro_data()
    fundas     = get_fundamentals(symbol)

    extra = [confidence]
    if fg:
        fg_norm = fg["value"] / 100
        extra.append(fg_norm if action == "BUY" else 1 - fg_norm)
    if news:
        extra.append((news["avg_score"] + 1) / 2)
    if twitter:
        extra.append((twitter["score"] + 1) / 2)
    final_confidence = round(np.mean(extra), 2)

    return {
        "symbol": symbol, "action": action, "color_emoji": color_emoji,
        "price": price, "change24": change24, "confidence": final_confidence,
        "votes": votes, "tp": round(price * 1.05, 2), "sl": round(price * 0.975, 2),
        "fg": fg, "news": news, "twitter": twitter,
        "order_book": order_book, "macro": macro, "fundas": fundas,
        "is_crypto": is_crypto
    }

def format_discord_message(s):
    conf_pct = int(s["confidence"] * 100)
    bar      = "█" * (conf_pct // 10) + "░" * (10 - conf_pct // 10)
    ch_em    = "📈" if s["change24"] >= 0 else "📉"
    now_str  = datetime.now(pytz.timezone("Asia/Kuala_Lumpur")).strftime("%d/%m/%Y à %H:%M")
    n_votes  = sum(1 for v in s["votes"].values() if v >= 0.5)

    L = []
    L += [f"╔══════════════════════════════════════════╗",
          f"{s['color_emoji']}  SIGNAL {s['action']} — {s['symbol']}",
          f"╚══════════════════════════════════════════╝", "",
          f"💰 Prix actuel      : ${s['price']:,.2f}",
          f"{ch_em} Variation 24h    : {'+' if s['change24']>=0 else ''}{s['change24']}%", "",
          f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
          f"🧠  CONSENSUS ML ({n_votes}/5 modèles)",
          f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
    for mn, prob in s["votes"].items():
        em = "🔵" if prob >= 0.5 else "🔴"
        lb = "BUY  ✅" if prob >= 0.5 else "HOLD ⏸️"
        L.append(f"  {em} {mn:<20} → {lb}")

    L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
          "📡  SIGNAUX DE MARCHÉ",
          "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
    if s["fg"]:
        L.append(f"  {s['fg']['emoji']} Fear & Greed       : {s['fg']['value']} — {s['fg']['label']}")
    if s["twitter"]:
        tw = s["twitter"]
        L.append(f"  🐦 Sentiment Twitter : {tw['label']} ({tw['score']:+.2f}) {tw['emoji']}")

    if s["news"]:
        ns = s["news"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "📰  ANALYSE NEWS — FinBERT NLP",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        for a in ns["articles"][:3]:
            em = "🟢" if a["sentiment"] == "positive" else ("🔴" if a["sentiment"] == "negative" else "🟡")
            L += [f"  {em} "{a['title']}"",
                  f"      → Sentiment : {a['sentiment'].capitalize()} ({a['score']:+.2f})"]
        avg_em = "🟢" if ns["avg_score"] > 0.1 else ("🔴" if ns["avg_score"] < -0.1 else "🟡")
        L.append(f"  📰 Score news global  : {ns['avg_score']:+.2f} {avg_em}")

    if s["order_book"]:
        ob = s["order_book"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "📖  ORDER BOOK & LIQUIDATIONS",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        if ob["support"]:    L.append(f"  🟢 Mur acheteur    : ${ob['support']:,.0f} (support fort)")
        if ob["resistance"]: L.append(f"  🔴 Mur vendeur     : ${ob['resistance']:,.0f} (résistance)")
        if ob["liquidations_m"] > 0:
            L.append(f"  ⚡ Liquidations    : ${ob['liquidations_m']}M")
        L.append(f"  📊 Ratio Buy/Sell  : {ob['buy_ratio']}% / {ob['sell_ratio']}%")

    if s["macro"]:
        mc = s["macro"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "🌍  MACRO-ÉCONOMIE",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        L += [f"  📈 S&P 500         : {mc['spx_change']:+.2f}% {'🟢' if mc['spx_change']>=0 else '🔴'}",
              f"  📈 NASDAQ          : {mc['ndq_change']:+.2f}% {'🟢' if mc['ndq_change']>=0 else '🔴'}",
              f"  💵 DXY (Dollar)    : {mc['dxy_change']:+.2f}% {'🟢' if mc['dxy_change']<=0 else '🔴'}"]

    if s["fundas"]:
        fd = s["fundas"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "📊  ANALYSE FONDAMENTALE",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        if fd["pe"]:         L.append(f"  💹 P/E Ratio       : {fd['pe']}")
        if fd["eps"]:        L.append(f"  💰 EPS             : ${fd['eps']}")
        if fd["rev_growth"]: L.append(f"  📈 Revenue Growth  : {fd['rev_growth']:+.1f}% YoY {'🟢' if fd['rev_growth']>0 else '🔴'}")
        if fd["margin"]:     L.append(f"  🏆 Marge nette     : {fd['margin']}% {'🟢' if fd['margin']>15 else '🟡'}")
        if fd["sector"]:     L.append(f"  🏭 Secteur         : {fd['sector']}")
        if fd["next_earn"]:
            L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                  "📅  CALENDRIER EARNINGS",
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                  f"  ⚠️  Prochain earnings : {fd['next_earn']}"]

    L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
          f"🎯  CONFIANCE GLOBALE : {conf_pct}%  {bar}",
          "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "",
          f"🎯 Take Profit  : ${s['tp']:,.2f}  (+5.0%)",
          f"🛑 Stop Loss    : ${s['sl']:,.2f}  (-2.5%)",
          f"⚖️  Risk/Reward  : 1 : 2.0"]

    warns = []
    if s["fundas"] and s["fundas"]["next_earn"]:
        warns.append(f"→ Earnings le {s['fundas']['next_earn']} — volatilité possible")
    if s["order_book"] and s["order_book"]["resistance"]:
        warns.append(f"→ Résistance forte à ${s['order_book']['resistance']:,.0f}")
    if s["fg"] and s["fg"]["value"] >= 75:
        warns.append("→ Fear & Greed en Extreme Greed — risque de correction")
    if s["macro"] and s["macro"]["dxy_change"] > 0.5:
        warns.append("→ Dollar en hausse — pression baissière possible")
    if warns:
        L += ["", "⚠️  POINTS DE VIGILANCE :"]
        L += [f"  {w}" for w in warns]

    L += ["", f"⏱️  Analysé le : {now_str}",
          f"🔁  Prochain scan dans : {SCAN_INTERVAL_MINUTES} min"]

    return "```\n" + "\n".join(L) + "\n```"

# ═══════════════════════════════════════════════════════
# MODULE 11 — DISCORD BOT
# ═══════════════════════════════════════════════════════

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)
trained_models = {}

async def analyze_and_send(symbol, channel):
    try:
        logger.info(f"Analyse {symbol}...")
        df_raw = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if df_raw.empty or len(df_raw) < 50:
            logger.warning(f"Pas assez de données pour {symbol}")
            return
        df = add_features(df_raw)

        if symbol not in trained_models or datetime.now().weekday() == 0:
            logger.info(f"Entraînement {symbol}...")
            trained_models[symbol] = train_ensemble(df)

        confidence, votes = predict_ensemble(trained_models[symbol], df)

        if SIGNAL_THRESHOLD > confidence > (1 - SIGNAL_THRESHOLD):
            logger.info(f"{symbol}: confiance {confidence:.2f} — HOLD ignoré")
            return

        signal  = build_signal(symbol, confidence, votes, df)
        message = format_discord_message(signal)
        for chunk in [message[i:i+1900] for i in range(0, len(message), 1900)]:
            await channel.send(chunk)
        logger.info(f"Signal {signal['action']} envoyé : {symbol} ({int(confidence*100)}%)")
    except Exception as e:
        logger.error(f"Erreur {symbol}: {e}")

@tasks.loop(minutes=SCAN_INTERVAL_MINUTES)
async def scheduled_scan():
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if not channel:
        logger.error("Channel Discord introuvable")
        return
    logger.info("=== Scan automatique démarré ===")
    for symbol in WATCHLIST:
        await analyze_and_send(symbol, channel)
        await asyncio.sleep(5)

@bot.event
async def on_ready():
    logger.info(f"Bot connecté : {bot.user}")
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        await channel.send(
            "```\n🤖 Trading Bot v3.2 démarré !\n"
            "📊 ML Ensemble + FinBERT NLP + Order Book + Macro + Fondamentaux\n"
            f"👁️  Watchlist : {', '.join(WATCHLIST)}\n"
            f"🔁 Scan toutes les {SCAN_INTERVAL_MINUTES} min\n```")
    scheduled_scan.start()

@bot.command(name="analyse")
async def cmd_analyse(ctx, symbol: str = "BTC-USD"):
    await ctx.send(f"⏳ Analyse de **{symbol.upper()}** en cours...")
    await analyze_and_send(symbol.upper(), ctx.channel)

@bot.command(name="top5")
async def cmd_top5(ctx):
    await ctx.send("⏳ Analyse top 5 en cours...")
    for sym in WATCHLIST[:5]:
        await analyze_and_send(sym, ctx.channel)
        await asyncio.sleep(3)

@bot.command(name="watchlist")
async def cmd_watchlist(ctx):
    await ctx.send(f"```\n📋 WATCHLIST\n\n"
                   f"🪙 Crypto  : {', '.join(WATCHLIST_CRYPTO)}\n"
                   f"📈 Actions : {', '.join(WATCHLIST_STOCKS)}\n```")

@bot.command(name="status")
async def cmd_status(ctx):
    next_scan = SCAN_INTERVAL_MINUTES - (datetime.now().minute % SCAN_INTERVAL_MINUTES)
    await ctx.send(f"```\n🤖 TRADING BOT v3.2 — STATUT\n\n"
                   f"🧠 ML Ensemble     : ✅ Actif (5 modèles)\n"
                   f"📰 NewsAPI NLP     : {'✅ Actif' if NEWSAPI_KEY else '❌ Manquant'}\n"
                   f"🐦 Twitter/X       : {'✅ Actif' if TWITTER_API_KEY else '⚠️ Non configuré'}\n"
                   f"📖 Binance OB      : {'✅ Actif' if BINANCE_API_KEY else '⚠️ Non configuré'}\n"
                   f"🌍 Macro (SPX/DXY) : ✅ Actif\n"
                   f"🔁 Prochain scan   : dans ~{next_scan} min\n```")

@bot.command(name="aide")
async def cmd_aide(ctx):
    await ctx.send("```\n📖 COMMANDES DISPONIBLES\n\n"
                   "!analyse [SYMBOL]  → Analyse un actif (ex: !analyse NVDA)\n"
                   "!top5              → Analyse les 5 premiers actifs\n"
                   "!watchlist         → Liste des actifs surveillés\n"
                   "!status            → Statut du bot\n"
                   "!aide              → Cette aide\n```")

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN manquant !")
        exit(1)
    if not DISCORD_CHANNEL_ID:
        logger.error("DISCORD_CHANNEL_ID manquant !")
        exit(1)
    logger.info("Démarrage Trading Bot v3.2...")
    bot.run(DISCORD_TOKEN)
