#!/usr/bin/env python3
"""
Trading Bot v3.2 Lite - Discord Signal Bot
ML Ensemble (Transformer + LSTM + RF + GB + SVM)
Sentiment: VADER + TextBlob (léger, pas de FinBERT)
Order Book Binance, Macro, Fondamentaux, Earnings
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
from datetime import datetime
import pytz
import requests

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Sentiment léger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
nltk.download("vader_lexicon", quiet=True)

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

SIGNAL_THRESHOLD      = 0.70
SCAN_INTERVAL_MINUTES = 60
SEQUENCE_LENGTH       = 20

vader = SentimentIntensityAnalyzer()

# ═══════════════════════════════════════════════════════
# MODULE 1 — INDICATEURS TECHNIQUES
# ═══════════════════════════════════════════════════════

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast).mean()
    ema_slow    = series.ewm(span=slow).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line, macd_line - signal_line

def compute_bollinger(series, period=20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + 2*std, sma, sma - 2*std

def add_features(df):
    df    = df.copy()
    close = df["Close"]
    vol   = df["Volume"]

    df["rsi"]          = compute_rsi(close)
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(close)
    df["bb_upper"], df["bb_mid"], df["bb_lower"]   = compute_bollinger(close)
    df["ema9"]         = close.ewm(span=9).mean()
    df["ema21"]        = close.ewm(span=21).mean()
    df["ema50"]        = close.ewm(span=50).mean()
    df["vol_ma20"]     = vol.rolling(20).mean()
    df["vol_ratio"]    = vol / (df["vol_ma20"] + 1e-9)
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
# MODULE 2 — MODELES DEEP LEARNING
# ═══════════════════════════════════════════════════════

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        enc_layer        = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                       dim_feedforward=128, dropout=dropout,
                                                       batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x):
        return self.classifier(self.transformer(self.input_proj(x))[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm       = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def train_deep(model, X_seq, y_seq, epochs=30, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    Xt = torch.FloatTensor(X_seq)
    yt = torch.LongTensor(y_seq)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        crit(model(Xt), yt).backward()
        opt.step()
    model.eval()
    return model

# ═══════════════════════════════════════════════════════
# MODULE 3 — ENSEMBLE 5 MODELES
# ═══════════════════════════════════════════════════════

def train_ensemble(df):
    X = df[FEATURE_COLS].values
    y = df["target"].values
    scaler   = MinMaxScaler()
    X_sc     = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_sc, y, test_size=0.2, shuffle=False)

    rf  = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    gb  = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    svm = SVC(probability=True, random_state=42).fit(Xtr, ytr)

    Xs, ys = build_sequences(X_sc, y, SEQUENCE_LENGTH)
    tr_model = train_deep(TransformerModel(len(FEATURE_COLS)), Xs, ys)
    ls_model = train_deep(LSTMModel(len(FEATURE_COLS)), Xs, ys)

    return {"scaler": scaler, "rf": rf, "gb": gb, "svm": svm,
            "transformer": tr_model, "lstm": ls_model}

def predict_ensemble(models, df):
    X    = models["scaler"].transform(df[FEATURE_COLS].values[-SEQUENCE_LENGTH-1:])
    Xf   = X[-1].reshape(1, -1)
    p_rf  = models["rf"].predict_proba(Xf)[0][1]
    p_gb  = models["gb"].predict_proba(Xf)[0][1]
    p_svm = models["svm"].predict_proba(Xf)[0][1]
    Xs   = torch.FloatTensor(X[-SEQUENCE_LENGTH:]).unsqueeze(0)
    with torch.no_grad():
        p_tr = torch.softmax(models["transformer"](Xs), dim=1)[0][1].item()
        p_ls = torch.softmax(models["lstm"](Xs), dim=1)[0][1].item()
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
        emoji = "😱" if value<=25 else ("😟" if value<=45 else
                ("😐" if value<=55 else ("😊" if value<=75 else "🤑")))
        return {"value": value, "label": label, "emoji": emoji}
    except Exception as e:
        logger.warning(f"Fear & Greed: {e}")
        return None

# ═══════════════════════════════════════════════════════
# MODULE 5 — SENTIMENT NEWS (VADER + TEXTBLOB)
# ═══════════════════════════════════════════════════════

def analyze_sentiment(text):
    vs     = vader.polarity_scores(text)["compound"]
    tb     = TextBlob(text).sentiment.polarity
    score  = round((vs + tb) / 2, 2)
    if score > 0.05:   return "positive", score
    elif score < -0.05: return "negative", score
    else:              return "neutral", score

def get_news_sentiment(symbol):
    if not NEWSAPI_KEY:
        return None
    query = symbol.replace("-USD", "").replace("-", " ")
    try:
        url  = (f"https://newsapi.org/v2/everything?q={query}"
                f"&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWSAPI_KEY}")
        arts = requests.get(url, timeout=8).json().get("articles", [])
        results = []
        for a in arts[:3]:
            title = a.get("title", "")
            if title:
                sentiment, score = analyze_sentiment(title)
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
        scores = [analyze_sentiment(t["text"])[1] for t in tweets[:5]]
        avg    = round(np.mean(scores), 2)
        emoji  = "🟢" if avg > 0.05 else ("🔴" if avg < -0.05 else "🟡")
        label  = "Positif" if avg > 0.05 else ("Négatif" if avg < -0.05 else "Neutre")
        return {"score": avg, "label": label, "emoji": emoji}
    except Exception as e:
        logger.warning(f"Twitter: {e}")
        return None

# ═══════════════════════════════════════════════════════
# MODULE 7 — ORDER BOOK BINANCE (OPTIONNEL)
# ═══════════════════════════════════════════════════════

def get_order_book(symbol):
    if not BINANCE_API_KEY:
        return None
    sym = symbol.replace("-USD", "USDT").replace("-", "")
    try:
        depth    = requests.get(f"https://api.binance.com/api/v3/depth?symbol={sym}&limit=20",
                                timeout=5).json()
        bids     = [(float(p), float(q)) for p, q in depth.get("bids", [])[:5]]
        asks     = [(float(p), float(q)) for p, q in depth.get("asks", [])[:5]]
        buy_vol  = sum(q for _, q in bids)
        sell_vol = sum(q for _, q in asks)
        total    = buy_vol + sell_vol + 1e-9
        try:
            liq_data = requests.get(
                f"https://fapi.binance.com/fapi/v1/allForceOrders?symbol={sym}&limit=10",
                timeout=5).json()
            liq_m = round(sum(float(o.get("origQty",0))*float(o.get("price",0))
                              for o in liq_data if isinstance(liq_data, list)) / 1e6, 1)
        except:
            liq_m = 0
        return {
            "buy_ratio":  round(buy_vol/total*100, 1),
            "sell_ratio": round(sell_vol/total*100, 1),
            "support":    max(p for p, _ in bids) if bids else None,
            "resistance": min(p for p, _ in asks) if asks else None,
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
            return round((df["Close"].iloc[-1]/df["Close"].iloc[-2]-1)*100, 2) if len(df)>=2 else 0.0
        return {"spx_change": pct("^GSPC"), "ndq_change": pct("^IXIC"), "dxy_change": pct("DX-Y.NYB")}
    except Exception as e:
        logger.warning(f"Macro: {e}")
        return None

# ═══════════════════════════════════════════════════════
# MODULE 9 — FONDAMENTAUX & EARNINGS (ACTIONS)
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
            try: next_earn = str(cal.iloc[0, 0])[:10]
            except: pass
        pe  = info.get("trailingPE")
        eps = info.get("trailingEps")
        rg  = info.get("revenueGrowth")
        mg  = info.get("profitMargins")
        return {
            "pe":        round(pe, 1) if pe else None,
            "eps":       round(eps, 2) if eps else None,
            "rev_growth":round(rg*100, 1) if rg else None,
            "margin":    round(mg*100, 1) if mg else None,
            "sector":    info.get("sector", "N/A"),
            "next_earn": next_earn
        }
    except Exception as e:
        logger.warning(f"Fundamentals {symbol}: {e}")
        return None

# ═══════════════════════════════════════════════════════
# MODULE 10 — CONSTRUCTION DU SIGNAL
# ═══════════════════════════════════════════════════════

def build_signal(symbol, confidence, votes, df):
    price     = df["Close"].iloc[-1]
    change24  = round((price / df["Close"].iloc[-2] - 1) * 100, 2)
    is_crypto = symbol in WATCHLIST_CRYPTO

    if confidence >= 0.65:   action, cem = "BUY",  "🟢"
    elif confidence <= 0.45: action, cem = "SELL", "🔴"
    else:                    action, cem = "HOLD", "🟡"

    fg   = get_fear_greed() if is_crypto else None
    news = get_news_sentiment(symbol)
    tw   = get_twitter_sentiment(symbol) if is_crypto else None
    ob   = get_order_book(symbol) if is_crypto else None
    mc   = get_macro_data()
    fd   = get_fundamentals(symbol)

    extra = [confidence]
    if fg:   extra.append(fg["value"]/100 if action=="BUY" else 1-fg["value"]/100)
    if news: extra.append((news["avg_score"]+1)/2)
    if tw:   extra.append((tw["score"]+1)/2)
    final = round(np.mean(extra), 2)

    return {"symbol": symbol, "action": action, "color_emoji": cem,
            "price": price, "change24": change24, "confidence": final,
            "votes": votes, "tp": round(price*1.05, 2), "sl": round(price*0.975, 2),
            "fg": fg, "news": news, "twitter": tw, "order_book": ob,
            "macro": mc, "fundas": fd, "is_crypto": is_crypto}

# ═══════════════════════════════════════════════════════
# MODULE 11 — FORMAT MESSAGE DISCORD
# ═══════════════════════════════════════════════════════

def format_message(s):
    cp   = int(s["confidence"]*100)
    bar  = "█"*(cp//10) + "░"*(10-cp//10)
    cem  = "📈" if s["change24"]>=0 else "📉"
    now  = datetime.now(pytz.timezone("Asia/Kuala_Lumpur")).strftime("%d/%m/%Y à %H:%M")
    nv   = sum(1 for v in s["votes"].values() if v>=0.5)

    L = [f"╔══════════════════════════════════════════╗",
         f"{s['color_emoji']}  SIGNAL {s['action']} — {s['symbol']}",
         f"╚══════════════════════════════════════════╝", "",
         f"💰 Prix actuel      : ${s['price']:,.2f}",
         f"{cem} Variation 24h    : {s['change24']:+.2f}%", "",
         f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
         f"🧠  CONSENSUS ML ({nv}/5 modèles)",
         f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
    for mn, prob in s["votes"].items():
        em = "🔵" if prob>=0.5 else "🔴"
        lb = "BUY  ✅" if prob>=0.5 else "HOLD ⏸️"
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
              "📰  ANALYSE NEWS (VADER+TextBlob)",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        for a in ns["articles"][:3]:
            em = "🟢" if a["sentiment"]=="positive" else ("🔴" if a["sentiment"]=="negative" else "🟡")
            L += [f"  {em} "{a['title']}"",
                  f"      → {a['sentiment'].capitalize()} ({a['score']:+.2f})"]
        avg_em = "🟢" if ns["avg_score"]>0.05 else ("🔴" if ns["avg_score"]<-0.05 else "🟡")
        L.append(f"  📰 Score global : {ns['avg_score']:+.2f} {avg_em}")

    if s["order_book"]:
        ob = s["order_book"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "📖  ORDER BOOK & LIQUIDATIONS",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        if ob["support"]:    L.append(f"  🟢 Support         : ${ob['support']:,.0f}")
        if ob["resistance"]: L.append(f"  🔴 Résistance      : ${ob['resistance']:,.0f}")
        if ob["liquidations_m"]>0: L.append(f"  ⚡ Liquidations    : ${ob['liquidations_m']}M")
        L.append(f"  📊 Buy/Sell        : {ob['buy_ratio']}% / {ob['sell_ratio']}%")

    if s["macro"]:
        mc = s["macro"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "🌍  MACRO-ÉCONOMIE",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              f"  📈 S&P 500  : {mc['spx_change']:+.2f}% {'🟢' if mc['spx_change']>=0 else '🔴'}",
              f"  📈 NASDAQ   : {mc['ndq_change']:+.2f}% {'🟢' if mc['ndq_change']>=0 else '🔴'}",
              f"  💵 DXY      : {mc['dxy_change']:+.2f}% {'🟢' if mc['dxy_change']<=0 else '🔴'}"]

    if s["fundas"]:
        fd = s["fundas"]
        L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
              "📊  ANALYSE FONDAMENTALE",
              "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        if fd["pe"]:         L.append(f"  💹 P/E Ratio  : {fd['pe']}")
        if fd["eps"]:        L.append(f"  💰 EPS        : ${fd['eps']}")
        if fd["rev_growth"]: L.append(f"  📈 Rev Growth : {fd['rev_growth']:+.1f}% {'🟢' if fd['rev_growth']>0 else '🔴'}")
        if fd["margin"]:     L.append(f"  🏆 Marge nette: {fd['margin']}%")
        if fd["sector"]:     L.append(f"  🏭 Secteur    : {fd['sector']}")
        if fd["next_earn"]:
            L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                  "📅  EARNINGS",
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                  f"  ⚠️  Prochain earnings : {fd['next_earn']}"]

    L += ["", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
          f"🎯  CONFIANCE GLOBALE : {cp}%  {bar}",
          "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", "",
          f"🎯 Take Profit : ${s['tp']:,.2f}  (+5.0%)",
          f"🛑 Stop Loss   : ${s['sl']:,.2f}  (-2.5%)",
          f"⚖️  Risk/Reward : 1 : 2.0"]

    warns = []
    if s["fundas"] and s["fundas"]["next_earn"]:
        warns.append(f"→ Earnings le {s['fundas']['next_earn']} — volatilité possible")
    if s["order_book"] and s["order_book"]["resistance"]:
        warns.append(f"→ Résistance forte à ${s['order_book']['resistance']:,.0f}")
    if s["fg"] and s["fg"]["value"] >= 75:
        warns.append("→ Extreme Greed — risque de correction")
    if s["macro"] and s["macro"]["dxy_change"] > 0.5:
        warns.append("→ Dollar en hausse — pression baissière")
    if warns:
        L += ["", "⚠️  POINTS DE VIGILANCE :"] + [f"  {w}" for w in warns]

    L += ["", f"⏱️  Analysé le : {now}",
          f"🔁  Prochain scan dans : {SCAN_INTERVAL_MINUTES} min"]

    return "```\n" + "\n".join(L) + "\n```"

# ═══════════════════════════════════════════════════════
# MODULE 12 — DISCORD BOT
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
            return
        df = add_features(df_raw)

        if symbol not in trained_models or datetime.now().weekday() == 0:
            logger.info(f"Entraînement {symbol}...")
            trained_models[symbol] = train_ensemble(df)

        confidence, votes = predict_ensemble(trained_models[symbol], df)

        if SIGNAL_THRESHOLD > confidence > (1 - SIGNAL_THRESHOLD):
            logger.info(f"{symbol}: HOLD ignoré ({confidence:.2f})")
            return

        signal  = build_signal(symbol, confidence, votes, df)
        message = format_message(signal)
        for chunk in [message[i:i+1900] for i in range(0, len(message), 1900)]:
            await channel.send(chunk)
        logger.info(f"✅ Signal {signal['action']} — {symbol} ({int(confidence*100)}%)")
    except Exception as e:
        logger.error(f"Erreur {symbol}: {e}")

@tasks.loop(minutes=SCAN_INTERVAL_MINUTES)
async def scheduled_scan():
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if not channel:
        return
    logger.info("=== Scan automatique ===")
    for symbol in WATCHLIST:
        await analyze_and_send(symbol, channel)
        await asyncio.sleep(5)

@bot.event
async def on_ready():
    logger.info(f"✅ Bot connecté : {bot.user}")
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    if channel:
        tw  = "✅" if TWITTER_API_KEY  else "⚠️ optionnel"
        bn  = "✅" if BINANCE_API_KEY  else "⚠️ optionnel"
        ns  = "✅" if NEWSAPI_KEY      else "❌ manquant"
        await channel.send(
            f"```\n🤖 Trading Bot v3.2 Lite démarré !\n"
            f"🧠 ML : Transformer + LSTM + RF + GB + SVM\n"
            f"📰 News NLP : VADER + TextBlob {ns}\n"
            f"🐦 Twitter : {tw} | 📖 Binance OB : {bn}\n"
            f"👁️  Watchlist : {len(WATCHLIST)} actifs\n"
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
    await ctx.send(f"```\n📋 WATCHLIST\n\n"
                   f"🪙 Crypto  : {chr(44).join(WATCHLIST_CRYPTO)}\n"
                   f"📈 Actions : {chr(44).join(WATCHLIST_STOCKS)}\n```")

@bot.command(name="status")
async def cmd_status(ctx):
    ns = datetime.now().minute % SCAN_INTERVAL_MINUTES
    next_s = SCAN_INTERVAL_MINUTES - ns
    await ctx.send(f"```\n🤖 TRADING BOT v3.2 Lite — STATUT\n\n"
                   f"🧠 ML Ensemble  : ✅ (5 modèles)\n"
                   f"📰 News VADER   : {'✅' if NEWSAPI_KEY else '❌'}\n"
                   f"🐦 Twitter      : {'✅' if TWITTER_API_KEY else '⚠️ optionnel'}\n"
                   f"📖 Binance OB   : {'✅' if BINANCE_API_KEY else '⚠️ optionnel'}\n"
                   f"🌍 Macro        : ✅\n"
                   f"🔁 Prochain scan: dans ~{next_s} min\n```")

@bot.command(name="aide")
async def cmd_aide(ctx):
    await ctx.send("```\n📖 COMMANDES\n\n"
                   "!analyse [SYMBOL] → Signal immédiat (ex: !analyse NVDA)\n"
                   "!top5             → Analyse les 5 premiers actifs\n"
                   "!watchlist        → Liste des actifs surveillés\n"
                   "!status           → Statut du bot\n"
                   "!aide             → Cette aide\n```")

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN manquant !")
        exit(1)
    if not DISCORD_CHANNEL_ID:
        logger.error("DISCORD_CHANNEL_ID manquant !")
        exit(1)
    logger.info("Démarrage Trading Bot v3.2 Lite...")
    bot.run(DISCORD_TOKEN)
