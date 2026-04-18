import os

# ── Discord ────────────────────────────────────────────────
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "")
CHANNEL_ID    = int(os.environ.get("CHANNEL_ID", "0"))

# ── Watchlist ──────────────────────────────────────────────
WATCHLIST_STOCKS = [
    "AAPL","NVDA","MSFT","TSLA","AMZN","META","GOOGL","JPM","NFLX","AMD",
    "MC.PA","TTE.PA","AI.PA","SAN.PA","BNP.PA",
]
WATCHLIST_CRYPTO = [
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","DOGE-USD",
    "AVAX-USD","DOT-USD","LINK-USD","MATIC-USD","LTC-USD","UNI-USD",
    "AAVE-USD","SHIB-USD","TRX-USD","XLM-USD","ATOM-USD",
    "MIOTA-USD","XTZ-USD","SAND-USD","MANA-USD","THETA-USD",
]

# ── Paramètres ─────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 60
MIN_CONFIDENCE        = 65
LOOKBACK_DAYS         = "6mo"
DAILY_REPORT_HOUR     = 8   # heure du rapport quotidien (UTC)

# ── Timeframes ─────────────────────────────────────────────
TIMEFRAMES = {
    "1h":  {"period": "60d",  "interval": "1h"},
    "4h":  {"period": "60d",  "interval": "90m"},  # proxy 4h
    "1d":  {"period": "6mo",  "interval": "1d"},
    "1wk": {"period": "2y",   "interval": "1wk"},
}
