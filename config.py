# ══════════════════════════════════════════════════════════
#  config.py — Configuration du StockBot Discord
#  ➡️  MODIFIE UNIQUEMENT CE FICHIER avant de lancer le bot
# ══════════════════════════════════════════════════════════

# ── 1. Token Discord du bot (obtenu sur discord.com/developers)
DISCORD_TOKEN = "METS_TON_TOKEN_ICI"

# ── 2. ID du canal Discord où envoyer les alertes
#   (clic droit sur le canal → "Copier l'ID" — mode développeur requis)
CHANNEL_ID = 123456789012345678  # ← remplace par ton vrai ID (nombre entier)

# ── 3. Watchlist Actions (tickers Yahoo Finance)
WATCHLIST_STOCKS = [
    "AAPL",   # Apple
    "NVDA",   # NVIDIA
    "MSFT",   # Microsoft
    "TSLA",   # Tesla
    "AMZN",   # Amazon
    "META",   # Meta
    "GOOGL",  # Alphabet
    "JPM",    # JPMorgan
    "NFLX",   # Netflix
    "AMD",    # AMD
    # ── Actions françaises / européennes ──
    "MC.PA",  # LVMH
    "TTE.PA", # TotalEnergies
    "AI.PA",  # Air Liquide
    "SAN.PA", # Sanofi
    "BNP.PA", # BNP Paribas
]

# ── 4. Watchlist Cryptos (format Yahoo Finance : SYMBOL-USD)
WATCHLIST_CRYPTO = [
    "BTC-USD",   # Bitcoin
    "ETH-USD",   # Ethereum
    "BNB-USD",   # Binance Coin
    "SOL-USD",   # Solana
    "XRP-USD",   # XRP
    "ADA-USD",   # Cardano
    "AVAX-USD",  # Avalanche
    "DOGE-USD",  # Dogecoin
    "DOT-USD",   # Polkadot
    "MATIC-USD", # Polygon
]

# ── 5. Intervalle de scan automatique (en minutes)
#   Plan gratuit : recommandé 60 min (évite les limitations Yahoo Finance)
#   Si tu veux plus fréquent : minimum 15 min
SCAN_INTERVAL_MINUTES = 60

# ── 6. Confiance minimale pour envoyer une alerte (0-100)
#   70 = seulement les signaux solides (recommandé)
#   60 = plus d'alertes mais moins fiables
MIN_CONFIDENCE = 70

# ── 7. Période d'analyse (jours de données à télécharger)
LOOKBACK_DAYS = "6mo"  # 6 mois de données OHLCV
