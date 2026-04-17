# ══════════════════════════════════════════════════════════
#  analyzer.py — Calcul des indicateurs techniques
# ══════════════════════════════════════════════════════════
import yfinance as yf
import numpy as np
import pandas as pd
from patterns import detect_patterns
from config import LOOKBACK_DAYS


def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=LOOKBACK_DAYS, interval="1d", progress=False, auto_adjust=True)
    if df.empty or len(df) < 30:
        raise ValueError(f"Données insuffisantes pour {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna()
    return df


def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def calc_macd(closes: pd.Series):
    ema12  = closes.ewm(span=12, adjust=False).mean()
    ema26  = closes.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return round(float(macd.iloc[-1]), 4), round(float(signal.iloc[-1]), 4), round(float(hist.iloc[-1]), 4)


def calc_bollinger(closes: pd.Series, period: int = 20):
    sma   = closes.rolling(period).mean()
    std   = closes.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    last  = float(closes.iloc[-1])
    u     = float(upper.iloc[-1])
    l     = float(lower.iloc[-1])
    pct_b = (last - l) / (u - l) if (u - l) != 0 else 0.5
    return round(pct_b, 3), round(u, 4), round(l, 4)


def calc_sma(closes: pd.Series, period: int) -> float:
    if len(closes) < period:
        return float(closes.mean())
    return round(float(closes.rolling(period).mean().iloc[-1]), 4)


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low  = df["Low"]
    close_prev = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low  - close_prev).abs()
    ], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]), 4)


def calc_volume_signal(df: pd.DataFrame) -> str:
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
    last_vol = df["Volume"].iloc[-1]
    ratio = last_vol / avg_vol if avg_vol > 0 else 1
    if ratio > 1.5:
        return "fort"
    elif ratio < 0.5:
        return "faible"
    return "normal"


def compute_signal(rsi, macd_val, macd_hist, pct_b, sma20, sma50, price, patterns, vol_signal):
    """
    Score multi-facteurs :
    - Indicateurs techniques classiques
    - Patterns graphiques détectés
    - Volume
    → Retourne (signal, confidence, reason)
    """
    score = 0
    reasons = []

    # RSI
    if rsi < 28:
        score += 3; reasons.append(f"RSI très survendu ({rsi})")
    elif rsi < 38:
        score += 2; reasons.append(f"RSI survendu ({rsi})")
    elif rsi < 45:
        score += 1; reasons.append(f"RSI bas ({rsi})")
    elif rsi > 72:
        score -= 3; reasons.append(f"RSI très surachat ({rsi})")
    elif rsi > 65:
        score -= 2; reasons.append(f"RSI surachat ({rsi})")
    elif rsi > 60:
        score -= 1; reasons.append(f"RSI élevé ({rsi})")

    # MACD
    if macd_val > 0 and macd_hist > 0:
        score += 2; reasons.append("MACD haussier + histogramme positif")
    elif macd_val > 0:
        score += 1; reasons.append("MACD positif")
    elif macd_val < 0 and macd_hist < 0:
        score -= 2; reasons.append("MACD baissier + histogramme négatif")
    elif macd_val < 0:
        score -= 1; reasons.append("MACD négatif")

    # Bollinger
    if pct_b < 0.10:
        score += 3; reasons.append("Prix sous bande basse (survente extrême)")
    elif pct_b < 0.20:
        score += 2; reasons.append("Prix proche bande basse")
    elif pct_b > 0.90:
        score -= 3; reasons.append("Prix au-dessus bande haute (achat excessif)")
    elif pct_b > 0.80:
        score -= 2; reasons.append("Prix proche bande haute")

    # Tendance SMA
    if sma20 > sma50 and price > sma20:
        score += 2; reasons.append("Tendance haussière (SMA20 > SMA50)")
    elif sma20 < sma50 and price < sma20:
        score -= 2; reasons.append("Tendance baissière (SMA20 < SMA50)")

    # Patterns graphiques (chaque pattern pèse fort)
    bullish_patterns = [p for p in patterns if any(k in p.lower() for k in
        ["double bottom", "triple bottom", "tête épaules inversé", "cup", "bullish", "marteau", "morning"])]
    bearish_patterns = [p for p in patterns if any(k in p.lower() for k in
        ["double top", "triple top", "tête épaules", "shooting", "bearish", "evening", "pendu"])]
    score += len(bullish_patterns) * 3
    score -= len(bearish_patterns) * 3

    # Volume
    if vol_signal == "fort" and score > 0:
        score += 1; reasons.append("Volume élevé confirme le signal")
    elif vol_signal == "fort" and score < 0:
        score -= 1; reasons.append("Volume élevé confirme la pression vendeuse")

    # Score → signal
    max_score = 14
    normalized = score / max_score  # -1 à +1
    confidence = int(min(98, max(30, 50 + normalized * 50)))

    if score >= 4:
        signal = "BUY"
    elif score <= -4:
        signal = "SELL"
    else:
        signal = "HOLD"

    reason = " + ".join(reasons[:3]) if reasons else "Pas de signal fort détecté"
    return signal, confidence, reason


def analyze_ticker(ticker: str) -> dict:
    df = fetch_data(ticker)
    closes = df["Close"]
    price  = float(closes.iloc[-1])
    prev   = float(closes.iloc[-2])
    chg    = round((price - prev) / prev * 100, 2)

    rsi               = calc_rsi(closes)
    macd_val, macd_sig, macd_hist = calc_macd(closes)
    pct_b, bb_upper, bb_lower     = calc_bollinger(closes)
    sma20             = calc_sma(closes, 20)
    sma50             = calc_sma(closes, 50)
    atr               = calc_atr(df)
    vol_signal        = calc_volume_signal(df)
    patterns          = detect_patterns(df)

    signal, confidence, reason = compute_signal(
        rsi, macd_val, macd_hist, pct_b,
        sma20, sma50, price, patterns, vol_signal
    )

    # Zones de trading basées sur ATR
    if signal == "BUY":
        entry  = round(price, 2)
        target = round(price + atr * 3.0, 2)
        stop   = round(price - atr * 1.5, 2)
    elif signal == "SELL":
        entry  = round(price, 2)
        target = round(price - atr * 3.0, 2)
        stop   = round(price + atr * 1.5, 2)
    else:
        entry  = round(price, 2)
        target = round(price + atr * 2.0, 2)
        stop   = round(price - atr * 1.5, 2)

    risk   = abs(price - stop)
    reward = abs(target - price)
    rr     = round(reward / risk, 2) if risk > 0 else 0

    return {
        "ticker":      ticker,
        "price":       price,
        "change_pct":  chg,
        "signal":      signal,
        "confidence":  confidence,
        "reason":      reason,
        "rsi":         rsi,
        "macd":        macd_val,
        "macd_hist":   macd_hist,
        "pct_b":       pct_b,
        "sma20":       sma20,
        "sma50":       sma50,
        "atr":         atr,
        "volume":      vol_signal,
        "patterns":    patterns,
        "entry":       entry,
        "target":      target,
        "stop":        stop,
        "risk_reward": rr,
    }
