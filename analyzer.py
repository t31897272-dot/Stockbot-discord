# ══════════════════════════════════════════════════════════
#  analyzer.py — Analyse multi-timeframe + tous indicateurs
# ══════════════════════════════════════════════════════════
import yfinance as yf
import pandas as pd
import numpy as np
from indicators import (
    calc_rsi, calc_stoch_rsi, calc_macd, calc_bollinger,
    calc_vwap, calc_sma, calc_ema, calc_atr,
    calc_fibonacci, calc_ichimoku, calc_volume_signal, near_fibonacci
)
from patterns import detect_patterns
from config import LOOKBACK_DAYS, TIMEFRAMES


def fetch_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty or len(df) < 20:
        raise ValueError(f"Données insuffisantes pour {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.dropna()


def score_single(rsi, stoch_k, macd_val, macd_hist, pct_b,
                 sma20, sma50, price, patterns, vol, ichimoku, fib_near):
    score = 0

    # RSI
    if rsi < 28:   score += 3
    elif rsi < 38: score += 2
    elif rsi < 45: score += 1
    elif rsi > 72: score -= 3
    elif rsi > 65: score -= 2
    elif rsi > 60: score -= 1

    # Stoch RSI
    if stoch_k < 20:   score += 2
    elif stoch_k < 30: score += 1
    elif stoch_k > 80: score -= 2
    elif stoch_k > 70: score -= 1

    # MACD
    if macd_val > 0 and macd_hist > 0:   score += 2
    elif macd_val > 0:                    score += 1
    elif macd_val < 0 and macd_hist < 0: score -= 2
    elif macd_val < 0:                   score -= 1

    # Bollinger
    if pct_b < 0.10:   score += 3
    elif pct_b < 0.20: score += 2
    elif pct_b > 0.90: score -= 3
    elif pct_b > 0.80: score -= 2

    # Tendance SMA
    if sma20 > sma50 and price > sma20:  score += 2
    elif sma20 < sma50 and price < sma20: score -= 2

    # Ichimoku
    if ichimoku["above_cloud"]:       score += 2
    if ichimoku["below_cloud"]:       score -= 2
    if ichimoku["tk_cross_bull"]:     score += 2
    if ichimoku["tk_cross_bear"]:     score -= 2

    # Fibonacci
    if fib_near in ("61.8%", "50.0%", "38.2%"): score += 1

    # Patterns
    bullish = [p for p in patterns if "🟢" in p]
    bearish = [p for p in patterns if "🔴" in p]
    score += len(bullish) * 3
    score -= len(bearish) * 3

    # Volume
    if vol == "fort":
        score += 1 if score > 0 else -1

    return score


def analyze_ticker(ticker: str) -> dict:
    # Données journalières principales
    df = fetch_data(ticker, period=LOOKBACK_DAYS, interval="1d")
    closes = df["Close"]
    price  = float(closes.iloc[-1])
    prev   = float(closes.iloc[-2])
    chg    = round((price - prev) / prev * 100, 2)

    # Indicateurs
    rsi                     = calc_rsi(closes)
    stoch_k, stoch_d        = calc_stoch_rsi(closes)
    macd_val, macd_sig, mhist = calc_macd(closes)
    pct_b, bb_up, bb_low    = calc_bollinger(closes)
    vwap                    = calc_vwap(df)
    sma20                   = calc_sma(closes, 20)
    sma50                   = calc_sma(closes, 50)
    ema9                    = calc_ema(closes, 9)
    atr                     = calc_atr(df)
    fib                     = calc_fibonacci(df)
    fib_near                = near_fibonacci(price, fib)
    ichi                    = calc_ichimoku(df)
    vol                     = calc_volume_signal(df)
    patterns                = detect_patterns(df)

    # Score timeframe journalier
    score_1d = score_single(rsi, stoch_k, macd_val, mhist, pct_b,
                             sma20, sma50, price, patterns, vol, ichi, fib_near)

    # Multi-timeframe : confirmation 1h
    mtf_bias = 0
    try:
        df_1h = fetch_data(ticker, period="60d", interval="1h")
        if len(df_1h) >= 20:
            rsi_1h = calc_rsi(df_1h["Close"])
            macd_1h, _, mhist_1h = calc_macd(df_1h["Close"])
            if rsi_1h < 40 and macd_1h > 0:  mtf_bias += 2
            elif rsi_1h > 60 and macd_1h < 0: mtf_bias -= 2
            else: mtf_bias += 1 if macd_1h > 0 else -1
    except Exception:
        pass

    # Multi-timeframe : confirmation hebdo
    try:
        df_wk = fetch_data(ticker, period="2y", interval="1wk")
        if len(df_wk) >= 20:
            rsi_wk = calc_rsi(df_wk["Close"])
            macd_wk, _, _ = calc_macd(df_wk["Close"])
            if rsi_wk < 45 and macd_wk > 0:  mtf_bias += 3
            elif rsi_wk > 60 and macd_wk < 0: mtf_bias -= 3
    except Exception:
        pass

    total_score = score_1d + mtf_bias
    max_score   = 22
    normalized  = total_score / max_score
    confidence  = int(min(98, max(30, 50 + normalized * 50)))

    if total_score >= 5:   signal = "BUY"
    elif total_score <= -5: signal = "SELL"
    else:                   signal = "HOLD"

    # Zones de trading
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

    # Raison principale
    reasons = []
    if rsi < 35:    reasons.append(f"RSI survendu ({rsi})")
    elif rsi > 65:  reasons.append(f"RSI surachat ({rsi})")
    if macd_val > 0 and mhist > 0: reasons.append("MACD haussier")
    elif macd_val < 0 and mhist < 0: reasons.append("MACD baissier")
    if ichi["above_cloud"]: reasons.append("Prix au-dessus du nuage Ichimoku")
    if ichi["below_cloud"]: reasons.append("Prix sous le nuage Ichimoku")
    if fib_near: reasons.append(f"Fibonacci {fib_near}")
    for p in patterns[:2]:
        reasons.append(p.split("—")[0].replace("🟢","").replace("🔴","").strip())
    if mtf_bias >= 2:  reasons.append("Confirmation multi-timeframe ✅")
    elif mtf_bias <= -2: reasons.append("Divergence multi-timeframe ⚠️")

    reason = " · ".join(reasons[:3]) if reasons else "Pas de signal fort"

    return {
        "ticker": ticker, "price": price, "change_pct": chg,
        "signal": signal, "confidence": confidence, "reason": reason,
        "rsi": rsi, "stoch_k": stoch_k, "stoch_d": stoch_d,
        "macd": macd_val, "macd_hist": mhist,
        "pct_b": pct_b, "bb_upper": bb_up, "bb_lower": bb_low,
        "vwap": vwap, "sma20": sma20, "sma50": sma50, "ema9": ema9,
        "atr": atr, "fibonacci": fib, "fib_near": fib_near,
        "ichimoku": ichi, "volume": vol, "patterns": patterns,
        "score_1d": score_1d, "mtf_bias": mtf_bias,
        "entry": entry, "target": target, "stop": stop, "risk_reward": rr,
    }
