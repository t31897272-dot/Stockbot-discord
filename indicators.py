# ══════════════════════════════════════════════════════════
#  indicators.py — Tous les indicateurs techniques
# ══════════════════════════════════════════════════════════
import pandas as pd
import numpy as np


def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def calc_stoch_rsi(closes: pd.Series, period: int = 14, smooth: int = 3):
    rsi_series = pd.Series(dtype=float)
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    min_rsi = rsi_series.rolling(period).min()
    max_rsi = rsi_series.rolling(period).max()
    stoch_rsi = (rsi_series - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    k = stoch_rsi.rolling(smooth).mean() * 100
    d = k.rolling(smooth).mean()
    return round(float(k.iloc[-1]), 2), round(float(d.iloc[-1]), 2)


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
    u, l  = float(upper.iloc[-1]), float(lower.iloc[-1])
    pct_b = (last - l) / (u - l) if (u - l) != 0 else 0.5
    return round(pct_b, 3), round(u, 4), round(l, 4)


def calc_vwap(df: pd.DataFrame) -> float:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return round(float(vwap.iloc[-1]), 4)


def calc_sma(closes: pd.Series, period: int) -> float:
    if len(closes) < period:
        return float(closes.mean())
    return round(float(closes.rolling(period).mean().iloc[-1]), 4)


def calc_ema(closes: pd.Series, period: int) -> float:
    return round(float(closes.ewm(span=period, adjust=False).mean().iloc[-1]), 4)


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low = df["High"], df["Low"]
    close_prev = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low  - close_prev).abs()
    ], axis=1).max(axis=1)
    return round(float(tr.rolling(period).mean().iloc[-1]), 4)


def calc_fibonacci(df: pd.DataFrame) -> dict:
    high = float(df["High"].max())
    low  = float(df["Low"].min())
    diff = high - low
    levels = {
        "0.0%":   round(high, 4),
        "23.6%":  round(high - 0.236 * diff, 4),
        "38.2%":  round(high - 0.382 * diff, 4),
        "50.0%":  round(high - 0.500 * diff, 4),
        "61.8%":  round(high - 0.618 * diff, 4),
        "78.6%":  round(high - 0.786 * diff, 4),
        "100%":   round(low, 4),
    }
    return levels


def calc_ichimoku(df: pd.DataFrame) -> dict:
    high, low, close = df["High"], df["Low"], df["Close"]
    # Tenkan-sen (9)
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    # Kijun-sen (26)
    kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2
    # Senkou Span A
    span_a = ((tenkan + kijun) / 2).shift(26)
    # Senkou Span B
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    price  = float(close.iloc[-1])
    sa     = float(span_a.iloc[-1]) if not pd.isna(span_a.iloc[-1]) else 0
    sb     = float(span_b.iloc[-1]) if not pd.isna(span_b.iloc[-1]) else 0
    above_cloud = price > max(sa, sb)
    below_cloud = price < min(sa, sb)
    tk_cross_bull = float(tenkan.iloc[-1]) > float(kijun.iloc[-1]) and \
                    float(tenkan.iloc[-2]) <= float(kijun.iloc[-2])
    tk_cross_bear = float(tenkan.iloc[-1]) < float(kijun.iloc[-1]) and \
                    float(tenkan.iloc[-2]) >= float(kijun.iloc[-2])
    return {
        "above_cloud":    above_cloud,
        "below_cloud":    below_cloud,
        "tk_cross_bull":  tk_cross_bull,
        "tk_cross_bear":  tk_cross_bear,
        "tenkan":         round(float(tenkan.iloc[-1]), 4),
        "kijun":          round(float(kijun.iloc[-1]), 4),
        "span_a":         round(sa, 4),
        "span_b":         round(sb, 4),
    }


def calc_volume_signal(df: pd.DataFrame) -> str:
    avg_vol  = df["Volume"].rolling(20).mean().iloc[-1]
    last_vol = df["Volume"].iloc[-1]
    ratio    = last_vol / avg_vol if avg_vol > 0 else 1
    if ratio > 1.5:   return "fort"
    elif ratio < 0.5: return "faible"
    return "normal"


def near_fibonacci(price: float, fib_levels: dict, tolerance: float = 0.01) -> str:
    for label, level in fib_levels.items():
        if abs(price - level) / level <= tolerance:
            return label
    return ""
