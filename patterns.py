# ══════════════════════════════════════════════════════════
#  patterns.py — Détection de patterns graphiques classiques
#  Inspiré de l'analyse technique chartiste (Murphy, Elder)
# ══════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
from typing import List


def find_local_minima(series: pd.Series, order: int = 5) -> List[int]:
    """Indices des minima locaux (creux)."""
    minima = []
    arr = series.values
    for i in range(order, len(arr) - order):
        if arr[i] == min(arr[i - order:i + order + 1]):
            minima.append(i)
    return minima


def find_local_maxima(series: pd.Series, order: int = 5) -> List[int]:
    """Indices des maxima locaux (sommets)."""
    maxima = []
    arr = series.values
    for i in range(order, len(arr) - order):
        if arr[i] == max(arr[i - order:i + order + 1]):
            maxima.append(i)
    return maxima


def detect_double_bottom(df: pd.DataFrame, tolerance: float = 0.03) -> bool:
    """
    Double Bottom (W) : deux creux à niveau similaire séparés par un sommet.
    Signal HAUSSIER puissant.
    """
    lows = df["Low"]
    minima = find_local_minima(lows, order=5)
    if len(minima) < 2:
        return False
    for i in range(len(minima) - 1):
        idx1, idx2 = minima[i], minima[i + 1]
        if idx2 - idx1 < 10:  # trop proches
            continue
        v1, v2 = float(lows.iloc[idx1]), float(lows.iloc[idx2])
        if abs(v1 - v2) / max(v1, v2) <= tolerance:
            # Vérifier qu'il y a bien un sommet entre les deux creux
            between_max = float(df["High"].iloc[idx1:idx2].max())
            if between_max > max(v1, v2) * 1.02:
                return True
    return False


def detect_double_top(df: pd.DataFrame, tolerance: float = 0.03) -> bool:
    """
    Double Top (M) : deux sommets à niveau similaire séparés par un creux.
    Signal BAISSIER puissant.
    """
    highs = df["High"]
    maxima = find_local_maxima(highs, order=5)
    if len(maxima) < 2:
        return False
    for i in range(len(maxima) - 1):
        idx1, idx2 = maxima[i], maxima[i + 1]
        if idx2 - idx1 < 10:
            continue
        v1, v2 = float(highs.iloc[idx1]), float(highs.iloc[idx2])
        if abs(v1 - v2) / max(v1, v2) <= tolerance:
            between_min = float(df["Low"].iloc[idx1:idx2].min())
            if between_min < min(v1, v2) * 0.98:
                return True
    return False


def detect_triple_bottom(df: pd.DataFrame, tolerance: float = 0.04) -> bool:
    """Triple Bottom : trois creux quasi-identiques. Signal TRÈS HAUSSIER."""
    lows = df["Low"]
    minima = find_local_minima(lows, order=5)
    if len(minima) < 3:
        return False
    for i in range(len(minima) - 2):
        idx1, idx2, idx3 = minima[i], minima[i + 1], minima[i + 2]
        if idx3 - idx1 < 20:
            continue
        v1 = float(lows.iloc[idx1])
        v2 = float(lows.iloc[idx2])
        v3 = float(lows.iloc[idx3])
        avg = (v1 + v2 + v3) / 3
        if max(abs(v1-avg), abs(v2-avg), abs(v3-avg)) / avg <= tolerance:
            return True
    return False


def detect_triple_top(df: pd.DataFrame, tolerance: float = 0.04) -> bool:
    """Triple Top : trois sommets quasi-identiques. Signal TRÈS BAISSIER."""
    highs = df["High"]
    maxima = find_local_maxima(highs, order=5)
    if len(maxima) < 3:
        return False
    for i in range(len(maxima) - 2):
        idx1, idx2, idx3 = maxima[i], maxima[i + 1], maxima[i + 2]
        if idx3 - idx1 < 20:
            continue
        v1 = float(highs.iloc[idx1])
        v2 = float(highs.iloc[idx2])
        v3 = float(highs.iloc[idx3])
        avg = (v1 + v2 + v3) / 3
        if max(abs(v1-avg), abs(v2-avg), abs(v3-avg)) / avg <= tolerance:
            return True
    return False


def detect_head_and_shoulders(df: pd.DataFrame) -> bool:
    """
    Tête & Épaules (baissier) : épaule gauche, tête (sommet) + épaule droite.
    Signal BAISSIER fort.
    """
    highs = df["High"]
    maxima = find_local_maxima(highs, order=5)
    if len(maxima) < 3:
        return False
    for i in range(len(maxima) - 2):
        ls = float(highs.iloc[maxima[i]])      # épaule gauche
        head = float(highs.iloc[maxima[i+1]])  # tête
        rs = float(highs.iloc[maxima[i+2]])    # épaule droite
        if head > ls * 1.03 and head > rs * 1.03:
            if abs(ls - rs) / ls < 0.05:
                return True
    return False


def detect_inverse_head_and_shoulders(df: pd.DataFrame) -> bool:
    """Tête & Épaules inversé (haussier)."""
    lows = df["Low"]
    minima = find_local_minima(lows, order=5)
    if len(minima) < 3:
        return False
    for i in range(len(minima) - 2):
        ls   = float(lows.iloc[minima[i]])
        head = float(lows.iloc[minima[i+1]])
        rs   = float(lows.iloc[minima[i+2]])
        if head < ls * 0.97 and head < rs * 0.97:
            if abs(ls - rs) / ls < 0.05:
                return True
    return False


def detect_cup_and_handle(df: pd.DataFrame) -> bool:
    """
    Cup & Handle (haussier) : forme en U avec petite consolidation.
    """
    closes = df["Close"].values
    if len(closes) < 40:
        return False
    n = len(closes)
    # Cherche un creux central dans les 2/3 du graphique
    mid_section = closes[n//4 : 3*n//4]
    left_high   = max(closes[:n//4])
    right_high  = max(closes[3*n//4:])
    cup_low     = min(mid_section)
    if left_high > cup_low * 1.08 and right_high > cup_low * 1.08:
        if abs(left_high - right_high) / left_high < 0.06:
            return True
    return False


def detect_bullish_flag(df: pd.DataFrame) -> bool:
    """
    Flag haussier : forte hausse suivie d'une légère consolidation.
    """
    closes = df["Close"]
    if len(closes) < 20:
        return False
    # Forte hausse sur les 10 premières bougies de la fenêtre
    first_half = closes.iloc[-20:-10]
    second_half = closes.iloc[-10:]
    gain_first = (first_half.iloc[-1] - first_half.iloc[0]) / first_half.iloc[0]
    # Consolidation légèrement baissière ou latérale
    move_second = (second_half.iloc[-1] - second_half.iloc[0]) / second_half.iloc[0]
    return gain_first > 0.05 and -0.04 < move_second < 0.01


def detect_bearish_flag(df: pd.DataFrame) -> bool:
    """Flag baissier : forte baisse suivie d'une légère consolidation."""
    closes = df["Close"]
    if len(closes) < 20:
        return False
    first_half  = closes.iloc[-20:-10]
    second_half = closes.iloc[-10:]
    drop_first  = (first_half.iloc[-1] - first_half.iloc[0]) / first_half.iloc[0]
    move_second = (second_half.iloc[-1] - second_half.iloc[0]) / second_half.iloc[0]
    return drop_first < -0.05 and -0.01 < move_second < 0.04


def detect_hammer(df: pd.DataFrame) -> bool:
    """
    Marteau (bougie haussière) : petite bougie avec longue mèche inférieure.
    """
    last = df.iloc[-1]
    body   = abs(float(last["Close"]) - float(last["Open"]))
    wick_l = float(last["Open"])  - float(last["Low"]) if last["Close"] > last["Open"] else float(last["Close"]) - float(last["Low"])
    wick_u = float(last["High"])  - max(float(last["Close"]), float(last["Open"]))
    total  = float(last["High"])  - float(last["Low"])
    if total == 0:
        return False
    return wick_l > 2 * body and wick_u < body * 0.5 and body / total < 0.35


def detect_shooting_star(df: pd.DataFrame) -> bool:
    """
    Étoile filante (bougie baissière) : petite bougie avec longue mèche supérieure.
    """
    last = df.iloc[-1]
    body   = abs(float(last["Close"]) - float(last["Open"]))
    wick_u = float(last["High"]) - max(float(last["Close"]), float(last["Open"]))
    wick_l = min(float(last["Close"]), float(last["Open"])) - float(last["Low"])
    total  = float(last["High"]) - float(last["Low"])
    if total == 0:
        return False
    return wick_u > 2 * body and wick_l < body * 0.5 and body / total < 0.35


def detect_engulfing(df: pd.DataFrame) -> tuple:
    """
    Englobante haussière ou baissière (2 bougies).
    Retourne (bullish: bool, bearish: bool)
    """
    if len(df) < 2:
        return False, False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    p_open, p_close = float(prev["Open"]), float(prev["Close"])
    c_open, c_close = float(curr["Open"]), float(curr["Close"])
    # Englobante haussière
    bullish = p_close < p_open and c_close > c_open and c_open < p_close and c_close > p_open
    # Englobante baissière
    bearish = p_close > p_open and c_close < c_open and c_open > p_close and c_close < p_open
    return bullish, bearish


def detect_support_resistance_breakout(df: pd.DataFrame) -> dict:
    """
    Détecte une cassure de support ou résistance.
    Retourne {"breakout_up": bool, "breakout_down": bool}
    """
    closes = df["Close"]
    highs  = df["High"]
    lows   = df["Low"]
    # Résistance = max des 20 dernières bougies (sauf la dernière)
    resistance = float(highs.iloc[-21:-1].max())
    support    = float(lows.iloc[-21:-1].min())
    last_close = float(closes.iloc[-1])
    breakout_up   = last_close > resistance * 1.005
    breakout_down = last_close < support * 0.995
    return {"breakout_up": breakout_up, "breakout_down": breakout_down, "resistance": resistance, "support": support}


# ── Main detection function ─────────────────────────────────────────────────
def detect_patterns(df: pd.DataFrame) -> List[str]:
    """
    Analyse tous les patterns et retourne une liste de strings descriptifs.
    Les patterns haussiers sont préfixés 🟢, baissiers 🔴.
    """
    found = []

    try:
        if detect_double_bottom(df):
            found.append("🟢 Double Bottom (W) — rebond probable")
        if detect_triple_bottom(df):
            found.append("🟢 Triple Bottom — support très fort")
        if detect_inverse_head_and_shoulders(df):
            found.append("🟢 Tête & Épaules Inversé — renversement haussier")
        if detect_cup_and_handle(df):
            found.append("🟢 Cup & Handle — continuation haussière")
        if detect_bullish_flag(df):
            found.append("🟢 Flag Haussier — continuation à la hausse")
        if detect_hammer(df):
            found.append("🟢 Marteau — bougie de renversement haussier")

        bull_eng, bear_eng = detect_engulfing(df)
        if bull_eng:
            found.append("🟢 Englobante Haussière — signal d'achat")

        if detect_double_top(df):
            found.append("🔴 Double Top (M) — retournement baissier")
        if detect_triple_top(df):
            found.append("🔴 Triple Top — résistance très forte")
        if detect_head_and_shoulders(df):
            found.append("🔴 Tête & Épaules — renversement baissier")
        if detect_bearish_flag(df):
            found.append("🔴 Flag Baissier — continuation à la baisse")
        if detect_shooting_star(df):
            found.append("🔴 Étoile Filante — bougie de renversement baissier")
        if bear_eng:
            found.append("🔴 Englobante Baissière — signal de vente")

        br = detect_support_resistance_breakout(df)
        if br["breakout_up"]:
            found.append(f"🟢 Cassure de résistance (${br['resistance']:.2f})")
        if br["breakout_down"]:
            found.append(f"🔴 Cassure de support (${br['support']:.2f})")

    except Exception as e:
        pass  # Silencieux pour ne pas bloquer l'analyse

    return found
