# ══════════════════════════════════════════════════════════
#  patterns.py — Tous les patterns graphiques + chandeliers
# ══════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
from typing import List


def _minima(series, order=5):
    arr = series.values
    return [i for i in range(order, len(arr)-order)
            if arr[i] == min(arr[i-order:i+order+1])]

def _maxima(series, order=5):
    arr = series.values
    return [i for i in range(order, len(arr)-order)
            if arr[i] == max(arr[i-order:i+order+1])]


# ── Patterns chartistes ────────────────────────────────────

def double_bottom(df, tol=0.03):
    lows, mins = df["Low"], _minima(df["Low"])
    for i in range(len(mins)-1):
        v1, v2 = float(lows.iloc[mins[i]]), float(lows.iloc[mins[i+1]])
        if mins[i+1]-mins[i] >= 10 and abs(v1-v2)/max(v1,v2) <= tol:
            if float(df["High"].iloc[mins[i]:mins[i+1]].max()) > max(v1,v2)*1.02:
                return True
    return False

def double_top(df, tol=0.03):
    highs, maxs = df["High"], _maxima(df["High"])
    for i in range(len(maxs)-1):
        v1, v2 = float(highs.iloc[maxs[i]]), float(highs.iloc[maxs[i+1]])
        if maxs[i+1]-maxs[i] >= 10 and abs(v1-v2)/max(v1,v2) <= tol:
            if float(df["Low"].iloc[maxs[i]:maxs[i+1]].min()) < min(v1,v2)*0.98:
                return True
    return False

def triple_bottom(df, tol=0.04):
    lows, mins = df["Low"], _minima(df["Low"])
    for i in range(len(mins)-2):
        v = [float(lows.iloc[mins[i+j]]) for j in range(3)]
        avg = sum(v)/3
        if mins[i+2]-mins[i] >= 20 and max(abs(x-avg) for x in v)/avg <= tol:
            return True
    return False

def triple_top(df, tol=0.04):
    highs, maxs = df["High"], _maxima(df["High"])
    for i in range(len(maxs)-2):
        v = [float(highs.iloc[maxs[i+j]]) for j in range(3)]
        avg = sum(v)/3
        if maxs[i+2]-maxs[i] >= 20 and max(abs(x-avg) for x in v)/avg <= tol:
            return True
    return False

def head_and_shoulders(df):
    highs, maxs = df["High"], _maxima(df["High"])
    for i in range(len(maxs)-2):
        ls = float(highs.iloc[maxs[i]])
        h  = float(highs.iloc[maxs[i+1]])
        rs = float(highs.iloc[maxs[i+2]])
        if h > ls*1.03 and h > rs*1.03 and abs(ls-rs)/ls < 0.05:
            return True
    return False

def inv_head_and_shoulders(df):
    lows, mins = df["Low"], _minima(df["Low"])
    for i in range(len(mins)-2):
        ls = float(lows.iloc[mins[i]])
        h  = float(lows.iloc[mins[i+1]])
        rs = float(lows.iloc[mins[i+2]])
        if h < ls*0.97 and h < rs*0.97 and abs(ls-rs)/ls < 0.05:
            return True
    return False

def cup_and_handle(df):
    c = df["Close"].values
    if len(c) < 40: return False
    n = len(c)
    lh = max(c[:n//4]); rh = max(c[3*n//4:])
    cl = min(c[n//4:3*n//4])
    return lh > cl*1.08 and rh > cl*1.08 and abs(lh-rh)/lh < 0.06

def bullish_flag(df):
    c = df["Close"]
    if len(c) < 20: return False
    g1 = (c.iloc[-15] - c.iloc[-20]) / c.iloc[-20]
    m2 = (c.iloc[-1]  - c.iloc[-10]) / c.iloc[-10]
    return g1 > 0.05 and -0.04 < m2 < 0.01

def bearish_flag(df):
    c = df["Close"]
    if len(c) < 20: return False
    g1 = (c.iloc[-15] - c.iloc[-20]) / c.iloc[-20]
    m2 = (c.iloc[-1]  - c.iloc[-10]) / c.iloc[-10]
    return g1 < -0.05 and -0.01 < m2 < 0.04

def breakout(df):
    highs, lows, closes = df["High"], df["Low"], df["Close"]
    res = float(highs.iloc[-21:-1].max())
    sup = float(lows.iloc[-21:-1].min())
    lc  = float(closes.iloc[-1])
    return {"up": lc > res*1.005, "down": lc < sup*0.995, "res": res, "sup": sup}


# ── Chandeliers japonais ────────────────────────────────────

def hammer(df):
    r = df.iloc[-1]
    o,h,l,c = float(r["Open"]),float(r["High"]),float(r["Low"]),float(r["Close"])
    body = abs(c-o); total = h-l
    if total == 0: return False
    wick_l = min(o,c) - l
    wick_u = h - max(o,c)
    return wick_l > 2*body and wick_u < body*0.5 and body/total < 0.35

def shooting_star(df):
    r = df.iloc[-1]
    o,h,l,c = float(r["Open"]),float(r["High"]),float(r["Low"]),float(r["Close"])
    body = abs(c-o); total = h-l
    if total == 0: return False
    wick_u = h - max(o,c)
    wick_l = min(o,c) - l
    return wick_u > 2*body and wick_l < body*0.5 and body/total < 0.35

def doji(df):
    r = df.iloc[-1]
    o,h,l,c = float(r["Open"]),float(r["High"]),float(r["Low"]),float(r["Close"])
    body = abs(c-o); total = h-l
    if total == 0: return False
    return body/total < 0.1

def engulfing(df):
    if len(df) < 2: return False, False
    p, r = df.iloc[-2], df.iloc[-1]
    po,pc = float(p["Open"]),float(p["Close"])
    ro,rc = float(r["Open"]),float(r["Close"])
    bull = pc < po and rc > ro and ro < pc and rc > po
    bear = pc > po and rc < ro and ro > pc and rc < po
    return bull, bear

def morning_star(df):
    if len(df) < 3: return False
    c1,c2,c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    d1 = float(c1["Close"]) - float(c1["Open"])   # bearish
    d2 = abs(float(c2["Close"]) - float(c2["Open"]))  # small
    d3 = float(c3["Close"]) - float(c3["Open"])   # bullish
    return d1 < -abs(d1)*0.6 and d2 < abs(d1)*0.3 and d3 > abs(d1)*0.5

def evening_star(df):
    if len(df) < 3: return False
    c1,c2,c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    d1 = float(c1["Close"]) - float(c1["Open"])
    d2 = abs(float(c2["Close"]) - float(c2["Open"]))
    d3 = float(c3["Close"]) - float(c3["Open"])
    return d1 > abs(d1)*0.6 and d2 < abs(d1)*0.3 and d3 < -abs(d1)*0.5

def three_white_soldiers(df):
    if len(df) < 3: return False
    candles = [df.iloc[-3], df.iloc[-2], df.iloc[-1]]
    return all(float(c["Close"]) > float(c["Open"]) for c in candles) and \
           float(candles[2]["Close"]) > float(candles[1]["Close"]) > float(candles[0]["Close"])

def three_black_crows(df):
    if len(df) < 3: return False
    candles = [df.iloc[-3], df.iloc[-2], df.iloc[-1]]
    return all(float(c["Close"]) < float(c["Open"]) for c in candles) and \
           float(candles[2]["Close"]) < float(candles[1]["Close"]) < float(candles[0]["Close"])

def harami(df):
    if len(df) < 2: return False
    p, r = df.iloc[-2], df.iloc[-1]
    po,pc = float(p["Open"]),float(p["Close"])
    ro,rc = float(r["Open"]),float(r["Close"])
    bull = pc < po and rc > ro and ro > pc and rc < po and (rc-ro) < (po-pc)*0.5
    bear = pc > po and rc < ro and ro < pc and rc > po and (ro-rc) < (pc-po)*0.5
    return bull, bear

def piercing_line(df):
    if len(df) < 2: return False
    p, r = df.iloc[-2], df.iloc[-1]
    p_bear = float(p["Close"]) < float(p["Open"])
    r_bull = float(r["Close"]) > float(r["Open"])
    mid_prev = (float(p["Open"]) + float(p["Close"])) / 2
    return p_bear and r_bull and float(r["Close"]) > mid_prev and float(r["Open"]) < float(p["Close"])

def dark_cloud_cover(df):
    if len(df) < 2: return False
    p, r = df.iloc[-2], df.iloc[-1]
    p_bull = float(p["Close"]) > float(p["Open"])
    r_bear = float(r["Close"]) < float(r["Open"])
    mid_prev = (float(p["Open"]) + float(p["Close"])) / 2
    return p_bull and r_bear and float(r["Close"]) < mid_prev and float(r["Open"]) > float(p["Close"])


# ── Master detection ────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> List[str]:
    found = []
    try:
        # Chartistes haussiers
        if double_bottom(df):   found.append("🟢 Double Bottom (W) — rebond probable")
        if triple_bottom(df):   found.append("🟢 Triple Bottom — support très fort")
        if inv_head_and_shoulders(df): found.append("🟢 Tête & Épaules Inversé")
        if cup_and_handle(df):  found.append("🟢 Cup & Handle")
        if bullish_flag(df):    found.append("🟢 Flag Haussier")
        if hammer(df):          found.append("🟢 Marteau")
        if morning_star(df):    found.append("🟢 Étoile du Matin")
        if three_white_soldiers(df): found.append("🟢 3 Soldats Blancs")
        if piercing_line(df):   found.append("🟢 Piercing Line")

        bull_eng, bear_eng = engulfing(df)
        if bull_eng: found.append("🟢 Englobante Haussière")

        bull_har, bear_har = harami(df)
        if bull_har: found.append("🟢 Harami Haussier")

        if doji(df): found.append("🟡 Doji — Indécision du marché")

        # Chartistes baissiers
        if double_top(df):   found.append("🔴 Double Top (M) — retournement baissier")
        if triple_top(df):   found.append("🔴 Triple Top — résistance très forte")
        if head_and_shoulders(df): found.append("🔴 Tête & Épaules")
        if bearish_flag(df): found.append("🔴 Flag Baissier")
        if shooting_star(df):found.append("🔴 Étoile Filante")
        if evening_star(df): found.append("🔴 Étoile du Soir")
        if three_black_crows(df): found.append("🔴 3 Corbeaux Noirs")
        if dark_cloud_cover(df):  found.append("🔴 Dark Cloud Cover")
        if bear_eng: found.append("🔴 Englobante Baissière")
        if bear_har: found.append("🔴 Harami Baissier")

        # Breakout
        br = breakout(df)
        if br["up"]:   found.append(f"🟢 Cassure résistance (${br['res']:.2f})")
        if br["down"]: found.append(f"🔴 Cassure support (${br['sup']:.2f})")

    except Exception:
        pass
    return found
