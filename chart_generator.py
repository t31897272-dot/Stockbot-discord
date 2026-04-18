# ══════════════════════════════════════════════════════════
#  chart_generator.py — Génère un graphique en image Discord
# ══════════════════════════════════════════════════════════
import io
import yfinance as yf
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def generate_chart(ticker: str, result: dict) -> io.BytesIO | None:
    if not MPL_AVAILABLE:
        return None
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna().tail(60)

        closes = df["Close"]
        sma20 = closes.rolling(20).mean()
        sma50 = closes.rolling(50).mean()
        bb_mid = closes.rolling(20).mean()
        bb_std = closes.rolling(20).std()
        bb_up  = bb_mid + 2*bb_std
        bb_low = bb_mid - 2*bb_std

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        facecolor="#1c1b19")
        for ax in [ax1, ax2]:
            ax.set_facecolor("#1c1b19")
            ax.tick_params(colors="#cdccca")
            ax.spines[:].set_color("#393836")

        x = range(len(df))
        # Bougies
        for i, (idx, row) in enumerate(df.iterrows()):
            o,h,l,c = float(row["Open"]),float(row["High"]),float(row["Low"]),float(row["Close"])
            color = "#00e5a8" if c >= o else "#fb7185"
            ax1.plot([i,i],[l,h], color=color, linewidth=0.8)
            ax1.bar(i, abs(c-o), bottom=min(o,c), color=color, width=0.6, alpha=0.9)

        ax1.plot(x, sma20, color="#fbbf24", linewidth=1.2, label="SMA20", alpha=0.8)
        ax1.plot(x, sma50, color="#5591c7", linewidth=1.2, label="SMA50", alpha=0.8)
        ax1.fill_between(x, bb_up, bb_low, alpha=0.08, color="#4f98a3", label="Bollinger")

        # Signal
        sig   = result["signal"]
        price = result["price"]
        color_sig = "#00e5a8" if sig=="BUY" else "#fb7185" if sig=="SELL" else "#fbbf24"
        ax1.axhline(price, color=color_sig, linewidth=1, linestyle="--", alpha=0.7)
        if sig in ("BUY","SELL"):
            ax1.axhline(result["target"], color="#00e5a8", linewidth=0.8, linestyle=":", alpha=0.6, label=f"TP ${result['target']:.2f}")
            ax1.axhline(result["stop"],   color="#fb7185", linewidth=0.8, linestyle=":", alpha=0.6, label=f"SL ${result['stop']:.2f}")

        ax1.set_title(f"{ticker} — {sig} | Confiance {result['confidence']}%",
                      color="#cdccca", fontsize=13, pad=10)
        ax1.legend(facecolor="#201f1d", edgecolor="#393836", labelcolor="#cdccca", fontsize=8)
        ax1.set_xlim(-1, len(df)+1)
        ax1.set_ylabel("Prix", color="#7a7974")
        ax1.tick_params(labelbottom=False)

        # Volume
        for i, (idx, row) in enumerate(df.iterrows()):
            o,c = float(row["Open"]),float(row["Close"])
            color = "#00e5a8" if c >= o else "#fb7185"
            ax2.bar(i, float(row["Volume"]), color=color, alpha=0.6, width=0.6)
        ax2.set_ylabel("Volume", color="#7a7974")
        ax2.set_xlim(-1, len(df)+1)

        # Dates
        step  = max(1, len(df)//8)
        ticks = list(range(0, len(df), step))
        labels= [df.index[i].strftime("%d/%m") for i in ticks]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels, color="#7a7974", fontsize=7)

        plt.tight_layout(pad=1.5)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor="#1c1b19")
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Chart error: {e}")
        return None
