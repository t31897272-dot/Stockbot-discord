import discord
from discord.ext import commands, tasks
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import os
from config import (
    DISCORD_TOKEN, CHANNEL_ID, WATCHLIST_STOCKS,
    WATCHLIST_CRYPTO, SCAN_INTERVAL_MINUTES, MIN_CONFIDENCE
)
from analyzer import analyze_ticker
from patterns import detect_patterns

# ── Bot setup ──────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

scan_running = False

# ── Colors ─────────────────────────────────────────────────────────────────
COLOR_BUY  = 0x00e5a8
COLOR_SELL = 0xfb7185
COLOR_HOLD = 0xfbbf24
COLOR_INFO = 0x5591c7

# ── Helpers ────────────────────────────────────────────────────────────────
def signal_color(sig):
    return {
        "BUY": COLOR_BUY,
        "SELL": COLOR_SELL,
        "HOLD": COLOR_HOLD
    }.get(sig, COLOR_INFO)

def signal_emoji(sig):
    return {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")

def build_embed(ticker, result, asset_type="stock"):
    sig    = result["signal"]
    conf   = result["confidence"]
    price  = result["price"]
    chg    = result["change_pct"]
    rsi    = result["rsi"]
    macd   = result["macd"]
    patterns = result["patterns"]
    reason = result["reason"]
    entry  = result["entry"]
    target = result["target"]
    stop   = result["stop"]
    rr     = result["risk_reward"]

    icon = "₿" if asset_type == "crypto" else "📈"
    title = f"{signal_emoji(sig)} {icon} {ticker} — {sig}"

    embed = discord.Embed(
        title=title,
        description=f"**{reason}**",
        color=signal_color(sig),
        timestamp=datetime.utcnow()
    )

    embed.add_field(
        name="💰 Prix & Variation",
        value=f"`${price:.2f}` | `{'+' if chg >= 0 else ''}{chg:.2f}%` aujourd'hui",
        inline=True
    )
    embed.add_field(
        name="📊 Confiance IA",
        value=f"`{conf}%` {'🔥' if conf >= 80 else '✅' if conf >= 65 else ''}",
        inline=True
    )
    embed.add_field(name="\u200b", value="\u200b", inline=True)

    embed.add_field(
        name="📐 Indicateurs",
        value=(
            f"RSI(14): `{rsi:.1f}` {'🔴' if rsi > 70 else '🟢' if rsi < 30 else '🟡'}\n"
            f"MACD: `{'↑ Haussier' if macd > 0 else '↓ Baissier'}` (`{macd:.3f}`)"
        ),
        inline=True
    )
    embed.add_field(
        name="🕯️ Patterns détectés",
        value="\n".join([f"• {p}" for p in patterns]) if patterns else "• Aucun pattern fort",
        inline=True
    )
    embed.add_field(name="\u200b", value="\u200b", inline=True)

    if sig in ("BUY", "SELL"):
        embed.add_field(
            name="🎯 Zones de trading",
            value=(
                f"Entrée :   `${entry:.2f}`\n"
                f"Objectif : `${target:.2f}` ✅\n"
                f"Stop-loss: `${stop:.2f}` ❌\n"
                f"Ratio R/R: `{rr:.1f}x`"
            ),
            inline=False
        )

    embed.set_footer(text=f"StockBot AI • {asset_type.upper()} • Données Yahoo Finance")
    return embed

# ── Events ─────────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    print(f"✅ Bot connecté : {bot.user} (ID: {bot.user.id})")
    print(f"📡 Canal d'alerte : {CHANNEL_ID}")
    auto_scan.start()
    print(f"⏰ Scan automatique toutes les {SCAN_INTERVAL_MINUTES} min")

# ── Auto Scan Task ──────────────────────────────────────────────────────────
@tasks.loop(minutes=SCAN_INTERVAL_MINUTES)
async def auto_scan():
    global scan_running
    if scan_running:
        return
    scan_running = True
    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        print(f"❌ Canal introuvable : {CHANNEL_ID}")
        scan_running = False
        return

    all_tickers = [(t, "stock") for t in WATCHLIST_STOCKS] + \
                  [(t, "crypto") for t in WATCHLIST_CRYPTO]

    signals_found = []
    for ticker, atype in all_tickers:
        try:
            result = analyze_ticker(ticker)
            if result["signal"] in ("BUY", "SELL") and result["confidence"] >= MIN_CONFIDENCE:
                signals_found.append((ticker, result, atype))
            await asyncio.sleep(1)
        except Exception as e:
            print(f"⚠️ Erreur {ticker}: {e}")

    if signals_found:
        summary = discord.Embed(
            title=f"🔔 {len(signals_found)} signal(s) détecté(s) !",
            description=(
                f"**{sum(1 for _,r,_ in signals_found if r['signal']=='BUY')} BUY** · "
                f"**{sum(1 for _,r,_ in signals_found if r['signal']=='SELL')} SELL**\n"
                f"Confiance minimale : {MIN_CONFIDENCE}%"
            ),
            color=COLOR_BUY,
            timestamp=datetime.utcnow()
        )
        summary.set_footer(text="StockBot AI • Analyse automatique")
        await channel.send(embed=summary)
        for ticker, result, atype in signals_found:
            embed = build_embed(ticker, result, atype)
            await channel.send(embed=embed)
            await asyncio.sleep(0.5)
    else:
        print(f"[{datetime.now().strftime('%H:%M')}] Scan terminé — aucun signal ≥{MIN_CONFIDENCE}%")

    scan_running = False

@auto_scan.before_loop
async def before_scan():
    await bot.wait_until_ready()

# ── Commands ────────────────────────────────────────────────────────────────
@bot.command(name="analyse", aliases=["a", "check"])
async def cmd_analyse(ctx, ticker: str = None):
    """!analyse AAPL — Analyse une action ou crypto"""
    if not ticker:
        await ctx.send("❌ Usage : `!analyse AAPL` ou `!analyse BTC-USD`")
        return
    ticker = ticker.upper()
    msg = await ctx.send(f"🔍 Analyse de `{ticker}` en cours…")
    try:
        result = analyze_ticker(ticker)
        atype = "crypto" if "-USD" in ticker or "-EUR" in ticker else "stock"
        embed = build_embed(ticker, result, atype)
        await msg.edit(content=None, embed=embed)
    except Exception as e:
        await msg.edit(content=f"❌ Erreur pour `{ticker}` : {e}")

@bot.command(name="scan", aliases=["s"])
async def cmd_scan(ctx):
    """!scan — Lance un scan immédiat de toute la watchlist"""
    await ctx.send("🔄 Scan de la watchlist en cours… (peut prendre 30-60 sec)")
    await auto_scan()

@bot.command(name="watchlist", aliases=["wl"])
async def cmd_watchlist(ctx):
    """!watchlist — Affiche la watchlist actuelle"""
    embed = discord.Embed(title="📋 Watchlist actuelle", color=COLOR_INFO)
    embed.add_field(
        name="📈 Actions",
        value="\n".join([f"`{t}`" for t in WATCHLIST_STOCKS]) or "Aucune",
        inline=True
    )
    embed.add_field(
        name="₿ Cryptos",
        value="\n".join([f"`{t}`" for t in WATCHLIST_CRYPTO]) or "Aucune",
        inline=True
    )
    embed.set_footer(text=f"Scan toutes les {SCAN_INTERVAL_MINUTES} min • Confiance min : {MIN_CONFIDENCE}%")
    await ctx.send(embed=embed)

@bot.command(name="prix", aliases=["p", "price"])
async def cmd_prix(ctx, ticker: str = None):
    """!prix TSLA — Prix rapide d'un ticker"""
    if not ticker:
        await ctx.send("❌ Usage : `!prix TSLA`")
        return
    ticker = ticker.upper()
    try:
        data = yf.Ticker(ticker).fast_info
        price = data.last_price
        prev  = data.previous_close
        chg   = (price - prev) / prev * 100
        color = COLOR_BUY if chg >= 0 else COLOR_SELL
        embed = discord.Embed(
            title=f"💰 {ticker}",
            description=f"**${price:.2f}** `{'+' if chg>=0 else ''}{chg:.2f}%`",
            color=color
        )
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Erreur : {e}")

@bot.command(name="aide", aliases=["help2", "h"])
async def cmd_aide(ctx):
    """!aide — Liste des commandes"""
    embed = discord.Embed(title="🤖 StockBot AI — Commandes", color=COLOR_INFO)
    cmds = [
        ("!analyse <TICKER>", "Analyse complète + signal BUY/SELL/HOLD"),
        ("!prix <TICKER>",    "Prix actuel rapide"),
        ("!scan",             "Lance un scan immédiat"),
        ("!watchlist",        "Voir la liste des actifs suivis"),
        ("!aide",             "Ce message"),
    ]
    for name, desc in cmds:
        embed.add_field(name=f"`{name}`", value=desc, inline=False)
    embed.set_footer(text="Exemples : !analyse AAPL | !analyse BTC-USD | !prix NVDA")
    await ctx.send(embed=embed)

# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
