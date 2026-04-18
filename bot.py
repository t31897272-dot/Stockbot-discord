# ══════════════════════════════════════════════════════════
#  bot.py — StockBot v2 — Toutes les fonctionnalités
# ══════════════════════════════════════════════════════════
import discord
from discord.ext import commands, tasks
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import asyncio, io

from config import (DISCORD_TOKEN, CHANNEL_ID, WATCHLIST_STOCKS,
                    WATCHLIST_CRYPTO, SCAN_INTERVAL_MINUTES, MIN_CONFIDENCE,
                    DAILY_REPORT_HOUR)
from analyzer  import analyze_ticker
from patterns  import detect_patterns
from news      import get_news_sentiment
from alerts    import (add_alert, check_alerts, add_position, check_positions,
                       list_alerts, list_positions)
from chart_generator import generate_chart

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

COLOR_BUY  = 0x00e5a8
COLOR_SELL = 0xfb7185
COLOR_HOLD = 0xfbbf24
COLOR_INFO = 0x5591c7

def sig_color(s): return {" BUY":COLOR_BUY,"SELL":COLOR_SELL,"HOLD":COLOR_HOLD}.get(s,COLOR_INFO)
def sig_emoji(s): return {"BUY":"🟢","SELL":"🔴","HOLD":"🟡"}.get(s,"⚪")


def build_embed(ticker, r, atype="stock"):
    sig, conf, price = r["signal"], r["confidence"], r["price"]
    chg, rsi, macd   = r["change_pct"], r["rsi"], r["macd"]
    stoch_k          = r.get("stoch_k", 0)
    patterns         = r["patterns"]
    ichi             = r.get("ichimoku", {})
    fib_near         = r.get("fib_near", "")
    vwap             = r.get("vwap", 0)
    news             = r.get("news", {"label": "neutre", "score": 0})
    mtf              = r.get("mtf_bias", 0)
    icon = "₿" if atype=="crypto" else "📈"

    embed = discord.Embed(
        title=f"{sig_emoji(sig)} {icon} {ticker} — {sig}",
        description=f"**{r['reason']}**",
        color=sig_color(f" {sig}"),
        timestamp=datetime.now(timezone.utc)
    )

    embed.add_field(
        name="💰 Prix",
        value=f"`${price:,.2f}` | `{'+' if chg>=0 else ''}{chg:.2f}%`",
        inline=True
    )
    embed.add_field(
        name="📊 Confiance",
        value=f"`{conf}%` {'🔥' if conf>=80 else '✅'}",
        inline=True
    )
    embed.add_field(
        name="📰 Sentiment",
        value=f"`{news['label']}`",
        inline=True
    )
    embed.add_field(
        name="📐 Indicateurs",
        value=(
            f"RSI(14): `{rsi:.1f}` {'🔴' if rsi>70 else '🟢' if rsi<30 else '🟡'}\n"
            f"Stoch RSI: `{stoch_k:.1f}` {'🔴' if stoch_k>80 else '🟢' if stoch_k<20 else '🟡'}\n"
            f"MACD: `{'↑' if macd>0 else '↓'} {macd:.3f}`\n"
            f"VWAP: `${vwap:,.2f}` {'✅' if price>vwap else '⚠️'}"
        ),
        inline=True
    )
    ichi_txt = "☁️ Au-dessus" if ichi.get("above_cloud") else "☁️ En dessous" if ichi.get("below_cloud") else "☁️ Dans le nuage"
    tk_txt   = "↑ TK Cross Bull" if ichi.get("tk_cross_bull") else "↓ TK Cross Bear" if ichi.get("tk_cross_bear") else ""
    embed.add_field(
        name="🌸 Ichimoku",
        value=f"{ichi_txt}\n{tk_txt or 'Pas de croisement'}" + (f"\nFibo: `{fib_near}`" if fib_near else ""),
        inline=True
    )
    embed.add_field(
        name="⏱️ Multi-timeframe",
        value=f"Biais: `{'🟢 +' if mtf>0 else '🔴 ' if mtf<0 else '🟡 '}{mtf}`",
        inline=True
    )
    embed.add_field(
        name="🕯️ Patterns détectés",
        value="\n".join([f"• {p}" for p in patterns[:5]]) or "• Aucun pattern fort",
        inline=False
    )
    if sig in ("BUY","SELL"):
        embed.add_field(
            name="🎯 Zones de trading",
            value=(
                f"Entrée :   `${r['entry']:,.2f}`\n"
                f"Objectif : `${r['target']:,.2f}` ✅\n"
                f"Stop-loss: `${r['stop']:,.2f}` ❌\n"
                f"Ratio R/R: `{r['risk_reward']:.1f}x`"
            ),
            inline=False
        )
    if news.get("headlines"):
        embed.add_field(
            name="📰 Dernières actualités",
            value="\n".join([f"• {h}" for h in news["headlines"][:2]]),
            inline=False
        )
    embed.set_footer(text=f"StockBot v2 • {atype.upper()} • Yahoo Finance")
    return embed


# ── Events ─────────────────────────────────────────────────
@bot.event
async def on_ready():
    print(f"✅ StockBot v2 connecté : {bot.user}")
    auto_scan.start()
    check_user_alerts.start()
    daily_report.start()
    print(f"⏰ Scan toutes les {SCAN_INTERVAL_MINUTES} min | Rapport à {DAILY_REPORT_HOUR}h UTC")


# ── Auto Scan ───────────────────────────────────────────────
@tasks.loop(minutes=SCAN_INTERVAL_MINUTES)
async def auto_scan():
    channel = bot.get_channel(CHANNEL_ID)
    if not channel: return
    all_tickers = [(t,"stock") for t in WATCHLIST_STOCKS] + \
                  [(t,"crypto") for t in WATCHLIST_CRYPTO]
    signals = []
    for ticker, atype in all_tickers:
        try:
            result = analyze_ticker(ticker)
            result["news"] = get_news_sentiment(ticker)
            # Bonus confiance si sentiment aligne le signal
            news_score = result["news"]["score"]
            if result["signal"]=="BUY"  and news_score > 0: result["confidence"] = min(98, result["confidence"]+5)
            if result["signal"]=="SELL" and news_score < 0: result["confidence"] = min(98, result["confidence"]+5)
            if result["signal"] in ("BUY","SELL") and result["confidence"] >= MIN_CONFIDENCE:
                signals.append((ticker, result, atype))
            await asyncio.sleep(1.5)
        except Exception as e:
            print(f"⚠️ {ticker}: {e}")

    if signals:
        summary = discord.Embed(
            title=f"🔔 {len(signals)} signal(s) — Scan automatique",
            description=(
                f"**{sum(1 for _,r,_ in signals if r['signal']=='BUY')} BUY** · "
                f"**{sum(1 for _,r,_ in signals if r['signal']=='SELL')} SELL**\n"
                f"Confiance min : {MIN_CONFIDENCE}%"
            ),
            color=COLOR_BUY,
            timestamp=datetime.now(timezone.utc)
        )
        await channel.send(embed=summary)
        for ticker, result, atype in signals:
            embed = build_embed(ticker, result, atype)
            chart = generate_chart(ticker, result)
            if chart:
                f = discord.File(chart, filename=f"{ticker}_chart.png")
                embed.set_image(url=f"attachment://{ticker}_chart.png")
                await channel.send(embed=embed, file=f)
            else:
                await channel.send(embed=embed)
            await asyncio.sleep(0.5)
    else:
        print(f"[{datetime.now().strftime('%H:%M')}] Scan OK — aucun signal ≥{MIN_CONFIDENCE}%")

@auto_scan.before_loop
async def before_scan():
    await bot.wait_until_ready()


# ── Vérification alertes prix ───────────────────────────────
@tasks.loop(minutes=5)
async def check_user_alerts():
    channel = bot.get_channel(CHANNEL_ID)
    if not channel: return
    all_users = set(list(channel.guild.members) if hasattr(channel, "guild") else [])
    for user in all_users:
        triggered = check_alerts(user.id)
        for msg in triggered:
            await channel.send(f"{user.mention} {msg}")
        pos_triggered = check_positions(user.id)
        for msg in pos_triggered:
            await channel.send(f"{user.mention} {msg}")

@check_user_alerts.before_loop
async def before_check():
    await bot.wait_until_ready()


# ── Rapport quotidien ───────────────────────────────────────
@tasks.loop(hours=1)
async def daily_report():
    now = datetime.now(timezone.utc)
    if now.hour != DAILY_REPORT_HOUR:
        return
    channel = bot.get_channel(CHANNEL_ID)
    if not channel: return

    all_data = []
    for t in WATCHLIST_STOCKS + WATCHLIST_CRYPTO:
        try:
            info  = yf.Ticker(t).fast_info
            price = float(info.last_price)
            prev  = float(info.previous_close)
            chg   = round((price-prev)/prev*100, 2)
            all_data.append((t, price, chg))
            await asyncio.sleep(0.5)
        except Exception:
            pass

    if not all_data: return
    best  = sorted(all_data, key=lambda x: x[2], reverse=True)[:3]
    worst = sorted(all_data, key=lambda x: x[2])[:3]

    embed = discord.Embed(
        title=f"📅 Rapport quotidien — {now.strftime('%d %B %Y')}",
        color=COLOR_INFO,
        timestamp=datetime.now(timezone.utc)
    )
    embed.add_field(
        name="📈 Meilleures performances",
        value="\n".join([f"`{t}` **+{c:.2f}%** (${p:,.2f})" for t,p,c in best]),
        inline=True
    )
    embed.add_field(
        name="📉 Plus fortes baisses",
        value="\n".join([f"`{t}` **{c:.2f}%** (${p:,.2f})" for t,p,c in worst]),
        inline=True
    )
    embed.add_field(
        name="ℹ️ Rappel",
        value=f"Lance `!scan` pour une analyse complète\n`!aide` pour toutes les commandes",
        inline=False
    )
    embed.set_footer(text=f"StockBot v2 • {len(all_data)} actifs surveillés")
    await channel.send(embed=embed)

@daily_report.before_loop
async def before_report():
    await bot.wait_until_ready()


# ══════════════════════════════════════════════════════════
#  COMMANDES
# ══════════════════════════════════════════════════════════

@bot.command(name="analyse", aliases=["a","check"])
async def cmd_analyse(ctx, ticker: str = None):
    if not ticker:
        await ctx.send("❌ Usage : `!analyse AAPL` ou `!analyse BTC-USD`"); return
    ticker = ticker.upper()
    msg = await ctx.send(f"🔍 Analyse complète de `{ticker}` en cours…")
    try:
        result       = analyze_ticker(ticker)
        result["news"] = get_news_sentiment(ticker)
        atype = "crypto" if any(x in ticker for x in ["-USD","-EUR","-BTC"]) else "stock"
        embed = build_embed(ticker, result, atype)
        chart = generate_chart(ticker, result)
        if chart:
            f = discord.File(chart, filename=f"{ticker}_chart.png")
            embed.set_image(url=f"attachment://{ticker}_chart.png")
            await msg.delete()
            await ctx.send(embed=embed, file=f)
        else:
            await msg.edit(content=None, embed=embed)
    except Exception as e:
        await msg.edit(content=f"❌ Erreur `{ticker}` : {e}")


@bot.command(name="scan", aliases=["s"])
async def cmd_scan(ctx):
    await ctx.send("🔄 Scan en cours… (30-90 sec)")
    await auto_scan()


@bot.command(name="prix", aliases=["p","price"])
async def cmd_prix(ctx, ticker: str = None):
    if not ticker:
        await ctx.send("❌ Usage : `!prix TSLA`"); return
    ticker = ticker.upper()
    try:
        info  = yf.Ticker(ticker).fast_info
        price = float(info.last_price)
        prev  = float(info.previous_close)
        chg   = round((price-prev)/prev*100, 2)
        color = COLOR_BUY if chg>=0 else COLOR_SELL
        embed = discord.Embed(
            title=f"💰 {ticker}",
            description=f"**${price:,.2f}** `{'+' if chg>=0 else ''}{chg:.2f}%`",
            color=color
        )
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ {e}")


@bot.command(name="news", aliases=["n","actualite"])
async def cmd_news(ctx, ticker: str = None):
    if not ticker:
        await ctx.send("❌ Usage : `!news AAPL`"); return
    ticker = ticker.upper()
    result = get_news_sentiment(ticker)
    embed  = discord.Embed(
        title=f"📰 Actualités — {ticker}",
        description=f"Sentiment : **{result['label']}** (score: `{result['score']}`)",
        color=COLOR_BUY if result["score"]>0 else COLOR_SELL if result["score"]<0 else COLOR_HOLD
    )
    if result["headlines"]:
        embed.add_field(
            name="Titres récents",
            value="\n".join([f"• {h}" for h in result["headlines"]]),
            inline=False
        )
    await ctx.send(embed=embed)


@bot.command(name="fib", aliases=["fibonacci"])
async def cmd_fib(ctx, ticker: str = None):
    if not ticker:
        await ctx.send("❌ Usage : `!fib AAPL`"); return
    ticker = ticker.upper()
    try:
        import yfinance as yf
        from indicators import calc_fibonacci
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        levels  = calc_fibonacci(df.dropna())
        price   = float(df["Close"].iloc[-1])
        embed   = discord.Embed(title=f"📐 Fibonacci — {ticker}", color=COLOR_INFO)
        lines   = []
        for label, lvl in levels.items():
            marker = " ← prix actuel" if abs(price-lvl)/lvl < 0.01 else ""
            lines.append(f"`{label}` → **${lvl:,.2f}**{marker}")
        embed.add_field(name="Niveaux de retracement", value="\n".join(lines), inline=False)
        embed.add_field(name="Prix actuel", value=f"`${price:,.2f}`", inline=True)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ {e}")


@bot.command(name="alerte", aliases=["alert"])
async def cmd_alerte(ctx, ticker: str = None, price: str = None):
    if not ticker or not price:
        await ctx.send("❌ Usage : `!alerte BTC-USD 90000`\nLe bot t'alertera quand BTC atteint $90 000"); return
    try:
        target = float(price.replace(",", ""))
        msg    = add_alert(ctx.author.id, ticker.upper(), target)
        await ctx.send(msg)
    except ValueError:
        await ctx.send("❌ Prix invalide. Exemple : `!alerte AAPL 200`")


@bot.command(name="alertes")
async def cmd_list_alertes(ctx):
    msg = list_alerts(ctx.author.id)
    embed = discord.Embed(title="🔔 Tes alertes de prix", description=msg, color=COLOR_INFO)
    await ctx.send(embed=embed)


@bot.command(name="achat", aliases=["buy","position"])
async def cmd_achat(ctx, ticker: str = None, entry: str = None, target: str = None, stop: str = None):
    if not all([ticker, entry, target, stop]):
        await ctx.send("❌ Usage : `!achat AAPL 180 195 174`\n(ticker entrée objectif stop-loss)"); return
    try:
        msg = add_position(ctx.author.id, ticker.upper(),
                           float(entry), float(target), float(stop))
        await ctx.send(msg)
    except ValueError:
        await ctx.send("❌ Valeurs invalides.")


@bot.command(name="positions")
async def cmd_positions(ctx):
    msg   = list_positions(ctx.author.id)
    embed = discord.Embed(title="📌 Tes positions en cours", description=msg, color=COLOR_INFO)
    await ctx.send(embed=embed)


@bot.command(name="watchlist", aliases=["wl"])
async def cmd_watchlist(ctx):
    embed = discord.Embed(title="📋 Watchlist", color=COLOR_INFO)
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
    embed.set_footer(text=f"Scan /{SCAN_INTERVAL_MINUTES} min • Confiance ≥{MIN_CONFIDENCE}%")
    await ctx.send(embed=embed)


@bot.command(name="rapport", aliases=["report"])
async def cmd_rapport(ctx):
    await ctx.send("📊 Génération du rapport en cours…")
    await daily_report()


@bot.command(name="aide", aliases=["help","h"])
async def cmd_aide(ctx):
    embed = discord.Embed(title="🤖 StockBot v2 — Commandes", color=COLOR_INFO)
    cmds = [
        ("!analyse <TICKER>",              "Analyse complète + graphique + news"),
        ("!prix <TICKER>",                 "Prix actuel rapide"),
        ("!news <TICKER>",                 "Sentiment des actualités"),
        ("!fib <TICKER>",                  "Niveaux de Fibonacci"),
        ("!scan",                          "Scan immédiat de la watchlist"),
        ("!alerte <TICKER> <PRIX>",        "Alerte quand le prix est atteint"),
        ("!alertes",                       "Voir tes alertes actives"),
        ("!achat <TICKER> <E> <TP> <SL>",  "Enregistrer une position"),
        ("!positions",                     "Voir tes positions en cours"),
        ("!watchlist",                     "Voir la liste des actifs suivis"),
        ("!rapport",                       "Rapport de performance immédiat"),
    ]
    for name, desc in cmds:
        embed.add_field(name=f"`{name}`", value=desc, inline=False)
    embed.set_footer(text="StockBot v2 • Analyse technique avancée")
    await ctx.send(embed=embed)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
