# ══════════════════════════════════════════════════════════
#  alerts.py — Alertes de prix personnalisées + suivi positions
# ══════════════════════════════════════════════════════════
from typing import Dict, List
import yfinance as yf

# Structure : {user_id: [{"ticker": str, "price": float, "direction": "above"/"below", "note": str}]}
price_alerts: Dict[int, List[dict]] = {}

# Structure : {user_id: [{"ticker": str, "entry": float, "target": float, "stop": float, "qty": str}]}
positions: Dict[int, List[dict]] = {}


def add_alert(user_id: int, ticker: str, target_price: float) -> str:
    try:
        current = float(yf.Ticker(ticker).fast_info.last_price)
        direction = "above" if target_price > current else "below"
        if user_id not in price_alerts:
            price_alerts[user_id] = []
        price_alerts[user_id].append({
            "ticker": ticker,
            "price": target_price,
            "direction": direction,
        })
        arrow = "↗️" if direction == "above" else "↘️"
        return (f"✅ Alerte créée : `{ticker}` {arrow} **${target_price:,.2f}**\n"
                f"Prix actuel : `${current:,.2f}`")
    except Exception as e:
        return f"❌ Erreur : {e}"


def check_alerts(user_id: int) -> List[str]:
    triggered = []
    if user_id not in price_alerts:
        return triggered
    remaining = []
    for alert in price_alerts[user_id]:
        try:
            current = float(yf.Ticker(alert["ticker"]).fast_info.last_price)
            hit = (alert["direction"] == "above" and current >= alert["price"]) or \
                  (alert["direction"] == "below" and current <= alert["price"])
            if hit:
                triggered.append(
                    f"🔔 **{alert['ticker']}** a atteint **${alert['price']:,.2f}** "
                    f"(actuel: ${current:,.2f})"
                )
            else:
                remaining.append(alert)
        except Exception:
            remaining.append(alert)
    price_alerts[user_id] = remaining
    return triggered


def add_position(user_id: int, ticker: str, entry: float, target: float, stop: float, qty: str = "?") -> str:
    if user_id not in positions:
        positions[user_id] = []
    positions[user_id].append({
        "ticker": ticker, "entry": entry,
        "target": target, "stop": stop, "qty": qty
    })
    rr = round(abs(target-entry)/abs(entry-stop), 2) if abs(entry-stop) > 0 else 0
    return (f"📌 Position enregistrée : `{ticker}`\n"
            f"Entrée: `${entry}` | TP: `${target}` | SL: `${stop}` | R/R: `{rr}x`")


def check_positions(user_id: int) -> List[str]:
    msgs = []
    if user_id not in positions:
        return msgs
    remaining = []
    for pos in positions[user_id]:
        try:
            current = float(yf.Ticker(pos["ticker"]).fast_info.last_price)
            pnl = round((current - pos["entry"]) / pos["entry"] * 100, 2)
            if current >= pos["target"]:
                msgs.append(f"🎯 **{pos['ticker']}** a atteint l'objectif `${pos['target']}` ! PnL: `+{pnl}%` 🎉")
            elif current <= pos["stop"]:
                msgs.append(f"🛑 **{pos['ticker']}** a touché le stop-loss `${pos['stop']}`. PnL: `{pnl}%`")
            else:
                remaining.append(pos)
        except Exception:
            remaining.append(pos)
    positions[user_id] = remaining
    return msgs


def list_alerts(user_id: int) -> str:
    if user_id not in price_alerts or not price_alerts[user_id]:
        return "Aucune alerte de prix active."
    lines = []
    for a in price_alerts[user_id]:
        arrow = "↗️" if a["direction"]=="above" else "↘️"
        lines.append(f"• `{a['ticker']}` {arrow} `${a['price']:,.2f}`")
    return "\n".join(lines)


def list_positions(user_id: int) -> str:
    if user_id not in positions or not positions[user_id]:
        return "Aucune position en cours."
    lines = []
    for p in positions[user_id]:
        try:
            current = float(yf.Ticker(p["ticker"]).fast_info.last_price)
            pnl = round((current - p["entry"]) / p["entry"] * 100, 2)
            emoji = "📈" if pnl >= 0 else "📉"
            lines.append(f"• `{p['ticker']}` {emoji} `{'+' if pnl>=0 else ''}{pnl}%` | TP `${p['target']}` | SL `${p['stop']}`")
        except Exception:
            lines.append(f"• `{p['ticker']}` | TP `${p['target']}` | SL `${p['stop']}`")
    return "\n".join(lines)
