# ══════════════════════════════════════════════════════════
#  news.py — Analyse du sentiment des actualités Yahoo Finance
# ══════════════════════════════════════════════════════════
import yfinance as yf

POSITIVE_WORDS = [
    "surge","rally","gain","rise","record","bull","breakout","beat",
    "profit","growth","strong","upgrade","buy","outperform","high",
    "hausse","monte","rebond","croissance","record","achat","bénéfice",
    "fort","positif","progression","momentum","optimiste",
]
NEGATIVE_WORDS = [
    "fall","drop","crash","decline","loss","bear","miss","weak",
    "downgrade","sell","underperform","low","risk","warning","fear",
    "baisse","chute","recul","perte","vente","faible","risque",
    "négatif","alerte","pessimiste","inquiétude","effondrement",
]


def get_news_sentiment(ticker: str) -> dict:
    """
    Retourne un score de sentiment basé sur les titres des news Yahoo Finance.
    Score > 0 = positif, < 0 = négatif
    """
    try:
        t       = yf.Ticker(ticker)
        news    = t.news or []
        if not news:
            return {"score": 0, "label": "neutre", "headlines": [], "count": 0}

        score      = 0
        headlines  = []
        analyzed   = 0

        for item in news[:10]:
            title = item.get("content", {}).get("title", "") or \
                    item.get("title", "") or ""
            if not title:
                continue
            title_lower = title.lower()
            pos = sum(1 for w in POSITIVE_WORDS if w in title_lower)
            neg = sum(1 for w in NEGATIVE_WORDS if w in title_lower)
            score += pos - neg
            headlines.append(title[:80])
            analyzed += 1

        if score > 1:   label = "positif 📈"
        elif score < -1: label = "négatif 📉"
        else:            label = "neutre ➡️"

        return {
            "score":     score,
            "label":     label,
            "headlines": headlines[:3],
            "count":     analyzed
        }
    except Exception:
        return {"score": 0, "label": "neutre", "headlines": [], "count": 0}
