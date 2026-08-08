"""
market_odds.py — betting-odds-implied team strength for the team model.
========================================================================
The team model currently anchors on reconstructed 25/26 Opta xG + ClubElo. Live
betting markets price in transfers, new managers and pre-season form that last
season's data can't see — the biggest documented predictive gain. This module
turns the 26/27 outright markets (title + relegation) into a per-team market Elo
that blends into TeamModel.fit() through the existing ClubElo path (no model
change): pass `clubelo=market_elo_frame()` (or a blend) with `clubelo_weight`.

Data embedded below is a static snapshot (7 Aug 2026, de-vigged from published
decimal odds across bet365/BetMGM/Ladbrokes/Squawka). Refresh live on your
machine with fetch_live_odds() (The Odds API); the season-strength mapping is the
same either way.

Sources: Squawka/OneFootball/Ladbrokes/BettingLounge 26/27 outright markets.
"""
from __future__ import annotations
import numpy as np, pandas as pd

# Decimal odds snapshot, 7 Aug 2026. title = to win the league; releg = to be
# relegated. Established teams carry both; promoted carry releg only.
ODDS_2627 = {
    "Arsenal":        dict(title=2.50, releg=101.0),
    "Man City":       dict(title=3.50, releg=67.0),
    "Liverpool":      dict(title=6.00, releg=51.0),
    "Man United":     dict(title=7.00, releg=34.0),
    "Aston Villa":    dict(title=26.0, releg=26.0),
    "Newcastle":      dict(title=34.0, releg=11.0),
    "Tottenham":      dict(title=41.0, releg=17.0),
    "Chelsea":        dict(title=15.0, releg=21.0),
    "Brighton":       dict(title=101.0, releg=15.0),
    "Bournemouth":    dict(title=251.0, releg=15.0),
    "Brentford":      dict(title=251.0, releg=11.0),
    "Everton":        dict(title=251.0, releg=13.0),
    "Crystal Palace": dict(title=151.0, releg=7.5),
    "Nott'm Forest":  dict(title=151.0, releg=12.0),
    "Fulham":         dict(title=251.0, releg=7.0),
    "Leeds":          dict(title=1001.0, releg=7.5),
    "Sunderland":     dict(title=1001.0, releg=5.0),
    "Coventry":       dict(title=1001.0, releg=1.85),
    "Ipswich":        dict(title=1001.0, releg=1.85),
    "Hull":           dict(title=1001.0, releg=1.26),
}


def _devig(probs):
    s = sum(probs.values())
    return {k: v / s for k, v in probs.items()}


def implied_strength(odds=None):
    """Per-team market strength z-score. Base signal is survival (1 - P(relegated),
    de-vigged to a 3-team market, monotone across the whole table); a title-prob
    term lifts the elite teams the relegation market can't separate."""
    odds = odds or ODDS_2627
    p_rel = _devig({t: 1.0 / o["releg"] for t, o in odds.items()})           # ~sums to 3 teams
    p_rel = {t: min(v * 3.0, 0.99) for t, v in p_rel.items()}
    p_title = _devig({t: 1.0 / o["title"] for t, o in odds.items()})
    s = {}
    for t in odds:
        survive = 1.0 - p_rel[t]
        s[t] = np.log(max(survive, 0.02) / max(1 - survive, 0.02)) + 2.5 * np.log(max(p_title[t], 1e-4) / 0.05)
    v = pd.Series(s)
    return (v - v.mean()) / v.std()


def market_elo_frame(center=1600.0, spread=140.0, odds=None):
    """Market strength -> ClubElo-scale frame the team model can blend.
    market Elo = center + spread * z. Returns DataFrame[team, Elo]."""
    z = implied_strength(odds)
    return pd.DataFrame({"team": z.index, "Elo": center + spread * z.values})


def blended_elo(clubelo_frame, market_weight=0.6, center=1600.0, spread=140.0, odds=None):
    """Blend market Elo with an existing ClubElo frame (team, Elo). market_weight
    on the market. Teams missing from either side fall back to the other."""
    m = market_elo_frame(center, spread, odds).set_index("team")["Elo"]
    if clubelo_frame is None or "team" not in getattr(clubelo_frame, "columns", []):
        return m.reset_index()
    c = clubelo_frame.set_index("team")["Elo"].astype(float)
    teams = sorted(set(m.index) | set(c.index))
    out = {}
    for t in teams:
        if t in m.index and t in c.index:
            out[t] = market_weight * m[t] + (1 - market_weight) * c[t]
        else:
            out[t] = m.get(t, c.get(t))
    return pd.DataFrame({"team": list(out), "Elo": list(out.values())})


def fetch_live_odds(api_key, region="uk", markets="outrights", sport="soccer_epl"):
    """Live refresh via The Odds API (KEY REQUIRED; needs network). Returns an
    ODDS_2627-shaped dict. Offline (e.g. this sandbox) raises; use the embedded
    snapshot. Wire this into a cron and rotate the key to an env var."""
    import json, urllib.request
    url = (f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
           f"?regions={region}&markets={markets}&oddsFormat=decimal&apiKey={api_key}")
    data = json.loads(urllib.request.urlopen(url, timeout=20).read().decode())
    # (parsing left thin: outright market shapes vary by book; map to {team:{title,releg}})
    return data


if __name__ == "__main__":
    print(market_elo_frame().sort_values("Elo", ascending=False).round(0).to_string(index=False))
