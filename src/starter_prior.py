"""
starter_prior.py — depth-aware start probability for COLD-START players
=======================================================================
The projection's minutes gate is its dominant single-GW lever. For players
with real minutes history the Beta(start_a, start_b) prior is well-identified.
For cold-start players (new signings, promoted-club squads, youth) the roster
builder falls back to a price-only regression, which badly over-rates fit-but-
fringe players at strong clubs (a £4.5m Man City centre-back inherits City's
clean-sheet EV because nothing says he won't start).

Ownership is the market's forecast of playing time and is the strongest
predictor of starts after minutes themselves (Spearman ~0.68 on 25/26; ~0.62
even among <=£5.5m players). It is NOT otherwise an input to the points model,
so using it to set the cold-start start prior is new information, not double
counting. This is a STATIC depth prior, not a week-to-week rotation multiplier
(rotation tested null — deliberately not modelled).

Applied to cold_start rows only. Established players keep their minutes-derived
Beta; apply_availability still overrides everyone on injuries and confirmed XIs.
"""
from __future__ import annotations
import numpy as np, pandas as pd, pickle, os

POS = ("GK", "DEF", "MID", "FWD")
CAL_PATH = "/tmp/own_start_cal.pkl"


def calibrate_ownership_start(panel_path="/tmp/pms_panel.pkl",
                              stats_csv="/home/claude/repo/FPL-Core-Insights-main/"
                                        "data/2025-2026/playerstats.csv",
                              team_games=38, min_rows=1):
    """Fit logit P(start per GW) ~ 1 + log(ownership) + price, per position,
    from 25/26. Start = played >=60 min in a match. Returns {pos: (b0,b1,b2)}."""
    panel = pd.read_pickle(panel_path)
    panel["is_start"] = (panel.mins >= 60).astype(float)
    gp = panel.groupby("player_id").agg(
        starts=("is_start", "sum"), rows=("is_start", "size"),
        pos=("pos", "last")).reset_index()
    last = pd.read_csv(stats_csv).sort_values("gw").groupby("id").last()
    gp["own"] = gp.player_id.map(last["selected_by_percent"])
    gp["price"] = gp.player_id.map(last["now_cost"])
    g = gp.dropna(subset=["own", "price"])
    g = g[g.rows >= min_rows]
    g["sr"] = (g.starts / team_games).clip(0.01, 0.99)
    g["logit"] = np.log(g.sr / (1 - g.sr))
    g["logown"] = np.log(g.own.clip(0.05, 95))
    cal = {}
    for p in POS:
        s = g[g.pos == p]
        if len(s) < 8:                          # fall back to pooled fit
            s = g
        X = np.column_stack([np.ones(len(s)), s.logown, s.price])
        b, *_ = np.linalg.lstsq(X, s.logit.values, rcond=None)
        cal[p] = b
    pickle.dump(cal, open(CAL_PATH, "wb"))
    return cal


def load_calibration():
    if os.path.exists(CAL_PATH):
        return pickle.load(open(CAL_PATH, "rb"))
    return calibrate_ownership_start()


def coldstart_start_prob(own, price, pos, cal):
    """Ownership+price -> P(start per GW), clipped to a sane [0.03, 0.97]."""
    b = cal.get(pos, cal.get("MID"))
    lo = b[0] + b[1] * np.log(max(float(own), 0.05)) + b[2] * float(price)
    return float(np.clip(1.0 / (1.0 + np.exp(-lo)), 0.03, 0.97))


def apply_coldstart_depth(players: pd.DataFrame, cal=None, kstart=6.0,
                          blend_price=0.35):
    """Reset start_a/start_b for cold_start rows from the ownership-aware prob.

    blend_price: weight on the ORIGINAL price-derived start prob (retains some
    of the roster builder's signal); the rest goes to the ownership model.
    Established rows (cold_start False/absent) are untouched.
    """
    if cal is None:
        cal = load_calibration()
    p = players.copy()
    if "cold_start" not in p.columns:
        return p
    for i, r in p.iterrows():
        if not bool(r.get("cold_start", False)):
            continue
        prior = r.start_a / (r.start_a + r.start_b) if (r.start_a + r.start_b) > 0 else 0.5
        own_p = coldstart_start_prob(r.get("own", 0.0), r.get("cost", 4.5), r.pos, cal)
        new_p = blend_price * prior + (1 - blend_price) * own_p
        p.at[i, "start_a"] = float(np.clip(new_p, 1e-3, 1) * kstart)
        p.at[i, "start_b"] = float(np.clip(1 - new_p, 1e-3, 1) * kstart)
    return p


def apply_minutes_shrinkage(players: pd.DataFrame, cal=None, k_min=900.0):
    """Evidence-weighted start prior for ALL players with a minutes column.

    Blends the historical start probability with the ownership-implied prior,
    weighting history by its precision: w_hist = minutes / (minutes + k_min).
    Rich-history players (Gabriel, 3900 min) are essentially untouched; thin-
    history players whose role has changed (Mosquera, 989 min, promoted into the
    XI by an injury ahead of him) are pulled toward the market's current view.
    This is the Bayesian-correct generalisation of the cold-start depth prior:
    minutes=0 -> w_hist=0 -> fully ownership-implied (identical to that prior).

    Preserves each Beta's concentration (total pseudo-count), shifting only the
    mean. Injuries/confirmed XIs still override afterward via apply_availability.
    """
    if cal is None:
        cal = load_calibration()
    p = players.copy()
    if "minutes" not in p.columns:
        return p
    for i, r in p.iterrows():
        if bool(r.get("cold_start", False)):
            continue                                  # handled by apply_coldstart_depth
        N = r.start_a + r.start_b
        if N <= 0:
            continue
        p_hist = r.start_a / N
        p_own = coldstart_start_prob(r.get("own", 0.0), r.get("cost", 4.5), r.pos, cal)
        w = float(r.get("minutes", 0.0)) / (float(r.get("minutes", 0.0)) + k_min)
        p_new = w * p_hist + (1 - w) * p_own
        p.at[i, "start_a"] = float(np.clip(p_new, 1e-3, 1) * N)
        p.at[i, "start_b"] = float(np.clip(1 - p_new, 1e-3, 1) * N)
    return p
