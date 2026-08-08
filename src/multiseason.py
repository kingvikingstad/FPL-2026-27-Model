from __future__ import annotations
import config
"""
multiseason.py — does prior-season data improve early-season projection?
========================================================================
The claim to test: two seasons of player data give better priors than one,
specifically in the small-sample regime (early gameweeks) where the model is
weakest and most used.

TEST DESIGN (clean, and it avoids circularity)
  Predict 2025-26 points in gameweeks 2..10 using only information available
  before each gameweek:
    (a) PRIOR SEASON only     — 2024-25 per-90 rates, frozen
    (b) CURRENT SEASON only   — rolling 25/26 rates to date (what the model does now)
    (c) HIERARCHICAL blend    — empirical-Bayes shrinkage of the current-season
                                rate toward the player's OWN prior-season rate,
                                weight = m/(m+K) in minutes
  Everything is leak-free. If (c) beats (b), prior-season data earns its place.

Players are joined across seasons on `player_code`, the stable FPL identifier.
"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats

BASE = config.REPO


def build_2425_panel():
    """2024-25 uses a flatter layout: playermatchstats/GW{n}/ and matches/GW{n}/."""
    players = pd.read_csv(f"{BASE}/2024-2025/players/players.csv")
    frames = []
    for gwdir in sorted(glob.glob(f"{BASE}/2024-2025/playermatchstats/GW*")):
        gw = int(os.path.basename(gwdir).replace("GW", ""))
        try:
            pms = pd.read_csv(f"{gwdir}/playermatchstats.csv")
        except Exception:
            continue
        pms["gameweek"] = gw
        frames.append(pms)
    d = pd.concat(frames, ignore_index=True)
    # Premier League only
    d = d[d.match_id.astype(str).str.contains("-prem-", na=False)].copy()
    d = d.merge(players[["player_id", "player_code", "web_name", "position"]],
                on="player_id", how="left")
    def num(c):
        if c not in d.columns:                       # 24/25 lacks some 25/26 fields
            return pd.Series(0.0, index=d.index)
        return pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d["mins"] = num("minutes_played")
    pen_att = num("penalties_scored") + num("penalties_missed")
    d["npxg"] = (num("xg") - 0.79 * pen_att).clip(lower=0)
    d["xa_"] = num("xa")
    d["defcon_raw"] = num("defensive_contributions")
    d["pos"] = d.position.map({"Goalkeeper": "GK", "Defender": "DEF",
                               "Midfielder": "MID", "Forward": "FWD"})
    return d


def season_rates(panel, min_minutes=450):
    g = panel.groupby(["player_code", "pos"], dropna=False)
    a = g.agg(mins=("mins", "sum"), npxg=("npxg", "sum"), xa=("xa_", "sum"),
              defcon=("defcon_raw", "sum")).reset_index()
    a = a[a.mins >= min_minutes].copy()
    n90 = a.mins / 90.0
    a["inv90"] = (a.npxg + a.xa) / n90
    a["dc90"] = a.defcon / n90
    return a


def run_test(K_grid=(0, 180, 450, 900, 1800, 4000), max_gw=10):
    # --- prior season (24/25) ---
    p24 = build_2425_panel()
    r24 = season_rates(p24).set_index("player_code")
    print(f"2024-25 panel: {len(p24)} player-matches, {p24.player_code.nunique()} players; "
          f"rates for {len(r24)} with >=450 min")

    # --- current season (25/26) with FPL points ---
    p25 = pd.read_pickle(config.PMS_PANEL)
    codes = pd.read_csv(f"{BASE}/2025-2026/players.csv")[["player_id", "player_code"]]
    p25 = p25.merge(codes, on="player_id", how="left")
    cur = p25.groupby(["player_code", "pos", "gameweek"], dropna=False).agg(
        mins=("mins", "sum"), npxg=("npxg", "sum"), xa=("xa_", "sum"),
        defcon=("defcon_raw", "sum"), pts=("total_points", "first")).reset_index()
    cur = cur.sort_values(["player_code", "gameweek"])
    g = cur.groupby("player_code")
    cur["cum_min"] = g["mins"].apply(lambda s: s.shift(1).cumsum()).values
    cur["cum_inv"] = g.apply(lambda d: (d.npxg + d.xa).shift(1).cumsum()).values
    cur["cur_rate"] = cur.cum_inv / (cur.cum_min / 90.0).replace(0, np.nan)
    cur["prior_rate"] = cur.player_code.map(r24["inv90"])
    cur["prior_min"] = cur.player_code.map(r24["mins"])

    d = cur[(cur.gameweek.between(2, max_gw)) & cur.pts.notna()
            & cur.prior_rate.notna()].copy()
    # positional fallback mean for the shrinkage target when a rate is missing
    posmean = d.groupby("pos").prior_rate.transform("mean")
    d["cur_rate_f"] = d.cur_rate.fillna(posmean)
    d = d[d.pos.isin(["MID", "FWD"])]
    print(f"test sample (MID/FWD, GW2-{max_gw}, present in BOTH seasons): {len(d)} rows, "
          f"{d.player_code.nunique()} players\n")

    out = []
    rho_prior = stats.spearmanr(d.prior_rate, d.pts).correlation
    rho_cur = stats.spearmanr(d.cur_rate_f, d.pts).correlation
    out.append({"model": "(a) prior season only (24/25 frozen)", "K": "-", "spearman": rho_prior})
    out.append({"model": "(b) current season only (rolling 25/26)", "K": "-", "spearman": rho_cur})
    for K in K_grid:
        m = d.cum_min.fillna(0)
        w = m / (m + K) if K > 0 else 1.0
        blend = w * d.cur_rate_f + (1 - w) * d.prior_rate
        out.append({"model": "(c) hierarchical blend", "K": K,
                    "spearman": stats.spearmanr(blend, d.pts).correlation})
    res = pd.DataFrame(out)
    print("Predicting next-GW points, early season (higher = better):")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = res[res.model.str.startswith("(c)")].sort_values("spearman", ascending=False).iloc[0]
    print(f"\nbest blend: K={best.K} -> {best.spearman:.4f}  "
          f"vs current-only {rho_cur:.4f}  (gain {best.spearman-rho_cur:+.4f})")

    # where does it help most? by how much 25/26 evidence exists so far
    print("\nGain by amount of current-season evidence (K=best):")
    Kb = int(best.K) if str(best.K).isdigit() else 900
    m = d.cum_min.fillna(0); w = m / (m + Kb)
    d["blend"] = w * d.cur_rate_f + (1 - w) * d.prior_rate
    d["bucket"] = pd.cut(m, [-1, 90, 270, 540, 900, 1e9],
                         labels=["<1 game", "1-3", "3-6", "6-10", "10+"])
    for b, s in d.groupby("bucket", observed=True):
        if len(s) < 50: continue
        rc = stats.spearmanr(s.cur_rate_f, s.pts).correlation
        rb = stats.spearmanr(s.blend, s.pts).correlation
        print(f"  {str(b):9s} n={len(s):5d}  current-only {rc:+.4f}  blended {rb:+.4f}  gain {rb-rc:+.4f}")
    return res


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    run_test()
