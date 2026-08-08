import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
final_2627.py — 26/27 projections on rebuilt per-match priors
=============================================================
Joins on **player_code**, the stable cross-season FPL identifier (Haaland is
223094 in both seasons, while player_id is reassigned 430 -> 411). This removes
the name-matching failures that previously mis-joined Reece/Daniel James and
Dean/Jordan Henderson.

Priors come from 25/26 PER-MATCH data (pms_priors), which beat the FPL
aggregates head-to-head on next-gameweek rank correlation (0.210 vs 0.181).
Penalty rates are MEASURED per taker rather than assumed.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys
import core_insights as ci
import pms_priors as pp
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
import signals as sg

REPO = config.REPO

# ---- 26/27 roster + signals (real) ---------------------------------------
d26, t26, gwsum = ci.load()   # already carries player_code from players.csv
assert "player_code" in d26.columns, "expected player_code from core_insights.load()"
sig = ci.to_signals(d26)

# ---- 25/26 per-match priors, keyed by stable player_code ------------------
rates = pp.season_rates()
pri = pp.to_model_priors(rates)
p25 = pd.read_csv(f"{REPO}/2025-2026/players.csv")[["player_id", "player_code"]]
pri = pri.merge(p25, on="player_id", how="left")
pri = pri.dropna(subset=["player_code"]).drop_duplicates("player_code")

# ---- build the 26/27 player frame ----------------------------------------
cal = calibrate_cold_start()
pmap = pri.set_index("player_code")
rows, n_carry, n_cold = [], 0, 0
for _, r in d26.iterrows():
    code = r.get("player_code")
    if pd.notna(code) and code in pmap.index:
        s = pmap.loc[code]
        row = {k: s[k] for k in ["npxgi_alpha", "npxgi_beta", "xa_alpha", "xa_beta",
                                 "defcon_alpha", "defcon_beta", "start_a", "start_b",
                                 "sub_app_rate"]}
        row.update({"id": r.player_id, "web_name": r.web_name, "pos": r.pos,
                    "team": r.team, "own": r.selected_by_percent, "cost": r.now_cost,
                    "minutes": s["minutes"], "cold_start": False,
                    "pen_xg90": float(s["pen_xg90_measured"]),
                    "xgot_faced_90": float(s["xgot_faced_90"])})
        rows.append(row); n_carry += 1
    else:
        cs = _coldstart_row(r.web_name, r.team, r.pos, r.now_cost,
                            r.selected_by_percent, cal)
        cs.update({"minutes": 0.0, "pen_xg90": 0.0, "xgot_faced_90": np.nan})
        rows.append(cs); n_cold += 1
players = pd.DataFrame(rows)
print(f"26/27 frame: {len(players)} players | carried forward {n_carry} | cold-start {n_cold}")

# availability + set-piece layer (real 26/27 flags)
players = sg.apply_availability(players, sig)
# set-piece: keep the MEASURED penalty rate for designated takers, else zero it
sigi = sig.set_index(sig.name.str.lower().str.strip())
pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())
players["pen_xg90"] = np.where(
    players.web_name.str.lower().str.strip().isin(pen1),
    players.pen_xg90.fillna(0).clip(lower=0.10),   # taker: at least the league norm
    0.0)
print(f"designated penalty takers with a measured rate: {(players.pen_xg90>0).sum()}")

# ---- team model: Elo (all 20) + per-club promoted priors ------------------
elo = ci.to_elo_frame(t26)
pclub = ci.promoted_prior_from_elo(t26)["per_club"]
tm = TeamModel(promoted_per_club=pclub).fit(clubelo=elo, clubelo_weight=0.45)
ts = tm.sample_2627(S=2000)

# ---- project --------------------------------------------------------------
for H in [3, 6, 38]:
    r = project(players, tm, ts, 1, H, S=2000).drop_duplicates(["player", "team"])
    r["ppm"] = r["mean"] / r["cost"].clip(lower=4.0)
    tag = {3: "gw1_3", 6: "gw1_6", 38: "season"}[H]
    r.round(2).to_csv(fos.path.join(config.OUTPUTS, "final_2627_{tag}.csv"), index=False)
    if H == 38:
        continue
    print("\n" + "#"*74)
    print(f"#  GW1-{H} — per-match priors, real roster/prices/availability")
    print("#"*74)
    for pos, n in [("GK", 3), ("DEF", 5), ("MID", 5), ("FWD", 4)]:
        print(f"\n-- {pos} --")
        print(r[r.pos == pos].nlargest(n, "mean")[
            ["player", "team", "cost", "mean", "p5", "p95", "ppm", "own"]
        ].round(2).to_string(index=False))
    print("\n>> CAPTAIN (ceiling):")
    print(r.nlargest(4, "p95")[["player", "pos", "team", "mean", "p95"]].round(1).to_string(index=False))
    print("\n>> VALUE (<=6.5m):")
    print(r[r.cost <= 6.5].nlargest(6, "ppm")[
        ["player", "pos", "team", "cost", "mean", "ppm"]].round(2).to_string(index=False))
