import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "scripts"))
import config
import os
"""
sweep_older_weight.py — how sensitive is the GW1-6 board to the between-season
weight? Rebuilds two-season priors across a grid of older_weight, holding the
team model, RNG seed, and depth prior fixed so every difference is the prior.
older_weight = 0.0 is single-season (25/26 only); 0.5 is current; 1.0 is equal.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys; import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src")); _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "scripts"))
import core_insights as ci, multiseason_priors as ms
import bayes_model
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
import signals as sg, starter_prior as sp

REPO = config.REPO
# Evidence, not a pipeline artifact: this sweep answers "how sensitive is the board
# to older_weight" once, and the answer is the justification for 0.5 in
# multiseason_priors. It lived in outputs/ and was registered in the manifest until
# 2026-09-08, where its real inputs include the daily feed and it therefore reported
# STALE on every run — a permanently red check nobody reads. It belongs with the
# other 23 study CSVs instead, beside the code that produced it.
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "older_weight_sweep.csv")
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
S = 3000
SEED = 7

d26, t26, _ = ci.load(base=f"{REPO}/2026-2027"); sig = ci.to_signals(d26)
cal_cs = calibrate_cold_start(hist_csv=config.COLDSTART_HIST)
cal_own = sp.calibrate_ownership_start()
pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())

# team model fixed across the whole sweep
elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
tm = TeamModel(promoted_per_club=pclub).fit(e0_path=config.E0_RECON, clubelo=elo, clubelo_weight=0.45)
bayes_model.rng = np.random.default_rng(SEED)      # fixed team draws
ts = tm.sample_2627(S=S)


def frame_for(weight):
    pri = ms.to_priors(ms.two_season_evidence(older_weight=weight))
    pri = pri.dropna(subset=["player_code"]).drop_duplicates("player_code")
    pmap = pri.set_index("player_code"); rows = []; carried = 0
    for _, r in d26.iterrows():
        c = r.get("player_code")
        if pd.notna(c) and c in pmap.index:
            s = pmap.loc[c]
            row = {k: s[k] for k in ["npxgi_alpha","npxgi_beta","xa_alpha","xa_beta",
                                     "defcon_alpha","defcon_beta","start_a","start_b","sub_app_rate","exp_minutes"]}
            row.update({"id": r.player_id, "web_name": r.web_name, "pos": r.pos, "team": r.team,
                        "own": r.selected_by_percent, "cost": r.now_cost, "cold_start": False,
                        "pen_xg90": float(s.get("pen_xg90_measured", 0) or 0)})
            rows.append(row); carried += 1
        else:
            rows.append(_coldstart_row(r.web_name, r.team, r.pos, r.now_cost, r.selected_by_percent, cal_cs))
    pl = pd.DataFrame(rows)
    pl = sp.apply_coldstart_depth(pl, cal_own)
    pl = sg.apply_availability(pl, sig, lineups=None)
    pl["pen_xg90"] = np.where(pl.web_name.str.lower().str.strip().isin(pen1),
                              pl.pen_xg90.fillna(0).clip(lower=0.10), 0.0)
    return pl, carried


means = {}; carried_n = {}
for w in GRID:
    pl, carried = frame_for(w)
    bayes_model.rng = np.random.default_rng(SEED)      # identical MC draws per config
    r = project(pl, tm, ts, 1, 6, S=S).drop_duplicates("id").set_index("id")
    means[w] = r["mean"]; carried_n[w] = carried
    means.setdefault("_meta", r[["player", "pos", "team", "cost", "own"]])

meta = means.pop("_meta")
M = pd.DataFrame(means)
M = meta.join(M)
M["range"] = M[GRID].max(axis=1) - M[GRID].min(axis=1)

print("carried-forward count by older_weight:",
      {w: carried_n[w] for w in GRID})

print("\n" + "=" * 78)
print("RANK STABILITY of the top board across older_weight")
print("=" * 78)
top50 = M.nlargest(50, 0.5).index
for w in GRID:
    rho = M.loc[top50, w].rank().corr(M.loc[top50, 0.5].rank(), method="spearman")
    j = len(set(M.nlargest(15, w).index) & set(M.nlargest(15, 0.5).index))
    print(f"  w={w}:  Spearman vs w=0.5 (top50) = {rho:.3f}   top-15 overlap = {j}/15")

print("\n" + "=" * 78)
print("MOST SENSITIVE players (largest EV range across the grid)")
print("=" * 78)
sens = M.nlargest(12, "range")
print(sens[["player", "pos", "team", "cost", "own"] + GRID + ["range"]].round(2).to_string(index=False))

print("\n" + "=" * 78)
print("ISAK — the headline prior call")
print("=" * 78)
isak = M[M.player == "Isak"]
print(isak[["player", "team", "cost", "own"] + GRID + ["range"]].round(2).to_string(index=False))

print("\n" + "=" * 78)
print("TOP 12 at the extremes: single-season (w=0.0) vs equal-weight (w=1.0)")
print("=" * 78)
comp = M.assign(single=M[0.0], w05=M[0.5], equal=M[1.0])
print("w=0.0:", list(comp.nlargest(12, "single").player))
print("w=0.5:", list(comp.nlargest(12, "w05").player))
print("w=1.0:", list(comp.nlargest(12, "equal").player))

M.round(3).to_csv(OUT, index=False)
