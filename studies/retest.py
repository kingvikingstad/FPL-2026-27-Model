import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
retest.py — rebuild the rate models on per-match data and retest every claim
============================================================================
Everything is leak-free: features for gameweek t are rolling means over matches
strictly before t, and all correlations are against NEXT-gameweek outcomes.

Tests:
  A. attacking signal — true npxG / xGOT / touches-in-box vs FPL's aggregate xG
     and vs realised goals (retests the "trust xG, drop goals" finding)
  B. defensive contribution — can the separate CBI components predict the FPL
     DefCon award better than the aggregate?
  C. goalkeepers — does `goals_prevented` add over saves?
  D. head-to-head: per-match rate model vs FPL-aggregate rate model, on
     next-gameweek points, walk-forward
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from fpl_xp_model import ols_robust

p = pd.read_pickle(config.PMS_PANEL).sort_values(["player_id", "gameweek"])

# ---------------------------------------------------------------------------
# rolling leak-free per-90 rates from PER-MATCH data
# ---------------------------------------------------------------------------
RATE_COLS = {
    "npxg": "npxg", "xa": "xa_", "xgot": "xgot_", "tob": "tob",
    "shots": "total_shots", "sot": "shots_on_target", "cc": "chances_created",
    "bcm": "big_chances_missed", "dribbles": "successful_dribbles",
    "cbi": "cbi", "tkl": "tkl", "rec": "rec", "defcon": "defcon_raw",
    "saves": "saves_", "gp": "goals_prevented", "xgotf": "xgot_faced",
    "goals": "goals", "npgoals": "np_goals", "assists": "assists",
}
g = p.groupby("player_id")
p["cum_min"] = g["mins"].apply(lambda s: s.shift(1).cumsum()).values
for name, col in RATE_COLS.items():
    p[f"cum_{name}"] = g[col].apply(lambda s: s.shift(1).cumsum()).values
    p[f"r_{name}"] = p[f"cum_{name}"] / (p["cum_min"] / 90.0).replace(0, np.nan)
p["prior_apps"] = g["mins"].apply(lambda s: (s.shift(1) > 0).cumsum()).values

d = p[(p.prior_apps >= 4) & (p.cum_min >= 270) & p.total_points.notna()].copy()
print(f"modelling rows: {len(d)}  players: {d.player_id.nunique()}  GW {d.gameweek.min()}-{d.gameweek.max()}")

# ===========================================================================
print("\n" + "#"*76)
print("# A. ATTACKING SIGNAL — retesting 'trust xG, drop goals' with TRUE npxG")
print("#"*76)
att = d[d.pos.isin(["MID", "FWD"])].copy()
att["att_pts"] = att.np_goals * np.where(att.pos == "FWD", 4, 5) + att.assists * 3
cands = [("rolling npxG/90 (true, pens removed)", "r_npxg"),
         ("rolling xG/90 (incl. pens)", None),
         ("rolling xGOT/90", "r_xgot"),
         ("rolling touches-in-box/90", "r_tob"),
         ("rolling shots/90", "r_shots"),
         ("rolling SoT/90", "r_sot"),
         ("rolling big-chances-missed/90", "r_bcm"),
         ("rolling ACTUAL np goals/90", "r_npgoals"),
         ("rolling xA/90", "r_xa"),
         ("rolling chances-created/90", "r_cc")]
att["r_xg_all"] = (att.cum_npxg + 0.79 * 0) / (att.cum_min / 90.0)  # placeholder
res = []
for lab, col in cands:
    if col is None:
        continue
    s = att[[col, "att_pts"]].dropna()
    if len(s) < 300:
        continue
    res.append({"signal": lab, "spearman": stats.spearmanr(s[col], s.att_pts).correlation, "n": len(s)})
r = pd.DataFrame(res).sort_values("spearman", ascending=False)
print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

s = att[["r_npxg", "r_npgoals", "r_xgot", "att_pts"]].dropna()
X = np.column_stack([np.ones(len(s)), s.r_npxg, s.r_npgoals, s.r_xgot])
o = ols_robust(X, s.att_pts.values, ["const", "npxG/90", "actual npG/90", "xGOT/90"])
print("\nJoint OLS (does actual finishing add over expected?), HC1 SEs:")
for n_, b, se, pv in zip(o.names, o.beta, o.se, o.pval):
    print(f"   {n_:16s} {b:+7.3f} (se {se:.3f}, p={pv:.4f})")

# ===========================================================================
print("\n" + "#"*76)
print("# B. DEFENSIVE CONTRIBUTION — separate components vs the aggregate")
print("#"*76)
dc = d[d.pos.isin(["DEF", "MID", "FWD"])].copy()
thr = np.where(dc.pos == "DEF", 10, 12)
dc["hit"] = (dc.defcon_raw >= thr).astype(int)
print(f"DefCon award rate: {dc.hit.mean():.3f}  (n={len(dc)})")
for lab, col in [("rolling aggregate defcon/90", "r_defcon"), ("rolling CBI/90", "r_cbi"),
                 ("rolling tackles/90", "r_tkl"), ("rolling recoveries/90", "r_rec")]:
    s = dc[[col, "hit"]].dropna()
    print(f"  {lab:30s} Spearman(next-match DefCon hit)={stats.spearmanr(s[col], s.hit).correlation:.4f}")
sub = dc.dropna(subset=["r_cbi", "r_tkl", "r_rec", "r_defcon"])
X = np.column_stack([np.ones(len(sub)), sub.r_cbi, sub.r_tkl, sub.r_rec])
o = ols_robust(X, sub.hit.values, ["const", "CBI/90", "tackles/90", "recoveries/90"])
print("\n  Components jointly (LPM, HC1):")
for n_, b, se, pv in zip(o.names, o.beta, o.se, o.pval):
    print(f"   {n_:16s} {b:+7.4f} (se {se:.4f}, p={pv:.4f})")
print(f"   R^2={o.r2:.3f}")

# ===========================================================================
print("\n" + "#"*76)
print("# C. GOALKEEPERS — does goals_prevented add over saves?")
print("#"*76)
gk = d[d.pos == "GK"].copy()
gk["gk_pts"] = gk.total_points
for lab, col in [("rolling saves/90", "r_saves"), ("rolling goals_prevented/90", "r_gp"),
                 ("rolling xGOT faced/90", "r_xgotf")]:
    s = gk[[col, "gk_pts"]].dropna()
    if len(s) > 100:
        print(f"  {lab:32s} Spearman(next-GW pts)={stats.spearmanr(s[col], s.gk_pts).correlation:+.4f}  n={len(s)}")
sub = gk.dropna(subset=["r_saves", "r_gp", "r_xgotf"])
if len(sub) > 100:
    X = np.column_stack([np.ones(len(sub)), sub.r_saves, sub.r_gp, sub.r_xgotf])
    o = ols_robust(X, sub.gk_pts.values, ["const", "saves/90", "goals_prevented/90", "xGOT faced/90"])
    print("\n  Joint OLS (HC1):")
    for n_, b, se, pv in zip(o.names, o.beta, o.se, o.pval):
        print(f"   {n_:22s} {b:+7.3f} (se {se:.3f}, p={pv:.4f})")

# ===========================================================================
print("\n" + "#"*76)
print("# D. HEAD-TO-HEAD — per-match rate model vs FPL-aggregate rate model")
print("#"*76)
h = pd.read_csv(config.FPL_DATA_STATS).sort_values(["id", "gameweek"])
gh = h.groupby("id")
h["cm"] = gh["minutes"].apply(lambda s: s.shift(1).cumsum()).values
for c, nm in [("expected_goals", "fx_xg"), ("expected_assists", "fx_xa"),
              ("defensive_contribution", "fx_dc"), ("total_shots", "fx_sh")]:
    h[nm] = gh[c].apply(lambda s: s.shift(1).cumsum()).values / (h["cm"] / 90.0).replace(0, np.nan)
h = h.rename(columns={"id": "player_id"})
merged = d.merge(h[["player_id", "gameweek", "fx_xg", "fx_xa", "fx_dc", "fx_sh"]],
                 on=["player_id", "gameweek"], how="inner")
merged["is_gk"] = (merged.pos == "GK").astype(int)
merged["is_def"] = (merged.pos == "DEF").astype(int)
merged["is_mid"] = (merged.pos == "MID").astype(int)

PMS = ["r_npxg", "r_xa", "r_xgot", "r_tob", "r_shots", "r_sot", "r_cbi", "r_tkl",
       "r_rec", "r_saves", "r_gp", "is_gk", "is_def", "is_mid"]
FPL = ["fx_xg", "fx_xa", "fx_dc", "fx_sh", "is_gk", "is_def", "is_mid"]

def walk(feats, name):
    sp, mae = [], []
    for t in range(10, 39):
        tr = merged[merged.gameweek < t].dropna(subset=feats + ["total_points"])
        te = merged[merged.gameweek == t].dropna(subset=feats + ["total_points"])
        if len(te) < 30 or len(tr) < 300:
            continue
        m = HistGradientBoostingRegressor(max_iter=250, max_depth=4, learning_rate=0.05,
                                          l2_regularization=1.0)
        m.fit(tr[feats].values, tr.total_points.values)
        pr = m.predict(te[feats].values)
        sp.append(stats.spearmanr(te.total_points, pr).correlation)
        mae.append(np.abs(te.total_points - pr).mean())
    return {"model": name, "Spearman": np.nanmean(sp), "MAE": np.mean(mae), "gws": len(sp)}

out = [walk(FPL, "FPL aggregates (old)"), walk(PMS, "per-match stats (new)"),
       walk(list(dict.fromkeys(PMS + FPL)), "both combined")]
print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
