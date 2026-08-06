"""
edge_study.py — what in gameweek t predicts gameweek t+1?
==========================================================
Captaincy is a HAUL problem, not a mean problem: you double the score, so what
matters is P(big return), and the tail is where captaincy is won or lost. So the
target here is P(points >= 10) next gameweek, alongside mean points.

Strictly leak-free: every feature is measured in gameweek t, every target in
gameweek t+1, and nothing from t+1 touches the features.

Hypotheses tested
  H1  raw xG/xGOT in t predicts t+1        (process carries over)
  H2  ACTUAL points in t predicts t+1      ("form" — probably weak)
  H3  xG UNDERperformance in t predicts a bounce in t+1 (mean reversion)
  H4  touches in opposition box / big chances missed carry signal
  H5  minutes stability matters more than any attacking stat
  H6  ownership/price momentum (transfers in) adds anything
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
import sys; sys.path.insert(0, "/home/claude/fpl")
from fpl_xp_model import ols_robust

p = pd.read_pickle("/tmp/pms_panel.pkl").sort_values(["player_id", "gameweek"])

# collapse to player-gameweek (doubles handled by summing)
agg = p.groupby(["player_id", "web_name", "pos", "team", "gameweek"], dropna=False).agg(
    mins=("mins", "sum"), npxg=("npxg", "sum"), xa=("xa_", "sum"), xgot=("xgot_", "sum"),
    np_goals=("np_goals", "sum"), assists=("assists", "sum"), shots=("total_shots", "sum"),
    sot=("shots_on_target", "sum"), tob=("tob", "sum"), cc=("chances_created", "sum"),
    bcm=("big_chances_missed", "sum"), defcon=("defcon_raw", "sum"),
    cbi=("cbi", "sum"), pts=("total_points", "first"),
).reset_index()

g = agg.groupby("player_id")
agg["pts_next"] = g["pts"].shift(-1)
agg["mins_next"] = g["mins"].shift(-1)
agg["haul_next"] = (agg.pts_next >= 10).astype(float)
agg["ret_next"] = (agg.pts_next >= 6).astype(float)
# rolling context (leak-free, strictly before t+1)
agg["mins_r3"] = g["mins"].transform(lambda s: s.rolling(3, min_periods=1).mean())
agg["npxg_r5"] = g["npxg"].transform(lambda s: s.rolling(5, min_periods=2).mean())
agg["pts_r5"] = g["pts"].transform(lambda s: s.rolling(5, min_periods=2).mean())
agg["xgi"] = agg.npxg + agg.xa
agg["xgi_r5"] = g["xgi"].transform(lambda s: s.rolling(5, min_periods=2).mean())
# H3: under/over-performance in t
agg["gminusxg"] = agg.np_goals - agg.npxg
agg["gminusxg_r5"] = g["gminusxg"].transform(lambda s: s.rolling(5, min_periods=2).mean())

d = agg[agg.pts_next.notna() & (agg.mins >= 45)].copy()
att = d[d.pos.isin(["MID", "FWD"])].copy()
print(f"sample: {len(d)} player-gameweeks with a following GW ({len(att)} MID/FWD)")
print(f"base rates: P(haul>=10 next) = {d.haul_next.mean():.4f} | "
      f"P(return>=6 next) = {d.ret_next.mean():.4f}")

# ---------------------------------------------------------------------------
print("\n" + "="*78)
print("SINGLE-FEATURE PREDICTIVE POWER (feature in GW t -> outcome in GW t+1)")
print("="*78)
FEATS = [("minutes in t", "mins"), ("rolling minutes (3)", "mins_r3"),
         ("npxG in t", "npxg"), ("xGOT in t", "xgot"), ("xG+xA in t", "xgi"),
         ("rolling xG+xA (5)", "xgi_r5"), ("rolling npxG (5)", "npxg_r5"),
         ("shots in t", "shots"), ("SoT in t", "sot"),
         ("touches in box in t", "tob"), ("chances created in t", "cc"),
         ("big chances missed in t", "bcm"),
         ("POINTS in t (form)", "pts"), ("rolling points (5)", "pts_r5"),
         ("goals minus xG in t", "gminusxg"), ("rolling G-xG (5)", "gminusxg_r5")]
rows = []
for lab, c in FEATS:
    s = att[[c, "pts_next", "haul_next"]].dropna()
    if len(s) < 300: continue
    rows.append({"feature": lab,
                 "rho_pts": stats.spearmanr(s[c], s.pts_next).correlation,
                 "rho_haul": stats.spearmanr(s[c], s.haul_next).correlation,
                 "n": len(s)})
r = pd.DataFrame(rows).sort_values("rho_haul", ascending=False)
print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------------------------------------
print("\n" + "="*78)
print("H3 — MEAN REVERSION: does UNDERperforming xG in t predict a bounce in t+1?")
print("="*78)
a = att[att.npxg >= 0.3].copy()          # players who actually had chances
a["bucket"] = pd.cut(a.gminusxg, [-9, -0.5, -0.15, 0.15, 0.5, 9],
                     labels=["big under", "under", "in line", "over", "big over"])
print(a.groupby("bucket", observed=True).agg(
    n=("pts_next", "size"), mean_pts_next=("pts_next", "mean"),
    P_haul_next=("haul_next", "mean"), npxg_t=("npxg", "mean")).round(3).to_string())
u = a[a.gminusxg <= -0.5]; o = a[a.gminusxg >= 0.5]
if len(u) > 50 and len(o) > 50:
    t_, pv = stats.ttest_ind(u.pts_next, o.pts_next, equal_var=False)
    print(f"\nunderperformers vs overperformers, next-GW points: "
          f"{u.pts_next.mean():.2f} vs {o.pts_next.mean():.2f} (t={t_:.2f}, p={pv:.4f})")

# controlled: does G-xG add anything ONCE you know the underlying xG rate?
s = a.dropna(subset=["xgi_r5", "gminusxg", "pts_next"])
X = np.column_stack([np.ones(len(s)), s.xgi_r5, s.gminusxg, s.mins_r3])
o2 = ols_robust(X, s.pts_next.values, ["const", "rolling xG+xA", "G-xG in t", "rolling mins"])
print("\nControlled OLS (HC1) — next-GW points:")
for n_, b, se, pv in zip(o2.names, o2.beta, o2.se, o2.pval):
    print(f"   {n_:16s} {b:+7.4f} (se {se:.4f}, p={pv:.4f})")

# ---------------------------------------------------------------------------
print("\n" + "="*78)
print("HAUL MODEL — what actually drives P(>=10 next GW)?")
print("="*78)
MOD = ["mins_r3", "xgi_r5", "npxg_r5", "xgot", "tob", "shots", "sot", "cc", "bcm",
       "pts_r5", "gminusxg_r5"]
s = att.dropna(subset=MOD + ["haul_next"])
# walk-forward AUC-style check
from sklearn.metrics import roc_auc_score
aucs = []
for t in range(12, 38):
    tr = s[s.gameweek < t]; te = s[s.gameweek == t]
    if len(te) < 40 or te.haul_next.nunique() < 2: continue
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.06)
    m.fit(tr[MOD].values, tr.haul_next.values)
    aucs.append(roc_auc_score(te.haul_next, m.predict(te[MOD].values)))
print(f"walk-forward AUC for P(haul next GW): {np.mean(aucs):.4f}  (0.5 = no skill, {len(aucs)} GWs)")

X = np.column_stack([np.ones(len(s))] + [s[c].values for c in MOD])
o3 = ols_robust(X, s.haul_next.values, ["const"] + MOD)
sig = pd.DataFrame({"feature": o3.names, "coef": o3.beta, "p": o3.pval}).iloc[1:]
sig = sig.reindex(sig.coef.abs().sort_values(ascending=False).index)
print("\nLinear probability model for haul (HC1), by |coef|:")
print(sig.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
</content>
