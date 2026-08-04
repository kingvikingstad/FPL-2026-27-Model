"""
cs_fixtures.py — most advantageous fixtures for defender clean sheets, GW1-10.
Team-level posterior clean-sheet probability per fixture on the validated,
market-calibrated team model. P(CS) = E_draws[exp(-lambda_against)] (integrates
team-strength uncertainty; Poisson zero-goal probability).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys; sys.path.insert(0, "/home/claude/fpl")
import core_insights as ci, bayes_model
from bayes_model import TeamModel
from schedule_2627 import schedule

REPO = "/home/claude/repo/FPL-Core-Insights-main/data"
GW_HI, S = 10, 6000

d26, t26, _ = ci.load(base=f"{REPO}/2026-2027")
elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
tm = TeamModel(promoted_per_club=pclub).fit(e0_path="/tmp/E0_recon.csv", clubelo=elo, clubelo_weight=0.45)
bayes_model.rng = np.random.default_rng(7); ts = tm.sample_2627(S=S)

idx = ts["idx"]; mu = ts["mu"]; home = ts["home"]; A = ts["att"]; D = ts["dfn"]
sched, long = schedule()
win = long[(long.gameweek >= 1) & (long.gameweek <= GW_HI)]

rows = []
for _, r in win.iterrows():
    t, o = r.team, r.opp
    if t not in idx or o not in idx:
        continue
    ti, oi = idx[t], idx[o]
    hopp = 0.0 if r.is_home else home           # home advantage applies to the OPPONENT if they're home
    lam_against = np.exp(mu + hopp + A[:, oi] - D[:, ti])
    cs = float(np.mean(np.exp(-lam_against)))    # posterior P(opponent scores 0)
    rows.append({"gw": int(r.gameweek), "team": t, "opp": o,
                 "venue": "H" if r.is_home else "A",
                 "xGA": float(lam_against.mean()), "cs_prob": cs})
fx = pd.DataFrame(rows)

# ---- top 10 individual fixtures ----
top = fx.sort_values("cs_prob", ascending=False).head(10).reset_index(drop=True)
print("=" * 68)
print("TOP 10 CLEAN-SHEET FIXTURES FOR DEFENDERS — GW1-10")
print("=" * 68)
top["fixture"] = top.apply(lambda x: f"{x.team} vs {x.opp} ({x.venue})", axis=1)
print(top[["gw", "fixture", "xGA", "cs_prob"]]
      .rename(columns={"xGA": "exp_GA", "cs_prob": "P(CS)"}).round(3).to_string(index=False))

# ---- which defending teams own the most top fixtures ----
print("\n" + "=" * 68)
print("BEST DEFENSIVE TEAMS TO TARGET OVER GW1-10 (mean CS prob, count of strong fixtures)")
print("=" * 68)
agg = (fx.groupby("team")
       .agg(n_fix=("cs_prob", "size"), mean_cs=("cs_prob", "mean"),
            strong_fix=("cs_prob", lambda s: (s >= 0.35).sum()),
            exp_cs_total=("cs_prob", "sum"))
       .sort_values("exp_cs_total", ascending=False).head(10))
print(agg.round(3).to_string())

fx.sort_values(["cs_prob"], ascending=False).round(4).to_csv(
    "/mnt/user-data/outputs/cs_fixtures_gw1_10.csv", index=False)
print("\nfull ranked fixture list -> cs_fixtures_gw1_10.csv")
