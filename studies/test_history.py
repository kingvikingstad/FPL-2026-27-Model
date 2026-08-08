import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Validate history.py estimators: simulate 30 seasons with KNOWN parameters
(reversion, between-season sd, promoted-team penalty, home advantage) and check
the estimators recover them."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import history as H

rng = np.random.default_rng(5)
TRUE = dict(revert=0.78, season_sd=0.18, promoted_att=-0.32, promoted_def=-0.30,
            home_adv=0.22, log_mu=np.log(1.35))
N_SEASONS, N_TEAMS = 30, 20

teams = [f"T{i:02d}" for i in range(40)]          # pool incl. promotion/relegation
active = teams[:N_TEAMS]
att = {t: rng.normal(0, 0.30) for t in active}
dfn = {t: rng.normal(0, 0.30) for t in active}
rows = []
for s in range(N_SEASONS):
    code = f"{s:02d}{s+1:02d}"
    for i, h in enumerate(active):
        for a in active:
            if h == a: continue
            lh = np.exp(TRUE["log_mu"] + TRUE["home_adv"] + att[h] - dfn[a])
            la = np.exp(TRUE["log_mu"] + att[a] - dfn[h])
            rows.append({"season": code, "HomeTeam": h, "AwayTeam": a,
                         "FTHG": rng.poisson(lh), "FTAG": rng.poisson(la)})
    # evolve: revert + innovation; relegate worst 3, promote 3 new with a penalty
    for t in active:
        att[t] = TRUE["revert"]*att[t] + rng.normal(0, TRUE["season_sd"])
        dfn[t] = TRUE["revert"]*dfn[t] + rng.normal(0, TRUE["season_sd"])
    strength = {t: att[t]+dfn[t] for t in active}
    worst = sorted(strength, key=strength.get)[:3]
    pool = [t for t in teams if t not in active]
    newcomers = list(rng.choice(pool, 3, replace=False))
    for t in worst: active.remove(t)
    for t in newcomers:
        active.append(t)
        att[t] = rng.normal(TRUE["promoted_att"], 0.15)
        dfn[t] = rng.normal(TRUE["promoted_def"], 0.15)

hist = pd.DataFrame(rows)
print(f"simulated {len(hist)} matches over {hist.season.nunique()} seasons\n")
rat, ha = H.season_ratings(hist, verbose=False)

rev = H.estimate_reversion(rat)
prom = H.estimate_promoted_prior(rat)
home = H.estimate_home_advantage(ha)

print("=== ESTIMATOR RECOVERY (true -> estimated) ===")
out = [
 ("revert (OLS, biased)", TRUE["revert"],   rev["revert_ols"]),
 ("revert (IV-corrected)",TRUE["revert"],   rev["revert"]),
 ("season_sd",         TRUE["season_sd"],    rev["season_sd"]),
 ("promoted_att_mean", TRUE["promoted_att"], prom["promoted_att_mean"]),
 ("promoted_def_mean", TRUE["promoted_def"], prom["promoted_def_mean"]),
 ("home_adv",          TRUE["home_adv"],     home["home_adv_all"]),
]
for name, t, e in out:
    print(f"  {name:20s} true={t:+.3f}  est={e:+.3f}  err={e-t:+.3f}")
print(f"\n  n_pairs={rev['n_pairs']}  n_promoted={prom['n_promoted']}")
print(f"  promoted sd: att={prom['promoted_att_sd']:.3f} def={prom['promoted_def_sd']:.3f}")
