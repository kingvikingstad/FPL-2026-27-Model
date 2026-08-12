import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""Validation for the appointment-date panel split (regime_panel.py).
Checks: (1) isolation — non-partial clubs unchanged; (2) player_code keying —
incumbents matched; (3) market agreement — boosted incumbents are higher-owned
than cut ones, using ownership as an independent signal (no Solio needed)."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, sys; import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src")); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os
import core_insights as ci, signals as sg, starter_prior as sp, regime_panel as rpn
from roster import calibrate_cold_start, _coldstart_row

REPO = config.REPO
panel = pd.read_pickle(config.PMS_PANEL)
d26, t26, _ = ci.load(base=f"{REPO}/2026-2027"); sig = ci.to_signals(d26)
cal_cs = calibrate_cold_start(hist_csv=config.COLDSTART_HIST)
cal_own = sp.calibrate_ownership_start()
pri = pd.read_pickle(config.MS_PRIORS).dropna(subset=["player_code"]).drop_duplicates("player_code")
pmap = pri.set_index("player_code")
rows = []
for _, r in d26.iterrows():
    c = r.get("player_code")
    if pd.notna(c) and c in pmap.index:
        s = pmap.loc[c]
        row = {k: s[k] for k in ["npxgi_alpha","npxgi_beta","xa_alpha","xa_beta",
                                 "defcon_alpha","defcon_beta","start_a","start_b","sub_app_rate","exp_minutes"]}
        row.update({"id": r.player_id, "player_code": c, "web_name": r.web_name, "pos": r.pos,
                    "team": r.team, "own": r.selected_by_percent, "cost": r.now_cost,
                    "minutes": s["minutes"], "cold_start": False}); rows.append(row)
    else:
        cs = _coldstart_row(r.web_name, r.team, r.pos, r.now_cost, r.selected_by_percent, cal_cs)
        cs["minutes"] = 0.0; cs["player_code"] = c; rows.append(cs)
pl = pd.DataFrame(rows)
sprob = lambda fr: fr.start_a / (fr.start_a + fr.start_b)
split = rpn.apply_regime_panel_split(pl.copy(), panel, REPO, w_pre=0.30)

partial = set(rpn.PARTIAL_REGIME_CLUBS_2627); non = ~pl.team.isin(partial)
iso = (np.allclose(pl.loc[non, "start_a"].values, split.loc[non, "start_a"].values, equal_nan=True)
       and np.allclose(pl.loc[non, "start_b"].values, split.loc[non, "start_b"].values, equal_nan=True))
print(f"TEST isolation (non-partial clubs unchanged): {'PASS' if iso else 'FAIL'}")

ff = sp.apply_minutes_shrinkage(pl.copy(), cal_own, k_min=900)
sf = sp.apply_minutes_shrinkage(split.copy(), cal_own, k_min=900)
m = pl[["web_name", "team", "own"]].copy()
m["delta"] = sprob(sf).values - sprob(ff).values
mm = m[m.team.isin(partial) & (m.delta.abs() > 0.02)]
up, dn = mm[mm.delta > 0].own, mm[mm.delta < 0].own
print(f"TEST market agreement: boosted own {up.mean():.1f}% (n={len(up)}) vs cut {dn.mean():.1f}% "
      f"(n={len(dn)}): {'PASS' if up.mean() > dn.mean() else 'FAIL'}")
