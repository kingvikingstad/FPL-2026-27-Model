import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys; import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src")); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import core_insights as ci, bayes_model, signals as sg, starter_prior as sp, defcon_env as de
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
from schedule_2627 import schedule
import os

REPO = config.REPO; S = 4000
d26, t26, _ = ci.load(base=f"{REPO}/2026-2027"); sig = ci.to_signals(d26)
cal_cs = calibrate_cold_start(hist_csv=config.COLDSTART_HIST); cal_own = sp.calibrate_ownership_start()
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
                    "minutes": s["minutes"], "cold_start": False,
                    "pen_xg90": float(s.get("pen_xg90_measured", 0) or 0)}); rows.append(row)
    else:
        cs = _coldstart_row(r.web_name, r.team, r.pos, r.now_cost, r.selected_by_percent, cal_cs)
        cs["minutes"] = 0.0; cs["player_code"] = c; rows.append(cs)
pl = pd.DataFrame(rows)
pl = sp.apply_coldstart_depth(pl, cal_own); pl = sp.apply_minutes_shrinkage(pl, cal_own, k_min=900)
pl = sg.apply_availability(pl, sig, lineups=None)

elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
tm = TeamModel(promoted_per_club=pclub).fit(e0_path=config.E0_RECON, clubelo=elo, clubelo_weight=0.45)
bayes_model.rng = np.random.default_rng(7); ts = tm.sample_2627(S=S)
# 26/27 projected xGA per team, GW1-10
idx = ts["idx"]; mu = ts["mu"]; home = ts["home"]; A = ts["att"]; D = ts["dfn"]
sched, long = schedule(); win = long[long.gameweek <= 10]
xga27 = {}
for t, g in win.groupby("team"):
    if t not in idx: continue
    v = [np.exp(mu + (0.0 if r.is_home else home) + A[:, idx[r.opp]] - D[:, idx[t]]).mean()
         for _, r in g.iterrows() if r.opp in idx]
    xga27[t] = float(np.mean(v))

pl_env = de.apply_defcon_environment(pl.copy(), xga27, REPO)

def run(frame):
    bayes_model.rng = np.random.default_rng(7)
    return project(frame, tm, ts, 1, 6, S=S).drop_duplicates(["player","team"]).set_index(["player","team"])

base = run(pl); env = run(pl_env)
common = base.index.intersection(env.index)
cmp = pd.DataFrame({"player": [i[0] for i in common], "pos": base.loc[common, "pos"], "team": [i[1] for i in common],
                    "dc_base": base.loc[common, "defcon_ev"], "dc_env": env.loc[common, "defcon_ev"]})
cmp["dc_delta"] = cmp.dc_env - cmp.dc_base

print("=" * 66)
print("§4.1(c) DefCon environment conditioning — Anderson natural experiment")
print("=" * 66)
a = cmp[cmp.player == "Anderson"]
print(a[["player", "team", "pos", "dc_base", "dc_env", "dc_delta"]].round(2).to_string(index=False))
print(f"\ndefcon_ev now surfaced on the board: {'YES' if 'defcon_ev' in base.columns else 'NO'} (§4.1a)")

print("\nBiggest DefCon DOWN-corrections (rate too high for the new/low-xGA environment):")
print(cmp.nsmallest(8, "dc_delta")[["player", "pos", "team", "dc_base", "dc_env", "dc_delta"]].round(2).to_string(index=False))
print("\nBiggest DefCon UP-corrections (moved into a higher-xGA environment):")
print(cmp.nlargest(6, "dc_delta")[["player", "pos", "team", "dc_base", "dc_env", "dc_delta"]].round(2).to_string(index=False))
