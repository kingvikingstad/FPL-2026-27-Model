import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
run_final_board.py — canonical GW1-H board with all validated improvements:
two-season priors -> cold-start depth prior -> evidence-weighted minutes shrinkage
-> (optional) appointment panel split / regime variance -> availability -> pen
assignment -> DefCon environment conditioning -> project (defcon_ev/cs_ev surfaced).

Env flags: REGIME_PANEL=on, REGIME=proposed, DEFCON_ENV=off (default on).
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd, sys; import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src")); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import core_insights as ci, bayes_model, signals as sg, starter_prior as sp, defcon_env as de
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
from schedule_2627 import schedule

REPO = config.REPO
GW_LO, GW_HI, S = 1, 6, 3000
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
                                 "defcon_alpha","defcon_beta","start_a","start_b","sub_app_rate"]}
        row.update({"id": r.player_id, "player_code": c, "web_name": r.web_name, "pos": r.pos,
                    "team": r.team, "own": r.selected_by_percent, "cost": r.now_cost,
                    "minutes": s["minutes"], "cold_start": False,
                    "pen_xg90": float(s.get("pen_xg90_measured", 0) or 0)}); rows.append(row)
    else:
        cs = _coldstart_row(r.web_name, r.team, r.pos, r.now_cost, r.selected_by_percent, cal_cs)
        cs["minutes"] = 0.0; cs["player_code"] = c; rows.append(cs)
pl = pd.DataFrame(rows)

pl = sp.apply_coldstart_depth(pl, cal_own)
if os.environ.get("REGIME_PANEL") == "on":
    import regime_panel as rpn
    pl = rpn.apply_regime_panel_split(pl, pd.read_pickle(config.PMS_PANEL), REPO, w_pre=0.30)
_regime = sp.REGIME_2627_PROPOSED if os.environ.get("REGIME") == "proposed" else {}
pl = sp.apply_regime_uncertainty(pl, regime=_regime, cal=cal_own, k_min=900)
pl = sg.apply_availability(pl, sig, lineups=None)
pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())
pl["pen_xg90"] = np.where(pl.web_name.str.lower().str.strip().isin(pen1),
                          pl.pen_xg90.fillna(0).clip(lower=0.10), 0.0)

elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
if os.environ.get("MARKET_ODDS") != "off":
    import market_odds as mo
    elo = mo.blended_elo(elo, market_weight=float(os.environ.get("MARKET_WEIGHT", "0.6")))
    print(f"[market-odds] blended betting-odds team strength into ClubElo "
          f"(weight {os.environ.get('MARKET_WEIGHT','0.6')})")
tm = TeamModel(promoted_per_club=pclub).fit(e0_path=config.E0_RECON, clubelo=elo, clubelo_weight=0.45)
bayes_model.rng = np.random.default_rng(7); ts = tm.sample_2627(S=S)

# projected xGA per team over the horizon (drives DefCon CBIT environment)
idx = ts["idx"]; mu = ts["mu"]; home = ts["home"]; A = ts["att"]; D = ts["dfn"]
sched, long = schedule(); win = long[(long.gameweek >= GW_LO) & (long.gameweek <= GW_HI)]
xga27 = {}
for t, g in win.groupby("team"):
    if t not in idx: continue
    xga27[t] = float(np.mean([np.exp(mu + (0.0 if r.is_home else home) + A[:, idx[r.opp]] - D[:, idx[t]]).mean()
                              for _, r in g.iterrows() if r.opp in idx]))

if os.environ.get("DEFCON_ENV") != "off":
    pl = de.apply_defcon_environment(pl, xga27, REPO)

bayes_model.rng = np.random.default_rng(7)
board = project(pl, tm, ts, GW_LO, GW_HI, S=S).drop_duplicates("id")
board["ppm"] = board["mean"] / board["cost"].clip(lower=4.0)
board.round(2).to_csv(os.path.join(config.OUTPUTS, "decision_gw1_6_defconenv.csv"), index=False)

print("=" * 78)
print(f"CANONICAL BOARD GW{GW_LO}-{GW_HI} (defcon_ev + cs_ev surfaced; DefCon env conditioning on)")
print("=" * 78)
cols = ["player", "pos", "team", "cost", "mean", "defcon_ev", "cs_ev", "p95", "own"]
print(board.nlargest(15, "mean")[cols].round(2).to_string(index=False))
print("\nTop DefCon assets (defcon_ev, fitted priors only, sub-8% owned) — the £4.5-5.5 strategy:")
dc = board[(board.pos.isin(["DEF", "MID"])) & (board.own < 8) & (board.cost <= 5.5)]
print(dc.nlargest(12, "defcon_ev")[["player", "pos", "team", "cost", "own", "defcon_ev", "mean"]].round(2).to_string(index=False))
