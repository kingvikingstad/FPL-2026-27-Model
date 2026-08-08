"""
gw_board.py — week-by-week per-player predictions, GW1..GW_HI.
==============================================================
Produces TRUE per-gameweek projections (project(gw,gw) per week, not a horizon
aggregate split proportionally), on the full validated stack: two-season priors,
depth prior, minutes shrinkage, availability, betting-odds team strength
(market_odds, embedded snapshot), DefCon environment (xGA + press index).

Solio weighting (offline): the cached Solio feed (data/solio_cache.md) carries
their GW1 projections. For matched players, GW1 is blended
    blend = w_ours * ours + (1 - w_ours) * solio        (default w_ours=0.5;
set SOLIO_W_OURS, or from inverse-MAE once results exist). Later GWs are ours
alone — Solio publishes single-GW only. Blended cells are flagged.

Outputs (in config.OUTPUTS):
  gw_board_long.csv  player x gw rows: mean, sd, p5..p95, defcon_ev, cs_ev, solio, blend
  gw_board_wide.csv  one row per player, gw1..gwN blended means + total
Env: GW_HI (default 10), SOLIO_W_OURS (default 0.5), SOLIO=off, MARKET_ODDS=off,
     MARKET_WEIGHT, DEFCON_ENV=off, REGIME_PANEL=on, REGIME=proposed.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import core_insights as ci, bayes_model, signals as sg, starter_prior as sp, defcon_env as de
import solio_ensemble as se
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
from schedule_2627 import schedule

GW_HI = int(os.environ.get("GW_HI", "10"))
S = int(os.environ.get("DRAWS", "3000"))
W_OURS = float(os.environ.get("SOLIO_W_OURS", "0.5"))

# ---------- frame (identical to run_final_board) ----------
d26, t26, _ = ci.load(base=config.repo("2026-2027")); sig = ci.to_signals(d26)
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
    pl = rpn.apply_regime_panel_split(pl, pd.read_pickle(config.PMS_PANEL), config.REPO, w_pre=0.30)
_regime = sp.REGIME_2627_PROPOSED if os.environ.get("REGIME") == "proposed" else {}
pl = sp.apply_regime_uncertainty(pl, regime=_regime, cal=cal_own, k_min=900)
pl = sg.apply_availability(pl, sig, lineups=None)
pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())
pl["pen_xg90"] = np.where(pl.web_name.str.lower().str.strip().isin(pen1),
                          pl.pen_xg90.fillna(0).clip(lower=0.10), 0.0)

# ---------- team model with betting-odds strength ----------
elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
if os.environ.get("MARKET_ODDS") != "off":
    import market_odds as mo
    elo = mo.blended_elo(elo, market_weight=float(os.environ.get("MARKET_WEIGHT", "0.6")))
    print(f"[market-odds] betting-odds team strength blended (weight {os.environ.get('MARKET_WEIGHT','0.6')})")
tm = TeamModel(promoted_per_club=pclub).fit(e0_path=config.E0_RECON, clubelo=elo, clubelo_weight=0.45)
bayes_model.rng = np.random.default_rng(7); ts = tm.sample_2627(S=S)

# DefCon environment (xGA over the full window + press index)
idx = ts["idx"]; mu = ts["mu"]; home = ts["home"]; A = ts["att"]; D = ts["dfn"]
sched, long = schedule(); win = long[long.gameweek <= GW_HI]
xga27 = {}
for t, g in win.groupby("team"):
    if t not in idx: continue
    xga27[t] = float(np.mean([np.exp(mu + (0.0 if r.is_home else home) + A[:, idx[r.opp]] - D[:, idx[t]]).mean()
                              for _, r in g.iterrows() if r.opp in idx]))
if os.environ.get("DEFCON_ENV") != "off":
    pl = de.apply_defcon_environment(pl, xga27, config.REPO)

# ---------- true per-GW projections ----------
frames = []
for gw in range(1, GW_HI + 1):
    bayes_model.rng = np.random.default_rng(7)
    g = project(pl, tm, ts, gw, gw, S=S).drop_duplicates("id")
    g["gw"] = gw; frames.append(g)
    print(f"[gw {gw}] projected {len(g)} players (top: {g.iloc[0].player} {g.iloc[0]['mean']:.2f})")
long_df = pd.concat(frames, ignore_index=True)

# ---------- Solio blend (GW1, offline from the cached feed) ----------
long_df["solio"] = np.nan; long_df["blended"] = long_df["mean"]; long_df["src"] = "model"
if os.environ.get("SOLIO") != "off":
    try:
        sol = se.parse_solio(se.fetch_solio())
        Sp = sol["players"]; sgw = sol.get("gameweek")
        if sgw is not None and 1 <= sgw <= GW_HI:
            g1 = long_df[long_df.gw == sgw]
            al = se.align_single_gw(g1.rename(columns={"mean": "mean"}), Sp)
            key2solio = al.set_index(["player", "team"])["solio_proj"].to_dict()
            mask = long_df.gw == sgw
            for i in long_df[mask].index:
                k = (long_df.at[i, "player"], long_df.at[i, "team"])
                if k in key2solio and pd.notna(key2solio[k]):
                    long_df.at[i, "solio"] = key2solio[k]
                    long_df.at[i, "blended"] = W_OURS * long_df.at[i, "mean"] + (1 - W_OURS) * key2solio[k]
                    long_df.at[i, "src"] = f"blend(w_ours={W_OURS})"
            n = long_df.loc[mask, "solio"].notna().sum()
            print(f"[solio] GW{sgw}: blended {n} matched players from the cached feed "
                  f"(w_ours={W_OURS}); other GWs are model-only")
        else:
            print(f"[solio] cached feed is GW{sgw} — outside window, no blend")
    except Exception as e:
        print(f"[solio] unavailable ({e}); model-only")

# ---------- outputs ----------
cols = ["player", "pos", "team", "cost", "own", "gw", "mean", "solio", "blended", "src",
        "sd", "defcon_ev", "cs_ev", "p5", "median", "p95"]
long_out = long_df[[c for c in cols if c in long_df.columns]].round(3)
long_out.to_csv(os.path.join(config.OUTPUTS, "gw_board_long.csv"), index=False)

wide = long_df.pivot_table(index=["player", "pos", "team", "cost", "own"],
                           columns="gw", values="blended", aggfunc="first")
wide.columns = [f"gw{int(c)}" for c in wide.columns]
wide["total"] = wide.sum(axis=1)
wide = wide.reset_index().sort_values("total", ascending=False)
wide.round(2).to_csv(os.path.join(config.OUTPUTS, "gw_board_wide.csv"), index=False)

print("\n" + "=" * 100)
print(f"WEEK-BY-WEEK BOARD  GW1-{GW_HI}  (GW1 Solio-blended where matched; betting-odds team strength on)")
print("=" * 100)
show = ["player", "pos", "team", "cost"] + [f"gw{g}" for g in range(1, min(GW_HI, 8) + 1)] + ["total"]
print(wide.head(15)[show].round(2).to_string(index=False))
print(f"\nwrote gw_board_long.csv ({len(long_out)} rows) and gw_board_wide.csv ({len(wide)} players) to {config.OUTPUTS}")
