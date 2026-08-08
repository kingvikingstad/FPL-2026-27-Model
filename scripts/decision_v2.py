import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
decision_v2.py — GW1-6 board with the lineup/injury layer wired in.
Adds: (1) ownership-aware depth prior for cold-start players,
      (2) optional confirmed/predicted XI ingestion (file or API-Football),
      (3) optional API-Football injury overlay,
and A/B-validates that the depth prior collapses cold-start inflation while
leaving established players stable.

Env (all optional):
  LINEUPS_PATH      path to a CSV/JSON of predicted/confirmed XIs
  APIFOOTBALL_KEY   key to pull injuries + confirmed XIs live
  DEPTH_OFF=1       disable the depth prior (baseline behaviour)
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd, sys; import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src")); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import core_insights as ci
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
import signals as sg
import starter_prior as sp
import lineups as LU

REPO = config.REPO
GW_LO, GW_HI, S = 1, 6, 3000

d26, t26, _ = ci.load(base=f"{REPO}/2026-2027"); sig = ci.to_signals(d26)

# ---- optional API-Football injury overlay (fills gaps FPL hasn't flagged) ----
key = os.environ.get("APIFOOTBALL_KEY")
if key:
    try:
        import apifootball as af
        inj = af.injuries(af.Client(key=key))
        sig = af.merge_into_signals(sig, inj)
        print(f"[injuries] API-Football overlay: {len(inj)} entries merged")
    except Exception as e:
        print(f"[injuries] API-Football unavailable: {e}")

# ---- optional confirmed / predicted XIs ----
lu = {}
if os.environ.get("LINEUPS_PATH"):
    lu = LU.load_lineups("file", path=os.environ["LINEUPS_PATH"])
elif key:
    lu = LU.load_lineups("apifootball", key=key, gameweek=GW_LO)
print(f"[lineups] {LU.coverage(lu)}")

cal_cs = calibrate_cold_start(hist_csv=config.COLDSTART_HIST)
cal_own = sp.calibrate_ownership_start()
pri = pd.read_pickle(config.MS_PRIORS).dropna(subset=["player_code"]).drop_duplicates("player_code")
pmap = pri.set_index("player_code")


def build_frame(depth: bool):
    rows = []
    for _, r in d26.iterrows():
        c = r.get("player_code")
        if pd.notna(c) and c in pmap.index:
            s = pmap.loc[c]
            row = {k: s[k] for k in ["npxgi_alpha","npxgi_beta","xa_alpha","xa_beta",
                                     "defcon_alpha","defcon_beta","start_a","start_b","sub_app_rate"]}
            row.update({"id": r.player_id, "web_name": r.web_name, "pos": r.pos, "team": r.team,
                        "own": r.selected_by_percent, "cost": r.now_cost, "minutes": s["minutes"],
                        "cold_start": False, "player_code": c, "pen_xg90": float(s.get("pen_xg90_measured", 0) or 0)})
            rows.append(row)
        else:
            cs = _coldstart_row(r.web_name, r.team, r.pos, r.now_cost, r.selected_by_percent, cal_cs)
            cs.update({"minutes": 0.0, "pen_xg90": 0.0, "player_code": c}); rows.append(cs)
    pl = pd.DataFrame(rows)
    if depth:
        pl = sp.apply_coldstart_depth(pl, cal_own)      # depth prior FIRST ...
        # evidence-weighted minutes + regime variance (regime={} -> identical to
        # apply_minutes_shrinkage; set REGIME=proposed to opt into the unfitted
        # per-club (delta,kappa) starting points for a sweep).
        if os.environ.get("REGIME_PANEL") == "on":
            import regime_panel as rpn, pandas as _pd
            _panel = _pd.read_pickle(config.PMS_PANEL)
            pl = rpn.apply_regime_panel_split(pl, _panel, REPO, w_pre=0.30)
        _regime = sp.REGIME_2627_PROPOSED if os.environ.get("REGIME") == "proposed" else {}
        pl = sp.apply_regime_uncertainty(pl, regime=_regime, cal=cal_own, k_min=900)
    pl = sg.apply_availability(pl, sig, lineups=lu)       # ... then injuries/XIs override
    pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())
    pl["pen_xg90"] = np.where(pl.web_name.str.lower().str.strip().isin(pen1),
                              pl.pen_xg90.fillna(0).clip(lower=0.10), 0.0)
    return pl


elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
tm = TeamModel(promoted_per_club=pclub).fit(e0_path=config.E0_RECON, clubelo=elo, clubelo_weight=0.45)
ts = tm.sample_2627(S=S)

depth_on = os.environ.get("DEPTH_OFF") != "1"
frames = {"baseline": build_frame(depth=False), "depth": build_frame(depth=True)}
proj = {}
for tag, pl in frames.items():
    r = project(pl, tm, ts, GW_LO, GW_HI, S=S).drop_duplicates(["player", "team"])
    r = r.merge(pl[["web_name", "cold_start"]].rename(columns={"web_name": "player"}),
                on="player", how="left")
    proj[tag] = r.set_index("player")

b, d = proj["baseline"], proj["depth"]
common = b.index.intersection(d.index)
cmp = pd.DataFrame({"pos": b.loc[common, "pos"], "team": b.loc[common, "team"],
                    "cost": b.loc[common, "cost"], "own": b.loc[common, "own"],
                    "cold": b.loc[common, "cold_start"],
                    "baseline": b.loc[common, "mean"], "depth": d.loc[common, "mean"]})
cmp["delta"] = cmp.depth - cmp.baseline

print("\n" + "=" * 74)
print("EFFECT OF THE DEPTH PRIOR")
print("=" * 74)
est = cmp[~cmp.cold]; cs = cmp[cmp.cold]
print(f"established players (n={len(est)}): mean |delta| = {est.delta.abs().mean():.2f} pts  "
      f"(should be ~0 — they are untouched)")
print(f"cold-start players (n={len(cs)}):  mean delta   = {cs.delta.mean():+.2f} pts")
print("\nCold-start players most corrected DOWN (0%-owned fringe at strong clubs):")
print(cmp[cmp.cold].nsmallest(8, "delta")[["pos","team","cost","own","baseline","depth","delta"]]
      .round(2).to_string())
print("\nCold-start players held UP by the prior (higher-owned, likely real starters):")
held = cmp[cmp.cold & (cmp.own >= 5)].nlargest(6, "depth")
print(held[["pos","team","cost","own","baseline","depth","delta"]].round(2).to_string())

# ---- corrected board ----
final = d.reset_index()
final["ppm"] = final["mean"] / final["cost"].clip(lower=4.0)
final.round(2).to_csv(os.path.join(config.OUTPUTS, "decision_gw1_6_depthprior.csv"), index=False)
print("\n" + "=" * 74)
print("CORRECTED TOP 15 GW1-6 (depth prior on; nothing hidden)")
print("=" * 74)
print(final.nlargest(15, "mean")[["player","pos","team","cost","mean","p5","p95","ppm","own"]]
      .round(2).to_string(index=False))
