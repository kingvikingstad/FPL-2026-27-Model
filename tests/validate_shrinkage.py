import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys; import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src")); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import core_insights as ci, bayes_model, signals as sg, starter_prior as sp, solio_ensemble as se
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
import os

REPO = config.REPO
d26, t26, _ = ci.load(base=f"{REPO}/2026-2027"); sig = ci.to_signals(d26)
cal_cs = calibrate_cold_start(hist_csv=config.COLDSTART_HIST)
cal_own = sp.calibrate_ownership_start()
pri = pd.read_pickle(config.MS_PRIORS).dropna(subset=["player_code"]).drop_duplicates("player_code")
pmap = pri.set_index("player_code")
pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())
S = se.parse_solio(se.fetch_solio())["players"]


def build(shrink):
    rows = []
    for _, r in d26.iterrows():
        c = r.get("player_code")
        if pd.notna(c) and c in pmap.index:
            s = pmap.loc[c]
            row = {k: s[k] for k in ["npxgi_alpha","npxgi_beta","xa_alpha","xa_beta",
                                     "defcon_alpha","defcon_beta","start_a","start_b","sub_app_rate"]}
            row.update({"id": r.player_id, "web_name": r.web_name, "pos": r.pos, "team": r.team,
                        "own": r.selected_by_percent, "cost": r.now_cost, "minutes": s["minutes"],
                        "cold_start": False, "pen_xg90": float(s.get("pen_xg90_measured", 0) or 0)})
            rows.append(row)
        else:
            cs = _coldstart_row(r.web_name, r.team, r.pos, r.now_cost, r.selected_by_percent, cal_cs)
            cs["minutes"] = 0.0; rows.append(cs)
    pl = pd.DataFrame(rows)
    pl = sp.apply_coldstart_depth(pl, cal_own)
    if shrink:
        pl = sp.apply_minutes_shrinkage(pl, cal_own, k_min=900)
    pl = sg.apply_availability(pl, sig, lineups=None)
    pl["pen_xg90"] = np.where(pl.web_name.str.lower().str.strip().isin(pen1),
                              pl.pen_xg90.fillna(0).clip(lower=0.10), 0.0)
    return pl


elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
tm = TeamModel(promoted_per_club=pclub).fit(e0_path=config.E0_RECON, clubelo=elo, clubelo_weight=0.45)

out = {}
for tag, shrink in [("baseline", False), ("shrinkage", True)]:
    pl = build(shrink)
    bayes_model.rng = np.random.default_rng(7); ts = tm.sample_2627(S=3000)
    bayes_model.rng = np.random.default_rng(7)
    proj = project(pl, tm, ts, 1, 1, S=3000).drop_duplicates("id")
    aligned = se.align_single_gw(proj, S)
    bm = se.benchmark(aligned)
    defs = aligned[aligned.pos == "DEF"]
    out[tag] = dict(bm=bm, def_gap=(defs.our_proj - defs.solio_proj).mean(),
                    mosquera=proj[proj.player == "Mosquera"]["mean"].values[:1],
                    gabriel=proj[proj.player == "Gabriel"]["mean"].values[:1],
                    haaland=proj[proj.player == "Haaland"]["mean"].values[:1])

print("=" * 66)
print("MINUTES SHRINKAGE — validated against the Solio benchmark")
print("=" * 66)
for tag in ("baseline", "shrinkage"):
    b = out[tag]["bm"]
    print(f"{tag:10s}  Pearson={b['pearson']:.3f}  Spearman={b['spearman']:.3f}  "
          f"MAE={b['mae']:.2f}  bias={b['bias_ours_minus_solio']:+.2f}  "
          f"DEF gap={out[tag]['def_gap']:+.2f}")
print()
for nm in ("mosquera", "gabriel", "haaland"):
    a = out["baseline"][nm]; c = out["shrinkage"][nm]
    if len(a) and len(c):
        print(f"{nm.capitalize():10s} {a[0]:.2f} -> {c[0]:.2f}  ({c[0]-a[0]:+.2f})")
print("\n(Solio: Mosquera 5.58, Gabriel 5.98, Haaland 7.24)")
