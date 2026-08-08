import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys; import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src")); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import core_insights as ci, bayes_model, signals as sg, starter_prior as sp
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
import os

REPO = config.REPO; S = 4000
d26, t26, _ = ci.load(base=f"{REPO}/2026-2027"); sig = ci.to_signals(d26)
cal_cs = calibrate_cold_start(hist_csv=config.COLDSTART_HIST)
cal_own = sp.calibrate_ownership_start()
pri = pd.read_pickle(config.MS_PRIORS).dropna(subset=["player_code"]).drop_duplicates("player_code")
pmap = pri.set_index("player_code")
pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())


def base_frame():
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
    return sp.apply_coldstart_depth(pl, cal_own)


def finish(pl):
    pl = sg.apply_availability(pl, sig, lineups=None)
    pl["pen_xg90"] = np.where(pl.web_name.str.lower().str.strip().isin(pen1),
                              pl.pen_xg90.fillna(0).clip(lower=0.10), 0.0)
    return pl


elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
tm = TeamModel(promoted_per_club=pclub).fit(e0_path=config.E0_RECON, clubelo=elo, clubelo_weight=0.45)
bayes_model.rng = np.random.default_rng(7); ts = tm.sample_2627(S=S)

b = base_frame()
# TEST 1 — baseline identity: regime={} == apply_minutes_shrinkage
f_shrink = sp.apply_minutes_shrinkage(b.copy(), cal_own, k_min=900)
f_regime0 = sp.apply_regime_uncertainty(b.copy(), regime={}, cal=cal_own, k_min=900)
ident = np.allclose(f_shrink.start_a, f_regime0.start_a) and np.allclose(f_shrink.start_b, f_regime0.start_b)
print(f"TEST 1  baseline identity (regime={{}} == shrinkage): {'PASS' if ident else 'FAIL'}")

# Build kappa=0 and kappa=0.8 frames (delta=1 -> isolate variance), proposed clubs
reg08 = {k: (1.0, v[1]) for k, v in sp.REGIME_2627_PROPOSED.items()}   # delta=1, keep kappa
f0 = finish(sp.apply_regime_uncertainty(b.copy(), regime={}, cal=cal_own))
fK = finish(sp.apply_regime_uncertainty(b.copy(), regime=reg08, cal=cal_own))


def proj(pl, hi):
    bayes_model.rng = np.random.default_rng(7)
    return project(pl, tm, ts, 1, hi, S=S).drop_duplicates("id").set_index("id")


regime_clubs = set(sp.REGIME_2627_PROPOSED)
for H in (1, 6):
    p0, pK = proj(f0, H), proj(fK, H)
    common = p0.index.intersection(pK.index)
    reg_ids = [i for i in common if p0.loc[i, "team"] in regime_clubs]
    non_ids = [i for i in common if p0.loc[i, "team"] not in regime_clubs]
    d_sd_reg = (pK.loc[reg_ids, "sd"] - p0.loc[reg_ids, "sd"]).mean()
    d_sd_non = (pK.loc[non_ids, "sd"] - p0.loc[non_ids, "sd"]).mean()
    d_mean_reg = (pK.loc[reg_ids, "mean"] - p0.loc[reg_ids, "mean"]).abs().mean()
    print(f"H={H}: regime ΔSD={d_sd_reg:+.3f}  non-regime ΔSD={d_sd_non:+.3f}  regime |Δmean|={d_mean_reg:.3f}")

# TEST 4 — horizon signature: H=1 ΔSD ~ 0, H=6 ΔSD > 0
p0_1, pK_1 = proj(f0, 1), proj(fK, 1)
p0_6, pK_6 = proj(f0, 6), proj(fK, 6)
reg_ids1 = [i for i in p0_1.index.intersection(pK_1.index) if p0_1.loc[i, "team"] in regime_clubs]
reg_ids6 = [i for i in p0_6.index.intersection(pK_6.index) if p0_6.loc[i, "team"] in regime_clubs]
h1 = (pK_1.loc[reg_ids1, "sd"] - p0_1.loc[reg_ids1, "sd"]).mean()
h6 = (pK_6.loc[reg_ids6, "sd"] - p0_6.loc[reg_ids6, "sd"]).mean()
print(f"\nTEST 4  horizon signature: H=1 ΔSD={h1:+.3f} (want ~0), H=6 ΔSD={h6:+.3f} (want >0): "
      f"{'PASS' if abs(h1) < 0.05 and h6 > 0.05 else 'FAIL'}")

# TEST 6 — mu^2 scaling: SD inflation correlates with EP, not flat
merged = pd.DataFrame({"mean": p0_6.loc[reg_ids6, "mean"],
                       "dsd": (pK_6.loc[reg_ids6, "sd"] - p0_6.loc[reg_ids6, "sd"])})
rho = merged["mean"].corr(merged["dsd"], method="spearman")
print(f"TEST 6  SD-inflation vs EP correlation (want >0): {rho:+.3f} {'PASS' if rho > 0.2 else 'WEAK'}")

# show the players most widened
merged["player"] = p0_6.loc[reg_ids6, "player"]; merged["team"] = p0_6.loc[reg_ids6, "team"]
print("\nMost-widened regime players (GW1-6 SD inflation):")
print(merged.nlargest(8, "dsd")[["player", "team", "mean", "dsd"]].round(2).to_string(index=False))
