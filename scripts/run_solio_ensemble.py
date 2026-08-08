import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys; import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src")); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import solio_ensemble as se
import core_insights as ci
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
import os
import signals as sg, starter_prior as sp

# ---------- 1. fetch + parse (correctness against a known-good source) ----------
md = se.fetch_solio()
sol = se.parse_solio(md)
S = sol["players"]
print(f"[parse] Solio GW{sol['gameweek']}: {len(S)} players parsed, "
      f"{S['solio_proj'].notna().sum()} with projections, "
      f"{S['prG'].notna().sum()} with prG")
print(S[["player", "team", "pos", "price", "solio_proj", "own", "prG", "prA"]].head(6).to_string(index=False))

print("\n[validate] reproduce Solio's OWN leverage differential ranking from components:")
S2 = S.dropna(subset=["solio_proj", "own"]).copy()
S2["lev"] = se.leverage(S2)
print("  our recomputed top-5 leverage:",
      list(S2.nlargest(5, "lev")["player"]))
print("  (compare against the 'Highest-leverage differentials' table in the live feed)")

# ---------- 2. build OUR single-GW (GW1) projection ----------
REPO = config.REPO
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
        row.update({"id": r.player_id, "web_name": r.web_name, "pos": r.pos, "team": r.team,
                    "own": r.selected_by_percent, "cost": r.now_cost, "cold_start": False,
                    "pen_xg90": float(s.get("pen_xg90_measured", 0) or 0)}); rows.append(row)
    else:
        rows.append(_coldstart_row(r.web_name, r.team, r.pos, r.now_cost, r.selected_by_percent, cal_cs))
pl = pd.DataFrame(rows)
pl = sp.apply_coldstart_depth(pl, cal_own)
pl = sg.apply_availability(pl, sig, lineups=None)
pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())
pl["pen_xg90"] = np.where(pl.web_name.str.lower().str.strip().isin(pen1),
                          pl.pen_xg90.fillna(0).clip(lower=0.10), 0.0)
elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
tm = TeamModel(promoted_per_club=pclub).fit(e0_path=config.E0_RECON, clubelo=elo, clubelo_weight=0.45)
ts = tm.sample_2627(S=3000)
ours_gw1 = project(pl, tm, ts, 1, 1, S=3000).drop_duplicates("id")   # SINGLE-GW basis
print(f"\n[ours] single-GW (GW1) projection built: {len(ours_gw1)} players "
      f"(top: {ours_gw1.iloc[0].player} {ours_gw1.iloc[0]['mean']:.2f})")

# ---------- 3. align + blend + benchmark ----------
aligned = se.align_single_gw(ours_gw1, S)
print(f"\n[align] matched {len(aligned)} players between our GW1 board and Solio's feed")
bm = se.benchmark(aligned)
print(f"[benchmark] n={bm['n_matched']}  Pearson={bm['pearson']:.3f}  "
      f"Spearman={bm['spearman']:.3f}  MAE={bm['mae']:.2f}  "
      f"bias(ours-solio)={bm['bias_ours_minus_solio']:+.2f}")

w = se.accuracy_weights(mae_ours=None, mae_solio=None)   # equal until we have realised results
blended = se.blend(aligned, w_ours=w)
print(f"\n[ensemble] w_ours={w:.2f} (equal — set from inverse-MAE once GW results land)")
print(blended[["player", "pos", "team", "our_proj", "solio_proj", "ensemble"]].head(10).round(2).to_string(index=False))

hi, lo = se.disagreements(aligned, n=8)
print("\n[disagreement] WE rate ABOVE Solio (differential edge or over-projection):")
print(hi.round(2).to_string(index=False))
print("\n[disagreement] WE rate BELOW Solio (blind spot or Solio over-projection):")
print(lo.round(2).to_string(index=False))

blended.round(3).to_csv(os.path.join(config.OUTPUTS, "solio_ensemble_demo.csv"), index=False)
print("\nNOTE: Solio's feed is 25/26 GW35; our board is 26/27 GW1. The numbers above")
print("exercise the plumbing on real data — level-valid ensembling begins once Solio")
print("publishes 26/27. Name-alignment, blend, and benchmark logic are season-agnostic.")
