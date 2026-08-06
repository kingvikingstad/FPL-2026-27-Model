"""Consolidated GW1-6 decision board on verified live 26/27 data.
Reuses the two-season frame build from final_ms, adds cold-start flags,
a realistic-pool ranking, and the GW1 captaincy/differential layer."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys; sys.path.insert(0, "/home/claude/fpl")
import core_insights as ci
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
import signals as sg
import captaincy as cap

REPO = "/home/claude/repo/FPL-Core-Insights-main/data"
d26, t26, _ = ci.load(base=f"{REPO}/2026-2027"); sig = ci.to_signals(d26)
cal = calibrate_cold_start(hist_csv="/tmp/coldstart_hist.csv")
pri = pd.read_pickle("/tmp/ms_priors.pkl").dropna(subset=["player_code"]).drop_duplicates("player_code")
pmap = pri.set_index("player_code")

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
        cs = _coldstart_row(r.web_name, r.team, r.pos, r.now_cost, r.selected_by_percent, cal)
        cs.update({"minutes": 0.0, "pen_xg90": 0.0}); rows.append(cs)
pl = pd.DataFrame(rows)
pl = sg.apply_availability(pl, sig)
pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())
pl["pen_xg90"] = np.where(pl.web_name.str.lower().str.strip().isin(pen1),
                          pl.pen_xg90.fillna(0).clip(lower=0.10), 0.0)

elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
tm = TeamModel(promoted_per_club=pclub).fit(e0_path="/tmp/E0_recon.csv", clubelo=elo, clubelo_weight=0.45)
ts = tm.sample_2627(S=3000)

proj = project(pl, tm, ts, 1, 6, S=3000).drop_duplicates(["player", "team"])
proj = proj.merge(pl[["web_name", "cold_start"]].rename(columns={"web_name": "player"}),
                  on="player", how="left")
proj["ppm"] = proj["mean"] / proj["cost"].clip(lower=4.0)
proj.round(2).to_csv("/mnt/user-data/outputs/decision_gw1_6_full.csv", index=False)

# Realistic pickable pool: exclude cold-start players nobody owns (lineup blind spot)
real = proj[~((proj.cold_start) & (proj.own < 1.0))].copy()
real.round(2).to_csv("/mnt/user-data/outputs/decision_gw1_6_realistic.csv", index=False)

print("=" * 72)
print("TOP 15 GW1-6  (realistic pool: cold-start 0%-owned excluded)")
print("=" * 72)
print(real.nlargest(15, "mean")[["player","pos","team","cost","mean","p5","p95","ppm","own"]].round(2).to_string(index=False))

print("\n" + "=" * 72)
print("BEST VALUE (ppm) among >=£5.5m, top of pool")
print("=" * 72)
val = real[(real.cost >= 5.5) & (real["mean"] > 12)].nlargest(12, "ppm")
print(val[["player","pos","team","cost","mean","ppm","own"]].round(2).to_string(index=False))

print("\n" + "=" * 72)
print("COLD-START INFLATION FLAG (0%-owned, projected > many premiums)")
print("=" * 72)
flagged = proj[(proj.cold_start) & (proj.own < 1.0)].nlargest(6, "mean")
print(flagged[["player","pos","team","cost","mean","own"]].round(2).to_string(index=False))

# ---- GW1 captaincy + differential layer ----
print("\n" + "=" * 72)
print("GW1 CAPTAINCY  (tail-ranked, template own>=15%)")
print("=" * 72)
capdf = cap.captain_picks(pl, tm, ts, gameweek=1, horizon=1, S=3000, top=10)
show = [c for c in ["player","team","ev_captain","ceiling","p_haul","p_blank","own","risk","regret","gain"] if c in capdf.columns]
print(capdf[show].round(3).head(10).to_string(index=False))

print("\nGW1 BEST DIFFERENTIALS (own<=10%)")
diff = cap.best_differential(pl, tm, ts, gameweek=1, horizon=1, S=3000, max_own=10)
show2 = [c for c in ["player","team","ev_captain","ceiling","p_haul","own","risk"] if c in diff.columns]
print(diff[show2].round(3).head(8).to_string(index=False))
