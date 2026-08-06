"""GW1-3 and GW1-6 holds using the REAL 2026/27 dataset."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys; sys.path.insert(0,'/home/claude/fpl')
import core_insights as ci
from bayes_model import TeamModel, project
from roster import build_players_2627
import signals as sg

d, t, gwsum = ci.load()
roster = ci.to_roster(d); sig = ci.to_signals(d)
elo = ci.to_elo_frame(t)
pp = ci.promoted_prior_from_elo(t)

players, _ = build_players_2627(roster)
players = sg.apply_availability(players, sig)
players = sg.apply_setpieces(players, sig)
n_cold = int(players.cold_start.sum())
print(f"roster {len(players)} players | cold-start {n_cold} | flagged {(d.status!='a').sum()}")

tm = TeamModel(promoted_per_club=pp["per_club"]).fit(clubelo=elo, clubelo_weight=0.45)
ts = tm.sample_2627(S=1500)
print(f"team model: Elo blended into {tm.clubelo_used} teams; per-club promoted priors {list(pp['per_club'].keys())}\n")

for H in [3, 6]:
    r = project(players, tm, ts, 1, H, S=1500).drop_duplicates(["player","team"])
    r["ppm"] = r["mean"] / r["cost"].clip(lower=4.0)
    r.round(2).to_csv(f"/mnt/user-data/outputs/holds_REAL_gw1_{H}.csv", index=False)
    print("#"*74); print(f"#  GW1-{H} HOLDS — REAL 26/27 roster, prices, availability"); print("#"*74)
    for pos, n in [("GK",3),("DEF",6),("MID",6),("FWD",4)]:
        s = r[r.pos==pos].nlargest(n,"mean")
        print(f"\n-- {pos} --")
        print(s[["player","team","cost","mean","p5","p95","ppm","own"]].round(2).to_string(index=False))
    print(f"\n>> CAPTAIN (ceiling p95): ")
    print(r.nlargest(4,"p95")[["player","pos","team","mean","p95"]].round(1).to_string(index=False))
    print(f"\n>> VALUE (pts/£m, <=6.5m):")
    print(r[r.cost<=6.5].nlargest(6,"ppm")[["player","pos","team","cost","mean","ppm"]].round(2).to_string(index=False))
    print()
</content>
