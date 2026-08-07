"""Where does single-gameweek FPL point variance actually come from?
Answers 'what information reduces noise' by measuring how much variance each
piece of knowledge would remove if you had it perfectly."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
p = pd.read_pickle("/tmp/pms_panel.pkl")
h = pd.read_csv("/mnt/user-data/uploads/fpl-data-stats.csv")
h["pos"] = h.element_type.map({1:"GK",2:"DEF",3:"MID",4:"FWD"})
d = h[h.gameweek.between(1,38)].copy()
# restrict to players in the squad-relevant pool (owned-worthy)
tot = d.groupby("id").minutes.sum()
keep = tot[tot>=450].index
d = d[d.id.isin(keep)].copy()
V = d.total_points.var()
print(f"pool: {len(d)} player-gameweeks, {d.id.nunique()} players")
print(f"TOTAL variance of single-GW points = {V:.3f}  (sd {np.sqrt(V):.2f})\n")

def rv(cond_cols, label):
    """variance remaining after conditioning on cond_cols (group means)"""
    grp = d.groupby(cond_cols).total_points.transform("mean")
    resid = d.total_points - grp
    rem = resid.var()
    print(f"  {label:52s} remaining {rem:6.3f}   explained {100*(1-rem/V):5.1f}%")
    return rem

print("VARIANCE EXPLAINED BY PERFECT FOREKNOWLEDGE OF:")
d["played"] = (d.minutes>0).astype(int)
d["min60"] = (d.minutes>=60).astype(int)
d["ga"] = d.goals + d.assists
d["cs"] = d.clean_sheet
d["dc_hit"] = (d.defensive_contribution >= np.where(d.pos=="DEF",10,12)).astype(int)
rv(["played"], "whether he plays at all")
rv(["min60"], "whether he plays 60+ minutes")
rv(["min60","pos"], "60+ mins x position")
rv(["min60","cs"], "60+ mins + clean sheet")
rv(["min60","dc_hit"], "60+ mins + defensive-contribution hit")
rv(["min60","ga"], "60+ mins + goals/assists")
rv(["min60","cs","dc_hit"], "60+ mins + clean sheet + defcon")
rv(["min60","ga","cs"], "60+ mins + G/A + clean sheet")
rv(["min60","ga","cs","dc_hit"], "60+ mins + G/A + CS + defcon (all but bonus)")

print("\nBY POSITION — variance and what dominates it:")
for pos in ["GK","DEF","MID","FWD"]:
    s = d[d.pos==pos]
    v = s.total_points.var()
    r_min = (s.total_points - s.groupby("min60").total_points.transform("mean")).var()
    r_ga  = (s.total_points - s.groupby(["min60","ga"]).total_points.transform("mean")).var()
    r_cs  = (s.total_points - s.groupby(["min60","cs"]).total_points.transform("mean")).var()
    print(f"  {pos}: var={v:5.2f} | minutes {100*(1-r_min/v):4.1f}% | +G/A {100*(1-r_ga/v):4.1f}% | +CS {100*(1-r_cs/v):4.1f}%")

print("\nHOW OFTEN DOES EACH EVENT ACTUALLY HAPPEN (per player-GW, 60+ min only)?")
s = d[d.min60==1]
print(f"  goal or assist        {(s.ga>0).mean():.3f}")
print(f"  clean sheet           {s.cs.mean():.3f}")
print(f"  defcon hit            {s.dc_hit.mean():.3f}")
print(f"  any bonus             {(s.get('bonus',pd.Series(0,index=s.index))>0).mean() if 'bonus' in s else float('nan'):.3f}")
