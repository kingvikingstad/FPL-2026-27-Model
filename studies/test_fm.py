import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import fm_priors as fmp
rng = np.random.default_rng(11)
# Build a synthetic FM export whose attributes GENERATE the observed 25/26 rates,
# so we can check calibrate() recovers the true signal.
hist = pd.read_csv(config.FPL_DATA_STATS)
hist["pos"] = hist.element_type.map({1:"GK",2:"DEF",3:"MID",4:"FWD"})
ag = hist.groupby(["web_name","pos"]).agg(
    minutes=("minutes","sum"),
    npxgi=("non_penalty_expected_goal_involvements","sum"),
    defcon=("defensive_contribution","sum"),
    starts=("minutes", lambda s:(s>=60).sum()), games=("minutes","size")).reset_index()
ag = ag[ag.minutes>=600].copy()
nnf = ag.minutes/90
ag["inv90"]=ag.npxgi/nnf; ag["dc90"]=ag.defcon/nnf; ag["sr"]=ag.starts/ag.games

# FM attributes correlated with the player's REAL rates (that's the ground truth
# a real FM database would approximately encode), plus scouting noise.
def attr_from(x, lo=6, hi=18, noise=2.0):
    r = (x - x.mean())/(x.std()+1e-9)
    return np.clip(12 + 2.2*r + rng.normal(0, noise, len(x)), lo, hi).round(0)

fm = pd.DataFrame({
  "name": ag.web_name, "team":"X", "pos": ag.pos,
  "Finishing": attr_from(ag.inv90), "OffTheBall": attr_from(ag.inv90),
  "Anticipation": attr_from(ag.inv90*0.5+ag.dc90*0.02),
  "Composure": attr_from(ag.inv90), "Technique": attr_from(ag.inv90),
  "Acceleration": attr_from(ag.inv90*0.3),
  "Passing": attr_from(ag.inv90*0.4), "Vision": attr_from(ag.inv90*0.5),
  "Crossing": attr_from(ag.inv90*0.3), "Flair": attr_from(ag.inv90*0.2),
  "Dribbling": attr_from(ag.inv90*0.4),
  "Tackling": attr_from(ag.dc90), "Marking": attr_from(ag.dc90),
  "Positioning": attr_from(ag.dc90), "WorkRate": attr_from(ag.dc90),
  "Aggression": attr_from(ag.dc90*0.7),
  "CurrentAbility": np.clip(100+40*((ag.sr-ag.sr.mean())/(ag.sr.std()+1e-9))+rng.normal(0,15,len(ag)),40,190).round(0),
  "NaturalFitness": attr_from(ag.sr), "Stamina": attr_from(ag.sr),
  "Determination": attr_from(ag.sr),
})
print("=== 1. CALIBRATION on FM x PL overlap ===")
m = fmp.calibrate(fm)

print("\n=== 2. COLD-START differentiation (no PL history) ===")
# three fictional promoted-team players: elite / average / poor attributes
def mk(name, lvl):
    row = {"name":name,"team":"Ipswich","pos":"MID"}
    for c in fmp.ALL_ATTRS: row[c] = lvl
    row["CurrentAbility"] = {18:150,12:110,7:75}[lvl]
    return row
fm_cold = pd.DataFrame([mk("Alpha Elite",18), mk("Beta Average",12), mk("Gamma Poor",7)])
players = pd.DataFrame([{
  "web_name":n,"pos":"MID","team":"Ipswich","minutes":0,"cold_start":True,
  "npxgi_alpha":0.20*1.5,"npxgi_beta":1.5,"xa_alpha":0.08*1.5,"xa_beta":1.5,
  "defcon_alpha":8.0*1.5,"defcon_beta":1.5,"start_a":2.0,"start_b":2.0}
  for n in ["Alpha Elite","Beta Average","Gamma Poor"]])
out = fmp.apply_fm_priors(players, fm_cold, m)
out["inv90_prior"]=out.npxgi_alpha/out.npxgi_beta
out["dc90_prior"]=out.defcon_alpha/out.defcon_beta
out["P(start)"]=out.start_a/(out.start_a+out.start_b)
print(out[["web_name","inv90_prior","dc90_prior","P(start)"]].round(3).to_string(index=False))

print("\n=== 3. FM weight DECAYS with observed minutes (prior vs likelihood) ===")
rows=[]
for mins in [0, 300, 900, 2000, 3400]:
    pl = pd.DataFrame([{"web_name":"Alpha Elite","pos":"MID","team":"X","minutes":mins,
      "cold_start":mins==0,"npxgi_alpha":0.20*1.5,"npxgi_beta":1.5,
      "xa_alpha":0.08*1.5,"xa_beta":1.5,"defcon_alpha":8.0*1.5,"defcon_beta":1.5,
      "start_a":2.0,"start_b":2.0}])
    o = fmp.apply_fm_priors(pl, fm_cold, m, verbose=False)
    rows.append({"observed_minutes":mins,"FM_weight":round(900/(900+mins),3),
                 "resulting_inv90_prior":round(float((o.npxgi_alpha/o.npxgi_beta).iloc[0]),3)})
print(pd.DataFrame(rows).to_string(index=False))
print("\n(observed rate baseline = 0.200; FM-elite pull shrinks as minutes accumulate)")
