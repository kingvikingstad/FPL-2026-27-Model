import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys; sys.path.insert(0,'/home/claude/fpl')
import core_insights as ci, pms_priors as pp
from bayes_model import TeamModel
from roster import calibrate_cold_start, _coldstart_row
import signals as sg, captaincy as cap
import os

# build the real 26/27 frame (same path as final_2627)
d26,t26,_ = ci.load(); sig = ci.to_signals(d26)
rates = pp.season_rates(); pri = pp.to_model_priors(rates)
p25 = pd.read_csv('/home/claude/repo/FPL-Core-Insights-main/data/2025-2026/players.csv')[['player_id','player_code']]
pri = pri.merge(p25,on='player_id',how='left').dropna(subset=['player_code']).drop_duplicates('player_code')
cal = calibrate_cold_start(); pmap = pri.set_index('player_code')
rows=[]
for _,r in d26.iterrows():
    c=r.get('player_code')
    if pd.notna(c) and c in pmap.index:
        s=pmap.loc[c]
        row={k:s[k] for k in ['npxgi_alpha','npxgi_beta','xa_alpha','xa_beta','defcon_alpha','defcon_beta','start_a','start_b','sub_app_rate']}
        row.update({'id':r.player_id,'web_name':r.web_name,'pos':r.pos,'team':r.team,
                    'own':r.selected_by_percent,'cost':r.now_cost,'minutes':s['minutes'],
                    'cold_start':False,'pen_xg90':float(s['pen_xg90_measured'])})
        rows.append(row)
    else:
        cs=_coldstart_row(r.web_name,r.team,r.pos,r.now_cost,r.selected_by_percent,cal)
        cs.update({'minutes':0.0,'pen_xg90':0.0}); rows.append(cs)
players=pd.DataFrame(rows)
players=sg.apply_availability(players,sig)
pen1=set(sig.loc[sig.pen_order==1,'name'].str.lower().str.strip())
players['pen_xg90']=np.where(players.web_name.str.lower().str.strip().isin(pen1),
                             players.pen_xg90.fillna(0).clip(lower=0.10),0.0)
elo=ci.to_elo_frame(t26); pclub=ci.promoted_prior_from_elo(t26)['per_club']
tm=TeamModel(promoted_per_club=pclub).fit(clubelo=elo,clubelo_weight=0.45)
ts=tm.sample_2627(S=3000)

for gw in [1,2]:
    c=cap.captain_picks(players,tm,ts,gameweek=gw,S=3000,top=10)
    print("="*96); print(f"CAPTAIN OPTIONS — GW{gw}   (template ref: {c.template_ref.iloc[0]})"); print("="*96)
    print(c[['player','pos','team','own','mean','ev_captain','floor_if_plays','ceiling','p_haul','p_blank','p_plays','cv','risk_profile']]
          .round(3).to_string(index=False))
    print()
    d=cap.best_differential(players,tm,ts,gameweek=gw,S=3000,max_own=10.0,top=8)
    print(f"--- BEST DIFFERENTIALS GW{gw} (own <=10%) ---")
    print(d[['player','pos','team','cost','own','mean','ceiling','p_haul','differential_score']].round(3).to_string(index=False))
    print()
    c.round(4).to_csv(fos.path.join(config.OUTPUTS, "captain_gw{gw}.csv"),index=False)
    d.round(4).to_csv(fos.path.join(config.OUTPUTS, "differentials_gw{gw}.csv"),index=False)
