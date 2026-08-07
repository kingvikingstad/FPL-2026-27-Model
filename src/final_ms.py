"""26/27 projections on TWO-SEASON hierarchical priors, compared to single-season."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys; sys.path.insert(0,'/home/claude/fpl')
import core_insights as ci, pms_priors as pp, multiseason_priors as ms
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
import signals as sg
REPO='/home/claude/repo/FPL-Core-Insights-main/data'

d26,t26,_ = ci.load(base=f'{REPO}/2026-2027'); sig = ci.to_signals(d26)
cal = calibrate_cold_start(hist_csv='/tmp/coldstart_hist.csv')

def build(prior_source):
    if prior_source=='single':
        rates=pp.season_rates(); pri=pp.to_model_priors(rates)
        p25=pd.read_csv(f'{REPO}/2025-2026/players.csv')[['player_id','player_code']]
        pri=pri.merge(p25,on='player_id',how='left')
    else:
        pri=pd.read_pickle('/tmp/ms_priors.pkl')
    pri=pri.dropna(subset=['player_code']).drop_duplicates('player_code')
    pmap=pri.set_index('player_code')
    rows=[]; carried=0
    for _,r in d26.iterrows():
        c=r.get('player_code')
        if pd.notna(c) and c in pmap.index:
            s=pmap.loc[c]
            row={k:s[k] for k in ['npxgi_alpha','npxgi_beta','xa_alpha','xa_beta',
                                  'defcon_alpha','defcon_beta','start_a','start_b','sub_app_rate']}
            row.update({'id':r.player_id,'web_name':r.web_name,'pos':r.pos,'team':r.team,
                        'own':r.selected_by_percent,'cost':r.now_cost,'minutes':s['minutes'],
                        'cold_start':False,'pen_xg90':float(s.get('pen_xg90_measured',0) or 0)})
            rows.append(row); carried+=1
        else:
            cs=_coldstart_row(r.web_name,r.team,r.pos,r.now_cost,r.selected_by_percent,cal)
            cs.update({'minutes':0.0,'pen_xg90':0.0}); rows.append(cs)
    pl=pd.DataFrame(rows)
    pl=sg.apply_availability(pl,sig)
    pen1=set(sig.loc[sig.pen_order==1,'name'].str.lower().str.strip())
    pl['pen_xg90']=np.where(pl.web_name.str.lower().str.strip().isin(pen1),
                            pl.pen_xg90.fillna(0).clip(lower=0.10),0.0)
    return pl, carried

elo=ci.to_elo_frame(t26); pclub=ci.promoted_prior_from_elo(t26)['per_club']
tm=TeamModel(promoted_per_club=pclub).fit(e0_path='/tmp/E0_recon.csv',clubelo=elo,clubelo_weight=0.45)
ts=tm.sample_2627(S=2000)

res={}
for src in ['single','multi']:
    pl,carried=build(src)
    print(f"{src:6s} priors: {carried}/{len(pl)} carried forward, {len(pl)-carried} cold-start")
    r=project(pl,tm,ts,1,6,S=2000).drop_duplicates(['player','team'])
    r['ppm']=r['mean']/r['cost'].clip(lower=4.0)
    res[src]=r.set_index('player')
    r.round(2).to_csv(f'/mnt/user-data/outputs/final_ms_{src}_gw1_6.csv',index=False)

a,b=res['single'],res['multi']
common=a.index.intersection(b.index)
cmp=pd.DataFrame({'pos':a.loc[common,'pos'],'team':a.loc[common,'team'],
                  'cost':a.loc[common,'cost'],'single':a.loc[common,'mean'],
                  'multi':b.loc[common,'mean'],
                  'sd_single':a.loc[common,'sd'],'sd_multi':b.loc[common,'sd']})
cmp['delta']=cmp['multi']-cmp['single']
cmp['sd_change']=cmp.sd_multi-cmp.sd_single
print(f"\ncommon players: {len(cmp)}")
print(f"mean |projection change|: {cmp.delta.abs().mean():.2f} pts over GW1-6")
print(f"mean change in posterior SD: {cmp.sd_change.mean():+.3f} "
      f"({'TIGHTER' if cmp.sd_change.mean()<0 else 'wider'} intervals)")
print("\nBiggest UPGRADES from adding 24/25 evidence:")
print(cmp.nlargest(8,'delta')[['pos','team','cost','single','multi','delta']].round(2).to_string())
print("\nBiggest DOWNGRADES:")
print(cmp.nsmallest(6,'delta')[['pos','team','cost','single','multi','delta']].round(2).to_string())
print("\nTop-10 GW1-6 under TWO-SEASON priors:")
print(b.nlargest(10,'mean')[['pos','team','cost','mean','p5','p95','ppm','own']].round(2).to_string())
