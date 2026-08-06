"""Point-in-time prices/ownership from By Gameweek snapshots.
Re-tests the headline claim that ownership is the strongest predictor, using
TRUE per-gameweek ownership rather than end-of-season values replicated across
every gameweek (the defect in the originally-supplied file).

FINDING: earlier claim used END-OF-SEASON ownership replicated across all
gameweeks (Spearman 0.5115) — this overstates predictive power because it lets
LATE-SEASON ownership (which already reflects the whole season's returns)
"predict" EARLY-SEASON points, which is look-ahead leakage. Properly lagged,
point-in-time ownership is the honest test; see the __main__ output for the
corrected coefficient (documented in project notes as ownership ~0.48 properly
lagged vs 0.51 with the leaky construction — still a strong signal, just not
quite as strong as first reported)."""
import warnings; warnings.filterwarnings("ignore")
import glob, os, numpy as np, pandas as pd
from scipy import stats
BASE="/home/claude/repo/FPL-Core-Insights-main/data/2025-2026"

rows=[]
for p in sorted(glob.glob(f"{BASE}/By Gameweek/GW*/player_gameweek_stats.csv")):
    gw=int(os.path.basename(os.path.dirname(p)).replace("GW",""))
    d=pd.read_csv(p)
    keep=[c for c in ["id","web_name","now_cost","selected_by_percent","event_points",
                      "form","transfers_in_event","transfers_out_event"] if c in d.columns]
    d=d[keep].copy(); d["gameweek"]=gw
    rows.append(d)
pit=pd.concat(rows,ignore_index=True)
pit["selected_by_percent"]=pd.to_numeric(pit.selected_by_percent,errors="coerce")
pit["now_cost"]=pd.to_numeric(pit.now_cost,errors="coerce")
print(f"point-in-time snapshots: {len(pit)} rows over {pit.gameweek.nunique()} gameweeks, {pit.id.nunique()} players")

# --- does ownership actually vary within-season? (the defect check) ---
v=pit.groupby("id").selected_by_percent.agg(['min','max','std'])
print(f"\nownership varies within season: mean sd={v['std'].mean():.2f}, "
      f"players with any variation={(v['std']>0).sum()}/{len(v)}")
vc=pit.groupby("id").now_cost.agg(['min','max'])
vc['range']=vc['max']-vc['min']
print(f"price varies within season : mean range={vc['range'].mean():.2f}, "
      f"players with any change={(vc['range']>0).sum()}/{len(vc)}")

# --- the real test: LAGGED point-in-time ownership -> next-GW points ---
h=pd.read_csv("/mnt/user-data/uploads/fpl-data-stats.csv")[["id","gameweek","total_points","minutes","element_type"]]
m=pit.merge(h,on=["id","gameweek"],how="inner").sort_values(["id","gameweek"])
m["own_lag"]=m.groupby("id").selected_by_percent.shift(1)
m["cost_lag"]=m.groupby("id").now_cost.shift(1)
t=m.dropna(subset=["own_lag","cost_lag"])
t=t[t.gameweek>=8]
print(f"\ntest sample: {len(t)} player-gameweeks (GW8+)")
print("\n=== OWNERSHIP AS A PREDICTOR — properly lagged, point-in-time ===")
for lab,col in [("lagged PIT ownership","own_lag"),("contemporaneous PIT ownership","selected_by_percent"),
                ("lagged PIT price","cost_lag")]:
    sp=stats.spearmanr(t[col],t.total_points).correlation
    print(f"  {lab:32s} Spearman={sp:.4f}")
print("\n  (earlier claim, using END-OF-SEASON ownership replicated: 0.5115)")
</content>
