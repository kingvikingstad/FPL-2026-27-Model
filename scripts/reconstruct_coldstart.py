import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
reconstruct_coldstart.py — build the cold-start calibration input
(coldstart_hist.csv) from the 25/26 per-match panel, using the model's own
event definitions (npxg, xa_, defcon_raw). Replaces the defective/unavailable
fpl-data-stats.csv that roster.calibrate_cold_start() expects, matching its
exact schema so it runs unchanged.

Requires /tmp/pms_panel.pkl (build_pms.build()). Edit REPO / paths as needed.
"""
import numpy as np, pandas as pd

REPO = config.REPO
PANEL = config.PMS_PANEL
OUT = config.COLDSTART_HIST


def build():
    panel = pd.read_pickle(PANEL)
    agg = panel.groupby("player_id").agg(
        minutes=("mins", "sum"), npxg=("npxg", "sum"), xa=("xa_", "sum"),
        defensive_contribution=("defcon_raw", "sum"), pos=("pos", "last")).reset_index()
    agg["non_penalty_expected_goal_involvements"] = agg.npxg + agg.xa
    agg["expected_assists"] = agg.xa
    agg["element_type"] = agg.pos.map({"GK": 1, "DEF": 2, "MID": 3, "FWD": 4})
    ps = pd.read_csv(f"{REPO}/2025-2026/playerstats.csv")
    price = ps.sort_values("gw").groupby("id")["now_cost"].last()
    agg["now_cost"] = agg.player_id.map(price)
    agg = agg.rename(columns={"player_id": "id"})
    agg["now_cost"] = np.where(agg.now_cost > 30, agg.now_cost / 10, agg.now_cost)  # millions guard
    out = agg[["id", "element_type", "minutes",
               "non_penalty_expected_goal_involvements", "expected_assists",
               "defensive_contribution", "now_cost"]].dropna(subset=["now_cost", "element_type"])
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} players, {(out.minutes >= 450).sum()} with >=450 min")
    return out


if __name__ == "__main__":
    build()
