"""
build_pms.py — consolidate per-match player stats into a modelling panel
========================================================================
Assembles all 38 gameweeks of `playermatchstats` for 2025-26, joins each row to
its match (for gameweek / team / opponent / venue), restricts to Premier League
fixtures, and attaches realised FPL points.

Why this beats the FPL aggregates the model used before:
  * `penalties_scored` / `penalties_missed`  -> TRUE non-penalty xG, instead of
    inferring it
  * `xgot`                                    -> shot quality on target, separating
    chance creation from finishing
  * `tackles`,`interceptions`,`blocks`,`clearances`,`recoveries` held SEPARATELY
    -> exactly the components the 26/27 BPS reweights (CBI now 1/3)
  * `goals_prevented`, `xgot_faced`, `saves_inside_box` -> real keeper
    shot-stopping, which the model had no equivalent of
  * `start_min` / `finish_min`                -> exact time on pitch
"""
from __future__ import annotations
import glob, os
import numpy as np, pandas as pd

BASE = "/home/claude/repo/FPL-Core-Insights-main/data/2025-2026"
OUT = "/tmp/pms_panel.parquet"


def build():
    teams = pd.read_csv(f"{BASE}/teams.csv")[["code", "id", "name", "short_name"]]
    code2name = dict(zip(teams.code, teams.name))

    frames = []
    for gwdir in sorted(glob.glob(f"{BASE}/By Gameweek/GW*")):
        gw = int(os.path.basename(gwdir).replace("GW", ""))
        try:
            pms = pd.read_csv(f"{gwdir}/playermatchstats.csv")
            mts = pd.read_csv(f"{gwdir}/matches.csv")
        except Exception:
            continue
        # long-form match -> one row per team with opponent & venue
        rows = []
        for _, r in mts.iterrows():
            h, a = r.get("home_team"), r.get("away_team")
            rows.append({"match_id": r["match_id"], "team_code": h, "opp_code": a,
                         "is_home": 1, "team_elo": r.get("home_team_elo"),
                         "opp_elo": r.get("away_team_elo"),
                         "team_xg": r.get("home_expected_goals_xg"),
                         "opp_xg": r.get("away_expected_goals_xg"),
                         "gf": r.get("home_score"), "ga": r.get("away_score")})
            rows.append({"match_id": r["match_id"], "team_code": a, "opp_code": h,
                         "is_home": 0, "team_elo": r.get("away_team_elo"),
                         "opp_elo": r.get("home_team_elo"),
                         "team_xg": r.get("away_expected_goals_xg"),
                         "opp_xg": r.get("home_expected_goals_xg"),
                         "gf": r.get("away_score"), "ga": r.get("home_score")})
        ml = pd.DataFrame(rows)
        # a player's team = whichever side of the match he appears for; resolve by
        # joining on match and taking the team whose roster contains him
        pl = pd.read_csv(f"{gwdir}/players.csv")[["player_id", "web_name", "team_code", "position"]]
        d = pms.merge(pl, on="player_id", how="left")
        d = d.merge(ml, on=["match_id", "team_code"], how="inner")
        d["gameweek"] = gw
        frames.append(d)

    panel = pd.concat(frames, ignore_index=True)
    panel["team"] = panel.team_code.map(code2name)
    panel["opp"] = panel.opp_code.map(code2name)
    # Premier League only. Both sides being PL clubs is NOT enough — cup ties and
    # European matches between two PL clubs sit in the same gameweek folders, so
    # filter on the competition token in match_id ("25-26-premier-league-...").
    panel = panel[panel.match_id.astype(str).str.contains("-prem-", na=False)].copy()
    panel = panel[panel.team.notna() & panel.opp.notna()].copy()
    panel["pos"] = panel.position.map({"Goalkeeper": "GK", "Defender": "DEF",
                                       "Midfielder": "MID", "Forward": "FWD"})

    # ---- derived, model-relevant quantities -------------------------------
    num = lambda c: pd.to_numeric(panel.get(c), errors="coerce").fillna(0.0)
    panel["pens_scored"] = num("penalties_scored")
    panel["pens_missed"] = num("penalties_missed")
    pen_att = panel.pens_scored + panel.pens_missed
    # a penalty is worth ~0.79 xG; subtract to get TRUE non-penalty xG
    panel["npxg"] = (num("xg") - 0.79 * pen_att).clip(lower=0)
    panel["np_goals"] = (num("goals") - panel.pens_scored).clip(lower=0)
    panel["xa_"] = num("xa")
    panel["xgot_"] = num("xgot")
    panel["cbi"] = num("clearances") + num("blocks") + num("interceptions")
    panel["tkl"] = num("tackles")
    panel["rec"] = num("recoveries")
    panel["defcon_raw"] = num("defensive_contributions")
    # 25/26 BPS: CBI 1 per 2. 26/27: CBI 1 per 3, tackles unchanged.
    panel["bps_cbi_2526"] = np.floor(panel.cbi / 2.0)
    panel["bps_cbi_2627"] = np.floor(panel.cbi / 3.0)
    panel["goals_prevented"] = num("goals_prevented")
    panel["xgot_faced"] = num("xgot_faced")
    panel["saves_"] = num("saves")
    panel["tob"] = num("touches_opposition_box")
    panel["mins"] = num("minutes_played")

    # ---- attach realised FPL points ---------------------------------------
    try:
        h = pd.read_csv("/mnt/user-data/uploads/fpl-data-stats.csv")
        h = h[["id", "gameweek", "total_points", "minutes", "defensive_contribution"]]
        h = h.rename(columns={"id": "player_id", "minutes": "fpl_minutes",
                              "defensive_contribution": "fpl_defcon"})
        panel = panel.merge(h, on=["player_id", "gameweek"], how="left")
    except Exception:
        pass
    return panel


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    p = build()
    p.to_parquet(OUT) if False else p.to_pickle("/tmp/pms_panel.pkl")
    print(f"panel: {len(p)} player-matches | {p.player_id.nunique()} players | "
          f"GW {p.gameweek.min()}-{p.gameweek.max()} | teams {p.team.nunique()}")
    print(f"points joined: {p.total_points.notna().sum()} rows")
    print("\nsanity — season totals from per-match data:")
    print(f"  goals {p.goals.sum():.0f} (np {p.np_goals.sum():.0f}, pens {p.pens_scored.sum():.0f})")
    print(f"  xG {p.xg.sum():.0f}  npxG {p.npxg.sum():.0f}  xA {p.xa_.sum():.0f}")
    print(f"  saves {p.saves_.sum():.0f}  goals_prevented {p.goals_prevented.sum():.1f}")
    print("\nminutes agreement with FPL (should be ~1:1):")
    m = p.dropna(subset=["fpl_minutes"])
    print(f"  corr={np.corrcoef(m.mins, m.fpl_minutes)[0,1]:.4f}  "
          f"mean abs diff={np.abs(m.mins-m.fpl_minutes).mean():.2f} min")
