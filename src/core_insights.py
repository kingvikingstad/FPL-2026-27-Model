from __future__ import annotations
import config
"""
core_insights.py — ingest the REAL 2026/27 dataset (FPL-Core-Insights)
======================================================================
Files (from data/2026-2027/ in olbauday/FPL-Core-Insights):

  players.csv            555 players: player_id, web_name, team_code, position
  playerstats.csv        pre-season snapshot (gw=1): now_cost, selected_by_percent,
                         status, chance_of_playing_*, ep_next, set-piece orders,
                         set_piece_threat, xG/xA/xGI per-90 priors, ICT
  teams.csv              20 teams incl. Coventry/Hull/Ipswich, with **Elo**
  team_history.csv       player_id x gw -> team_code  (tracks in-season moves)
  gameweek_summaries.csv 38 deadlines; GW1 = 2026-08-21, nothing finished yet

This replaces every stopgap the model was using:
  * demo roster            -> real 555-player roster, transfers already resolved
  * 25/26 end-of-season £  -> real 26/27 launch prices
  * illustrative signals   -> real status / chance_of_playing / set-piece orders
  * ClubElo name-matching  -> Elo shipped in teams.csv, keyed by FPL team code
  * guessed promoted prior -> Elo-derived prior for Coventry/Hull/Ipswich
"""
import numpy as np, pandas as pd

POS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD"}
# Core-Insights names -> the short names used across the model / schedule
TEAM_NORM = {
    "Man Utd": "Man United", "Spurs": "Tottenham", "Coventry City": "Coventry",
    "Hull City": "Hull", "Ipswich Town": "Ipswich", "Nott'm Forest": "Nott'm Forest",
}
def norm_team(n): return TEAM_NORM.get(str(n).strip(), str(n).strip())

UPLOADS = "/mnt/user-data/uploads"


def load(base=UPLOADS):
    """Return (players, teams, gameweeks) with everything joined and normalised."""
    p = pd.read_csv(f"{base}/players.csv")
    s = pd.read_csv(f"{base}/playerstats.csv")
    t = pd.read_csv(f"{base}/teams.csv")
    gw = pd.read_csv(f"{base}/gameweek_summaries.csv")

    s = s.drop(columns=[c for c in ["first_name", "second_name", "web_name"] if c in s.columns])
    d = s.merge(p, left_on="id", right_on="player_id", how="left")
    d = d.merge(t[["code", "name", "short_name", "elo"]], left_on="team_code",
                right_on="code", how="left")
    d["pos"] = d["position"].map(POS)
    d["team"] = d["name"].map(norm_team)
    t = t.assign(team=t["name"].map(norm_team))
    return d, t, gw


# ---------------------------------------------------------------------------
# Bridges into the existing model
# ---------------------------------------------------------------------------
def to_roster(d):
    """Roster frame for roster.build_players_2627 (prices already in millions)."""
    return pd.DataFrame({
        "name": d.web_name, "team": d.team, "pos": d.pos,
        "price": pd.to_numeric(d.now_cost, errors="coerce"),
        "own": pd.to_numeric(d.selected_by_percent, errors="coerce").fillna(0.0),
    })


def to_signals(d):
    """Signal frame for signals.apply_availability / apply_setpieces."""
    chance = pd.to_numeric(d.get("chance_of_playing_next_round"), errors="coerce")
    return pd.DataFrame({
        "name": d.web_name,
        "status": d.get("status", "a").fillna("a"),
        # FPL leaves chance null when there is no doubt -> treat as fully available
        "chance_play": (chance.fillna(100) / 100.0).clip(0, 1),
        "news": "",
        "ep_next": pd.to_numeric(d.get("ep_next"), errors="coerce").fillna(0.0),
        "pen_order": pd.to_numeric(d.get("penalties_order"), errors="coerce"),
        "fk_order": pd.to_numeric(d.get("direct_freekicks_order"), errors="coerce"),
        "corner_order": pd.to_numeric(d.get("corners_and_indirect_freekicks_order"), errors="coerce"),
    })


def to_elo_frame(t):
    """Elo for ALL 20 teams of 26/27 — including the promoted three, which the
    model previously had to cover with a generic promoted prior."""
    return t[["team", "elo"]].rename(columns={"team": "team", "elo": "Elo"})


def promoted_prior_from_elo(t, promoted=("Coventry", "Hull", "Ipswich"), scale=0.0016):
    """Convert the Elo gap to a centred log attack/defence prior for the promoted
    clubs, replacing the history-calibrated generic prior. `scale` maps Elo points
    to log-strength; 0.0016 ~ a 400-Elo gap being worth ~0.64 in log terms, in
    line with the 30-season promoted-team estimate (-0.29 attack)."""
    e = t.set_index("team")["elo"].astype(float)
    centred = e - e.mean()
    out = {}
    for club in promoted:
        if club in centred.index:
            out[club] = float(centred[club] * scale)
    if not out:
        return None
    vals = np.array(list(out.values()))
    return {"per_club": out, "mean": float(vals.mean()), "sd": float(max(vals.std(), 0.12))}


def price_changes_vs_last_season(d, hist_csv=config.FPL_DATA_STATS):
    """Compare 26/27 launch prices with 25/26 end-of-season prices."""
    try:
        h = pd.read_csv(hist_csv).sort_values(["id", "gameweek"])
        last = h.groupby("web_name").now_cost.last()
    except Exception:
        return pd.DataFrame()
    m = d[["web_name", "pos", "team", "now_cost", "selected_by_percent"]].copy()
    m["price_2526"] = m.web_name.map(last)
    m["delta"] = m.now_cost - m.price_2526
    return m.dropna(subset=["price_2526"]).sort_values("delta", ascending=False)


if __name__ == "__main__":
    d, t, gw = load()
    print(f"players {len(d)} | teams {len(t)} | gameweeks {len(gw)}")
    print(f"GW1 deadline: {gw.loc[gw.id == 1, 'deadline_time'].iloc[0]}")
    print(f"flagged (not 'a'): {(d.status != 'a').sum()}")
    print(f"penalty takers (order 1): {(d.penalties_order == 1).sum()}")
    print("\nElo-derived promoted prior:", promoted_prior_from_elo(t))
