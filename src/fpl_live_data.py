"""
fpl_live_data.py — REAL data captured from the live FPL API (bootstrap-static)
==============================================================================
Fetched 2026-07-23. IMPORTANT: the endpoint is still serving the **2025/26**
season, not 2026/27:
  * game_config.static_content_url ends ".../plfpl-production/2025_26/"
  * Gameweek 38 (deadline 2026-05-24) is flagged is_current, all events finished
  * the team list contains Burnley, West Ham and Wolves, and does NOT contain
    Coventry, Hull or Ipswich
So the 26/27 game has not opened yet. What IS live and captured here:

 1. FPL's OWN team strength ratings (all 20 teams, venue-split) — a genuinely new
    prior signal, analogous to ClubElo but produced by FPL itself.
 2. The final 2025/26 league table (validates our team-strength posteriors).
 3. The official scoring config (validates every constant in fpl_xp_model).
 4. Real set-piece / penalty orders (previously only illustrative in the demo).

The full 840-player roster can't be transcribed through a fetch, so `signals.
load_fpl_signals()` remains the production path — it reads the same JSON directly
in a networked environment.
"""
import pandas as pd

# ---------------------------------------------------------------------------
# 1. FPL team strength ratings (live, all 20 teams). Scale ~900-1400.
#    Columns: FPL id, name, overall strength tier (1-5), and venue-split
#    attack/defence ratings.
# ---------------------------------------------------------------------------
TEAM_STRENGTH = pd.DataFrame([
    # id, name,            tier, ovr_h, ovr_a, att_h, att_a, def_h, def_a, final_pos
    (1,  "Arsenal",        5, 1305, 1370, 1340, 1390, 1270, 1350, 1),
    (2,  "Aston Villa",    3, 1155, 1230, 1140, 1190, 1170, 1270, 4),
    (3,  "Burnley",        2,  975, 1045,  910, 1050, 1040, 1040, 19),
    (4,  "Bournemouth",    3, 1145, 1200, 1060, 1160, 1230, 1240, 6),
    (5,  "Brentford",      3, 1125, 1215, 1110, 1170, 1140, 1260, 9),
    (6,  "Brighton",       3, 1165, 1230, 1130, 1200, 1200, 1260, 8),
    (7,  "Chelsea",        4, 1185, 1225, 1130, 1170, 1240, 1280, 10),
    (8,  "Crystal Palace", 3, 1140, 1160, 1130, 1150, 1150, 1170, 15),
    (9,  "Everton",        3, 1140, 1145, 1140, 1140, 1140, 1150, 13),
    (10, "Fulham",         3, 1095, 1190, 1080, 1170, 1110, 1210, 11),
    (11, "Leeds",          3, 1095, 1195, 1090, 1160, 1100, 1230, 14),
    (12, "Liverpool",      4, 1235, 1275, 1170, 1200, 1300, 1350, 5),
    (13, "Man City",       4, 1260, 1350, 1190, 1310, 1330, 1390, 2),
    (14, "Man United",     4, 1230, 1250, 1170, 1170, 1290, 1330, 3),
    (15, "Newcastle",      3, 1115, 1205, 1120, 1120, 1110, 1290, 12),
    (16, "Nott'm Forest",  3, 1130, 1160, 1110, 1160, 1150, 1160, 16),
    (17, "Sunderland",     3, 1090, 1155, 1090, 1190, 1090, 1120, 7),
    (18, "Tottenham",      3, 1145, 1145, 1120, 1120, 1150, 1190, 17),
    (19, "West Ham",       3, 1095, 1130, 1060, 1080, 1130, 1180, 18),
    (20, "Wolves",         2,  975, 1060,  950, 1050, 1000, 1070, 20),
], columns=["fpl_id", "team", "tier", "ovr_home", "ovr_away",
            "att_home", "att_away", "def_home", "def_away", "final_pos"])

RELEGATED_2526 = ["West Ham", "Burnley", "Wolves"]      # confirmed: pos 18/19/20
PROMOTED_2627 = ["Coventry", "Hull", "Ipswich"]

# ---------------------------------------------------------------------------
# 2. Official scoring config (game_config.scoring) — validates our constants
# ---------------------------------------------------------------------------
OFFICIAL_SCORING = {
    "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "assists": 3,
    "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},   # per 2 conceded
    "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
    "penalties_saved": 5, "penalties_missed": -2,
    "yellow_cards": -1, "red_cards": -3, "own_goals": -2,
    "long_play": 2, "short_play": 1, "saves": 1, "bonus": 1,
}
# game_settings confirming the rules file: 20 transfers cap, 4 extra saved FTs
# (=5 max), 100.0m budget (squad_total_spend 1000), 3 per club, 15-man squad.
OFFICIAL_SETTINGS = {"squad_size": 15, "squad_play": 11, "team_limit": 3,
                     "total_spend_tenths": 1000, "transfers_cap": 20,
                     "max_extra_free_transfers": 4, "sell_on_fee": 0.5}

# ---------------------------------------------------------------------------
# 3. Real set-piece / penalty orders (sample verified from the live payload).
#    Production path pulls all 840 players via signals.load_fpl_signals().
# ---------------------------------------------------------------------------
SET_PIECE_SAMPLE = pd.DataFrame([
    # name, team, penalties_order, direct_fk_order, corners_order
    ("Saka",      "Arsenal", 1,    2,    2),
    ("Rice",      "Arsenal", None, 1,    1),
    ("Ødegaard",  "Arsenal", 3,    None, 4),
    ("Trossard",  "Arsenal", 4,    None, None),
    ("Madueke",   "Arsenal", None, None, 3),
], columns=["name", "team", "pen_order", "fk_order", "corner_order"])


# ---------------------------------------------------------------------------
# Bridge: FPL team strength -> model prior (same shape as the ClubElo blend)
# ---------------------------------------------------------------------------
def fpl_strength_ratings():
    """Return a frame with centred log attack/defence strength per team, derived
    from FPL's own ratings. Higher = better in BOTH columns (defence is inverted
    so it reads like our model's `defence` parameter)."""
    import numpy as np
    d = TEAM_STRENGTH.copy()
    d["att_raw"] = (d.att_home + d.att_away) / 2
    d["def_raw"] = (d.def_home + d.def_away) / 2
    d["attack"] = np.log(d.att_raw) - np.log(d.att_raw).mean()
    d["defence"] = np.log(d.def_raw) - np.log(d.def_raw).mean()
    d["Elo"] = (d.ovr_home + d.ovr_away) / 2      # ClubElo-compatible column
    return d[["team", "attack", "defence", "Elo", "tier", "final_pos"]]


def as_clubelo_frame():
    """Adapter so FPL strength can be passed anywhere ClubElo is accepted
    (e.g. TeamModel(clubelo=...)). Only the 17 continuing clubs will match
    2026/27; the three relegated sides are dropped by the caller."""
    d = fpl_strength_ratings()
    return d[["team", "Elo"]].copy()


if __name__ == "__main__":
    r = fpl_strength_ratings().sort_values("attack", ascending=False)
    print("FPL's own team strength ratings (live, 2025/26):")
    print(r.round(3).to_string(index=False))
    print(f"\nRelegated (pos 18-20): {RELEGATED_2526}")
    print(f"Promoted for 26/27:    {PROMOTED_2627} (no FPL data yet)")
