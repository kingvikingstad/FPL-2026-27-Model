"""
regime_panel.py — appointment-date panel split for PARTIAL-regime clubs.
=========================================================================
For clubs whose current manager took over PART-WAY through 25/26 (Carrick at
Man United, De Zerbi at Spurs), a flat minutes multiplier (delta) discounts the
whole season uniformly and throws away the information that a chunk of 25/26 was
already played under the current regime. This is the cleaner alternative flagged
in the regime note: weight post-appointment matches at full strength and only
discount the pre-appointment ones, then rebuild the start (minutes) prior from
the re-weighted panel.

Effect: incumbents who were regulars under the NEW manager get a tighter, more
relevant start prior; players who only featured under the OLD manager are
correctly discounted — without the blunt across-the-board haircut.

Scope: applies only to players who were at the partial-regime club in 25/26.
Transferred-in players (Mbeumo/Šeško/Cunha at Man Utd) have no current-regime
data at the new club and are left to the cold-start / two-season path untouched.

Off by default. `partial_regime=None` -> no-op. All settings sweepable.
"""
from __future__ import annotations
import numpy as np, pandas as pd

# {26/27 club name : appointment gameweek in 25/26}. Matches at gw >= this are
# current-regime. Sample sizes from the regime note (Man Utd ~19, Spurs ~9).
PARTIAL_REGIME_CLUBS_2627 = {"Man United": 20, "Tottenham": 30}

# 25/26 panel club names -> 26/27 frame club names
PANEL_TO_FRAME = {"Man Utd": "Man United", "Spurs": "Tottenham"}




def _panel_club_code_map(panel, repo):
    """panel player_id -> (frame_club, player_code) using 25/26 players.csv."""
    pl = pd.read_csv(f"{repo}/2025-2026/players.csv")
    teams = pd.read_csv(f"{repo}/2025-2026/teams.csv").set_index("code")["name"].to_dict()
    pl["club"] = pl.team_code.map(teams).map(lambda c: PANEL_TO_FRAME.get(c, c))
    club = pl.set_index("player_id")["club"].to_dict()
    code = pl.set_index("player_id")["player_code"].to_dict()
    return club, code


def regime_start_priors(panel: pd.DataFrame, repo: str, partial_regime=None,
                        w_pre=0.30, a0=2.0, b0=2.0, revert=0.70) -> pd.DataFrame:
    """Per-player-CODE regime-weighted (start_a, start_b) for partial-regime
    incumbents. Keyed on player_code (stable across seasons), never player_id
    (reassigned). w_pre discounts pre-appointment matches (post = 1.0); eff_frac
    is the regime-weighted / total games ratio for effective-minutes shrinkage."""
    partial = PARTIAL_REGIME_CLUBS_2627 if partial_regime is None else partial_regime
    club, code = _panel_club_code_map(panel, repo)
    p = panel.copy()
    p["frame_club"] = p.player_id.map(club)
    p["player_code"] = p.player_id.map(code)
    p = p[p.frame_club.isin(partial) & p.player_code.notna()].copy()
    if not len(p):
        return pd.DataFrame(columns=["player_code", "frame_club", "start_a", "start_b",
                                     "wgames", "wstarts", "eff_frac"])
    p["appt_gw"] = p.frame_club.map(partial)
    p["w"] = np.where(p.gameweek >= p.appt_gw, 1.0, w_pre)
    p["is_start"] = (p.mins >= 60).astype(float)
    g = p.groupby("player_code").apply(lambda d: pd.Series({
        "wgames": d.w.sum(), "wstarts": (d.w * d.is_start).sum(),
        "games": float(len(d)), "frame_club": d.frame_club.iloc[0]})).reset_index()
    g["start_a"] = a0 + revert * g.wstarts
    g["start_b"] = b0 + revert * (g.wgames - g.wstarts).clip(lower=0)
    g["eff_frac"] = (g.wgames / g.games).clip(0.0, 1.0)
    return g[["player_code", "frame_club", "start_a", "start_b", "wgames", "wstarts", "eff_frac"]]


def apply_regime_panel_split(players: pd.DataFrame, panel: pd.DataFrame, repo: str,
                             partial_regime=None, w_pre=0.30) -> pd.DataFrame:
    """Override start_a/start_b for partial-regime INCUMBENTS (at the club in both
    25/26 and 26/27) with the appointment-weighted prior, and shrink their effective
    minutes to the current-regime sample. Matches on player_code. `players` must
    carry a 'player_code' column. Non-partial clubs and players who left are
    untouched. Apply BEFORE apply_minutes_shrinkage."""
    if "player_code" not in players.columns:
        raise KeyError("apply_regime_panel_split needs a 'player_code' column on the frame")
    rp = regime_start_priors(panel, repo, partial_regime=partial_regime, w_pre=w_pre)
    if not len(rp):
        return players
    out = players.copy()
    rp_map = rp.set_index("player_code")
    n = 0
    for i, r in out.iterrows():
        pc = r.get("player_code")
        if pd.isna(pc) or pc not in rp_map.index or bool(r.get("cold_start", False)):
            continue
        if r.get("team") != rp_map.loc[pc, "frame_club"]:     # must still be at the club
            continue
        out.at[i, "start_a"] = float(rp_map.loc[pc, "start_a"])
        out.at[i, "start_b"] = float(rp_map.loc[pc, "start_b"])
        if "minutes" in out.columns:
            out.at[i, "minutes"] = float(r["minutes"]) * float(rp_map.loc[pc, "eff_frac"])
        n += 1
    print(f"[regime-panel] appointment-weighted start prior applied to {n} incumbents "
          f"({', '.join(sorted(set(rp.frame_club)))})")
    return out
