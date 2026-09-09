from __future__ import annotations
import config
"""
defcon_env.py — condition the DefCon rate on the team defensive-action environment.
===================================================================================
The simulator draws DefCon counts from a player's fitted per-90 rate x minutes, with
NO team term. That treats DefCon as a team-invariant player trait. It is not: DefCon is
repeatable *conditional on environment*. Elliot Anderson banked 52 DefCon points at
Forest (xGA ~1.5); at Man City (xGA ~1.0) the same rate over-projects him, because CBIT/
CBIRT volume scales with how much defending the team does.

Fix: scale each player's defcon rate by a per-player environment factor
        factor = ( xGA_2627[current club] / xGA_ref[25/26 club] ) ** beta_pos
- DEF (CBIT, threshold 10): clearances/blocks dominate, ~monotone in xGA -> beta = 1.0.
- MID/FWD (CBIRT, threshold 12): adds recoveries, which a high-press possession side
  generates even at low xGA -> beta < 1 (dampened), so City's press returns some volume.

Keyed on player_code (stable). Players whose 25/26 club is unknown (new to PL / cold
start) use the league-average xGA as the reference, i.e. environment relative to average.
Clipped to a sane band. Off by default via clip=(1,1); pass real xGA maps to activate.
"""
import numpy as np, pandas as pd

# Two channels, each on its correct driver (DefCon handoff §4.2):
#   DEF  (CBIT, thr 10): clearances/blocks ~ monotone in team xGA        -> XGA_BETA
#   MID/FWD (CBIRT, thr 12): recoveries ~ press intensity (PPDA), NOT xGA -> PRESS_BETA
# A high-press side generates recoveries even at low xGA (Anderson holds up at City
# under Maresca; Solio 56% confirms). Press conditioning captures that; xGA would not.
XGA_BETA = {"DEF": 1.0, "GK": 0.0, "MID": 0.0, "FWD": 0.0}
PRESS_BETA = {"DEF": 0.0, "GK": 0.0, "MID": 0.5, "FWD": 0.5}


def team_xga_2526(e0_path=config.E0_RECON):
    """25/26 xGA per team (reconstructed def_xg = goals-against rate)."""
    import betting_features as bf
    return bf.team_ratings(bf.build(e0_path))["def_xg"].to_dict()


def _panel_code_club(repo, season="2025-2026"):
    """player_code -> 25/26 club name (frame naming)."""
    from regime_panel import PANEL_TO_FRAME
    pl = pd.read_csv(f"{repo}/{season}/players.csv")
    teams = pd.read_csv(f"{repo}/{season}/teams.csv").set_index("code")["name"].to_dict()
    pl["club"] = pl.team_code.map(teams).map(lambda c: PANEL_TO_FRAME.get(c, c))
    return pl.dropna(subset=["player_code"]).set_index("player_code")["club"].to_dict()


def apply_defcon_environment(players: pd.DataFrame, xga_2627: dict, repo: str,
                             xga_2526: dict = None, clip=(0.6, 1.6),
                             xga_beta=None, press_beta=None, gw=None):
    """Scale defcon_alpha by the position-aware environment factor: DEF on the team
    xGA ratio (CBIT), MID/FWD on the press-intensity ratio (CBIRT). Requires a
    'player_code' column. xga_2627: {team: projected mean xGA}. Returns a copy."""
    if "player_code" not in players.columns:
        raise KeyError("apply_defcon_environment needs a 'player_code' column")
    if xga_2526 is None:
        xga_2526 = team_xga_2526()
    xga_beta = xga_beta or XGA_BETA
    press_beta = press_beta or PRESS_BETA
    try:
        import press_index as pix
    except Exception:
        pix = None
    code_club = _panel_code_club(repo)
    league_ref = float(np.mean(list(xga_2526.values())))
    out = players.copy(); moved = []
    for i, r in out.iterrows():
        if pd.isna(r.get("defcon_alpha")):
            continue
        ref_club = code_club.get(r.get("player_code"))
        bx = xga_beta.get(r.pos, 0.0); bp = press_beta.get(r.pos, 0.0)
        factor = 1.0
        if bx > 0:                                              # CBIT / xGA channel
            tgt = xga_2627.get(r.team)
            if tgt is not None:
                ref = xga_2526.get(ref_club, league_ref) if ref_club else league_ref
                ref = ref if ref and ref > 0 else league_ref
                factor *= (tgt / ref) ** bx
        if bp > 0 and pix is not None:                          # CBIRT / press channel
            # `gw` bounds the measured-press revision to gameweeks completed by then.
            # Without it `press_factor` reads EVERY completed gameweek at call time, so a
            # board rebuilt for a past week is conditioned on results from that week and
            # after — look-ahead that inflates any backtest through a club-level,
            # persistent channel. None keeps the old behaviour (all completed), which is
            # correct for a forward board.
            tgt_p = pix.press_factor(r.team, "2627", gw)        # league/ppda (higher=more press)
            ref_p = pix.press_factor(ref_club, "2526") if ref_club else 1.0
            if ref_p and ref_p > 0:
                factor *= (tgt_p / ref_p) ** bp
        factor = float(np.clip(factor, clip[0], clip[1]))
        out.at[i, "defcon_alpha"] = r.defcon_alpha * factor
        if abs(factor - 1.0) > 0.05:
            moved.append((r.web_name, r.pos, r.team, round(factor, 2)))
    n_def = sum(1 for m in moved if m[1] == "DEF"); n_mid = len(moved) - n_def
    print(f"[defcon-env] rescaled {len(moved)} players (DEF/xGA: {n_def}, MID-FWD/press: {n_mid})")
    return out
