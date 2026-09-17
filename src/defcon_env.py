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
        factor = ( xGA_2627[current club] / xGA_ref ) ** beta_pos
- DEF (CBIT, threshold 10): clearances/blocks dominate, ~monotone in xGA -> beta = 1.0.
- MID/FWD (CBIRT, threshold 12): adds recoveries, which a high-press possession side
  generates even at low xGA -> beta < 1 (dampened), so City's press returns some volume.

THE REFERENCE DEPENDS ON WHERE EACH PART OF ALPHA WAS MEASURED (2026-09-16).
`defcon_alpha` = prior share + evidence share (multiseason_priors.to_priors):
- the EVIDENCE share (revert x 25/26 DefCon count) was produced at the player's 25/26
  club, so it is referenced to that club: xGA_2627[club] / xGA_2526[25/26 club];
- the PRIOR share (pooled rate x k0: RATE_DEF_POOLED, a CB/FB role rate, a position rate,
  or a re-listed player's own pool) was pooled over the whole league, so it is referenced
  to the LEAGUE: xGA_2627[club] / mean xGA_2526, and press_factor / 1.0 (press_factor is
  league/ppda, so the league is 1 by construction).
Scaling the whole alpha by the club ratio referenced a league-pooled prior to one club.
For a player whose 25/26 DefCon is all null (22 January arrivals, since exposure dropped
null minutes) alpha IS the prior, and the bias was the full club/league xGA ratio —
downward for anyone whose 25/26 club conceded more than average. A player with no 25/26
club already used the league reference; a pure-prior player with a known club now gets
exactly the same factor, which the selftest checks.

Keyed on player_code (stable). Each share's factor is clipped to `clip` separately.
Requires `defcon_prior_alpha` (ms_priors.pkl built on or after 2026-09-16, or
roster._coldstart_row); a frame without it raises rather than silently scaling the whole
alpha by the club ratio again.
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
    pl["club"] = pl["team_code"].map(teams).map(lambda c: PANEL_TO_FRAME.get(c, c))
    return pl.dropna(subset=["player_code"]).set_index("player_code")["club"].to_dict()


def apply_defcon_environment(players: pd.DataFrame, xga_2627: dict, repo: str,
                             xga_2526: dict = None, clip=(0.6, 1.6),
                             xga_beta=None, press_beta=None, gw=None,
                             code_club: dict = None, press=None):
    """Scale defcon_alpha by the position-aware environment factor: DEF on the team
    xGA ratio (CBIT), MID/FWD on the press-intensity ratio (CBIRT). The prior share of
    alpha is referenced to the league, the evidence share to the 25/26 club.
    Requires 'player_code' and 'defcon_prior_alpha' columns. xga_2627: {team: projected
    mean xGA}. `code_club` / `press` inject the 25/26 club map and the press_index
    module (selftest). Returns a copy."""
    if "player_code" not in players.columns:
        raise KeyError("apply_defcon_environment needs a 'player_code' column")
    if "defcon_prior_alpha" not in players.columns:
        raise KeyError("apply_defcon_environment needs 'defcon_prior_alpha' (the prior's "
                       "share of defcon_alpha) — rebuild ms_priors.pkl with build_all")
    if xga_2526 is None:
        xga_2526 = team_xga_2526()
    xga_beta = xga_beta or XGA_BETA
    press_beta = press_beta or PRESS_BETA
    pix = press
    if pix is None:
        try:
            import press_index as pix
        except Exception:
            pix = None
    if code_club is None:
        code_club = _panel_code_club(repo)
    league_ref = float(np.mean(list(xga_2526.values())))
    out = players.copy(); moved = []
    for i, r in out.iterrows():
        alpha = r["defcon_alpha"]
        if pd.isna(alpha):
            continue
        prior = r["defcon_prior_alpha"]
        if pd.isna(prior):
            raise ValueError(f"defcon_prior_alpha is null for {r.get('web_name')} "
                             f"({r.get('player_code')}) with defcon_alpha set")
        prior = float(np.clip(prior, 0.0, alpha))
        evid = float(alpha) - prior
        ref_club = code_club.get(r["player_code"])
        pos, team = r["pos"], r["team"]
        bx = xga_beta.get(pos, 0.0); bp = press_beta.get(pos, 0.0)
        f_ev = 1.0; f_pr = 1.0
        if bx > 0:                                              # CBIT / xGA channel
            tgt = xga_2627.get(team)
            if tgt is not None:
                ref = xga_2526.get(ref_club, league_ref) if ref_club else league_ref
                ref = ref if ref and ref > 0 else league_ref
                f_ev *= (tgt / ref) ** bx
                f_pr *= (tgt / league_ref) ** bx
        if bp > 0 and pix is not None:                          # CBIRT / press channel
            # `gw` bounds the measured-press revision to gameweeks completed by then.
            # Without it `press_factor` reads EVERY completed gameweek at call time, so a
            # board rebuilt for a past week is conditioned on results from that week and
            # after — look-ahead that inflates any backtest through a club-level,
            # persistent channel. None keeps the old behaviour (all completed), which is
            # correct for a forward board.
            tgt_p = pix.press_factor(team, "2627", gw)          # league/ppda (higher=more press)
            ref_p = pix.press_factor(ref_club, "2526") if ref_club else 1.0
            if ref_p and ref_p > 0:
                f_ev *= (tgt_p / ref_p) ** bp
            f_pr *= tgt_p ** bp                                 # league press_factor == 1
        f_ev = float(np.clip(f_ev, clip[0], clip[1]))
        f_pr = float(np.clip(f_pr, clip[0], clip[1]))
        new = prior * f_pr + evid * f_ev
        out.at[i, "defcon_alpha"] = new
        out.at[i, "defcon_prior_alpha"] = prior * f_pr
        factor = new / alpha if alpha > 0 else 1.0
        if abs(factor - 1.0) > 0.05:
            moved.append((r.get("web_name"), pos, team, round(factor, 2)))
    n_def = sum(1 for m in moved if m[1] == "DEF"); n_mid = len(moved) - n_def
    print(f"[defcon-env] rescaled {len(moved)} players (DEF/xGA: {n_def}, MID-FWD/press: {n_mid})")
    return out


def selftest():
    """Offline. The prior share is referenced to the league, the evidence to the club."""
    import types
    xga26 = {"Hi": 2.0, "Lo": 1.0}                 # league mean 1.5
    xga27 = {"Hi": 2.0, "Lo": 1.0, "Mid": 1.5}
    cc = {1: "Hi", 2: "Hi", 3: "Lo", 4: "Hi"}      # 5 has no 25/26 club
    k0 = 3.0; rate = 7.678
    P = rate * k0
    fr = pd.DataFrame([
        # 1: all-null 25/26 DefCon at a high-xGA club, staying there — pure prior
        {"player_code": 1, "web_name": "prior_known", "pos": "DEF", "team": "Hi",
         "defcon_alpha": P, "defcon_prior_alpha": P, "defcon_beta": k0},
        # 5: the same prior with NO 25/26 club — must get the identical factor
        {"player_code": 5, "web_name": "prior_unknown", "pos": "DEF", "team": "Hi",
         "defcon_alpha": P, "defcon_prior_alpha": P, "defcon_beta": k0},
        # 2: pure evidence at Hi, staying at Hi — environment unchanged, factor 1
        {"player_code": 2, "web_name": "evid_stay", "pos": "DEF", "team": "Hi",
         "defcon_alpha": 50.0, "defcon_prior_alpha": 0.0, "defcon_beta": 8.0},
        # 3: mixed, Lo -> Mid
        {"player_code": 3, "web_name": "mixed_move", "pos": "DEF", "team": "Mid",
         "defcon_alpha": P + 40.0, "defcon_prior_alpha": P, "defcon_beta": 9.0},
        # 4: a midfielder — xGA beta 0, press stubbed off below
        {"player_code": 4, "web_name": "mid", "pos": "MID", "team": "Lo",
         "defcon_alpha": 30.0, "defcon_prior_alpha": 25.0, "defcon_beta": 4.0},
    ])
    nopress = types.SimpleNamespace(press_factor=lambda club, season, gw=None: 1.0)
    o = apply_defcon_environment(fr, xga27, None, xga_2526=xga26, code_club=cc,
                                 press=nopress).set_index("player_code")
    a = o["defcon_alpha"]
    # pure prior: league-referenced, and identical with or without a known 25/26 club
    assert np.isclose(a[1], P * 2.0 / 1.5), a[1]
    assert np.isclose(a[1], a[5]), (a[1], a[5])
    # the old whole-alpha club ratio would have left player 1 at P (Hi/Hi): biased down
    assert a[1] > P
    # pure evidence at an unchanged club is unchanged — the old behaviour, exactly
    assert np.isclose(a[2], 50.0), a[2]
    # mixed: each share on its own reference
    assert np.isclose(a[3], P * 1.5 / 1.5 + 40.0 * 1.5 / 1.0), a[3]
    assert np.isclose(o.loc[3, "defcon_prior_alpha"], P)
    # beta untouched; MID unmoved with press neutral
    assert np.allclose(o["defcon_beta"], fr.set_index("player_code")["defcon_beta"])
    assert np.isclose(a[4], 30.0)

    # press channel: club 25/26 press 2.0 -> 26/27 press 1.0; league reference 1.0
    pf = {("Lo", "2627"): 1.0, ("Hi", "2526"): 2.0}
    stub = types.SimpleNamespace(press_factor=lambda club, season, gw=None: pf[(club, season)])
    cc4 = {4: "Hi"}
    o2 = apply_defcon_environment(fr[fr["player_code"] == 4], xga27, None, xga_2526=xga26,
                                  code_club=cc4, press=stub)
    exp = 25.0 * 1.0 ** 0.5 + 5.0 * (1.0 / 2.0) ** 0.5
    assert np.isclose(o2["defcon_alpha"].iloc[0], exp), (o2["defcon_alpha"].iloc[0], exp)

    # clip is applied per share: evidence ratio 4 clips to 1.6, prior ratio 1 does not
    fr3 = pd.DataFrame([{"player_code": 9, "web_name": "clip", "pos": "DEF", "team": "X",
                         "defcon_alpha": 10.0 + 10.0, "defcon_prior_alpha": 10.0}])
    o3 = apply_defcon_environment(fr3, {"X": 4.0}, None, xga_2526={"Y": 1.0, "Z": 7.0},
                                  code_club={9: "Y"}, press=nopress)
    assert np.isclose(o3["defcon_alpha"].iloc[0], 10.0 * 1.0 + 10.0 * 1.6)

    # a frame without the prior share must fail loudly, not revert to whole-alpha scaling
    try:
        apply_defcon_environment(fr.drop(columns="defcon_prior_alpha"), xga27, None,
                                 xga_2526=xga26, code_club=cc, press=nopress)
        raise AssertionError("missing defcon_prior_alpha did not raise")
    except KeyError:
        pass
    print("defcon_env selftest ok")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
