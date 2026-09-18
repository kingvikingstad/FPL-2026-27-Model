import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
export_projection_detail.py — one row per player per gameweek, with everything behind it
=========================================================================================
`gw_board_long.csv` gives the projected score but not the reasoning. This joins that same
projection to the two things needed to audit or argue with it:

  * THE FIXTURE — opponent, venue, the opponent's attack and defence strength on the
    model's own scale, and the per-fixture lambdas the projection is built from
    (lam_for, lam_against). "Relative to the strength of the team they are playing" is not
    a separate calculation here; it IS the projection, so the lambdas are exported rather
    than a proxy difficulty rating.
  * THE TEAM LAYER BEHIND THAT FIXTURE — the full betting-odds provenance chain for the
    player's own club (published title/relegation decimal odds -> de-vigged probabilities
    -> market strength z -> market Elo -> blend with ClubElo -> fitted posterior attack
    and defence), the derived match-outcome distribution (win/draw/loss, clean sheet,
    BTTS), and the press/PPDA covariate that conditions the MID/FWD DefCon channel. These
    come from `export_team_projections.build()` rather than a second TeamModel fit, so the
    team file and this file are guaranteed to agree.
  * EVERY INSTALLED PRIOR — Gamma shapes for attacking involvement, assists and DefCon
    (with their implied per-90 rates), the Beta for starts, expected minutes given a
    start, penalty rate, and set-piece duty. These are what the simulation draws from, so
    a surprising projection can be traced to the prior that produced it.

IT ANNOTATES THE BOARD, IT DOES NOT RECOMPUTE IT
-------------------------------------------------
An earlier version re-ran `project()` to rebuild the numbers. That was wrong. The
simulation consumes a seeded RNG stream player by player, so any difference in ROW ORDER
hands each player a different draw and the output diverges by Monte Carlo noise — measured
at 0.066 mean and 0.546 max, which looks like a modelling disagreement and is not one.
The projection is read from the board and treated as authoritative. Only team-level
quantities are recomputed, and those do not depend on player ordering.

JOIN KEYS
---------
On (player, team, gw), never on name alone: 17 web_names are shared by two different
players (there is a Dasilva at Coventry and another at Brentford), and a name-keyed join
silently fans out. A row-count assertion enforces this.

Run:  python scripts/export_projection_detail.py
Out:  outputs/projection_detail_gw1_10.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import core_insights as ci, signals as sg, starter_prior as sp
import bayes_model
import inseason
from roster import calibrate_cold_start, _coldstart_row

# Must match the board's draw count. The team sampler is seeded (rng=default_rng(7)) but
# S changes the draw stream itself, so a different S reports lambdas the board never used.
S = int(_os.environ.get("DRAWS", "3000"))
GW_HI = int(_os.environ.get("GW_HI", "10"))
# Must match the board. gw_board applies predicted XIs to ONE gameweek and this
# export exists to explain that board, so it has to name the same week — derived by the
# SAME function. This was a hardcoded 1 until 2026-09-10, so on a GW4 board the
# `p_start_prior_predxi` column described GW1's team news.
PRED_XI_GW = int(_os.environ.get("PRED_XI_GW") or inseason.next_open_gw())
OUT = _os.path.join(config.OUTPUTS, f"projection_detail_gw1_{GW_HI}.csv")
PRIOR_COLS = ["npxgi_alpha", "npxgi_beta", "xa_alpha", "xa_beta",
              "defcon_alpha", "defcon_beta", "start_a", "start_b",
              "sub_app_rate", "exp_minutes"]


def player_priors():
    """The prior set as actually installed, after cold-start fill, regime adjustment,
    availability and the set-piece override — i.e. what the simulation saw."""
    d26, t26, _ = ci.load(base=config.repo("2026-2027")); sig = ci.to_signals(d26)
    cal_cs = calibrate_cold_start(hist_csv=config.COLDSTART_HIST)
    cal_own = sp.calibrate_ownership_start()
    pri = pd.read_pickle(config.MS_PRIORS).dropna(subset=["player_code"]) \
            .drop_duplicates("player_code")
    pmap = pri.set_index("player_code")
    rows = []
    for _, r in d26.iterrows():
        c = r.get("player_code")
        if pd.notna(c) and c in pmap.index:
            s = pmap.loc[c]
            row = {k: s[k] for k in PRIOR_COLS}
            row.update({"player_code": c, "web_name": r.web_name, "pos": r.pos,
                        "team": r.team, "minutes_prior": s["minutes"],
                        "cold_start": False,
                        "pen_xg90": float(s.get("pen_xg90_measured", 0) or 0)})
        else:
            row = _coldstart_row(r.web_name, r.team, r.pos, r.now_cost,
                                 r.selected_by_percent, cal_cs)
            row.update({"minutes_prior": 0.0, "player_code": c, "cold_start": True})
        rows.append(row)
    pl = pd.DataFrame(rows)
    pl = sp.apply_coldstart_depth(pl, cal_own)
    # Mirror the board's flags EXACTLY. This export exists to explain the board, so a
    # prior column that differs from the one the simulation drew from is worse than no
    # column at all. Previously this hardcoded regime={} and skipped the panel split,
    # which was silently correct only while both were off by default.
    if _os.environ.get("REGIME_PANEL") == "on":
        import regime_panel as rpn
        pl = rpn.apply_regime_panel_split(pl, pd.read_pickle(config.PMS_PANEL),
                                          config.REPO, w_pre=0.30)
    _regime, _regime_label = sp.resolve_regime()
    print(f"[export] regime: {_regime_label}")
    pl = sp.apply_regime_uncertainty(pl, regime=_regime, cal=cal_own, k_min=900)
    # Mirror the board's in-season and XI-constraint layers, in the board's order. This
    # export exists to EXPLAIN the board, so a prior column built without a layer the
    # simulation used describes a run that never happened.
    if _os.environ.get("INSEASON", "on").lower() in ("on", "1", "true"):
        import inseason as _ins
        _upto = int(_os.environ.get("INSEASON_UPTO", "38"))
        # Both minutes-prior levers are mirrored here for the reason stated above: this
        # export explains the board, so a lever the board applied and this did not would
        # produce a p_start_prior column describing a run that never happened. The cap was
        # previously missing, which made the two diverge whenever INSEASON_KAPPA was set.
        # ON by default since 2026-09-17, mirroring gw_board.py — see the note at its
        # `_kap` resolution and PROJECT_KNOWLEDGE §6.9. `INSEASON_KAPPA=off` disables it
        # in both places, and this export must follow the board or the p_start_prior
        # column it publishes describes a run that never happened.
        _k = _os.environ.get("INSEASON_KAPPA", str(_ins.START_KAPPA)).strip()
        if _k.lower() in ("off", "0", "none", "inf"):
            _k = ""
        if _k:
            pl, _ = _ins.cap_start_prior(pl, kappa=float(_k), verbose=False)
        _l = _os.environ.get("INSEASON_LAM", "").strip()
        _apps = _ins.appearances(upto_gw=_upto, verbose=False,
                                 lam=float(_l) if _l else _ins.RECENCY_LAM)
        pl, _ = _ins.update_minutes(
            pl, _apps, weight=float(_os.environ.get("INSEASON_W_MIN", _ins.W_MINUTES)),
            verbose=False)
        if _os.environ.get("INSEASON_EXP_MINUTES", "off").lower() in ("on", "1", "true"):
            pl, _ = _ins.update_exp_minutes(
                pl, _apps, k_half=float(_os.environ.get("INSEASON_EXP_K",
                                                        _ins.EXP_MINUTES_K)),
                verbose=False)
        pl, _ = _ins.update_rates(
            pl, _ins.rates(upto_gw=_upto, verbose=False),
            weight=float(_os.environ.get("INSEASON_W_RATE", _ins.W_RATE)),
            verbose=False)
    # Same availability source as the board, or the exported p_start_prior describes a
    # simulation that never ran. See gw_board.py for why live is preferred.
    sig["player_code"] = d26["player_code"].values
    if _os.environ.get("LIVE_FPL", "on").lower() not in ("off", "0"):
        try:
            _live = sg.fetch_live_signals()
            if d26["player_code"].isin(_live["player_code"]).mean() >= 0.9:
                _sp = sig[["player_code", "pen_order", "fk_order", "corner_order"]]
                sig = _live.merge(_sp, on="player_code", how="left")
                print("[export] availability: live FPL endpoint")
        except Exception as e:
            print(f"[export] live availability unavailable ({type(e).__name__}); snapshot")
    pl = sg.apply_availability(pl, sig, lineups=None)
    # AFTER availability, with the ruled-out held — the board's order since 2026-09-08.
    # This ran BEFORE availability until 2026-09-10, the ordering the board abandoned
    # because it normalised each club to eleven and then let injuries delete players out
    # of that eleven (league 220.0 -> 184.9 expected starters). So every p_start_prior
    # here was deflated at exactly the clubs with the most team news.
    if _os.environ.get("XI_CONSTRAINT", "on").lower() not in ("off", "0"):
        pl, _ = sp.apply_xi_constraint(pl, hold=pl.get("avail_ruled_out"), verbose=False)
    # Default "fpl", the board's since 2026-09-08. This still defaulted to "observed" —
    # the n=1 estimator studies/penalty_assignment.py argues against, which let a single
    # realised penalty overturn a declared taker — so the exported pen_xg90 credited
    # penalties to players the board did not.
    _spt = _os.environ.get("FPL_SETPIECE", "fpl").lower()
    import set_piece_takers as spt
    if _spt not in ("off", "0", "fpl", "observed"):
        sig = spt.apply_to_signals(sig, d26[["web_name", "team", "player_code"]],
                                   mode="fill" if _spt == "fill" else "override",
                                   verbose=False)
    # player_code, not web_name — see set_piece_takers.pen1_codes. Same window as the
    # board: 38 means "no cut", which press_measured and observed_takers read as None.
    _upto_all = int(_os.environ.get("INSEASON_UPTO", "38"))
    _pen_src = d26[["web_name", "team", "player_code"]].copy()
    if "penalties_order" in d26.columns:
        _pen_src["penalties_order"] = d26["penalties_order"]
    pen1 = spt.pen1_codes(_pen_src, mode=_spt,
                          upto_gw=None if _upto_all >= 38 else _upto_all)
    pl["pen_xg90"] = np.where(pl.player_code.isin(pen1),
                              pl.pen_xg90.fillna(0).clip(lower=0.10), 0.0)
    # Cold-start players carry no exp_minutes, so `_minutes_if_start` falls back to the
    # positional constant. Leaving the column null would hide the number the simulation
    # actually used, which defeats the point of this export — so fill it and say so.
    pos_min = bayes_model.MINUTES_IF_START
    have = pl["exp_minutes"].notna() if "exp_minutes" in pl.columns else pd.Series(
        False, index=pl.index)
    if "exp_minutes" not in pl.columns:
        pl["exp_minutes"] = np.nan
    pl["exp_minutes_src"] = np.where(have, "player history", "positional fallback")
    pl["exp_minutes"] = np.where(have, pl["exp_minutes"],
                                 pl["pos"].map(pos_min).fillna(85.3))
    pl["p_start_prior"] = pl["start_a"] / (pl["start_a"] + pl["start_b"])
    # Predicted-XI state is gameweek-specific, so it cannot replace the base prior in a
    # frame whose rows span ten gameweeks. Both are exported: `p_start_prior` is what
    # GW2+ projected from, `p_start_prior_predxi` is what the team-news gameweek used.
    pl["p_start_prior_predxi"] = pl["p_start_prior"]
    if _os.environ.get("PRED_XI", "on").lower() not in ("off", "0"):
        try:
            import predicted_xi as pxi
            _sq = d26[["web_name", "team", "player_code", "first_name", "second_name"]]
            # same multi-source consensus the board uses, or the exported prior
            # describes a simulation that never ran
            _cons = pxi.consensus(_sq, gw=PRED_XI_GW, verbose=False)
            if len(_cons):
                _plp, _ = pxi.apply_consensus(pl, _cons, _sq, verbose=False)
            else:
                _plp, _ = pxi.apply_soft(
                    pl, pxi.resolve(pxi.load(gw=PRED_XI_GW), _sq, verbose=False),
                    verbose=False)
            # the board caps a lineup source at what the injury feed permits; mirror it
            _plp, _ = pxi.apply_injury_ceiling(_plp, sig, verbose=False)
            _plp, _ = pxi.apply_minutes_caps(_plp, gw=PRED_XI_GW, verbose=False)
            pl["p_start_prior_predxi"] = (_plp["start_a"] /
                                          (_plp["start_a"] + _plp["start_b"]))
            n = int((pl["p_start_prior_predxi"] - pl["p_start_prior"]).abs().gt(0.25).sum())
            print(f"[export] predicted XI: {n} players move >0.25 in the team-news gameweek")
        except Exception as e:
            print(f"[export] predicted XI skipped ({type(e).__name__})")
    pl["npxgi_rate90"] = pl["npxgi_alpha"] / pl["npxgi_beta"]
    pl["xa_rate90"] = pl["xa_alpha"] / pl["xa_beta"]
    pl["defcon_rate90"] = pl["defcon_alpha"] / pl["defcon_beta"]
    return pl, sig, d26, t26


def fixtures():
    """The team layer, taken from export_team_projections so the two files cannot
    disagree. That export owns the team model (odds -> market Elo -> blend -> fitted
    posterior); duplicating the build here previously meant two TeamModel fits that
    could drift apart under a flag change. Team-level only, so nothing depends on
    player row order.

    Returns (per team-gameweek frame, per-team season frame)."""
    import export_team_projections as etp
    # Pin the team layer to THIS export's window. The two used to agree only because both
    # read GW_HI with the same default; that default is 38 there and 10 here since
    # 2026-09-10, and the season table's window sums (xga_env, exp_pts) depend on it.
    etp.GW_HI = GW_HI
    T, G = etp.build()

    keep_gw = ["team", "gw", "opponent", "is_home", "lam_for", "lam_for_p5",
               "lam_for_p95", "lam_against", "lam_against_p5", "lam_against_p95",
               "p_clean_sheet", "p_concede_2plus", "p_win", "p_draw", "p_loss",
               "p_btts", "exp_league_pts", "home_discount_applied"]
    F = G[keep_gw].copy()
    F = F.rename(columns={"p_clean_sheet": "clean_sheet_prob"})

    # opponent-side context
    opp = T[["team", "att_strength", "def_strength", "att_rank", "def_rank",
             "market_strength_z", "blended_elo"]].rename(columns={
        "team": "opponent", "att_strength": "opp_att_strength",
        "def_strength": "opp_def_strength", "att_rank": "opp_att_rank",
        "def_rank": "opp_def_rank", "market_strength_z": "opp_market_z",
        "blended_elo": "opp_blended_elo"})
    F = F.merge(opp, on="opponent", how="left")

    # own-side context, including the betting-odds provenance chain
    own = T[["team", "att_strength", "def_strength", "att_rank", "def_rank",
             "title_odds_dec", "releg_odds_dec", "p_title_devig", "p_relegation",
             "market_strength_z", "market_elo", "clubelo_base", "blended_elo",
             "market_weight", "promoted", "ppda_2627", "press_factor",
             "press_is_new_manager_estimate", "xga_env"]].rename(columns={
        "att_strength": "own_att_strength", "def_strength": "own_def_strength",
        "att_rank": "own_att_rank", "def_rank": "own_def_rank",
        "market_strength_z": "own_market_z", "blended_elo": "own_blended_elo"})
    F = F.merge(own, on="team", how="left")

    F["attack_ease"] = F["lam_for"] / F["lam_for"].mean()
    return F


def main():
    board = pd.read_csv(_os.path.join(config.OUTPUTS, "gw_board_long.csv"))
    n0 = len(board)
    assert not board.duplicated(["player", "team", "gw"]).any(), \
        "board has duplicate (player, team, gw) — the join key is not unique"
    print(f"[export] board: {n0} rows, {board.groupby(['player','team']).ngroups} players")

    pl, sig, d26, t26 = player_priors()
    F = fixtures()
    print(f"[export] fixtures: {len(F)} team-gameweeks over GW1-{GW_HI}")

    out = board.merge(F, on=["team", "gw"], how="left")
    assert len(out) == n0, f"fixture join fanned out: {n0} -> {len(out)}"

    pcols = (["web_name", "team", "player_code", "cold_start", "minutes_prior",
              "p_start_prior", "p_start_prior_predxi", "npxgi_rate90", "xa_rate90",
              "defcon_rate90", "pen_xg90", "exp_minutes_src"] + PRIOR_COLS)
    P = pl[[c for c in pcols if c in pl.columns]].rename(columns={"web_name": "player"})
    P = P.drop_duplicates(["player", "team"])
    # The BOARD is the authority on player_code now that gw_board writes it, so drop the
    # prior frame's copy before joining on (player, team). Leaving both makes pandas
    # suffix them to player_code_x / _y, and the name join further down — which keys on a
    # plain `player_code` — then fails with a bare KeyError. Regression introduced when
    # player_code was added to the board output on 2026-08-22.
    if "player_code" in P.columns and "player_code" in out.columns:
        P = P.drop(columns=["player_code"])
    out = out.merge(P, on=["player", "team"], how="left")
    assert len(out) == n0, f"prior join fanned out: {n0} -> {len(out)}"

    # set-piece duty, keyed on name+club for the same reason
    s = sig.copy()
    s["player"] = s["name"]
    name_team = d26[["web_name", "team"]].rename(columns={"web_name": "player"})
    s = s.merge(name_team, on="player", how="left").drop_duplicates(["player", "team"])
    out = out.merge(s[["player", "team", "pen_order", "fk_order", "corner_order"]],
                    on=["player", "team"], how="left")
    assert len(out) == n0, f"set-piece join fanned out: {n0} -> {len(out)}"

    # Disambiguated names. `player` is FPL's web_name and is NOT unique — 15 surnames
    # in this squad cover 32 players — so anything keyed or charted on it merges people.
    import player_names as pn
    nm = pn.canonical(d26, t26)[["player_code", "display_name", "unique_label",
                                 "team_code", "name_ambiguous"]]
    n_before = len(out)
    out = out.merge(nm, on="player_code", how="left")
    assert len(out) == n_before, "name join fanned out"
    front = ["display_name", "unique_label", "team", "team_code", "pos", "gw"]
    out = out[[c for c in front if c in out.columns] +
              [c for c in out.columns if c not in front]]
    print(f"[export] names: {int(out.drop_duplicates('player_code')['name_ambiguous'].sum())} "
          f"players carry a shared surname and are qualified in display_name")

    out = out.sort_values(["gw", "mean"], ascending=[True, False])
    out.to_csv(OUT, index=False)
    print(f"[export] wrote {len(out)} rows x {len(out.columns)} columns -> {OUT}")
    miss = out["lam_for"].isna().sum()
    print(f"[export] rows without a fixture (blank gameweek): {miss}")
    print(f"[export] cold-start players: {int(out.drop_duplicates(['player','team'])['cold_start'].sum())}")


if __name__ == "__main__":
    main()
