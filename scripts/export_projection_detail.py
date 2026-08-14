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
from bayes_model import TeamModel
from schedule_2627 import schedule
from roster import calibrate_cold_start, _coldstart_row

S = 1500
GW_HI = 10
OUT = _os.path.join(config.OUTPUTS, "projection_detail_gw1_10.csv")
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
    pl = sp.apply_regime_uncertainty(pl, regime={}, cal=cal_own, k_min=900)
    pl = sg.apply_availability(pl, sig, lineups=None)
    if _os.environ.get("FPL_SETPIECE", "override").lower() not in ("off", "0"):
        import set_piece_takers as spt
        sig = spt.apply_to_signals(sig, d26[["web_name", "team", "player_code"]],
                                   mode="override", verbose=False)
    pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())
    pl["pen_xg90"] = np.where(pl.web_name.str.lower().str.strip().isin(pen1),
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
    pl["npxgi_rate90"] = pl["npxgi_alpha"] / pl["npxgi_beta"]
    pl["xa_rate90"] = pl["xa_alpha"] / pl["xa_beta"]
    pl["defcon_rate90"] = pl["defcon_alpha"] / pl["defcon_beta"]
    return pl, sig, d26, t26


def fixtures(t26):
    """Team-level only, so nothing here depends on player row order."""
    elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
    if _os.environ.get("MARKET_ODDS") != "off":
        import market_odds as mo
        elo = mo.blended_elo(elo, market_weight=float(
            _os.environ.get("MARKET_WEIGHT", "0.6")))
    tm = TeamModel(promoted_per_club=pclub).fit(e0_path=config.E0_RECON, clubelo=elo,
                                                clubelo_weight=0.45)
    bayes_model.rng = np.random.default_rng(7)
    ts = tm.sample_2627(S=S)
    idx, mu, home, A, D = ts["idx"], ts["mu"], ts["home"], ts["att"], ts["dfn"]

    strength = pd.DataFrame({
        "team": list(idx.keys()),
        "att_strength": [float(A[:, idx[t]].mean()) for t in idx],
        "def_strength": [float(D[:, idx[t]].mean()) for t in idx]})
    strength["att_rank"] = strength["att_strength"].rank(ascending=False).astype(int)
    strength["def_rank"] = strength["def_strength"].rank(ascending=False).astype(int)

    _, long = schedule()
    rows = []
    for _, r in long[long.gameweek <= GW_HI].iterrows():
        if r.team not in idx or r.opp not in idx:
            continue
        he = bayes_model._home_effect(home, r.gameweek, bool(r.is_home))
        ho = bayes_model._home_effect(home, r.gameweek, not bool(r.is_home))
        rows.append({
            "team": r.team, "gw": int(r.gameweek), "opponent": r.opp,
            "is_home": int(bool(r.is_home)),
            "lam_for": float(np.exp(mu + he + A[:, idx[r.team]] - D[:, idx[r.opp]]).mean()),
            "lam_against": float(np.exp(mu + ho + A[:, idx[r.opp]] - D[:, idx[r.team]]).mean())})
    F = pd.DataFrame(rows)
    F = F.merge(strength.rename(columns={
        "team": "opponent", "att_strength": "opp_att_strength",
        "def_strength": "opp_def_strength", "att_rank": "opp_att_rank",
        "def_rank": "opp_def_rank"}), on="opponent", how="left")
    F = F.merge(strength[["team", "att_strength", "def_strength"]].rename(columns={
        "att_strength": "own_att_strength", "def_strength": "own_def_strength"}),
        on="team", how="left")
    F["attack_ease"] = F["lam_for"] / F["lam_for"].mean()
    F["clean_sheet_prob"] = np.exp(-F["lam_against"])
    return F


def main():
    board = pd.read_csv(_os.path.join(config.OUTPUTS, "gw_board_long.csv"))
    n0 = len(board)
    assert not board.duplicated(["player", "team", "gw"]).any(), \
        "board has duplicate (player, team, gw) — the join key is not unique"
    print(f"[export] board: {n0} rows, {board.groupby(['player','team']).ngroups} players")

    pl, sig, d26, t26 = player_priors()
    F = fixtures(t26)
    print(f"[export] fixtures: {len(F)} team-gameweeks over GW1-{GW_HI}")

    out = board.merge(F, on=["team", "gw"], how="left")
    assert len(out) == n0, f"fixture join fanned out: {n0} -> {len(out)}"

    pcols = (["web_name", "team", "player_code", "cold_start", "minutes_prior",
              "p_start_prior", "npxgi_rate90", "xa_rate90", "defcon_rate90", "pen_xg90",
              "exp_minutes_src"] + PRIOR_COLS)
    P = pl[[c for c in pcols if c in pl.columns]].rename(columns={"web_name": "player"})
    P = P.drop_duplicates(["player", "team"])
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

    out = out.sort_values(["gw", "mean"], ascending=[True, False])
    out.to_csv(OUT, index=False)
    print(f"[export] wrote {len(out)} rows x {len(out.columns)} columns -> {OUT}")
    miss = out["lam_for"].isna().sum()
    print(f"[export] rows without a fixture (blank gameweek): {miss}")
    print(f"[export] cold-start players: {int(out.drop_duplicates(['player','team'])['cold_start'].sum())}")


if __name__ == "__main__":
    main()
