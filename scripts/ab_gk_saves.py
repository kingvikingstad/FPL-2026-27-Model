import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
ab_gk_saves.py — the board gate for the goalkeeper save term (PROJECT_KNOWLEDGE §6.11)
=====================================================================================
`studies/gk_saves.py` fixed the FORM of the term (proportional, E[saves] = 2.059 * lam_against,
NB2 alpha 0.044). This checks the BOARD built with it, against gates G1-G4 registered in
that study's docstring BEFORE either board was built:

  G1  every non-GK row bit-identical across arms; every GK row's non-save components
      (app/att/cs/defcon/conc) bit-identical — the save draw has its own streams;
  G2  across clubs, the starting keeper's save_ev per match is Spearman >= 0.5 with the
      club's mean lam_against (negative ordering = failure);
  G3  no goalkeeper row has a negative `mean`;
  G4  league-mean save pts per starting-keeper match within +-20% of the study's pooled
      historical 0.626.
  G5  (diagnostic, NOT registered, gates nothing) closed-form E[floor(saves/3)] at each
      club-gw's lam_against against the board's per-match save_ev — the analytic direction.

Both arms are `scripts/gw_board.py` runs, same seed, LIVE_FPL=off, SOLIO=off, differing only
in FPL_GK_SAVES; each writes to its own FPL_OUTPUTS. Compared on `mean`, never `blended`.

Usage:  ab_gk_saves.py <off_outputs_dir> <on_outputs_dir>      |   --selftest
"""
import numpy as np, pandas as pd
from scipy import stats

HIST_SAVE_PTS = 0.626            # studies/gk_saves.py, pooled 24/25+25/26 per team-match
SAVE_R, SAVE_ALPHA = 2.05922, 0.04439
KEYS = ["player_code", "gw"]
NONSAVE = ["app_ev", "att_ev", "cs_ev", "defcon_ev", "conc_ev"]
START_APP = 1.5                  # app_ev above this = near-certain starter (P(60+) > 0.75)


def exp_save_pts(lam, smax=60):
    n = 1.0 / SAVE_ALPHA
    p = n / (n + SAVE_R * np.asarray(lam, float))
    s = np.arange(smax + 1)[:, None]
    return (np.floor(s / 3.0) * stats.nbinom.pmf(s, n, p[None, :])).sum(0)


def starters(on, fix):
    """Starting keeper per club-gw in single-fixture weeks, save_ev per match played."""
    one = fix.groupby(["team", "gw"]).size().rename("nfix").reset_index()
    one = one[one["nfix"] == 1]
    gk = on[(on["pos"] == "GK") & (on["app_ev"] > START_APP)]
    gk = gk.merge(one[["team", "gw"]], on=["team", "gw"], how="inner")
    gk = gk.merge(fix[["team", "gw", "lam_against"]], on=["team", "gw"], how="left")
    # per match PLAYED: appearance points ~ 2 x P(plays) for a starter
    gk = gk.assign(save_per_match=gk["save_ev"] / (gk["app_ev"] / 2.0))
    return gk


def gate(off, on, fix):
    m = off.merge(on, on=KEYS, suffixes=("_off", "_on"))
    out = {}
    gkm = m["pos_on"] == "GK"
    cols_all = [c for c in off.columns if c not in KEYS + ["player", "pos", "team", "src",
                                                          "solio", "blended", "save_ev"]]
    same_out = all(np.array_equal(m.loc[~gkm, f"{c}_off"].to_numpy(),
                                  m.loc[~gkm, f"{c}_on"].to_numpy(), equal_nan=True)
                   for c in cols_all)
    same_gk = all(np.array_equal(m.loc[gkm, f"{c}_off"].to_numpy(),
                                 m.loc[gkm, f"{c}_on"].to_numpy(), equal_nan=True)
                  for c in NONSAVE)
    out["G1"] = (same_out and same_gk,
                 f"outfield identical={same_out} (n={int((~gkm).sum())}), "
                 f"GK non-save components identical={same_gk} (n={int(gkm.sum())})")

    st = starters(on, fix)
    club = st.groupby("team")[["save_per_match", "lam_against"]].mean()
    rho = stats.spearmanr(club["save_per_match"], club["lam_against"]).correlation
    out["G2"] = (rho >= 0.5, f"club Spearman(save_ev/match, lam_against) = {rho:+.3f} "
                             f"over {len(club)} clubs")

    gk_on = on[on["pos"] == "GK"]
    mn = gk_on["mean"].min()
    out["G3"] = (mn >= 0, f"min GK mean {mn:+.4f} (off arm: "
                          f"{off.loc[off['pos'] == 'GK', 'mean'].min():+.4f})")

    lm = st["save_per_match"].mean()
    out["G4"] = (abs(lm / HIST_SAVE_PTS - 1) <= 0.20,
                 f"mean save pts per starting-keeper match {lm:.3f} vs historical "
                 f"{HIST_SAVE_PTS} ({lm / HIST_SAVE_PTS - 1:+.1%})")

    cf = exp_save_pts(st["lam_against"].to_numpy())
    diff = st["save_per_match"].to_numpy() - cf
    out["G5"] = (None, f"board minus closed form per match: mean {diff.mean():+.4f}, "
                       f"max |.| {np.abs(diff).max():.4f} (n={len(st)}; MC + posterior "
                       f"spread of lam, not a gate)")
    return out, m, st, club


def report(off, on, fix):
    res, m, st, club = gate(off, on, fix)
    for k, (ok, msg) in res.items():
        tag = "info" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {k} {tag:4s}  {msg}")
    gkm = m["pos_on"] == "GK"
    d = (m.loc[gkm, "mean_on"] - m.loc[gkm, "mean_off"])
    print(f"\n  GK mean shift per row: mean {d.mean():+.3f}, starters "
          f"{(m.loc[gkm & (m['app_ev_on'] > START_APP), 'mean_on'] - m.loc[gkm & (m['app_ev_on'] > START_APP), 'mean_off']).mean():+.3f}")
    print("\n  clubs by starting-keeper save pts per match (top/bottom 4):")
    c = club.sort_values("save_per_match")
    print(pd.concat([c.head(4), c.tail(4)]).round(3).to_string())
    return all(ok for ok, _ in res.values() if ok is not None)


def selftest():
    rng = np.random.default_rng(1)
    teams = [f"T{i}" for i in range(20)]
    lam = dict(zip(teams, np.linspace(0.8, 2.2, 20)))
    fix = pd.DataFrame([{"team": t, "gw": g, "lam_against": lam[t]} for t in teams
                        for g in (1, 2)])
    rows = []
    for i, t in enumerate(teams):
        for g in (1, 2):
            rows.append({"player_code": i, "player": f"k{i}", "pos": "GK", "team": t, "gw": g,
                         "mean": 3.0, "app_ev": 1.9, "att_ev": 0.0, "cs_ev": 1.0,
                         "defcon_ev": 0.0, "conc_ev": -0.4, "sd": 2.0, "save_ev": 0.0})
            rows.append({"player_code": 100 + i, "player": f"d{i}", "pos": "DEF", "team": t,
                         "gw": g, "mean": 3.5, "app_ev": 1.9, "att_ev": 0.4, "cs_ev": 1.0,
                         "defcon_ev": 0.5, "conc_ev": -0.4, "sd": 2.5, "save_ev": 0.0})
    off = pd.DataFrame(rows)
    on = off.copy()
    k = on["pos"] == "GK"
    sv = exp_save_pts(on.loc[k, "team"].map(lam).to_numpy()) * 0.95
    on.loc[k, "save_ev"] = sv
    on.loc[k, "mean"] = on.loc[k, "mean"] + sv
    res, *_ = gate(off, on, fix)
    assert all(ok for ok, _ in res.values() if ok is not None), res
    # a change that leaks into an outfield row must fail G1
    bad = on.copy(); bad.loc[~k, "mean"] += 1e-9
    assert not gate(off, bad, fix)[0]["G1"][0]
    # reversed ordering (strong defences save more) must fail G2
    rev = on.copy(); rev.loc[k, "save_ev"] = rev.loc[k, "save_ev"].to_numpy()[::-1]
    assert not gate(off, rev, fix)[0]["G2"][0]
    # at the league-average lambda the closed form sits near the historical per-match rate
    assert abs(exp_save_pts(np.array([1.40]))[0] - HIST_SAVE_PTS) < 0.1
    print("ab_gk_saves selftest OK")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        selftest()
        _sys.exit(0)
    a, b = _sys.argv[1], _sys.argv[2]
    off = pd.read_csv(_os.path.join(a, "gw_board_long.csv"))
    on = pd.read_csv(_os.path.join(b, "gw_board_long.csv"))
    fix = pd.read_csv(_os.path.join(b, "team_projections_gw1_38.csv"))
    print(f"A/B FPL_GK_SAVES off ({a}) vs on ({b}); {len(on)} player-gameweeks")
    _sys.exit(0 if report(off, on, fix) else 1)
