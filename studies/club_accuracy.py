import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "scripts"))
import config
"""
club_accuracy.py — is projection error a property of the CLUB, or is it noise?
=============================================================================
"Are players on certain teams predicted more correctly?"  err = projected - actual
throughout, matching `score_gw`: POSITIVE IS OVER-PROJECTION.

Ranking 20 clubs by MAE always produces a spread — under pure noise the best and worst
club differ by roughly 2*sd/sqrt(n) with n~28 players, which is most of the spread
actually seen. A ranking is therefore not evidence. Two tests are, and both are
pre-registered here.

Raw MAE is also confounded by SCALE: a club whose players score few points has small
absolute errors by construction, so "predicted well" and "scores little" are the same
number. Every group comparison below is therefore reported both raw and divided by the
club's own SD of actual points.

PRE-REGISTERED, decision rules fixed before the numbers were seen
-----------------------------------------------------------------
  TEST A (persistence).  Correlate each club's mean signed error in GW1 against the same
      club's mean signed error in GW2, across the clubs scored in both. A real club
      effect — a squad the model systematically misreads — persists across weeks. Noise
      does not. Same logic as the split-half reliability used for the style indicators,
      and it is the test that matters.
      RULE: club effect supported only if r > 0 with one-sided p < 0.05.

  TEST B (variance).  Permutation test: shuffle club labels across players within a
      gameweek, and ask how often the between-club variance of mean error exceeds the
      observed value.
      RULE: supported only if p < 0.05.

  DECLARED REAL only if BOTH clear. Otherwise this is a null and NO club-level
  correction is built — per CLAUDE.md, a tested null stays dead.

Reported either way, because they are the interesting descriptive cuts:
  - the club bias table, with the noise band it has to beat,
  - promoted vs established, and regime vs continuity clubs, the two groups the model
    itself flags as having weak priors.

CAVEAT: GW2 bonus is not confirmed and one fixture is outstanding, so GW2 points can
still move. With 18 clubs in both weeks the MDE on TEST A is r ~ 0.40 at 80% power —
this cannot detect a small club effect, only a large one. Absence of evidence here is
weak evidence of absence, and is recorded as such.
"""
import numpy as np, pandas as pd
from scipy import stats
import score_gw as sg
import press_index as px
from schedule_2627 import PROMOTED

BOARDS = {1: "gw1_board_locked_2026-08-21.csv",
          2: "gw2_board_reconstructed_from-2026-08-26-data.csv"}
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "club_accuracy.csv")


def frame(gw, fname):
    """Player-level actual vs projected for one gameweek, restricted to clubs whose
    fixture is complete."""
    L = pd.read_csv(_os.path.join(config.PREDICTIONS, fname))
    L = L[L["gw"] == gw] if "gw" in L.columns else L
    col = "blended" if "blended" in L.columns else "mean"
    A = sg.actuals(gw)
    clubs, _, _ = sg.played_clubs(gw)
    # The GW1 lock predates `player_code` on the board output, so it can only be joined
    # on name+club. `score_gw` makes the same fallback and says so; mirrored here rather
    # than silently dropping GW1, which would leave TEST A with nothing to correlate.
    if "player_code" in L.columns and L["player_code"].notna().any():
        M = L.merge(A, on="player_code", how="inner", suffixes=("", "_act"))
    else:
        M = L.merge(A, left_on=["player", "team"], right_on=["web_name", "team"],
                    how="inner")
        print(f"  [gw{gw}] joined on name+club — that lock predates player_code")
    M = M[M["team"].isin(clubs)].copy()
    M["gw"] = gw
    M["proj"] = pd.to_numeric(M[col], errors="coerce")
    M["act"] = pd.to_numeric(M["total_points"], errors="coerce")
    # Sign convention follows `score_gw.line`, which is mean(model - actual): POSITIVE
    # IS OVER-PROJECTION. Deliberately matched so the repo carries ONE convention -
    # the natural residual (actual - model) points the other way, and two conventions
    # in one repo is how a winner's-curse story gets told backwards.
    M["err"] = M["proj"] - M["act"]                     # + = model OVER-projected
    M["mins"] = pd.to_numeric(M["minutes"], errors="coerce").fillna(0)
    return M.dropna(subset=["proj", "act"])[
        ["gw", "player_code", "web_name", "team", "proj", "act", "err", "mins"]]


def permutation_p(d, iters=10000, seed=7):
    """P(between-club variance of mean error >= observed) with club labels shuffled
    within gameweek."""
    rng = np.random.default_rng(seed)

    def between(frame_):
        g = frame_.groupby(["gw", "team"])["err"].mean()
        return float(g.groupby(level=0).var(ddof=1).mean())

    obs = between(d)
    hits = 0
    x = d.copy()
    for _ in range(iters):
        x["team"] = x.groupby("gw")["team"].transform(
            lambda s: rng.permutation(s.values))
        hits += between(x) >= obs
    return obs, (hits + 1) / (iters + 1)


def main():
    d = pd.concat([frame(gw, f) for gw, f in BOARDS.items()], ignore_index=True)
    print("=" * 78)
    print("CLUB-LEVEL PROJECTION ACCURACY   (err = projected - actual; + = OVER-projected)")
    print("=" * 78)
    print(f"  {len(d)} player-gameweeks, {d.team.nunique()} clubs, GW{sorted(d.gw.unique())}")
    print("  GW2 bonus is NOT confirmed - these numbers can still move.")

    g = d.groupby("team").agg(n=("err", "size"),
                              mae=("err", lambda s: s.abs().mean()),
                              bias=("err", "mean")).reset_index()
    app = d[d.mins > 0].groupby("team").agg(
        n_app=("err", "size"), mae_app=("err", lambda s: s.abs().mean()),
        bias_app=("err", "mean")).reset_index()
    g = g.merge(app, on="team", how="left")
    g["promoted"] = g.team.isin(PROMOTED)
    g["regime"] = g.team.isin(px.REGIME_PRESS_CLUBS)
    g = g.sort_values("mae")
    print("\nPER CLUB, pooled GW1+GW2, sorted by MAE")
    print(g[["team", "n", "mae", "bias", "n_app", "mae_app", "bias_app"]]
          .round(2).to_string(index=False))

    sd = d.err.std(ddof=1)
    nbar = g.n.mean()
    se = sd / np.sqrt(nbar)
    print(f"\n  residual sd {sd:.2f}; with ~{nbar:.0f} players/club the SE of a club mean")
    print(f"  is {se:.2f}, so under PURE NOISE club biases should span about "
          f"+/-{1.96 * se:.2f}.")
    print(f"  Observed span {g.bias.min():+.2f} to {g.bias.max():+.2f}.")

    print("\n" + "-" * 78)
    print("TEST A - does a club's error PERSIST from GW1 to GW2?")
    print("-" * 78)
    piv = d.groupby(["team", "gw"])["err"].mean().unstack("gw").dropna()
    a_pass = False
    if piv.shape[1] == 2 and len(piv) >= 5:
        r, p2 = stats.pearsonr(piv[1], piv[2])
        rho, _ = stats.spearmanr(piv[1], piv[2])
        p1 = p2 / 2 if r > 0 else 1 - p2 / 2
        a_pass = (r > 0) and (p1 < 0.05)
        print(f"  clubs in both weeks: {len(piv)}")
        print(f"  r = {r:+.3f}   rho = {rho:+.3f}   one-sided p = {p1:.3f}")
        print(f"  -> {'PASS' if a_pass else 'FAIL'} the pre-registered rule")
        p2f = piv.copy()
        p2f.columns = ["gw1_bias", "gw2_bias"]
        print("\n" + p2f.round(2).sort_values("gw1_bias").to_string())
    else:
        print("  not enough overlap")

    print("\n" + "-" * 78)
    print("TEST B - is between-club variance larger than chance?")
    print("-" * 78)
    obs, pb = permutation_p(d)
    b_pass = pb < 0.05
    print(f"  observed between-club variance of mean error: {obs:.4f}")
    print(f"  permutation p = {pb:.4f}   -> {'PASS' if b_pass else 'FAIL'}")

    print("\n" + "-" * 78)
    print("PRE-EXISTING WEAK-PRIOR GROUPS")
    print("-" * 78)
    # Scale-adjusted: |err| / SD(actual points) within the same group. A club whose
    # players never score has small absolute errors for free; dividing by the outcome
    # spread asks whether the model is doing better RELATIVE TO WHAT THERE WAS TO PREDICT.
    for lab, mask in (("promoted", d.team.isin(PROMOTED)),
                      ("regime", d.team.isin(px.REGIME_PRESS_CLUBS))):
        a, b = d[mask], d[~mask]
        _, pv = stats.ttest_ind(a.err.abs(), b.err.abs(), equal_var=False)
        sa, sb = a.act.std(ddof=1), b.act.std(ddof=1)
        na, nb = a.err.abs().mean() / sa, b.err.abs().mean() / sb
        _, pvn = stats.ttest_ind(a.err.abs() / sa, b.err.abs() / sb, equal_var=False)
        print(f"  {lab:9s} raw MAE  {a.err.abs().mean():.2f} (n={len(a)}) vs "
              f"{b.err.abs().mean():.2f} (n={len(b)})   Welch p={pv:.3f}")
        print(f"  {'':9s} SD(act)  {sa:.2f} vs {sb:.2f}  -> scale-adjusted "
              f"{na:.3f} vs {nb:.3f}   Welch p={pvn:.3f}")

    print("\n" + "=" * 78)
    if a_pass and b_pass:
        print("VERDICT: club effect SUPPORTED by both tests.")
    else:
        print("VERDICT: NULL - club identity does not explain projection error beyond")
        print("         noise on the evidence available. No club-level correction is")
        print("         built. Per CLAUDE.md a tested null stays dead; revisit only with")
        print("         a pre-registered re-test once more gameweeks have accrued.")
    print("=" * 78)
    g.to_csv(OUT, index=False)
    print(f"[wrote] {OUT}")
    return g


if __name__ == "__main__":
    main()
