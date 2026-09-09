import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
postgw_review.py — the after-every-gameweek chore, in one runner.
==================================================================
    python scripts/postgw_review.py --gw 2      review one gameweek
    python scripts/postgw_review.py --auto      review the newest COMPLETE, unreviewed GW
    python scripts/postgw_review.py --selftest  offline wiring check

WHY --auto EXISTS
------------------
Gameweeks do not end on a schedule a cron can express: GW1 finished on a Monday, GW2 on
a Monday three days after it began, and a midweek round ends on a Thursday. Anything
pinned to a weekday either fires early on an unfinished week or misses one entirely.
So the trigger is a STATE test, not a time: run daily, and act only when some gameweek
is (a) complete and (b) has no review doc yet. Running it ten times in a day is a no-op
nine times.

WHAT "COMPLETE" MEANS, AND THE TRAP IN IT
------------------------------------------
`matches.csv` in a GW folder is NOT the gameweek's fixture list. In the 26/27 GW2 folder
it carried 20 rows: the 10 league fixtures plus 10 midweek cup ties whose opponents are
EFL clubs. Cup rows are identifiable because their team codes are absent from teams.csv,
so they map to NaN. Counting raw `finished` there would call GW2 complete while three
league games were still to play. This module reuses `inseason.played()`, which maps
codes through teams.csv and drops what does not resolve, so only league fixtures count.

BONUS IS NOT FINAL UNTIL `finished`
------------------------------------
FPL marks a match `finished_provisional` as soon as it ends but `finished` only once
bonus is confirmed, and every total_points can still move by up to 3 in between. The
review refuses to write a doc for a gameweek that is not fully `finished` unless forced,
because a review quoting provisional bonus gets quietly cited later as if it were final.
"""
import argparse
import datetime as _dt
import glob
import numpy as np
import pandas as pd
from scipy import stats

import inseason as ins

DOC_GLOB = "GW{gw}_REVIEW_*.md"


def league_fixtures(gw, base=None):
    """Only the league fixtures for `gw`, cup ties dropped. Returns (n_total, n_done).

    The module docstring above says cup rows are identifiable because their team codes
    are absent from teams.csv. That is true of a tie against an EFL club and FALSE of a
    cup tie between two Premier League clubs, which resolves on both sides and passed
    the old `both codes are PL codes` test. 26/27 GW2 contained exactly one —
    `26-27-efl-cup-nottingham-forest-vs-leeds-united` — so the round reported as 11/11
    when the league had 10 fixtures.

    Both were finished, so `n_done == n_fix` still held and the review ran; the failure
    mode it was one unfinished cup tie away from is worse. `--auto` gates on that
    equality, so an incomplete PL-vs-PL cup tie would have withheld a review of a
    COMPLETE league round indefinitely, and the wrong denominator is written into the
    doc as fact.

    Filtering on the `-prem-` token instead is what `inseason` and `repo_events` already
    do, and it does not care who the opponent is.
    """
    base = base or config.repo("2026-2027")
    d = _os.path.join(base, "By Gameweek", f"GW{gw}")
    f = _os.path.join(d, "matches.csv")
    if not _os.path.exists(f):
        return 0, 0
    m = pd.read_csv(f)
    lg = m[m["match_id"].astype(str).str.contains("-prem-", na=False)]
    return len(lg), int(lg["finished"].sum())


def actuals(gw, base=None):
    base = base or config.repo("2026-2027")
    d = _os.path.join(base, "By Gameweek", f"GW{gw}")
    roster = pd.read_csv(_os.path.join(base, "players.csv"))
    g = pd.read_csv(_os.path.join(d, "player_gameweek_stats.csv")).merge(
        roster[["player_id", "player_code"]], left_on="id", right_on="player_id", how="left")
    g = g.dropna(subset=["player_code"]).drop_duplicates("player_code")
    g["player_code"] = g["player_code"].astype(int)
    return g.set_index("player_code")[["minutes", "total_points", "bonus"]]


def prediction(gw, supplied=None):
    """The board to score for `gw`, and WHICH OF THREE KINDS of evidence it is.

    Three cases, deliberately not two. `score_gw._basis()` already grades provenance
    three ways and this runner graded it two, so a gameweek carrying a genuine
    pre-kickoff board but no lock — GW2 and GW3 are both that shape — produced a review
    doc asserting that no pre-deadline prediction existed. That is false, it is false in
    the direction that UNDERSTATES the model, and the doc is the artifact that gets
    cited months later when nobody re-derives it.

      lock       a board saved before the deadline. Proves its own information set by
                 existing; nothing has to be taken on trust.
      supplied   passed with --prediction. The CALLER asserts the inputs predate
                 kickoff and the file does not prove it. Admissible because
                 out-of-sample is a property of what the board KNEW, not of when it was
                 written to disk — the rule `score_gw.main()` already states.
      unlocked   today's board. A diagnostic, never a forecast.

    Returns the basis as a STRING rather than the old bool: the caller has to
    distinguish `supplied` from `lock` to write an honest header, and a bool cannot.
    """
    if supplied:
        return (pd.read_csv(supplied),
                f"supplied-prekickoff: {_os.path.basename(supplied)}", "supplied")
    hits = sorted(glob.glob(_os.path.join(config.PREDICTIONS, f"gw{gw}_board_locked_*.csv")))
    if hits:
        return pd.read_csv(hits[-1]), f"locked: {_os.path.basename(hits[-1])}", "lock"
    p = _os.path.join(config.OUTPUTS, "gw_board_long.csv")
    return pd.read_csv(p), "UNLOCKED current board (not a pre-deadline prediction)", "unlocked"


def review(gw, force=False, write=True, supplied=None):
    n_fix, n_done = league_fixtures(gw)
    if n_fix == 0:
        print(f"GW{gw}: no fixture data yet.")
        return None
    complete = n_done == n_fix
    print(f"GW{gw}: {n_done}/{n_fix} league fixtures finished (bonus confirmed).")
    if not complete and not force:
        print("  not complete — nothing written. Re-run when it is, or pass --force.")
        return None

    act = actuals(gw)
    board, prov, basis = prediction(gw, supplied=supplied)
    b = board[board["gw"] == gw].dropna(subset=["player_code"]).drop_duplicates("player_code")
    b["player_code"] = b["player_code"].astype(int)
    m = b.merge(act, on="player_code", how="inner")
    if not len(m):
        print("  no players joined on player_code — check the board carries it.")
        return None

    ap = m[m["minutes"] > 0]
    hd = {
        "n": len(m), "n_app": len(ap),
        "r": stats.pearsonr(m["mean"], m.total_points)[0],
        "rho": stats.spearmanr(m["mean"], m.total_points)[0],
        "mae": (m["mean"] - m.total_points).abs().mean(),
        "bias": (m["mean"] - m.total_points).mean(),
        "r_app": stats.pearsonr(ap["mean"], ap.total_points)[0] if len(ap) > 2 else np.nan,
        "mae_app": (ap["mean"] - ap.total_points).abs().mean() if len(ap) else np.nan,
        "bias_app": (ap["mean"] - ap.total_points).mean() if len(ap) else np.nan,
        "flat_mae": (2.0 - m.total_points).abs().mean(),
    }
    m = m.copy(); m["err"] = m.total_points - m["mean"]
    misses = m[(m["mean"] >= 3.0) & (m["minutes"] == 0)].sort_values("mean", ascending=False)
    over = m.nlargest(10, "err")
    under = m[m["mean"] >= 3].nsmallest(10, "err")
    club = m.groupby("team").agg(proj=("mean", "sum"), actual=("total_points", "sum"))
    club["diff"] = club.actual - club.proj

    print(f"  prediction scored: {prov}")
    print(f"  r={hd['r']:.3f} rho={hd['rho']:.3f} MAE={hd['mae']:.2f} bias={hd['bias']:+.2f}"
          f"   (flat-2.0 baseline MAE {hd['flat_mae']:.2f})")
    print(f"  appeared only (n={hd['n_app']}): r={hd['r_app']:.3f} MAE={hd['mae_app']:.2f}"
          f" bias={hd['bias_app']:+.2f}")
    print(f"  availability misses (proj>=3.0, 0 min): {len(misses)}"
          f" carrying {misses['mean'].sum():.1f} projected points")

    if not write:
        return hd
    stamp = _dt.date.today().isoformat()
    path = _os.path.join(config.ROOT, "docs", f"GW{gw}_REVIEW_{stamp}.md")
    with open(path, "w", encoding="utf-8") as fh:
        w = fh.write
        w(f"# GW{gw} review — {stamp}\n\n")
        w(f"Auto-generated by `scripts/postgw_review.py`. "
          f"{n_done}/{n_fix} league fixtures, bonus confirmed.\n\n")
        w(f"Prediction scored: **{prov}**\n\n")
        if basis == "unlocked":
            w("> **Not a pre-deadline prediction.** No locked board existed for this "
              "gameweek, so this scores the current board and the numbers are a "
              "diagnostic, not a forecasting result. Save a board before the next "
              "deadline to `predictions/`.\n\n")
        elif basis == "supplied":
            w("> **Pre-kickoff board, not a lock.** The caller asserts this file's "
              "inputs predate the gameweek's kickoffs, which makes it a genuine "
              "forecast — out-of-sample is a property of what the board KNEW, not of "
              "when it was written. But nothing IN the file proves its pull date, "
              "which a lock proves by construction. Weight it below a locked row; "
              "see `predictions/README.md`.\n\n")
        w("## 1. Headline\n\n| | r | rho | MAE | bias |\n|---|---|---|---|---|\n")
        w(f"| model, all players (n={hd['n']}) | {hd['r']:.3f} | {hd['rho']:.3f} | "
          f"{hd['mae']:.2f} | {hd['bias']:+.2f} |\n")
        w(f"| model, appeared only (n={hd['n_app']}) | {hd['r_app']:.3f} | — | "
          f"{hd['mae_app']:.2f} | {hd['bias_app']:+.2f} |\n")
        w(f"| baseline: everyone gets 2.0 | — | — | {hd['flat_mae']:.2f} | — |\n\n")
        w(f"Correlation falls {hd['r']:.3f} -> {hd['r_app']:.3f} once restricted to players "
          f"who appeared. That gap is how much of the week's skill was knowing who plays.\n\n")
        w(f"## 2. Availability misses\n\n{len(misses)} players projected >=3.0 played zero "
          f"minutes, carrying {misses['mean'].sum():.1f} projected points that could never "
          f"be scored.\n\n| player | club | proj |\n|---|---|---|\n")
        for r in misses.head(12).itertuples():
            w(f"| {r.player} | {r.team} | {getattr(r, 'mean'):.2f} |\n")
        w("\n## 3. Biggest surprises\n\n**Overperformed**\n\n| player | club | proj | actual |\n|---|---|---|---|\n")
        for r in over.itertuples():
            w(f"| {r.player} | {r.team} | {getattr(r, 'mean'):.2f} | {int(r.total_points)} |\n")
        w("\n**Underperformed (projected >=3)**\n\n| player | club | proj | actual | min |\n|---|---|---|---|---|\n")
        for r in under.itertuples():
            w(f"| {r.player} | {r.team} | {getattr(r, 'mean'):.2f} | {int(r.total_points)} | {int(r.minutes)} |\n")
        w("\n## 4. Per club\n\n| club | proj | actual | diff |\n|---|---|---|---|\n")
        for t, r in club.sort_values("diff", ascending=False).iterrows():
            w(f"| {t} | {r.proj:.1f} | {int(r.actual)} | {r['diff']:+.1f} |\n")
        w("\n## 5. Still to do by hand\n\n"
          "- [ ] Enter your own and the tracked squads' scores: "
          f"`python scripts/track.py --gw {gw} --result MINE HYBRID MODEL`\n"
          f"- [ ] Freeze this week's plan squads into `predictions/gw{gw}_scored/` — "
          "`plan_constrained.py` overwrites them and they are not reproducible.\n"
          "- [ ] Save a locked board before the NEXT deadline, or the next review is a "
          "diagnostic too.\n"
          "- [ ] Read section 4 against section 2 before concluding anything about the "
          "team layer: most per-club deficit is the availability miss landing club by club.\n")
    print(f"  wrote {path}")
    return hd


def auto(max_gw=38):
    """Newest complete, unreviewed gameweek. Silent no-op when there isn't one."""
    for gw in range(max_gw, 0, -1):
        n_fix, n_done = league_fixtures(gw)
        if n_fix == 0 or n_done != n_fix:
            continue
        if glob.glob(_os.path.join(config.ROOT, "docs", DOC_GLOB.format(gw=gw))):
            print(f"[auto] newest complete gameweek is GW{gw}; already reviewed. Nothing to do.")
            return None
        print(f"[auto] GW{gw} is complete and unreviewed.")
        return review(gw)
    print("[auto] no complete gameweek found.")
    return None


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    gwd = _os.path.join(d, "By Gameweek", "GW9"); _os.makedirs(gwd)
    pd.DataFrame({"code": [1, 2], "id": [1, 2], "name": ["A", "B"]}).to_csv(
        _os.path.join(gwd, "teams.csv"), index=False)
    # Two league fixtures, a cup tie against a club absent from teams.csv, AND a cup tie
    # between two PREMIER LEAGUE clubs. The last one is the case that broke this: it
    # resolves on both sides, so the old "both codes are PL codes" test counted it as
    # league and reported 26/27 GW2 as 11/11 when the round was 10 fixtures. It is
    # UNFINISHED here on purpose — that is the shape that would have made `--auto`
    # withhold a review of a complete league round indefinitely.
    pd.DataFrame({
        "match_id": ["26-27-prem-a-vs-b", "26-27-prem-b-vs-a",
                     "26-27-efl-cup-a-vs-minnow", "26-27-efl-cup-a-vs-b"],
        "home_team": [1.0, 2.0, 1.0, 1.0], "away_team": [2.0, 1.0, 99.0, 2.0],
        "finished": [True, True, False, False]}).to_csv(
        _os.path.join(gwd, "matches.csv"), index=False)
    n, done = league_fixtures(9, base=d)
    assert (n, done) == (2, 2), (
        f"cup ties must be excluded — including an all-PL one — got {(n, done)}")
    # a genuinely incomplete league round is not complete
    pd.DataFrame({"match_id": ["26-27-prem-a-vs-b", "26-27-prem-b-vs-a"],
                  "home_team": [1.0, 2.0], "away_team": [2.0, 1.0],
                  "finished": [True, False]}).to_csv(
        _os.path.join(gwd, "matches.csv"), index=False)
    n, done = league_fixtures(9, base=d)
    assert (n, done) == (2, 1), (n, done)
    # Provenance is graded three ways, not two. A gameweek with a genuine pre-kickoff
    # board but no lock (GW2 and GW3) used to fall through to the `unlocked` branch and
    # the doc asserted no pre-deadline prediction existed — false, and false in the
    # direction that understates the model. Only the `supplied` branch is exercised
    # here: it is the one that takes a path, so it is the one that stays offline.
    f = _os.path.join(d, "supplied.csv")
    pd.DataFrame({"gw": [3], "player_code": [1], "mean": [4.0],
                  "player": ["A"], "team": ["X"]}).to_csv(f, index=False)
    board, prov, basis = prediction(3, supplied=f)
    assert basis == "supplied", basis
    assert "supplied-prekickoff" in prov and "supplied.csv" in prov, prov
    assert len(board) == 1, len(board)

    print("SELFTEST OK: cup ties are excluded from the completeness test — both against "
          "an unmapped club and between two PL clubs, which the old code counted as "
          "league — a part-played league round is not reported complete, and a "
          "supplied pre-kickoff board grades as `supplied`, not `unlocked`.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prediction", help="score this board instead of the "
                                         "locked one; asserts its inputs "
                                         "predate kickoff")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.auto:
        auto()
    elif a.gw:
        review(a.gw, force=a.force, supplied=a.prediction)
    else:
        print(__doc__)
