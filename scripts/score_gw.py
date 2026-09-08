import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
score_gw.py — score a locked prediction against what actually happened.
========================================================================
The validation PROJECT_KNOWLEDGE §6.6 has been waiting for. Everything else in this repo
compares projections to other projections (Solio, FFS) or to measured base rates; this
compares them to points that were actually scored.

It reads `predictions/gw{N}_board_locked_*.csv` — the board as it stood BEFORE the
deadline, which is the only thing that can honestly be scored — and the realised
`player_gameweek_stats.csv` for that gameweek.

WHAT IT REPORTS, AND WHY EACH ONE
-----------------------------------
  HEADLINE      r / rho / MAE / bias against two naive baselines. A model that cannot
                beat "everyone gets 2.0" has not earned its complexity.
  CONDITIONAL   the same, restricted to players who actually appeared. This is the
                diagnostic that matters: the gap between the two says how much of the
                skill is knowing WHO PLAYS versus how well they play. In GW1 the
                correlation roughly halved, which is the project's own central claim
                about minutes being the dominant lever, confirmed on real data.
  DECISION      the same metrics restricted to the players a manager actually chooses
                between. The headline is flattering: most of its correlation is the
                model separating starters from non-starters, a distinction nobody is
                deciding. Measured on GW1, r fell from 0.530 across all 577 listed
                players to roughly zero inside the top 50 by projection — while the
                OUTCOME variance rose (SD 2.72 -> 3.68), so it is not that there was
                less to predict there. Range restriction explains part of that
                mechanically; it does not explain the bias column, which is the row to
                watch: +0.48 across the field against +2.29 for the top 15.
  MISSES        players projected 3.0+ who played zero minutes, and the reverse. Given
                the above, this is the error class worth attacking.
  PER CLUB      where the team layer was wrong, which is a different failure from
                getting a player wrong.

PROVISIONAL DATA
----------------
FPL marks a gameweek `finished` only once bonus is confirmed. Before that every
`total_points` can still move by up to 3. The runner checks and says so rather than
quietly scoring numbers that are about to change.

Run:  python scripts/score_gw.py            (latest locked prediction)
      python scripts/score_gw.py --gw 1
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import json
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats as st
import core_insights as ci

LEDGER = _os.path.join(config.PREDICTIONS, "scoring_ledger.csv")


def _basis(path, supplied):
    """What kind of evidence this row is scoring — recorded in the ledger, not inferred
    later from a filename.

    Until 2026-09-08 the ledger had no such column, so GW1 (a true pre-deadline lock)
    and GW2 (a board reconstructed on 2026-08-31 from a 2026-08-26 data pull) sat in one
    table as equal rows. They are not equal. The distinction is the whole subject of
    `predictions/README.md`, and `forecast-scorer` is required to state it as BASIS on
    every report, which it cannot do from a table that does not carry it.

    `supplied` is the weaker claim deliberately: --prediction asserts only that the
    inputs predate kickoff. Out-of-sample is a property of what the board KNEW, so a
    reconstruction from a pre-deadline pull is a genuine forecast — but nothing in the
    file proves the pull date, and a lock proves it by construction. Ranking them in
    the data lets a later reader weight them differently without re-deriving why.
    """
    if supplied:
        return "supplied-prekickoff"
    # "lock" deliberately, not "early-lock": predictions/README.md gives "early" a
    # specific meaning — a SECONDARY board taken days out as insurance while a
    # `_deadline` primary is still to come. GW1's lock is neither secondary nor early
    # in that sense; it is the only one, taken about six hours out. Reporting it under
    # the insurance label would understate the project's single strongest row.
    return "deadline-lock" if "_deadline" in _os.path.basename(path) else "lock"


def _locked(gw):
    """The locked prediction to score for `gw`.

    A gameweek may carry MORE THAN ONE lock — an early one taken days out as insurance
    and a `_deadline` one taken on the day, which is the model at the information it
    would actually have had. Which of those gets scored must be fixed in advance and by
    the code, not left to a filename sorting the right way by luck: picking after the
    results are in is choosing the flattering answer, and the whole point of this
    directory is that it cannot be done.

    So a `*_deadline*.csv` lock ALWAYS wins when one exists, regardless of dates. Any
    other lock is a secondary observation — useful for asking what a week of team news
    was worth, never for the headline score.
    """
    hits = sorted(glob.glob(_os.path.join(config.PREDICTIONS, f"gw{gw}_board_locked_*.csv")))
    if not hits:
        raise SystemExit(
            f"no locked prediction for GW{gw} in {config.PREDICTIONS}. A board can only be "
            f"scored if it was saved BEFORE the deadline — see predictions/README.md.")
    primary = [h for h in hits if "_deadline" in _os.path.basename(h)]
    if primary:
        if len(hits) > len(primary):
            print(f"[score] {len(hits)} locks for GW{gw}; scoring the pre-declared "
                  f"primary: {_os.path.basename(primary[-1])}")
        return primary[-1]
    if len(hits) > 1:
        print(f"[score] {len(hits)} locks for GW{gw} and none marked `_deadline`; "
              f"scoring the latest: {_os.path.basename(hits[-1])}")
    return hits[-1]


def actuals(gw, base=None):
    """Realised points for one gameweek, keyed on player_code."""
    base = base or config.repo("2026-2027")
    p = _os.path.join(base, "By Gameweek", f"GW{gw}", "player_gameweek_stats.csv")
    a = pd.read_csv(p)
    roster = pd.read_csv(_os.path.join(base, "players.csv"))
    teams = pd.read_csv(_os.path.join(base, "teams.csv"))
    roster = roster.merge(teams[["code", "name"]], left_on="team_code",
                          right_on="code", how="left")
    roster["team"] = roster["name"].map(ci.norm_team)
    A = a.merge(roster[["player_id", "player_code", "web_name", "team"]],
                left_on="id", right_on="player_id", how="left", suffixes=("", "_r"))
    keep = ["player_code", "web_name", "team", "minutes", "total_points"]
    return A[[c for c in keep if c in A.columns]].dropna(subset=["team"])


def played_clubs(gw, base=None):
    """Clubs whose LEAGUE fixture is complete, how many are done, and how many there are.

    A gameweek folder's `matches.csv` is NOT the gameweek's league fixture list. The 26/27
    GW2 folder carries 20 rows: 10 league fixtures plus 10 midweek cup ties. Counting them
    all had two consequences, both live before 2026-09-06:

      * the completeness denominator was the cup-inflated one — GW2 read as 20/20 done
        when the league round was 11 fixtures — so a week could be called final while
        league games were still to play, which is exactly what `final` must never do;
      * a cup opponent is an EFL club whose code is absent from teams.csv, so it mapped to
        NaN, and `norm_team` stringifies its input — producing a phantom club literally
        named "nan" in the returned set and a `clubs` count of 21 in the scoring ledger.

    `postgw_review` documents this trap and avoids it by routing through
    `inseason.played()`. This filters on the same `-prem-` token that `inseason` and
    `repo_events` use, and additionally drops any club that failed to resolve, so an
    unmapped code in a genuine league row cannot reintroduce the phantom.
    """
    base = base or config.repo("2026-2027")
    m = pd.read_csv(_os.path.join(base, "By Gameweek", f"GW{gw}", "matches.csv"))
    m = m[m["match_id"].astype(str).str.contains("-prem-", na=False)]
    t = pd.read_csv(_os.path.join(base, "teams.csv"))
    nm = dict(zip(t["code"].astype(float), t["name"]))
    m["H"] = m["home_team"].astype(float).map(nm).map(ci.norm_team)
    m["A"] = m["away_team"].astype(float).map(nm).map(ci.norm_team)
    done = m[m["finished"] == True]                                    # noqa: E712
    clubs = {c for c in (set(done["H"]) | set(done["A"]))
             if isinstance(c, str) and c and c.lower() != "nan"}
    return clubs, len(done), len(m)


def _final(gw):
    """Is the gameweek bonus-confirmed? Asked of the live API, not the snapshot."""
    try:
        req = urllib.request.Request(
            "https://fantasy.premierleague.com/api/bootstrap-static/",
            headers={"User-Agent": "Mozilla/5.0"})
        d = json.loads(urllib.request.urlopen(req, timeout=25).read())
        e = [x for x in d["events"] if x["id"] == int(gw)][0]
        return bool(e["finished"]), bool(e["data_checked"]), e.get("average_entry_score")
    except Exception:
        return None, None, None


def line(tag, x, y, pad=22):
    r = st.pearsonr(x, y)[0] if np.std(x) > 0 else np.nan
    rho = st.spearmanr(x, y)[0] if np.std(x) > 0 else np.nan
    print(f"  {tag:{pad}s} r={r:6.3f}  rho={rho:6.3f}  MAE={np.abs(x - y).mean():5.2f}  "
          f"bias={np.mean(x - y):+5.2f}")
    return {"r": r, "rho": rho, "mae": float(np.abs(x - y).mean()),
            "bias": float(np.mean(x - y))}


def main(gw, prediction=None):
    # `_locked` enforces that only a board saved BEFORE the deadline can be scored. That
    # guard exists so an unlocked board cannot be scored by accident, which would silently
    # report in-sample fit as if it were a forecast. `--prediction` is the deliberate
    # exception, and it announces itself: the caller asserts the file's information
    # predates the gameweek. Out-of-sample is a property of what the board KNEW, not of
    # when the file was written — a board built from a data pull that predates kickoff is
    # a genuine forecast even if it was written to disk afterwards.
    if prediction:
        path = prediction
        print("[provenance] scoring an UNLOCKED file supplied with --prediction:")
        print(f"             {path}")
        print(f"             Honest only if its inputs predate the GW{gw} kickoffs."
              f" Not a lock.")
    else:
        path = _locked(gw)
    L = pd.read_csv(path)
    L = L[L["gw"] == gw] if "gw" in L.columns else L
    A = actuals(gw)
    clubs, n_done, n_fix = played_clubs(gw)
    fin, checked, avg = _final(gw)

    print("=" * 78)
    print(f"GW{gw} SCORING   prediction: {_os.path.basename(path)}")
    print("=" * 78)
    print(f"  fixtures complete   {n_done}/{n_fix}")
    if fin is not None:
        print(f"  bonus confirmed     {'yes' if checked else 'NO — points can still move ±3'}"
              + (f"   (FPL average {avg})" if avg else ""))
    if n_done < n_fix:
        print(f"  scoring only the {len(clubs)} clubs whose fixture is complete")

    # Join on player_code where both sides carry it. A board that lacks the column can
    # still be scored on the stable key IF a crosswalk sidecar was built for it — see
    # scripts/crosswalk_gw1_lock.py. That path exists because the name+club fallback
    # below does not merely lose rows, it loses DIFFERENT rows over time: FPL rewrites
    # web_name mid-season, so the same board scored a fortnight apart gave n=577 then
    # n=561, and the ledger recorded the decay as a result. Name joins are forbidden by
    # CLAUDE.md for exactly this reason; the fallback is kept only so an un-crosswalked
    # board degrades loudly instead of refusing to score at all.
    if "player_code" not in L.columns or not L["player_code"].notna().any():
        side = _os.path.join(_os.path.dirname(path), "crosswalks",
                             _os.path.splitext(_os.path.basename(path))[0]
                             + ".player_code.csv")
        if _os.path.exists(side):
            xw = pd.read_csv(side)
            xw = xw[xw["player_code"].notna()][["player", "team", "player_code"]]
            L = L.merge(xw, on=["player", "team"], how="left")
            print(f"  crosswalk           {len(xw)} of {len(L)} rows keyed from "
                  f"{_os.path.basename(side)}")

    if "player_code" in L.columns and L["player_code"].notna().any():
        M = L.merge(A, on="player_code", how="inner", suffixes=("", "_act"))
        key = "player_code"
    else:
        M = L.merge(A, left_on=["player", "team"], right_on=["web_name", "team"],
                    how="inner")
        key = "(player, team)"
        print("  NOTE joined on name+club — UNSTABLE. This board predates player_code "
              "and has no crosswalk sidecar; the matched set will drift as FPL edits "
              "web_name. Build one with scripts/crosswalk_gw1_lock.py")
    M = M[M["team"].isin(clubs)].copy()
    print(f"  matched             {len(M)} players on {key}\n")
    if len(M) < 30:
        raise SystemExit("too few matched players to score")

    y = M["total_points"].values
    print("MODEL vs ACTUAL")
    res = {"gw": gw, "n": len(M), "clubs": len(clubs), "final": bool(checked),
           "basis": _basis(path, bool(prediction)),
           "source": _os.path.basename(path)}
    res.update({f"model_{k}": v for k, v in line("model mean", M["mean"].values, y).items()})
    if "blended" in M.columns:
        line("model blended", M["blended"].values, y)
    line("baseline: price x0.55", M["cost"].values * 0.55, y)
    line("baseline: flat 2.0", np.full(len(M), 2.0), y)

    ap = M[M["minutes"] > 0]
    print(f"\nCONDITIONAL ON APPEARING (n={len(ap)})")
    r2 = line("model mean", ap["mean"].values, ap["total_points"].values)
    res.update({f"appeared_{k}": v for k, v in r2.items()})
    print(f"  -> correlation falls {res['model_r']:.3f} -> {r2['r']:.3f}; the difference is "
          f"how much of the skill is knowing who plays")

    # DECISION-RELEVANT ACCURACY. The headline above is computed over every listed
    # player, and it is flattering: most of that correlation is the model separating
    # players who start from players who do not. Nobody chooses between a nailed premium
    # and a fourth-choice keeper. The set that matters is the one a manager actually
    # picks from, so it is scored explicitly rather than left to be inferred.
    #
    # Expect r to FALL under restriction — truncating the predictor's range does that
    # mechanically, whatever the model's quality. What is NOT mechanical is the bias
    # column: if the model over-projects its own top picks by more than it over-projects
    # the field, that is a winner's curse at the top of the ranking, and it is
    # actionable in a way the headline number never surfaces.
    print("\nDECISION-RELEVANT SUBSETS (r falls under range restriction by construction;"
          " watch `bias`)")
    print(f"  {'set':24s}{'n':>5s}{'r':>8s}{'rho':>8s}{'MAE':>7s}{'bias':>8s}"
          f"{'SD(act)':>9s}")
    def _sub(lab, d):
        if len(d) < 8:
            return None
        x, y = d["mean"].values, d["total_points"].values
        rr = st.pearsonr(x, y)[0] if np.std(x) > 0 else np.nan
        rr2 = st.spearmanr(x, y)[0] if np.std(x) > 0 else np.nan
        print(f"  {lab:24s}{len(d):>5d}{rr:>8.3f}{rr2:>8.3f}"
              f"{np.abs(x - y).mean():>7.2f}{np.mean(x - y):>+8.2f}{np.std(y):>9.2f}")
        return {"set": lab, "n": len(d), "r": rr, "rho": rr2,
                "mae": float(np.abs(x - y).mean()), "bias": float(np.mean(x - y))}
    subs = [_sub("all listed", M)]
    for nn in (100, 50, 30, 15):
        subs.append(_sub(f"top {nn} by projection", M.nlargest(nn, "mean")))
    own = pd.to_numeric(M.get("own"), errors="coerce")
    if own is not None and own.notna().any():
        subs.append(_sub("ownership >= 5%", M[own >= 5]))
    subs = [x for x in subs if x]
    top15 = next((x for x in subs if x["set"].startswith("top 15")), None)
    if top15 and abs(top15["bias"]) > 2 * abs(res.get("model_bias", 0.0)) + 0.5:
        # `line` computes bias as mean(model - actual), so POSITIVE is over-projection.
        # This used to hardcode the word "over-projected" and then print a signed number,
        # so a negative bias rendered as "over-projected by -2.10" — which reads as the
        # opposite of what it is, and is exactly the direction a winner's-curse story
        # would be built on. Name the direction from the sign.
        _d = "over" if top15["bias"] > 0 else "under"
        print(f"  NOTE the top 15 are {_d}-projected by {abs(top15['bias']):.2f} against "
              f"{res.get('model_bias', 0.0):+.2f} for the field (+ = over-projected).")
        print(f"       One gameweek is not evidence; watch the SIGN accumulate, not just "
              f"the size.")
    for x in subs:
        res[f"bias_{x['set'].replace(' ', '_').replace('>=', 'ge')}"] = x["bias"]

    z = M[(M["mean"] >= 3.0) & (M["minutes"] == 0)].sort_values("mean", ascending=False)
    print(f"\nAVAILABILITY MISSES — projected 3.0+, played 0 minutes ({len(z)}, "
          f"{z['mean'].sum():.1f} projected points)")
    for _, r in z.head(10).iterrows():
        print(f"    {r['player']:20s} {r['team']:15s} {r['mean']:.2f}")
    res["misses"] = len(z); res["missed_points"] = float(z["mean"].sum())

    u = M[(M["mean"] <= 1.5) & (M["total_points"] >= 8)].sort_values("total_points",
                                                                    ascending=False)
    print(f"\nMISSED HAULS — projected <=1.5, scored 8+ ({len(u)})")
    for _, r in u.head(8).iterrows():
        print(f"    {r['player']:20s} {r['team']:15s} proj {r['mean']:.2f} -> "
              f"{int(r['total_points'])} pts")

    print("\nPER CLUB (projected total vs actual)")
    g = M.groupby("team").apply(lambda d: pd.Series(
        {"n": len(d), "model": d["mean"].sum(), "actual": d["total_points"].sum()}),
        include_groups=False)
    g["diff"] = g["actual"] - g["model"]
    print(g.sort_values("diff").round(1).to_string())

    row = pd.DataFrame([res])
    if _os.path.exists(LEDGER):
        old = pd.read_csv(LEDGER)
        row = pd.concat([old[old["gw"] != gw], row], ignore_index=True)
    row.sort_values("gw").to_csv(LEDGER, index=False)
    print(f"\n[wrote] {LEDGER}")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int)
    ap.add_argument("--prediction", help="score this file instead of the locked board; "
                                         "only honest if its inputs predate the gameweek")
    a = ap.parse_args()
    if a.gw:
        main(a.gw, a.prediction)
    else:
        hits = sorted(glob.glob(_os.path.join(config.PREDICTIONS, "gw*_board_locked_*.csv")))
        if not hits:
            raise SystemExit("no locked predictions yet")
        g = int(_os.path.basename(hits[-1]).split("_")[0][2:])
        main(g)
