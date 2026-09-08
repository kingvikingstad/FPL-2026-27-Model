import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
minutes_per_start.py — is a player's early minutes-per-start worth anything?
=============================================================================
`src/inseason.py` updates the START prior from realised starts and stops there.
`exp_minutes` — the minutes a player is expected to last GIVEN a start — is never
touched by any in-season path (grep it: no hits in inseason.py). The appearance feed
carries the minutes column and nothing consumes it.

The consequence is that a player who starts every week and is withdrawn at half time is
indistinguishable, to the model, from one who plays every minute. Both register as
"started"; both keep whatever `exp_minutes` their prior gave them. It biases the
projection twice, because `exp_minutes` scales attacking exposure as well as appearance
points. Tzolis is the case that prompted this: cold start, p_start 0.97 from the
ownership calibration, two starts from two, 120 minutes across them.

The obvious fix is to update `exp_minutes` the way the other channels update. Whether
that fix is worth making is an empirical question, and this study answers it BEFORE any
code path is added.

PRE-REGISTERED — WRITTEN BEFORE THE RESULTS WERE SEEN
------------------------------------------------------
Unit: player-season. For each player with a usable prior season and each cutoff k:

    prior    his mean minutes-per-start in season s-1        (the analogue of exp_minutes)
    early    his mean minutes-per-start over his FIRST k starts of season s
    target   his mean minutes-per-start over his REMAINING starts of season s

    estimate(w) = (1 - w) * prior + w * early

`w = 0` is the model as it stands: the channel off. w is fitted by grid search on
OUT-OF-SAMPLE error — leave-one-season-out, fitted on every season but s and evaluated
on s — because an in-sample w is optimistically large, and large is precisely the
direction that would make the model over-react to two half-time substitutions.

PRIMARY ENDPOINT   RMSE of predicted minutes-per-start over the remainder of the season.
DECISION RULE      The channel ships ON at cutoff k only if BOTH hold:
                     (1) the LOSO-fitted w_k cuts RMSE by at least MIN_GAIN (2%)
                         against w = 0, pooled over held-out seasons; and
                     (2) the improvement has the same sign in at least CONSISTENT (3)
                         of the held-out seasons.
                   Anything else is recorded as a NULL and no code path is added.
                   Nulls are deliverables.

SECONDARY (reported, never gating)  the share of players for whom the update would move
`exp_minutes` by more than ten minutes, which says whether this is a tail correction or a
broad one.

WHAT THIS STUDY DOES NOT ESTABLISH
-----------------------------------
That the channel improves POINTS. It establishes that early minutes-per-start predicts
later minutes-per-start, which is a statement about the parameter, not about the
projection built on it. Points depend on minutes through appearance thresholds (1 point,
then 2 at sixty) and through exposure scaling on the attacking rates, and neither is
linear in `exp_minutes`. A blend of a MINUTES prior against a POINTS target is a scale
mismatch and answers nothing — an earlier draft of this file reported exactly that and
it has been removed rather than left in looking like evidence. The honest points test is
a board-level A/B once `update_exp_minutes` exists: same seed, channel off against
channel on, scored on real gameweeks. That is a separate piece of work and this study
does not pre-empt its result.

WHY MINUTES-PER-START AND NOT MINUTES
--------------------------------------
Total minutes confounds two things the model already separates: whether he starts (the
Beta prior, already updated) and how long he lasts given a start (`exp_minutes`, never
updated). Only the second is in question here, so the estimand conditions on starting.
Substitute appearances are excluded from both the estimate and the target for the same
reason — `start_minutes` in `inseason.appearances` exists for exactly this denominator.

DATA
----
Player-gameweek panel from `config.HISTORY`, keyed on player_code — never on name,
because names repeat across clubs and seasons. FPL only began publishing a real `starts`
column in 2022-23; earlier seasons have it inferred from a minutes threshold, which
miscounts a 60-minute substitute as a starter. Seasons whose starts are derived are
DROPPED rather than used with a caveat, and the study reports which survived.

Run:  python studies/minutes_per_start.py
Out:  studies/minutes_per_start.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import fpl_history as fh

# --- pre-registered constants -------------------------------------------------
CUTOFFS = (1, 2, 3, 5, 8, 10)
MIN_PRIOR_STARTS = 5      # a prior mean needs some support
MIN_REMAIN_STARTS = 3     # a target mean needs some support
W_GRID = np.round(np.arange(0.0, 1.001, 0.01), 3)
MIN_GAIN = 0.02           # 2% RMSE reduction required to ship
CONSISTENT = 3            # held-out seasons that must agree in sign
MOVE_THRESHOLD = 10.0     # minutes; "would this actually move exp_minutes?"
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "minutes_per_start.csv")


def usable_seasons(panel):
    """Seasons whose `is_start` was read, not inferred from minutes."""
    ok = []
    for s, g in panel.groupby("season"):
        if not bool(g["starts_derived"].any()):
            ok.append(s)
    return sorted(ok)


def build(panel, seasons):
    """One row per (player_code, season, cutoff) with prior, early and target."""
    panel = panel[panel["is_start"] == True].copy()          # noqa: E712
    panel["mins"] = panel["mins"].clip(upper=90)
    panel = panel.sort_values(["player_code", "season", "gw"])

    # prior: mean minutes-per-start in the previous season
    per = (panel.groupby(["player_code", "season"])
                .agg(mps=("mins", "mean"), n=("mins", "size")).reset_index())
    order = {s: i for i, s in enumerate(sorted(panel["season"].unique()))}
    per["si"] = per["season"].map(order)
    prev = per[per["n"] >= MIN_PRIOR_STARTS][["player_code", "si", "mps"]].copy()
    prev["si"] = prev["si"] + 1
    prev = prev.rename(columns={"mps": "prior"})

    rows = []
    for (code, season), g in panel.groupby(["player_code", "season"]):
        if season not in seasons:
            continue
        si = order[season]
        p = prev[(prev["player_code"] == code) & (prev["si"] == si)]
        if not len(p):
            continue
        prior = float(p["prior"].iloc[0])
        m = g["mins"].to_numpy(dtype=float)
        pts = g["points"].to_numpy(dtype=float)
        for k in CUTOFFS:
            if len(m) < k + MIN_REMAIN_STARTS:
                continue
            rows.append({
                "player_code": code, "season": season, "k": k,
                "prior": prior,
                "early": float(m[:k].mean()),
                "target": float(m[k:].mean()),
                "early_pts": float(pts[:k].mean()),
                "target_pts": float(pts[k:].mean()),
                "n_starts": len(m),
            })
    return pd.DataFrame(rows)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def best_w(df, target="target", early="early"):
    """Grid-search the blend weight that minimises RMSE on `df`."""
    best, bw = np.inf, 0.0
    for w in W_GRID:
        e = _rmse((1 - w) * df["prior"] + w * df[early], df[target])
        if e < best:
            best, bw = e, float(w)
    return bw, best


def evaluate(D, target="target", early="early"):
    """Leave-one-season-out: fit w on the other seasons, score on the held-out one."""
    out = []
    seasons = sorted(D["season"].unique())
    for k in CUTOFFS:
        Dk = D[D["k"] == k]
        if len(Dk) < 50:
            continue
        per_season, gains = [], []
        for s in seasons:
            tr, te = Dk[Dk["season"] != s], Dk[Dk["season"] == s]
            if len(tr) < 30 or len(te) < 20:
                continue
            w, _ = best_w(tr, target, early)
            e_w = _rmse((1 - w) * te["prior"] + w * te[early], te[target])
            e_0 = _rmse(te["prior"], te[target])
            gain = (e_0 - e_w) / e_0 if e_0 else 0.0
            per_season.append({"season": s, "w": w, "rmse_w": e_w, "rmse_0": e_0,
                               "gain": gain, "n": len(te)})
            gains.append(gain)
        if not per_season:
            continue
        ps = pd.DataFrame(per_season)
        # pooled: one fitted w per fold, errors accumulated across all held-out rows
        num = float((ps["rmse_w"] ** 2 * ps["n"]).sum() / ps["n"].sum())
        den = float((ps["rmse_0"] ** 2 * ps["n"]).sum() / ps["n"].sum())
        pooled_w, pooled_0 = np.sqrt(num), np.sqrt(den)
        pooled_gain = (pooled_0 - pooled_w) / pooled_0 if pooled_0 else 0.0
        w_all, _ = best_w(Dk, target, early)
        out.append({
            "k": k, "n": int(len(Dk)), "folds": len(ps),
            "w_fitted": w_all,
            "w_loso_mean": float(ps["w"].mean()),
            "rmse_w0": pooled_0, "rmse_w": pooled_w,
            "gain": pooled_gain,
            "n_positive": int((ps["gain"] > 0).sum()),
            "per_season": ps,
        })
    return out


def main():
    print("=" * 78)
    print("MINUTES-PER-START — is an early in-season signal worth anything?")
    print("=" * 78)
    print(f"pre-registered: ship at k only if gain >= {MIN_GAIN:.0%} AND the sign holds "
          f"in >= {CONSISTENT} held-out seasons")

    panel = fh.load_history(verbose=False)
    seasons = usable_seasons(panel)
    print(f"\nseasons with a READ starts column: {seasons}")
    if len(seasons) < 2:
        print("NOT ENOUGH SEASONS — cannot evaluate out of sample. No claim made.")
        return 0
    # a season is evaluable only if the one before it is also usable (it supplies the prior)
    order = {s: i for i, s in enumerate(sorted(panel["season"].unique()))}
    evaluable = [s for s in seasons if any(order.get(t, -9) == order[s] - 1 for t in seasons)]
    print(f"evaluable (previous season also usable): {evaluable}")
    if len(evaluable) < 2:
        print("NOT ENOUGH EVALUABLE SEASONS — no claim made.")
        return 0

    D = build(panel, set(evaluable))
    print(f"player-season-cutoff rows: {len(D)}  "
          f"({D['player_code'].nunique()} players)")

    res = evaluate(D)
    print("\n" + "-" * 78)
    print("PRIMARY — minutes per start over the remainder of the season")
    print("-" * 78)
    print(f"{'k':>3} {'n':>6} {'w_loso':>7} {'RMSE w=0':>9} {'RMSE w':>8} {'gain':>7} "
          f"{'folds+':>7}  verdict")
    ship = []
    rows_csv = []
    for r in res:
        ok = (r["gain"] >= MIN_GAIN) and (r["n_positive"] >= CONSISTENT)
        verdict = "SHIP" if ok else "null"
        if ok:
            ship.append(r)
        print(f"{r['k']:>3} {r['n']:>6} {r['w_loso_mean']:>7.2f} {r['rmse_w0']:>9.2f} "
              f"{r['rmse_w']:>8.2f} {r['gain']:>6.1%} "
              f"{r['n_positive']:>3}/{r['folds']:<3}  {verdict}")
        rows_csv.append({k: v for k, v in r.items() if k != "per_season"})
        rows_csv[-1]["endpoint"] = "minutes_per_start"
        rows_csv[-1]["verdict"] = verdict

    # --- secondary: is this a tail correction? ---
    print("\n" + "-" * 78)
    print("SECONDARY — would the update actually move anything?")
    print("-" * 78)
    for r in res:
        k = r["k"]
        Dk = D[D["k"] == k]
        w = r["w_loso_mean"]
        move = (w * (Dk["early"] - Dk["prior"])).abs()
        print(f"  k={k:<3} at w={w:.2f}: median |move| {move.median():5.2f} min, "
              f"{100 * (move > MOVE_THRESHOLD).mean():5.1f}% of players move more than "
              f"{MOVE_THRESHOLD:.0f} minutes")

    out = pd.DataFrame(rows_csv)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(out)} rows)")

    print("\n" + "=" * 78)
    if ship:
        ks = ", ".join(str(r["k"]) for r in ship)
        print(f"RESULT: the decision rule is MET at k = {ks}.")
        print("        Shipping weights (leave-one-season-out means):")
        for r in ship:
            print(f"          k={r['k']:<3} w={r['w_loso_mean']:.2f}  "
                  f"({r['gain']:.1%} RMSE reduction)")
        print("        The channel is worth building. It belongs in src/inseason.py as")
        print("        `update_exp_minutes`, off by default until wired and re-tested.")
        print()
        print("        SCOPE OF THE CLAIM. This says early minutes-per-start predicts later")
        print("        minutes-per-start. It does NOT say the board scores better: points")
        print("        depend on minutes through the 60-minute threshold and through")
        print("        exposure scaling, neither linear in exp_minutes. The points question")
        print("        needs a board-level A/B on real gameweeks, same seed, channel off")
        print("        against on. Until that runs the channel ships OFF by default.")
    else:
        print("RESULT: NULL. The decision rule is met at no cutoff.")
        print("        An early minutes-per-start signal does not beat the prior-season")
        print("        mean out of sample by the pre-registered margin. The channel is")
        print("        NOT built, and this file is the record of why — so it does not get")
        print("        proposed again under a different name.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
