from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
new_manager_debut.py — how do OFFSEASON hires start, against what was expected of them?
========================================================================================
Restricted to managers appointed between the end of season t and matchday 1 of season
t+1 — i.e. they get a full pre-season and no mid-season handover. Their first 3, 6 and 12
matches are scored against what their club's PRIOR-SEASON strength predicts.

THE SELECTION PROBLEM, WHICH IS THE WHOLE DIFFICULTY
-----------------------------------------------------
Clubs that change manager in the summer are not a random sample — they change BECAUSE the
previous season disappointed. So "new-manager teams start below league average" is
guaranteed and meaningless. The only honest comparison is against what their own prior
strength predicts, which is why every number here is a RESIDUAL from a baseline fitted on
the clubs that did NOT change:

    baseline:  early_N(t+1) = a + b * full_season_strength(t)      [fitted on no-change]
    residual:  actual - baseline_prediction                         [scored on changers]

Fitting the baseline on non-changers only matters: fitting it on everyone would let the
changers drag the line toward themselves and shrink their own residual.

WHAT THIS CANNOT SEPARATE
-------------------------
A summer managerial change usually arrives with squad investment. Any effect measured here
is "new manager AND whatever else the club did that summer", not the manager alone. There
is no way around that with this data, and it is the main reason not to read a positive
result as "the manager caused it".

Strength = season-standardised non-penalty xG difference per match, so eras compare.
Understat 2014/15-2025/26. Manager dates: data/manager_changes.csv (club + date only;
see late_form_carryover.py for the provenance caveat on that file).

Run:  python studies/new_manager_debut.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import late_form_carryover as lfc

WINDOWS = (3, 6, 12)
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "new_manager_debut.csv")


def build_pairs(tm, flags):
    """One row per surviving (team, t -> t+1) with early-window means for each N."""
    order = sorted(tm["season"].unique(), key=lfc.season_start)
    fl = flags.set_index(["season", "team"])
    out = []
    for s0, s1 in zip(order[:-1], order[1:]):
        a = tm[tm["season"] == s0]; b = tm[tm["season"] == s1]
        for team, ga in a.groupby("team"):
            gb = b[b["team"] == team].sort_values("md")
            if len(gb) < max(WINDOWS) or len(ga) < 20:
                continue
            key = (s0, team)
            row = {"season": s0, "next": s1, "team": team,
                   "full": ga["z"].mean(),
                   "new_mgr": bool(fl.loc[key, "close_season_change"])
                   if key in fl.index else False}
            for n in WINDOWS:
                row[f"early_{n}"] = gb["z"].head(n).mean()
            out.append(row)
    return pd.DataFrame(out)


def residuals(P, n, boot=4000, seed=0):
    """Baseline fitted on NON-changers, applied to changers."""
    base = P[~P["new_mgr"]]
    y = base[f"early_{n}"].values
    b, se, r2 = lfc.ols([base["full"]], y)
    pred = lambda f: b[0] + b[1] * f
    P = P.copy()
    P[f"resid_{n}"] = P[f"early_{n}"] - pred(P["full"])
    chg = P[P["new_mgr"]]
    rng = np.random.default_rng(seed)
    v = chg[f"resid_{n}"].values
    bs = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"window": n, "n_changers": len(v), "n_baseline": len(base),
            "baseline_slope": b[1], "baseline_r2": r2,
            "mean_resid": float(v.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "share_beating_expectation": float((v > 0).mean())}, P


def main():
    tm = lfc.build()
    flags = lfc.manager_flags(tm)
    P = build_pairs(tm, flags)
    n_chg = int(P["new_mgr"].sum())
    print(f"[study] {len(P)} surviving team season-pairs; {n_chg} with an OFFSEASON "
          f"managerial change")
    print(f"        windows: first {WINDOWS} matches of season t+1")

    # selection check — confirm changers really are the disappointing clubs
    print("\n=== selection check: who changes manager in the summer? ===")
    g = P.groupby("new_mgr")["full"].agg(["count", "mean"])
    g.index = ["no change", "new manager"]
    print(g.round(3).to_string())
    print("  ^ if 'new manager' prior strength is lower, the selection is real and a")
    print("    raw league-relative comparison would be meaningless.")

    rows = []
    for n in WINDOWS:
        r, P = residuals(P, n)
        rows.append(r)
    R = pd.DataFrame(rows)

    print("\n" + "=" * 82)
    print("PERFORMANCE VS PRIOR-SEASON STRENGTH  (residual from a baseline fitted on")
    print("non-changers; units = SD of season-standardised npxG difference per match)")
    print("=" * 82)
    print(f"  {'window':>7s} {'n':>4s} {'mean resid':>11s} {'95% CI':>20s} "
          f"{'beat exp':>9s} {'base slope':>11s}")
    for _, r in R.iterrows():
        sig = "  *" if not (r.ci_lo <= 0 <= r.ci_hi) else ""
        print(f"  first {int(r.window):2d} {int(r.n_changers):4d} {r.mean_resid:+11.3f} "
              f"  ({r.ci_lo:+.3f}, {r.ci_hi:+.3f}) {r.share_beating_expectation:9.1%} "
              f"{r.baseline_slope:11.3f}{sig}")
    print("\n  * = 95% CI excludes zero")

    # ---- robustness on the one result that clears the bar ----
    print("\n=== ROBUSTNESS on the 12-match result (n=35, CI only just excludes 0) ===")
    v = P[P["new_mgr"]]["resid_12"].values
    k = int((v > 0).sum())
    # exact two-sided sign test
    from math import comb
    p_sign = 2 * sum(comb(len(v), i) for i in range(k, len(v) + 1)) / 2 ** len(v)
    print(f"  sign test: {k}/{len(v)} beat expectation, two-sided p = {min(p_sign,1):.4f}")
    tv = np.sort(v)[2:-2]
    print(f"  trimmed mean (drop 2 highest and 2 lowest): {tv.mean():+.3f} "
          f"vs untrimmed {v.mean():+.3f}")
    print("  leave-one-season-out means (is one transition carrying it?):")
    chg = P[P["new_mgr"]]
    los = []
    for s in sorted(chg["season"].unique(), key=lfc.season_start):
        sub = chg[chg["season"] != s]["resid_12"]
        los.append(sub.mean())
        print(f"    drop {s}: {sub.mean():+.3f}  (n={len(sub)})")
    print(f"  range across leave-one-out: {min(los):+.3f} to {max(los):+.3f}")

    # practical magnitude — z is per-MATCH standardised, which is a noisy unit
    sd_match = tm.groupby("season")["diff"].std().mean()
    print(f"\n  practical size: 1 z unit = {sd_match:.2f} npxG difference per match,")
    print(f"  so +{R.mean_resid.iloc[2]:.3f} z = {R.mean_resid.iloc[2]*sd_match:+.3f} "
          f"npxG/match = {R.mean_resid.iloc[2]*sd_match*12:+.2f} npxG across the 12.")

    # trajectory: is there a settling-in pattern within the first 12?
    print("\n=== trajectory — matches 1-3 vs 4-6 vs 7-12 (changers only) ===")
    chg = P[P["new_mgr"]]
    seg = {}
    for lab, lo, hi in (("matches 1-3", 0, 3), ("matches 4-6", 3, 6), ("matches 7-12", 6, 12)):
        vals = []
        for _, r in chg.iterrows():
            gb = tm[(tm["season"] == r["next"]) & (tm["team"] == r["team"])].sort_values("md")
            if len(gb) >= hi:
                vals.append(gb["z"].values[lo:hi].mean() - r["full"] * R.baseline_slope.iloc[1])
        seg[lab] = (np.mean(vals), len(vals))
    for k, (m, c) in seg.items():
        print(f"  {k:14s} mean gap vs prior strength {m:+.3f}   (n={c})")
    print("  ^ a rising line would mean new managers settle in; flat means no bedding-in")

    # secondary: does prior strength predict them AS WELL as it predicts everyone else?
    print("\n=== does prior strength still predict a new-manager team? ===")
    for n in WINDOWS:
        sub = P[P["new_mgr"]]
        b, se, r2 = lfc.ols([sub["full"]], sub[f"early_{n}"].values)
        b0, se0, r20 = lfc.ols([P[~P["new_mgr"]]["full"]],
                               P[~P["new_mgr"]][f"early_{n}"].values)
        print(f"  first {n:2d}:  new-mgr slope {b[1]:+.3f} (se {se[1]:.3f}, r2 {r2:.3f})"
              f"   vs no-change slope {b0[1]:+.3f} (se {se0[1]:.3f}, r2 {r20:.3f})")
    print("  ^ a much flatter slope would mean a new manager genuinely resets the team")

    keep = ["season", "next", "team", "full", "new_mgr"] + \
           [f"early_{n}" for n in WINDOWS] + [f"resid_{n}" for n in WINDOWS]
    P[keep].to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
