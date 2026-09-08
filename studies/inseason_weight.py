import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
inseason_weight.py — how much is an early-season result worth?
===============================================================
Calibration for `src/inseason.py`. The model currently projects 26/27 entirely from
25/26 and never updates (docs/GW1_SCORING_2026-08-22.md §4). Turning that on needs one
number: how much weight a club's first k matches deserve against its prior-season
strength. That number is measured here, not asserted.

THIS IS A CALIBRATION, NOT A HYPOTHESIS TEST
---------------------------------------------
The primary output is a fitted weight w_k, so there is no null to reject and no
multiplicity to correct on it. Two genuine hypotheses ride along and ARE pre-registered
with a Bonferroni correction, because they are the kind of claim this repo keeps having
to reject:

  H-A  PROCESS BEATS RESULTS. Early shots-on-target difference predicts rest-of-season
       strength at least as well as early goal difference. If true, the in-season update
       must run on process, not on scorelines — which is the same principle as the
       project's standing "xG over goals" rule, tested here on the update channel.

  H-B  OVER-PERFORMANCE DOES NOT PERSIST. A club whose early goal difference exceeds
       what its shot difference implies gets no credit for the excess: the residual has
       no predictive value for the rest of the season.

  alpha = 0.025 each.

Prompted concretely by Hull 2-0 Man United in GW1 2026/27: Hull won on 1.26 xG to 1.81,
with 94% of their xG from set plays and five goalkeeper saves. H-B is the question of
whether that win means anything for Hull's season beyond the three points.

METHOD
------
Unit: club-season. For each club and each k:

    prior      club's prior-season goal difference per game (GD/g)
    early_g    GD/g over the club's OWN matches 1..k
    early_s    shots-on-target difference per game over the same matches
    target     GD/g over matches k+1..N        <- what we are predicting

The weight is defined by the blend that best predicts `target`:

    estimate = (1 - w) * prior + w * early

and w_k is found by grid search on OUT-OF-SAMPLE error — fitted on every season but s,
evaluated on s, for every s. An in-sample w would be optimistically large, which is
exactly the direction that would make the model over-react to early results.

Promoted clubs get their own arm: they have no prior PL season, so their `prior` is the
mean promoted-club GD/g, and the question becomes how fast their own results should
displace that base rate.

DATA
----
football-data.co.uk, cached by `history.download_seasons`. Goals and dates from 1993/94
(31 seasons); shots and shots-on-target from 2000/01 (24 seasons), so H-A and H-B run on
the shorter panel. 2003/04 and 2004/05 are absent from the cache and are simply missing.

Run:  python studies/inseason_weight.py
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
import history

KS = [1, 2, 3, 5, 8, 10]
GRID = np.round(np.arange(0.0, 1.0001, 0.01), 4)
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "inseason_weight.csv")


def load(cache_dir=None):
    """Club-match long form with goals and, where available, shots on target."""
    cache_dir = cache_dir or _os.path.join(config.SCRATCH, "fd_hist")
    frames = []
    for p in sorted(glob.glob(_os.path.join(cache_dir, "E0_*.csv"))):
        d = pd.read_csv(p, encoding="latin-1")
        if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"}.issubset(d.columns):
            continue
        d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"]).copy()
        d["season"] = _os.path.basename(p)[3:-4]
        d["date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        d = d.dropna(subset=["date"])
        for c in ("HST", "AST"):
            if c not in d.columns:
                d[c] = np.nan
        d["HomeTeam"] = d["HomeTeam"].replace(history.NAME_NORM)
        d["AwayTeam"] = d["AwayTeam"].replace(history.NAME_NORM)
        frames.append(d[["season", "date", "HomeTeam", "AwayTeam",
                         "FTHG", "FTAG", "HST", "AST"]])
    d = pd.concat(frames, ignore_index=True)
    d["yr"] = d["season"].map(history.season_start_year)
    h = d.rename(columns={"HomeTeam": "team", "AwayTeam": "opp", "FTHG": "gf",
                          "FTAG": "ga", "HST": "sf", "AST": "sa"})
    a = d.rename(columns={"AwayTeam": "team", "HomeTeam": "opp", "FTAG": "gf",
                          "FTHG": "ga", "AST": "sf", "HST": "sa"})
    cols = ["season", "yr", "date", "team", "opp", "gf", "ga", "sf", "sa"]
    L = pd.concat([h[cols], a[cols]], ignore_index=True)
    L["pts"] = np.where(L.gf > L.ga, 3, np.where(L.gf == L.ga, 1, 0))
    L = L.sort_values(["yr", "team", "date"]).reset_index(drop=True)
    L["md"] = L.groupby(["yr", "team"]).cumcount() + 1
    return L


def panel(L, k):
    """One row per club-season at horizon k."""
    seasons = sorted(L.yr.unique())
    prev_gd = {}
    for yr in seasons:
        p = L[L.yr == yr - 1]
        if len(p):
            g = p.groupby("team").apply(
                lambda d: (d.gf - d.ga).mean(), include_groups=False)
            for t, v in g.items():
                prev_gd[(yr, t)] = float(v)
    rows = []
    for yr in seasons:
        S = L[L.yr == yr]
        prev_teams = set(L[L.yr == yr - 1]["team"])
        if not prev_teams:
            continue
        for team, g in S.groupby("team"):
            early = g[g.md <= k]
            rest = g[g.md > k]
            if len(early) < k or len(rest) < 5:
                continue
            rows.append({
                "yr": yr, "season": g["season"].iloc[0], "team": team,
                "promoted": int(team not in prev_teams),
                "prior": prev_gd.get((yr, team), np.nan),
                "early_g": float((early.gf - early.ga).mean()),
                "early_s": float((early.sf - early.sa).mean()),
                "early_pts": float(early.pts.mean()),
                "target": float((rest.gf - rest.ga).mean()),
                "rest_pts": float(rest.pts.mean()),
                "rest_n": len(rest),
            })
    return pd.DataFrame(rows)


def best_w(df, early_col, prior_col="prior"):
    """Out-of-sample optimal blend weight, by leave-one-season-out grid search."""
    d = df.dropna(subset=[prior_col, early_col, "target"])
    if len(d) < 50:
        return np.nan, np.nan, 0
    errs = np.zeros(len(GRID))
    for yr in sorted(d.yr.unique()):
        te = d[d.yr == yr]
        if not len(te):
            continue
        for i, w in enumerate(GRID):
            est = (1 - w) * te[prior_col].values + w * te[early_col].values
            errs[i] += float(np.sum((est - te["target"].values) ** 2))
    n = len(d)
    rmse = np.sqrt(errs / n)
    j = int(np.argmin(errs))
    return float(GRID[j]), float(rmse[j]), n


def r2_oos(df, cols):
    """Leave-one-season-out R^2 of an OLS on `cols` predicting target."""
    d = df.dropna(subset=cols + ["target"])
    if len(d) < 50:
        return np.nan, 0
    sse = sst = 0.0
    for yr in sorted(d.yr.unique()):
        tr, te = d[d.yr != yr], d[d.yr == yr]
        if len(te) == 0 or len(tr) < 30:
            continue
        X = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
        b = np.linalg.pinv(X.T @ X) @ X.T @ tr["target"].values
        Xe = np.column_stack([np.ones(len(te))] + [te[c].values for c in cols])
        pred = Xe @ b
        sse += float(np.sum((te["target"].values - pred) ** 2))
        sst += float(np.sum((te["target"].values - tr["target"].mean()) ** 2))
    return (1 - sse / sst if sst else np.nan), len(d)


def ols_cluster(y, X, groups):
    from scipy import stats as st
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gv in np.unique(groups):
        m = groups == gv
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    V = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1))
    se = np.sqrt(np.diag(V)); t = b / se
    p = 2 * (1 - st.t.cdf(np.abs(t), df=max(G - 1, 1)))
    return b, se, t, p


def main():
    L = load()
    print(f"[data] {L.yr.nunique()} seasons, {len(L):,} club-matches; "
          f"shots present from {int(L.dropna(subset=['sf']).yr.min())}")

    rows = []
    print("\n" + "=" * 78)
    print("1. HOW MUCH WEIGHT DO A CLUB'S FIRST k MATCHES DESERVE?")
    print("   estimate = (1-w)*prior_season_GD + w*early_GD, w by leave-one-season-out")
    print("=" * 78)
    print(f"  {'k':>3s}  {'n':>5s}  {'w(goals)':>9s} {'w(shots)':>9s}   "
          f"{'R2 prior':>9s} {'R2 +goals':>10s} {'R2 +shots':>10s}")
    for k in KS:
        P = panel(L, k)
        est = P[P.promoted == 0]
        wg, _, n = best_w(est, "early_g")
        ws, _, _ = best_w(est.dropna(subset=["early_s"]), "early_s")
        r0, _ = r2_oos(est, ["prior"])
        r1, _ = r2_oos(est, ["prior", "early_g"])
        r2, _ = r2_oos(est.dropna(subset=["early_s"]), ["prior", "early_s"])
        print(f"  {k:>3d}  {n:>5d}  {wg:>9.2f} {ws:>9.2f}   "
              f"{r0:>9.3f} {r1:>10.3f} {r2:>10.3f}")
        rows.append({"arm": "established", "k": k, "n": n, "w_goals": wg,
                     "w_shots": ws, "r2_prior": r0, "r2_goals": r1, "r2_shots": r2})

    print("\n" + "=" * 78)
    print("2. PROMOTED CLUBS — no prior PL season, so how fast do their own results")
    print("   displace the promoted base rate?")
    print("=" * 78)
    print(f"  {'k':>3s}  {'n':>4s}  {'w(goals)':>9s}   {'base GD/g':>10s} {'rest GD/g':>10s}")
    for k in KS:
        P = panel(L, k)
        pro = P[P.promoted == 1].copy()
        if len(pro) < 30:
            continue
        base = pro["target"].mean()
        pro["prior"] = base                       # the base rate IS their prior
        wg, _, n = best_w(pro, "early_g")
        print(f"  {k:>3d}  {n:>4d}  {wg:>9.2f}   {base:>10.3f} {pro.target.mean():>10.3f}")
        rows.append({"arm": "promoted", "k": k, "n": n, "w_goals": wg,
                     "w_shots": np.nan, "r2_prior": np.nan,
                     "r2_goals": np.nan, "r2_shots": np.nan})

    print("\n" + "=" * 78)
    print("3. PRE-REGISTERED HYPOTHESES (Bonferroni alpha = 0.025)")
    print("=" * 78)
    P3 = panel(L, 3).dropna(subset=["early_s"])
    est3 = P3[P3.promoted == 0]

    # H-A: process vs results
    rg, _ = r2_oos(est3, ["prior", "early_g"])
    rs, _ = r2_oos(est3, ["prior", "early_s"])
    rb, _ = r2_oos(est3, ["prior", "early_g", "early_s"])
    d = est3.dropna(subset=["prior", "early_g", "early_s", "target"])
    X = np.column_stack([np.ones(len(d)), d.prior, d.early_g, d.early_s])
    b, se, t, p = ols_cluster(d.target.values, X, d.yr.values)
    print(f"\n  H-A  process vs results, k=3, n={len(d)}")
    print(f"       out-of-sample R2:  prior+goals {rg:.4f} | prior+shots {rs:.4f} | "
          f"both {rb:.4f}")
    print(f"       in one model:  early_g {b[2]:+.4f} (p={p[2]:.3f})   "
          f"early_s {b[3]:+.4f} (p={p[3]:.3f})")

    # H-B: does over-performance persist?
    # residual of early goal difference on early shot difference = the unexplained part
    dd = est3.dropna(subset=["early_g", "early_s", "prior", "target"]).copy()
    Xs = np.column_stack([np.ones(len(dd)), dd.early_s.values])
    bb = np.linalg.pinv(Xs.T @ Xs) @ Xs.T @ dd.early_g.values
    dd["overperf"] = dd.early_g.values - Xs @ bb
    X2 = np.column_stack([np.ones(len(dd)), dd.prior, dd.early_s, dd.overperf])
    b2, se2, t2, p2 = ols_cluster(dd.target.values, X2, dd.yr.values)
    print(f"\n  H-B  does early over-performance persist?  n={len(dd)}")
    print(f"       beta(overperf) {b2[3]:+.4f} GD/g per unit   SE {se2[3]:.4f}   "
          f"t={t2[3]:+.2f}   p={p2[3]:.3f}   "
          f"{'PERSISTS' if p2[3] < 0.025 else 'DOES NOT PERSIST'}")
    print(f"       for comparison, beta(early_s) {b2[2]:+.4f} (p={p2[2]:.3f})")
    print(f"       MDE at 80% power: {2.8*se2[3]:.4f} GD/g")

    # the same question for promoted clubs specifically — the Hull case
    pro3 = P3[P3.promoted == 1].dropna(subset=["early_g", "early_s", "target"]).copy()
    if len(pro3) > 40:
        Xp = np.column_stack([np.ones(len(pro3)), pro3.early_s.values])
        bp = np.linalg.pinv(Xp.T @ Xp) @ Xp.T @ pro3.early_g.values
        pro3["overperf"] = pro3.early_g.values - Xp @ bp
        X3 = np.column_stack([np.ones(len(pro3)), pro3.early_s, pro3.overperf])
        b3, se3, t3, p3v = ols_cluster(pro3.target.values, X3, pro3.yr.values)
        print(f"\n  H-B, PROMOTED CLUBS ONLY (the Hull case)  n={len(pro3)}")
        print(f"       beta(overperf) {b3[2]:+.4f}  SE {se3[2]:.4f}  p={p3v[2]:.3f}")
        print(f"       beta(early_s)  {b3[1]:+.4f}  SE {se3[1]:.4f}  p={p3v[1]:.3f}")
        print(f"       MDE {2.8*se3[2]:.4f} GD/g")

        # survival: does an early win change a promoted club's odds of staying up?
        pro3["won_early"] = (pro3.early_pts > 0.9).astype(int)
        S = L.groupby(["yr", "team"], as_index=False).agg(pts=("pts", "sum"))
        S["rank"] = S.groupby("yr")["pts"].rank(ascending=False, method="min")
        S["n_clubs"] = S.groupby("yr")["team"].transform("size")
        S["relegated"] = (S["rank"] > S["n_clubs"] - 3).astype(int)
        pro3 = pro3.merge(S[["yr", "team", "relegated", "pts"]], on=["yr", "team"], how="left")
        g = pro3.groupby("won_early").agg(n=("team", "size"),
                                          relegated=("relegated", "mean"),
                                          final_pts=("pts", "mean"))
        print(f"\n  PROMOTED CLUBS: early points (matches 1-3) vs survival")
        print(g.to_string())
        rows.append({"arm": "H-B promoted overperf", "k": 3, "n": len(pro3),
                     "w_goals": np.nan, "w_shots": np.nan, "r2_prior": np.nan,
                     "r2_goals": b3[2], "r2_shots": p3v[2]})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
