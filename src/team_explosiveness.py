from __future__ import annotations
import config
"""
team_explosiveness.py — do teams differ in how EXPLOSIVE their scoring is?
===========================================================================
The engine draws team goals as `rng.poisson(lam)`. Poisson fixes variance = mean, so it
has no freedom to say that one side's goals arrive in bursts and another's dribble out
one at a time. If teams genuinely differ in that second moment, every tail quantity
built on top — `captaincy.p_haul`, `ceiling`, the differential edge — is mis-stated for
the teams at the extremes, and captaincy is chosen precisely on the tail.

So the question is worth asking. The answer, measured on 25/26, is TWO nulls and one
real correction that points the OPPOSITE way to the premise:

  1. [NULL] Teams do not differ in dispersion. Per-club Pearson dispersion ranges 0.65
     to 2.28, but its split-half reliability is r=+0.006 against a simulated
     true-Poisson null of +0.012 with a 95% band of (-0.426, +0.480). Entirely noise:
     with 38 matches a club's dispersion is barely estimated at all.

  2. [NULL] Return concentration (are a team's goal involvements bunched into one
     player?) has split-half r=+0.36 against a shuffled-club null 95% band of
     (-0.42, +0.40). Borderline, inside the band, not established.

  3. [VERIFIED] The league's UPPER TAIL IS THINNER THAN POISSON, not fatter. Teams
     score 4+ goals on 4.08% of team-matches where an independent-Poisson model with
     the same lambdas implies 6.54%.

WHY (3) NEEDED A SIMULATED NULL, NOT A Z-TEST
----------------------------------------------
`lambda` has to be estimated, and P(4+) is convex in lambda, so a NOISY lambda-hat
inflates the implied tail by Jensen and manufactures a thin-tail result out of nothing.
An in-sample fit has the opposite problem: 41 parameters on 760 observations chases the
data and drove the naive Pearson dispersion down to 0.887, which looked like strong
under-dispersion and was pure overfit — cross-fitting moved it to 1.056. The per-club
dispersion spread is likewise an in-sample illusion: 0.63-1.40 in sample, 0.65-2.28
cross-fitted, and neither range carries signal.

So the test simulates from a TRUE Poisson with the fitted lambdas, pushes it through the
identical cross-fit pipeline, and compares. The null's own P(4+) gap is -0.0077 (that is
the convexity bias, made visible). The observed -0.0247 sits at the 0.2nd percentile of
that null, and P(5+) at the 1.5th. The thin tail survives.

WHAT THIS MEANS FOR THE MODEL  [JUDGMENT]
------------------------------------------
The engine OVERSTATES blowouts. Not by a little: it implies 60% more 4+ goal team-games
than happen. Those are exactly the fixtures a captaincy pick concentrates on, so the
error lands where it is most expensive.

The correction here is a non-parametric calibration of the count distribution, and it is
OFF BY DEFAULT (`apply_tail_calibration` must be called explicitly). It is not wired
into `TeamModel`, because changing the goal distribution moves the clean-sheet engine
that is validated at GA r=0.89 / CS r=0.93, and that revalidation is a separate exercise
against scored gameweeks. Shipping it silently would be the failure mode this project is
organised against.

NOTE ON THE MARKET GUARD
------------------------
`style_matchup.beats_the_market()` gates new team signals, but it is a Poisson SCORE
test on the MEAN — it asks whether a covariate predicts goals beyond the market lambda.
It is structurally incapable of detecting a second-moment claim: a distribution can match
the market's mean exactly and still have the wrong tail. So that gate neither passes nor
fails this, and calling it here would be a category error. The relevant validation is
tail calibration against out-of-sample scorelines, which is what this module measures.
"""
import glob
import os
import numpy as np
import pandas as pd

MAX_GOALS = 9


def team_match_goals(season="2025-2026", base=None):
    """One row per team-match: club, opponent, venue, goals scored, gameweek."""
    root = base or config.repo(season)
    fs = sorted(glob.glob(os.path.join(root, "By Gameweek", "GW*", "matches.csv")))
    if not fs:
        return pd.DataFrame()
    m = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    m = m[m["match_id"].astype(str).str.contains("-prem-", na=False)]
    m = m.drop_duplicates("match_id")
    t = pd.read_csv(os.path.join(root, "teams.csv"))
    c2n = dict(zip(pd.to_numeric(t["code"], errors="coerce"), t["name"]))
    m["home"] = pd.to_numeric(m["home_team"], errors="coerce").map(c2n)
    m["away"] = pd.to_numeric(m["away_team"], errors="coerce").map(c2n)
    m = m.dropna(subset=["home", "away"])
    hs = pd.to_numeric(m["home_score"], errors="coerce")
    as_ = pd.to_numeric(m["away_score"], errors="coerce")
    long = pd.concat([
        m.assign(club=m["home"], opp=m["away"], g=hs, is_home=True),
        m.assign(club=m["away"], opp=m["home"], g=as_, is_home=False)],
        ignore_index=True)
    return long.dropna(subset=["g"])[["match_id", "gameweek", "club", "opp", "g",
                                      "is_home"]].reset_index(drop=True)


def _design(df, ci, T):
    n = len(df)
    X = np.zeros((n, 2 * T + 1))
    X[np.arange(n), df["club"].map(ci).values] = 1.0
    X[np.arange(n), T + df["opp"].map(ci).values] = 1.0
    X[:, -1] = df["is_home"].astype(float).values
    return X


def _poisson_fit(X, y, ridge=1e-3, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        lam = np.exp(np.clip(X @ b, -8, 4))
        z = X @ b + (y - lam) / np.maximum(lam, 1e-9)
        A = X.T @ (X * lam[:, None]) + ridge * np.eye(X.shape[1])
        b = np.linalg.solve(A, X.T @ (lam * z))
    return b


def crossfit_lambda(long):
    """Out-of-sample lambda for every team-match, by fitting attack/defence/home on one
    half of the season and predicting the other.

    In-sample lambdas are not usable here: 41 parameters on 760 rows overfits enough to
    push Pearson dispersion to 0.887 when the honest value is 1.056.
    """
    clubs = sorted(long["club"].unique())
    ci = {c: i for i, c in enumerate(clubs)}
    T = len(clubs)
    med = long["gameweek"].median()
    i1 = np.flatnonzero((long["gameweek"] <= med).values)
    i2 = np.flatnonzero((long["gameweek"] > med).values)
    X = _design(long, ci, T)
    y = long["g"].values.astype(float)
    lam = np.full(len(long), np.nan)
    for tr, te in ((i1, i2), (i2, i1)):
        b = _poisson_fit(X[tr], y[tr])
        lam[te] = np.exp(np.clip(X[te] @ b, -8, 4))
    if np.isfinite(lam).all() and lam.mean() > 0:
        lam = lam * (y.mean() / lam.mean())   # halves differ in scoring rate
    return lam


def tail_calibration(long, lam=None):
    """Empirical P(G=k) against the Poisson-implied P(G=k), per count.

    Non-parametric on purpose: a named distribution (negative binomial, COM-Poisson)
    would impose a shape, and the finding here is about the top of the distribution
    only. Counts 4 and above are pooled because 760 matches cannot resolve them apart.
    """
    from scipy import stats
    if lam is None:
        lam = crossfit_lambda(long)
    y = long["g"].values.astype(float)
    rows = []
    for k in range(0, 4):
        obs = float((y == k).mean())
        imp = float(stats.poisson.pmf(k, lam).mean())
        rows.append({"goals": str(k), "observed": obs, "poisson": imp,
                     "ratio": obs / imp if imp > 0 else np.nan})
    obs = float((y >= 4).mean())
    imp = float(stats.poisson.sf(3, lam).mean())
    rows.append({"goals": "4+", "observed": obs, "poisson": imp,
                 "ratio": obs / imp if imp > 0 else np.nan})
    return pd.DataFrame(rows)


def dispersion_by_team(long, lam=None):
    """Per-club Pearson dispersion, plus the split-half reliability that decides
    whether the spread is signal. Compare against `dispersion_null` before believing it.
    """
    if lam is None:
        lam = crossfit_lambda(long)
    d = long.assign(lam=lam)
    d["pear2"] = (d["g"] - d["lam"]) ** 2 / np.maximum(d["lam"], 1e-9)
    g = d.groupby("club").agg(n=("pear2", "size"), dispersion=("pear2", "mean"))
    med = d["gameweek"].median()
    a = d[d["gameweek"] <= med].groupby("club")["pear2"].mean()
    b = d[d["gameweek"] > med].groupby("club")["pear2"].mean()
    j = pd.concat([a.rename("h1"), b.rename("h2")], axis=1).dropna()
    g["reliability"] = float(j["h1"].corr(j["h2"])) if len(j) >= 5 else np.nan
    return g.sort_values("dispersion", ascending=False).reset_index()


def dispersion_null(long, lam=None, n_sim=200, seed=0):
    """What that reliability looks like when the data really ARE Poisson."""
    if lam is None:
        lam = crossfit_lambda(long)
    rng = np.random.default_rng(seed)
    med = long["gameweek"].median()
    lo = long["gameweek"].values <= med
    out = []
    for _ in range(n_sim):
        ys = rng.poisson(lam)
        p2 = (ys - lam) ** 2 / np.maximum(lam, 1e-9)
        t = pd.DataFrame({"club": long["club"].values, "p2": p2, "lo": lo})
        a = t[t["lo"]].groupby("club")["p2"].mean()
        b = t[~t["lo"]].groupby("club")["p2"].mean()
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        out.append(float(j["a"].corr(j["b"])))
    return np.array(out)


def tail_null(long, n_sim=400, seed=0):
    """Distribution of the observed-minus-implied tail gap under a TRUE Poisson,
    pushed through the identical cross-fit pipeline. This is what separates a real
    thin tail from the convexity bias of an estimated lambda."""
    from scipy import stats
    clubs = sorted(long["club"].unique())
    ci = {c: i for i, c in enumerate(clubs)}
    T = len(clubs)
    X = _design(long, ci, T)
    y = long["g"].values.astype(float)
    lam_true = np.exp(np.clip(X @ _poisson_fit(X, y), -8, 4))
    rng = np.random.default_rng(seed)
    med = long["gameweek"].median()
    i1 = np.flatnonzero((long["gameweek"] <= med).values)
    i2 = np.flatnonzero((long["gameweek"] > med).values)

    def gaps(yy):
        lam = np.full(len(yy), np.nan)
        for tr, te in ((i1, i2), (i2, i1)):
            lam[te] = np.exp(np.clip(X[te] @ _poisson_fit(X[tr], yy[tr]), -8, 4))
        lam = lam * (yy.mean() / lam.mean())
        return {k: float((yy >= k).mean()) - float(stats.poisson.sf(k - 1, lam).mean())
                for k in (3, 4, 5)}

    sims = [gaps(rng.poisson(lam_true).astype(float)) for _ in range(n_sim)]
    return gaps(y), pd.DataFrame(sims)


def apply_tail_calibration(lam, cal, max_goals=MAX_GOALS):
    """Reshape a Poisson(lam) pmf by the measured calibration ratios, renormalised.

    OFF BY DEFAULT — nothing calls this. Wiring it into `TeamModel` moves the
    clean-sheet engine that is validated at GA r=0.89 / CS r=0.93, so it needs a
    revalidation against scored gameweeks first.
    """
    from scipy import stats
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    ks = np.arange(0, max_goals + 1)
    pmf = stats.poisson.pmf(ks[None, :], lam[:, None])
    ratio = np.ones(len(ks))
    lut = dict(zip(cal["goals"], cal["ratio"]))
    for k in ks:
        ratio[k] = lut.get(str(k), lut.get("4+", 1.0)) if k < 4 else lut.get("4+", 1.0)
    out = pmf * ratio[None, :]
    s = out.sum(axis=1, keepdims=True)
    return out / np.maximum(s, 1e-12)


def selftest():
    from scipy import stats
    rng = np.random.default_rng(0)
    clubs = [f"C{i}" for i in range(20)]
    rows = []
    gw = 0
    for rnd in range(38):
        gw += 1
        perm = rng.permutation(clubs)
        for i in range(0, 20, 2):
            rows.append(dict(match_id=f"m{rnd}_{i}", gameweek=gw, club=perm[i],
                             opp=perm[i + 1], is_home=True))
            rows.append(dict(match_id=f"m{rnd}_{i}", gameweek=gw, club=perm[i + 1],
                             opp=perm[i], is_home=False))
    L = pd.DataFrame(rows)
    att = dict(zip(clubs, rng.normal(0, 0.3, 20)))
    L["lam_true"] = np.exp(0.3 + L["club"].map(att) - L["opp"].map(att)
                           + 0.1 * L["is_home"])

    # 1. TRUE Poisson data must NOT be flagged as having a thin tail
    L1 = L.assign(g=rng.poisson(L["lam_true"]).astype(float))
    obs, sims = tail_null(L1[["match_id", "gameweek", "club", "opp", "g", "is_home"]],
                          n_sim=120, seed=1)
    pct = 100 * (sims[4] < obs[4]).mean()
    assert 1.0 <= pct <= 99.0, f"Poisson data wrongly flagged, pct={pct:.1f}"

    # 2. genuinely thin-tailed data MUST be flagged: cap goals at 3
    y2 = np.minimum(rng.poisson(L["lam_true"]), 3).astype(float)
    L2 = L.assign(g=y2)
    obs2, sims2 = tail_null(L2[["match_id", "gameweek", "club", "opp", "g", "is_home"]],
                            n_sim=120, seed=2)
    pct2 = 100 * (sims2[4] < obs2[4]).mean()
    assert pct2 < 5.0, f"planted thin tail not detected, pct={pct2:.1f}"

    # 3. calibration must renormalise and move mass off the tail
    cal = tail_calibration(L2[["match_id", "gameweek", "club", "opp", "g", "is_home"]])
    pmf = apply_tail_calibration(np.array([1.4, 2.0]), cal)
    assert np.allclose(pmf.sum(axis=1), 1.0), pmf.sum(axis=1)
    raw_tail = stats.poisson.sf(3, 1.4)
    assert pmf[0, 4:].sum() < raw_tail, "calibration should thin the tail here"

    # 4. dispersion reliability on true Poisson must sit inside its own null
    disp = dispersion_by_team(L1[["match_id", "gameweek", "club", "opp", "g", "is_home"]])
    null = dispersion_null(L1[["match_id", "gameweek", "club", "opp", "g", "is_home"]],
                           n_sim=60, seed=3)
    r = disp["reliability"].iloc[0]
    assert np.percentile(null, 1) <= r <= np.percentile(null, 99), (
        f"Poisson dispersion reliability {r:.3f} outside its own null")
    print("SELFTEST OK: true-Poisson data not flagged, planted thin tail detected "
          f"({pct2:.1f}th pctile), calibration renormalises and thins, dispersion "
          "reliability sits inside its null.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    print(__doc__)
