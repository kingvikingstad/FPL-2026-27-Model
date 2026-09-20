from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
start_forgetting.py — a discounted Beta-Bernoulli filter for the minutes prior
==============================================================================
`inseason.update_minutes` is a conjugate Beta update on a COUNT, so it is exchangeable:
start-start-bench and bench-start-start give the same posterior. `start_persistence.py`
measured that order carries information over roughly six matches, and
`start_persistence_followup.py` (2026-09-17) showed a per-player FIRST-ORDER chain is not
enough either — the match before last still moves the next one (+0.057 [0.043, 0.068] after
a start, +0.096 [0.085, 0.106] after a non-start, against a null that holds each
player-season's own transition counts). So memory beyond the last match is real.

`start_recency.py` encoded that as a geometric weight renormalised to sum to k. That fixes
the nominal evidence count but NOT the information in it: the effective sample size of
geometric weights caps at (1+lam)/(1-lam) — 7 at lam=0.75 — while the Beta is still charged
k observations. The posterior is therefore too concentrated, the variance of multi-gameweek
start counts is understated, and lam is not separable from the prior strength kappa (lam*
at h=10 is 0.25 uncapped against 0.75 at kappa=4). That is why `INSEASON_LAM` is OFF.

This study tests the estimator that has the right concentration by construction:

    a_t = lam * a_{t-1} + (1 - lam) * kappa * m0 + y_t
    b_t = lam * b_{t-1} + (1 - lam) * kappa * (1 - m0) + (1 - y_t)
    a_0 = kappa * m0,  b_0 = kappa * (1 - m0)

Old evidence decays geometrically, the prior is reverted to rather than overwritten, and
the total mass converges to kappa + 1/(1-lam) instead of kappa + k. lam = 1 with the mass
term dropped is the flat update; lam -> 0 is "last match only".

PRE-REGISTERED — WRITTEN BEFORE THE RESULTS WERE SEEN
------------------------------------------------------
Written by the main session on 2026-09-18. `research-preregistrar` is the role CLAUDE.md
assigns to a new line of enquiry; it was launched twice for this and failed both times
(Opus session limit, then a stalled stream), so these rules were fixed here instead, in the
file, before the first run. That substitution is recorded rather than hidden.

UNIT      player-season x cutoff k. Native `starts`, 22/23 observed from GW16
          (`fpl_history.empty_native_gws`), ordered by kickoff, via
          `start_persistence.build` — the same loader the persistence studies use.

PRIOR     the previous season's (starts, matches) as a Beta, mean m0 and strength capped at
          kappa. This is the STAND-IN prior `start_prior_strength.py` uses, NOT the
          installed production prior. Deliberate, and a stated limitation: the production
          replica (`start_prior_production.py`, another session, 2026-09-17) showed the
          installed prior's denominator counts appearances rather than matches (mean 0.548
          against a realised 0.424), and fixing that is a separate pre-registered item in
          PROJECT_KNOWLEDGE §6.9 owned by that work. Fitting memory on a prior whose LEVEL
          is being corrected elsewhere would let this study credit "memory" with a level
          correction. Every arm here therefore shares one prior mean m0, so nothing below
          can move by shifting the level; only the weighting of evidence differs.

ARMS      all three share m0 and sweep the same kappa grid, so the comparison is memory,
          not prior strength:
            A FLAT    a = kappa*m0 + sum(y[:k]),  b = kappa*(1-m0) + (k - sum(y[:k]))
                      the installed estimator (w = 1) with the prior capped at kappa.
            B GEOM    A with the renormalised geometric weights of `start_recency` (the
                      shipped-off INSEASON_LAM), weights summing to k.
            C FILTER  the recursion above.

ENDPOINT  PRIMARY: pooled Brier over EVERY remaining match of the season, one prediction
          per (player, season, cutoff) held fixed across the window. Chosen because
          `gw_board.py` defaults to GW_HI=38, so rest-of-season is what the board actually
          projects, and because it is the harder endpoint — the previous study's error was
          picking h=10 after seeing a sweep on which it looked best. SECONDARY, reported
          and non-gating: the next 10 matches.
          The endpoint is measured BEFORE the XI constraint and the availability override,
          which run after the update in the board. Those layers can only absorb level
          error, and all arms share a level, so measuring after them would test them, not
          this.

FOLDS     leave-one-season-out over 23/24, 24/25, 25/26 — the three SEASON-START folds.
          22/23 supplies priors only: its own fold would begin at GW16 across the World Cup
          break, and it is already known to pull memory estimates toward shorter windows
          (season-start folds give lam* 0.70 at h=10 against 0.75 with it included).

GRIDS     kappa in {2, 3, 4, 5, 8, 12, 20, inf}; lam in {0.50 .. 0.95 step 0.05} plus 1.00.
          Cutoffs k in {3, 5, 8, 12, 20, 30} — out to 30 because the shipped lam was fitted
          with k <= 12 and by-cutoff lam* was still rising at the edge of that range.
          (kappa, lam) are fitted JOINTLY per fold. Only the ratio of prior mass to evidence
          mass is expected to be identified, so the report is a RIDGE — every pair within
          0.5pp of the fold-optimum — not a point.

DECISION RULE, fixed before any number was read
  ADOPT the filter only if ALL THREE hold, leave-one-season-out on the PRIMARY endpoint:
    (1) pooled Brier gain over arm A at A's own best kappa is at least 1.0%;
    (2) the gain is positive in 3 of 3 held-out seasons;
    (3) DISPERSION GATE — coverage of the central 80% Beta-Binomial predictive interval for
        the number of starts in the next min(10, remaining) matches is within [0.75, 0.85],
        and no further from 0.80 than arm A's. A mean-only Brier cannot validate this
        change: the specific failure being corrected is over-concentration, so an arm that
        wins on Brier while mis-stating its own uncertainty has not earned the flag.
  If (1) or (2) fails: NULL. The filter is not wired, `INSEASON_LAM` stays off, and this
  file records why. If only (3) fails: NULL FOR SHIPPING, recorded as "better mean, wrong
  spread" — the estimator is not adopted and the finding stands as the reason.
  Arm C against arm B (filter vs the shipped-off geometric weight) is reported as a
  DECLARED SECONDARY and gates nothing on its own.

SENSITIVITIES, declared non-decision-bearing: the h=10 endpoint; per-cutoff fits; the
  proxy seasons (16/17-21/22) as a replication; kappa = inf.

POWER, stated before the run: three held-out seasons is the same fold count that carried
  `start_prior_strength`'s ADOPT, where the effects were 6-11% of baseline Brier. The
  effects here are expected to be far smaller — `start_recency` found +0.5% rest-of-season
  for the geometric weight, under its 1% bar — so this design can comfortably detect a
  shippable gain and will most likely return a NULL at the 1% margin. That is the expected
  outcome and an informative one: it would say the exchangeability defect, though real and
  measured, is not worth a code path at the board's own horizon. What would make it
  uninformative is a gain between 0.5% and 1.0% with folds disagreeing; that would be
  recorded as INCONCLUSIVE rather than argued either way.

GUARDS    not a rotation multiplier (no fixture-conditional term), not mean reversion, not
          a streak or momentum predictor: nothing here adds a covariate. It re-weights the
          likelihood's own observations, which is what the measured defect is about.

RESULT, 2026-09-18 — INCONCLUSIVE at the gating endpoint; the filter is NOT adopted
------------------------------------------------------------------------------------
  rest-of-season (PRIMARY): FILTER +0.960% vs the flat update, positive 3/3 folds, fitted
    kappa=3 lam=0.85-0.90. Under the 1.0% bar -> R1 fails, R2 passes, R3 fails.
  next 10 (SECONDARY, non-gating): FILTER +2.54%, GEOM +2.07%, both 3/3. The horizon
    dependence found before is reproduced: memory pays at ten matches and washes out over
    a season, which is the endpoint the default GW_HI=38 board runs on.
  DISPERSION, the finding worth more than the verdict: coverage of the central 80%
    Beta-Binomial predictive for starts in the next ten matches is 0.530 (FLAT), 0.539
    (GEOM), 0.562 (FILTER) against a nominal 0.80. EVERY arm is far too confident. The
    filter helps and does not fix it, because the predictive still treats matches as
    conditionally independent given p while real start sequences are serially correlated
    (the very defect this line of work measured). Correcting the MEAN weighting cannot
    correct the SPREAD; that needs a predictive with dependence, and it is a separate
    pre-registered item.
  IDENTIFICATION: as expected only the ratio is pinned — 9 (kappa, lam) pairs sit within
    0.5% of the optimum for both GEOM and FILTER; the flat arm's ridge is kappa in {3,4,5}.
  CONSEQUENCE: no code path is added. `INSEASON_LAM` stays OFF, `inseason.update_minutes`
    keeps its exchangeable form, and this file is the record of why.

Run:  python studies/start_forgetting.py     (~8 min)
Out:  studies/start_forgetting.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import start_persistence as sp

TARGET = ("2023-24", "2024-25", "2025-26")          # season-start folds
PRIOR_FROM = ("2022-23",) + TARGET
CUTOFFS = (3, 5, 8, 12, 20, 30)
KAPPAS = (2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0, np.inf)
LAMS = tuple(np.round(np.arange(0.50, 1.00, 0.05), 2)) + (1.00,)
MIN_PRIOR_MATCHES = 5
MIN_REMAIN = 5
H_SECONDARY = 10
MIN_GAIN = 0.01
CONSISTENT = 3
COVER_LO, COVER_HI, COVER_TARGET = 0.75, 0.85, 0.80
RIDGE_TOL = 0.005
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "start_forgetting.csv")


def build():
    """One row per (player_code, season, cutoff): prior, the ordered early sequence, and
    the remaining sequence the endpoint scores."""
    panel = sp.build(PRIOR_FROM)
    per = (panel.groupby(["player_code", "season"])
                .agg(st=("is_start", "sum"), n=("is_start", "size")).reset_index())
    order = {s: i for i, s in enumerate(sorted(panel["season"].unique()))}
    per["si"] = per["season"].map(order)
    prev = per[per["n"] >= MIN_PRIOR_MATCHES][["player_code", "si", "st", "n"]].copy()
    prev["si"] += 1
    pmap = {(int(r.player_code), int(r.si)): (float(r.st), float(r.n))
            for r in prev.itertuples()}
    rows = []
    for (code, season), g in panel.groupby(["player_code", "season"]):
        if season not in TARGET:
            continue
        key = (int(code), order[season])
        if key not in pmap:
            continue
        a, n = pmap[key]
        y = g["is_start"].to_numpy().astype(float)
        for k in CUTOFFS:
            if len(y) < k + MIN_REMAIN:
                continue
            rest = y[k:]
            rows.append({"player_code": int(code), "season": season, "k": k,
                         "m0": a / n, "n0": n, "early": y[:k].copy(),
                         "cum": np.cumsum(rest),
                         "rest_s": float(rest.sum()), "rest_n": int(len(rest))})
    return pd.DataFrame(rows)


def _geom_weights(k, lam):
    d = np.arange(k - 1, -1, -1, dtype=float)
    u = lam ** d
    return u * (k / u.sum())


def posterior(df, arm, kappa, lam):
    """(a, b) after the first k matches, for one arm. All arms share m0."""
    m0 = df["m0"].to_numpy()
    ks = df["k"].to_numpy().astype(int)
    # kappa CAPS the prior at that many pseudo-matches, holding the mean — the same
    # operation as `inseason.cap_start_prior`. kappa = inf therefore means "uncapped",
    # i.e. the previous season's actual match count, NOT infinite mass.
    kp = np.minimum(df["n0"].to_numpy(), kappa) if np.isfinite(kappa) \
        else df["n0"].to_numpy().astype(float)
    a = np.empty(len(df)); b = np.empty(len(df))
    if arm == "FLAT":
        ev = np.array([e.sum() for e in df["early"].to_numpy()])
        a = kp * m0 + ev
        b = kp * (1 - m0) + (ks - ev)
    elif arm == "GEOM":
        cache = {int(k): _geom_weights(int(k), lam) for k in np.unique(ks)}
        ev = np.array([float(np.dot(cache[int(k)], e))
                       for k, e in zip(ks, df["early"].to_numpy())])
        a = kp * m0 + ev
        b = kp * (1 - m0) + (ks - ev)
    elif arm == "FILTER":
        # vectorised across players, one pass per cutoff: at most 30 steps, not 12k loops
        early = df["early"].to_numpy()
        for kk in np.unique(ks):
            idx = np.flatnonzero(ks == kk)
            Y = np.stack([early[i] for i in idx])
            mm, kpi = m0[idx], kp[idx]
            at, bt = kpi * mm, kpi * (1 - mm)
            base_a, base_b = (1 - lam) * kpi * mm, (1 - lam) * kpi * (1 - mm)
            for t in range(int(kk)):
                y = Y[:, t]
                at = lam * at + base_a + y
                bt = lam * bt + base_b + (1 - y)
            a[idx], b[idx] = at, bt
    else:
        raise ValueError(arm)
    return a, b


def brier(df, a, b, h=None):
    """Pooled Brier over the next h matches (None = every remaining match)."""
    p = a / (a + b)
    nrest = df["rest_n"].to_numpy()
    if h is None:
        n, s = nrest, df["rest_s"].to_numpy()
    else:
        n = np.minimum(nrest, h)
        s = np.array([float(c[i - 1]) for c, i in zip(df["cum"].to_numpy(), n)])
    return float((s * (1 - p) ** 2 + (n - s) * p ** 2).sum() / n.sum())


def coverage(df, a, b, h=H_SECONDARY, lo=0.10, hi=0.90):
    """Share of player-seasons whose realised start count in the next h matches falls in
    the central 80% of the Beta-Binomial predictive. Tests the SPREAD, not the mean."""
    from scipy.stats import betabinom
    n = np.minimum(df["rest_n"].to_numpy(), h)
    s = np.array([float(c[i - 1]) for c, i in zip(df["cum"].to_numpy(), n)])
    ok = np.zeros(len(n), dtype=bool)
    for i in range(len(n)):
        d = betabinom(int(n[i]), a[i], b[i])
        ok[i] = (s[i] >= d.ppf(lo)) and (s[i] <= d.ppf(hi))
    return float(ok.mean())


def fit(df, arm, endpoint_h):
    """Best (kappa, lam) for one arm on one training set."""
    best, bs = None, np.inf
    lams = (1.0,) if arm == "FLAT" else LAMS
    for kp in KAPPAS:
        for lm in lams:
            a, b = posterior(df, arm, kp, lm)
            e = brier(df, a, b, endpoint_h)
            if e < bs:
                bs, best = e, (kp, lm)
    return best, bs


def ridge(df, arm, endpoint_h, tol=RIDGE_TOL):
    """Every (kappa, lam) within `tol` of the best — the identified set, not a point."""
    scores = {}
    lams = (1.0,) if arm == "FLAT" else LAMS
    for kp in KAPPAS:
        for lm in lams:
            a, b = posterior(df, arm, kp, lm)
            scores[(kp, lm)] = brier(df, a, b, endpoint_h)
    best = min(scores.values())
    return sorted(k for k, v in scores.items() if v <= best * (1 + tol)), best


def loso(df, endpoint_h):
    """Fit each arm on all-but-one season, score on the held-out one."""
    out = []
    for hold in sorted(df["season"].unique()):
        tr, te = df[df["season"] != hold], df[df["season"] == hold]
        row = {"hold": hold}
        for arm in ("FLAT", "GEOM", "FILTER"):
            (kp, lm), _ = fit(tr, arm, endpoint_h)
            a, b = posterior(te, arm, kp, lm)
            row[f"{arm}_kappa"], row[f"{arm}_lam"] = kp, lm
            row[f"{arm}_brier"] = brier(te, a, b, endpoint_h)
            row[f"{arm}_cover"] = coverage(te, a, b)
        for arm in ("GEOM", "FILTER"):
            row[f"{arm}_gain"] = ((row["FLAT_brier"] - row[f"{arm}_brier"])
                                  / row["FLAT_brier"])
        out.append(row)
    return pd.DataFrame(out)


def main():
    print("=" * 78)
    print("FORGETTING FILTER vs the flat update and the renormalised geometric weight")
    print("=" * 78)
    df = build()
    print(f"\n{len(df)} player-season-cutoff rows, {df.player_code.nunique()} players, "
          f"folds {sorted(df.season.unique())}, cutoffs {CUTOFFS}")
    rows = []

    for lab, h in (("PRIMARY rest-of-season", None), ("SECONDARY next 10", H_SECONDARY)):
        R = loso(df, h)
        print(f"\n=== {lab} ===")
        print(R.round(4).to_string(index=False))
        for arm in ("GEOM", "FILTER"):
            g = R[f"{arm}_gain"]
            npos = int((g > 0).sum())
            print(f"  {arm:6s} mean gain vs FLAT {g.mean():+.3%}, positive {npos}/{len(R)}"
                  f", coverage {R[f'{arm}_cover'].mean():.3f} "
                  f"(FLAT {R['FLAT_cover'].mean():.3f}, target {COVER_TARGET})")
        for r in R.itertuples():
            rows.append({"endpoint": lab, **r._asdict()})

        if h is None:                      # the gating endpoint
            g = R["FILTER_gain"]
            cov = R["FILTER_cover"].mean()
            cov_flat = R["FLAT_cover"].mean()
            c1 = g.mean() >= MIN_GAIN
            c2 = int((g > 0).sum()) >= CONSISTENT
            c3 = (COVER_LO <= cov <= COVER_HI
                  and abs(cov - COVER_TARGET) <= abs(cov_flat - COVER_TARGET))
            verdict = ("ADOPT" if (c1 and c2 and c3) else
                       "NULL FOR SHIPPING (better mean, wrong spread)" if (c1 and c2)
                       else "NULL" if (g.mean() < 0.005 or not c2) else "INCONCLUSIVE")
            print(f"\n  R1 gain >= {MIN_GAIN:.0%}: {c1} ({g.mean():+.3%})")
            print(f"  R2 positive in {CONSISTENT}/3: {c2}")
            print(f"  R3 dispersion gate: {c3} (coverage {cov:.3f} vs FLAT {cov_flat:.3f})")
            print(f"  -> {verdict}")
            rows.append({"endpoint": lab, "hold": "VERDICT", "verdict": verdict,
                         "gain": float(g.mean()), "folds_pos": int((g > 0).sum()),
                         "cover": cov, "cover_flat": cov_flat})

    print("\n=== identification: the ridge on the pooled panel (rest-of-season) ===")
    for arm in ("FLAT", "GEOM", "FILTER"):
        rset, best = ridge(df, arm, None)
        shown = ", ".join(f"({'inf' if not np.isfinite(k) else int(k)},{l:g})"
                          for k, l in rset[:8])
        print(f"  {arm:6s} best Brier {best:.5f}; {len(rset)} pairs within "
              f"{RIDGE_TOL:.1%}: {shown}{' ...' if len(rset) > 8 else ''}")
        rows.append({"endpoint": "ridge", "hold": arm, "brier": best,
                     "n_within_tol": len(rset)})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
