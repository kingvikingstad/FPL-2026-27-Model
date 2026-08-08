import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
matchup_design.py — test designs for three constructs, with the confound that
kills the naive version of each.  All findings below are from simulation
(`--selftest`), not from fitted PL data.

  C1  intended style     — what a side sets up to do before information arrives
  C2  skill to dictate   — whether that intent survives contact with the opponent
  C3  state proclivity   — goal-rate response to being ahead / behind

THE STRUCTURE THAT LINKS THEM
-----------------------------
C2 turns C1 into a first stage.  Intent is the assignment, realised style is
the treatment, and relative skill governs compliance:

    M_ij = w_ij * S_i + (1 - w_ij) * S_j,    w_ij = logit^-1( kappa*(q_i - q_j) )

If kappa is large, then in a mismatch the realised style of the match IS the
stronger side's style — which is already collinear with strength.  So intent
has independent explanatory power ONLY in evenly-matched fixtures.  This is
the same conclusion the GW1-6 leverage analysis reached from the other
direction: the high-lambda fixtures are big-six-vs-promoted mismatches, which
is exactly where style carries the least independent information.

---------------------------------------------------------------------------
C1 — MEASURING INTENT
---------------------------------------------------------------------------
Measure style on 0-0 minutes only.  Budget (simulated):
  * 45% of all league minutes are played at 0-0
  * ~1,548 such minutes per team per season = 17 matches' worth
The 0-0 window is SHORTER for teams in high-scoring matches, so raw 0-0
totals are exposure-biased toward defensive sides.  Always rate-per-minute,
always weight by exposure, never compare raw counts.

---------------------------------------------------------------------------
C2 — SKILL TO DICTATE  (the best-powered of the three)
---------------------------------------------------------------------------
Reduced form first, no structure required:

    (M_ij - S_i) = a + b * [ (S_j - S_i) * (q_j - q_i) ] + c*(S_j - S_i)
                                                         + d*(q_j - q_i) + e

b > 0 means the better side drags the match toward its own intent.  Simulated
t-statistics on ONE season (380 matches): kappa=0.5 -> t=7.3; kappa=1.0 ->
t=13.6; kappa=2.0 -> t=20.7; and correctly t=-0.6 under the null.  Structural
kappa recovers to about +/-0.07.  This is the construct to build first.

q_i: use att+dfn from TeamModel, or market-implied strength.
S_i: the C1 intent scores.
M_ij: realised match style on the same axes, whole match.

---------------------------------------------------------------------------
C3 — STATE PROCLIVITY  (the trap)
---------------------------------------------------------------------------
Two separate biases, in opposite directions.

(a) SELECTION.  State is not randomly assigned.  High-lambda matches leave
    0-0 fast and spend most minutes non-level, so non-level minutes are drawn
    from the high-lambda tail.  Simulated with a TRUE effect of exactly zero:

        naive (state dummy only)          +0.134   = +14.3% goal rate
        + team strength + time            +0.076   =  +7.9%
        + MARKET-implied match total      +0.031   =  +3.1%   <-- clean
        + true match lambda               +0.026   =  +2.6%

    Controlling for the market total recovers a true +0.150 to within 0.002.
    "Goals beget goals" is mostly this artifact.  You already ingest the odds;
    the total-goals line is the control variable.

(b) DEPLETION.  Match fixed effects are the WRONG fix.  State is a function
    of cumulative goals within the match, so conditioning on the match total
    mechanically depletes the later state — Nickell bias.  Simulated: match FE
    returns -0.067 against a true +0.150.  Naive is biased up, FE biased
    down; they bracket the truth but neither is usable.

(c) DO NOT AGGREGATE INTO SPELLS.  A level spell ends when a goal arrives, so
    its goals-per-minute is mechanically 1/(waiting time), and E[1/T] >
    1/E[T].  Spell aggregation flips the sign of the estimate.  Use one row
    per minute (or per 5-min block) with the state fixed at the START of the
    interval.

(d) PER-TEAM PROCLIVITY IS NOT IDENTIFIABLE.  20 free team betas:

        seasons   SE(own beta)   MDE at 80% power
           1          0.276           117%
           2          0.192            71%
           3          0.156            55%

    No real state response is 55% of a goal rate.  "Which teams open up after
    conceding" cannot be answered per team from PL data.

(e) THE VIABLE VERSION mediates the state response through intent —
    beta_i = beta0 + beta1 * S_i — which is 2 parameters, not 20, and is
    exactly the hypothesis anyway:

        seasons   MDE(beta0)   MDE(beta1)
           1          19%          22%
           2          13%          14%
           3          10%          12%

    One season is marginal; two is workable if the true differential is large.

---------------------------------------------------------------------------
BUILD ORDER
---------------------------------------------------------------------------
1. C1 intent scores on 0-0 minutes, exposure-weighted, with split-half
   reliability reported (see style_matchup.py `style_reliability`).
2. C2 reduced form. Well powered in one season. If b is not significantly
   positive, stop — style does not survive contact and C3(e) has no regressor.
3. C3(e) minute-level hazard, market total as control, state x intent.
4. Only then ask whether any of it beats the market on the residual.

Run: python matchup_design.py --selftest
"""
from __future__ import annotations
import argparse
import numpy as np


def poisson_irls(X, y, iters=100, ridge=1e-8):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        lam = np.exp(np.clip(X @ b, -12, 12))
        H = (X * lam[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - lam))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    lam = np.exp(np.clip(X @ b, -12, 12))
    H = (X * lam[:, None]).T @ X + ridge * np.eye(X.shape[1])
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def dictation_test(M, S_own, S_opp, q_own, q_opp):
    """C2 reduced form. Returns (b, se, t) on the pull term. b>0 => the
    stronger side imposes its intended style on the match."""
    dev = M - S_own
    pull = (S_opp - S_own) * (q_opp - q_own)
    X = np.column_stack([np.ones(len(M)), pull, S_opp - S_own, q_opp - q_own])
    b, *_ = np.linalg.lstsq(X, dev, rcond=None)
    r = dev - X @ b
    s2 = r @ r / (len(M) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return b[1], se, b[1] / se


def state_response(goals, is_nonlevel, market_total, minute_frac, intent=None):
    """C3(e). One row per MINUTE; state fixed at the start of the minute.
    market_total = log market-implied total goals for the match (the control
    that kills the selection bias). intent = match-level intent score; if
    given, the state x intent interaction is estimated."""
    cols = [np.ones(len(goals)), is_nonlevel]
    names = ["const", "state"]
    if intent is not None:
        cols += [is_nonlevel * intent, intent]
        names += ["state x intent", "intent"]
    cols += [market_total, minute_frac]
    names += ["market_total", "minute"]
    b, se = poisson_irls(np.column_stack(cols), np.asarray(goals, float))
    return {n: (bi, si) for n, bi, si in zip(names, b, se)}


def selftest():
    rng = np.random.default_rng(19)
    T = 20
    S = rng.normal(0, 1, T)
    q = rng.normal(0, 1, T)
    KAP = 1.0
    i = rng.integers(0, T, 380); j = rng.integers(0, T, 380)
    k = i != j; i, j = i[k], j[k]
    w = 1 / (1 + np.exp(-KAP * (q[i] - q[j])))
    M = w * S[i] + (1 - w) * S[j] + rng.normal(0, 0.35, len(i))
    b, se, t = dictation_test(M, S[i], S[j], q[i], q[j])
    print(f"C2  pull coef {b:+.4f} (se {se:.4f}, t={t:+.1f}) at kappa={KAP}")
    assert t > 4, "C2 should be strongly powered in one season"

    b0, b0se, *_ = 0, 0, 0
    rows = []
    for m in range(760):
        a_, d_ = rng.integers(0, T), rng.integers(0, T)
        if a_ == d_:
            continue
        unobs = rng.normal(0, 0.25)
        obs = 0.5 * (q[a_] + q[d_])
        mkt = obs + np.sqrt(.75) * unobs + rng.normal(0, .05)
        base = np.log(2.88 / 90) + obs + unobs
        sty = (S[a_] + S[d_]) / 2
        gd = 0
        for tm in range(90):
            st = 0.0 if gd == 0 else 1.0
            g = rng.poisson(np.exp(base + (0.15 + 0.12 * sty) * st + .35 * tm / 90))
            rows.append((g, st, mkt, tm / 90, sty))
            if g:
                gd += rng.choice([-1, 1]) * g
    a = np.array(rows).T
    res = state_response(a[0], a[1], a[2], a[3], intent=a[4])
    for nm in ("state", "state x intent"):
        est, s = res[nm]
        print(f"C3  {nm:<16} {est:+.4f} (se {s:.4f})")
    assert abs(res["state"][0] - 0.15) < 3 * res["state"][1], "state effect off"
    print("SELFTEST OK — C2 powered at one season; C3 recovers the true state "
          "effect once the market total is controlled for.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
    else:
        print(__doc__)
