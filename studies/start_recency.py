from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
start_recency.py — should the minutes update be ORDER-AWARE?
=============================================================
`inseason.update_minutes` is a conjugate Beta update: `start_a += w*starts`. A Beta
update is EXCHANGEABLE — it reads a count, so start-start-bench and bench-start-start
produce an identical posterior. `studies/start_persistence.py` measured that this is the
wrong likelihood:

    excess over a permutation null holding each player-season's start count fixed,
    P(start at t+1 | streak k)  [corrected 2026-09-16 — 22/23 GW1-15 `starts` were a
    literal 0 and had inflated each value]:
      k=1 +0.164   k=3 +0.122   k=5 +0.076   k=6 +0.060   k=8 +0.015   k=10 +0.002

Ordering carries information over roughly 6-8 matches. This study tests one encoding of
it — discounting realised matches by RECENCY with a geometric weight u_d = lam^d on the
match d back from the cutoff — as a weight on the Beta update's own observations, not a
new predictor. (A `streak_k` covariate is not built, by `inseason`'s rule against streak
terms. This docstring first called it "WRONG, double counting"; that does not follow,
because the persistence null conditions on the full-season count a forecaster cannot see.
It is untested, not refuted. The first version also said recency weights "handle the
benching asymmetry automatically"; they cannot, since the weights do not depend on whether
a match was a start, and that asymmetry has no null.)

IDENTIFICATION — why the weights are renormalised, and what that does NOT buy
-------------------------------------------------------------------------------
Weights are rescaled to sum to k, so the NOMINAL evidence count is w*k for every lam and
lam = 1 recovers today's flat update exactly. Without it, lam < 1 would visibly shrink the
in-season weight. As first written, this section claimed that made the test one of ORDER
alone, separate from the W_MINUTES/START_KAPPA ridge. [WRONG — corrected 2026-09-16,
stats-referee.] A fixed sum is not fixed information: the effective sample size of
geometric weights caps at (1+lam)/(1-lam), 7 at lam=0.75, while the Beta is charged k. So
lam trades off against kappa — the results show it, lam* at h=10 being 0.25 uncapped and
0.75 at kappa=4 — and the implied posterior is over-concentrated. Read every lam* below as
conditional on its kappa.

PRE-REGISTERED — WRITTEN BEFORE THE RESULTS WERE SEEN
------------------------------------------------------
Unit: player-season x cutoff k. Prior (a, b) = previous season's (starts, non-starts),
optionally capped at kappa holding the mean. Posterior

    p_hat = (a' + w * sum_i u_i y_i) / (a' + b' + w * k),   u_i = lam^d_i renormalised to k

PRIMARY ENDPOINT   Brier on the NEXT match's start indicator. The mechanism measured is
                   a one-step transition, so the one-step forecast is where it must show
                   up. Pooled over player-seasons.
SECONDARY          Brier on every remaining match (the horizon the board actually
                   projects). A recency effect should be WEAKER here and that is expected,
                   not a failure — it is reported to size the horizon decay, not to gate.
DECISION RULE      Adopt lam < 1 only if, leave-one-season-out on the PRIMARY endpoint:
                     (1) pooled Brier gain over lam = 1 is at least MIN_GAIN (1%); and
                     (2) the gain is positive in at least CONSISTENT (3) held-out seasons.
                   Otherwise NULL: the constant is not touched, `update_minutes` keeps its
                   exchangeable form, and this file records why.

TWO THINGS WERE ADDED AFTER THE FIRST RUN — both flagged as post-hoc
--------------------------------------------------------------------
1. LAM_GRID was extended DOWNWARD. The first grid stopped at 0.50 and every LOSO fold
   returned exactly 0.50, so the search was pinned to a boundary and no fitted value was
   a fit. Extending an unbracketed grid is a correction, not a second look: the decision
   rule and the endpoint are unchanged.
2. A HORIZON SWEEP. The first run showed the two endpoints disagreeing hard — big gains
   one match out, roughly nothing over the rest of the season, and at kappa=4 the rest
   endpoint got WORSE at lam=0.50. That is not noise, it is the substantive result: the
   optimal memory length is a function of FORECAST HORIZON. `lam_by_horizon` measures it
   directly, scoring the pooled Brier over the next h matches for h = 1, 2, 3, 5, 10 and
   rest-of-season. It is exploratory and gates nothing; it exists because the board
   projects ten gameweeks, so a lam fitted at h=1 would be tuned on the wrong endpoint.

Both prior strengths are swept (kappa = inf, today's production default; kappa = 4, the
`start_prior_strength` corner that INSEASON_KAPPA turns on) because a recency effect can
only show up if the prior is light enough for realised matches to move anything.

SCOPE   Target seasons use the native `starts` column. Prior seasons may be pre-2022/23,
        where the start definition is the mins>=60 proxy — that only sets a base rate. On
        the same rows the proxy ranks players like native starts but sits consistently LOW:
        -0.017 to -0.020 absolute (about -6.5%), P(proxy|native) = 0.93, P(native|proxy)
        ~ 0.99 in every native season [VERIFIED 2026-09-17]. Cold-start players
        have no prior season and never enter; their prior comes from `starter_prior`.
        22/23 is observed from GW16 (its GW1-15 `starts` are structural zeros, dropped by
        `start_persistence.build`), so its fold starts mid-season against a 21/22 prior —
        a different setting from the three season-start folds.

RESULT, 2026-09-16 (corrected data) — the full record is docs/START_PERSISTENCE_2026-09-07.md
  Pre-registered gate: ADOPT, +10.9% at kappa=4 and +5.5% uncapped, 4/4 folds each.
  Horizon sweep (exploratory, kappa=4): lam* 0.35/0.45/0.55/0.60/0.75/0.82 at
  h = 1/2/3/5/10/rest, gains +10.9/+7.8/+5.5/+3.4/+1.6/+0.5%. lam=0.75 is DERIVED only
  under (h=10, kappa=4, W=1, previous-season prior, k<=12, match grain) with the mid-season
  22/23 fold included; the three season-start folds alone give lam*=0.70 at h=10 (+1.5%, 3/3),
  gate +10.3% (3/3), rest-of-season 2/3 folds [VERIFIED 2026-09-17]. At fixed lam=0.75
  the latest season gains +0.81% at h=10 and -0.42% rest-of-season. The board now defaults
  to GW_HI=38, so point 2 above ("the board projects ten gameweeks") no longer holds.

Run:  python studies/start_recency.py
Out:  studies/start_recency.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import start_persistence as sp

# --- pre-registered constants -------------------------------------------------
TARGET = ("2022-23", "2023-24", "2024-25", "2025-26")     # native `starts`
PRIOR_FROM = ("2021-22",) + TARGET                        # supply priors for TARGET
# 0.05..0.45 added after the first run: every LOSO fold returned 0.50, the LOWEST value
# then tested, so the search was pinned to a boundary and nothing was actually fitted.
LAM_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
CUTOFFS = (3, 5, 8, 12)
KAPPAS = (np.inf, 4.0)
HORIZONS = (1, 2, 3, 5, 10, None)          # None = rest of season (post-hoc sweep)
W = 1.0                    # inseason.W_MINUTES; held fixed, already fitted elsewhere
MIN_PRIOR_MATCHES = 5
MIN_REMAIN = 5
MIN_GAIN = 0.01
CONSISTENT = 3
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "start_recency.csv")


def build():
    """One row per (player_code, season, cutoff): prior, ordered early sequence, target."""
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
                         "prior_a": a, "prior_b": n - a,
                         "early": y[:k].copy(),
                         "next_y": float(y[k]),
                         "cum": np.cumsum(rest),          # starts within the next h
                         "rest_s": float(rest.sum()), "rest_n": int(len(rest))})
    return pd.DataFrame(rows)


def _weights(k, lam):
    """u_d = lam^d for d = k-1 ... 0 (0 = most recent), renormalised to sum to k."""
    d = np.arange(k - 1, -1, -1, dtype=float)
    u = lam ** d
    return u * (k / u.sum())


def phat(df, lam, kappa, w=W):
    a = df["prior_a"].to_numpy(); b = df["prior_b"].to_numpy()
    tot = a + b
    scale = np.where(tot > kappa, kappa / np.maximum(tot, 1e-9), 1.0) \
        if np.isfinite(kappa) else np.ones_like(tot)
    a2, b2 = a * scale, b * scale
    ks = df["k"].to_numpy()
    wcache = {int(k): _weights(int(k), lam) for k in np.unique(ks)}
    ev = np.array([float(np.dot(wcache[int(k)], e))
                   for k, e in zip(ks, df["early"].to_numpy())])
    return (a2 + w * ev) / (a2 + b2 + w * ks)


def brier_next(df, lam, kappa):
    p = phat(df, lam, kappa)
    return float(np.mean((p - df["next_y"].to_numpy()) ** 2))


def brier_rest(df, lam, kappa):
    p = phat(df, lam, kappa)
    s = df["rest_s"].to_numpy(); n = df["rest_n"].to_numpy()
    return float((s * (1 - p) ** 2 + (n - s) * p ** 2).sum() / n.sum())


def brier_horizon(df, lam, kappa, h=None):
    """Pooled Brier over the next `h` matches (h=None -> rest of season).

    This is the object the board is actually scored on: a start probability held fixed
    across a projection window, charged against every match inside it. h=1 reduces to
    `brier_next`; h=None reduces to `brier_rest`.
    """
    if h is None:
        return brier_rest(df, lam, kappa)
    p = phat(df, lam, kappa)
    nrest = df["rest_n"].to_numpy()
    nh = np.minimum(nrest, h)
    s = np.array([float(c[i - 1]) for c, i in zip(df["cum"].to_numpy(), nh)])
    return float((s * (1 - p) ** 2 + (nh - s) * p ** 2).sum() / nh.sum())


def loso(df, score, kappa):
    """Fit lam on all-but-one season, score on the held-out one."""
    out = []
    for hold in sorted(df["season"].unique()):
        tr = df[df["season"] != hold]
        te = df[df["season"] == hold]
        if not len(tr) or not len(te):
            continue
        lam_hat = min(LAM_GRID, key=lambda L: score(tr, L, kappa))
        b_base = score(te, 1.0, kappa)
        b_new = score(te, lam_hat, kappa)
        out.append(dict(hold=hold, lam=lam_hat, base=b_base, new=b_new,
                        gain=(b_base - b_new) / b_base))
    return out


def main():
    print("\n=== building panel ===")
    df = build()
    print(f"{len(df)} player-season-cutoff rows, "
          f"{df.player_code.nunique()} players, seasons {sorted(df.season.unique())}")

    rows = []
    for kappa in KAPPAS:
        kl = "inf" if not np.isfinite(kappa) else f"{kappa:g}"
        for ename, score in (("next", brier_next), ("rest", brier_rest)):
            print(f"\n=== kappa={kl}  endpoint={ename} ===")
            print(f"  {'lam':>5}  {'brier':>9}  {'gain vs lam=1':>14}  half-life(matches)")
            b1 = score(df, 1.0, kappa)
            for lam in LAM_GRID:
                b = score(df, lam, kappa)
                hl = np.log(0.5) / np.log(lam) if lam < 1 else np.inf
                print(f"  {lam:>5.2f}  {b:>9.5f}  {(b1 - b) / b1:>+13.3%}  "
                      f"{hl:>6.1f}" if np.isfinite(hl) else
                      f"  {lam:>5.2f}  {b:>9.5f}  {(b1 - b) / b1:>+13.3%}     flat")
                rows.append(dict(scope="grid", kappa=kl, endpoint=ename, lam=lam,
                                 hold="ALL", brier=b, base=b1, gain=(b1 - b) / b1,
                                 n=len(df)))

            folds = loso(df, score, kappa)
            npos = sum(1 for f in folds if f["gain"] > 0)
            med_lam = float(np.median([f["lam"] for f in folds]))
            pooled = float(np.mean([f["gain"] for f in folds]))
            print(f"  LOSO: lam_hat by fold "
                  f"{ {f['hold']: f['lam'] for f in folds} }")
            for f in folds:
                print(f"    hold {f['hold']}  lam={f['lam']:.2f}  "
                      f"base={f['base']:.5f} new={f['new']:.5f} gain={f['gain']:+.3%}")
            verdict = ("ADOPT" if (ename == "next" and pooled >= MIN_GAIN
                                   and npos >= CONSISTENT) else "null")
            print(f"  mean LOSO gain {pooled:+.3%}, positive in {npos}/{len(folds)} "
                  f"folds, median lam {med_lam:.2f}  ->  {verdict}")
            for f in folds:
                rows.append(dict(scope="loso", kappa=kl, endpoint=ename, lam=f["lam"],
                                 hold=f["hold"], brier=f["new"], base=f["base"],
                                 gain=f["gain"], n=int((df.season == f["hold"]).sum())))
            rows.append(dict(scope="verdict", kappa=kl, endpoint=ename, lam=med_lam,
                             hold=verdict, brier=np.nan, base=np.nan, gain=pooled,
                             n=npos))

    # --- POST-HOC: how long should the memory be as a function of HORIZON? -----
    # The board holds one start probability across a projection window and is charged
    # against every match inside it, so the horizon is not a detail — it selects lam.
    print("\n=== POST-HOC: lam* by forecast horizon (LOSO, gates nothing) ===")
    for kappa in KAPPAS:
        kl = "inf" if not np.isfinite(kappa) else f"{kappa:g}"
        print(f"\n  --- kappa={kl} ---")
        print(f"  {'horizon':>8}  {'lam*':>5}  {'half-life':>9}  {'LOSO gain':>10}  "
              f"{'folds+':>6}")
        for h in HORIZONS:
            score = (lambda d, L, K, _h=h: brier_horizon(d, L, K, _h))
            folds = loso(df, score, kappa)
            if not folds:
                continue
            med = float(np.median([f["lam"] for f in folds]))
            gain = float(np.mean([f["gain"] for f in folds]))
            npos = sum(1 for f in folds if f["gain"] > 0)
            hl = np.log(0.5) / np.log(med) if med < 1 else np.inf
            label = "rest" if h is None else str(h)
            print(f"  {label:>8}  {med:>5.2f}  "
                  f"{(f'{hl:.1f}' if np.isfinite(hl) else 'flat'):>9}  "
                  f"{gain:>+10.3%}  {npos}/{len(folds)}")
            rows.append(dict(scope="horizon", kappa=kl, endpoint=f"h{label}", lam=med,
                             hold="LOSO", brier=np.nan, base=np.nan, gain=gain, n=npos))

    # per-cutoff detail at the production prior: where does order actually pay?
    print("\n=== gain by cutoff (kappa=4, endpoint=next) ===")
    for k in CUTOFFS:
        d = df[df["k"] == k]
        if not len(d):
            continue
        b1 = brier_next(d, 1.0, 4.0)
        best = min(LAM_GRID, key=lambda L: brier_next(d, L, 4.0))
        bb = brier_next(d, best, 4.0)
        print(f"  k={k:>2}  n={len(d):>5}  lam*={best:.2f}  "
              f"brier {b1:.5f} -> {bb:.5f}  ({(b1 - bb) / b1:+.2%})")
        rows.append(dict(scope="by_cutoff", kappa="4", endpoint="next", lam=best,
                         hold=f"k{k}", brier=bb, base=b1, gain=(b1 - bb) / b1, n=len(d)))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
