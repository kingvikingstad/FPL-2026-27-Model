import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
press_k_multiseason.py — set `press_measured.K_DEFAULT` on the feed actually used.
==================================================================================
K_DEFAULT = 40 was a compromise between two estimates that disagreed:

  (a) Understat, 12 seasons, 220 team-seasons, held-out truth: optimum k = 11.5,
      MSE -27.5% against prior-only, flat basin 9..15.
  (b) the same backtest run OPERATIONALLY on the feed the model could compute:
      optimum k = 44-50, and k = 12 lost.

The gap had two candidate explanations that one season could not separate: the feed's
definition mismatch against Understat, and the fact that the only testable transition
(2024/25 -> 2025/26) was the most persistent of the eleven, r = 0.874, which flatters any
prior. `press_feed_compare.py` re-ran (b) with PitchAPI and it FAILED the same way -
optimum k ~ 120 - which is evidence for the second explanation, not the first, because
PitchAPI's definition mismatch is much smaller (r = 0.9677 vs 0.9370) and the result got
WORSE rather than better. A cross-source, single-season test cannot answer this.

WHAT IS DIFFERENT HERE
----------------------
PitchAPI supplies 2021/22..2025/26 of PPDA under ONE definition, so prior, measurement
and truth are the same quantity. That removes the source mismatch entirely and gives four
season transitions instead of one. It is the analogue of (a), run on the feed the board
will actually use.

PRE-REGISTERED — rules fixed before any number was seen
-------------------------------------------------------
  ESTIMATOR   For club c in season s: prior = c's mean log PPDA in s-1 (league mean of
              s-1 if c was not in the division); measured = mean log PPDA over the first
              n matches of s; truth = mean log PPDA over the REMAINING matches of s, so
              the two estimators cannot share noise with what they are scored against.
              Optimum k minimises pooled MSE.

  ROBUSTNESS  Leave-one-season-out: refit k with each transition held out. Adopt a new
              K_DEFAULT only if the optimum is inside [0.6x, 1.6x] of the pooled value in
              EVERY fold. A parameter that moves when one season is removed has been
              fitted to that season.

  ADOPT       the pooled optimum, rounded to the nearest integer, only if its gain over
              prior-only is positive in every fold as well. Otherwise KEEP 40 and record
              why. Weak-prior clubs (promoted, no previous season) get their own k from
              the same fit, since their prior is a league mean rather than a measurement.

  This study does NOT get to relax the standard that produced 40 just because a better
  feed arrived. It replaces a compromise between two mismatched tests with one properly
  powered test on the right feed - or it fails, and 40 stands.

OUTCOME: NULL — THE TEST CANNOT BE RUN
--------------------------------------
[VERIFIED 2026-08-31] PitchAPI's `/advanced` routes, which carry PPDA, exist for 2025/26
onward and essentially nowhere earlier: 0/380 matches rated in 2021/22, 0/380 in 2022/23,
23/380 in 2023/24, 0/380 in 2024/25, against 380/380 in 2025/26. The `/v1/leagues` season
list advertises 2021/2022..2026/2027 because FIXTURES go back that far; the analytics do
not, and nothing in the season list says so. Confirmed against the live API with the cache
bypassed, so it is the service and not a cached failure.

One season of PPDA cannot yield a season-transition backtest, so the pre-registered
estimator has no data and K_DEFAULT stays 40. That is now a stronger statement than the
compromise it started as: not "two tests disagreed and 40 splits them", but "the only
well-powered test available is the 12-season Understat one (k = 11.5), the only
operational test is a single favourable transition (k = 120), and no feed in existence
can currently break the tie."

The file is kept rather than deleted because the pre-registration and the estimator are
correct and become runnable the moment a second season of `/advanced` exists — around the
end of 26/27. It exits 0 with this notice rather than raising, so the harness stays green.
"""
import numpy as np
import pandas as pd

TEMP = _os.environ.get("TEMP", "/tmp")
HIST = _os.path.join(TEMP, "pa_hist.csv")
CUR = _os.path.join(TEMP, "pa_2526.csv")
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "press_k_multiseason.csv")
KS = np.arange(0.5, 200, 0.25)


MIN_SEASONS = 2          # a transition needs a season and its predecessor


def panel():
    fr = [pd.read_csv(p) for p in (HIST, CUR) if _os.path.exists(p)]
    if not fr:
        print("no PitchAPI pull on disk; nothing to fit. See the docstring.")
        raise SystemExit(0)
    d = pd.concat(fr, ignore_index=True)
    d = d[d["defending.ppda"] > 0].copy()
    d["l"] = np.log(d["defending.ppda"])
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["season", "team", "date"])


def build(d):
    seasons = sorted(d.season.unique())
    prev = {}
    for a, b in zip(seasons, seasons[1:]):
        g = d[d.season == a].groupby("team")["l"].mean()
        prev[b] = (g.to_dict(), float(g.mean()))
    rows = []
    for s in seasons[1:]:
        pmap, pleague = prev[s]
        for t, g in d[d.season == s].groupby("team"):
            L = g["l"].values
            if len(L) < 20:
                continue
            pri = pmap.get(t, pleague)
            promoted = t not in pmap
            for n in range(1, 26):
                if len(L) - n < 8:
                    break
                rows.append((s, t, promoted, n, pri, L[:n].mean(), L[n:].mean()))
    return pd.DataFrame(rows, columns=["season", "team", "promoted", "n",
                                       "prior", "meas", "truth"])


def fit(b):
    """(best k, MSE at it, prior-only MSE, gain)."""
    if not len(b):
        return np.nan, np.nan, np.nan, np.nan
    base = float(((b.prior - b.truth) ** 2).mean())
    mse = {}
    for k in KS:
        w = b.n / (b.n + k)
        mse[k] = float(((w * b.meas + (1 - w) * b.prior - b.truth) ** 2).mean())
    kb = min(mse, key=mse.get)
    return kb, mse[kb], base, 1 - mse[kb] / base


def main():
    d = panel()
    print("=" * 78)
    print("K FOR THE PRESS BLEND, FITTED ON THE PITCHAPI FEED ITSELF")
    print("=" * 78)
    seasons = sorted(d.season.unique())
    print(f"  {len(d)} team-matches, seasons {seasons}")
    full = [s for s in seasons if (d.season == s).sum() >= 600]      # ~760 in a full season
    if len(full) < MIN_SEASONS:
        print("")
        print(f"  usable (near-complete) seasons: {full or 'none'}")
        print("  NULL — a season-transition backtest needs at least two, and PitchAPI's")
        print("  /advanced coverage begins at 2025/26. K_DEFAULT stays 40; see the")
        print("  docstring for the verified per-season coverage counts.")
        print("=" * 78)
        raise SystemExit(0)
    b = build(d)
    print(f"  backtest rows {len(b)}, team-seasons {b.groupby(['season','team']).ngroups}, "
          f"promoted {b[b.promoted].groupby(['season','team']).ngroups}")

    est = b[~b.promoted]
    kb, m, base, gain = fit(est)
    print(f"\nESTABLISHED CLUBS (prior = last season's own measurement)")
    print(f"  pooled optimum k = {kb:.2f}   MSE {m:.4f}   prior-only {base:.4f}   "
          f"gain {gain:+.1%}")
    for k in (8, 12, 20, 30, 40, 60):
        kk = min(KS, key=lambda x: abs(x - k))
        w = est.n / (est.n + kk)
        e = float(((w * est.meas + (1 - w) * est.prior - est.truth) ** 2).mean())
        print(f"     k={k:4.0f}  gain {1 - e / base:+.1%}")

    print(f"\n  LEAVE-ONE-SEASON-OUT")
    folds = []
    for s in sorted(est.season.unique()):
        k2, _, _, g2 = fit(est[est.season != s])
        folds.append((s, k2, g2))
        print(f"     drop {s}:  k = {k2:6.2f}   gain {g2:+.1%}")
    lo, hi = 0.6 * kb, 1.6 * kb
    stable = all(lo <= k2 <= hi for _, k2, _ in folds)
    positive = all(g2 > 0 for _, _, g2 in folds) and gain > 0
    print(f"     stability band [{lo:.1f}, {hi:.1f}] -> "
          f"{'STABLE' if stable else 'UNSTABLE'};  gains all positive: {positive}")

    prom = b[b.promoted]
    kp, _, basep, gainp = fit(prom)
    print(f"\nWEAK-PRIOR CLUBS (promoted; prior = league mean)")
    print(f"  n team-seasons {prom.groupby(['season','team']).ngroups}   "
          f"optimum k = {kp:.2f}   gain {gainp:+.1%}")

    print("\n" + "=" * 78)
    if stable and positive:
        print(f"VERDICT: ADOPT K_DEFAULT = {round(kb)}  (was 40)")
        print(f"         and K_WEAK_PRIOR = {round(kp)}  (was 20)")
        w10 = 10 / (10 + kb)
        print(f"         weight on measured press at GW10: {w10:.2f} (was 0.20)")
    else:
        print("VERDICT: KEEP K_DEFAULT = 40. The optimum is not stable across")
        print("         leave-one-season-out folds, so it is fitted to a season rather")
        print("         than to the game.")
    print("=" * 78)
    pd.DataFrame(folds, columns=["held_out_season", "k", "gain"]).to_csv(OUT, index=False)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
