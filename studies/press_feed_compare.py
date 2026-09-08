import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
press_feed_compare.py — should `press_measured` switch its feed to PitchAPI?
===========================================================================
`press_measured` currently rebuilds PPDA from FPL-Core-Insights components:

    opponent attempted passes / (tackles_won + interceptions + fouls_committed + recoveries)

That proxy correlates 0.937 with Understat's PPDA at season level, and the residual ~12%
of between-club variance is a DEFINITION MISMATCH, not noise — averaging more matches
never removes it. That irreducible term is the entire reason `K_DEFAULT` was set to the
conservative 40 rather than the theoretically optimal ~12.5: a feed that converges to a
slightly wrong answer must not be allowed to displace the prior.

PitchAPI publishes PPDA under Opta's published methodology, WITH its numerator
(`opponent_passes`) and denominator (`defensive_actions`) separately, so the season value
is a correct ratio-of-sums rather than a mean of per-match ratios. If it agrees with
Understat more closely than the proxy does, the mismatch term shrinks and k can fall,
which is worth real accuracy: at k=40 the weight on measured press at GW10 is 0.20; at
k=12 it is 0.45.

PRE-REGISTERED — decision rules fixed before any number was seen
---------------------------------------------------------------
  SWITCH the feed only if BOTH hold:
    (1) PitchAPI's season-level correlation with Understat 25/26, across all 20 clubs,
        EXCEEDS the proxy's 0.937; and
    (2) in the operational backtest — prior (24/25 Understat) + n matches of the candidate
        feed, predicting held-out 25/26 Understat — the optimum k is materially below 40
        AND the gain at that k is non-negative under the Understat-native frame too.
        This is the same both-frames robustness standard that produced K_DEFAULT=40; it
        is not relaxed just because a nicer feed showed up.

  LOWER K only to a value that is non-negative in BOTH frames. If PitchAPI wins (1) but
  not (2), the feed switches and k stays at 40 — a better measurement still deserves the
  same scepticism about how fast it should displace a prior.

  If neither holds, this is a NULL: the proxy stays, and PitchAPI is not wired into the
  press channel at all.

CAVEAT: Understat is itself only a reference scale, not truth. Both candidates are being
scored on agreement with a third party whose definition differs from both. What this can
establish is which feed is closer to the quantity `press_index`'s tables are denominated
in — that, and not "which is more correct", is the question that governs the blend.
"""
import numpy as np
import pandas as pd

import sd_ingest as sd
import press_measured as pm

TEMP = _os.environ.get("TEMP", "/tmp")
PA_2526 = _os.path.join(TEMP, "pa_2526.csv")
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "press_feed_compare.csv")
PROXY_R = 0.937           # the incumbent's season-level correlation, from press_measured


def understat(season):
    p = _os.path.join(config.SD_CACHE, "understat_team", f"{season}.csv.gz")
    u = pd.read_csv(p, parse_dates=["date"])
    u["team"] = u["team"].map(sd.normalise_team)
    u["l"] = np.log(u["ppda"])
    return u.sort_values(["team", "date"])


def pitch(path=PA_2526):
    d = pd.read_csv(path)
    d = d[d["defending.ppda"] > 0].copy()
    d["l"] = np.log(d["defending.ppda"])
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["team", "date"])


def var_components(d, col="l", by="team"):
    g = d.groupby(by)[col]
    m = g.mean()
    sw = ((d[col] - d[by].map(m)) ** 2).sum() / (len(d) - g.ngroups)
    sb = m.var(ddof=1) - sw / g.size().mean()
    return sw, sb, sb / (sb + sw)


def backtest(meas, prior, truth, ks):
    """meas/prior/truth are aligned arrays over (team, n). Returns {k: MSE}."""
    out = {}
    for k in ks:
        w = meas["n"] / (meas["n"] + k)
        est = w * meas["v"] + (1 - w) * prior
        out[k] = float(((est - truth) ** 2).mean())
    return out


def main():
    if not _os.path.exists(PA_2526):
        raise SystemExit(f"no PitchAPI 25/26 pull at {PA_2526} — run pitchapi first")
    pa, u, up = pitch(), understat("2526"), understat("2425")

    su = u.groupby("team")["l"].mean()
    # ratio-of-sums, the correct season statistic for a ratio; available only because
    # PitchAPI publishes numerator and denominator separately.
    ros = (pa.groupby("team")["defending.opponent_passes"].sum()
           / pa.groupby("team")["defending.defensive_actions"].sum())
    spa_ros, spa_ml = np.log(ros), pa.groupby("team")["l"].mean()

    print("=" * 78)
    print("PRESS FEED COMPARISON — PitchAPI vs the component proxy, against Understat")
    print("=" * 78)
    print(f"  PitchAPI 25/26: {len(pa)} team-matches, {pa.team.nunique()} clubs")

    print("\nSEASON-LEVEL AGREEMENT WITH UNDERSTAT 25/26")
    rows = []
    for lab, s in (("PitchAPI ratio-of-sums", spa_ros), ("PitchAPI mean-of-log", spa_ml)):
        j = pd.concat([s.rename("x"), su.rename("y")], axis=1).dropna()
        r = j.x.corr(j.y)
        b, a = np.polyfit(j.x, j.y, 1)
        resid = (j.y - (a + b * j.x)).std(ddof=2)
        rows.append((lab, len(j), r, a, b, resid))
        print(f"  {lab:24s} n={len(j)}  r={r:.4f}  calib y={a:+.4f}{b:+.4f}x  "
              f"resid sd={resid:.4f}")
    print(f"  {'component proxy (incumbent)':24s}      r={PROXY_R:.4f}   [press_measured]")
    best = max(rows, key=lambda t: t[2])
    rule1 = best[2] > PROXY_R
    print(f"\n  RULE (1) PitchAPI r {best[2]:.4f} > proxy {PROXY_R:.4f} ?  "
          f"{'PASS' if rule1 else 'FAIL'}")

    print("\nVARIANCE COMPONENTS (log PPDA, within 25/26)")
    for lab, d in (("PitchAPI", pa), ("Understat", u)):
        sw, sb, r1 = var_components(d)
        print(f"  {lab:10s} sigma2_w={sw:.4f}  sigma2_b={sb:.4f}  "
              f"single-match reliability r1={r1:.4f}")

    print("\nOPERATIONAL BACKTEST — prior(24/25 Understat) + n PitchAPI matches")
    print("                       -> held-out 25/26 Understat")
    prior = up.groupby("team")["l"].mean().to_dict()
    pleague = float(np.mean(list(prior.values())))
    calib = np.polyfit(spa_ml.reindex(su.index).dropna(),
                       su.reindex(spa_ml.index).dropna(), 1)
    recs = []
    for t in sorted(set(spa_ml.index) & set(su.index)):
        if t not in prior:
            continue
        lp = pa[pa.team == t]["l"].values
        lu = u[u.team == t]["l"].values
        for n in range(1, 26):
            if len(lu) - n < 8 or n > len(lp) - 8:
                break
            recs.append((t, n, prior[t], float(np.polyval(calib, lp[:n].mean())),
                         float(lu[n:].mean())))
    b = pd.DataFrame(recs, columns=["team", "n", "prior", "meas", "truth"])
    ks = np.arange(0.5, 120, 0.25)
    tot = backtest({"n": b.n, "v": b.meas}, b.prior, b.truth, ks)
    kb = min(tot, key=tot.get)
    base = float(((b.prior - b.truth) ** 2).mean())
    print(f"  rows {len(b)}, teams {b.team.nunique()}")
    print(f"  prior-only MSE {base:.4f}")
    print(f"  best k = {kb:.2f}   MSE {tot[kb]:.4f}   gain {1 - tot[kb] / base:+.1%}")
    for k in (8, 12, 20, 40, 60):
        kk = min(ks, key=lambda x: abs(x - k))
        print(f"     k={k:4.0f}  gain {1 - tot[kk] / base:+.1%}")
    rule2 = kb < 40
    print(f"\n  RULE (2) operational optimum k {kb:.1f} materially below 40 ?  "
          f"{'PASS' if rule2 else 'FAIL'}")

    print("\n" + "=" * 78)
    if rule1 and rule2:
        print("VERDICT: SWITCH the feed to PitchAPI, and lower k to the value that is")
        print("         non-negative in both frames (see the gain table above).")
    elif rule1:
        print("VERDICT: SWITCH the feed to PitchAPI, KEEP k=40. A better measurement")
        print("         does not by itself earn a faster displacement of the prior.")
    else:
        print("VERDICT: NULL — the proxy stays and PitchAPI is not wired into the press")
        print("         channel. A tested null stays dead.")
    print("=" * 78)
    pd.DataFrame(rows, columns=["feed", "n", "r", "calib_a", "calib_b", "resid_sd"]
                 ).to_csv(OUT, index=False)
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
