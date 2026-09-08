"""
press_switchover.py — when is measured PPDA worth more than the judgment forecast?
=================================================================================
`press_index.PPDA_2627` is a manager-aware [JUDGMENT] forecast that scales every MID/FWD
DefCon rate through `defcon_env`'s CBIRT channel. `press_measured` revises it against
26/27 results with weight w = n/(n+k). This study is where k comes from, and it is the
evidence behind every constant in that module.

PRE-REGISTERED before results were seen:
  H1  A club's PPDA is a real season-long trait but a single match is mostly noise.
      RULE: report single-match reliability r1 = sigma2_b/(sigma2_b+sigma2_w). If
      r1 > 0.4 the "one match is noise" claim is withdrawn.
  H2  Blending an in-season measurement into a previous-season prior beats the prior.
      RULE: out-of-sample MSE against HELD-OUT matches, over >= 200 team-seasons. The
      blend must beat prior-only by >= 5% at its optimal k or the channel stays off.
  H3  The optimal k is constant in n (the inverse-variance model's prediction).
      RULE: if best-k drifts monotonically with n by more than 2x across n = 1..25 the
      single-constant form is rejected in favour of a saturating weight.
  H4  The repo proxy is an adequate stand-in for Understat PPDA.
      RULE: season-level correlation >= 0.90 across 20 clubs, else the feed is rejected.

OUTCOMES [2026-08-28]
  H1  CONFIRMED. r1 = 0.168 (Understat, 12 seasons, 9120 team-matches). A GW1 press
      table is ~92% noise. The GW1 measured-vs-assumed correlation of +0.14 that
      prompted this study is fully consistent with the assumptions being CORRECT: the
      ceiling on that correlation at n=1 is sqrt(0.168) = 0.41.
  H2  CONFIRMED on the Understat-native feed (-27.5% MSE at k=11.5, 220 team-seasons),
      NOT CONFIRMED on the operationally available proxy feed, where the optimum is
      k = 44-50 for +9% and k = 12 loses 3.6%. Only one season is testable that way.
      Resolution: K_DEFAULT = 40, the value non-negative under BOTH. Recorded as a
      partial null — the optimistic k is not available on the evidence.
  H3  CONFIRMED on the Understat feed: best-k ranges 9.75-14.25 over n = 1..25 with no
      trend, against a 2x rejection band. The single-constant form stands.
  H4  CONFIRMED at r = 0.937, but only after two specification findings (see below).
      The first proxy tried scored 0.878 and would have been rejected.

TWO FINDINGS THAT WOULD HAVE SILENTLY CORRUPTED THE FEED
  - `tackles` (attempted) is 100% NULL in 26/27 while present in 25/26. A proxy
    calibrated on 25/26 using it imports fine and divides by the wrong denominator all
    season. `tackles_won` is the only populated form.
  - `recoveries` belongs in the denominator (0.878 -> 0.937). It is the repo's closest
    analogue to Understat's "challenges", and the action CBIRT is actually driven by.

NOT THE EXPLANATION, tested: schedule confounding. A club's first n fixtures are not
opponent-balanced and PPDA depends on the opponent, but opponent-adjusting the feed with
a two-way additive model moves the optimum only from k = 49.5 to 44.2. Applied anyway
(principled, free); it is not where the gap between the two backtests comes from.

Run:  python studies/press_switchover.py
Needs the soccerdata Understat cache (.cache/soccerdata/understat_team/*.csv.gz) for the
multi-season legs; the proxy legs need only the FPL repo.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config, sd_ingest as sd, press_measured as pm

SEASONS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122",
           "2223", "2324", "2425", "2526"]
OUT = os.path.join(os.path.dirname(__file__), "press_switchover.csv")


def understat_panel():
    fr = []
    for s in SEASONS:
        p = os.path.join(config.SD_CACHE, "understat_team", f"{s}.csv.gz")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d["season"] = s
        d["team"] = d["team"].map(sd.normalise_team)
        fr.append(d)
    if not fr:
        return None
    d = pd.concat(fr, ignore_index=True)
    d["l"] = np.log(d["ppda"])
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["season", "team", "date"])


def variance_components(d):
    rows = []
    for s, g in d.groupby("season"):
        tm = g.groupby("team")["l"]
        m = tm.mean()
        sw = ((g["l"] - g["team"].map(m)) ** 2).sum() / (len(g) - g["team"].nunique())
        sb = m.var(ddof=1) - sw / tm.size().mean()
        rows.append((s, sw, sb, sb / (sb + sw)))
    return pd.DataFrame(rows, columns=["season", "sigma2_w", "sigma2_b", "r1"])


def persistence(d):
    ss = d.groupby(["season", "team"])["l"].mean().unstack(0)
    out = []
    for a, b in zip(SEASONS, SEASONS[1:]):
        if a not in ss.columns or b not in ss.columns:
            continue
        j = ss[[a, b]].dropna()
        out.append((f"{a}->{b}", len(j), j[a].corr(j[b])))
    return pd.DataFrame(out, columns=["pair", "n_clubs", "r"])


def backtest(d, ks):
    """Prior = previous season mean; measured = first n matches; truth = the REST.

    Truth is held out from both estimators, so neither shares sampling noise with it and
    the argmin over k is unbiased.
    """
    prev = {}
    for a, b in zip(SEASONS, SEASONS[1:]):
        g = d[d["season"] == a].groupby("team")["l"].mean()
        if len(g):
            prev[b] = (g.to_dict(), float(g.mean()))
    rows = []
    for s in SEASONS[1:]:
        if s not in prev:
            continue
        pmap, pleague = prev[s]
        for t, g in d[d["season"] == s].groupby("team"):
            L = g["l"].to_numpy()
            if len(L) < 20:
                continue
            for n in range(1, 26):
                if len(L) - n < 8:
                    break
                rows.append((s, t, t not in pmap, n, pmap.get(t, pleague),
                             L[:n].mean(), L[n:].mean()))
    b = pd.DataFrame(rows, columns=["season", "team", "promoted", "n",
                                    "prior", "meas", "truth"])
    return b


def _best_k(sub, ks):
    mse = {}
    for k in ks:
        w = sub["n"] / (sub["n"] + k)
        mse[k] = float(((w * sub["meas"] + (1 - w) * sub["prior"] - sub["truth"]) ** 2).mean())
    kb = min(mse, key=mse.get)
    base = float(((sub["prior"] - sub["truth"]) ** 2).mean())
    return kb, mse[kb], base, mse


def main():
    ks = np.arange(0.5, 140, 0.25)
    rows_out = []
    d = understat_panel()
    if d is None:
        print("no Understat cache — skipping the multi-season legs")
    else:
        vc = variance_components(d)
        sw, sb = vc["sigma2_w"].mean(), vc["sigma2_b"].mean()
        r1 = sb / (sb + sw)
        print("=" * 74)
        print("H1  variance components, Understat log-PPDA")
        print("=" * 74)
        print(f"  seasons {len(vc)}  team-matches {len(d)}")
        print(f"  sigma2_w {sw:.4f}   sigma2_b {sb:.4f}   single-match r1 {r1:.4f}")
        print(f"  VERDICT: {'CONFIRMED' if r1 <= 0.4 else 'WITHDRAWN'} (rule: r1 <= 0.40)")
        print(f"  matches needed for reliability 0.7: {0.7 * (1 - r1) / (r1 * 0.3):.0f}")

        pe = persistence(d)
        rbar = pe["r"].mean()
        tau2 = sb * (1 - rbar ** 2)
        print(f"\n  year-over-year persistence  mean r {rbar:.3f} over {len(pe)} transitions")
        print(f"  -> prior tau2 {tau2:.4f}   theory k = sigma2_w/tau2 = {sw / tau2:.1f}")

        b = backtest(d, ks)
        print("\n" + "=" * 74)
        print("H2/H3  out-of-sample backtest, Understat feed")
        print("=" * 74)
        drift = []
        for n in range(1, 26):
            sub = b[(b["n"] == n) & (~b["promoted"])]
            if len(sub) < 50:
                continue
            kb, m, base, _ = _best_k(sub, ks)
            drift.append(kb)
        sub = b[~b["promoted"]]
        kb, m, base, _ = _best_k(sub, ks)
        print(f"  team-seasons {b.groupby(['season', 'team']).ngroups}   rows {len(sub)}")
        print(f"  best k {kb:.2f}   MSE {m:.4f} vs prior-only {base:.4f}   gain {1 - m / base:+.1%}")
        print(f"  VERDICT H2: {'CONFIRMED' if 1 - m / base >= 0.05 else 'NULL'} (rule: >= 5%)")
        print(f"  best-k over n=1..25: {min(drift):.2f}-{max(drift):.2f}"
              f"  ratio {max(drift) / min(drift):.2f}")
        print(f"  VERDICT H3: {'CONFIRMED' if max(drift) / min(drift) < 2 else 'REJECTED'}"
              f" (rule: < 2x drift)")
        sp = b[b["promoted"]]
        if len(sp) > 50:
            kp, mp_, bp, _ = _best_k(sp, ks)
            print(f"  promoted clubs ({sp.groupby(['season','team']).ngroups} team-seasons):"
                  f" best k {kp:.2f}  gain {1 - mp_ / bp:+.1%}"
                  f"   -> weaker prior switches faster")
        for n in [1, 3, 5, 10, 19, 38]:
            rows_out.append(("weight_schedule", n, pm.blend_weight(n), ""))

    # ---- H4: the proxy the module actually uses
    print("\n" + "=" * 74)
    print("H4  repo proxy vs Understat, 25/26")
    print("=" * 74)
    try:
        a, bb, r = pm.fit_calibration("2025-2026")
        print(f"  log_understat = {a:.5f} + {bb:.5f} * log_proxy    r = {r:.4f}")
        print(f"  frozen in press_measured: A {pm.CALIB_A:.5f}  B {pm.CALIB_B:.5f}")
        print(f"  VERDICT: {'CONFIRMED' if r >= 0.90 else 'REJECTED'} (rule: r >= 0.90)")
        rows_out.append(("calibration", 20, r, f"a={a:.5f} b={bb:.5f}"))
    except Exception as e:                                    # noqa: BLE001
        print(f"  unavailable ({type(e).__name__}: {e}) — needs the Understat cache")

    print("\n" + "=" * 74)
    print(f"SHIPPED: k = {pm.K_DEFAULT:g} (weak prior {pm.K_WEAK_PRIOR:g})")
    print("=" * 74)
    print("  " + "  ".join(f"GW{n}={pm.blend_weight(n):.2f}"
                           for n in [1, 3, 5, 10, 19, 38]))
    if rows_out:
        pd.DataFrame(rows_out, columns=["quantity", "n", "value", "note"]).to_csv(OUT, index=False)
        print(f"\nwrote {OUT}")


def selftest():
    """Offline. The backtest machinery must recover a known k on synthetic data."""
    rng = np.random.default_rng(11)
    sw, tau2 = 0.20, 0.02
    rows = []
    for s in SEASONS[1:3]:
        for t in range(20):
            true = rng.normal(0, np.sqrt(0.04))
            prior = true + rng.normal(0, np.sqrt(tau2))
            L = true + rng.normal(0, np.sqrt(sw), 38)
            for n in range(1, 26):
                rows.append((s, f"T{t}", False, n, prior, L[:n].mean(), L[n:].mean()))
    b = pd.DataFrame(rows, columns=["season", "team", "promoted", "n",
                                    "prior", "meas", "truth"])
    kb, m, base, _ = _best_k(b, np.arange(0.5, 60, 0.25))
    want = sw / tau2
    assert 0.5 * want < kb < 2.0 * want, f"recovered k={kb:.1f}, planted {want:.1f}"
    assert m < base, "blend must beat the prior when the model is true"
    print("press_switchover selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
