from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
early_dispersion.py — does the model under-disperse team strength in the opening weeks?
=======================================================================================
`team_archetype_study.py` found that archetype MISMATCH drives goals about three times
harder in GW1-6 than later (+0.185 vs +0.062 goals per rank-gap). If that is a real
property of the early season rather than an artefact of coarse rank buckets, the model is
wrong in a specific, fixable way: it forms

    lambda = exp(mu + home + att - def)

with the same (att - def) response in every gameweek. A steeper early response would mean
strong-vs-weak fixtures are MORE lopsided in GW1-6 than the model projects — good fixtures
better, bad fixtures worse — which changes every early-season captaincy and
cheap-defender call.

WHY THIS REPEATS THE TEST ON A CONTINUOUS SCALE
------------------------------------------------
The rank-gap version bucketed teams into four archetypes, so "3× steeper" could be an
artefact of where the bucket edges fall. The model does not use buckets; it uses a
continuous strength difference. So the estimand here is the slope of match xG on
(own season strength - opponent season strength), which is exactly the quantity
`att - def` stands in for, measured separately for GW1-6 and GW7+.

Both slopes come from the SAME season-long ratings, so any difference is about the early
window, not about knowing less early. Clustered by team-season throughout.

Run:  python studies/early_dispersion.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest
import team_archetype_study as tas
import late_form_carryover as lfc

EARLY = 6
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "early_dispersion.csv")


def build():
    tm = tas.build_panel()
    # season-long ratings from ALL matches, so the early and late windows are scored
    # against identical measures of who is good
    att = tm.groupby(["season", "team"])["npxg"].mean().rename("A")
    dfn = tm.groupby(["season", "team"])["npxga"].mean().rename("D")
    tm = tm.join(att, on=["season", "team"]).join(dfn.rename("Dopp"), on=["season", "opp"])
    lg = tm.groupby("season")["npxg"].transform("mean")
    # strength difference in log space, the scale the model actually works on
    tm["sdiff"] = np.log(tm["A"] / lg) - np.log(tm["Dopp"] / lg)
    tm["y"] = np.log(np.maximum(tm["npxg"], 0.05) / lg)
    return tm.dropna(subset=["sdiff", "y"])


def slope_ci(d, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    units = d.groupby(["season", "team"]).ngroup().values
    uniq = np.unique(units)
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(units == u) for u in pick])
        s = d.iloc[idx]
        b, _, _ = lfc.ols([s["sdiff"]], s["y"].values)
        out.append(b[1])
    return np.array(out)


def main():
    d = build()
    print(f"[study] {len(d)} team-matches, {d.season.nunique()} seasons")
    early = d[d.md <= EARLY]
    rest = d[d.md > EARLY]

    be, _, r2e = lfc.ols([early["sdiff"]], early["y"].values)
    br, _, r2r = lfc.ols([rest["sdiff"]], rest["y"].values)
    print("\n" + "=" * 74)
    print("RESPONSE OF MATCH xG TO STRENGTH DIFFERENCE, early vs rest")
    print("=" * 74)
    print(f"  {'window':10s} {'n':>6s} {'slope':>8s} {'r2':>7s}")
    print(f"  {'GW1-' + str(EARLY):10s} {len(early):6d} {be[1]:8.4f} {r2e:7.4f}")
    print(f"  {'GW' + str(EARLY+1) + '+':10s} {len(rest):6d} {br[1]:8.4f} {r2r:7.4f}")

    bse, bsr = slope_ci(early), slope_ci(rest, seed=1)
    diff = bse - bsr
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"\n  early slope 95% CI ({np.percentile(bse,2.5):.4f}, {np.percentile(bse,97.5):.4f})")
    print(f"  rest  slope 95% CI ({np.percentile(bsr,2.5):.4f}, {np.percentile(bsr,97.5):.4f})")
    print(f"  DIFFERENCE (early - rest): {be[1]-br[1]:+.4f}  95% CI ({lo:+.4f}, {hi:+.4f})"
          f"{'  *' if not (lo <= 0 <= hi) else '  <- overlaps zero'}")
    print("\n  A slope near 1.0 means the model's multiplicative form is already right.")
    print("  Only a significantly STEEPER early slope would justify a GW1-6 amplifier.")

    # phase profile, to see whether anything special happens in the opening weeks
    print("\n  slope by phase:")
    for lo_, hi_, lab in [(1, 3, "GW1-3"), (4, 6, "GW4-6"), (7, 12, "GW7-12"),
                          (13, 19, "GW13-19"), (20, 38, "GW20-38")]:
        s = d[(d.md >= lo_) & (d.md <= hi_)]
        b, _, _ = lfc.ols([s["sdiff"]], s["y"].values)
        print(f"    {lab:9s} n={len(s):5d}  slope {b[1]:.4f}")

    # and the same on GOALS, since that is what scores points
    d["yg"] = np.log(np.maximum(d["gf"], 0.5) / d.groupby("season")["gf"].transform("mean"))
    bge, _, _ = lfc.ols([d[d.md <= EARLY]["sdiff"]], d[d.md <= EARLY]["yg"].values)
    bgr, _, _ = lfc.ols([d[d.md > EARLY]["sdiff"]], d[d.md > EARLY]["yg"].values)
    print(f"\n  same test on GOALS: early {bge[1]:.4f} vs rest {bgr[1]:.4f}")

    # ---- the better-specified version, and the one that actually decides it ----
    # The regression above takes log of a SINGLE match's xG, which is dominated by noise
    # and floored often enough to compress the slope — hence r2 ~ 0.005 and an interval
    # wide enough to admit almost anything. Ask the question the model's form asks
    # instead: build the multiplicative expectation the model would use, and test whether
    # its RESIDUAL still depends on the strength difference. If the form is right the
    # slope is zero; a positive early slope is exactly the under-dispersion claimed.
    lg = d.groupby("season")["npxg"].transform("mean")
    d["exp_mult"] = d["A"] * d["Dopp"] / lg
    d["resid"] = d["npxg"] - d["exp_mult"]
    print("\n" + "=" * 74)
    print("RESIDUAL TEST — does the multiplicative form leave strength on the table?")
    print("=" * 74)
    print(f"  {'window':10s} {'n':>6s} {'slope':>9s} {'95% CI':>22s}")
    res = {}
    for lab, s, seed in (("GW1-6", d[d.md <= EARLY], 2), ("GW7+", d[d.md > EARLY], 3)):
        b, _, _ = lfc.ols([s["sdiff"]], s["resid"].values)
        rng = np.random.default_rng(seed)
        units = s.groupby(["season", "team"]).ngroup().values
        uniq = np.unique(units)
        bs = []
        for _ in range(1200):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(units == u) for u in pick])
            q = s.iloc[idx]
            bb, _, _ = lfc.ols([q["sdiff"]], q["resid"].values)
            bs.append(bb[1])
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        res[lab] = np.array(bs)
        print(f"  {lab:10s} {len(s):6d} {b[1]:9.4f}   ({lo_:+.4f}, {hi_:+.4f})")
    dd = res["GW1-6"] - res["GW7+"]
    lo_, hi_ = np.percentile(dd, [2.5, 97.5])
    print(f"\n  DIFFERENCE (early - rest): {dd.mean():+.4f}  95% CI ({lo_:+.4f}, {hi_:+.4f})"
          f"{'  *' if not (lo_ <= 0 <= hi_) else '  <- overlaps zero'}")
    print("\n  Zero in both windows means the model's multiplicative form already captures")
    print("  the matchup, and a GW1-6 amplifier would be fitting noise.")

    pd.DataFrame([{"window": "early", "slope": be[1], "n": len(early)},
                  {"window": "rest", "slope": br[1], "n": len(rest)}]).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
