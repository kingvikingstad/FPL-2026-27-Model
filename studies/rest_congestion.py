from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
rest_congestion.py — does rest before a fixture move the result, and by how much?
=================================================================================
The model has NOTHING for fixture congestion. lambda = exp(mu + home + att - def) is the
same whether a side has had three days or nine. If rest matters, that is a per-FIXTURE
adjustment the board could use immediately — the same shape as the GW1-3 home discount
already applied, and available for every gameweek rather than just the opening three.

ESTIMAND: THE REST DIFFERENTIAL, NOT RAW REST
---------------------------------------------
Raw rest days are confounded with everything about the fixture calendar — the clubs on
short rest in midweek rounds are disproportionately the good ones (cup runs, Europe,
TV picks). What is exogenous-ish within a fixture is the DIFFERENCE between the two sides'
rest, and it is symmetric by construction: whatever one team gains the other loses. So the
regressor is `rest_diff` = own days since last PL match minus the opponent's, and the
outcome is opponent- and venue-adjusted residual xG, which already nets out both teams'
season-long quality.

KNOWN MEASUREMENT LIMIT, STATED UP FRONT
-----------------------------------------
Only PREMIER LEAGUE matches are visible here, so a club playing a Thursday Europa tie
looks fully rested by Sunday. That is real measurement error and it ATTENUATES the
estimate toward zero — so a null here is weaker evidence than a null usually is, and any
effect that does show up is, if anything, understated. European fixtures are not in the
Understat panel and adding them would mean scraping every club's full calendar.

Run:  python studies/rest_congestion.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest
import midtable_fade as mf
import late_form_carryover as lfc

OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "rest_congestion.csv")


def build():
    d = mf.build()
    d = d.sort_values(["season", "team", "date"])
    d["prev"] = d.groupby(["season", "team"])["date"].shift(1)
    d["rest"] = (d["date"] - d["prev"]).dt.days
    # opponent's rest for the same fixture
    key = d.set_index(["season", "match_id", "team"])["rest"]
    d["opp_rest"] = key.reindex(
        pd.MultiIndex.from_arrays([d["season"], d["match_id"], d["opp"]])).values
    d["rest_diff"] = d["rest"] - d["opp_rest"]

    # opponent- and venue-adjusted residual, as elsewhere in this project
    att = d.groupby(["season", "team"])["npxg"].mean().rename("A")
    dfn = d.groupby(["season", "team"])["npxga"].mean().rename("D")
    lg = d.groupby("season")["npxg"].mean().rename("L")
    d = d.join(att, on=["season", "team"]).join(
        dfn.rename("Dopp"), on=["season", "opp"]).join(lg, on="season")
    hm = d[d.is_home]["npxg"].mean() / d[~d.is_home]["npxg"].mean()
    d["exp_xg"] = d["A"] * d["Dopp"] / d["L"] * np.where(
        d["is_home"], np.sqrt(hm), 1 / np.sqrt(hm))
    d["resid"] = d["npxg"] - d["exp_xg"]
    # conceding side of the same idea, for clean-sheet relevance
    d["exp_xga"] = d["Dopp"] * 0 + d.groupby(["season", "opp"])["npxg"].transform("mean")
    return d.dropna(subset=["rest", "opp_rest"])


def cluster_ci(d, xcol, ycol, n=600, seed=0):
    rng = np.random.default_rng(seed)
    units = d.groupby(["season", "team"]).ngroup().values
    uniq = np.unique(units)
    bs = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(units == u) for u in pick])
        s = d.iloc[idx]
        b, _, _ = lfc.ols([s[xcol]], s[ycol].values)
        bs.append(b[1])
    return np.percentile(bs, [2.5, 97.5])


def main():
    d = build()
    print(f"[study] {len(d)} team-matches with a rest measure, "
          f"{d.season.nunique()} seasons")
    print("\n  rest-days distribution (days since that team's previous PL match):")
    r = d["rest"].clip(upper=14)
    print(r.value_counts().sort_index().head(14).to_string())

    print("\n" + "=" * 74)
    print("1. RAW REST — confounded, shown for context only")
    print("=" * 74)
    b = pd.cut(d["rest"], [0, 3, 4, 5, 6, 7, 8, 99],
               labels=["<=3", "4", "5", "6", "7", "8", "9+"])
    print(d.groupby(b).agg(n=("resid", "size"), resid=("resid", "mean"),
                           npxg=("npxg", "mean")).round(3).to_string())

    print("\n" + "=" * 74)
    print("2. REST DIFFERENTIAL — the estimand (own rest minus opponent's)")
    print("=" * 74)
    bd = pd.cut(d["rest_diff"], [-99, -4, -2, -0.5, 0.5, 2, 4, 99],
                labels=["<=-4", "-3..-2", "-1", "0", "+1", "+2..+3", ">=+4"])
    tab = d.groupby(bd).agg(n=("resid", "size"), resid_xg=("resid", "mean"),
                            npxg=("npxg", "mean")).round(3)
    print(tab.to_string())

    bb, ss, r2 = lfc.ols([d["rest_diff"]], d["resid"].values)
    lo, hi = cluster_ci(d, "rest_diff", "resid")
    print(f"\n  resid_xg ~ rest_diff : {bb[1]:+.5f} xG per extra day of rest advantage")
    print(f"    naive se {ss[1]:.5f} (t {bb[1]/ss[1]:+.2f}) | clustered 95% CI "
          f"({lo:+.5f}, {hi:+.5f})"
          f"{'  *' if not (lo <= 0 <= hi) else '  (not significant)'}")

    # restrict to the fixtures where congestion is actually plausible
    sub = d[d["rest_diff"].abs() >= 2]
    bb2, ss2, _ = lfc.ols([sub["rest_diff"]], sub["resid"].values)
    lo2, hi2 = cluster_ci(sub, "rest_diff", "resid", seed=1)
    print(f"\n  restricted to |rest_diff| >= 2 days (n={len(sub)}):")
    print(f"    {bb2[1]:+.5f} per day, clustered 95% CI ({lo2:+.5f}, {hi2:+.5f})"
          f"{'  *' if not (lo2 <= 0 <= hi2) else '  (not significant)'}")

    print("\n" + "=" * 74)
    print("3. SHORT TURNAROUND — the sharp version: 3 days or fewer vs a rested opponent")
    print("=" * 74)
    d["short"] = (d["rest"] <= 3) & (d["opp_rest"] >= 5)
    d["opp_short"] = (d["opp_rest"] <= 3) & (d["rest"] >= 5)
    for lab, m in (("on <=3 days vs opponent on 5+", d[d["short"]]),
                   ("balanced (both 5+)", d[(d["rest"] >= 5) & (d["opp_rest"] >= 5)]),
                   ("opponent on <=3, we had 5+", d[d["opp_short"]])):
        if len(m) < 20:
            print(f"  {lab:32s} n={len(m):5d}  (too few)"); continue
        print(f"  {lab:32s} n={len(m):5d}  resid xG {m['resid'].mean():+.4f}  "
              f"raw npxG {m['npxg'].mean():.3f}")
    a = d[d["short"]]["resid"].values
    c = d[(d["rest"] >= 5) & (d["opp_rest"] >= 5)]["resid"].values
    if len(a) >= 20:
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(a, len(a), replace=True).mean()
                       - rng.choice(c, len(c), replace=True).mean() for _ in range(6000)])
        lo3, hi3 = np.percentile(bs, [2.5, 97.5])
        print(f"\n  short-rest penalty vs balanced: {a.mean()-c.mean():+.4f} xG  "
              f"95% CI ({lo3:+.4f}, {hi3:+.4f})"
              f"{'  *' if not (lo3 <= 0 <= hi3) else '  (not significant)'}")

    print("\n" + "=" * 74)
    print("4. WHAT WOULD IT BE WORTH? translate to the model's units")
    print("=" * 74)
    lam = d["npxg"].mean()
    print(f"  league mean npxG per team-match = {lam:.3f}")
    print(f"  a 1-day rest advantage moves xG by {bb[1]:+.5f} = "
          f"{bb[1]/lam*100:+.2f}% of a typical team-match")
    print(f"  a 4-day swing (Thu-Sun vs a week off) = {4*bb[1]:+.4f} xG = "
          f"{4*bb[1]/lam*100:+.1f}%")
    print("  compare: the GW1-3 home-advantage discount already applied is worth ~15% of")
    print("  a home side's edge. An effect an order of magnitude smaller is not worth a")
    print("  model parameter.")

    d[["season", "team", "opp", "md", "rest", "opp_rest", "rest_diff", "resid"]].to_csv(
        OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
