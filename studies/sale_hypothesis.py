from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
sale_hypothesis.py — does losing your best attacker cause the mid-table early fade?
===================================================================================
midtable_fade.py left two live explanations for the prior-7-to-12 GW1-6 fade after ruling
out schedule and squad churn. This tests the second one:

    Mid-table clubs are the ones who sell a leading attacker to a top-6 club each summer
    and replace him late. Bottom clubs have nobody worth buying; top-6 clubs retain. That
    would produce a temporary early deficit which recovers as the replacement beds in —
    the same shape, for a completely different reason than European qualifying.

MEASURE: `departed_xg_share` — the fraction of a club's non-penalty xG from season t-1
produced by players who did not appear for that club in season t. Built from Understat
player-match data across all 12 seasons, keyed on Understat's own player id (a crosswalk
to player_code is not needed: both sides of the comparison are Understat).

WHAT MAKES THIS A REAL TEST RATHER THAN A CORRELATION
------------------------------------------------------
The outcome is a WITHIN-CLUB early-minus-rest gap, so any level effect — a club that lost
its striker is simply worse all year — cancels out. The hypothesis is specifically about
a TEMPORARY deficit that recovers, so the estimand has to be the early-vs-late shape, not
the level. Three things have to hold for the hypothesis to survive:

  1. the mechanism must be PRESENT: prior-7-12 clubs must actually lose more attacking
     output than other archetypes;
  2. it must BITE: departed xG share must predict a bigger early-minus-rest deficit;
  3. it must EXPLAIN: controlling for it should shrink the prior-7-12 early effect.

Failing (1) kills it outright. Passing (1) but failing (2) means mid-table clubs do sell
more but it costs them nothing early.

Run:  python studies/sale_hypothesis.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest
import team_archetype_study as tas
import late_form_carryover as lfc
import midtable_fade as mf

SEASONS = ["%02d%02d" % (y % 100, (y + 1) % 100) for y in range(2014, 2026)]
EARLY = 6
ORDER = ["prior top-6", "prior 7-12", "prior 13-20", "promoted"]
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "sale_hypothesis.csv")


def departures():
    """Per (season, club): share of the PREVIOUS season's xG that left the club, and
    whether the club's single leading xG contributor left."""
    ps = sd_ingest.understat_player_season(SEASONS)
    ps["team"] = ps["team"].map(sd_ingest.normalise_team)
    ps["xg"] = pd.to_numeric(ps["np_xg"], errors="coerce").fillna(0.0)
    ps["minutes"] = pd.to_numeric(ps["minutes"], errors="coerce").fillna(0.0)
    ps = ps[["season", "team", "understat_player_id", "xg", "minutes"]]
    # who appeared for the club at all in a given season
    appeared = {(s, t): set(g["understat_player_id"])
                for (s, t), g in ps[ps["minutes"] > 0].groupby(["season", "team"])}

    rows = []
    for i in range(1, len(SEASONS)):
        prev, cur = SEASONS[i - 1], SEASONS[i]
        for (s, t), g in ps[ps["season"] == prev].groupby(["season", "team"]):
            if (cur, t) not in appeared:
                continue                      # relegated
            stay = appeared[(cur, t)]
            tot = g["xg"].sum()
            if tot <= 0:
                continue
            gone = g[~g["understat_player_id"].isin(stay)]
            top = g.sort_values("xg", ascending=False).iloc[0]
            rows.append({
                "season": cur, "team": t,
                "departed_xg_share": gone["xg"].sum() / tot,
                "prev_xg_total": tot,
                "top_scorer_left": bool(top["understat_player_id"] not in stay),
                "n_departed": int(len(gone)),
            })
    return pd.DataFrame(rows)


def main():
    d = mf.build()                              # team-matches with archetypes
    # opponent- and venue-adjusted residual, exactly as in midtable_fade
    att = d.groupby(["season", "team"])["npxg"].mean().rename("A")
    dfn = d.groupby(["season", "team"])["npxga"].mean().rename("D")
    lg = d.groupby("season")["npxg"].mean().rename("L")
    d = d.join(att, on=["season", "team"]).join(
        dfn.rename("Dopp"), on=["season", "opp"]).join(lg, on="season")
    hm = d[d.is_home]["npxg"].mean() / d[~d.is_home]["npxg"].mean()
    d["exp_xg"] = d["A"] * d["Dopp"] / d["L"] * np.where(
        d["is_home"], np.sqrt(hm), 1 / np.sqrt(hm))
    d["resid"] = d["npxg"] - d["exp_xg"]

    dep = departures()
    print(f"[study] departures computed for {len(dep)} club-seasons across "
          f"{dep.season.nunique()} seasons")
    print(f"        mean departed xG share {dep.departed_xg_share.mean():.3f}, "
          f"top scorer left in {dep.top_scorer_left.mean():.1%} of club-seasons")

    m = d.merge(dep, on=["season", "team"], how="inner")
    # early-minus-rest residual gap per club-season
    gaps = []
    for (s, t, a), g in m.groupby(["season", "team", "arch"]):
        e = g[g.md <= EARLY]["resid"].mean()
        r = g[g.md > EARLY]["resid"].mean()
        gaps.append({"season": s, "team": t, "arch": a, "gap": e - r,
                     "departed": g["departed_xg_share"].iloc[0],
                     "top_left": bool(g["top_scorer_left"].iloc[0])})
    G = pd.DataFrame(gaps)
    print(f"        {len(G)} club-seasons with both a fade measure and a departure measure")

    # ------------------------------------------------- 1. is the mechanism present?
    print("\n" + "=" * 76)
    print("1. IS THE MECHANISM PRESENT? do mid-table clubs lose more attacking output?")
    print("=" * 76)
    print(f"  {'archetype':14s} {'n':>4s} {'departed xG share':>18s} {'top scorer left':>16s}")
    for a in ORDER:
        s = G[G["arch"] == a]
        if not len(s):
            continue
        print(f"  {a:14s} {len(s):4d} {s['departed'].mean():18.3f} "
              f"{s['top_left'].mean():15.1%}")
    rng = np.random.default_rng(0)
    mid = G[G["arch"] == "prior 7-12"]["departed"].values
    oth = G[G["arch"] != "prior 7-12"]["departed"].values
    bs = np.array([rng.choice(mid, len(mid), replace=True).mean()
                   - rng.choice(oth, len(oth), replace=True).mean() for _ in range(4000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n  prior 7-12 minus the rest: {mid.mean()-oth.mean():+.3f}  "
          f"95% CI ({lo:+.3f}, {hi:+.3f})"
          f"{'  *' if not (lo <= 0 <= hi) else '  <- overlaps zero'}")
    print("  If this overlaps zero the hypothesis fails at step 1: mid-table clubs are")
    print("  not losing more attacking output than anyone else.")

    # ------------------------------------------------- 2. does it bite?
    print("\n" + "=" * 76)
    print("2. DOES IT BITE? does losing xG predict a bigger EARLY deficit?")
    print("=" * 76)
    b, se, r2 = lfc.ols([G["departed"]], G["gap"].values)
    print(f"  gap ~ departed xG share : {b[1]:+.4f} (se {se[1]:.4f}, t {b[1]/se[1]:+.2f}) "
          f" r2={r2:.4f}  n={len(G)}")
    bt, st, _ = lfc.ols([G["top_left"].astype(float)], G["gap"].values)
    print(f"  gap ~ top scorer left   : {bt[1]:+.4f} (se {st[1]:.4f}, "
          f"t {bt[1]/st[1]:+.2f})")
    print("  a NEGATIVE coefficient = losing more xG means a worse start relative to")
    print("  that club's own rest-of-season")

    q = pd.qcut(G["departed"], 4, labels=["Q1 keep most", "Q2", "Q3", "Q4 lose most"])
    print("\n  by quartile of departed xG share:")
    print(G.groupby(q)["gap"].agg(["count", "mean"]).round(4).to_string())

    # ------------------------------------------------- 3. does it explain the fade?
    print("\n" + "=" * 76)
    print("3. DOES IT EXPLAIN THE FADE? add it alongside the prior-7-12 indicator")
    print("=" * 76)
    G["is_mid"] = (G["arch"] == "prior 7-12").astype(float)
    b1, s1, _ = lfc.ols([G["is_mid"]], G["gap"].values)
    b2, s2, _ = lfc.ols([G["is_mid"], G["departed"]], G["gap"].values)
    print(f"  gap ~ mid                     : mid {b1[1]:+.4f} (se {s1[1]:.4f})")
    print(f"  gap ~ mid + departed xG share : mid {b2[1]:+.4f} (se {s2[1]:.4f}), "
          f"departed {b2[2]:+.4f} (se {s2[2]:.4f})")
    shrink = (1 - abs(b2[1]) / abs(b1[1])) * 100 if b1[1] else np.nan
    print(f"\n  the prior-7-12 coefficient moves by {shrink:+.1f}% when departures are")
    print("  controlled for. A large shrink would mean sales explain the fade;")
    print("  roughly no movement means they do not.")

    G.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
