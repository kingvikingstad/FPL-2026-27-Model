from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
tournament_summers.py — do major-tournament summers change the opening gameweeks?
==================================================================================
Directly relevant to 2026/27, which follows the June-July 2026 World Cup with a GW1
deadline of 21 Aug 2026 — a short turnaround for exactly the players concentrated at the
strongest clubs.

THE HONEST HEADLINE FIRST: THIS IS UNDERPOWERED BY CONSTRUCTION
----------------------------------------------------------------
Understat gives 12 PL seasons. Five follow a World Cup or Euro. That is the entire
sample, and no amount of match-level data fixes it — 380 matches in a season are not 380
independent observations of "what a tournament summer does", because the treatment is
applied once per season. The unit of analysis is therefore the SEASON: n=5 vs n=7.

With that n, only a very large effect is detectable. So this study reports effect sizes
with intervals and an explicit power statement, uses EXACT permutation tests (every
possible split of 12 seasons is enumerable — C(12,5)=792), and does not pretend a null
is evidence of absence.

DESIGN — difference-in-differences
-----------------------------------
Comparing tournament seasons to others directly would confound the tournament with era
drift (league scoring rose from 2.57 to 3.28 goals/game over this window). Comparing
early to late within a season removes era but not the general early-season pattern already
documented in early_season_goals.py. So the estimand is the DIFFERENCE OF DIFFERENCES:

    (GW1-6 minus rest-of-season)  in tournament years
  - (GW1-6 minus rest-of-season)  in other years

which nets out both era and the ordinary early-season pattern, leaving the part
attributable to the summer.

CLASSIFICATION (major = World Cup or Euro, the tournaments with most PL players)
--------------------------------------------------------------------------------
  2014/15  WC Brazil        12 Jun - 13 Jul 2014
  2016/17  Euro France      10 Jun - 10 Jul 2016  (+ Copa Centenario)
  2018/19  WC Russia        14 Jun - 15 Jul 2018
  2021/22  Euro (2020)      11 Jun - 11 Jul 2021  (+ Copa)
  2024/25  Euro Germany     14 Jun - 14 Jul 2024  (+ Copa USA)

Controls, with their caveats recorded rather than hidden:
  2015/16  Copa Chile only            -> secondary grouping treats this as treated
  2017/18  clean control
  2019/20  Copa + AFCON summer 2019   -> secondary grouping treats this as treated
  2020/21  COVID. The 19/20 season ended Jul 2020 and 20/21 began Sep 2020 — a ~6-week
           offseason for EVERY club, which is a bigger disruption than a tournament, plus
           empty stadiums. Kept in the control group for the primary test but reported
           separately, because it biases the control group toward looking disrupted and
           therefore biases the estimate TOWARD zero.
  2022/23  clean control (the 2022 WC was Nov-Dec, mid-season — it cannot affect GW1-6)
  2023/24  clean control
  2025/26  FIFA Club World Cup ran Jun-Jul 2025 and Man City and Chelsea went deep
           (Chelsea won it). A partial, club-specific version of the same treatment.

Run:  python studies/tournament_summers.py
"""
import warnings; warnings.filterwarnings("ignore")
from itertools import combinations
import numpy as np, pandas as pd
import early_season_goals as esg
import team_archetype_study as tas

EARLY = 6
MAJOR = {"1415", "1617", "1819", "2122", "2425"}          # WC or Euro
ALSO_CONF = {"1516", "1920"}                              # Copa / AFCON only
COVID = "2021"
CWC_PARTIAL = "2526"
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "tournament_summers.csv")


def season_metrics(tm):
    """Per (season, window) metrics. window: 'early' = GW1-EARLY, 'rest' = GW7+."""
    rows = []
    for s, g in tm.groupby("season"):
        for lab, w in (("early", g[g.md <= EARLY]), ("rest", g[g.md > EARLY])):
            h, a = w[w.is_home]["gf"].mean(), w[~w.is_home]["gf"].mean()
            rows.append({
                "season": s, "window": lab,
                "total_goals": w["total_goals"].mean(),
                "home_adv": h - a,
                "cs_rate": w["cs"].mean(),
                "goal_spread": w.groupby("team")["gf"].mean().std(),
            })
    return pd.DataFrame(rows)


def did(M, metric, treated):
    """Difference-in-differences plus an EXACT permutation p-value."""
    piv = M.pivot(index="season", columns="window", values=metric)
    piv["gap"] = piv["early"] - piv["rest"]
    t = piv.loc[sorted(treated), "gap"].values
    c = piv.loc[[s for s in piv.index if s not in treated], "gap"].values
    obs = t.mean() - c.mean()

    seasons = list(piv.index)
    k = len(treated)
    stats = []
    for combo in combinations(range(len(seasons)), k):
        m = np.zeros(len(seasons), bool); m[list(combo)] = True
        stats.append(piv["gap"].values[m].mean() - piv["gap"].values[~m].mean())
    stats = np.array(stats)
    p = float((np.abs(stats) >= abs(obs) - 1e-12).mean())
    return {"metric": metric, "treated_gap": t.mean(), "control_gap": c.mean(),
            "did": obs, "p_exact": p, "n_treated": len(t), "n_control": len(c),
            "perm_sd": float(stats.std())}


def main():
    tm = esg.build()
    M = season_metrics(tm)
    seasons = sorted(tm.season.unique())
    print(f"[study] {len(seasons)} seasons; treated (WC/Euro summer) = "
          f"{sorted(MAJOR)}")
    print(f"        n = {len(MAJOR)} treated vs {len(seasons)-len(MAJOR)} control. "
          f"Underpowered by construction — read the power note at the end.")

    print("\n" + "=" * 84)
    print("DIFFERENCE-IN-DIFFERENCES: (GW1-6 minus rest) in tournament years vs others")
    print("=" * 84)
    print(f"  {'metric':14s} {'tourn gap':>10s} {'ctrl gap':>10s} {'DiD':>9s} "
          f"{'exact p':>9s} {'perm sd':>9s}")
    rows = []
    for metric in ("total_goals", "home_adv", "cs_rate", "goal_spread"):
        r = did(M, metric, MAJOR)
        rows.append(r)
        star = " *" if r["p_exact"] < 0.05 else ""
        print(f"  {metric:14s} {r['treated_gap']:+10.3f} {r['control_gap']:+10.3f} "
              f"{r['did']:+9.3f} {r['p_exact']:9.3f} {r['perm_sd']:9.3f}{star}")
    print("  (exact p from all C(12,5)=792 possible assignments — no asymptotics)")

    # secondary grouping: count Copa/AFCON summers as treated too
    print("\n  SECONDARY — counting Copa/AFCON summers (15/16, 19/20) as treated too:")
    alt = MAJOR | ALSO_CONF
    for metric in ("total_goals", "home_adv", "cs_rate", "goal_spread"):
        r = did(M, metric, alt)
        print(f"    {metric:14s} DiD {r['did']:+7.3f}  exact p {r['p_exact']:.3f} "
              f"({r['n_treated']} vs {r['n_control']})")

    # does the COVID control season drive the result?
    print("\n  SENSITIVITY — dropping 2020/21 (COVID: ~6-week offseason for everyone):")
    M2 = M[M.season != COVID]
    for metric in ("total_goals", "home_adv", "cs_rate", "goal_spread"):
        r = did(M2, metric, MAJOR)
        print(f"    {metric:14s} DiD {r['did']:+7.3f}  exact p {r['p_exact']:.3f}")

    # ---- the mechanism test: do the STRONG clubs suffer more? ----
    print("\n" + "=" * 84)
    print("MECHANISM: tournament load concentrates at strong clubs, so if the effect is")
    print("real the prior-top-6 advantage should SHRINK in GW1-6 of tournament years")
    print("=" * 84)
    arch = tas.archetypes(tm)
    t2 = tm.merge(arch[["season", "team", "arch"]], on=["season", "team"], how="left")
    t2 = t2[t2["arch"].notna() & (t2["arch"] != "(no prior season)")]
    rows2 = []
    for s, g in t2.groupby("season"):
        for lab, w in (("early", g[g.md <= EARLY]), ("rest", g[g.md > EARLY])):
            top = w[w.arch == "prior top-6"]
            oth = w[w.arch != "prior top-6"]
            if len(top) and len(oth):
                rows2.append({"season": s, "window": lab,
                              "top6_edge": top["gf"].mean() - oth["gf"].mean()})
    G = pd.DataFrame(rows2)
    r = did(G, "top6_edge", MAJOR & set(G.season.unique()))
    print(f"  top-6 goal edge, early-minus-rest:")
    print(f"    tournament years {r['treated_gap']:+.3f}   others {r['control_gap']:+.3f}"
          f"   DiD {r['did']:+.3f}   exact p {r['p_exact']:.3f}")
    print("  a NEGATIVE DiD would mean the strong clubs start slower after a tournament")

    # ---- power ----
    print("\n" + "=" * 84)
    print("POWER — what could this design actually have detected?")
    print("=" * 84)
    for r in rows:
        mde = 1.96 * r["perm_sd"]
        print(f"  {r['metric']:14s} permutation sd {r['perm_sd']:.3f}  ->  only effects "
              f"larger than about {mde:.3f} are detectable")
    print("\n  Anything smaller than that is invisible here regardless of whether it is")
    print("  real. Treat every null below as 'not detectable at n=5 vs 7', NOT as 'zero'.")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")

    print("\n=== per-season detail (gap = GW1-6 minus rest) ===")
    piv = M.pivot(index="season", columns="window", values="total_goals")
    piv["goals_gap"] = piv["early"] - piv["rest"]
    ha = M.pivot(index="season", columns="window", values="home_adv")
    piv["home_gap"] = ha["early"] - ha["rest"]
    piv["summer"] = ["WC/Euro" if s in MAJOR else
                     ("Copa/AFCON" if s in ALSO_CONF else
                      ("COVID" if s == COVID else
                       ("ClubWC" if s == CWC_PARTIAL else "-")))
                     for s in piv.index]
    print(piv[["summer", "goals_gap", "home_gap"]].round(3).to_string())


if __name__ == "__main__":
    main()
