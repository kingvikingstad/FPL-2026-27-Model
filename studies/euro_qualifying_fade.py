from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
euro_qualifying_fade.py — settling the mid-table early fade
============================================================
Last surviving explanation for the prior-7-to-12 GW1-6 fade, after schedule
(midtable_fade.py), squad churn (transfer_churn.py) and player sales
(sale_hypothesis.py) were all eliminated:

    Conference/Europa League qualifying rounds are played in JULY and AUGUST. A club
    finishing 7th-8th plays competitive football before the season starts and midweek
    games through the opening gameweeks, while top-6 clubs enter at the group stage in
    September and 13-20 clubs have no European football at all.

Until now this was tested with prior rank as a PROXY for European participation. This
study replaces the proxy with the real thing: data/european_qualifying.csv, the English
clubs that actually played July/August European ties, scraped from the UEFA competition
qualifying pages on Wikipedia.

THE TEST THAT SETTLES IT
------------------------
Compare treated and untreated clubs WITHIN the prior-7-12 bucket. That removes archetype
entirely as a confound — same bucket, same classification, differing only in whether they
actually played qualifying football in July/August. If the mechanism is real the fade
should sit in the treated clubs and be absent from the untreated ones.

Two placebos come free:
  * Tottenham 2020/21 qualified for Europa League qualifying but COVID pushed the rounds
    to SEPTEMBER. Same competition, same qualification route, wrong month — so under this
    mechanism they should NOT fade.
  * Top-6 clubs that played CL play-offs in August (Man United 15/16, Man City 16/17,
    Liverpool 17/18) are treated but sit in a bucket showing no fade at all.

PROVENANCE
----------
Scraped via LLM summarisation of Wikipedia's UEFA qualifying pages, and that step is
demonstrably unreliable: the 2023-24 page falsely reported no English clubs when Aston
Villa played the play-off round, and the 2017-18 page contradicted itself within one
answer. Both were caught and corrected by cross-checking the competition page. The ROUND
is recorded rather than exact dates, because rounds were verified and several dates were
not. Treat the file as good-but-hand-checked, not authoritative.

Run:  python studies/euro_qualifying_fade.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest
import midtable_fade as mf
import late_form_carryover as lfc

EARLY = 6
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "euro_qualifying_fade.csv")


def load_treatment(valid):
    e = pd.read_csv(config.EUROPEAN_QUALIFYING)
    e["season"] = e["season"].astype(str).str.zfill(4)
    e["club"] = e["club"].map(sd_ingest.normalise_team)
    bad = {(s, c) for s, c in zip(e["season"], e["club"])} - valid
    if bad:
        raise RuntimeError(
            f"european_qualifying.csv references club-seasons not in the Understat panel: "
            f"{sorted(bad)}. A wrong club name or a club that was not in the PL that year.")
    print(f"[validate] all {len(e)} rows map onto real PL club-seasons")
    return e


def gaps_by_club_season(d):
    rows = []
    for (s, t, a), g in d.groupby(["season", "team", "arch"]):
        rows.append({"season": s, "team": t, "arch": a,
                     "gap": g[g.md <= EARLY]["resid"].mean() - g[g.md > EARLY]["resid"].mean(),
                     "early": g[g.md <= EARLY]["resid"].mean(),
                     "rest": g[g.md > EARLY]["resid"].mean()})
    return pd.DataFrame(rows)


def boot_mean(v, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(n)])
    return b


def main():
    d = mf.build()
    att = d.groupby(["season", "team"])["npxg"].mean().rename("A")
    dfn = d.groupby(["season", "team"])["npxga"].mean().rename("D")
    lg = d.groupby("season")["npxg"].mean().rename("L")
    d = d.join(att, on=["season", "team"]).join(
        dfn.rename("Dopp"), on=["season", "opp"]).join(lg, on="season")
    hm = d[d.is_home]["npxg"].mean() / d[~d.is_home]["npxg"].mean()
    d["exp_xg"] = d["A"] * d["Dopp"] / d["L"] * np.where(
        d["is_home"], np.sqrt(hm), 1 / np.sqrt(hm))
    d["resid"] = d["npxg"] - d["exp_xg"]

    G = gaps_by_club_season(d)
    valid = set(zip(G["season"], G["team"]))
    e = load_treatment(valid)

    treat = set(zip(e.loc[e["jul_aug"] == 1, "season"], e.loc[e["jul_aug"] == 1, "club"]))
    sept = set(zip(e.loc[e["jul_aug"] == 0, "season"], e.loc[e["jul_aug"] == 0, "club"]))
    G["euro_jul_aug"] = [(s, t) in treat for s, t in zip(G["season"], G["team"])]
    G["euro_sept"] = [(s, t) in sept for s, t in zip(G["season"], G["team"])]

    print(f"[study] {len(G)} club-seasons; {G.euro_jul_aug.sum()} played July/August "
          f"European ties, {G.euro_sept.sum()} played the same rounds in September")
    print("\n  treated club-seasons by archetype:")
    print(G[G.euro_jul_aug].groupby("arch").size().to_string())

    # ---------------------------------------------------------- THE TEST
    print("\n" + "=" * 76)
    print("THE TEST — treated vs untreated WITHIN the prior-7-12 bucket")
    print("=" * 76)
    mid = G[G["arch"] == "prior 7-12"]
    t = mid[mid["euro_jul_aug"]]["gap"].values
    u = mid[~mid["euro_jul_aug"] & ~mid["euro_sept"]]["gap"].values
    print(f"  {'group':34s} {'n':>4s} {'early-minus-rest':>17s} {'95% CI':>20s}")
    for lab, v in ((f"played Jul/Aug Euro qualifying", t),
                   (f"no European football", u)):
        b = boot_mean(v)
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"  {lab:34s} {len(v):4d} {v.mean():+17.3f}   ({lo:+.3f}, {hi:+.3f})")
    bt, bu = boot_mean(t, seed=1), boot_mean(u, seed=2)
    diff = bt - bu
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"\n  DIFFERENCE (treated minus untreated): {t.mean()-u.mean():+.3f}  "
          f"95% CI ({lo:+.3f}, {hi:+.3f})"
          f"{'  *' if not (lo <= 0 <= hi) else '  <- overlaps zero'}")
    print("  If the mechanism is real the fade lives in the treated clubs and the")
    print("  untreated mid-table clubs should look ordinary.")

    # ---------------------------------------------------------- placebos
    print("\n" + "=" * 76)
    print("PLACEBOS")
    print("=" * 76)
    sp = G[G["euro_sept"]]
    if len(sp):
        print(f"  September qualifying (COVID 20/21), n={len(sp)}: "
              f"gap {sp['gap'].mean():+.3f}")
        print("    same competition and route, wrong month — should show NO fade")
    t6 = G[(G["arch"] == "prior top-6")]
    t6t = t6[t6["euro_jul_aug"]]
    t6u = t6[~t6["euro_jul_aug"]]
    print(f"\n  prior top-6 who played Aug CL play-offs, n={len(t6t)}: "
          f"gap {t6t['gap'].mean():+.3f}")
    print(f"  prior top-6 who did not,                n={len(t6u)}: "
          f"gap {t6u['gap'].mean():+.3f}")

    # ---------------------------------------------------------- pooled
    print("\n" + "=" * 76)
    print("POOLED — does treatment explain the prior-7-12 indicator?")
    print("=" * 76)
    G["is_mid"] = (G["arch"] == "prior 7-12").astype(float)
    G["treat"] = G["euro_jul_aug"].astype(float)
    b1, s1, _ = lfc.ols([G["is_mid"]], G["gap"].values)
    b2, s2, _ = lfc.ols([G["is_mid"], G["treat"]], G["gap"].values)
    print(f"  gap ~ mid            : mid {b1[1]:+.4f} (se {s1[1]:.4f})")
    print(f"  gap ~ mid + treated  : mid {b2[1]:+.4f} (se {s2[1]:.4f}), "
          f"treated {b2[2]:+.4f} (se {s2[2]:.4f})")
    if b1[1]:
        print(f"\n  the prior-7-12 coefficient moves by "
              f"{(1-abs(b2[1])/abs(b1[1]))*100:+.1f}% once treatment is controlled for")

    G.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
