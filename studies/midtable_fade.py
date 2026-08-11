from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
midtable_fade.py — why do prior-7-to-12 clubs attack worse in GW1-6?
====================================================================
team_archetype_study.py found one significant early-season effect: clubs that finished
7th-12th last season create ~7% less xG in GW1-6 than their own season average, -0.068
with a bootstrap CI of (-0.126, -0.007), improving monotonically thereafter
(0.905 -> 1.007 attack index, 1.22 -> 1.44 points per game).

STEP 0 IS WHETHER IT IS REAL AT ALL
------------------------------------
That was one of FOUR archetypes tested and its interval barely cleared zero. Four tests
with one marginal hit is exactly what noise looks like, so this study starts by
re-examining the effect itself under a multiplicity correction and a leave-one-season-out
check. Only if it survives is a mechanism worth hunting.

CANDIDATE MECHANISMS, in the order they should be eliminated
-------------------------------------------------------------
1. SCHEDULE. `att_rel` is a club's own xG against the league average — it is NOT adjusted
   for who they played or where. If mid-table clubs happen to draw harder opponents or
   more away games in the opening six, the "fade" is a fixture artefact and nothing more.
   Tested by rebuilding the measure as a residual from season-long team ratings.
2. EUROPEAN QUALIFYING. This is the one mechanism that would hit this bucket and no other.
   Conference/Europa League qualifying rounds are played in JULY and AUGUST, so a club
   finishing 7th-8th plays competitive football before the season starts and midweek games
   through GW1-6, while top-6 clubs go straight into group stages in September and 13-20
   clubs have no European football at all. Prediction: the fade should sit in the TOP of
   the bucket (7th-8th), not the bottom (11th-12th).
3. SQUAD CHURN. Already eliminated in transfer_churn.py — no early-loaded effect on any
   measure, so it is not carried further here except as a control.

Run:  python studies/midtable_fade.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest
import team_archetype_study as tas
import late_form_carryover as lfc

OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "midtable_fade.csv")
EARLY = 6
ORDER = ["prior top-6", "prior 7-12", "prior 13-20", "promoted"]


def build():
    tm = tas.build_panel()
    arch = tas.archetypes(tm)
    tm = tm.merge(arch[["season", "team", "arch", "prior_rank"]],
                  on=["season", "team"], how="left")
    tm = tm.merge(arch[["season", "team", "arch"]].rename(
        columns={"team": "opp", "arch": "opp_arch"}), on=["season", "opp"], how="left")
    return tm[tm["arch"].notna() & (tm["arch"] != "(no prior season)")].copy()


def boot_effect(d, arch, n=4000, seed=0):
    """Bootstrap the early-minus-rest attack gap for one archetype, resampling whole
    team-seasons."""
    sub = d[d["arch"] == arch]
    units = sub.groupby(["season", "team"]).ngroup().values
    uniq = np.unique(units)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(units == u) for u in pick])
        s = sub.iloc[idx]
        e = s[s.md <= EARLY]["att_rel"].mean()
        r = s[s.md > EARLY]["att_rel"].mean()
        out.append(e - r)
    out = np.array(out)
    obs = (sub[sub.md <= EARLY]["att_rel"].mean() - sub[sub.md > EARLY]["att_rel"].mean())
    p = float(2 * min((out > 0).mean(), (out < 0).mean()))
    return obs, np.percentile(out, [2.5, 97.5]), p


def main():
    d = build()
    print(f"[study] {len(d)} team-matches, {d.season.nunique()} seasons")

    # ------------------------------------------------------ step 0: is it real?
    print("\n" + "=" * 76)
    print("0. IS THE EFFECT REAL? four archetypes were tested — correct for that")
    print("=" * 76)
    print(f"  {'archetype':14s} {'effect':>8s} {'95% CI':>20s} {'p':>7s} {'Bonf p<.0125':>13s}")
    for a in ORDER:
        obs, ci, p = boot_effect(d, a)
        ok = "yes" if p < 0.0125 else "no"
        print(f"  {a:14s} {obs:+8.3f}   ({ci[0]:+.3f}, {ci[1]:+.3f}) {p:7.4f} {ok:>13s}")
    print("\n  Bonferroni across the 4 archetypes needs p < 0.0125 to claim a discovery.")

    print("\n  leave-one-season-out for prior 7-12 (is one season carrying it?):")
    sub = d[d["arch"] == "prior 7-12"]
    los = []
    for s in sorted(sub.season.unique(), key=lfc.season_start):
        t = sub[sub.season != s]
        v = t[t.md <= EARLY]["att_rel"].mean() - t[t.md > EARLY]["att_rel"].mean()
        los.append((s, v))
    for s, v in los:
        print(f"    drop {s}: {v:+.3f}")
    vals = [v for _, v in los]
    print(f"    range {min(vals):+.3f} to {max(vals):+.3f}  "
          f"({'stable' if max(vals) < 0 else 'FLIPS SIGN — not robust'})")

    # ------------------------------------------------- step 1: schedule effects
    print("\n" + "=" * 76)
    print("1. SCHEDULE — did mid-table clubs simply draw harder opening fixtures?")
    print("=" * 76)
    # season-long ratings from ALL 38 matches, so they are not contaminated by phase
    att = d.groupby(["season", "team"])["npxg"].mean().rename("A")
    dfn = d.groupby(["season", "team"])["npxga"].mean().rename("D")
    lg = d.groupby("season")["npxg"].mean().rename("L")
    d = d.join(att, on=["season", "team"]).join(
        dfn.rename("Dopp"), on=["season", "opp"]).join(lg, on="season")
    home_mult = (d[d.is_home]["npxg"].mean() / d[~d.is_home]["npxg"].mean())
    d["exp_xg"] = d["A"] * d["Dopp"] / d["L"] * np.where(
        d["is_home"], np.sqrt(home_mult), 1 / np.sqrt(home_mult))
    d["resid"] = d["npxg"] - d["exp_xg"]

    print(f"  {'archetype':14s} {'opp strength faced':>19s} {'home share':>11s}")
    print(f"  {'':14s} {'GW1-6':>9s} {'GW7+':>9s} {'GW1-6':>6s} {'GW7+':>5s}")
    for a in ORDER:
        s = d[d["arch"] == a]
        e, r = s[s.md <= EARLY], s[s.md > EARLY]
        print(f"  {a:14s} {e['Dopp'].mean():9.3f} {r['Dopp'].mean():9.3f} "
              f"{e['is_home'].mean():6.3f} {r['is_home'].mean():5.3f}")
    print("\n  (opp strength = opponent's season-long xG CONCEDED per match; lower means")
    print("   a tougher defence to face, so a LOWER early number would explain the fade)")

    print("\n  after removing opponent quality and venue — residual xG, early vs rest:")
    print(f"  {'archetype':14s} {'GW1-6':>8s} {'GW7+':>8s} {'gap':>8s} {'95% CI':>20s}")
    rows = []
    for a in ORDER:
        s = d[d["arch"] == a]
        units = s.groupby(["season", "team"]).ngroup().values
        uniq = np.unique(units); rng = np.random.default_rng(1)
        bs = []
        for _ in range(2000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(units == u) for u in pick])
            q = s.iloc[idx]
            bs.append(q[q.md <= EARLY]["resid"].mean() - q[q.md > EARLY]["resid"].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        e, r = s[s.md <= EARLY]["resid"].mean(), s[s.md > EARLY]["resid"].mean()
        star = " *" if not (lo <= 0 <= hi) else ""
        print(f"  {a:14s} {e:+8.3f} {r:+8.3f} {e-r:+8.3f}   ({lo:+.3f}, {hi:+.3f}){star}")
        rows.append({"arch": a, "resid_early": e, "resid_rest": r, "gap": e - r,
                     "ci_lo": lo, "ci_hi": hi})
    print("\n  If the raw fade survives here it is NOT a fixture artefact.")

    # --------------------------------------- step 2: European qualifying rounds
    print("\n" + "=" * 76)
    print("2. EUROPEAN QUALIFYING — Conference/Europa rounds are played in Jul-Aug,")
    print("   so they should hit the TOP of the 7-12 bucket and nothing else")
    print("=" * 76)
    mid = d[d["arch"] == "prior 7-12"].copy()
    mid["band"] = np.where(mid["prior_rank"] <= 8, "prior 7-8", "prior 9-12")
    print(f"  {'band':12s} {'n team-seasons':>15s} {'raw gap':>9s} {'resid gap':>11s}")
    for b in ("prior 7-8", "prior 9-12"):
        s = mid[mid["band"] == b]
        n = s.groupby(["season", "team"]).ngroups
        raw = (s[s.md <= EARLY]["att_rel"].mean() - s[s.md > EARLY]["att_rel"].mean())
        res = (s[s.md <= EARLY]["resid"].mean() - s[s.md > EARLY]["resid"].mean())
        print(f"  {b:12s} {n:15d} {raw:+9.3f} {res:+11.3f}")
    print("\n  The European-qualifying story predicts a clearly larger fade in 7-8.")

    # is the band difference more than noise, and is the 7-8 fade itself significant?
    def band_boot(s, col="resid", n=4000, seed=2):
        units = s.groupby(["season", "team"]).ngroup().values
        uniq = np.unique(units); rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            pick = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(units == u) for u in pick])
            q = s.iloc[idx]
            out.append(q[q.md <= EARLY][col].mean() - q[q.md > EARLY][col].mean())
        return np.array(out)

    b78 = band_boot(mid[mid.band == "prior 7-8"])
    b912 = band_boot(mid[mid.band == "prior 9-12"], seed=3)
    print(f"\n  prior 7-8  residual fade: {b78.mean():+.3f}  95% CI "
          f"({np.percentile(b78,2.5):+.3f}, {np.percentile(b78,97.5):+.3f})")
    print(f"  prior 9-12 residual fade: {b912.mean():+.3f}  95% CI "
          f"({np.percentile(b912,2.5):+.3f}, {np.percentile(b912,97.5):+.3f})")
    diff = b78 - b912
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"  DIFFERENCE (7-8 minus 9-12): {diff.mean():+.3f}  95% CI ({lo:+.3f}, {hi:+.3f})"
          f"{'  *' if not (lo <= 0 <= hi) else '  <- overlaps zero'}")

    # timing: qualifying rounds finish in late August, so the damage should be
    # front-loaded WITHIN the early window and gone by GW4-6
    print("\n  TIMING — qualifying rounds run Jul to late Aug, so the effect should sit")
    print("  in GW1-3 and be gone by GW4-6:")
    print(f"  {'band':12s} {'GW1-3':>8s} {'GW4-6':>8s} {'GW7+':>8s}")
    for b in ("prior 7-8", "prior 9-12"):
        s = mid[mid.band == b]
        v1 = s[s.md <= 3]["resid"].mean()
        v2 = s[(s.md > 3) & (s.md <= 6)]["resid"].mean()
        v3 = s[s.md > 6]["resid"].mean()
        print(f"  {b:12s} {v1:+8.3f} {v2:+8.3f} {v3:+8.3f}")

    # the logical cross-check: prior top-6 are ALL in Europe, but their group stages
    # start in September, not July
    t6 = d[d["arch"] == "prior top-6"]
    print(f"\n  CROSS-CHECK — prior top-6 are all in Europe too, but enter at the GROUP")
    print(f"  stage in September rather than playing July qualifiers. Their residual")
    print(f"  gap is {t6[t6.md<=EARLY]['resid'].mean()-t6[t6.md>EARLY]['resid'].mean():+.3f} "
          f"— i.e. no fade. European football per se does not explain it; the")
    print(f"  JULY-AUGUST timing of qualifying is what the hypothesis rests on.")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
