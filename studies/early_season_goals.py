from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
early_season_goals.py — are the opening gameweeks a different scoring environment?
==================================================================================
GOALS, not xG. The board's team layer is xG-driven, so if actual goals in the opening
weeks systematically diverge from the xG baseline, every projection in the horizon the
board actually publishes (GW1-10) is shifted.

    lam = exp(mu + home + att - def)          bayes_model, prior mean log(LEAGUE_MU=1.40)
    attacking returns scale with lam_for / LEAGUE_MU
    clean sheets and goals conceded are Poisson(lam_against)

So a league-level early-season scoring factor is a ONE-PARAMETER change: multiply lam in
the affected gameweeks. That is the integration point this study is aimed at.

DESIGN — why the comparison is WITHIN season
--------------------------------------------
League scoring drifted from 2.57 goals/game (14/15) to 3.28 (23/24). Pooling raw goals
across twelve seasons would confound era with matchday. Every comparison here is
early-window against THE SAME SEASON's remainder, and the unit of analysis is the SEASON
(n=12 paired observations), not the match. Twelve paired seasons is a small but honest
sample; 4,560 matches would be a fake one, because matches within a season are not
independent draws of the thing being measured.

Windows are cumulative from matchday 1 (first 3, 6, 12) because that is how a manager
plans, with matchdays 13+ as the within-season baseline.

Run:  python studies/early_season_goals.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest

SEASONS = ["%02d%02d" % (y % 100, (y + 1) % 100) for y in range(2014, 2026)]
WINDOWS = (3, 6, 12)
BASE_FROM = 13
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "early_season_goals.csv")


def build():
    tm = sd_ingest.understat_team_match(SEASONS)
    tm["team"] = tm["team"].map(sd_ingest.normalise_team)
    tm["opp"] = tm["opp"].map(sd_ingest.normalise_team)
    tm["date"] = pd.to_datetime(tm["date"], errors="coerce")
    tm = tm.sort_values(["season", "team", "date"])
    tm["md"] = tm.groupby(["season", "team"]).cumcount() + 1
    # goals against: the opponent's goals in the same match
    ga = tm.set_index(["season", "match_id", "team"])["goals"]
    tm["ga"] = ga.reindex(
        pd.MultiIndex.from_arrays([tm["season"], tm["match_id"], tm["opp"]])).values
    tm["gf"] = tm["goals"]
    tm["cs"] = (tm["ga"] == 0).astype(float)
    tm["total_goals"] = tm["gf"] + tm["ga"]
    tm["pts"] = np.where(tm.gf > tm.ga, 3, np.where(tm.gf == tm.ga, 1, 0))
    tm["xg_all"] = tm["xg"]                      # penalty-inclusive, matches goals
    tm["finishing"] = tm["gf"] - tm["xg_all"]    # over/under-performance of xG
    return tm.dropna(subset=["ga"])


def paired(tm, col, agg="mean"):
    """Per-season early-window value and same-season baseline (md>=13). Returns a frame
    with one row per (season, window) plus the paired difference and ratio."""
    rows = []
    for season, g in tm.groupby("season"):
        base = g[g["md"] >= BASE_FROM][col].agg(agg)
        for n in WINDOWS:
            early = g[g["md"] <= n][col].agg(agg)
            rows.append({"season": season, "window": n, "early": early,
                         "baseline": base, "diff": early - base,
                         "ratio": early / base if base else np.nan})
    return pd.DataFrame(rows)


def summarise(P, label, unit="", pct=False):
    print(f"\n{label}")
    print(f"  {'window':>8s} {'early':>8s} {'md13+':>8s} {'diff':>8s} "
          f"{'95% CI on diff':>20s} {'ratio':>7s} {'seasons up':>11s}")
    out = []
    for n in WINDOWS:
        s = P[P["window"] == n]
        d = s["diff"].values
        # paired across seasons: bootstrap the mean of 12 season-level differences
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = " *" if not (lo <= 0 <= hi) else ""
        up = int((d > 0).sum())
        print(f"  first {n:2d} {s['early'].mean():8.3f} {s['baseline'].mean():8.3f} "
              f"{d.mean():+8.3f}   ({lo:+.3f}, {hi:+.3f}) "
              f"{s['ratio'].mean():7.3f} {up:6d}/{len(d)}{sig}")
        out.append({"metric": label, "window": n, "early": s["early"].mean(),
                    "baseline": s["baseline"].mean(), "diff": d.mean(),
                    "ci_lo": lo, "ci_hi": hi, "ratio": s["ratio"].mean(),
                    "seasons_up": up, "n_seasons": len(d)})
    return out


def main():
    tm = build()
    print(f"[study] {len(tm)} team-matches, {tm.season.nunique()} seasons "
          f"({SEASONS[0]}-{SEASONS[-1]}); baseline = matchday {BASE_FROM}+")
    print(f"        unit of analysis = SEASON (n={tm.season.nunique()} paired), not match")

    rows = []
    rows += summarise(paired(tm, "total_goals"),
                      "TOTAL GOALS PER MATCH (both teams)")
    rows += summarise(paired(tm, "xg_all"),
                      "xG PER TEAM-MATCH (penalty-inclusive)")
    rows += summarise(paired(tm, "finishing"),
                      "FINISHING: goals minus xG per team-match")
    rows += summarise(paired(tm, "cs"),
                      "CLEAN-SHEET RATE")
    rows += summarise(paired(tm[tm.is_home], "gf"),
                      "HOME GOALS PER MATCH")
    rows += summarise(paired(tm[~tm.is_home], "gf"),
                      "AWAY GOALS PER MATCH")

    # Home advantage. Home goals and away goals each move the same way but neither
    # clears zero on its own; the DIFFERENCE is the quantity of interest and is far
    # better determined, because differencing within a season removes that season's
    # scoring level entirely.
    print("\nHOME ADVANTAGE (home goals minus away goals per match) — paired by season")
    print(f"  {'window':>8s} {'early':>8s} {'md13+':>8s} {'diff':>8s} "
          f"{'95% CI on diff':>20s} {'log h early':>12s} {'seasons down':>13s}")
    rng = np.random.default_rng(0)
    for n in WINDOWS:
        e, b, lh = [], [], []
        for season, g in tm.groupby("season"):
            ew = g[g["md"] <= n]; bw = g[g["md"] >= BASE_FROM]
            eh, ea = ew[ew.is_home]["gf"].mean(), ew[~ew.is_home]["gf"].mean()
            bh, ba = bw[bw.is_home]["gf"].mean(), bw[~bw.is_home]["gf"].mean()
            e.append(eh - ea); b.append(bh - ba)
            lh.append(np.log(eh / ea) if ea > 0 else np.nan)
        e, b, lh = np.array(e), np.array(b), np.array(lh)
        d = e - b
        bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = " *" if not (lo <= 0 <= hi) else ""
        print(f"  first {n:2d} {e.mean():8.3f} {b.mean():8.3f} {d.mean():+8.3f} "
              f"  ({lo:+.3f}, {hi:+.3f}) {np.nanmean(lh):12.3f} "
              f"{int((d < 0).sum()):8d}/{len(d)}{sig}")
        rows.append({"metric": "HOME ADVANTAGE (goals)", "window": n,
                     "early": e.mean(), "baseline": b.mean(), "diff": d.mean(),
                     "ci_lo": lo, "ci_hi": hi,
                     "ratio": e.mean() / b.mean() if b.mean() else np.nan,
                     "seasons_up": int((d > 0).sum()), "n_seasons": len(d)})
    lb = np.log(tm[tm.is_home & (tm.md >= BASE_FROM)]["gf"].mean()
                / tm[~tm.is_home & (tm.md >= BASE_FROM)]["gf"].mean())
    print(f"\n  baseline log home advantage (md13+) = {lb:.3f}; "
          f"bayes_model home_prior = 0.184 (season-average, calibrated)")
    print("  ^ compare 'log h early' against these — the model applies ONE value to "
          "every gameweek.")

    # ---- matchups: does the archetype matrix look different early? ----
    print("\nMATCHUP TYPES — total match goals by archetype pairing, early vs later")
    import team_archetype_study as tas
    arch = tas.archetypes(tm)
    t2 = tm.merge(arch[["season", "team", "arch"]], on=["season", "team"], how="left")
    t2 = t2.merge(arch[["season", "team", "arch"]].rename(
        columns={"team": "opp", "arch": "opp_arch"}), on=["season", "opp"], how="left")
    t2 = t2[t2["arch"].notna() & t2["opp_arch"].notna()
            & (t2["arch"] != "(no prior season)") & (t2["opp_arch"] != "(no prior season)")]
    order = ["prior top-6", "prior 7-12", "prior 13-20", "promoted"]
    for lab, sub in (("first 6 matchdays", t2[t2["md"] <= 6]),
                     ("matchday 13+", t2[t2["md"] >= BASE_FROM])):
        m = sub.pivot_table(index="arch", columns="opp_arch", values="total_goals",
                            aggfunc="mean").reindex(order)[order]
        print(f"\n  {lab} (mean total goals in the match):")
        print(m.round(2).to_string())
    print("\n  cell counts (first 6):")
    print(t2[t2["md"] <= 6].pivot_table(index="arch", columns="opp_arch",
                                        values="total_goals", aggfunc="size")
          .reindex(order)[order].to_string())

    # Sixteen cells x two windows invites a false positive. Test ONE hypothesis
    # instead: does archetype MISMATCH drive goals harder early than later? Rank the
    # archetypes 1-4 and regress total goals on the absolute gap.
    print("\n  MISMATCH TEST — total goals ~ |archetype gap|, early vs later")
    rank = {a: i for i, a in enumerate(order)}
    t2 = t2.assign(gap=(t2["arch"].map(rank) - t2["opp_arch"].map(rank)).abs())
    for lab, sub in (("first 6 matchdays", t2[t2["md"] <= 6]),
                     ("matchday 13+", t2[t2["md"] >= BASE_FROM])):
        import late_form_carryover as lfc
        b, se, r2 = lfc.ols([sub["gap"]], sub["total_goals"].values)
        # cluster the CI by season, since matches within a season are not independent
        seasons = sub["season"].unique()
        bs = []
        for _ in range(2000):
            pick = rng.choice(seasons, len(seasons), replace=True)
            s2 = pd.concat([sub[sub.season == s] for s in pick])
            bb, _, _ = lfc.ols([s2["gap"]], s2["total_goals"].values)
            bs.append(bb[1])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"    {lab:20s} slope {b[1]:+.4f} goals per rank-gap  "
              f"95% CI ({lo:+.3f}, {hi:+.3f})   n={len(sub)}")
    print("    ^ a steeper early slope means mismatches blow out more in the opening weeks")

    # the standout cell, tested directly and clustered by season
    print("\n  STANDOUT CELL — prior top-6 vs promoted, first 6 vs matchday 13+")
    cell = t2[((t2["arch"] == "prior top-6") & (t2["opp_arch"] == "promoted"))
              | ((t2["arch"] == "promoted") & (t2["opp_arch"] == "prior top-6"))]
    e = cell[cell["md"] <= 6]; l = cell[cell["md"] >= BASE_FROM]
    seasons = cell["season"].unique()
    bs = []
    for _ in range(4000):
        pick = rng.choice(seasons, len(seasons), replace=True)
        ee = pd.concat([e[e.season == s] for s in pick])["total_goals"]
        ll = pd.concat([l[l.season == s] for s in pick])["total_goals"]
        if len(ee) and len(ll):
            bs.append(ee.mean() - ll.mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    d = e["total_goals"].mean() - l["total_goals"].mean()
    print(f"    early {e['total_goals'].mean():.2f} (n={len(e)}) vs later "
          f"{l['total_goals'].mean():.2f} (n={len(l)})   diff {d:+.2f}  "
          f"95% CI ({lo:+.2f}, {hi:+.2f})"
          f"{'  *' if not (lo <= 0 <= hi) else '  (not significant)'}")
    print("    NB: one cell chosen AFTER seeing the matrix — treat as a hypothesis,")
    print("    not a finding, however the interval falls.")

    # dispersion — FPL cares about the tail, not just the mean
    print("\nDISPERSION of total goals per match (blowouts and blanks)")
    print(f"  {'window':>8s} {'sd':>7s} {'P(0-1 gls)':>11s} {'P(4+ gls)':>10s}")
    for n in list(WINDOWS) + [None]:
        sub = tm[tm["md"] <= n] if n else tm[tm["md"] >= BASE_FROM]
        lab = f"first {n:2d}" if n else "  md13+ "
        print(f"  {lab} {sub['total_goals'].std():7.3f} "
              f"{(sub['total_goals'] <= 1).mean():11.3f} "
              f"{(sub['total_goals'] >= 4).mean():10.3f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
