from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
team_archetype_study.py — what kind of team, against what kind of opponent, when?
=================================================================================
Twelve Understat seasons (2014/15-2025/26, 9,120 team-matches) used as a TEAM-level
fixture predictor. Four questions, all aimed at the same end: knowing before a gameweek
which fixtures are likely to produce goals and which are likely to produce clean sheets.

  Q1  How do promoted sides and prior-season top-six fare across the season?
  Q2  Does offensive or defensive quality carry over to a new season?
  Q3  Which archetypes over/under-perform specifically in EARLY fixtures?
  Q4  Which opponent types drive high- and low-scoring games?

TWO RULES THIS STUDY OBEYS
--------------------------
1. NO LOOKAHEAD. Every archetype is built from information available BEFORE a ball is
   kicked: whether a club was promoted, and where it finished LAST season. Bucketing by
   how a team ends up doing this season would be leakage — it would "predict" GW3 using
   the final table. That is the difference between a usable fixture signal and a
   backtest that cannot be traded.

2. SEASON-RELATIVE UNITS. League scoring has drifted hard over these twelve seasons
   (2.57 goals/game in 14/15, 3.28 in 23/24). Everything is expressed as a RATIO to that
   season's league mean, so a 1.20 means "20% above the league that year", not "20%
   above 2014". Pooling raw goals across eras would confound era with archetype.

xG, not goals, wherever a rate is being estimated — 38 matches is far too few for goals
to be a stable measure of quality, which is the whole reason the model is xG-based.

Run:  python studies/team_archetype_study.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest

SEASONS = ["%02d%02d" % (y % 100, (y + 1) % 100) for y in range(2014, 2026)]
OUT_DIR = _os.path.dirname(_os.path.abspath(__file__))

# Phase boundaries by matchday. GW1-6 is the horizon the board actually projects.
PHASES = [(1, 6, "GW1-6 early"), (7, 19, "GW7-19 autumn"),
          (20, 30, "GW20-30 winter"), (31, 38, "GW31-38 run-in")]


def season_start(code):
    yy = int(code[:2])
    return (1900 + yy) if yy >= 90 else (2000 + yy)


def build_panel():
    tm = sd_ingest.understat_team_match(SEASONS, )
    tm["team"] = tm["team"].map(sd_ingest.normalise_team)
    tm["opp"] = tm["opp"].map(sd_ingest.normalise_team)
    tm["date"] = pd.to_datetime(tm["date"], errors="coerce")
    tm = tm.sort_values(["season", "team", "date"])
    tm["md"] = tm.groupby(["season", "team"]).cumcount() + 1      # matchday 1..38
    tm["gf"] = tm["goals"]
    tm["ga"] = tm.merge(
        tm[["season", "match_id", "team", "goals"]],
        left_on=["season", "match_id", "opp"], right_on=["season", "match_id", "team"],
        how="left", suffixes=("", "_o"))["goals_o"].values
    tm["pts"] = np.where(tm.gf > tm.ga, 3, np.where(tm.gf == tm.ga, 1, 0))
    tm["cs"] = (tm["ga"] == 0).astype(float)
    tm["phase"] = pd.cut(tm["md"], bins=[0] + [b for _, b, _ in PHASES],
                         labels=[lab for _, _, lab in PHASES])
    # season-relative scaling
    for c, out in (("npxg", "att_rel"), ("npxga", "def_rel")):
        mu = tm.groupby("season")[c].transform("mean")
        tm[out] = tm[c] / mu
    tm["tot_xg"] = tm["npxg"] + tm["npxga"]
    mu_t = tm.groupby("season")["tot_xg"].transform("mean")
    tm["tot_rel"] = tm["tot_xg"] / mu_t
    return tm


def archetypes(tm):
    """Pre-season-knowable label per (season, team): promoted, or prior-season finish."""
    fin = (tm.groupby(["season", "team"])
             .agg(pts=("pts", "sum"), gd=("gf", "sum")).reset_index())
    fin["rank"] = fin.groupby("season")["pts"].rank(ascending=False, method="first")
    order = sorted(tm["season"].unique(), key=season_start)
    prev = {s: order[i - 1] if i else None for i, s in enumerate(order)}

    rank_prev = fin.set_index(["season", "team"])["rank"].to_dict()
    rows = []
    for _, r in fin.iterrows():
        p = prev[r["season"]]
        pr = rank_prev.get((p, r["team"])) if p else None
        if p is None:
            lab, prank = "(no prior season)", np.nan
        elif pr is None:
            lab, prank = "promoted", np.nan
        elif pr <= 6:
            lab, prank = "prior top-6", pr
        elif pr <= 12:
            lab, prank = "prior 7-12", pr
        else:
            lab, prank = "prior 13-20", pr
        rows.append({"season": r["season"], "team": r["team"], "arch": lab,
                     "prior_rank": prank, "pts": r["pts"], "rank": r["rank"]})
    return pd.DataFrame(rows)


def main():
    tm = build_panel()
    arch = archetypes(tm)
    tm = tm.merge(arch[["season", "team", "arch", "prior_rank"]],
                  on=["season", "team"], how="left")
    tm = tm.merge(arch[["season", "team", "arch"]].rename(
        columns={"team": "opp", "arch": "opp_arch"}), on=["season", "opp"], how="left")
    live = tm[tm["arch"] != "(no prior season)"].copy()
    print(f"[study] {len(tm)} team-matches, {tm.season.nunique()} seasons; "
          f"{len(live)} usable once a prior season is required")

    # ------------------------------------------------------------------ Q1 + Q3
    print("\n" + "=" * 78)
    print("Q1/Q3  ATTACK & DEFENCE BY ARCHETYPE AND PHASE  (1.00 = league mean that season)")
    print("=" * 78)
    piv = live.pivot_table(index="arch", columns="phase",
                           values=["att_rel", "def_rel"], aggfunc="mean")
    att = piv["att_rel"].round(3); dfn = piv["def_rel"].round(3)
    order = ["prior top-6", "prior 7-12", "prior 13-20", "promoted"]
    print("\nxG CREATED (higher = better attack):")
    print(att.reindex(order).to_string())
    print("\nxG CONCEDED (LOWER = better defence):")
    print(dfn.reindex(order).to_string())

    early = PHASES[0][2]
    rest = [p[2] for p in PHASES[1:]]
    print("\nEARLY-SEASON EFFECT (GW1-6 minus rest-of-season):")
    eff = pd.DataFrame({
        "att_early_minus_rest": att[early] - att[rest].mean(axis=1),
        "def_early_minus_rest": dfn[early] - dfn[rest].mean(axis=1),
    }).reindex(order).round(3)
    eff["reading"] = np.where(
        eff.att_early_minus_rest > 0.02, "attacks BETTER early",
        np.where(eff.att_early_minus_rest < -0.02, "attacks WORSE early", "flat"))
    print(eff.to_string())
    att.reindex(order).to_csv(_os.path.join(OUT_DIR, "team_arch_attack_by_phase.csv"))
    dfn.reindex(order).to_csv(_os.path.join(OUT_DIR, "team_arch_defence_by_phase.csv"))

    # points and clean sheets, the FPL-facing versions
    print("\nPOINTS PER GAME and CLEAN-SHEET RATE by archetype and phase:")
    pp = live.pivot_table(index="arch", columns="phase", values="pts", aggfunc="mean")
    cs = live.pivot_table(index="arch", columns="phase", values="cs", aggfunc="mean")
    print("\n  points/game:");   print(pp.reindex(order).round(2).to_string())
    print("\n  clean-sheet rate:"); print(cs.reindex(order).round(3).to_string())

    # ------------------------------------------------------------------ Q2
    print("\n" + "=" * 78)
    print("Q2  DOES QUALITY CARRY OVER? season t -> t+1, surviving teams only")
    print("=" * 78)
    ts = (live.groupby(["season", "team"])
              .agg(att=("npxg", "mean"), dfn=("npxga", "mean")).reset_index())
    order_s = sorted(ts["season"].unique(), key=season_start)
    pairs = []
    for a, b in zip(order_s[:-1], order_s[1:]):
        x = ts[ts.season == a].set_index("team")
        y = ts[ts.season == b].set_index("team")
        for t in x.index.intersection(y.index):
            pairs.append({"att0": x.loc[t, "att"], "att1": y.loc[t, "att"],
                          "def0": x.loc[t, "dfn"], "def1": y.loc[t, "dfn"]})
    P = pd.DataFrame(pairs)
    res = {}
    for k, lab in (("att", "ATTACK  (npxG created/match)"),
                   ("def", "DEFENCE (npxG conceded/match)")):
        x, y = P[f"{k}0"].values, P[f"{k}1"].values
        slope = float(np.polyfit(x, y, 1)[0]); r = float(np.corrcoef(x, y)[0, 1])
        res[k] = (slope, r)
        print(f"  {lab}: slope={slope:.3f}  r={r:.3f}  r2={r*r:.3f}  n={len(P)}")
    print(f"\n  -> {'ATTACK' if res['att'][1] > res['def'][1] else 'DEFENCE'} is the more "
          f"persistent trait (r {max(res['att'][1], res['def'][1]):.3f} vs "
          f"{min(res['att'][1], res['def'][1]):.3f})")
    print("  Both are measured on SURVIVING teams only — relegation censors the worst,")
    print("  so these slopes understate true persistence at the bottom.")

    # --- is the early-season effect real? cluster bootstrap over team-seasons ---
    print("\n  significance of the early-season effect (bootstrap over team-seasons,")
    print("  resampling whole (season, team) units because a club's matches correlate):")
    rng = np.random.default_rng(0)
    units = live.groupby(["season", "team"]).ngroup().values
    uniq = np.unique(units)
    boot = {a: [] for a in order}
    for _ in range(400):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(units == u) for u in pick])
        s = live.iloc[idx]
        e = s[s.phase == early].groupby("arch")["att_rel"].mean()
        r = s[s.phase != early].groupby("arch")["att_rel"].mean()
        for a in order:
            if a in e.index and a in r.index:
                boot[a].append(e[a] - r[a])
    print(f"    {'archetype':14s} {'effect':>8s} {'95% CI':>18s}   {'n team-seasons':>14s}")
    for a in order:
        b = np.array(boot[a])
        lo, hi = np.percentile(b, [2.5, 97.5])
        n = live[live.arch == a].groupby(["season", "team"]).ngroups
        sig = "" if lo <= 0 <= hi else "  <- excludes 0"
        print(f"    {a:14s} {eff.loc[a,'att_early_minus_rest']:+8.3f} "
              f"  ({lo:+.3f}, {hi:+.3f})   {n:14d}{sig}")

    # --- Q2 caveat: are the two traits measured with equal reliability? ---
    print("\n  reliability check for Q2 (split each team-season's matches in half):")
    rel = {}
    for col, lab in (("npxg", "att"), ("npxga", "def")):
        rs = []
        for _ in range(60):
            h = live.copy()
            h["half"] = rng.integers(0, 2, len(h))
            g = h.groupby(["season", "team", "half"])[col].mean().unstack()
            g = g.dropna()
            r = float(np.corrcoef(g[0], g[1])[0, 1])
            rs.append(2 * r / (1 + r))            # Spearman-Brown to full length
        rel[lab] = float(np.mean(rs))
    print(f"    attack reliability {rel['att']:.3f}   defence reliability {rel['def']:.3f}")
    a_dis = res["att"][1] / np.sqrt(rel["att"] * rel["att"])
    d_dis = res["def"][1] / np.sqrt(rel["def"] * rel["def"])
    print(f"    disattenuated r:  attack {a_dis:.3f}   defence {d_dis:.3f}")
    print("    ^ if the raw gap survives this, attack really is the more persistent")
    print("      trait; if it collapses, the gap was just defence being measured worse.")

    # ------------------------------------------------------------------ Q4
    print("\n" + "=" * 78)
    print("Q4  SCORING ENVIRONMENT BY OPPONENT TYPE  (what fixtures produce goals?)")
    print("=" * 78)
    q4 = live[live["opp_arch"].notna() & (live["opp_arch"] != "(no prior season)")]
    m = q4.pivot_table(index="arch", columns="opp_arch", values="att_rel", aggfunc="mean")
    print("\nxG CREATED by row-team against column-opponent (1.00 = league mean):")
    print(m.reindex(order)[order].round(3).to_string())
    csm = q4.pivot_table(index="arch", columns="opp_arch", values="cs", aggfunc="mean")
    print("\nCLEAN-SHEET RATE for row-team against column-opponent:")
    print(csm.reindex(order)[order].round(3).to_string())
    m.reindex(order)[order].to_csv(_os.path.join(OUT_DIR, "team_arch_xg_matrix.csv"))
    csm.reindex(order)[order].to_csv(_os.path.join(OUT_DIR, "team_arch_cs_matrix.csv"))

    print("\nTOTAL match xG by opponent archetype (both teams combined, season-relative):")
    tot = (q4.groupby("opp_arch")["tot_rel"].agg(["mean", "count"])
             .reindex(order).round(3))
    tot.columns = ["total_xg_rel", "n"]
    print(tot.to_string())
    print("\n  -> above 1.00 means matches AGAINST this archetype are higher-scoring")
    print("     than the league average, counting both teams.")

    # home/away split, since that is half of what a fixture is
    print("\nHOME/AWAY, season-relative xG created:")
    ha = live.pivot_table(index="arch", columns="is_home", values="att_rel", aggfunc="mean")
    ha.columns = ["away", "home"]
    ha["home_edge"] = ha["home"] - ha["away"]
    print(ha.reindex(order).round(3).to_string())
    print(f"\n-> wrote 4 matrices to {OUT_DIR}")


if __name__ == "__main__":
    main()
