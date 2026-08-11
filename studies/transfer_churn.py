from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
transfer_churn.py — does heavy transfer activity cost you, and when in the season?
==================================================================================
Transaction counts (arrivals + departures) per club-season from transfermarkt, 2014/15 to
2025/26, against league outcome measured MONTH BY MONTH.

TWO CONFOUNDS THAT WOULD OTHERWISE MANUFACTURE A RESULT
--------------------------------------------------------
1. PROMOTED CLUBS churn enormously and are weak. Left in, they alone produce a large
   "churn causes bad results" effect that is really just "promoted clubs are worse".
   Excluded from the primary test; reported separately.
2. CHURN IS NOT MONOTONE IN QUALITY. Chelsea made 42 transactions in 14/15 and Man City
   31 in 16/17 — the biggest clubs trade heavily too, mostly loans and squad churn that
   never touches the first eleven. So churn is high at BOTH ends of the table. Every model
   here conditions on prior-season strength, without which the raw correlation is
   uninterpretable.

The estimand is the interaction: does churn hurt MORE in the opening months than later?
That is the question that would justify a gameweek-specific adjustment, and it is
separable from any level effect.

DATA PROVENANCE — read before trusting
---------------------------------------
data/transfer_counts.csv is transcribed from transfermarkt via an LLM summarisation step,
one page per season. Transcription errors are real: one page returned a header labelled
"2024/25" for the 2025/26 season (the club list gave it away). So `validate()` checks
every season's club set against Understat's for the same season and refuses to proceed on
a mismatch — a wrong-season page cannot pass that check silently.

Independently cross-checked against squad turnover computed from the LOCAL vaastav data
(players who actually appeared, keyed on permanent player_code) for the seasons where the
club naming there is reliable. If the two disagree, believe neither.

Run:  python studies/transfer_churn.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest
import early_season_goals as esg
import late_form_carryover as lfc
import team_archetype_study as tas

OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "transfer_churn.csv")
MONTH_ORDER = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
MONTH_NAME = {8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
              1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May"}


def load_transfers(tm):
    t = pd.read_csv(config.TRANSFER_COUNTS)
    t["season"] = t["season"].astype(str).str.zfill(4)
    t["club"] = t["club"].map(sd_ingest.normalise_team)
    t["moves"] = t["arrivals"] + t["departures"]
    validate(t, tm)
    return t


def validate(t, tm):
    """A wrong-season page must not pass silently."""
    bad = []
    for s, g in t.groupby("season"):
        got = set(g["club"])
        want = set(tm[tm["season"] == s]["team"].map(sd_ingest.normalise_team))
        if got != want:
            bad.append((s, sorted(want - got), sorted(got - want)))
    if bad:
        msg = "; ".join(f"{s}: missing {m}, extra {e}" for s, m, e in bad)
        raise RuntimeError(
            f"transfer_counts.csv club sets do not match Understat: {msg}. "
            f"Most likely a season page was transcribed under the wrong label.")
    print(f"[validate] all {t.season.nunique()} seasons match Understat's club sets "
          f"exactly ({len(t)} club-seasons)")


def build_panel():
    tm = esg.build()
    tm["team"] = tm["team"].map(sd_ingest.normalise_team)
    tm["month"] = tm["date"].dt.month
    t = load_transfers(tm)

    # prior-season strength, the control that makes churn interpretable
    ts = tm.groupby(["season", "team"])["z"].mean().rename("strength").reset_index() \
        if "z" in tm.columns else None
    tm["diff"] = tm["gf"] - tm["ga"]
    g = tm.groupby("season")["diff"]
    tm["z"] = (tm["diff"] - g.transform("mean")) / g.transform("std")
    ts = tm.groupby(["season", "team"])["z"].mean().rename("strength").reset_index()

    order = sorted(tm["season"].unique(), key=lfc.season_start)
    prev = {s: order[i - 1] if i else None for i, s in enumerate(order)}
    ts_prev = ts.copy()
    ts_prev["season"] = ts_prev["season"].map({v: k for k, v in prev.items() if v})
    ts_prev = ts_prev.dropna(subset=["season"]).rename(columns={"strength": "prior"})

    m = tm.merge(t[["season", "club", "moves", "arrivals"]],
                 left_on=["season", "team"], right_on=["season", "club"], how="left")
    m = m.merge(ts_prev[["season", "team", "prior"]], on=["season", "team"], how="left")
    m["promoted"] = m["prior"].isna()
    # churn standardised WITHIN season, so league-wide changes in trading volume
    # (which have trended up) do not masquerade as club-level churn
    gg = m.groupby("season")["moves"]
    m["churn_z"] = (m["moves"] - gg.transform("mean")) / gg.transform("std")
    return m


def cross_check(t):
    """Correlate the scraped counts against squad turnover computed from the LOCAL
    vaastav data — players who actually appeared, keyed on permanent player_code.

    Restricted to 2021/22 onward: merged_gw.csv only carries a club NAME from 2020/21,
    and before that the only club key is a season-specific integer id that FPL reassigns,
    so cross-season squad comparison is impossible there. Two independent sources
    agreeing is the only real check on an LLM-transcribed table.
    """
    import fpl_history as fh
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    FPL_TO_FRAME = {"Man Utd": "Man United", "Spurs": "Tottenham",
                    "Nott'm Forest": "Nott'm Forest"}
    squads = {}
    for s in seasons:
        raw = fh._read_csv(_os.path.join(config.history(s), "players_raw.csv"))
        gw = fh._read_csv(_os.path.join(config.history(s), "gws", "merged_gw.csv"))
        if "team" not in gw.columns:
            continue
        code = dict(zip(raw["id"], raw["code"]))
        gw["player_code"] = gw["element"].map(code)
        g = gw[(gw["minutes"] > 0) & gw["player_code"].notna()]
        for club, sub in g.groupby("team"):
            club = FPL_TO_FRAME.get(str(club), str(club))
            squads[(s, club)] = set(sub["player_code"])

    def tag(s):                      # '2021-22' -> '2122'
        a, b = s.split("-"); return a[2:] + b

    rows = []
    for i in range(1, len(seasons)):
        p, c = seasons[i - 1], seasons[i]
        for (ss, club), cur in list(squads.items()):
            if ss != c or (p, club) not in squads:
                continue
            rows.append({"season": tag(c), "club": club,
                         "n_new_local": len(cur - squads[(p, club)])})
    L = pd.DataFrame(rows)
    if L.empty:
        print("\n[cross-check] local turnover unavailable — skipped")
        return
    j = L.merge(t, on=["season", "club"], how="inner")
    if len(j) < 20:
        print(f"\n[cross-check] only {len(j)} overlapping club-seasons — too few")
        return
    r = j["n_new_local"].corr(j["arrivals"])
    rs = j["n_new_local"].corr(j["arrivals"], method="spearman")
    print(f"\n[cross-check] scraped arrivals vs LOCAL squad turnover, n={len(j)} "
          f"club-seasons ({j.season.nunique()} seasons)")
    print(f"  Pearson {r:+.3f}   Spearman {rs:+.3f}   "
          f"mean scraped {j.arrivals.mean():.1f} vs local new players "
          f"{j.n_new_local.mean():.1f}")
    print("  transfermarkt counts every transaction (loans, youth, reserves); the local")
    print("  figure counts only players who went on to appear — so the LEVELS should")
    print("  differ and only the correlation is informative.")


def main():
    m = build_panel()
    try:
        cross_check(pd.read_csv(config.TRANSFER_COUNTS).assign(
            season=lambda d: d["season"].astype(str).str.zfill(4),
            club=lambda d: d["club"].map(sd_ingest.normalise_team)))
    except Exception as e:
        print(f"[cross-check] skipped: {type(e).__name__}: {e}")
    est = m[~m["promoted"]].copy()
    print(f"[study] {len(m)} team-matches; {len(est)} with a prior PL season "
          f"({est.groupby(['season','team']).ngroups} club-seasons)")

    # ---------------------------------------------------------- churn vs table
    print("\n" + "=" * 78)
    print("1. CHURN vs THE TABLE  (does trading a lot go with finishing worse?)")
    print("=" * 78)
    cs = (est.groupby(["season", "team"])
             .agg(moves=("moves", "first"), churn_z=("churn_z", "first"),
                  prior=("prior", "first"), pts=("pts", "sum"),
                  z=("z", "mean")).reset_index())
    cs["rank"] = cs.groupby("season")["pts"].rank(ascending=False)
    r_raw = cs["churn_z"].corr(cs["pts"])
    b, se, r2 = lfc.ols([cs["churn_z"]], cs["pts"].values)
    b2, se2, r22 = lfc.ols([cs["prior"], cs["churn_z"]], cs["pts"].values)
    print(f"  raw correlation churn vs points: {r_raw:+.3f}  (n={len(cs)})")
    print(f"  points ~ churn                  : {b[1]:+.3f} pts per 1 SD of churn "
          f"(se {se[1]:.3f}, t {b[1]/se[1]:+.2f})  r2={r2:.3f}")
    print(f"  points ~ prior strength + churn : {b2[2]:+.3f} pts per 1 SD of churn "
          f"(se {se2[2]:.3f}, t {b2[2]/se2[2]:+.2f})  r2={r22:.3f}")
    print(f"    (prior-strength coefficient {b2[1]:+.3f}, t {b2[1]/se2[1]:+.2f})")
    print("\n  ^ the second line is the interpretable one. Churn correlates with a weak")
    print("    prior season, so the unconditional number is mostly that.")

    print("\n  promoted clubs, excluded above (they churn hardest and are weakest):")
    pr = m[m["promoted"]].groupby(["season", "team"]).agg(
        moves=("moves", "first"), pts=("pts", "sum")).reset_index()
    print(f"    n={len(pr)}  mean moves {pr.moves.mean():.1f} vs "
          f"{cs.moves.mean():.1f} for established clubs;  mean points "
          f"{pr.pts.mean():.1f} vs {cs.pts.mean():.1f}")

    # ---------------------------------------------------------- by month
    print("\n" + "=" * 78)
    print("2. BY MONTH — is the churn penalty concentrated early?")
    print("=" * 78)
    print(f"  {'month':>6s} {'n':>5s} {'churn coef':>11s} {'se':>7s} {'t':>7s}")
    rows = []
    for mo in MONTH_ORDER:
        sub = est[est["month"] == mo]
        if len(sub) < 100:
            continue
        bb, ss, _ = lfc.ols([sub["prior"], sub["churn_z"]], sub["z"].values)
        rows.append({"month": MONTH_NAME[mo], "n": len(sub), "coef": bb[2],
                     "se": ss[2], "t": bb[2] / ss[2]})
        print(f"  {MONTH_NAME[mo]:>6s} {len(sub):5d} {bb[2]:+11.4f} {ss[2]:7.4f} "
              f"{bb[2]/ss[2]:+7.2f}")
    R = pd.DataFrame(rows)
    print("\n  outcome = season-standardised goal difference per match, controlling for")
    print("  prior-season strength. A negative coefficient = high-churn clubs do worse.")

    # formal test of the interaction, clustered by club-season
    est2 = est.copy()
    est2["early"] = (est2["month"].isin([8, 9])).astype(float)
    inter = est2["churn_z"] * est2["early"]
    bi, si, _ = lfc.ols([est2["prior"], est2["churn_z"], est2["early"], inter],
                        est2["z"].values)
    # cluster bootstrap over club-seasons
    rng = np.random.default_rng(0)
    units = est2.groupby(["season", "team"]).ngroup().values
    uniq = np.unique(units)
    bs = []
    for _ in range(600):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(units == u) for u in pick])
        s = est2.iloc[idx]
        bb, _, _ = lfc.ols([s["prior"], s["churn_z"], s["early"],
                            s["churn_z"] * s["early"]], s["z"].values)
        bs.append(bb[4])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n  INTERACTION churn x (Aug-Sep): {bi[4]:+.4f}  95% CI ({lo:+.4f}, {hi:+.4f})"
          f"{'  *' if not (lo <= 0 <= hi) else '  (not significant)'}")
    print("  (cluster bootstrap over club-seasons — matches within a club-season are")
    print("   not independent)")

    # ------------------------------------------------------------------
    # A null on a badly-measured regressor is not much of a null. The scraped count
    # correlates only ~0.5 with actual squad turnover, so attenuation alone could
    # flatten a real effect. Repeat on the LOCAL measure: the share of each club's
    # minutes played by players who did not appear for it last season. Fewer seasons,
    # but measured exactly rather than scraped, and it is the disruption that actually
    # reaches the pitch.
    print("\n" + "=" * 78)
    print("3. ROBUSTNESS — same test on directly-measured squad disruption")
    print("=" * 78)
    try:
        D = local_disruption()
    except Exception as e:
        print(f"  unavailable: {type(e).__name__}: {e}")
        D = None
    if D is not None and len(D) > 40:
        e2 = est.merge(D, on=["season", "team"], how="inner")
        gg = e2.groupby("season")["new_minute_share"]
        e2["disr_z"] = (e2["new_minute_share"] - gg.transform("mean")) / gg.transform("std")
        print(f"  {e2.groupby(['season','team']).ngroups} club-seasons, "
              f"{e2.season.nunique()} seasons")
        print(f"  {'month':>6s} {'n':>5s} {'disruption coef':>16s} {'se':>7s} {'t':>7s}")
        for mo in MONTH_ORDER:
            sub = e2[e2["month"] == mo]
            if len(sub) < 60:
                continue
            bb, ss, _ = lfc.ols([sub["prior"], sub["disr_z"]], sub["z"].values)
            print(f"  {MONTH_NAME[mo]:>6s} {len(sub):5d} {bb[2]:+16.4f} {ss[2]:7.4f} "
                  f"{bb[2]/ss[2]:+7.2f}")
        e2["early"] = e2["month"].isin([8, 9]).astype(float)
        bi2, si2, _ = lfc.ols([e2["prior"], e2["disr_z"], e2["early"],
                               e2["disr_z"] * e2["early"]], e2["z"].values)
        print(f"\n  INTERACTION disruption x (Aug-Sep): {bi2[4]:+.4f} (se {si2[4]:.4f}, "
              f"t {bi2[4]/si2[4]:+.2f})")
        bb, ss, _ = lfc.ols([e2["prior"], e2["disr_z"]], e2["z"].values)
        # The naive se above is computed over ~3,200 MATCH rows as if they were
        # independent. They are not — a club-season contributes 38 correlated matches
        # and only its single disruption value. Cluster over club-seasons.
        u = e2.groupby(["season", "team"]).ngroup().values
        uu = np.unique(u)
        bs2 = []
        for _ in range(600):
            pick = rng.choice(uu, len(uu), replace=True)
            idx = np.concatenate([np.flatnonzero(u == q) for q in pick])
            s = e2.iloc[idx]
            c, _, _ = lfc.ols([s["prior"], s["disr_z"]], s["z"].values)
            bs2.append(c[2])
        lo2, hi2 = np.percentile(bs2, [2.5, 97.5])
        print(f"\n  season-long disruption coefficient: {bb[2]:+.4f}")
        print(f"    naive se {ss[2]:.4f} (t {bb[2]/ss[2]:+.2f})  <- overstates certainty")
        print(f"    clustered 95% CI ({lo2:+.4f}, {hi2:+.4f})"
              f"{'  *' if not (lo2 <= 0 <= hi2) else '  (not significant)'}")
        print("\n  CAUSALITY WARNING on the season-long number: `new_minute_share`")
        print("  counts JANUARY arrivals too, and clubs sign in January BECAUSE the")
        print("  first half went badly. That is reverse causation, and it inflates any")
        print("  negative season-long coefficient. The by-month interaction — the actual")
        print("  question — is far less exposed, since it compares months within clubs.")

    R.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


def local_disruption():
    """Share of a club's minutes played by players who did not appear for it last
    season. Exact, local, and the version of 'churn' that actually reaches the pitch."""
    import fpl_history as fh
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    FPL_TO_FRAME = {"Man Utd": "Man United", "Spurs": "Tottenham"}
    mins = {}
    for s in seasons:
        raw = fh._read_csv(_os.path.join(config.history(s), "players_raw.csv"))
        gw = fh._read_csv(_os.path.join(config.history(s), "gws", "merged_gw.csv"))
        if "team" not in gw.columns:
            continue
        code = dict(zip(raw["id"], raw["code"]))
        gw["player_code"] = gw["element"].map(code)
        g = gw[(gw["minutes"] > 0) & gw["player_code"].notna()]
        for club, sub in g.groupby("team"):
            club = FPL_TO_FRAME.get(str(club), str(club))
            mins[(s, club)] = sub.groupby("player_code")["minutes"].sum()

    def tag(s):
        a, b = s.split("-"); return a[2:] + b

    rows = []
    for i in range(1, len(seasons)):
        p, c = seasons[i - 1], seasons[i]
        for (ss, club), cur in mins.items():
            if ss != c or (p, club) not in mins:
                continue
            new = set(cur.index) - set(mins[(p, club)].index)
            tot = cur.sum()
            rows.append({"season": tag(c), "team": club,
                         "new_minute_share": cur.reindex(list(new)).sum() / tot
                         if tot else np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
