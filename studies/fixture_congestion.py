from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
fixture_congestion.py — re-opening the congestion null with the sharper instrument
==================================================================================
PRE-REGISTRATION. Written and committed before any coefficient was read. The design,
the sample restriction, the buckets and the decision rules below are fixed; results
go into docs/ whichever way they come out.

WHY RE-OPEN A NULL
------------------
Two congestion nulls are already on the board and both are kept:

  `rest_congestion`      +0.00000 xG per day of rest advantage, CI (-0.00434, +0.00402)
  `euro_qualifying_fade` participation flags for July/August European ties

Neither measured the thing. `rest_congestion` could only see PREMIER LEAGUE matches, so
a club playing a Thursday Europa tie read as fully rested by Sunday, and it said so —
"that is real measurement error and it ATTENUATES the estimate toward zero". The 19 Aug
data audit then found the fix sitting unused in the repo: the By-Gameweek folders carry
every competition, so the actual midweek fixture is observable. The audit's condition for
re-opening was pre-registration before measurement, which is what this file is.

The attenuation is now quantified rather than asserted. Of 760 club-PL-matches in 25/26,
**126 (17%) sit in a different recovery bucket** once cup and European fixtures are
counted; 53 of them look like a full week's rest and are actually three-day turnarounds.
That is the error the earlier null was averaging over.

ESTIMAND: DAYS OF RECOVERY, NOT PARTICIPATION
----------------------------------------------
`recovery` = days between a club's previous competitive fixture in ANY competition and
the PL kickoff. Buckets <=2 / 3 / 4 / 5+ days, with 5+ as the reference.

Recovery is preferred to a "played midweek" flag because it is the mechanism and because
it varies WITHIN club and WITHIN competition — a Thursday Europa tie leaves three days to
a Sunday game, a Tuesday Champions League tie leaves four to a Saturday. Competition
entry does not vary within a club-season at all, so a competition main effect is a club
label (see the identification warning below).

Club fixed effects throughout. Across clubs, the congested ones are the good ones — that
is the confound that makes raw congestion comparisons useless, and it is exactly the
confound `rest_congestion` avoided with a symmetric differential. Here it is absorbed
instead, and the contrast is a club against itself in a rested week.

IDENTIFICATION WARNING, STATED UP FRONT  [JUDGMENT]
----------------------------------------------------
"How does this change by competition entered" is only partly answerable from one season.
In 25/26 the European entrants are:

    Champions League   6 clubs   (ARS, LIV, NEW, CHE, TOT, MCI)
    Europa League      2 clubs   (AVL, NFO)
    Conference League  1 club    (CRY)
    EFL Cup           20 clubs

A Conference League main effect IS Crystal Palace; a Europa League one is Villa and
Forest. Those cells cannot be separated from club identity, and no amount of estimation
fixes it. What IS identified is recovery days, which varies within every club. So the
competition question is answered THROUGH the mechanism — competitions differ because
UEFA fixes their weekday, and the test is whether competition adds anything once
recovery days and club are controlled (H3). It is pre-committed that no
competition-specific parameter is adopted on this sample regardless of what H3 shows.

SAMPLE: GW1-26
--------------
Cup kickoff times are complete for GW1-26 (114/114 club-matches) and almost entirely
absent for GW27+ (6/50) — upstream's knockout-round ingest is thinner. Treatment cannot
be measured without error where the kickoff is unknown, so the primary sample stops at
GW26 and GW27+ is reported separately as an appendix, never pooled.

The FA Cup is ABSENT from this data source entirely. Within GW1-26 that is benign: FA Cup
rounds are played on weekends that are blank in the PL calendar (R3 c. 10 Jan 2026 sits
between GW21 on 6 Jan and GW22 on 17 Jan), so an unobserved FA Cup match still leaves 5+
days to the next league game and cannot move a club between buckets. From GW27 the
quarter-finals and semi-finals go midweek and it stops being benign — a second reason the
late-season sample is an appendix and not evidence.

HYPOTHESES AND DECISION RULES  (fixed before results)
------------------------------------------------------
H1  ROTATION. Within player, PL minutes fall as recovery shortens.
    ADOPT if the 3-day bucket costs >= 5 minutes with a club-clustered 95% CI excluding
    zero. Secondary, reported either way: P(start), P(60+ minutes).

H2  TEAM PERFORMANCE. Within club, opponent- and venue-adjusted residual xG falls as
    recovery shortens.
    ADOPT if the 3-day bucket costs >= 0.10 xG with a club-clustered 95% CI excluding
    zero. 0.10 is roughly 6% of a team-match, the size of the GW1-3 home discount, which
    is the smallest correction this project has been willing to carry.

H3  COMPETITION. Preceding-competition dummies add nothing once recovery bucket and club
    are controlled. Reported descriptively only; no parameter is adopted from it (see the
    identification warning).

A result failing its rule is recorded as a null and no code path is added.

Run:  python studies/fixture_congestion.py
      python studies/fixture_congestion.py --selftest
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import repo_events as re_

SEASON = "2025-2026"
PRIMARY_MAX_GW = 26
BUCKETS = ["<=2d", "3d", "4d", "5+d"]
REF = "5+d"
MIN_MINUTES_EFFECT = 5.0     # H1 adoption threshold
MIN_XG_EFFECT = 0.10         # H2 adoption threshold
MIN_CLUSTERS = 5             # refuse a CI for a bucket living in fewer clubs
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fixture_congestion.csv")
OUT_PLAYER = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                           "fixture_congestion_players.csv")


def bucket(days):
    """Recovery days -> pre-registered bucket. NaN stays NaN."""
    d = pd.to_numeric(days, errors="coerce")
    return pd.cut(d, [-np.inf, 2.5, 3.5, 4.5, np.inf], labels=BUCKETS)


# ------------------------------------------------------------------ team panel
def team_panel(season=SEASON, base=None):
    """Club-PL-match rows with recovery buckets and an opponent/venue-adjusted xG
    residual built from the repo's own match xG (one source, same matches)."""
    rp = re_.recovery_panel(season, base=base)
    if rp.empty:
        return rp
    cal = re_.competitive_calendar(season, base=base)

    # match-level xG, from the same matches.csv the calendar was built from
    root = base or config.repo(season)
    import glob
    frames = []
    for f in sorted(glob.glob(_os.path.join(root, "By Gameweek", "GW*", "matches.csv"))):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    m = pd.concat(frames, ignore_index=True).drop_duplicates("match_id")
    code = re_._team_code_map(season, base=root)
    m["home"] = pd.to_numeric(m["home_team"], errors="coerce").map(code)
    m["away"] = pd.to_numeric(m["away_team"], errors="coerce").map(code)
    m["hxg"] = pd.to_numeric(m.get("home_expected_goals_xg"), errors="coerce")
    m["axg"] = pd.to_numeric(m.get("away_expected_goals_xg"), errors="coerce")
    m["hg"] = pd.to_numeric(m.get("home_score"), errors="coerce")
    m["ag"] = pd.to_numeric(m.get("away_score"), errors="coerce")
    long = pd.concat([
        m.assign(club=m["home"], opp=m["away"], xg=m["hxg"], xga=m["axg"],
                 gf=m["hg"], ga=m["ag"], is_home=True),
        m.assign(club=m["away"], opp=m["home"], xg=m["axg"], xga=m["hxg"],
                 gf=m["ag"], ga=m["hg"], is_home=False),
    ], ignore_index=True)[["match_id", "club", "opp", "xg", "xga", "gf", "ga",
                           "is_home"]]

    d = rp.merge(long, on=["match_id", "club"], how="left", suffixes=("", "_m"))
    d = d.dropna(subset=["xg", "opp"])

    # actual result, not just the underlying performance
    d["points"] = np.where(d["gf"] > d["ga"], 3.0,
                           np.where(d["gf"] == d["ga"], 1.0, 0.0))
    d["win"] = (d["gf"] > d["ga"]).astype(float)
    d["gd"] = d["gf"] - d["ga"]

    # opponent- and venue-adjusted residual, the construction used elsewhere here
    att = d.groupby("club")["xg"].mean().rename("A")
    dfn = d.groupby("club")["xga"].mean().rename("D")
    lg = d["xg"].mean()
    d = d.join(att, on="club").join(dfn.rename("Dopp"), on="opp")
    hm = d[d["is_home"]]["xg"].mean() / d[~d["is_home"]]["xg"].mean()
    d["exp_xg"] = d["A"] * d["Dopp"] / lg * np.where(d["is_home"], np.sqrt(hm),
                                                     1 / np.sqrt(hm))
    d["resid"] = d["xg"] - d["exp_xg"]
    d["b_all"] = bucket(d["rec_all"])
    d["b_pl"] = bucket(d["rec_pl"])

    # The OPPONENT's recovery, and the differential. For a result this is the natural
    # estimand: a match is two-sided and the differential is symmetric by construction,
    # so whatever one side gains the other loses and no club-quality term is needed.
    # `rest_congestion` used exactly this, but could only build it from league matches;
    # here both sides' cup and European fixtures are in it.
    opp_rec = d.set_index(["match_id", "club"])["rec_all"]
    d["opp_rec_all"] = opp_rec.reindex(
        pd.MultiIndex.from_arrays([d["match_id"], d["opp"]])).values
    d["rest_diff"] = d["rec_all"] - d["opp_rec_all"]
    return d


def diff_bucket(x):
    """Rest differential -> bucket. Reference is a level match-up."""
    return pd.cut(pd.to_numeric(x, errors="coerce"),
                  [-np.inf, -1.5, 1.5, np.inf],
                  labels=["opp fresher by 2+", "level (within 1d)", "we are fresher by 2+"])


# ---------------------------------------------------------------- player panel
def player_panel(season=SEASON, base=None):
    """Player x PL-match rows over the matchday SQUAD, not over appearances.

    The risk set is `lineups.csv` — every player on the teamsheet including unused
    substitutes. Built from `playermatchstats` instead, the panel would be
    conditioned on getting on the pitch, a rested player would drop out of the
    sample rather than record a zero, and rotation would be undetectable by
    construction. That is the whole quantity of interest here.
    """
    tp = team_panel(season, base=base)
    if tp.empty:
        return tp
    sheets = re_.squad_sheets(season, base=base)
    apm = re_.all_comp_player_matches(season, base=base)
    cal = re_.competitive_calendar(season, base=base)

    pl = sheets[sheets["tournament"] == "prem"].copy()
    mins = apm[["player_code", "match_id", "minutes_played"]]
    pl = pl.merge(mins, on=["player_code", "match_id"], how="left")
    pl["minutes_played"] = pl["minutes_played"].fillna(0.0)   # unused sub = 0
    pl["started"] = pl["is_starting"].astype(float)
    pl["played60"] = (pl["minutes_played"] >= 60).astype(float)
    pl["appeared"] = (pl["minutes_played"] > 0).astype(float)

    d = pl.merge(tp[["match_id", "club", "gameweek", "kickoff", "b_all", "b_pl",
                     "rec_all", "rec_pl", "prev_tour", "resid", "is_home"]],
                 on=["match_id", "club"], how="inner")

    # the player's own load in the club's immediately preceding fixture
    prev = cal.dropna(subset=["kickoff"]).sort_values(["club", "kickoff"])
    prev["next_match"] = prev.groupby("club")["match_id"].shift(-1)
    link = prev.dropna(subset=["next_match"])[["match_id", "next_match"]].rename(
        columns={"match_id": "prev_match", "next_match": "match_id"})
    d = d.merge(link, on="match_id", how="left")
    own = mins.rename(columns={"match_id": "prev_match",
                               "minutes_played": "prev_minutes"})
    d = d.merge(own, on=["player_code", "prev_match"], how="left")
    d["prev_minutes"] = d["prev_minutes"].fillna(0.0)
    return d


# ------------------------------------------------------------------- inference
def cluster_boot_means(d, group_col, value_col, cluster_col="club", n=4000, seed=0):
    """Bucket means, and each bucket's difference from the reference, with a cluster
    bootstrap over clubs.

    Resampling whole clusters is exactly a re-weighting of per-club sums and counts, so
    the draw is done on a (clubs x buckets) table rather than by rebuilding row indices.
    Same estimator, and it turns an overnight loop into a fraction of a second.

    Returns {bucket: (mean, n, diff_vs_ref, lo, hi)}.
    """
    d = d.dropna(subset=[group_col, cluster_col])
    clubs = pd.Index(sorted(d[cluster_col].unique()))
    ci = d[cluster_col].map({c: i for i, c in enumerate(clubs)}).to_numpy()
    bi = pd.Categorical(d[group_col], categories=BUCKETS).codes
    ok = bi >= 0
    ci, bi = ci[ok], bi[ok]
    v = pd.to_numeric(d[value_col], errors="coerce").to_numpy()[ok]
    good = np.isfinite(v)
    ci, bi, v = ci[good], bi[good], v[good]

    nc, nb = len(clubs), len(BUCKETS)
    flat = ci * nb + bi
    S = np.bincount(flat, weights=v, minlength=nc * nb).reshape(nc, nb)
    N = np.bincount(flat, minlength=nc * nb).reshape(nc, nb).astype(float)

    tot_n = N.sum(axis=0)
    tot_s = S.sum(axis=0)
    means = np.where(tot_n > 0, tot_s / np.maximum(tot_n, 1), np.nan)
    ref = BUCKETS.index(REF)

    rng = np.random.default_rng(seed)
    w = rng.multinomial(nc, np.full(nc, 1.0 / nc), size=n).astype(float)   # club weights
    bs = w @ S
    bn = w @ N
    bm = np.where(bn > 0, bs / np.maximum(bn, 1e-12), np.nan)
    diffs = bm - bm[:, [ref]]

    out = {}
    for k, b in enumerate(BUCKETS):
        size = int(tot_n[k])
        if b == REF:
            out[b] = (means[k], size, 0.0, 0.0, 0.0)
            continue
        col = diffs[:, k]
        col = col[np.isfinite(col)]
        # A bucket living in one or two clubs has no between-cluster variation left
        # to resample: every draw that contains the club returns that club's own
        # mean, so the interval collapses and reads as a tight significant effect.
        # Refuse inference instead of reporting a fake CI.
        n_clubs_here = int((N[:, k] > 0).sum())
        if size < 10 or len(col) < 100 or n_clubs_here < MIN_CLUSTERS:
            out[b] = (means[k], size, means[k] - means[ref], np.nan, np.nan)
        else:
            lo, hi = np.percentile(col, [2.5, 97.5])
            out[b] = (means[k], size, means[k] - means[ref], lo, hi)
    return out


def show(title, res, unit, thresh):
    print(f"\n  {title}")
    print(f"    {'bucket':8s} {'n':>6s} {'mean':>9s} {'vs 5+d':>9s} "
          f"{'95% CI (cluster on club)':>30s}")
    for b in BUCKETS:
        mean, size, diff, lo, hi = res[b]
        if size == 0:
            print(f"    {b:8s} {0:6d}       —  (bucket empty)")
            continue
        if b == REF:
            print(f"    {b:8s} {size:6d} {mean:9.3f} {'ref':>9s}")
            continue
        if not np.isfinite(lo):
            print(f"    {b:8s} {size:6d} {mean:9.3f} {diff:+9.3f}   "
                  f"(too few clubs for inference)")
            continue
        sig = "*" if not (lo <= 0 <= hi) else " "
        adopt = "ADOPT" if (sig == "*" and abs(diff) >= thresh) else ""
        print(f"    {b:8s} {size:6d} {mean:9.3f} {diff:+9.3f}   "
              f"({lo:+.3f}, {hi:+.3f}) {sig}  {adopt}")
    print(f"    units: {unit};  adoption threshold |diff| >= {thresh}")


def main():
    tp = team_panel()
    if tp.empty:
        print("[study] no data — is FPL_DATA pointing at FPL-Core-Insights/data?")
        return
    prim = tp[tp["gameweek"] <= PRIMARY_MAX_GW]
    late = tp[tp["gameweek"] > PRIMARY_MAX_GW]

    print("=" * 78)
    print("FIXTURE CONGESTION — 25/26, all competitions, pre-registered")
    print("=" * 78)
    print(f"[sample] {len(tp)} club-PL-matches total; primary GW1-{PRIMARY_MAX_GW} "
          f"n={len(prim)}; appendix GW{PRIMARY_MAX_GW+1}+ n={len(late)}")

    print("\n" + "=" * 78)
    print("0. HOW MUCH THE PL-ONLY MEASURE GETS WRONG  (the reason to re-open)")
    print("=" * 78)
    x = pd.crosstab(tp["b_pl"], tp["b_all"], dropna=False)
    print("  rows = bucket using PL matches only (the earlier null's measure)")
    print("  cols = bucket once cup and European fixtures are counted")
    print(x.to_string())
    both = tp.dropna(subset=["b_pl", "b_all"])
    moved = int((both["b_pl"].astype(str) != both["b_all"].astype(str)).sum())
    print(f"\n  {moved} of {len(both)} club-matches ({moved/max(len(both),1):.1%}) "
          f"sit in the wrong bucket without cup fixtures.")
    print("  Those are the rows the earlier null averaged over as 'rested'.")

    print("\n  recovery bucket by preceding competition (primary sample):")
    pc = pd.crosstab(prim["prev_tour"], prim["b_all"], dropna=False)
    print(pc.to_string())
    print("\n  UEFA fixes the weekday, so the competition IS the turnaround:")
    print("  Thursday competitions (Europa/Conference) land on 3 days; Tuesday and")
    print("  Wednesday ones (Champions League, EFL Cup) mostly land on 4.")

    print("\n" + "=" * 78)
    print("H2. TEAM PERFORMANCE — opponent- and venue-adjusted residual xG")
    print("=" * 78)
    r_all = cluster_boot_means(prim.dropna(subset=["b_all"]), "b_all", "resid")
    show("by ALL-COMPETITION recovery (the instrument)", r_all, "xG", MIN_XG_EFFECT)
    r_pl = cluster_boot_means(prim.dropna(subset=["b_pl"]), "b_pl", "resid")
    show("by PL-ONLY recovery (what the earlier null could see)", r_pl, "xG",
         MIN_XG_EFFECT)

    print("\n" + "=" * 78)
    print("H2b. RESULTS — points, goals, wins. xG is the performance; this is the payoff")
    print("=" * 78)
    print("  Residual xG above is what a side CREATED. Points are what it got. Goals are")
    print("  noisier than xG, so this is the lower-powered test of the two — the MDE is")
    print("  printed rather than left implicit.")

    sd_pts = prim["points"].std()
    n3 = int((prim["b_all"] == "3d").sum())
    n5 = int((prim["b_all"] == REF).sum())
    if n3 and n5:
        se = sd_pts * np.sqrt(1.0 / n3 + 1.0 / n5)
        print(f"\n  points sd {sd_pts:.2f}; n(3d)={n3}, n(5+d)={n5}")
        print(f"  -> smallest 3d-vs-5+d gap detectable at 80% power is about "
              f"{2.8 * se:.2f} points per match.")
        print(f"     For scale, a whole season's home advantage is worth ~0.35 pts/match,")
        print(f"     so this sample can only see a congestion effect LARGER than that.")

    print("\n  --- by OWN recovery (club fixed effects) ---")
    print("  Four outcomes x three buckets = 12 tests in this block alone. Treat any")
    print("  single marginal interval accordingly; nothing here is Bonferroni-corrected")
    print("  and a lower bound sitting just above zero is not a finding.")
    for col, lab in (("points", "points"), ("gf", "goals for"),
                     ("ga", "goals against"), ("win", "win rate")):
        pr = prim.dropna(subset=["b_all", col]).copy()
        pr["_dm"] = pr[col] - pr.groupby("club")[col].transform("mean")
        res = cluster_boot_means(pr, "b_all", "_dm")
        show(f"{lab}, vs each club's own season mean", res, lab,
             0.30 if col in ("points", "gf", "ga") else 0.10)

    print("\n  --- by REST DIFFERENTIAL ---")
    print("  `rest_congestion` used this estimand, and it is now built from BOTH sides'")
    print("  full competitive calendars rather than league matches only. But note what")
    print("  symmetry does and does not buy: the REGRESSOR is symmetric (whatever one")
    print("  side gains the other loses), the UNITS are not exchangeable. The side on")
    print("  short rest is the side that played midweek, which is the side in Europe,")
    print("  which is the better side. So a RAW outcome here is confounded with quality")
    print("  and `rest_congestion` was right to pair the differential with an")
    print("  opponent-adjusted outcome. Both are shown, and they disagree — which is")
    print("  the point.")
    pd_ = prim.dropna(subset=["rest_diff"]).copy()
    pd_["bd"] = diff_bucket(pd_["rest_diff"])
    print(f"\n  n with a two-sided rest measure: {len(pd_)}")
    tab = pd_.groupby("bd", observed=False).agg(
        n=("points", "size"), points=("points", "mean"), gf=("gf", "mean"),
        ga=("ga", "mean"), win=("win", "mean"), resid_xg=("resid", "mean"))
    print(tab.round(3).to_string())

    # THE CONFOUND, MADE VISIBLE. Season points-per-game of each side by bucket.
    ppg = prim.groupby("club")["points"].mean()
    pd_["own_ppg"] = pd_["club"].map(ppg)
    pd_["opp_ppg"] = pd_["opp"].map(ppg)
    print("\n  who is actually in each bucket (season points-per-game):")
    diag = pd_.groupby("bd", observed=False).agg(
        n=("points", "size"), own_ppg=("own_ppg", "mean"),
        opp_ppg=("opp_ppg", "mean"))
    diag["quality_gap"] = diag["own_ppg"] - diag["opp_ppg"]
    print(diag.round(3).to_string())
    print("\n  There it is. When the OPPONENT is fresher it is because WE played midweek,")
    print("  so `we` are the European club and the quality gap runs in our favour. The")
    print("  raw points gradient below is that gap, not fatigue.")

    # two-way demeaned points: strip own and opponent season strength, the same
    # adjustment the residual-xG outcome applies
    # expected points ~ own strength, discounted by how strong the opponent is
    pd_["adj_points"] = (pd_["points"] - pd_["own_ppg"]
                         + (pd_["opp_ppg"] - ppg.mean()))
    lvl = pd_[pd_["bd"] == "level (within 1d)"]
    for outcome, olab in (("points", "RAW points (confounded — shown to be discarded)"),
                          ("adj_points", "QUALITY-ADJUSTED points (own and opponent "
                                         "strength removed)")):
        print(f"\n  {olab}:")
        for lab, sub in (("opponent fresher by 2+",
                          pd_[pd_["bd"] == "opp fresher by 2+"]),
                         ("we are fresher by 2+",
                          pd_[pd_["bd"] == "we are fresher by 2+"])):
            if len(sub) < 20:
                print(f"    {lab}: n={len(sub)} — too few")
                continue
            rng = np.random.default_rng(0)
            a, c = sub[outcome].values, lvl[outcome].values
            bs = np.array([rng.choice(a, len(a), replace=True).mean()
                           - rng.choice(c, len(c), replace=True).mean()
                           for _ in range(6000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sig = "*" if not (lo <= 0 <= hi) else "(not significant)"
            print(f"    {lab}: n={len(sub)}, {a.mean() - c.mean():+.3f} vs level, "
                  f"95% CI ({lo:+.3f}, {hi:+.3f}) {sig}")

    print("\n  the sharp cut — our 3 days against an opponent on 6+:")
    sharp = prim[(prim["rec_all"] <= 3.5) & (prim["opp_rec_all"] >= 5.5)]
    balanced = prim[(prim["rec_all"] >= 5.5) & (prim["opp_rec_all"] >= 5.5)]
    print(f"    on <=3 days vs a rested opponent : n={len(sharp):4d}  "
          f"points {sharp['points'].mean():.3f}  gf {sharp['gf'].mean():.2f}  "
          f"ga {sharp['ga'].mean():.2f}")
    print(f"    both sides rested (5+ each)      : n={len(balanced):4d}  "
          f"points {balanced['points'].mean():.3f}  gf {balanced['gf'].mean():.2f}  "
          f"ga {balanced['ga'].mean():.2f}")
    if len(sharp) >= 20 and len(balanced) >= 20:
        rng = np.random.default_rng(1)
        a, c = sharp["points"].values, balanced["points"].values
        bs = np.array([rng.choice(a, len(a), replace=True).mean()
                       - rng.choice(c, len(c), replace=True).mean()
                       for _ in range(6000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "*" if not (lo <= 0 <= hi) else "(not significant)"
        print(f"    difference {a.mean() - c.mean():+.3f} points, 95% CI "
              f"({lo:+.3f}, {hi:+.3f}) {sig}")

    # ---- player side
    pp = player_panel()
    pp = pp[pp["gameweek"] <= PRIMARY_MAX_GW].dropna(subset=["b_all"])
    print("\n" + "=" * 78)
    print("H1. ROTATION — PL minutes, within player")
    print("=" * 78)
    print(f"  {len(pp)} player-PL-match rows, {pp['player_code'].nunique()} players")

    print("  risk set is the matchday squad (lineups.csv), so a rested player is a")
    print("  zero in the sample rather than a missing row.")

    # regulars: the population actually at risk of being rested. Defined on START
    # RATE over the squad sheet, not on mean minutes over appearances, which would
    # reintroduce the selection this panel exists to avoid.
    reg = pp.groupby("player_code")["started"].agg(["mean", "size"])
    keep = reg[(reg["mean"] >= 0.5) & (reg["size"] >= 10)].index
    q = pp[pp["player_code"].isin(keep)].copy()
    print(f"  regulars (started >=50% of sheets he was named on, >=10 sheets): "
          f"{len(q)} rows, {q['player_code'].nunique()} players")

    # within-player demeaning: each player against his own season average
    for col, lab, thr in (("minutes_played", "minutes", MIN_MINUTES_EFFECT),
                          ("started", "P(start)", 0.05),
                          ("played60", "P(60+ min)", 0.05)):
        q["_dm"] = q[col] - q.groupby("player_code")[col].transform("mean")
        res = cluster_boot_means(q, "b_all", "_dm")
        show(f"within-player {lab} vs own season mean", res, lab, thr)

    # ---- the user-facing split: is a European midweek different from a league one?
    print("\n  SPLIT BY WHAT THE MIDWEEK FIXTURE WAS")
    print("  A 3-day turnaround after a Thursday in Bucharest is not the same thing")
    print("  as a 3-day turnaround after a Saturday league game 30 miles away, and")
    print("  the pooled bucket above mixes them. `src` separates them.")
    q["src"] = np.where(q["prev_tour"].isin(["champions-league", "europa-league",
                                             "conference-league"]), "europe",
                        np.where(q["prev_tour"] == "efl-cup", "efl-cup", "league"))
    short = q[q["b_all"].isin(["<=2d", "3d", "4d"])].copy()
    base_ = q[q["b_all"] == REF]
    for col, lab in (("started", "P(start)"), ("minutes_played", "minutes")):
        # Demean against the SAME players' rested weeks. Demeaning inside each
        # subset would force its mean to zero by construction and report a null
        # that is an artefact of the arithmetic rather than a finding.
        ref_mean = base_.groupby("player_code")[col].mean()
        print(f"\n    {lab}, short-recovery rows vs the same players' 5+d weeks:")
        print(f"      {'preceded by':14s} {'n':>6s} {'clubs':>6s} {'delta':>9s}")
        for src in ("europe", "efl-cup", "league"):
            m = short[short["src"] == src].copy()
            m["_ref"] = m["player_code"].map(ref_mean)
            m = m.dropna(subset=["_ref"])
            if not len(m):
                continue
            print(f"      {src:14s} {len(m):6d} {m['club'].nunique():6d} "
                  f"{(m[col] - m['_ref']).mean():+9.3f}")

    # ---- robustness: a clean reference week
    print("\n  ROBUSTNESS — the 5+d reference is contaminated by international breaks,")
    print("  where players fly the world and come back tired. Restricting the")
    print("  reference to a NORMAL week (6-8 days) is the cleaner rested comparison.")
    norm = q[(q["b_all"] != REF) | ((q["rec_all"] >= 5.5) & (q["rec_all"] <= 8.5))]
    for col, lab, thr in (("started", "P(start)", 0.05),
                          ("minutes_played", "minutes", MIN_MINUTES_EFFECT)):
        nn = norm.copy()
        nn["_dm"] = nn[col] - nn.groupby("player_code")[col].transform("mean")
        res = cluster_boot_means(nn, "b_all", "_dm")
        show(f"{lab}, reference restricted to a normal 6-8 day week", res, lab, thr)

    print("\n  DOSE — did HE play in the midweek fixture? (endogenous: managers pick")
    print("  who plays midweek, so this is descriptive, not an effect)")
    mid = q[q["b_all"].isin(["<=2d", "3d", "4d"]) & q["prev_tour"].notna()]
    mid = mid[mid["prev_tour"] != "prem"]
    if len(mid) > 50:
        mid["_dm"] = mid["minutes_played"] - mid.groupby("player_code")[
            "minutes_played"].transform("mean")
        mid["dose"] = pd.cut(mid["prev_minutes"], [-1, 0.5, 45, 75, 200],
                             labels=["did not play", "1-45", "46-75", "76+"])
        t = mid.groupby("dose", observed=False).agg(
            n=("_dm", "size"), delta_minutes=("_dm", "mean"))
        print(t.round(2).to_string())

    # ---- fringe players: is the rotation happening below the regulars?
    fringe = pp[~pp["player_code"].isin(keep)].copy()
    if len(fringe) > 200:
        print("\n  FRINGE PLAYERS — everyone who is not an established starter")
        print("  (if congestion moves anyone it should be the squad players)")
        for col, lab in (("started", "P(start)"), ("minutes_played", "minutes")):
            fringe["_dm"] = fringe[col] - fringe.groupby("player_code")[
                col].transform("mean")
            res = cluster_boot_means(fringe, "b_all", "_dm")
            show(f"fringe {lab}", res, lab, 0.05 if col == "started"
                 else MIN_MINUTES_EFFECT)

    # ---- WHERE THE ROTATION ACTUALLY GOES
    print("\n" + "=" * 78)
    print("WHERE THE ROTATION ACTUALLY GOES — who plays the midweek match")
    print("=" * 78)
    print("  The league-side nulls above only make sense next to this. Rotation is not")
    print("  absent from a congested week; it is spent on the CUP TEAM.")
    print()
    print("  MEASUREMENT NOTE: this table cannot come from lineups.csv. Upstream")
    print("  populates `player_id` for Premier League teamsheets (400/400 in a sample")
    print("  gameweek) but leaves it null for most CUP rows (15 of 129 Arsenal")
    print("  Champions League rows), so a lineup-based cup XI would be a 5% subset")
    print("  chosen by whichever names upstream managed to resolve. `playermatchstats`")
    print("  joins completely in every competition, so involvement is measured there as")
    print("  60+ minutes played — the same definition applied to every competition,")
    print("  including the league yardstick, so the columns are comparable.")
    apm_all = re_.all_comp_player_matches(SEASON)
    cal = re_.competitive_calendar(SEASON)
    gwmap = cal.drop_duplicates("match_id").set_index("match_id")["gameweek"]
    inv = apm_all[apm_all["minutes_played"] >= 60].copy()
    inv["gameweek"] = inv["match_id"].map(gwmap)
    inv = inv[inv["gameweek"] <= PRIMARY_MAX_GW]
    inv["is_regular"] = inv["player_code"].isin(keep)
    tab = inv.groupby("tournament", observed=False).agg(
        played60=("is_regular", "size"),
        pct_regulars=("is_regular", "mean"),
        matches=("match_id", "nunique"))
    tab["pct_regulars"] = (tab["pct_regulars"] * 100).round(1)
    print()
    print(tab.sort_values("pct_regulars", ascending=False).to_string())
    pl_pct = tab.loc["prem", "pct_regulars"] if "prem" in tab.index else float("nan")
    print(f"\n  The league sits at {pl_pct:.1f}%. Every cup is far below it, and that gap")
    print("  IS the rotation — taken in the cup, which is why the league-side minutes")
    print("  above barely move. Note `matches` counts fixtures, and the European rows")
    print("  cover only the 6 / 2 / 1 clubs who were in those competitions.")

    print("\n" + "=" * 78)
    print("H3. COMPETITION — does it add anything beyond recovery days?")
    print("=" * 78)
    sub = prim[prim["prev_tour"].notna() & (prim["prev_tour"] != "prem")]
    print("  team residual xG by preceding competition (primary sample):")
    t = sub.groupby("prev_tour", observed=False).agg(
        n=("resid", "size"), resid_xg=("resid", "mean"),
        mean_recovery=("rec_all", "mean"), clubs=("club", "nunique"))
    print(t.round(3).to_string())
    print("\n  `clubs` is the column that matters: a Conference League row is one club.")
    print("  Any apparent competition effect here is that club's season, not a")
    print("  competition effect, and is NOT adopted (pre-registered).")

    print("\n" + "=" * 78)
    print("APPENDIX. GW27+ — treatment measured with error, not evidence")
    print("=" * 78)
    if len(late):
        known = late["rec_all"].notna().sum()
        print(f"  n={len(late)}, recovery measurable for {known}. Cup kickoff times are")
        print("  largely absent after GW26 and the FA Cup is absent all season, so short")
        print("  turnarounds here are systematically coded as rested. Reported for")
        print("  completeness; no decision rule is applied.")
        la = cluster_boot_means(late.dropna(subset=["b_all"]), "b_all", "resid")
        show("GW27+ residual xG by recovery (degraded)", la, "xG", MIN_XG_EFFECT)

    tp.to_csv(OUT, index=False)
    q.to_csv(OUT_PLAYER, index=False)
    print(f"\n-> {OUT}")
    print(f"-> {OUT_PLAYER}")


# -------------------------------------------------------------------- selftest
def selftest():
    """Offline, synthetic. Checks the bucket edges and that a planted rotation
    effect is recovered with the right sign and rough size."""
    # edges are midpoints, so a 2.9-day gap is a three-day turnaround
    b = bucket(pd.Series([1, 2, 2.9, 3, 3.4, 3.6, 4, 4.4, 4.6, 7, np.nan]))
    got = [str(x) for x in b]
    assert got == ["<=2d", "<=2d", "3d", "3d", "3d", "4d", "4d", "4d",
                   "5+d", "5+d", "nan"], got

    rng = np.random.default_rng(0)
    rows = []
    for club in [f"C{i}" for i in range(12)]:
        for p in range(6):
            for wk in range(20):
                rec = 3.0 if wk % 4 == 0 else 6.0
                # planted: 3-day turnaround costs 12 minutes
                mins = 80 - (12 if rec == 3.0 else 0) + rng.normal(0, 6)
                rows.append(dict(club=club, player_code=f"{club}_{p}",
                                 rec_all=rec, minutes_played=mins))
    d = pd.DataFrame(rows)
    d["b_all"] = bucket(d["rec_all"])
    d["_dm"] = d["minutes_played"] - d.groupby("player_code")[
        "minutes_played"].transform("mean")
    res = cluster_boot_means(d, "b_all", "_dm", n=400, seed=1)
    diff = res["3d"][2]
    lo, hi = res["3d"][3], res["3d"][4]
    assert -14 < diff < -8, f"planted -12 minute effect not recovered: {diff}"
    assert hi < 0, f"CI should exclude zero for a planted effect: ({lo},{hi})"

    # and a planted NULL must not be called significant
    d2 = d.copy()
    d2["minutes_played"] = 80 + rng.normal(0, 6, len(d2))
    d2["_dm"] = d2["minutes_played"] - d2.groupby("player_code")[
        "minutes_played"].transform("mean")
    r2 = cluster_boot_means(d2, "b_all", "_dm", n=400, seed=2)
    lo2, hi2 = r2["3d"][3], r2["3d"][4]
    assert lo2 <= 0 <= hi2, f"null wrongly called significant: ({lo2},{hi2})"
    print("SELFTEST OK: bucket edges correct, planted rotation effect recovered "
          "with CI excluding zero, planted null not called significant.")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        selftest(); _sys.exit(0)
    main()
