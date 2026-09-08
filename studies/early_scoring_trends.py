from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
early_scoring_trends.py — do four club characteristics predict EARLY-SEASON SCORING?
====================================================================================
Four hypotheses, all previously examined in this repo against *xG strength residuals*.
This study re-runs them against a different estimand — GOALS ACTUALLY SCORED, and match
TOTALS — over a much longer panel, and pre-registers the decision rule before looking.

    H1  prior-season final position
    H2  prior-season final-10 form (residual above that season's own baseline)
    H3  offseason managerial change
    H4  squad turnover (arrivals + departures)

WHY RE-TEST SOMETHING ALREADY CALLED NULL
------------------------------------------
Not to get a different answer. Three reasons that are legitimate, and one that is not:

  * DIFFERENT OUTCOME. `midtable_fade`, `late_form_carryover`, `new_manager_debut` and
    `transfer_churn` all score clubs on xG-based strength or on league points. None asks
    whether these characteristics move GOALS, and none asks about MATCH TOTALS (both
    teams), which is the quantity that drives over/under markets and the one FPL
    attacking returns concentrate in. `early_season_goals` did test totals, but only
    globally — never crossed with a club characteristic.
  * LONGER PANEL for H1/H2. The xG studies are capped at the 12 Understat seasons
    (2014/15 on). Goals go back to 1993/94 — 29 usable seasons, ~2.4x the club-seasons.
  * SCHEDULE ADJUSTMENT MADE EXPLICIT. Every measure here is a residual from
    opponent-and-venue-adjusted expectation, so "they had easy early fixtures" cannot
    masquerade as an effect.

  NOT a reason: hoping a null flips. Goals are NOISIER than xG per match, so on the
  shared 12 seasons this test has LESS power than the xG version, not more. Where the
  panel is the same length, a null here adds little; where it is 2.4x longer, it adds
  real information. That asymmetry is stated in the power table below and must be read
  before the p-values.

THE MULTIPLICITY TRAP THIS STUDY IS BUILT TO AVOID
---------------------------------------------------
`midtable_fade` died of exactly this: one marginal hit out of four archetypes tested,
p=0.039 against the 0.0125 that four tests require. Testing four hypotheses again is the
same trap, so the correction is pre-committed HERE, before any result is seen:

    PRIMARY OUTCOME   match total goals residual, early window minus rest of season
    PRIMARY TESTS     4 (one per hypothesis)
    ALPHA             0.05 / 4 = 0.0125, two-sided          <- pre-committed
    SECONDARY         team goals-scored residual; reported as support only, never as
                      the basis for a verdict, and NOT counted toward the correction
    DECISION          "confirmed" requires p < 0.0125 on the PRIMARY outcome AND the
                      same sign on the secondary AND leave-one-season-out stability
    NULLS             reported with the MDE, so "no effect" is separable from
                      "no power" — the distinction `tournament_summers` had to make

H1 is tested as a CONTINUOUS rank, not as four archetype buckets. Bucketing was what
generated the midtable_fade false positive; a continuous slope is one test, not four,
and has more power.

WITHIN-CLUB DIFFERENCING
------------------------
Good clubs score more all season, so a cross-sectional "clubs with characteristic X score
more early" is mostly club quality. Every outcome here is EARLY minus REST-OF-SEASON for
the same club in the same season, which differences club quality out entirely. What
survives is a genuine early-vs-late shift.

Run:  python studies/early_scoring_trends.py
      python studies/early_scoring_trends.py --selftest
Out:  studies/early_scoring_trends.csv        (the four tests)
      studies/early_scoring_panel.csv         (the club-season panel behind them)
"""
import glob
import numpy as np
import pandas as pd
from scipy import stats

EARLY_N = 6                    # matches per club counted as "early"
ALPHA = 0.05 / 4               # pre-committed Bonferroni over the four primary tests
POWER_Z = 0.8416               # z for 80% power
OUT_TESTS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "early_scoring_trends.csv")
OUT_PANEL = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "early_scoring_panel.csv")

# football-data club spellings -> the names used by manager_changes / transfer_counts
NAME_NORM = {
    "Man United": "Man United", "Man Utd": "Man United", "Manchester United": "Man United",
    "Man City": "Man City", "Manchester City": "Man City",
    "Tottenham": "Tottenham", "Spurs": "Tottenham",
    "Newcastle": "Newcastle", "Nott'm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest", "Sheffield United": "Sheffield Utd",
    "Sheffield Weds": "Sheffield Wed", "QPR": "QPR", "West Brom": "West Brom",
    "Wolves": "Wolves", "Leeds": "Leeds", "Brighton": "Brighton",
    "Bournemouth": "Bournemouth", "Cardiff": "Cardiff", "Hull": "Hull",
    "Ipswich": "Ipswich", "Coventry": "Coventry", "Luton": "Luton",
}


def _norm(t):
    return NAME_NORM.get(str(t).strip(), str(t).strip())


# ---------------------------------------------------------------- data
def load_matches(cache_dir=None):
    """Every cached football-data season, long form (one row per club-match)."""
    cache_dir = cache_dir or _os.path.join(config.SCRATCH, "fd_hist")
    rows = []
    for path in sorted(glob.glob(_os.path.join(cache_dir, "E0_*.csv"))):
        code = _os.path.basename(path)[3:7]
        try:
            d = pd.read_csv(path, encoding="latin-1")
        except Exception:
            continue
        if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(d.columns):
            continue
        d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        d["date"] = pd.to_datetime(d.get("Date"), dayfirst=True, errors="coerce")
        if d["date"].isna().all():
            d["date"] = pd.RangeIndex(len(d))
        d = d.sort_values("date").reset_index(drop=True)
        for _, r in d.iterrows():
            h, a = _norm(r["HomeTeam"]), _norm(r["AwayTeam"])
            gh, ga = float(r["FTHG"]), float(r["FTAG"])
            rows.append({"season": code, "date": r["date"], "team": h, "opp": a,
                         "is_home": 1, "gf": gh, "ga": ga, "total": gh + ga})
            rows.append({"season": code, "date": r["date"], "team": a, "opp": h,
                         "is_home": 0, "gf": ga, "ga": gh, "total": gh + ga})
    m = pd.DataFrame(rows)
    m["season"] = m["season"].astype(str).str.zfill(4)   # keep "0001" from becoming 1
    m = m.sort_values(["season", "team", "date"])
    m["md"] = m.groupby(["season", "team"]).cumcount() + 1
    return m


def adjusted_residuals(m):
    """Residual of each club-match from an opponent- and venue-adjusted expectation.

    Multiplicative Poisson-style season model, fitted within season:
        E[gf] = league_mean * att(team) * def(opp) * venue
    `att` and `def` are that club's own SEASON-LONG rates, so the residual asks "did they
    score more early than their own season and their own fixtures imply?" — which removes
    both club quality and schedule. Two damped iterations are enough; this is a control,
    not the estimand.
    """
    out = []
    for season, g in m.groupby("season"):
        g = g.copy()
        mu = g["gf"].mean()
        home_f = (g.loc[g.is_home == 1, "gf"].mean() / mu) if mu > 0 else 1.0
        away_f = (g.loc[g.is_home == 0, "gf"].mean() / mu) if mu > 0 else 1.0
        att = g.groupby("team")["gf"].mean() / mu
        dfn = g.groupby("team")["ga"].mean() / mu
        for _ in range(2):
            venue = np.where(g.is_home == 1, home_f, away_f)
            pred = mu * g["team"].map(att).values * g["opp"].map(dfn).values * venue
            ratio = g["gf"].values / np.maximum(pred, 1e-6)
            adj = pd.Series(ratio, index=g.index).groupby(g["team"]).mean()
            att = att * adj.reindex(att.index).fillna(1.0) ** 0.5
        venue = np.where(g.is_home == 1, home_f, away_f)
        g["exp_gf"] = mu * g["team"].map(att).values * g["opp"].map(dfn).values * venue
        g["exp_ga"] = mu * g["opp"].map(att).values * g["team"].map(dfn).values * \
            np.where(g.is_home == 1, away_f, home_f)
        g["exp_total"] = g["exp_gf"] + g["exp_ga"]
        g["res_gf"] = g["gf"] - g["exp_gf"]
        g["res_total"] = g["total"] - g["exp_total"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


def club_season_panel(m):
    """early-minus-rest residuals per club-season, plus prior-season features."""
    m = m.copy()
    m["window"] = np.where(m.md <= EARLY_N, "early", "rest")
    piv = m.pivot_table(index=["season", "team"], columns="window",
                        values=["res_gf", "res_total"], aggfunc="mean")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    P = piv.reset_index()
    P["d_total"] = P["res_total_early"] - P["res_total_rest"]     # PRIMARY outcome
    P["d_gf"] = P["res_gf_early"] - P["res_gf_rest"]              # secondary

    # prior-season table position and last-10 form residual
    pts = m.assign(p=np.where(m.gf > m.ga, 3, np.where(m.gf == m.ga, 1, 0)))
    tbl = pts.groupby(["season", "team"]).agg(
        points=("p", "sum"), gd=("gf", "sum")).reset_index()
    tbl["gd"] = tbl["gd"] - pts.groupby(["season", "team"])["ga"].sum().values
    tbl["position"] = tbl.groupby("season")[["points", "gd"]].apply(
        lambda x: x["points"].rank(ascending=False, method="first")).values

    last10 = m[m.groupby(["season", "team"])["md"].transform("max") - m["md"] < 10]
    l10 = last10.groupby(["season", "team"])["res_gf"].mean().rename("late10_resid")
    full = m.groupby(["season", "team"])["res_gf"].mean().rename("full_resid")
    form = pd.concat([l10, full], axis=1).reset_index()
    form["late10_form"] = form["late10_resid"] - form["full_resid"]

    prior = tbl.merge(form, on=["season", "team"])
    codes = sorted(m["season"].unique())
    nxt = {c: codes[i + 1] for i, c in enumerate(codes[:-1])}
    prior["next"] = prior["season"].map(nxt)
    prior = prior.dropna(subset=["next"])
    prior = prior.rename(columns={"season": "prior_season", "position": "prior_position",
                                  "points": "prior_points"})
    P = P.merge(prior[["next", "team", "prior_season", "prior_position", "prior_points",
                       "late10_form"]],
                left_on=["season", "team"], right_on=["next", "team"], how="left")
    P["promoted"] = P["prior_position"].isna()
    return P


def add_manager_and_churn(P):
    """H3/H4 features. Both sources start in 2014/15, which caps their power."""
    P = P.copy()
    P["offseason_mgr"] = np.nan
    P["churn"] = np.nan
    try:
        mc = pd.read_csv(config.MANAGER_CHANGES)
        mc["date"] = pd.to_datetime(mc["date"], errors="coerce")
        mc["club"] = mc["club"].map(_norm)
        # offseason = a change dated between 1 May and 15 Aug, credited to the season
        # that starts that summer
        off = mc[mc["date"].dt.month.between(5, 8)].copy()
        off["season"] = off["date"].dt.year.map(
            lambda y: f"{str(y)[2:]}{str(y + 1)[2:]}")
        key = set(zip(off["season"], off["club"]))
        seasons_covered = set(off["season"])
        P.loc[P["season"].isin(seasons_covered), "offseason_mgr"] = [
            1.0 if (s, t) in key else 0.0
            for s, t in zip(P.loc[P["season"].isin(seasons_covered), "season"],
                            P.loc[P["season"].isin(seasons_covered), "team"])]
    except Exception as e:
        print(f"  [warn] manager changes unavailable: {type(e).__name__}")
    try:
        tc = pd.read_csv(config.TRANSFER_COUNTS)
        tc["club"] = tc["club"].map(_norm)
        # season codes are "0001"/"1415"; read_csv turns them into ints and eats the
        # leading zero, so a naive merge raises on object-vs-int64 and a coerced one
        # would silently match nothing. Pad both sides back to 4-character strings.
        tc["season"] = tc["season"].astype(str).str.zfill(4)
        tc["churn"] = tc["arrivals"] + tc["departures"]
        P = P.drop(columns=["churn"]).merge(
            tc[["season", "club", "churn"]].rename(columns={"club": "team"}),
            on=["season", "team"], how="left")
        n = int(P["churn"].notna().sum())
        print(f"  [churn] matched {n} club-seasons "
              f"({tc['season'].nunique()} seasons in the source)")
        if n == 0:
            raise RuntimeError("churn joined to nothing — check club spellings")
    except Exception as e:
        print(f"  [warn] transfer counts unavailable: {type(e).__name__}: {e}")
        if "churn" not in P.columns:
            P["churn"] = np.nan
    return P


# ---------------------------------------------------------------- inference
def cluster_ols(y, x, cluster):
    """Slope of y on x with an intercept, SE clustered on `cluster`.

    Matches within a season share weather, ball, refereeing directives and league-wide
    scoring level, so treating club-seasons as independent understates the SE. Clustering
    on season is the conservative choice and is what the season-level confounds require.
    """
    d = pd.DataFrame({"y": y, "x": x, "c": cluster}).dropna()
    if len(d) < 12 or d["x"].nunique() < 2:
        return dict(n=len(d), coef=np.nan, se=np.nan, t=np.nan, p=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, clusters=d["c"].nunique())
    X = np.column_stack([np.ones(len(d)), d["x"].values])
    yv = d["y"].values
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ yv
    resid = yv - X @ beta
    meat = np.zeros((2, 2))
    for _, gi in d.groupby("c").indices.items():
        Xg, rg = X[gi], resid[gi]
        s = Xg.T @ rg
        meat += np.outer(s, s)
    G = d["c"].nunique()
    scale = G / max(G - 1, 1)
    V = XtX_inv @ meat @ XtX_inv * scale
    se = float(np.sqrt(max(V[1, 1], 0)))
    t = float(beta[1] / se) if se > 0 else np.nan
    dof = max(G - 1, 1)
    p = float(2 * (1 - stats.t.cdf(abs(t), dof))) if np.isfinite(t) else np.nan
    crit = stats.t.ppf(1 - 0.025, dof)
    return dict(n=int(len(d)), coef=float(beta[1]), se=se, t=t, p=p,
                ci_lo=float(beta[1] - crit * se), ci_hi=float(beta[1] + crit * se),
                clusters=int(G))


def mde(se, alpha=ALPHA):
    """Minimum detectable effect at 80% power for the pre-committed alpha."""
    if not np.isfinite(se):
        return np.nan
    return (stats.norm.ppf(1 - alpha / 2) + POWER_Z) * se


def loso(P, ycol, xcol):
    """Leave-one-season-out: the sign must be stable to count as confirmed.
    NB stability is NOT evidence of a real effect — a pooled fluke is also LOO-stable
    (`euro_qualifying_fade` made exactly that point). It is a necessary check, not a
    sufficient one."""
    coefs = []
    for s in sorted(P["season"].dropna().unique()):
        sub = P[P["season"] != s]
        r = cluster_ols(sub[ycol], sub[xcol], sub["season"])
        if np.isfinite(r["coef"]):
            coefs.append(r["coef"])
    if not coefs:
        return dict(loso_min=np.nan, loso_max=np.nan, loso_sign_stable=False)
    return dict(loso_min=float(np.min(coefs)), loso_max=float(np.max(coefs)),
                loso_sign_stable=bool(np.sign(np.min(coefs)) == np.sign(np.max(coefs))))


HYPOTHESES = [
    ("H1 prior-season final position", "prior_position",
     "higher number = finished lower. Continuous rank, ONE test, not four buckets."),
    ("H2 prior-season last-10 form", "late10_form",
     "goals-scored residual over the final 10, above that club's own season baseline."),
    ("H3 offseason manager change", "offseason_mgr",
     "appointed between 1 May and 15 Aug. 2014/15 on only."),
    ("H4 squad turnover", "churn",
     "arrivals + departures. 2014/15 on only; transfermarkt transcription."),
]


def positive_control(m, verbose=True):
    """Can this panel recover an effect known to be real?

    Four nulls are only worth reporting if the machinery can detect something. The known
    effect is early-season home advantage: `early_season_goals.py` measured it ~0.15
    goals LOWER over matchdays 1-3 on 12 Understat seasons, and that finding is already
    applied in `bayes_model._home_effect`. This re-tests it on 31 seasons of raw results
    — an independent and 2.5x longer panel — paired within season.
    """
    def ha(df):
        return (df.loc[df.is_home == 1, "gf"].mean() -
                df.loc[df.is_home == 0, "gf"].mean())
    a = m[m.md <= 3].groupby("season").apply(ha)
    b = m[m.md >= 13].groupby("season").apply(ha)
    j = pd.concat([a.rename("early"), b.rename("late")], axis=1).dropna()
    d = j["early"] - j["late"]
    t, p = stats.ttest_1samp(d, 0)
    res = {"seasons": int(len(j)), "diff": float(d.mean()),
           "ci_lo": float(d.mean() - 1.96 * d.sem()),
           "ci_hi": float(d.mean() + 1.96 * d.sem()),
           "t": float(t), "p": float(p), "seasons_down": int((d < 0).sum()),
           "recovered": bool(p < 0.05)}
    if verbose:
        print("=" * 100)
        print("POSITIVE CONTROL — can this panel find an effect known to exist?")
        print("=" * 100)
        print(f"  early-season home advantage, md1-3 vs md13-38, paired over "
              f"{res['seasons']} seasons")
        print(f"  difference {res['diff']:+.4f} goals  CI ({res['ci_lo']:+.4f}, "
              f"{res['ci_hi']:+.4f})  p = {res['p']:.4f}")
        print(f"  {res['seasons_down']}/{res['seasons']} seasons show the drop")
        print(f"  -> {'RECOVERED' if res['recovered'] else 'NOT RECOVERED'}  "
              f"(early_season_goals measured -0.152 on 12 Understat seasons; the model "
              f"already applies it)")
        if not res["recovered"]:
            print("  !! the four nulls below are NOT interpretable — fix this first")
        print()
    return res


def run(verbose=True):
    if verbose:
        print("Loading match history ...")
    m = load_matches()
    if m.empty:
        raise RuntimeError("no cached football-data seasons found")
    m = adjusted_residuals(m)
    P = club_season_panel(m)
    P = add_manager_and_churn(P)
    P.to_csv(OUT_PANEL, index=False)

    if verbose:
        print(f"  {m.season.nunique()} seasons, {len(m):,} club-matches, "
              f"{len(P)} club-seasons")
        print(f"  early window = first {EARLY_N} matches per club\n")
    pc = positive_control(m, verbose=verbose)
    if verbose:
        print("=" * 100)
        print("POWER, COMPUTED BEFORE THE TESTS  (pre-committed alpha = "
              f"{ALPHA:.4f}, two-sided, 80% power)")
        print("=" * 100)

    rows = []
    for label, col, note in HYPOTHESES:
        prim = cluster_ols(P["d_total"], P[col], P["season"])
        sec = cluster_ols(P["d_gf"], P[col], P["season"])
        m_prim = mde(prim["se"])
        sd_x = P[col].std()
        rows.append({"hypothesis": label, "predictor": col, "note": note,
                     "n": prim["n"], "seasons": prim["clusters"],
                     "mde_per_unit": m_prim,
                     "mde_per_sd": m_prim * sd_x if np.isfinite(m_prim) else np.nan,
                     "coef_total": prim["coef"], "se_total": prim["se"],
                     "p_total": prim["p"], "ci_lo": prim["ci_lo"], "ci_hi": prim["ci_hi"],
                     "coef_gf": sec["coef"], "p_gf": sec["p"],
                     "effect_per_sd_total": prim["coef"] * sd_x
                     if np.isfinite(prim["coef"]) else np.nan})
    R = pd.DataFrame(rows)

    if verbose:
        for _, r in R.iterrows():
            print(f"  {r['hypothesis']:34s} n={r['n']:4d}  seasons={r['seasons']:2d}  "
                  f"MDE = {r['mde_per_sd']:+.4f} goals/match per SD")
        print("\n  Read this first: an MDE larger than the effect anyone cares about "
              "means a null here\n  is 'not detectable', not 'zero'.\n")

    # only now look at the results
    for i, r in R.iterrows():
        col = r["predictor"]
        sig = np.isfinite(r["p_total"]) and r["p_total"] < ALPHA
        same_sign = (np.isfinite(r["coef_gf"]) and
                     np.sign(r["coef_gf"]) == np.sign(r["coef_total"]))
        lo = loso(P, "d_total", col) if sig else dict(loso_sign_stable=False,
                                                     loso_min=np.nan, loso_max=np.nan)
        R.loc[i, "loso_min"] = lo["loso_min"]
        R.loc[i, "loso_max"] = lo["loso_max"]
        R.loc[i, "secondary_agrees"] = same_sign
        R.loc[i, "verdict"] = ("CONFIRMED" if (sig and same_sign and
                                               lo["loso_sign_stable"])
                               else "null" if np.isfinite(r["p_total"])
                               else "no data")

    R["control_home_adv_diff"] = pc["diff"]
    R["control_p"] = pc["p"]
    R["control_recovered"] = pc["recovered"]
    R.to_csv(OUT_TESTS, index=False)

    if verbose:
        print("=" * 100)
        print("RESULTS — primary outcome: match total goals, early minus rest of season")
        print("=" * 100)
        for _, r in R.iterrows():
            print(f"\n{r['hypothesis']}")
            print(f"  {r['note']}")
            print(f"  n={r['n']} club-seasons over {r['seasons']} seasons")
            print(f"  coef {r['coef_total']:+.5f} per unit  "
                  f"({r['effect_per_sd_total']:+.4f} goals/match per SD)")
            print(f"  95% CI ({r['ci_lo']:+.5f}, {r['ci_hi']:+.5f})   "
                  f"p = {r['p_total']:.4f}   alpha = {ALPHA:.4f}")
            print(f"  secondary (goals scored) coef {r['coef_gf']:+.5f}, "
                  f"p = {r['p_gf']:.4f}")
            print(f"  VERDICT: {r['verdict']}")
        print(f"\nwrote {OUT_TESTS}\nwrote {OUT_PANEL}")
    return R, P


def selftest():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(14):
        code = f"{s:02d}{s+1:02d}"
        teams = [f"T{i}" for i in range(20)]
        for t in teams:
            for md in range(1, 39):
                opp = teams[(int(t[1:]) + md) % 20]
                if opp == t:
                    opp = teams[(int(t[1:]) + md + 1) % 20]
                rows.append({"season": code, "date": md, "team": t, "opp": opp,
                             "is_home": md % 2, "gf": rng.poisson(1.4),
                             "ga": rng.poisson(1.4)})
    m = pd.DataFrame(rows)
    m["total"] = m["gf"] + m["ga"]
    m["md"] = m.groupby(["season", "team"]).cumcount() + 1
    m = adjusted_residuals(m)
    P = club_season_panel(m)

    assert len(P) > 200, "panel too small"
    # residuals must be centred by construction
    assert abs(m["res_gf"].mean()) < 0.05, f"residuals not centred: {m['res_gf'].mean()}"
    # pure noise -> the four tests must not fire
    r = cluster_ols(P["d_total"], P["prior_position"], P["season"])
    assert r["n"] > 100 and np.isfinite(r["p"]), "cluster_ols failed on noise"
    assert r["p"] > ALPHA, f"false positive on pure noise: p={r['p']:.4f}"

    # a planted effect MUST be recovered, or a null here means nothing
    P2 = P.dropna(subset=["prior_position"]).copy()
    P2["d_total"] = P2["d_total"] + 0.05 * (P2["prior_position"] - 10.5)
    r2 = cluster_ols(P2["d_total"], P2["prior_position"], P2["season"])
    assert r2["p"] < ALPHA, f"planted effect not recovered: p={r2['p']:.4f}"
    assert abs(r2["coef"] - 0.05) < 0.02, f"planted coef off: {r2['coef']:.4f}"

    assert mde(0.01) > 0 and np.isfinite(mde(0.01))
    print("SELFTEST OK: residuals centred, no false positive on noise, planted effect "
          "recovered at the pre-committed alpha, MDE finite.")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        selftest(); _sys.exit(0)
    import warnings; warnings.filterwarnings("ignore")
    run()
