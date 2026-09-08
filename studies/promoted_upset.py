import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
promoted_upset.py — does losing to a newly-promoted club early predict a bad season?
=====================================================================================
Prompted by Hull 2-0 Man United, GW1 2026/27. This is therefore a POST-HOC question
asked after seeing a striking result, which is the single highest-risk setting for a
false positive in this whole project. The design below is fixed before any estimate is
produced, and the decision rule with it.

PRE-REGISTRATION  [fixed 2026-08-22, before results were seen]
---------------------------------------------------------------
H1 (PRIMARY, one test). A non-promoted club that LOSES to a newly-promoted club within
its own first three matches of a season underperforms over the rest of that season
(its matches 4..N), relative to what its prior-season strength implies.

  estimator   OLS   rest_ppg ~ beta*lost_to_promoted_early + gamma*prior_ppg + const
              cluster-robust SE by season (a season's clubs share opponents and a
              common scoring environment, so match-level shocks are not independent)
  sample      1994/95 .. 2025/26. A season needs its predecessor to identify who was
              promoted, so 1993/94 provides only the lookup. Clubs promoted INTO the
              season are excluded as treated units: they have no prior PL season, so
              `prior_ppg` is undefined and they are not the population of interest.
  decision    reject H0 only if p < 0.05 on beta AND the sign is stable under
              leave-one-season-out. Report the minimum detectable effect either way.

SECONDARY, and they are the interesting ones. Declared here so they cannot be presented
later as if they had been the plan all along. Bonferroni across the two: alpha = 0.025.

  S1 TIMING (placebo). Same regression, treatment = lost to a promoted club in matches
     20-22 instead of 1-3. If an early defeat carries something specific — a narrative
     effect, a confidence effect, a manager under pressure — early should show what
     mid-season does not. If the two coefficients match, the mechanism is not timing;
     it is that a club which loses to promoted opposition was simply weaker than its
     prior season said.

  S2 SPECIFICITY. Treatment = lost in matches 1-3 to any club that finished BOTTOM SIX
     the previous season, promoted clubs excluded. If "newly promoted" carries no
     information beyond "weak opponent", beta_S2 ~= beta_H1 and the promotion framing
     adds nothing.

WHAT THE REPO ALREADY SAYS ABOUT THIS CLASS OF HYPOTHESIS
----------------------------------------------------------
Every neighbouring narrative tested here has come back null: `new_manager_debut` (no
early bump or penalty), `late_form_carryover` (+0.0000 r2), `midtable_fade` (closed as a
probable false positive after failing Bonferroni), `transfer_churn` (not front-loaded),
`tournament_summers` (null on every metric), `early_season_goals` (no global early
effect). The prior on this one being real is correspondingly low, and the burden is on
the data.

RESULT  [2026-08-22] — NULL on the primary test, and the secondaries explain why
---------------------------------------------------------------------------------
31 seasons, 1993/94-2025/26, 23,888 club-matches, 496 club-seasons, 60 treated.

  H1  lost to promoted, own matches 1-3    beta -0.0439 ppg  SE 0.0385  p=0.263  NULL
  S1  same, matches 20-22 (placebo)        beta -0.0560 ppg  SE 0.0513  p=0.284  null
  S2  lost to prior-bottom-6, matches 1-3  beta -0.0531 ppg  SE 0.0510  p=0.307  null

All three coefficients are the same size. That is the finding, not an aside:

  * TIMING CARRIES NOTHING. The mid-season placebo is if anything LARGER than the
    early-season effect. If an opening-weeks defeat had a psychological or narrative
    component, S1 is where it would separate. It does not.
  * "PROMOTED" CARRIES NOTHING BEYOND "WEAK". S2 uses prior-bottom-six opponents with
    promoted clubs removed and lands on the same coefficient. The promotion framing is
    not doing any work.
  * MOST OF THE RAW GAP IS SELECTION. Uncontrolled, treated clubs finish 0.150 ppg
    worse. Controlling for prior-season PPG removes 71% of it. Treated clubs averaged
    1.358 prior ppg against 1.492 for everyone else: clubs that lose to promoted sides
    were ALREADY weaker, and that is what the raw gap is mostly measuring.

Power: MDE 0.108 ppg at 80%, about 3.8 points over a 35-match remainder. So this rules
out anything larger than ~4 points; it cannot rule out something smaller. The null is
"not detectable at this sample size", not "exactly zero" — the same caveat
`tournament_summers` carries.

Leave-one-season-out: sign stable 29/29, beta in [-0.057, -0.026]. **This is not
evidence against noise.** A pooled fluke is also LOO-stable; `euro_qualifying_fade`
established that caveat in this repo and it applies here unchanged.

EXPLORATORY, NOT PRE-REGISTERED — strong clubs only
----------------------------------------------------
The prompting case was a STRONG club losing to a promoted one, which the pooled test
does not isolate. Restricted to the top quartile of prior-season PPG (>= 1.74):
n=131, treated=7, beta -0.032 ppg, p=0.844, MDE 0.448 ppg = 15.7 points. Underpowered
to the point of being uninformative as a test, so it is reported as case evidence:

  season  club         lost to      md   prior ppg   finished
  0910    Man United   Burnley       2       2.37    2nd, 85 pts
  1617    Leicester    Hull          1       2.13    12th, 44 pts
  9596    Blackburn    Bolton        3       2.12    6th, 61 pts
  1314    Man City     Cardiff       2       2.05    1st, 86 pts  <- won the title
  0203    Newcastle    Man City      2       1.87    3rd, 69 pts
  0102    Liverpool    Bolton        2       1.82    2nd, 80 pts
  0001    Leeds        Man City      3       1.82    4th, 68 pts

Five of seven finished top six; one won the league. The two that fell away — Blackburn
95/96 and Leicester 16/17 — were both DEFENDING CHAMPIONS, a regression pattern with a
well-known cause of its own and no connection to who beat them in August.

VERDICT: no model change. Do not adjust any club's projection because it lost to a
promoted side, in the opening weeks or otherwise. Price the club at its strength.

Run:  python studies/promoted_upset.py
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
import history

EARLY = (1, 3)          # a club's own matches 1..3
MID = (20, 22)          # placebo window, same width
BOTTOM_N = 6            # "weak opponent" definition for S2
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "promoted_upset.csv")


def load_with_dates(cache_dir=None):
    """Season results WITH dates, so a club's own matchday can be ordered.

    `history.download_seasons` drops Date, and matchday cannot be reconstructed without
    it — calendar matchweek is not the same thing as a club's own Nth fixture once games
    are rearranged, and it is the club's own first three that the hypothesis is about.
    """
    cache_dir = cache_dir or _os.path.join(config.SCRATCH, "fd_hist")
    frames = []
    for p in sorted(glob.glob(_os.path.join(cache_dir, "E0_*.csv"))):
        code = _os.path.basename(p)[3:-4]
        d = pd.read_csv(p, encoding="latin-1")
        need = {"HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"}
        if not need.issubset(d.columns):
            continue
        d = d.dropna(subset=list(need))
        d["season"] = code
        d["date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        d = d.dropna(subset=["date"])
        d["HomeTeam"] = d["HomeTeam"].replace(history.NAME_NORM)
        d["AwayTeam"] = d["AwayTeam"].replace(history.NAME_NORM)
        frames.append(d[["season", "date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]])
    out = pd.concat(frames, ignore_index=True)
    out["yr"] = out["season"].map(history.season_start_year)
    return out.sort_values(["yr", "date"]).reset_index(drop=True)


def long_form(d):
    """One row per club-match, with the club's own matchday index and points won."""
    h = d.rename(columns={"HomeTeam": "team", "AwayTeam": "opp",
                          "FTHG": "gf", "FTAG": "ga"}).assign(home=1)
    a = d.rename(columns={"AwayTeam": "team", "HomeTeam": "opp",
                          "FTAG": "gf", "FTHG": "ga"}).assign(home=0)
    L = pd.concat([h[["season", "yr", "date", "team", "opp", "gf", "ga", "home"]],
                   a[["season", "yr", "date", "team", "opp", "gf", "ga", "home"]]],
                  ignore_index=True)
    L["pts"] = np.where(L.gf > L.ga, 3, np.where(L.gf == L.ga, 1, 0))
    L = L.sort_values(["yr", "team", "date"])
    L["md"] = L.groupby(["yr", "team"]).cumcount() + 1
    return L.reset_index(drop=True)


def season_tables(L):
    """Per club-season: games, points, ppg, and final rank."""
    t = (L.groupby(["yr", "season", "team"], as_index=False)
           .agg(games=("pts", "size"), pts=("pts", "sum"),
                gf=("gf", "sum"), ga=("ga", "sum")))
    t["ppg"] = t["pts"] / t["games"]
    t["gd"] = t["gf"] - t["ga"]
    t["rank"] = t.groupby("yr")["pts"].rank(ascending=False, method="min")
    t["n_clubs"] = t.groupby("yr")["team"].transform("size")
    return t


def build(L, tab, window, opponent="promoted"):
    """One row per treated-eligible club-season.

    `opponent="promoted"` -> lost to a club newly promoted INTO this season.
    `opponent="bottom"`   -> lost to a club that finished bottom-`BOTTOM_N` last season
                             and was NOT promoted (S2's specificity contrast).
    """
    prev = {}          # yr -> set of clubs in the previous season
    prior = {}         # (yr, team) -> prior-season ppg
    bottom = {}        # yr -> set of clubs bottom-N last season, excluding promoted
    for yr in sorted(tab.yr.unique()):
        p = tab[tab.yr == yr - 1]
        prev[yr] = set(p["team"])
        for _, r in p.iterrows():
            prior[(yr, r["team"])] = r["ppg"]
        if len(p):
            bottom[yr] = set(p.nlargest(BOTTOM_N, "rank")["team"])
    lo, hi = window
    rows = []
    for yr in sorted(tab.yr.unique()):
        if not prev.get(yr):
            continue                                   # no predecessor season
        promoted = set(tab[tab.yr == yr]["team"]) - prev[yr]
        if opponent == "promoted":
            targets = promoted
        else:
            targets = bottom.get(yr, set()) - promoted
        if not targets:
            continue
        S = L[L.yr == yr]
        for team, g in S.groupby("team"):
            if team in promoted:
                continue                               # promoted clubs are not the population
            pp = prior.get((yr, team))
            if pp is None:
                continue
            w = g[(g.md >= lo) & (g.md <= hi)]
            faced = int(w["opp"].isin(targets).sum())
            lost = int(((w["opp"].isin(targets)) & (w["pts"] == 0)).sum())
            rest = g[g.md > hi]
            if not len(rest):
                continue
            rows.append({"yr": yr, "season": g["season"].iloc[0], "team": team,
                         "prior_ppg": pp, "faced": faced, "treated": int(lost > 0),
                         "n_lost": lost,
                         "window_ppg": float(w["pts"].mean()) if len(w) else np.nan,
                         "rest_ppg": float(rest["pts"].mean()),
                         "rest_games": len(rest),
                         "rest_gd": float((rest["gf"] - rest["ga"]).mean())})
    return pd.DataFrame(rows)


def ols_cluster(y, X, groups):
    """OLS with cluster-robust (CR0) standard errors. Returns (beta, se, t, p, n)."""
    from scipy import stats as st
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    resid = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gval in np.unique(groups):
        m = groups == gval
        Xg, ug = X[m], resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    dof_c = G / max(G - 1, 1)
    V = XtX_inv @ meat @ XtX_inv * dof_c
    se = np.sqrt(np.diag(V))
    t = b / se
    p = 2 * (1 - st.t.cdf(np.abs(t), df=max(G - 1, 1)))
    return b, se, t, p, len(y)


def fit(df, label, alpha=0.05):
    """Primary estimator: rest_ppg ~ treated + prior_ppg, clustered by season."""
    d = df.dropna(subset=["rest_ppg", "prior_ppg"])
    X = np.column_stack([np.ones(len(d)), d["treated"].values, d["prior_ppg"].values])
    b, se, t, p, n = ols_cluster(d["rest_ppg"].values, X, d["yr"].values)
    # minimum detectable effect at 80% power, two-sided
    mde = 2.8 * se[1]
    print(f"\n  {label}")
    print(f"    n={n} club-seasons, treated={int(d.treated.sum())}, "
          f"seasons={d.yr.nunique()}")
    print(f"    beta(treated)  {b[1]:+.4f} ppg   SE {se[1]:.4f}   t={t[1]:+.2f}   "
          f"p={p[1]:.3f}   {'SIGNIFICANT' if p[1] < alpha else 'null'} at alpha={alpha}")
    print(f"    gamma(prior_ppg) {b[2]:+.4f}     (the control doing its job)")
    print(f"    raw gap (no control): "
          f"{d[d.treated == 1]['rest_ppg'].mean() - d[d.treated == 0]['rest_ppg'].mean():+.4f} ppg")
    print(f"    MDE at 80% power: {mde:.3f} ppg "
          f"= {mde * 35:.1f} points over a 35-match remainder")
    return {"label": label, "n": n, "treated": int(d.treated.sum()),
            "beta": b[1], "se": se[1], "t": t[1], "p": p[1],
            "gamma_prior": b[2], "mde_ppg": mde}


def loso(df):
    """Leave-one-season-out: is the sign stable, or carried by one year?"""
    signs, betas = [], []
    for yr in sorted(df.yr.unique()):
        d = df[(df.yr != yr)].dropna(subset=["rest_ppg", "prior_ppg"])
        X = np.column_stack([np.ones(len(d)), d["treated"].values, d["prior_ppg"].values])
        b, *_ = ols_cluster(d["rest_ppg"].values, X, d["yr"].values)
        betas.append(b[1]); signs.append(np.sign(b[1]))
    same = int(max((np.array(signs) > 0).sum(), (np.array(signs) < 0).sum()))
    print(f"    leave-one-season-out: beta in [{min(betas):+.4f}, {max(betas):+.4f}], "
          f"sign stable in {same}/{len(signs)} folds")
    # NB stability is NOT evidence against noise. A pooled fluke is also LOO-stable —
    # the same caveat euro_qualifying_fade records.
    return {"loso_min": min(betas), "loso_max": max(betas),
            "loso_sign_stable": same, "loso_folds": len(signs)}


def main():
    d = load_with_dates()
    L = long_form(d)
    tab = season_tables(L)
    print(f"[data] {tab.yr.nunique()} seasons, {len(L):,} club-matches, "
          f"{tab.yr.min()}-{tab.yr.max()}")

    H1 = build(L, tab, EARLY, "promoted")
    S1 = build(L, tab, MID, "promoted")
    S2 = build(L, tab, EARLY, "bottom")

    print("\n" + "=" * 74)
    print("H1 (PRIMARY) — lost to a promoted club in own matches 1-3")
    print("=" * 74)
    r1 = fit(H1, "rest-of-season PPG"); r1.update(loso(H1))

    print("\n" + "=" * 74)
    print("SECONDARY (Bonferroni alpha = 0.025)")
    print("=" * 74)
    r2 = fit(S1, "S1 PLACEBO: lost to a promoted club in matches 20-22", alpha=0.025)
    r3 = fit(S2, "S2 SPECIFICITY: lost to a prior-bottom-6 club in matches 1-3",
             alpha=0.025)

    # exposure check: does simply FACING promoted opposition early matter?
    print("\n" + "=" * 74)
    print("CONTEXT")
    print("=" * 74)
    e = H1.groupby("faced")["rest_ppg"].agg(["size", "mean"])
    print("  rest-of-season PPG by number of promoted opponents faced in matches 1-3:")
    print(e.to_string())
    t1 = H1[H1.treated == 1]
    print(f"\n  treated clubs: {len(t1)} over {H1.yr.nunique()} seasons "
          f"({len(t1)/H1.yr.nunique():.1f} per season)")
    print(f"  their prior-season PPG {t1.prior_ppg.mean():.3f} vs "
          f"{H1[H1.treated==0].prior_ppg.mean():.3f} for the rest "
          f"-> treated clubs were ALREADY weaker, which is what the control removes")

    # EXPLORATORY, declared as such: the prompting case was a strong club, which the
    # pooled test does not isolate. n=7 treated, so this is case evidence, not a test.
    print("\n" + "=" * 74)
    print("EXPLORATORY (not pre-registered) — strong clubs only")
    print("=" * 74)
    qq = H1.prior_ppg.quantile(0.75)
    strong = H1[H1.prior_ppg >= qq]
    Xs = np.column_stack([np.ones(len(strong)), strong.treated.values,
                          strong.prior_ppg.values])
    bs, ses, ts, ps, ns = ols_cluster(strong.rest_ppg.values, Xs, strong.yr.values)
    print(f"  prior PPG >= {qq:.2f}: n={ns}, treated={int(strong.treated.sum())}")
    print(f"  beta {bs[1]:+.4f} ppg  SE {ses[1]:.4f}  p={ps[1]:.3f}   "
          f"MDE {2.8*ses[1]:.3f} ppg = {2.8*ses[1]*35:.1f} pts")
    print("  -> underpowered by an order of magnitude; see the docstring for the seven cases")
    r4 = {"label": "EXPLORATORY strong clubs (prior ppg >= q75)", "n": ns,
          "treated": int(strong.treated.sum()), "beta": bs[1], "se": ses[1],
          "t": ts[1], "p": ps[1], "gamma_prior": bs[2], "mde_ppg": 2.8 * ses[1]}

    res = pd.DataFrame([r1, r2, r3, r4])
    res.to_csv(OUT, index=False)
    print(f"\n[wrote] {OUT}")
    return res


if __name__ == "__main__":
    main()
