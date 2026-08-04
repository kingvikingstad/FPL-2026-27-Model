"""
style_matchup.py — team/manager play-style axes and style x style interaction
effects on goal supply.  OFF BY DEFAULT.  Nothing here is validated yet.

WHAT THIS DOES
--------------
1. Builds continuous style scores per team-season from event-derived metrics
   (defensive line height, press intensity, defensive-action mix, build-up
   directness, territorial control).
2. Fits, on top of the existing att/dfn factorisation, an interaction term:

       log lam_ij = mu + home + att[i] - dfn[j] + theta' z_ij

   where z_ij are OWN-style x OPPONENT-style products.
3. Tests whether theta survives (a) team fixed effects, (b) multiple testing,
   (c) an out-of-sample horse race against market-implied lambda.

THE CENTRAL IDENTIFICATION POINT — read before using
----------------------------------------------------
Style is (near) constant for a team within a season.  att[i] and dfn[i] are
therefore COLLINEAR with any style MAIN effect.  "High-line teams concede
more" is NOT estimable here and never was: it is absorbed by dfn[i].  Every
public claim of that form is re-labelled team quality.

Only the INTERACTION is identified, off within-team variation in opponent
style.  `check_identification()` enforces this by refusing to fit any spec
whose style main effects are not in the null space of the FE design.

POWER — why the outcome variable is goals-adjacent, not goals
-------------------------------------------------------------
Simulated MDE at 80% power, Bonferroni over 4 scalar interactions, 380
matches, regressor standardised to unit variance (theta = log goal-rate
change per 1sd of the interaction):

    outcome                       lambda     SE      MDE
    goals                           1.35   0.031    10.8%
    shots on target                 4.50   0.017     5.7%
    shots                          12.50   0.010     3.4%
    final-third entries            45.00   0.002     0.8%

Plausible true tactical effects are ~3-8%.  A single season of GOALS cannot
detect them.  So the model is fitted on shot COUNTS and shot QUALITY
separately and recomposed into lambda:

    lam = E[shots] * E[xG per shot]

An interaction that raises volume while lowering quality nets to zero on
goals; fitting the two margins separately is what makes that visible.

ATTENUATION
-----------
Style scores are noisy season averages.  Measurement error enters the
interaction MULTIPLICATIVELY (both sides are noisy), so attenuation goes as
reliability^2.  Simulated recovery of a true theta=0.10 over 3 seasons:
reliability 0.96 -> 93%; 0.89 -> 86%; 0.80 -> 77%; 0.67 -> 64%.  Divide any
MDE above by the recovery fraction.  Use split-half reliability from
`style_reliability()` to report this rather than guessing.

DATA REQUIRED (none of it is in this repo; all free)
----------------------------------------------------
  FBref/Opta team-match logs : tackles by pitch third, clearances, blocks,
      interceptions, passes by length, progressive distance, touches by third
  Understat                  : PPDA, deep completions, xG per shot
  FBref possession-sequence  : passes per sequence, direct-attack count,
      sequence start distance from own goal
  Offsides won (any source)  : cleanest free proxy for defensive line height

Run:  python style_matchup.py --selftest
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Style axes
# ---------------------------------------------------------------------------
# Each axis is a signed combination of observable per-match metrics. Signs are
# fixed by definition, NOT fitted -- these are measurement models, not
# hypotheses. The hypotheses live in Theta.
#
# Deliberately NOT included: manager-identity dummies. A manager effect that
# is constant across opponents is, again, absorbed by att/dfn. Manager
# identity matters here only as a way to TRANSFER a style prior to a club
# where the current regime has no match history (see `transfer_prior`).

STYLE_AXES = {
    # + = higher defensive line, further from own goal
    "line_height": {
        "offsides_won_p90":          +1.0,   # purest free signal
        "def_action_dist_from_goal": +1.0,   # mean distance of T+I+clearances
        "clearances_p90":            -0.6,   # deep block correlate
        "opp_pass_completion_own_3rd": -0.4,
    },
    # + = more aggressive pressing
    "press": {
        "ppda":                      -1.0,   # inverted: low PPDA = high press
        "tackles_att_3rd_share":     +1.0,
        "opp_long_pass_share":       +0.5,   # pressed teams go long
    },
    # + = more direct / faster vertical progression
    "directness": {
        "passes_per_sequence":       -1.0,
        "direct_attacks_p90":        +1.0,   # >=3 passes, >=60% forward, fast
        "progressive_dist_per_pass": +0.8,
        "long_pass_share":           +0.6,
    },
    # + = more territorial control
    "control": {
        "field_tilt":                +1.0,   # final-third touch share (beats poss%)
        "possession":                +0.5,
        "seq_start_dist_from_goal":  +0.4,   # build from deep vs. win it high
    },
}

# --- Theta: the hypotheses. Each entry is (own_axis, opp_axis). --------------
# Restricted a priori to 4 scalar terms, NOT a free 4x4 matrix. The power
# table above is why: a free Theta costs 16 tests and ~4pp of MDE for
# interactions with no mechanism behind them.
#
# Sign column is the PRIOR direction, recorded so that a wrong-signed
# significant estimate is treated as a red flag (likely leakage from team
# quality), not as a finding.
CANDIDATE_INTERACTIONS = [
    # attacker's directness vs. defender's line height -> balls in behind
    ("directness", "line_height", +1, "long ball beats the high line"),
    # attacker's build-from-deep vs. defender's press -> forced turnovers high
    ("control",    "press",       -1, "press disrupts deep build-up"),
    # attacker's control vs. defender's (negative) line height -> low block
    ("control",    "line_height", +1, "possession blunted by a deep block"),
    # attacker's press vs. defender's control -> counter-press regains
    ("press",      "control",     +1, "high press punishes possession teams"),
]


def build_style_scores(team_match: pd.DataFrame,
                       game_state_filter: bool = True) -> pd.DataFrame:
    """team_match: one row per team-match with the raw metrics named in
    STYLE_AXES. Returns team-season z-scored style loadings.

    game_state_filter: restrict to minutes while the score is LEVEL. This is
    not optional in spirit -- a team 1-0 up drops its line and stops pressing,
    so unfiltered season averages measure results, not intent, and the good
    teams look 'deep and direct' purely because they lead more often. If your
    source has no state split, pass the first-half-while-level subset.
    """
    d = team_match.copy()
    if game_state_filter:
        if "state" not in d.columns:
            raise ValueError(
                "game_state_filter=True needs a 'state' column (level/ahead/"
                "behind). Unfiltered style scores are confounded with results."
            )
        d = d[d.state == "level"]

    out = {}
    for axis, weights in STYLE_AXES.items():
        missing = [m for m in weights if m not in d.columns]
        if missing:
            raise ValueError(f"axis '{axis}' missing metrics: {missing}")
        z = pd.DataFrame(index=d.index)
        for metric, w in weights.items():
            col = d[metric].astype(float)
            z[metric] = w * (col - col.mean()) / (col.std(ddof=0) + 1e-12)
        out[axis] = z.mean(axis=1)
    S = pd.DataFrame(out)
    S[["team", "season"]] = d[["team", "season"]]
    S = S.groupby(["team", "season"]).mean().reset_index()
    for axis in STYLE_AXES:
        S[axis] = (S[axis] - S[axis].mean()) / (S[axis].std(ddof=0) + 1e-12)
    return S


def style_reliability(team_match: pd.DataFrame, n_splits: int = 200,
                      seed: int = 0) -> pd.Series:
    """Split-half reliability of each axis within team-season. Feed the result
    into the attenuation correction -- do not assume the scores are clean."""
    rng = np.random.default_rng(seed)
    rel = {a: [] for a in STYLE_AXES}
    for _ in range(n_splits):
        d = team_match.copy()
        d["half"] = rng.integers(0, 2, len(d))
        try:
            a = build_style_scores(d[d.half == 0], game_state_filter=False)
            b = build_style_scores(d[d.half == 1], game_state_filter=False)
        except ValueError:
            continue
        m = a.merge(b, on=["team", "season"], suffixes=("_a", "_b"))
        for ax in STYLE_AXES:
            r = np.corrcoef(m[f"{ax}_a"], m[f"{ax}_b"])[0, 1]
            rel[ax].append(2 * r / (1 + r))          # Spearman-Brown
    return pd.Series({a: float(np.nanmean(v)) for a, v in rel.items()})


# ---------------------------------------------------------------------------
# 2. Estimation
# ---------------------------------------------------------------------------
def _fe_design(att_idx, dfn_idx, is_home, T):
    X = np.zeros((len(att_idx), 2 + 2 * (T - 1)))
    X[:, 0] = 1.0
    X[:, 1] = is_home
    for r, (i, j) in enumerate(zip(att_idx, dfn_idx)):
        if i < T - 1:
            X[r, 2 + i] = 1.0
        if j < T - 1:
            X[r, 2 + (T - 1) + j] = -1.0
    return X


def check_identification(X_fe: np.ndarray, Z: np.ndarray, tol=1e-6) -> dict:
    """Refuse to fit if the style block is (near-)spanned by the FE block.
    Returns the share of each z column's variance left after projecting out
    the fixed effects. Anything below ~5% is not an interaction, it is team
    quality wearing a costume."""
    P = X_fe @ np.linalg.pinv(X_fe)
    resid = Z - P @ Z
    keep = (resid ** 2).sum(0) / ((Z - Z.mean(0)) ** 2).sum(0).clip(tol)
    return {"resid_var_share": keep,
            "identified": bool(np.all(keep > 0.05))}


def poisson_irls(X, y, ridge=0.0, iters=100):
    b = np.zeros(X.shape[1])
    R = ridge * np.eye(X.shape[1])
    R[0, 0] = R[1, 1] = 0.0                      # never penalise mu / home
    for _ in range(iters):
        lam = np.exp(np.clip(X @ b, -12, 12))
        H = (X * lam[:, None]).T @ X + R + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - lam) - R @ b)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    lam = np.exp(np.clip(X @ b, -12, 12))
    H = (X * lam[:, None]).T @ X + R + 1e-9 * np.eye(X.shape[1])
    return b, np.linalg.inv(H)


def fit_interaction(matches: pd.DataFrame, styles: pd.DataFrame,
                    outcome: str = "shots", ridge: float = 1.0,
                    interactions=None) -> pd.DataFrame:
    """matches: long form, one row per team-match, columns
       [team, opp, season, is_home, <outcome>].
    Fit on 'shots' first, then repeat on 'xg_per_shot' -- lambda is the
    product, and the two margins can move in opposite directions."""
    inter = interactions or CANDIDATE_INTERACTIONS
    S = styles.set_index(["team", "season"])
    teams = sorted(set(matches.team) | set(matches.opp))
    idx = {t: i for i, t in enumerate(teams)}

    Zcols, names = [], []
    for own_ax, opp_ax, sign, label in inter:
        own = np.array([S.loc[(r.team, r.season), own_ax] for r in matches.itertuples()])
        opp = np.array([S.loc[(r.opp, r.season), opp_ax] for r in matches.itertuples()])
        z = own * opp
        Zcols.append((z - z.mean()) / (z.std(ddof=0) + 1e-12))   # unit variance
        names.append(f"{own_ax} x opp_{opp_ax}")
    Z = np.column_stack(Zcols)

    X_fe = _fe_design([idx[t] for t in matches.team],
                      [idx[t] for t in matches.opp],
                      matches.is_home.astype(float).values, len(teams))
    ident = check_identification(X_fe, Z)
    if not ident["identified"]:
        raise RuntimeError(
            "style block is collinear with team fixed effects "
            f"(residual variance shares {np.round(ident['resid_var_share'], 3)}). "
            "Whatever this would estimate is team quality, not matchup.")

    X = np.hstack([X_fe, Z])
    b, cov = poisson_irls(X, matches[outcome].values.astype(float), ridge=ridge)
    k = len(names)
    th, se = b[-k:], np.sqrt(np.diag(cov)[-k:])
    m = len(inter)
    return pd.DataFrame({
        "interaction": names,
        "prior_sign": [s for _, _, s, _ in inter],
        "mechanism": [lab for _, _, _, lab in inter],
        "theta": th, "se": se, "z": th / se,
        "pct_per_sd": 100 * (np.exp(th) - 1),
        "sig_bonferroni": np.abs(th / se) > abs(_z_crit(0.05 / m)),
        "resid_var_share": ident["resid_var_share"],
    })


def _z_crit(alpha):
    """Two-sided normal critical value without scipy."""
    p = 1 - alpha / 2
    # Acklam-style rational approximation, plenty accurate here
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------------------
# 3. The decisive test
# ---------------------------------------------------------------------------
def beats_the_market(matches: pd.DataFrame, styles: pd.DataFrame,
                     theta: pd.DataFrame, market_lambda_col: str = "lam_mkt"):
    """Bookmakers price tactical matchups. The prior should be that most of
    this is already in the odds. The only question that matters for the
    pipeline is whether z_ij predicts the residual

        log(goals_actual) - log(lam_market)

    If it does not, the interaction is real football and worthless alpha:
    do NOT add it to TeamModel, because it will double-count against the
    market-calibrated clean-sheet engine that is already validated at
    GA r=0.89 / CS r=0.93.

    Returns (coef, se) of z on the market residual, per interaction.
    """
    S = styles.set_index(["team", "season"])
    rows = []
    for (own_ax, opp_ax, sign, label), th in zip(CANDIDATE_INTERACTIONS, theta.theta):
        own = np.array([S.loc[(r.team, r.season), own_ax] for r in matches.itertuples()])
        opp = np.array([S.loc[(r.opp, r.season), opp_ax] for r in matches.itertuples()])
        z = own * opp
        z = (z - z.mean()) / (z.std(ddof=0) + 1e-12)
        # Poisson score test of z given a fixed market offset
        lam = matches[market_lambda_col].values.astype(float)
        y = matches["goals"].values.astype(float)
        num = float(z @ (y - lam))
        den = float(np.sqrt((z ** 2) @ lam))
        rows.append({"interaction": f"{own_ax} x opp_{opp_ax}",
                     "score_z": num / den if den > 0 else np.nan,
                     "adds_over_market": abs(num / den) > 2.5 if den > 0 else False})
    return pd.DataFrame(rows)


def transfer_prior(new_club: str, manager_prev_club_styles: pd.DataFrame,
                   n_matches_observed: int, k: float = 25.0) -> pd.Series:
    """The ONLY place manager identity legitimately enters: a club with a new
    regime has no current-style observations, so borrow the manager's style
    from their previous club and shrink toward league mean as the new club's
    own matches accumulate. Weight = n / (n + k).

    This is a variance statement ("we don't know this club's style yet"), not
    a directional one -- consistent with the regime-discount logic already in
    starter_prior.py, and unlike the rejected style multipliers it makes no
    claim about which way returns move.
    """
    w = n_matches_observed / (n_matches_observed + k)
    prev = manager_prev_club_styles.mean()
    return (1 - w) * prev          # caller blends with the observed new-club score


# ---------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(7)
    T, seasons = 20, 3
    teams = [f"T{i:02d}" for i in range(T)]
    Sm = rng.normal(0, 1, (T, len(STYLE_AXES)))
    Sm -= Sm.mean(0); Sm /= Sm.std(0)
    styles = pd.DataFrame(Sm, columns=list(STYLE_AXES))
    styles["team"] = teams
    styles["season"] = "S1"
    att = rng.normal(0, .3, T); dfn = rng.normal(0, .3, T)

    ai = list(STYLE_AXES).index("directness")
    bi = list(STYLE_AXES).index("line_height")
    TRUE = 0.06                                    # 6.2% per sd, on SHOTS

    rows = []
    for _ in range(380 * seasons):
        i, j = rng.integers(0, T), rng.integers(0, T)
        if i == j:
            continue
        for (a_, d_, hm) in ((i, j, 1.0), (j, i, 0.0)):   # both perspectives
            z = Sm[a_, ai] * Sm[d_, bi]
            lam = np.exp(np.log(12.5) + .05 * hm + att[a_] - dfn[d_] + TRUE * z)
            rows.append({"team": teams[a_], "opp": teams[d_], "season": "S1",
                         "is_home": hm, "shots": rng.poisson(lam)})
    m = pd.DataFrame(rows)

    res = fit_interaction(m, styles, outcome="shots", ridge=0.0)
    hit = res[res.interaction == "directness x opp_line_height"].iloc[0]
    print(res[["interaction", "theta", "se", "z", "pct_per_sd",
               "sig_bonferroni"]].round(4).to_string(index=False))
    assert abs(hit.theta - TRUE) < 3 * hit.se, "true effect not recovered"
    assert hit.sig_bonferroni, "true effect not detected at Bonferroni"
    others = res[res.interaction != "directness x opp_line_height"]
    assert not others.sig_bonferroni.any(), "false positive on a null channel"
    print(f"\nSELFTEST OK: recovered {hit.theta:.4f} vs true {TRUE:.4f} "
          f"(se {hit.se:.4f}); 3 null channels correctly non-significant.")

    # The guard must reject a style MAIN effect -- the thing everyone reports.
    S2 = styles.set_index(["team", "season"])
    main = np.array([S2.loc[(r.team, r.season), "line_height"]
                     for r in m.itertuples()])[:, None]
    tix = {t: i for i, t in enumerate(teams)}
    Xfe = _fe_design([tix[t] for t in m.team], [tix[t] for t in m.opp],
                     m.is_home.values.astype(float), T)
    rep = check_identification(Xfe, main)
    share = float(rep["resid_var_share"][0])
    assert not rep["identified"], "guard failed to reject a style main effect"
    print(f"GUARD OK: 'high line concedes more' main effect retains only "
          f"{100*share:.2f}% of its variance after team FE -> unidentified, "
          f"as it must be.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        print(__doc__)
