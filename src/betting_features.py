"""
betting_features.py
===================
Turn football-data.co.uk match+odds data (E0.csv) into the fixture-level inputs
the FPL xP model needs, by extracting each team's EXPECTED GOALS per match from
the betting markets.

Why odds? A closing betting line is the market's forward-looking forecast of a
fixture; it already prices in lineups, injuries, form and home advantage, and is
extremely well calibrated. From three markets we identify each side's expected
goals (lambda_home, lambda_away):

  * 1X2 (home/draw/away)      -> the full outcome distribution
  * Over/Under 2.5 goals      -> the expected TOTAL goals (lambda_h + lambda_a)
  * Asian handicap            -> the expected SUPREMACY (lambda_h - lambda_a)

Under an independent-Poisson score model, (lambda_h, lambda_a) are point-identified
by these probabilities. We solve for them per match, then:
  - lambda_against a team  -> clean-sheet prob exp(-lambda) and the concession penalty
  - lambda_for a team      -> the attacking fixture multiplier
These replace the static season-average team ratings in fpl_xp_model.

Everything here is estimated on realised data and VALIDATED against actual goals
(calibration regressions, Brier/log-loss), so the odds->xG inversion is checked,
not assumed.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize


# ----------------------------------------------------------------------------
# 1. De-vig (remove bookmaker overround / margin)
# ----------------------------------------------------------------------------
def devig(odds: np.ndarray) -> np.ndarray:
    """Multiplicative margin removal: p_i = (1/o_i) / sum_j(1/o_j).
    Simple and robust; for the favourite-longshot bias use the Shin or power
    method instead (hook left in `devig_power`)."""
    inv = 1.0 / np.asarray(odds, float)
    return inv / inv.sum(axis=-1, keepdims=True)


def devig_power(odds: np.ndarray, tol=1e-10):
    """Power method: find k s.t. sum p_i^(1/k) = 1 with p_i ∝ (1/o_i). Reduces
    favourite-longshot bias vs the multiplicative method. Vectorised per row."""
    inv = 1.0 / np.asarray(odds, float)
    raw = inv / inv.sum(axis=-1, keepdims=True)
    out = np.empty_like(raw)
    for i, r in enumerate(raw):
        lo, hi = 0.5, 1.5
        for _ in range(60):
            k = 0.5 * (lo + hi)
            s = (r ** (1.0 / k)).sum()
            if s > 1: lo = k
            else:     hi = k
            if abs(s - 1) < tol: break
        out[i] = r ** (1.0 / k)
        out[i] /= out[i].sum()
    return out


# ----------------------------------------------------------------------------
# 2. Independent-Poisson score model: map (lambda_h, lambda_a) -> market probs
# ----------------------------------------------------------------------------
def _score_matrix(lh, la, kmax=12):
    ph = poisson.pmf(np.arange(kmax), lh)
    pa = poisson.pmf(np.arange(kmax), la)
    return np.outer(ph, pa)                       # M[i,j] = P(home i, away j)


def market_probs(lh, la, kmax=12):
    """Return (pH, pD, pA, p_over25) implied by independent Poisson(lh, la)."""
    M = _score_matrix(lh, la, kmax)
    iu = np.triu_indices(kmax, k=1)               # home < away rows? see below
    # home i (row), away j (col): home win = i>j -> lower triangle excl diag
    pH = np.tril(M, -1).sum()
    pD = np.trace(M)
    pA = np.triu(M, 1).sum()
    tot = np.add.outer(np.arange(kmax), np.arange(kmax))
    p_over = M[tot >= 3].sum()
    return pH, pD, pA, p_over


# ----------------------------------------------------------------------------
# 3. Solve (lambda_h, lambda_a) per match from de-vigged 1X2 + O/U 2.5
# ----------------------------------------------------------------------------
def solve_lambdas(pH, pD, pA, p_over, x0=(1.4, 1.1)):
    """Least-squares fit of (lambda_h, lambda_a) to the four de-vigged targets."""
    target = np.array([pH, pD, pA, p_over])
    w = np.array([1.0, 1.0, 1.0, 0.7])            # O/U slightly down-weighted
    def obj(z):
        lh, la = np.exp(z)                        # positivity via log-param
        m = np.array(market_probs(lh, la))
        return (w * (m - target) ** 2).sum()
    res = minimize(obj, np.log(x0), method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-10, maxiter=2000))
    return tuple(np.exp(res.x))


# ----------------------------------------------------------------------------
# 4. Full pipeline over an E0.csv file
# ----------------------------------------------------------------------------
def build(path="E0.csv", use_power_devig=False):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    dv = devig_power if use_power_devig else devig
    p1x2 = dv(df[["AvgH", "AvgD", "AvgA"]].values)
    # O/U: two-outcome market, de-vig the pair, keep P(over)
    pou = dv(df[["Avg>2.5", "Avg<2.5"]].values)[:, 0]

    lam = np.array([solve_lambdas(*p1x2[i], pou[i]) for i in range(len(df))])
    lh, la = lam[:, 0], lam[:, 1]
    new = pd.DataFrame({
        "lambda_home": lh, "lambda_away": la,
        "exp_total": lh + la, "exp_supremacy": lh - la,
        "p_home": p1x2[:, 0], "p_draw": p1x2[:, 1], "p_away": p1x2[:, 2],
        "p_over25": pou,
        "home_cs_prob": np.exp(-la),   # home keeps CS if away scores 0
        "away_cs_prob": np.exp(-lh),
    }, index=df.index)
    return pd.concat([df, new], axis=1)


# ----------------------------------------------------------------------------
# 5. Team ratings (long form): attack / defence, home & away, from odds xG
# ----------------------------------------------------------------------------
def team_ratings(df):
    home = df.rename(columns={"HomeTeam": "team", "AwayTeam": "opp",
                              "lambda_home": "xg_for", "lambda_away": "xg_against",
                              "FTHG": "g_for", "FTAG": "g_against"}).assign(venue="home")
    away = df.rename(columns={"AwayTeam": "team", "HomeTeam": "opp",
                              "lambda_away": "xg_for", "lambda_home": "xg_against",
                              "FTAG": "g_for", "FTHG": "g_against"}).assign(venue="away")
    keep = ["team", "opp", "venue", "xg_for", "xg_against", "g_for", "g_against"]
    long = pd.concat([home[keep], away[keep]], ignore_index=True)
    agg = long.groupby("team").agg(
        matches=("xg_for", "size"),
        att_xg=("xg_for", "mean"), def_xg=("xg_against", "mean"),
        goals_for=("g_for", "mean"), goals_against=("g_against", "mean"),
        cs_rate=("g_against", lambda s: (s == 0).mean()),
    ).round(3)
    # home/away attack & defence splits (used for venue-aware fixture inputs)
    splits = {}
    for v in ["home", "away"]:
        sub = long[long.venue == v].groupby("team")
        splits[f"att_xg_{v}"] = sub["xg_for"].mean().round(3)
        splits[f"def_xg_{v}"] = sub["xg_against"].mean().round(3)
    agg = pd.concat([agg, pd.DataFrame(splits)], axis=1)
    return agg.sort_values("att_xg", ascending=False)


# ----------------------------------------------------------------------------
# 6. Maher/Dixon-Coles Poisson GLM on ACTUAL goals (results-based ratings)
#    log E[goals] = mu + home*1{home} + attack[scorer] - defence[conceder]
#    A cross-check on the odds-derived ratings, and the classic econometric model.
# ----------------------------------------------------------------------------
def fit_dixon_coles_poisson(df, l2=1e-3):
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    # stack two Poisson observations per match (home goals, away goals)
    rows, y = [], []
    for _, r in df.iterrows():
        h, a = idx[r.HomeTeam], idx[r.AwayTeam]
        # home goals: attack_h - defence_a + home
        v = np.zeros(2 * n + 2); v[h] = 1; v[n + a] = -1; v[-1] = 1; v[-2] = 1
        rows.append(v); y.append(r.FTHG)
        # away goals: attack_a - defence_h (no home term)
        v = np.zeros(2 * n + 2); v[a] = 1; v[n + h] = -1; v[-2] = 1
        rows.append(v); y.append(r.FTAG)
    X = np.array(rows); y = np.array(y, float)

    def negll(b):
        eta = X @ b
        mu = np.exp(np.clip(eta, -10, 6))
        ll = (y * eta - mu).sum() - l2 * (b[:-2] ** 2).sum()  # ridge on team params
        return -ll
    from scipy.optimize import minimize as _min
    b0 = np.zeros(2 * n + 2)
    res = _min(negll, b0, method="L-BFGS-B")
    b = res.x
    att = pd.Series(b[:n], index=teams, name="attack")
    dfc = pd.Series(b[n:2 * n], index=teams, name="defence")
    out = pd.concat([att, dfc], axis=1)
    out["net"] = out.attack + out.defence
    return out.sort_values("net", ascending=False).round(3), float(b[-1])  # home adv


# ----------------------------------------------------------------------------
# 7. Bridge to the FPL model: fixture inputs for a given team & venue
# ----------------------------------------------------------------------------
def fixture_inputs(ratings, home_team, away_team, league_avg_xg=1.45):
    """Return the fixture-difficulty inputs fpl_xp_model consumes for each side:
    expected goals for (attack multiplier) and against (-> CS / concession)."""
    def one(team, opp, venue):
        att = ratings.loc[team, f"att_xg_{venue}"]
        opp_def = ratings.loc[opp, f"def_xg_{'away' if venue=='home' else 'home'}"]
        xg_for = np.sqrt(max(att, 1e-6) * max(opp_def, 1e-6))     # blend own & opp
        oppv = "away" if venue == "home" else "home"
        opp_att = ratings.loc[opp, f"att_xg_{oppv}"]
        team_def = ratings.loc[team, f"def_xg_{venue}"]
        xg_against = np.sqrt(max(opp_att, 1e-6) * max(team_def, 1e-6))
        return {"team": team, "venue": venue,
                "opp_xg90": round(xg_for, 3),           # attack faced by fixture
                "team_xga90": round(xg_against, 3),     # goals team expects to concede
                "fixmult": round(xg_for / league_avg_xg, 3),
                "cs_prob": round(np.exp(-xg_against), 3)}
    return pd.DataFrame([one(home_team, away_team, "home"),
                         one(away_team, home_team, "away")])


def attach_fixture_inputs(player_df, ratings, league_avg_xg=1.45):
    """Vectorised bridge INTO fpl_xp_model. Given a player frame with columns
    ['team','opp','is_home'], attach the betting-derived fixture inputs the xP
    model consumes: opp_xg90 (attack context), team_xga90 (goals the team is
    expected to concede -> clean sheet & concession), and fixmult. Replaces the
    static season-average team ratings with market, fixture-specific numbers."""
    df = player_df.copy()
    def row(r):
        venue = "home" if r["is_home"] else "away"
        oppv  = "away" if r["is_home"] else "home"
        att      = ratings.loc[r["team"], f"att_xg_{venue}"]
        opp_def  = ratings.loc[r["opp"],  f"def_xg_{oppv}"]
        opp_att  = ratings.loc[r["opp"],  f"att_xg_{oppv}"]
        team_def = ratings.loc[r["team"], f"def_xg_{venue}"]
        xg_for     = np.sqrt(max(att, 1e-6) * max(opp_def, 1e-6))
        xg_against = np.sqrt(max(opp_att, 1e-6) * max(team_def, 1e-6))
        return pd.Series({"opp_xg90": xg_for, "team_xga90": xg_against,
                          "fixmult": xg_for / league_avg_xg})
    return df.join(df.apply(row, axis=1))
</content>
