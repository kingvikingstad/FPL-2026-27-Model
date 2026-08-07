"""
fpl_xp_model.py
================
A component-based expected-points (xP) regression model for Fantasy Premier
League, grounded in (a) the official FPL scoring rules and (b) FBref player &
team datapoints (https://fbref.com/en/comps/9/Premier-League-Stats).

DESIGN PHILOSOPHY
-----------------
FPL points are a *deterministic, position-weighted function of underlying match
events* whose coefficients are fixed by the rules. So we do NOT regress total
points directly. Instead:

        E[Points_it | X] = sum_k  c_k * E[N_k,it | X]

where c_k are the KNOWN scoring weights (see SCORING below) and N_k are scoring
events (goals, assists, clean sheet, saves, defensive-contribution threshold,
concessions, cards, bonus, appearance). We estimate each E[N_k | X] with an
appropriate GLM on FBref rates, then compose. A final OLS of realised points on
the predicted components gives interpretable inference and calibration checks.

STAGES
  1. Minutes / availability (two-part gate) -> appearance points, scales everything
  2. Goals        (Poisson/NegBin on npxG90 + shots, + penalties)
  3. Assists      (Poisson/NegBin on xA90 + key passes)
  4. Team goals conceded -> clean-sheet prob + concession penalty (Poisson)
  5. GK saves     (Poisson on opponent SoT * save%)
  6. Defensive Contribution (threshold prob on CBIT[+recoveries] per-90)
  7. Bonus (BPS)  (reduced-form Poisson on BPS drivers)
  8. Compose xP, then calibration/inference OLS with robust SEs

Dependencies: numpy, pandas, scipy, scikit-learn. Optional: soccerdata (FBref
ingest), statsmodels (richer inference; a numpy robust-OLS fallback is built in).

Author: (built for an econometrician who lives in per-90 space)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats
from sklearn.linear_model import PoissonRegressor, LogisticRegression

# =============================================================================
# 1. FPL SCORING RULES  (transcribed directly from the rules document)
# =============================================================================
# position codes: GK, DEF, MID, FWD
GOAL_POINTS      = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_PTS  = {"GK": 4,  "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS    = 3
SAVES_PER_POINT  = 3           # 1 pt per 3 saves (GK)
PEN_SAVE_PTS     = 5
PEN_MISS_PTS     = -2
YELLOW_PTS       = -1
RED_PTS          = -3
OWN_GOAL_PTS     = -2
DEFCON_PTS       = 2           # defensive-contribution award
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}  # CBIT (DEF) / CBIT+rec (MID,FWD)
APP_SHORT_PTS    = 1           # 1-59 mins
APP_LONG_PTS     = 2           # 60+ mins
# concessions: -1 for every 2 goals conceded while on pitch (GK/DEF only)


def concession_penalty_expectation(mu_conceded: np.ndarray) -> np.ndarray:
    """E[ -floor(G/2) ] for G ~ Poisson(mu). Applies to GK/DEF.
    Computed exactly over the Poisson pmf (truncated tail is negligible)."""
    mu = np.asarray(mu_conceded, float)
    kmax = int(max(20, np.nanmax(mu) * 4 + 10)) if mu.size else 20
    ks = np.arange(0, kmax)
    pmf = stats.poisson.pmf(ks[None, :], mu[:, None])          # (n, kmax)
    penalty = -np.floor(ks / 2.0)                              # (kmax,)
    return (pmf * penalty[None, :]).sum(axis=1)


def clean_sheet_prob(mu_conceded: np.ndarray) -> np.ndarray:
    """P(0 conceded) = exp(-mu) under Poisson. (Minutes gate applied separately.)"""
    return np.exp(-np.asarray(mu_conceded, float))


def threshold_prob_poisson(rate_per90: np.ndarray, exp_minutes: np.ndarray,
                           threshold: int) -> np.ndarray:
    """P(count >= threshold) where count ~ Poisson(rate_per90 * minutes/90).
    Used for the Defensive Contribution award."""
    mu = np.asarray(rate_per90, float) * np.asarray(exp_minutes, float) / 90.0
    return stats.poisson.sf(threshold - 1, mu)   # sf(t-1) = P(X >= t)


# =============================================================================
# 2. DATA INGEST  (FBref via soccerdata, with a documented CSV fallback)
# =============================================================================
def load_fbref(season: str = "2024-2025", league: str = "ENG-Premier League"):
    """Pull the FBref tables this model consumes. Requires `pip install soccerdata`
    and outbound network. Returns a dict of DataFrames keyed by stat type.

    We read the same tables surfaced on the league page you linked:
    standard, shooting, passing, goal_shot_creation, defense, playing_time,
    misc, keeper, keeper_adv  (plus team-level standard for opponent strength).
    """
    import soccerdata as sd
    fb = sd.FBref(leagues=league, seasons=season)
    stat_types = ["standard", "shooting", "passing", "goal_shot_creation",
                  "defense", "playing_time", "misc", "keeper", "keeper_adv"]
    players = {s: fb.read_player_season_stats(stat_type=s) for s in stat_types}
    teams   = fb.read_team_season_stats(stat_type="standard")
    return {"players": players, "teams": teams}


# The engineered player-match / player-season frame the model expects. Build this
# from FBref columns; names below are the canonical inputs used downstream.
REQUIRED_COLUMNS = [
    "player", "team", "pos",           # pos in {GK,DEF,MID,FWD}
    "minutes",                          # minutes played in window
    "npxg", "xg", "sh", "sot", "pk", "pkatt",   # shooting / penalties
    "xa", "kp", "sca", "gca",          # creation
    "tkl", "int", "blocks", "clr",     # defensive actions (CBIT parts)
    "recoveries",                      # miscellaneous (for MID/FWD defcon)
    "saves", "sota",                   # goalkeeping (saves, shots on target against)
    "team_xga90", "opp_xg90",          # team defensive rating / opponent attack
    "is_home",                         # fixture context (1/0)
    "starts", "matches",               # for the minutes model
]


# =============================================================================
# 3. FEATURE ENGINEERING:  per-90, recency weights, empirical-Bayes shrinkage
# =============================================================================
def per90(count, minutes, eps=1e-9):
    return np.asarray(count, float) / (np.asarray(minutes, float) / 90.0 + eps)


def empirical_bayes_shrink(rate, minutes, pos, prior_by_pos=None, K=None):
    """Shrink a noisy per-90 rate toward its positional prior mean.

        shrunk = w*rate + (1-w)*prior ,  w = minutes / (minutes + K)

    K is a pseudo-count (minutes) governing shrinkage strength; if not supplied
    it is estimated from the between/within variance ratio (method of moments).
    This is the single most important correction for FBref rate stats computed
    over small samples (early season / low-minute players)."""
    rate = np.asarray(rate, float); minutes = np.asarray(minutes, float)
    pos = np.asarray(pos)
    out = rate.copy()
    for p in np.unique(pos):
        m = pos == p
        r, mn = rate[m], minutes[m]
        if prior_by_pos and p in prior_by_pos:
            prior = prior_by_pos[p]
        else:
            wsum = mn.sum()
            prior = (r * mn).sum() / wsum if wsum > 0 else np.nanmean(r)
        if K is None:
            # MoM: K ~ within-var / between-var, in minutes units (rough, robust)
            between = np.nanvar(r) + 1e-9
            within  = np.nanmean(np.abs(r - prior)) + 1e-9
            k = float(np.clip(within / between * 90.0, 90.0, 1500.0))
        else:
            k = K
        w = mn / (mn + k)
        out[m] = w * r + (1 - w) * prior
    return out


def recency_weight(gw_index, current_gw, half_life=5.0):
    """Exponential recency weights for pooling multiple gameweeks:
    weight = 0.5 ** ((current_gw - gw)/half_life). Feed as sample_weight."""
    return 0.5 ** ((current_gw - np.asarray(gw_index, float)) / half_life)


def fixture_multiplier(opp_xga90, league_mean_xga90):
    """Multiplicative fixture-difficulty factor for attacking output:
    easier defence (higher xGA allowed) -> >1. Centered at league mean."""
    return np.asarray(opp_xga90, float) / (league_mean_xga90 + 1e-9)


# =============================================================================
# 4. ROBUST OLS  (numpy; HC1 standard errors) -- inference layer w/o statsmodels
# =============================================================================
@dataclass
class OLSResult:
    names: list
    beta: np.ndarray
    se: np.ndarray
    tstat: np.ndarray
    pval: np.ndarray
    r2: float
    n: int

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "coef": self.beta, "robust_se": self.se,
            "t": self.tstat, "p": self.pval,
        }, index=self.names)


def ols_robust(X: np.ndarray, y: np.ndarray, names=None) -> OLSResult:
    """OLS with HC1 heteroskedasticity-robust standard errors. X should already
    include an intercept column if wanted."""
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # HC1 sandwich
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    cov = XtX_inv @ S @ XtX_inv * (n / (n - k))
    se = np.sqrt(np.diag(cov))
    tstat = beta / se
    pval = 2 * stats.t.sf(np.abs(tstat), df=n - k)
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    names = names or [f"x{i}" for i in range(k)]
    return OLSResult(names, beta, se, tstat, pval, r2, n)


# =============================================================================
# 5. COMPONENT MODELS
# =============================================================================
@dataclass
class FPLxPModel:
    half_life: float = 5.0
    use_negbin_check: bool = True
    league_mean_xga90: float = 1.35          # ~PL average; recomputed from data if provided
    fitted_: dict = field(default_factory=dict)

    # ---- 5a. Minutes / availability: two-part gate ------------------------
    def fit_minutes(self, df):
        """P(start) via logistic on recent start share; expected minutes via a
        simple conditional-mean model. Returns nothing; stores fitted objects."""
        X = df[["start_share", "avail"]].fillna(0).values
        y_start = (df["starts"] / df["matches"].clip(lower=1) > 0.5).astype(int).values
        clf = LogisticRegression(max_iter=1000).fit(X, y_start) \
            if len(np.unique(y_start)) > 1 else None
        self.fitted_["minutes_clf"] = clf
        # expected minutes given typical role (data-driven, shrunk)
        self.fitted_["mean_min_start"] = df.loc[y_start == 1, "min_per_start"].mean() \
            if (y_start == 1).any() else 80.0
        return self

    def predict_minutes(self, df):
        clf = self.fitted_.get("minutes_clf")
        X = df[["start_share", "avail"]].fillna(0).values
        p_start = clf.predict_proba(X)[:, 1] if clf is not None else \
            df["start_share"].clip(0, 1).values
        mean_start_min = self.fitted_.get("mean_min_start", 80.0)
        exp_min = p_start * mean_start_min + (1 - p_start) * df["sub_min"].fillna(15).values
        # Prefer directly-observed historical rates (leak-free: built from GWs<t);
        # fall back to a role-based heuristic only where history is missing.
        p60 = df["p60_hist"].values if "p60_hist" in df else (p_start*0.9 + (1-p_start)*0.10)
        p_app = df["papp_hist"].values if "papp_hist" in df else \
                (p_start*0.99 + (1-p_start)*df["sub_app_rate"].fillna(0.5).values)
        p60 = np.clip(p60, 0, 1); p_app = np.clip(np.maximum(p_app, p60), 0, 1)
        return pd.DataFrame({"exp_min": exp_min, "p60": p60, "p_app": p_app,
                             "p_start": p_start}, index=df.index)

    # ---- generic Poisson GLM with optional overdispersion diagnostic -------
    def _poisson(self, X, y, sample_weight=None, offset_log_min=None):
        # sklearn PoissonRegressor has no offset; fold minutes in as a feature
        # by modelling counts with log(minutes/90) appended, coef free-estimated.
        glm = PoissonRegressor(alpha=1e-4, max_iter=2000)
        glm.fit(X, y, sample_weight=sample_weight)
        mu = glm.predict(X)
        # overdispersion (Cameron-Trivedi): regress (var proxy) on mu
        if self.use_negbin_check and len(y) > 20:
            z = ((y - mu) ** 2 - y) / np.maximum(mu, 1e-6)
            alpha_hat = np.average(z, weights=sample_weight) if sample_weight is not None else z.mean()
            od = float(alpha_hat)   # >0 suggests NegBin would fit better
        else:
            od = np.nan
        return glm, od

    # ---- 5b. Goals ---------------------------------------------------------
    # Poisson with exposure handled as a free-coefficient log(minutes/90) feature
    # (the exposure coef estimates ~1). At prediction we plug in EXPECTED minutes,
    # so no post-hoc rescale is needed and predictions cannot double-count minutes.
    def _log90(self, minutes):
        return np.log(np.clip(np.asarray(minutes, float), 5.0, None) / 90.0)

    @staticmethod
    def _lograte(x, c=0.05):
        """log-transform a non-negative rate so the Poisson log-link becomes a
        multiplicative (constant-elasticity) model: exp(beta*log x) = x**beta."""
        return np.log(np.clip(np.asarray(x, float), 0, None) + c)

    def _build_goal_X(self, df, minutes_col):
        return np.column_stack([
            self._lograte(df["npxg90_s"]), self._lograte(df["sh90_s"]),
            self._lograte(df["fixmult"], c=0.01),
            df["is_home"].fillna(0).values, df["is_fwd"].values,
            df["is_mid"].values, df["is_def"].values,
            self._log90(df[minutes_col]),
        ])

    def fit_goals(self, df, sw=None):
        X = self._build_goal_X(df, "minutes"); y = df["goals"].values
        glm, od = self._poisson(X, y, sw)
        self.fitted_["goals"] = glm; self.fitted_["goals_overdisp"] = od
        return self

    def predict_goals_per_match(self, df):
        return np.clip(self.fitted_["goals"].predict(self._build_goal_X(df, "exp_min")), 0, 2.5)

    # ---- 5c. Assists -------------------------------------------------------
    def _build_assist_X(self, df, minutes_col):
        return np.column_stack([
            self._lograte(df["xa90_s"]), self._lograte(df["kp90_s"]),
            self._lograte(df["sca90_s"]), self._lograte(df["fixmult"], c=0.01),
            df["is_home"].fillna(0).values, self._log90(df[minutes_col]),
        ])

    def fit_assists(self, df, sw=None):
        X = self._build_assist_X(df, "minutes"); y = df["assists"].values
        glm, od = self._poisson(X, y, sw)
        self.fitted_["assists"] = glm; self.fitted_["assists_overdisp"] = od
        return self

    def predict_assists_per_match(self, df):
        return np.clip(self.fitted_["assists"].predict(self._build_assist_X(df, "exp_min")), 0, 2.5)

    # ---- 5d. Team goals conceded (clean sheets + concession penalty) -------
    def predict_conceded_mu(self, df):
        """Expected goals conceded by the player's team, opponent- and venue-
        adjusted. mu = team_xga90 blended with opponent attack, * home factor."""
        team_def = df["team_xga90"].fillna(self.league_mean_xga90).values
        opp_att  = df["opp_xg90"].fillna(self.league_mean_xga90).values
        home_adj = np.where(df["is_home"].values == 1, 0.90, 1.10)  # home concede less
        mu = np.sqrt(np.clip(team_def, 1e-6, None) * np.clip(opp_att, 1e-6, None)) * home_adj
        return mu

    # ---- 5e. GK saves ------------------------------------------------------
    def predict_saves(self, df):
        """Expected saves = opponent shots-on-target-against * save%, scaled by
        minutes. Points = floor(saves/3); we take E[floor(S/3)] via Poisson."""
        exp_sota = df["sota90"].fillna(3.2).values * (df["exp_min"] / 90.0).values
        save_pct = df["save_pct"].fillna(0.69).values
        mu_saves = exp_sota * save_pct
        # E[floor(S/3)] under Poisson(mu_saves)
        kmax = int(max(30, np.nanmax(mu_saves) * 4 + 10))
        ks = np.arange(kmax)
        pmf = stats.poisson.pmf(ks[None, :], mu_saves[:, None])
        pts = np.floor(ks / SAVES_PER_POINT)
        return (pmf * pts[None, :]).sum(axis=1)

    # ---- 5f. Defensive Contribution ---------------------------------------
    def predict_defcon_points(self, df):
        """2 pts if DEF hits 10 CBIT, or MID/FWD hits 12 CBIT+recoveries, in a
        match. Expected value = 2 * P(count >= threshold)."""
        cbit90 = df["cbit90_s"].values.copy()
        # MID/FWD also count recoveries toward the 12 threshold
        is_midfwd = df["pos"].isin(["MID", "FWD"]).values
        rate = np.where(is_midfwd, df["cbitr90_s"].values, cbit90)
        thr = df["pos"].map(DEFCON_THRESHOLD).fillna(999).astype(float).values
        p = threshold_prob_poisson(rate, df["exp_min"].values, thr.astype(int))
        # GK effectively never qualifies
        p = np.where(df["pos"].values == "GK", 0.0, p)
        return DEFCON_PTS * p

    # ---- 5g. Bonus (reduced-form BPS proxy) --------------------------------
    def fit_bonus(self, df, sw=None):
        # trained on realised events (learning the BPS->bonus mapping); at predict
        # time the SAME columns are supplied as model-predicted expectations.
        feats = ["goals", "assists", "cs_flag", "tkl90_s", "sot90_s", "passcomp_tier"]
        X = df[feats].fillna(0).values
        y = df["bonus"].values
        glm, _ = self._poisson(X, y, sw)
        self.fitted_["bonus"] = (glm, feats)
        return self

    def predict_bonus_from_expected(self, exp_goals, exp_assists, p_cs, df):
        glm, feats = self.fitted_["bonus"]
        X = np.column_stack([
            exp_goals, exp_assists, p_cs,
            df["tkl90_s"].fillna(0).values, df["sot90_s"].fillna(0).values,
            df["passcomp_tier"].fillna(2).values,
        ])
        return np.clip(glm.predict(X), 0, 3.0)

    # ---- 6. COMPOSE expected points ---------------------------------------
    def predict_xp(self, df):
        mn = self.predict_minutes(df)
        d = df.join(mn)
        pos = d["pos"].values

        goals   = self.predict_goals_per_match(d)
        assists = self.predict_assists_per_match(d)
        mu_con  = self.predict_conceded_mu(d)
        p_cs    = clean_sheet_prob(mu_con) * d["p60"].values    # need 60+ mins
        conc    = np.where(np.isin(pos, ["GK", "DEF"]),
                           concession_penalty_expectation(mu_con) * d["p60"].values, 0.0)
        saves   = np.where(pos == "GK", self.predict_saves(d), 0.0)
        defcon  = self.predict_defcon_points(d)
        bonus   = self.predict_bonus_from_expected(goals, assists, p_cs, d)

        goal_pts = goals * np.vectorize(GOAL_POINTS.get)(pos)
        cs_pts   = p_cs * np.vectorize(CLEAN_SHEET_PTS.get)(pos)
        app_pts  = APP_SHORT_PTS * (d["p_app"].values - d["p60"].values).clip(0) \
                   + APP_LONG_PTS * d["p60"].values
        # disciplinary priors (expected cards from per-90 history)
        card_pts = YELLOW_PTS * d.get("yc90", pd.Series(0, index=d.index)).values \
                   * (d["exp_min"].values / 90.0)

        xp = (app_pts + goal_pts + assists * ASSIST_POINTS + cs_pts
              + saves + defcon + bonus + conc + card_pts)

        return pd.DataFrame({
            "xP": xp, "app_pts": app_pts, "goal_pts": goal_pts,
            "assist_pts": assists * ASSIST_POINTS, "cs_pts": cs_pts,
            "save_pts": saves, "defcon_pts": defcon, "bonus_pts": bonus,
            "concede_pts": conc, "card_pts": card_pts,
            "exp_goals": goals, "exp_assists": assists, "p_cs": p_cs,
            "exp_min": d["exp_min"].values,
        }, index=df.index)

    # ---- 7. Calibration / inference OLS -----------------------------------
    def calibration_regression(self, realized_points, xp_frame):
        """Regress realised FPL points on predicted xP (with intercept). A well
        calibrated model gives slope ~ 1, intercept ~ 0. Robust (HC1) SEs."""
        y = np.asarray(realized_points, float)
        X = np.column_stack([np.ones_like(y), xp_frame["xP"].values])
        return ols_robust(X, y, names=["intercept", "xP"])

    def component_diagnostic_regression(self, realized_points, xp_frame):
        """Regress realised points on each predicted component; slopes near 1
        and jointly significant indicate the scoring-rule mapping is unbiased."""
        cols = ["app_pts", "goal_pts", "assist_pts", "cs_pts", "save_pts",
                "defcon_pts", "bonus_pts", "concede_pts"]
        y = np.asarray(realized_points, float)
        X = np.column_stack([np.ones_like(y)] + [xp_frame[c].values for c in cols])
        return ols_robust(X, y, names=["intercept"] + cols)


# =============================================================================
# 8. WALK-FORWARD BACKTEST (no leakage: fit on GWs < t, predict GW t)
# =============================================================================
def walk_forward_backtest(panel: pd.DataFrame, gw_col="gw", start_gw=6):
    """panel: engineered player-gameweek rows with realised 'points' and all
    feature columns. Returns per-GW error metrics and pooled calibration."""
    records, preds = [], []
    for t in sorted(panel[gw_col].unique()):
        if t < start_gw:
            continue
        train = panel[panel[gw_col] < t]
        test  = panel[panel[gw_col] == t]
        sw = recency_weight(train[gw_col].values, t)
        m = FPLxPModel()
        m.fit_minutes(train).fit_goals(train, sw).fit_assists(train, sw).fit_bonus(train, sw)
        xp = m.predict_xp(test)
        err = test["points"].values - xp["xP"].values
        records.append({"gw": t, "n": len(test),
                        "MAE": np.mean(np.abs(err)),
                        "RMSE": np.sqrt(np.mean(err ** 2)),
                        "bias": np.mean(err),
                        "spearman": stats.spearmanr(test["points"].values,
                                                    xp["xP"].values).correlation})
        tmp = test[[gw_col]].copy(); tmp["points"] = test["points"].values
        tmp["xP"] = xp["xP"].values
        preds.append(tmp)
    metrics = pd.DataFrame(records)
    pooled = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return metrics, pooled


if __name__ == "__main__":
    print("fpl_xp_model loaded. See run_demo.py for a synthetic end-to-end test,")
    print("and load_fbref() to ingest real FBref data in a networked environment.")
