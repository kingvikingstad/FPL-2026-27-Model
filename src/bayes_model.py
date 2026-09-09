from __future__ import annotations
import config
"""
bayes_model.py  —  Hierarchical Bayesian FPL projection for 2026/27
===================================================================
Produces POSTERIOR-PREDICTIVE distributions (mean, credible intervals, ceiling,
floor, P(threshold)) for player points over any horizon of the real 26/27
schedule. Three Bayesian layers, all implemented in numpy/scipy (no PPL needed):

 1. TEAM strengths  — hierarchical Poisson (attack/defence/home) fit on 25/26
    results, with a Gaussian prior centred on the betting-odds ratings (prior
    fusion). MAP + Laplace posterior. Promoted teams (Coventry, Hull, Ipswich)
    have NO 25/26 data, so they receive informative promoted-team priors; their
    posterior = prior until 26/27 games arrive (honest Bayesian updating).
 2. PLAYER rates    — conjugate Gamma–Poisson for per-90 attacking involvement &
    defensive-contribution, partially pooled to position means, with a
    new-season reversion discount on the evidence (last season informs but does
    not dictate next season).
 3. AVAILABILITY    — Beta–Binomial start probability; the dominant, and noisiest,
    driver of multi-week totals.

Composition is Monte-Carlo: draw team strengths, player rates and availability,
push each draw through the FPL scoring rules across the player's fixtures, and
accumulate. Design choices are the ones the earlier analysis validated: xG over
goals, light shrinkage, short-memory form, bonus loads onto goals, and the
forward schedule with double/blank handling.
"""
import numpy as np, pandas as pd
from scipy import stats
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import betting_features as bf
from schedule_2627 import schedule, PROMOTED
from fpl_xp_model import (GOAL_POINTS, CLEAN_SHEET_PTS, ASSIST_POINTS,
                          DEFCON_THRESHOLD, DEFCON_PTS)

rng = np.random.default_rng(7)
# Seed for the PER-PLAYER generators used inside project(). Runners already rebind
# `rng` before each call; this is the equivalent knob for the player layer.
PROJECT_SEED = 7


def _player_key(p):
    """A stable identifier for one player's random stream.

    Must not change when unrelated players are added, removed or reordered, so it keys on
    `player_code` — the project's stable join key — and falls back to a CRC of name+club.
    Python's built-in hash() is salted per process and would silently break reproducibility
    between runs, so it is not used.
    """
    code = p.get("player_code") if hasattr(p, "get") else None
    if code is not None and code == code and str(code) not in ("", "None"):
        try:
            return int(code)
        except (TypeError, ValueError):
            pass
    import zlib
    return int(zlib.crc32(f"{p.get('web_name')}|{p.get('team')}".encode("utf-8")))


def _player_rng(p, seed=None, gw=0):
    """An independent, reproducible generator for one player IN ONE GAMEWEEK WINDOW.

    THE POINT. project() used to draw every player from one shared sequential stream, so
    the draws a player received depended on every player simulated before him. That is not
    merely an ordering quirk: numpy's beta/gamma/poisson use rejection sampling, so the
    number of raw bits consumed depends on the PARAMETER VALUES. Changing one player's
    start_a therefore re-randomised everyone downstream — measured at 48 players moved,
    19 of them at other clubs, from a single perturbation.

    The board was still reproducible given identical input, so the determinism check
    passed, but no A/B was clean: a measured difference was the real effect plus a
    re-randomisation term. Per-player streams remove that term entirely.

    Team-level correlation is unaffected. It lives in `tsamp`, drawn once outside this
    function and indexed identically for every player, so draw s still means the same
    team-strength world for everyone. What becomes independent across players is exactly
    what the model already assumes is conditionally independent given team strength.

    THE `gw` ARGUMENT  [ADDED 2026-09-08]
    -------------------------------------
    The stream was keyed on (seed, player_code) alone. `gw_board` calls
    `project(gw, gw, ...)` once per gameweek, so every one of those 38 calls handed a
    player the IDENTICAL stream and therefore the identical draws: measured, the start
    indicator vector for draw s was bitwise the same across gameweek calls, agreement
    1.0000. In path s a player started every week of the season or none of them.

    Per-gameweek marginals were unaffected, which is why the board invariants and the
    determinism check both passed. What was wrong is any aggregation ACROSS gameweeks
    built from these draws: perfect comonotonicity in the start dimension inflates a
    multi-week variance by roughly H instead of sqrt(H). That is latent while DUMP_DRAWS
    covers one gameweek and becomes load-bearing for the transfer/chip solver, which is
    the first consumer that sums draws over a horizon.

    Keying on the window restores independence across gameweeks while keeping the
    property the per-player streams were introduced for: the stream is still a pure
    function of (seed, player_code, gw), so a board is still reproducible and an A/B is
    still free of the re-randomisation term.
    """
    return np.random.default_rng([int(PROJECT_SEED if seed is None else seed),
                                  _player_key(p), int(gw)])
LEAGUE_MU = 1.40
BET_NAME = {"Man Utd": "Man United", "Spurs": "Tottenham"}
# per-position bonus that rides along with a goal (recovered in the earlier
# reverse-engineering: DEF +1.0, MID +1.3, FWD +1.3 pts/goal above rulebook)
# Recovered from 25/26, then RESCALED for the 2026/27 BPS changes
# (CBI 1/3 was 1/2; being-tackled -1 removed; +1 per non-penalty big-chance save).
# Position multipliers measured from the full-season bonus recompute (fpl.red/bonus,
# 137 bonus-relevant players): DEF -10.4%, MID +4.5%, FWD +4.1%, GK +1.9%.
BPS_2627_MULT = {"GK": 1.019, "DEF": 0.896, "MID": 1.045, "FWD": 1.041}
_BONUS_PER_GOAL_2526 = {"GK": 1.0, "DEF": 1.0, "MID": 1.3, "FWD": 1.3}
_BONUS_PER_CS_2526 = {"GK": 0.5, "DEF": 0.4, "MID": 0.0, "FWD": 0.0}
BONUS_PER_GOAL = {k: v * BPS_2627_MULT[k] for k, v in _BONUS_PER_GOAL_2526.items()}
BONUS_PER_CS = {k: v * BPS_2627_MULT[k] for k, v in _BONUS_PER_CS_2526.items()}
BONUS_PER_ASSIST = 0.6 * 1.045

# =============================================================================
# LAYER 1 — hierarchical Bayesian team model (MAP + Laplace posterior)
# =============================================================================
# --- team hyperparameters -----------------------------------------------------
# These were hard-coded guesses. They are now read from data/team_hyperparams.json,
# estimated from 31 seasons of results (see scripts/calibrate_team_history.py and
# docs/TEAM_FIXTURE_FINDINGS.md). GUESSES is kept as the fallback when the file is
# absent, and as the A/B baseline: set FPL_TEAM_HYPER=guess to restore the old values
# without editing code.
GUESSES = {"home_prior": (0.26, 0.08), "promoted_att": (-0.20, 0.30),
           "promoted_def": (-0.22, 0.30), "revert": 0.85, "season_sd": 0.15,
           "home_early_discount": 0.0, "home_early_last_gw": 3}


def _load_hyperparams():
    import json
    if _os.environ.get("FPL_TEAM_HYPER", "").lower() in ("guess", "0", "off"):
        return dict(GUESSES), "hard-coded guesses (FPL_TEAM_HYPER=guess)"
    try:
        with open(config.TEAM_HYPERPARAMS, encoding="utf-8") as fh:
            d = json.load(fh)
        out = dict(GUESSES)
        for k in ("revert", "season_sd"):
            out[k] = float(d[k])
        out["home_early_discount"] = float(d.get("home_early_discount", 0.0))
        out["home_early_last_gw"] = int(d.get("home_early_last_gw", 3))
        for k in ("home_prior", "promoted_att", "promoted_def"):
            out[k] = tuple(d[k])
        n = d.get("_provenance", {}).get("n_seasons", "?")
        return out, f"calibrated on {n} seasons ({_os.path.basename(config.TEAM_HYPERPARAMS)})"
    except Exception:
        return dict(GUESSES), "hard-coded guesses (no calibration file)"


HYPER, HYPER_SOURCE = _load_hyperparams()


class TeamModel:
    def __init__(self, prior_sd=0.35, home_prior=None,
                 promoted_att=None, promoted_def=None,
                 promoted_per_club=None):
        # Defaults come from HYPER (calibrated); pass explicitly to override.
        self.prior_sd = prior_sd
        self.home_prior = home_prior if home_prior is not None else HYPER["home_prior"]
        self.promoted_att = (promoted_att if promoted_att is not None
                             else HYPER["promoted_att"])
        self.promoted_def = (promoted_def if promoted_def is not None
                             else HYPER["promoted_def"])
        # optional {club: centred_log_strength} from Elo -> club-specific priors
        self.promoted_per_club = promoted_per_club or {}

    def fit(self, e0_path=None, clubelo=None, clubelo_weight=0.35):
        bet = bf.build(e0_path or config.E0_RECON)
        rat = bf.team_ratings(bet)                       # betting-odds team ratings
        self.ratings = rat
        teams = sorted(set(bet.HomeTeam) | set(bet.AwayTeam))
        self.idx = {t: i for i, t in enumerate(teams)}; self.teams = teams
        T = len(teams)
        # betting-informed prior means (centred log ratings)
        la = np.log(rat["att_xg"].reindex(teams).values)
        ld = np.log(rat["def_xg"].reindex(teams).values)
        m_att = la - la.mean()
        m_def = -(ld - ld.mean())                        # lower xGA -> higher strength
        # optional ClubElo blend into the prior means (dynamic, cross-season)
        if clubelo is not None:
            try:
                e = clubelo.set_index("team")["Elo"].astype(float)
                le = np.log(e / e.mean())
                for i, t in enumerate(teams):
                    if t in le.index:
                        m_att[i] = (1 - clubelo_weight) * m_att[i] + clubelo_weight * le[t]
                        m_def[i] = (1 - clubelo_weight) * m_def[i] + clubelo_weight * le[t]
                self.clubelo_used = int(sum(t in le.index for t in teams))
            except Exception:
                self.clubelo_used = 0
        else:
            self.clubelo_used = 0
        # parameter vector theta = [mu, home, att(T), def(T)]
        n = 2 + 2 * T
        m0 = np.concatenate([[np.log(LEAGUE_MU), self.home_prior[0]], m_att, m_def])
        prec = np.zeros(n)
        prec[0] = 1 / 0.5 ** 2; prec[1] = 1 / self.home_prior[1] ** 2
        prec[2:] = 1 / self.prior_sd ** 2
        P0 = np.diag(prec)
        # design matrix: two Poisson rows per match
        rows, y = [], []
        for _, r in bet.iterrows():
            h, a = self.idx[r.HomeTeam], self.idx[r.AwayTeam]
            v = np.zeros(n); v[0] = 1; v[1] = 1; v[2 + h] = 1; v[2 + T + a] = -1
            rows.append(v); y.append(r.FTHG)
            v = np.zeros(n); v[0] = 1; v[2 + a] = 1; v[2 + T + h] = -1
            rows.append(v); y.append(r.FTAG)
        X = np.array(rows); y = np.array(y, float)
        # Newton MAP on penalised Poisson log-likelihood
        theta = m0.copy()
        for _ in range(60):
            eta = X @ theta; mu = np.exp(np.clip(eta, -8, 6))
            grad = X.T @ (mu - y) + P0 @ (theta - m0)
            H = X.T @ (X * mu[:, None]) + P0
            step = np.linalg.solve(H, grad)
            theta -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.theta = theta
        self.cov = np.linalg.inv(H)                      # Laplace posterior covariance
        self.T = T; self.n = n
        return self

    def _promoted_prior_draw(self, S):
        # weaker attack, leakier defence, wide uncertainty (centred log scale).
        # Defaults are guesses; supply history-calibrated values via __init__.
        att = rng.normal(self.promoted_att[0], self.promoted_att[1], S)
        dfn = rng.normal(self.promoted_def[0], self.promoted_def[1], S)
        return att, dfn

    def sample_2627(self, S=1500, revert=None, season_sd=None):
        """Draw S joint samples of (mu, home, att, def) for the 20 teams of
        2026/27. Returning teams: Laplace posterior, reverted to mean and with
        added between-season variance. Promoted teams: promoted prior.

        `revert`/`season_sd` default to the calibrated HYPER values. The previous
        default of 0.85 reverted teams toward the mean roughly four times harder than
        31 seasons support (measured 0.963, IV-corrected for errors-in-variables —
        the naive OLS slope of 0.680 is the attenuated one that made 0.85 look
        plausible)."""
        revert = HYPER["revert"] if revert is None else revert
        season_sd = HYPER["season_sd"] if season_sd is None else season_sd
        draw = rng.multivariate_normal(self.theta, self.cov, size=S)  # (S, n)
        mu, home = draw[:, 0], draw[:, 1]
        att = draw[:, 2:2 + self.T]; dfn = draw[:, 2 + self.T:]
        sched, _ = schedule()
        teams_2627 = sorted(set(sched.home) | set(sched.away))
        A = np.zeros((S, len(teams_2627))); D = np.zeros((S, len(teams_2627)))
        idx2 = {t: i for i, t in enumerate(teams_2627)}
        extra = rng.normal(0, season_sd, (S, len(teams_2627)))
        for t, j in idx2.items():
            if t in PROMOTED:
                if t in self.promoted_per_club:
                    mu_t = self.promoted_per_club[t]
                    sd_t = max(self.promoted_att[1], 0.12)
                    A[:, j] = rng.normal(mu_t, sd_t, S)
                    D[:, j] = rng.normal(mu_t, sd_t, S)
                else:
                    a, d = self._promoted_prior_draw(S)
                    A[:, j] = a; D[:, j] = d
            else:
                i = self.idx[t]
                A[:, j] = revert * att[:, i] + extra[:, j]
                D[:, j] = revert * dfn[:, i] + extra[:, j]
        self.teams_2627 = teams_2627; self.idx2 = idx2
        return dict(mu=mu, home=home, att=A, dfn=D, teams=teams_2627, idx=idx2)


# =============================================================================
# LAYER 2 & 3 — player rate (Gamma–Poisson) and availability (Beta–Binomial)
# =============================================================================
def player_posteriors(csv=config.FPL_DATA_STATS,
                      revert=0.70, min_minutes=450):
    d = pd.read_csv(csv)
    d["pos"] = d.element_type.map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    d["team"] = d.team_name.replace(BET_NAME)
    agg = d.groupby(["id", "web_name", "pos", "team"]).agg(
        minutes=("minutes", "sum"),
        npxgi=("non_penalty_expected_goal_involvements", "sum"),
        xa=("expected_assists", "sum"),
        defcon=("defensive_contribution", "sum"),
        starts=("minutes", lambda s: (s >= 60).sum()),
        apps=("minutes", lambda s: (s > 0).sum()),
        games=("minutes", "size"),
        own=("selected_by_percent", "last"),
        cost=("now_cost", "last"),
    ).reset_index()
    agg = agg[agg.minutes >= min_minutes].copy()
    nnf = agg.minutes / 90.0
    # position prior means (per-90) for partial pooling
    for col in ["npxgi", "xa", "defcon"]:
        rate = agg[col] / nnf
        pri = agg.groupby("pos")[col].transform("sum") / agg.groupby("pos")["minutes"].transform("sum") * 90
        agg[f"{col}_prior"] = pri
        # Gamma posterior with reverted evidence: alpha0 from prior mean * k0,
        # evidence discounted by `revert` to widen for the new season.
        k0 = 3.0                                    # prior strength in 90s
        alpha0 = pri * k0; beta0 = k0
        agg[f"{col}_alpha"] = alpha0 + revert * agg[col]
        agg[f"{col}_beta"] = beta0 + revert * nnf
    # availability: Beta on P(start | in squad), reverted; sub-appearance rate
    a0, b0 = 2.0, 2.0
    agg["start_a"] = a0 + revert * agg.starts
    agg["start_b"] = b0 + revert * (agg.games - agg.starts).clip(lower=0)
    agg["sub_app_rate"] = (agg.apps - agg.starts).clip(lower=0) / agg.games
    return agg


# =============================================================================
# COMPOSITION — Monte-Carlo posterior predictive over a horizon
# =============================================================================
# --- minutes given an appearance ------------------------------------------------
# `start` is drawn from Beta(start_a, start_b), whose counts were fitted on
# `minutes >= 60` — so the "start" branch means "played 60+", and played60 (hence
# clean-sheet eligibility and the 2-point appearance) is right by construction.
# The EXPOSURE was not: m90 = mins/90 multiplies attacking involvement, penalty xG and
# DefCon counts, and assuming 90 minutes for everyone who clears 60 inflates all three.
#
# [VERIFIED 2026-08-11, studies/minutes_distribution.py, 23,059 appearances over 24/25
# and 25/26] mean minutes GIVEN 60+ is 85.3, not 90, and the shortfall is strongly
# positional — only 52% of midfielders and 43% of forwards who clear the hour finish the
# match, against 82% of defenders and 99.5% of keepers. So the inflation lands hardest on
# exactly the players whose attacking return drives the projection.
#
# The conditional MEAN is the correct substitution rather than a sampled distribution:
# exposure enters linearly (E[Poisson(lam*m90)] = lam*E[m90]) and the 60-minute threshold
# has already been passed by construction on this branch.
MINUTES_IF_START = {"GK": 89.9, "DEF": 87.5, "MID": 83.1, "FWD": 81.5}
MINUTES_IF_SUB = 22.0          # measured; the previous constant was 20.0


def _minutes_if_start(pos, player=None):
    """Expected minutes given the player started.

    Prefers the player's own shrunk history (`exp_minutes`, built in
    multiseason_priors.to_priors) over the positional constant. That refinement beats the
    constant on out-of-sample MAE 1.968 vs 2.649 and removes its +1.2 minute bias — see
    studies/minutes_persistence.py. The positional constant remains the fallback for
    players with no 60+ appearances, which is the right default and matches the previous
    behaviour exactly.

    FPL_MINUTES_MODEL: `flat` restores the original 90/20; `positional` pins to the
    constant and ignores player history, which is the A/B baseline for the refinement.
    """
    mode = _os.environ.get("FPL_MINUTES_MODEL", "").lower()
    if mode in ("flat", "90", "off"):
        return 90.0
    base = MINUTES_IF_START.get(pos, 85.3)
    if mode == "positional" or player is None:
        return base
    v = getattr(player, "exp_minutes", None)
    if v is None or not np.isfinite(v) or v <= 0:
        return base
    return float(v)


def _minutes_if_sub():
    if _os.environ.get("FPL_MINUTES_MODEL", "").lower() in ("flat", "90", "off"):
        return 20.0
    return MINUTES_IF_SUB


def _home_effect(home, gameweek, is_home):
    """Home advantage for a given gameweek and side, in log space.

    SPLIT SYMMETRICALLY, which matters. The measured early-season pattern is that home
    goals FALL (-0.064 in logs over md1-3) and away goals RISE (+0.076), with total match
    goals unchanged — that null is itself measured, and firmly (+0.011 to +0.020 across
    the first 3/6/12, all CIs spanning zero). This model carries home advantage
    asymmetrically (h added to the home side only), so shaving h alone would lower home
    lambda while leaving away lambda untouched, quietly dropping total goals in GW1-3 and
    contradicting the total-goals null. Applying half the discount to the home side and
    half as a bonus to the away side reproduces both halves of the observed pattern and
    leaves the match total where the data says it should be.

    Home advantage is NOT constant across a season. [VERIFIED 2026-08-11,
    studies/early_season_goals.py, 12 Understat seasons] within-season log home
    advantage over matchdays 1-3 runs ~0.15 below the rest of the season — 10 of 12
    seasons down, and robust to which baseline is used (-0.152 vs md4+, -0.151 vs md7+,
    -0.163 vs md20+). Home goals fall AND away goals rise, so the two halves agree.

    Deliberately a SINGLE step rather than a schedule. Matchdays 4-6 show no significant
    discount (-0.071, CI -0.155..+0.023), and there is no monotone trend across the
    season (per-season slope CI spans zero) — the raw segment profile is non-monotone
    (peaking md7-12, dipping md13-19) and a multi-step schedule fitted to it would
    encode noise. One step is what the data supports.

    Applies to lam_for and lam_against alike, so it reaches both attacking returns and
    clean sheets. Set FPL_TEAM_HYPER=guess to disable (discount 0).
    """
    disc = HYPER.get("home_early_discount", 0.0)
    early = False
    if disc and gameweek is not None:
        try:
            early = int(gameweek) <= int(HYPER.get("home_early_last_gw", 3))
        except (TypeError, ValueError):
            early = False
    # always return something shaped like `home` (an array of S posterior draws), so
    # callers never have to care which branch they got
    if not early:
        return home if is_home else np.zeros_like(home)
    half = disc / 2.0
    # a discount larger than the fitted advantage would flip home into a disadvantage,
    # which nothing in the data supports; floor at zero
    return np.maximum(home - half, 0.0) if is_home else np.zeros_like(home) + half


def project(players, tm, tsamp, gw_lo, gw_hi, S=1500, seed=None, return_draws=False):
    """Posterior-predictive points per player over [gw_lo, gw_hi].

    `return_draws=True` additionally returns the raw (n_players, S) draw matrix in the
    row order of the returned frame. Summaries cannot substitute for it whenever the
    question is about a SUM of players — a squad total, a bench boost — because
    percentiles are not additive: the 95th percentile of a sum is not the sum of the
    95th percentiles, and assuming otherwise overstates the upper tail badly. It is off
    by default because the matrix is large and nothing in the normal board path needs it.
    """
    sched, long = schedule()
    win = long[(long.gameweek >= gw_lo) & (long.gameweek <= gw_hi)]
    # per-team, per-fixture expected goals for/against for all S draws
    idx2 = tsamp["idx"]; mu = tsamp["mu"]; home = tsamp["home"]
    A = tsamp["att"]; D = tsamp["dfn"]
    # build, per team, arrays of (n_fix, S) lambda_for and lambda_against
    fix_by_team = {}
    for team, g in win.groupby("team"):
        if team not in idx2:
            continue
        lf, la = [], []
        for _, r in g.iterrows():
            if r.opp not in idx2:
                continue
            ti, oi = idx2[team], idx2[r.opp]
            gw = getattr(r, "gameweek", None)
            h = _home_effect(home, gw, bool(r.is_home))          # this team's side
            hopp = _home_effect(home, gw, not bool(r.is_home))   # the opponent's
            lam_for = np.exp(mu + h + A[:, ti] - D[:, oi])
            lam_against = np.exp(mu + hopp + A[:, oi] - D[:, ti])
            lf.append(lam_for); la.append(lam_against)
        fix_by_team[team] = (np.array(lf), np.array(la))   # (nfix, S)

    out = []
    draw_rows = []
    for _, p in players.iterrows():
        if p.team not in fix_by_team:
            continue
        lam_for, lam_against = fix_by_team[p.team]         # (nfix, S)
        nfix = lam_for.shape[0]
        if nfix == 0:
            continue
        pos = p.pos
        # this player's own stream — see _player_rng
        prng = _player_rng(p, seed, gw_lo)
        # availability draws (shared across the window -> nailed/rotation risk)
        p_start = prng.beta(p.start_a, p.start_b, S)         # (S,)
        # attacking involvement rate (per 90) and defcon rate draws
        # rate draws, sanitised: a NaN/inf prior would crash the Poisson and a
        # runaway rate is not physically meaningful (no player exceeds these).
        def _safe(draw, hi):
            d = np.asarray(draw, float)
            return np.clip(np.nan_to_num(d, nan=0.0, posinf=hi, neginf=0.0), 0.0, hi)
        inv_rate = _safe(prng.gamma(max(float(np.nan_to_num(p.npxgi_alpha, nan=0.05)), 1e-6),
                                   1 / max(float(np.nan_to_num(p.npxgi_beta, nan=1.5)), 1e-6), S), 2.0)
        xa_rate  = _safe(prng.gamma(max(float(np.nan_to_num(p.xa_alpha, nan=0.02)), 1e-6),
                                   1 / max(float(np.nan_to_num(p.xa_beta, nan=1.5)), 1e-6), S), 1.5)
        dc_rate  = _safe(prng.gamma(max(float(np.nan_to_num(p.defcon_alpha, nan=5.0)), 1e-6),
                                   1 / max(float(np.nan_to_num(p.defcon_beta, nan=1.5)), 1e-6), S), 40.0)
        pts = np.zeros(S)
        dc_pts = np.zeros(S); cs_pts = np.zeros(S)
        app_pts = np.zeros(S); att_pts = np.zeros(S); conc_pts = np.zeros(S)
        for f in range(nfix):
            start = prng.random(S) < p_start
            sub   = (~start) & (prng.random(S) < p.sub_app_rate)
            mins  = np.where(start, _minutes_if_start(pos, p),
                             np.where(sub, _minutes_if_sub(), 0.0))
            played = mins > 0; played60 = mins >= 60
            m90 = mins / 90.0
            # attacking: split xGI into goals vs assists via xa share
            share_a = np.clip(xa_rate / np.maximum(inv_rate, 1e-6), 0, 1)
            exp_involve = inv_rate * m90 * (lam_for[f] / LEAGUE_MU)
            open_goals = prng.poisson(np.maximum(exp_involve * (1 - share_a), 0))
            # penalties are separate from non-penalty xGI: add for the designated
            # taker only (pen_xg90 set by the set-piece layer, else 0)
            pen_xg90 = float(getattr(p, "pen_xg90", 0.0) or 0.0)
            pen_goals = prng.poisson(np.maximum(pen_xg90 * m90, 0)) if pen_xg90 > 0 else 0
            goals   = open_goals + pen_goals
            assists = prng.poisson(np.maximum(exp_involve * share_a, 0))
            gp = goals * GOAL_POINTS[pos] + goals * BONUS_PER_GOAL[pos]
            ap = assists * (ASSIST_POINTS + BONUS_PER_ASSIST)
            # clean sheet / concession (need 60 mins), team-level.
            # ONE realisation of goals conceded drives both, because they are the same
            # event. The previous version drew lam_against TWICE and derived the clean
            # sheet from one draw and the concession penalty from the other, so a
            # simulation path could hand a defender clean-sheet points AND a two-goal
            # concession penalty in the same match — a state that cannot occur.
            # E[csp] and E[concp] were each individually right, so `mean` was unbiased
            # and this was invisible there; what was wrong was their JOINT distribution,
            # i.e. sd, p5/p95 and the captaincy tail, which is precisely what the
            # posterior draws are retained for.
            conc = prng.poisson(lam_against[f])
            cs = (conc == 0) & played60
            csp = cs * (CLEAN_SHEET_PTS[pos] + BONUS_PER_CS[pos])
            concp = np.where(np.isin(pos, ["GK", "DEF"]) & played60,
                             -np.floor(conc / 2), 0.0)
            # defensive contribution threshold (per match)
            thr = DEFCON_THRESHOLD.get(pos, 999)
            dc_cnt = prng.poisson(np.maximum(dc_rate * m90, 0))
            dcp = np.where(dc_cnt >= thr, DEFCON_PTS, 0.0)
            appp = np.where(played60, 2.0, np.where(played, 1.0, 0.0))
            pts += appp + gp + ap + csp + concp + dcp
            dc_pts += dcp; cs_pts += csp
            app_pts += appp; att_pts += gp + ap; conc_pts += concp
        q = np.percentile(pts, [5, 25, 50, 75, 95])
        if return_draws:
            draw_rows.append(pts.copy())
        out.append({"id": p.id, "player": p.web_name, "pos": pos, "team": p.team,
                    "own": p.own, "cost": p.cost, "nfix": nfix,
                    "mean": pts.mean(), "sd": pts.std(),
                    "defcon_ev": dc_pts.mean(), "cs_ev": cs_pts.mean(),
                    # the remaining components, so the five sum back to `mean` exactly
                    # and a decomposition needs no residual bucket
                    "app_ev": app_pts.mean(), "att_ev": att_pts.mean(),
                    "conc_ev": conc_pts.mean(),
                    "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4]})
    res = pd.DataFrame(out).sort_values("mean", ascending=False)
    if return_draws:
        # `res` is sorted by mean, `draw_rows` is in append order. Reindex the matrix by
        # the same permutation or every draw row is attributed to the wrong player —
        # silently, and only detectably in the tails.
        D = np.array(draw_rows)[res.index.to_numpy()] if len(draw_rows) else np.empty((0, S))
        return res.reset_index(drop=True), D
    return res.reset_index(drop=True)


if __name__ == "__main__":
    print("Fitting hierarchical Bayesian team model (25/26 results + betting priors)...")
    tm = TeamModel().fit()
    tsamp = tm.sample_2627(S=1500)
    # team strength posterior summary (attack & defence, points-scale)
    A, D = tsamp["att"], tsamp["dfn"]
    tstab = pd.DataFrame({
        "team": tsamp["teams"],
        "attack": A.mean(0), "attack_sd": A.std(0),
        "defence": D.mean(0), "defence_sd": D.std(0),
    })
    tstab["net"] = tstab.attack + tstab.defence
    tstab["promoted"] = tstab.team.isin(PROMOTED)
    tstab = tstab.sort_values("net", ascending=False)
    tstab.round(3).to_csv(_os.path.join(config.OUTPUTS, "team_strength_posteriors_2627.csv"), index=False)
    print("\nTeam strength posteriors (net = attack+defence):")
    print(tstab.round(3).to_string(index=False))

    print("\nComputing player posteriors and projecting 2026/27 ...")
    players = player_posteriors()
    for lo, hi, tag in [(1, 6, "GW1-6"), (1, 38, "full-season")]:
        res = project(players, tm, tsamp, lo, hi, S=1500)
        res.round(2).to_csv(_os.path.join(config.OUTPUTS, f"projection_2627_{tag}.csv"), index=False)
        print(f"\n=== 2026/27 {tag}: top 15 by posterior-mean points (90% CI) ===")
        show = res.head(15)[["player","pos","team","nfix","mean","p5","p95","own"]]
        print(show.round(1).to_string(index=False))
