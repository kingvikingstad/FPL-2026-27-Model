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
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
import sys; sys.path.insert(0, "/home/claude/fpl")
import betting_features as bf
from schedule_2627 import schedule, PROMOTED
from fpl_xp_model import (GOAL_POINTS, CLEAN_SHEET_PTS, ASSIST_POINTS,
                          DEFCON_THRESHOLD, DEFCON_PTS)

rng = np.random.default_rng(7)
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
class TeamModel:
    def __init__(self, prior_sd=0.35, home_prior=(0.26, 0.08),
                 promoted_att=(-0.20, 0.30), promoted_def=(-0.22, 0.30),
                 promoted_per_club=None):
        # promoted_* default to educated guesses; history.calibrate_all()
        # estimates them from 30 seasons of results -> pass them in.
        self.prior_sd = prior_sd; self.home_prior = home_prior
        self.promoted_att = promoted_att; self.promoted_def = promoted_def
        # optional {club: centred_log_strength} from Elo -> club-specific priors
        self.promoted_per_club = promoted_per_club or {}

    def fit(self, e0_path="/mnt/user-data/uploads/E0.csv", clubelo=None, clubelo_weight=0.35):
        bet = bf.build(e0_path)
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

    def sample_2627(self, S=1500, revert=0.85, season_sd=0.15):
        """Draw S joint samples of (mu, home, att, def) for the 20 teams of
        2026/27. Returning teams: Laplace posterior, reverted to mean and with
        added between-season variance. Promoted teams: promoted prior."""
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
def player_posteriors(csv="/mnt/user-data/uploads/fpl-data-stats.csv",
                      revert=0.70, min_minutes=450, agg=None):
    """Gamma-Poisson rate posteriors + Beta availability posterior per player.

    `agg` — optional pre-built season-total frame (see
    `roster.season_aggregate_from_bootstrap`). Supplying it lets the whole prior
    layer run off the live FPL bootstrap payload, with no dependency on the
    legacy per-gameweek `fpl-data-stats.csv` panel.
    """
    if agg is None:
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
    else:
        agg = agg.copy()
        agg["team"] = agg.team.replace(BET_NAME)
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
def project(players, tm, tsamp, gw_lo, gw_hi, S=1500, market=None,
            market_weight=0.5, return_fixtures=False):
    """Monte-Carlo posterior predictive over gameweeks [gw_lo, gw_hi].

    Calling this with gw_lo == gw_hi yields a TRUE single-gameweek posterior
    (the window filter below is what makes that work), not a share of a
    multi-week aggregate.

    `market` — optional {(gameweek, team): (lambda_for, lambda_against)} from
    `market_odds.fixture_odds_to_market_map`. Where a fixture is priced, the
    model's own expected goals are blended with the market's in LOG space
    (geometric blend, which keeps the result positive and is the natural scale
    for a Poisson rate). `market_weight` is the weight on the market: 0 ignores
    it entirely, 1 defers to it completely. The blend is applied to the whole
    posterior draw vector, so model uncertainty is preserved rather than being
    collapsed to a point estimate by the market.
    """
    sched, long = schedule()
    win = long[(long.gameweek >= gw_lo) & (long.gameweek <= gw_hi)]
    # per-team, per-fixture expected goals for/against for all S draws
    idx2 = tsamp["idx"]; mu = tsamp["mu"]; home = tsamp["home"]
    A = tsamp["att"]; D = tsamp["dfn"]
    market = market or {}
    w = float(np.clip(market_weight, 0.0, 1.0))
    n_priced = 0
    # build, per team, arrays of (n_fix, S) lambda_for and lambda_against
    fix_by_team = {}
    fixture_rows = []
    for team, g in win.groupby("team"):
        if team not in idx2:
            continue
        lf, la = [], []
        for _, r in g.iterrows():
            if r.opp not in idx2:
                continue
            ti, oi = idx2[team], idx2[r.opp]
            h = home if r.is_home else 0.0
            lam_for = np.exp(mu + h + A[:, ti] - D[:, oi])
            hopp = 0.0 if r.is_home else home
            lam_against = np.exp(mu + hopp + A[:, oi] - D[:, ti])
            mk = market.get((int(r.gameweek), team))
            if mk is not None and w > 0:
                m_for, m_against = mk
                if np.isfinite(m_for) and m_for > 0:
                    lam_for = np.exp((1 - w) * np.log(lam_for) + w * np.log(m_for))
                if np.isfinite(m_against) and m_against > 0:
                    lam_against = np.exp((1 - w) * np.log(lam_against) + w * np.log(m_against))
                n_priced += 1
            lf.append(lam_for); la.append(lam_against)
            if return_fixtures:
                fixture_rows.append({
                    "gameweek": int(r.gameweek), "team": team, "opp": r.opp,
                    "is_home": int(r.is_home), "xg_for": float(lam_for.mean()),
                    "xg_against": float(lam_against.mean()),
                    "cs_prob": float(np.exp(-lam_against).mean()),
                    "market_priced": int(mk is not None)})
        fix_by_team[team] = (np.array(lf), np.array(la))   # (nfix, S)
    project.last_market_fixtures = n_priced

    out = []
    for _, p in players.iterrows():
        if p.team not in fix_by_team:
            continue
        lam_for, lam_against = fix_by_team[p.team]         # (nfix, S)
        nfix = lam_for.shape[0]
        if nfix == 0:
            continue
        pos = p.pos
        # availability draws (shared across the window -> nailed/rotation risk)
        p_start = rng.beta(p.start_a, p.start_b, S)         # (S,)
        # attacking involvement rate (per 90) and defcon rate draws
        # rate draws, sanitised: a NaN/inf prior would crash the Poisson and a
        # runaway rate is not physically meaningful (no player exceeds these).
        def _safe(draw, hi):
            d = np.asarray(draw, float)
            return np.clip(np.nan_to_num(d, nan=0.0, posinf=hi, neginf=0.0), 0.0, hi)
        inv_rate = _safe(rng.gamma(max(float(np.nan_to_num(p.npxgi_alpha, nan=0.05)), 1e-6),
                                   1 / max(float(np.nan_to_num(p.npxgi_beta, nan=1.5)), 1e-6), S), 2.0)
        xa_rate  = _safe(rng.gamma(max(float(np.nan_to_num(p.xa_alpha, nan=0.02)), 1e-6),
                                   1 / max(float(np.nan_to_num(p.xa_beta, nan=1.5)), 1e-6), S), 1.5)
        dc_rate  = _safe(rng.gamma(max(float(np.nan_to_num(p.defcon_alpha, nan=5.0)), 1e-6),
                                   1 / max(float(np.nan_to_num(p.defcon_beta, nan=1.5)), 1e-6), S), 40.0)
        pts = np.zeros(S)
        for f in range(nfix):
            start = rng.random(S) < p_start
            sub   = (~start) & (rng.random(S) < p.sub_app_rate)
            mins  = np.where(start, 90.0, np.where(sub, 20.0, 0.0))
            played = mins > 0; played60 = mins >= 60
            m90 = mins / 90.0
            # attacking: split xGI into goals vs assists via xa share
            share_a = np.clip(xa_rate / np.maximum(inv_rate, 1e-6), 0, 1)
            exp_involve = inv_rate * m90 * (lam_for[f] / LEAGUE_MU)
            open_goals = rng.poisson(np.maximum(exp_involve * (1 - share_a), 0))
            # penalties are separate from non-penalty xGI: add for the designated
            # taker only (pen_xg90 set by the set-piece layer, else 0)
            pen_xg90 = float(getattr(p, "pen_xg90", 0.0) or 0.0)
            pen_goals = rng.poisson(np.maximum(pen_xg90 * m90, 0)) if pen_xg90 > 0 else 0
            goals   = open_goals + pen_goals
            assists = rng.poisson(np.maximum(exp_involve * share_a, 0))
            gp = goals * GOAL_POINTS[pos] + goals * BONUS_PER_GOAL[pos]
            ap = assists * (ASSIST_POINTS + BONUS_PER_ASSIST)
            # clean sheet / concession (need 60 mins), team-level
            cs = (rng.poisson(lam_against[f]) == 0) & played60
            csp = cs * (CLEAN_SHEET_PTS[pos] + BONUS_PER_CS[pos])
            conc = rng.poisson(lam_against[f])
            concp = np.where(np.isin(pos, ["GK", "DEF"]) & played60,
                             -np.floor(conc / 2), 0.0)
            # defensive contribution threshold (per match)
            thr = DEFCON_THRESHOLD.get(pos, 999)
            dc_cnt = rng.poisson(np.maximum(dc_rate * m90, 0))
            dcp = np.where(dc_cnt >= thr, DEFCON_PTS, 0.0)
            appp = np.where(played60, 2.0, np.where(played, 1.0, 0.0))
            pts += appp + gp + ap + csp + concp + dcp
        q = np.percentile(pts, [5, 25, 50, 75, 95])
        out.append({"id": p.id, "player": p.web_name, "pos": pos, "team": p.team,
                    "own": p.own, "cost": p.cost, "nfix": nfix,
                    "mean": pts.mean(), "sd": pts.std(),
                    "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
                    # Decision-relevant tail summaries. Captaincy is a haul
                    # problem and bench cover is a floor problem, so surface both
                    # rather than making the user re-derive them from the CSV.
                    # Thresholds scale with the window so they stay meaningful
                    # whether this is one gameweek or twelve: "ceiling" = a 6+
                    # points-per-fixture return rate, "floor" = 2 or fewer.
                    "p_ceiling": float((pts >= 6.0 * nfix).mean()),
                    "p_floor": float((pts <= 2.0 * nfix).mean()),
                    "pts_per_fix": float(pts.mean() / nfix),
                    "exp_starts": float(np.round(p.start_a / (p.start_a + p.start_b) * nfix, 2)),
                    "cold_start": bool(getattr(p, "cold_start", False))})
    res = pd.DataFrame(out).sort_values("mean", ascending=False).reset_index(drop=True)
    if return_fixtures:
        return res, pd.DataFrame(fixture_rows)
    return res


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
    tstab.round(3).to_csv("/mnt/user-data/outputs/team_strength_posteriors_2627.csv", index=False)
    print("\nTeam strength posteriors (net = attack+defence):")
    print(tstab.round(3).to_string(index=False))

    print("\nComputing player posteriors and projecting 2026/27 ...")
    players = player_posteriors()
    for lo, hi, tag in [(1, 6, "GW1-6"), (1, 38, "full-season")]:
        res = project(players, tm, tsamp, lo, hi, S=1500)
        res.round(2).to_csv(f"/mnt/user-data/outputs/projection_2627_{tag}.csv", index=False)
        print(f"\n=== 2026/27 {tag}: top 15 by posterior-mean points (90% CI) ===")
        show = res.head(15)[["player","pos","team","nfix","mean","p5","p95","own"]]
        print(show.round(1).to_string(index=False))
