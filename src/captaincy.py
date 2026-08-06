"""
captaincy.py — captain picks, risk profiles, and differentials
===============================================================
Captaincy is a TAIL problem, not a mean problem. You double the score, so the
question is P(big return), and the empirical work (edge_study.py) says:

  * rolling xG+xA over ~5 gameweeks is the best single haul predictor
    (Spearman 0.114 with P(haul>=10) next GW)
  * rolling points (form) adds independently (p=0.009 in the haul LPM)
  * SINGLE-gameweek stats are mostly noise — the rolling versions beat them on
    every metric
  * "he's due / underlying numbers say he'll bounce" is a MYTH: goals-minus-xG
    has essentially zero predictive value (rho -0.018 for hauls), and is
    insignificant once you control for the underlying rate (p=0.48)
  * the realistic ceiling for haul prediction is modest — walk-forward AUC 0.568

So the model keeps the full posterior predictive rather than collapsing to a
mean, and captaincy is ranked on the tail.

Risk framework
--------------
  EV          2 x mean — what actually maximises expected score
  ceiling     p95 of the doubled score
  floor       p5  — the downside if it blanks
  P(haul)     P(>=10 raw points)
  P(blank)    P(<=2 raw points)
  regret      P(this blanks AND the template captain hauls) — the rank cost of
              going against the field
  edge        differential upside: P(haul) x (1 - effective ownership)
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
from fpl_xp_model import (GOAL_POINTS, CLEAN_SHEET_PTS, ASSIST_POINTS,
                          DEFCON_THRESHOLD, DEFCON_PTS)
import bayes_model as bm

rng = np.random.default_rng(11)


# ---------------------------------------------------------------------------
# Draw the full posterior-predictive point distribution per player
# ---------------------------------------------------------------------------
def point_draws(players, tm, tsamp, gw_lo, gw_hi, S=4000):
    """Like bayes_model.project but RETAINS the raw draws, which is what the
    tail metrics need. Returns (index, draws array of shape (n_players, S))."""
    from schedule_2627 import schedule
    _, long = schedule()
    win = long[(long.gameweek >= gw_lo) & (long.gameweek <= gw_hi)]
    idx = tsamp["idx"]; mu = tsamp["mu"]; home = tsamp["home"]
    A = tsamp["att"]; D = tsamp["dfn"]

    fix = {}
    for team, g in win.groupby("team"):
        if team not in idx: continue
        lf, la = [], []
        for _, r in g.iterrows():
            if r.opp not in idx: continue
            h = home if r.is_home else 0.0
            ho = 0.0 if r.is_home else home
            lf.append(np.exp(mu + h + A[:, idx[team]] - D[:, idx[r.opp]]))
            la.append(np.exp(mu + ho + A[:, idx[r.opp]] - D[:, idx[team]]))
        fix[team] = (np.array(lf), np.array(la))

    names, draws = [], []
    for _, p in players.iterrows():
        if p.team not in fix: continue
        lam_for, lam_against = fix[p.team]
        nfix = lam_for.shape[0]
        if nfix == 0: continue
        Sd = min(S, lam_for.shape[1])
        pos = p.pos
        p_start = rng.beta(max(p.start_a, 1e-3), max(p.start_b, 1e-3), Sd)
        safe = lambda a, b, hi, dflt: np.clip(np.nan_to_num(
            rng.gamma(max(float(np.nan_to_num(a, nan=dflt)), 1e-6),
                      1 / max(float(np.nan_to_num(b, nan=1.5)), 1e-6), Sd),
            nan=0.0, posinf=hi), 0, hi)
        inv = safe(p.npxgi_alpha, p.npxgi_beta, 2.0, 0.05)
        xa = safe(p.xa_alpha, p.xa_beta, 1.5, 0.02)
        dc = safe(p.defcon_alpha, p.defcon_beta, 40.0, 5.0)
        pen90 = float(getattr(p, "pen_xg90", 0.0) or 0.0)
        pts = np.zeros(Sd)
        for f in range(nfix):
            start = rng.random(Sd) < p_start
            sub = (~start) & (rng.random(Sd) < float(p.get("sub_app_rate", 0.3) or 0.3))
            mins = np.where(start, 90.0, np.where(sub, 20.0, 0.0))
            played, p60 = mins > 0, mins >= 60
            m90 = mins / 90.0
            share_a = np.clip(xa / np.maximum(inv, 1e-6), 0, 1)
            ei = inv * m90 * (lam_for[f][:Sd] / bm.LEAGUE_MU)
            goals = rng.poisson(np.maximum(ei * (1 - share_a), 0))
            if pen90 > 0:
                goals = goals + rng.poisson(np.maximum(pen90 * m90, 0))
            asts = rng.poisson(np.maximum(ei * share_a, 0))
            gp = goals * (GOAL_POINTS[pos] + bm.BONUS_PER_GOAL[pos])
            ap = asts * (ASSIST_POINTS + bm.BONUS_PER_ASSIST)
            conc = rng.poisson(lam_against[f][:Sd])
            cs = (conc == 0) & p60
            csp = cs * (CLEAN_SHEET_PTS[pos] + bm.BONUS_PER_CS[pos])
            concp = np.where(np.isin(pos, ["GK", "DEF"]) & p60, -np.floor(conc / 2), 0.0)
            thr = DEFCON_THRESHOLD.get(pos, 999)
            dcp = np.where(rng.poisson(np.maximum(dc * m90, 0)) >= thr, DEFCON_PTS, 0.0)
            app = np.where(p60, 2.0, np.where(played, 1.0, 0.0))
            pts += app + gp + ap + csp + concp + dcp
        names.append({"player": p.web_name, "pos": pos, "team": p.team,
                      "cost": float(p.get("cost", np.nan)),
                      "own": float(p.get("own", 0) or 0)})
        draws.append(pts)
    return pd.DataFrame(names), np.array(draws)


# ---------------------------------------------------------------------------
# The captaincy function
# ---------------------------------------------------------------------------
def captain_picks(players, tm, tsamp, gameweek, horizon=1, S=4000,
                  haul=10, blank=2, top=12, min_own_template=15.0):
    """Rank captaincy options for a gameweek on the TAIL of the posterior.

    Returns a frame with EV, ceiling, floor, P(haul), P(blank), a risk label,
    regret against the template captain, and a differential edge score.
    """
    meta, draws = point_draws(players, tm, tsamp, gameweek, gameweek + horizon - 1, S=S)
    if not len(meta):
        return pd.DataFrame()
    m = meta.copy()
    m["mean"] = draws.mean(1)
    m["ev_captain"] = 2 * m["mean"]
    m["sd"] = draws.std(1)
    m["floor"] = np.percentile(draws, 5, axis=1)
    m["median"] = np.percentile(draws, 50, axis=1)
    m["ceiling"] = np.percentile(draws, 95, axis=1)
    m["p_haul"] = (draws >= haul).mean(1)
    m["p_blank"] = (draws <= blank).mean(1)
    # coefficient of variation: risk per unit of expected return
    m["cv"] = m["sd"] / m["mean"].clip(lower=0.1)
    # floor conditional on actually appearing — a 5th percentile of 0 for everyone
    # just restates rotation risk; this is the actionable downside.
    m["floor_if_plays"] = [np.percentile(draws[i][draws[i] > 0], 10)
                           if (draws[i] > 0).sum() > 20 else 0.0 for i in range(len(m))]
    m["p_plays"] = (draws > 0).mean(1)

    # template captain = highest-owned among the strong options
    pool = m[m["mean"] >= m["mean"].quantile(0.90)]
    tmpl_i = pool["own"].idxmax() if len(pool) else m["mean"].idxmax()
    tmpl = m.loc[tmpl_i, "player"]
    tmpl_draws = draws[m.index.get_loc(tmpl_i)]
    # regret: P(mine blanks AND template hauls) — the rank cost of deviating
    m["regret"] = [((draws[i] <= blank) & (tmpl_draws >= haul)).mean() for i in range(len(m))]
    # gain: P(mine hauls AND template blanks)
    m["gain"] = [((draws[i] >= haul) & (tmpl_draws <= blank)).mean() for i in range(len(m))]
    m["edge_ratio"] = m["gain"] / m["regret"].clip(lower=1e-4)

    # effective ownership proxy: captaincy concentrates on the obvious pick
    m["eo_proxy"] = m["own"] * (1 + (m["mean"] / m["mean"].max()))
    m["differential_edge"] = m["p_haul"] * (1 - (m["eo_proxy"] / 100).clip(0, 0.95))

    # thresholds computed on the FULL candidate pool (viable captains only),
    # not the truncated top-N, or the percentiles are meaningless
    pool_v = m[m["mean"] >= m["mean"].quantile(0.80)]
    cv_lo, cv_hi = pool_v["cv"].quantile(0.35), pool_v["cv"].quantile(0.70)
    haul_hi = pool_v["p_haul"].quantile(0.70)

    def label(r):
        if r["own"] >= min_own_template and r["mean"] >= pool_v["mean"].quantile(0.5):
            return "TEMPLATE (protects rank)"
        if r["own"] < 10 and r["p_haul"] >= haul_hi:
            return "DIFFERENTIAL (rank gain)"
        if r["cv"] >= cv_hi and r["p_haul"] >= haul_hi:
            return "AGGRESSIVE (high variance)"
        if r["cv"] <= cv_lo:
            return "SAFE (low variance)"
        return "BALANCED"
    m["risk_profile"] = m.apply(label, axis=1)
    m["template_ref"] = tmpl
    return m.sort_values("ev_captain", ascending=False).head(top).reset_index(drop=True)


def best_differential(players, tm, tsamp, gameweek, horizon=1, S=4000,
                      max_own=10.0, min_mean=None, top=10):
    """Low-owned players with genuine haul probability — ranked on rank-gain
    potential rather than raw projection."""
    meta, draws = point_draws(players, tm, tsamp, gameweek, gameweek + horizon - 1, S=S)
    m = meta.copy()
    m["mean"] = draws.mean(1)
    m["ceiling"] = np.percentile(draws, 95, axis=1)
    m["p_haul"] = (draws >= 10).mean(1)
    m["p_blank"] = (draws <= 2).mean(1)
    if min_mean is None:
        min_mean = m["mean"].quantile(0.70)
    d = m[(m.own <= max_own) & (m["mean"] >= min_mean)].copy()
    d["differential_score"] = d.p_haul * (1 - d.own / 100.0) * (d["mean"] / m["mean"].max())
    return d.sort_values("differential_score", ascending=False).head(top).reset_index(drop=True)
</content>
