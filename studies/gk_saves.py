from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
gk_saves.py — the form of the goalkeeper save term (PROJECT_KNOWLEDGE §6.11)
============================================================================
`bayes_model.project()` credits a keeper appearance + clean sheet − floor(conceded/2) and
NOTHING for saves, although FPL pays 1 pt per 3. That the term belongs in the model is not a
hypothesis — it is a scoring rule the composition omits. What this study decides is its FORM
and its SPREAD, and it is the gate the term passes before `FPL_GK_SAVES` defaults on.

MECHANISM [DERIVED]. If shots on target faced ~ Poisson(σ) and each is independently a goal
(prob c) or a save, then by Poisson thinning goals ~ Poisson(σc) and saves ~ Poisson(σ(1−c))
are INDEPENDENT GIVEN σ, and E[saves] = λ(1−c)/c ∝ λ = E[goals against]. So in `project()`
the coupling of saves to concessions runs through the shared λ_against draw, not through the
goals realisation. §6.11's "drawn from the same shot realisation", read literally, would
over-couple them. A residual match-level shock (shot volume, game state) can still correlate
the two given λ; that is measured below, not assumed.

PRE-REGISTERED — WRITTEN BEFORE THE RESULTS WERE SEEN
------------------------------------------------------
Written by the main session on 2026-09-28. `research-preregistrar` is the assigned role; it
could not be launched (the harness's action classifier returned no verdict on every launch
attempt), so the rules were fixed here, in the file, before the first run — the same
recorded substitution `start_forgetting.py` makes. Before writing, only file HEADERS were
read: no saves value, no fit.

UNIT      team-match: one club's keeper saves in one Premier League match (`home_keeper_saves`
          / `away_keeper_saves` in FPL-Core-Insights `matches.csv`; `match_id` contains
          `-prem-`; finished only; de-duplicated on match_id). Team level because in the
          board a save count is a property of the MATCH, shared by whichever keeper plays,
          exactly as goals conceded are (`_team_rng`). The board thins it by the keeper's
          own minutes.

REGRESSOR λ̂  ex-ante expected goals against. The board's λ_against does not exist for past
          matches, so: a Poisson GLM  ga ~ 1 + (opp_elo − team_elo)/100 + is_home, fitted on
          the TRAINING season only and applied to the test season. λ̂ is a calibrated
          predictor (E[ga | λ̂] ≈ λ̂), so under thinning E[saves | λ̂] = r·E[λ | λ̂] = r·λ̂:
          the proportional arm is identified through a calibrated proxy without
          errors-in-variables attenuation (Berkson, not classical, error). The free-elasticity
          arm is the test of that structure, not the default.

ARMS      each with its own negative-binomial (NB2) dispersion α, fitted by ML on train:
            A0 CONST   E[saves] = m                      (unconditioned)
            A1 PROP    E[saves] = r · λ̂                  (thinning)
            A2 ELAST   E[saves] = exp(a + b · log λ̂)     (free elasticity)

FOLDS     train 24/25 → test 25/26, and train 25/26 → test 24/25.
          26/27 GW1-5 is a DECLARED SECONDARY replication (train on 24/25+25/26); it gates
          nothing — ~100 team-matches.

ENDPOINT  PRIMARY: mean held-out log score of SAVE POINTS per team-match, i.e. of
          K = floor(saves/3) under each arm's NB predictive,
          P(K=k) = F(3k+2) − F(3k−1). Points are what the board consumes, and the floor makes
          the mean and the tail matter jointly.

DECISION RULE, fixed before any number was read
  R1  Conditioning: the arm shipped is A1 or A2 only if it beats A0 on the primary endpoint in
      BOTH folds; otherwise A0 ships (a constant rate is still better than the absent term —
      the term ships ON either way, the rule governs its form).
  R2  Elasticity: A2 replaces A1 only if A2 − A1 > 0.005 nats per team-match in BOTH folds.
      Otherwise A1, on parsimony and the thinning structure. The shipped parameters are
      refitted on 24/25 + 25/26 pooled.
  R3  Dispersion: randomised-PIT coverage of the central 80% interval for SAVES under the
      shipped arm, pooled over both test folds, must lie in [0.75, 0.85]. If it does not,
      the term still ships (a mean correction of ~40% of a keeper's projection outranks a
      spread defect), but the result is recorded as "right mean, wrong spread" and the
      miss is carried to PROJECT_KNOWLEDGE beside §6.12.
  R4  Coupling: Pearson correlation between the save residual (y − μ)/sd under the shipped
      arm and the concession residual (ga − λ̂)/√λ̂, pooled over test folds. If |ρ| ≤ 0.10
      (≈3 SE at n≈1,500) saves are drawn independently given λ_against. If |ρ| > 0.10 the
      independent draw is NOT shipped as final: a shared shock is required and becomes its
      own item, and the term ships flagged as coupling-incomplete.
  R5  Definition: FotMob `keeper_saves` must be the quantity FPL scores. Declared check on
      26/27 GW1-5: Σ FPL `saves` (player_gameweek_stats, GK) against Σ keeper_saves per
      (club, gw) for single-match club-weeks. If the pooled ratio FPL/FotMob lies outside
      [0.95, 1.05], the shipped rate is multiplied by that ratio.

SCOPE     IN: 1 pt per 3 saves. OUT, recorded as open: penalty saves (+5; ~0.1 pens faced
          per team-match × ~0.2 saved ≈ 0.1 pt/gw — real, and its own small item) and the
          save-driven BPS/bonus (the BONUS_PER_* table carries no save term; a bonus change
          belongs to the §6.10 BPS work).

BOARD GATE, after the form is fixed (scripts/ab_gk_saves.py, same seed, flag off vs on,
          compared on `mean`, never `blended`):
    G1  every non-GK row is bit-identical, and every GK row's non-save components are
        bit-identical (the save draw has its own streams);
    G2  across clubs, starting-keeper save_ev per match is Spearman ≥ 0.5 with the club's
        mean λ_against — weaker defences earn more. A NEGATIVE ordering is a failure;
    G3  no goalkeeper row has a negative `mean` (tighter than the harness invariant,
        which since 2026-09-08 tolerates down to −0.05 and so may already be green);
    G4  league-mean save pts per starting-keeper match lies within ±20% of the pooled
        historical rate from this study.
  Any G failure: the flag does not default on and the failure is recorded.

GUARDS    Not a team-level signal entering TeamModel — it consumes λ_against, adds nothing to
          it, so `beats_the_market` does not apply. Revives nothing in the null register
          (rotation, mean reversion, congestion, explosiveness, style): no covariate is added
          to any existing channel.

POWER     ~760 team-matches per season. The between-club range in saves per match is
          expected to be ~2x (strong vs weak defences); under A0 vs A1 that is a large
          log-score effect at this n. R2's 0.005-nat margin is ~1% of the expected
          per-match log score and guards against adopting A2 on a small in-sample slope.

RESULT
------
(not yet run)
"""
import glob
import numpy as np, pandas as pd
from scipy import optimize, stats

SEASONS = {"2024-2025": "matches/GW*/matches.csv",
           "2025-2026": "By Gameweek/GW*/matches.csv",
           "2026-2027": "By Gameweek/GW*/matches.csv"}
OUT_CSV = _os.path.join(config.STUDIES, "gk_saves.csv")
MARGIN_A2 = 0.005
RHO_MAX = 0.10


# ----------------------------------------------------------------------------- data
def _team_rows(m: pd.DataFrame, season: str) -> pd.DataFrame:
    m = m[m["match_id"].astype(str).str.contains("-prem-", na=False)]
    m = m[m["finished"].astype(str).str.upper() == "TRUE"]
    m = m.drop_duplicates("match_id").sort_values("kickoff_time")
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({
            "season": season, "gameweek": m["gameweek"].to_numpy(),
            "match_id": m["match_id"].to_numpy(),
            "team": m[f"{side}_team"].to_numpy(), "opp": m[f"{other}_team"].to_numpy(),
            "is_home": 1 if side == "home" else 0,
            "team_elo": pd.to_numeric(m[f"{side}_team_elo"], errors="coerce").to_numpy(),
            "opp_elo": pd.to_numeric(m[f"{other}_team_elo"], errors="coerce").to_numpy(),
            # goals against = the OTHER side's score
            "ga": pd.to_numeric(m[f"{other}_score"], errors="coerce").to_numpy(),
            "saves": pd.to_numeric(m[f"{side}_keeper_saves"], errors="coerce").to_numpy()}))
    d = pd.concat(rows, ignore_index=True)
    return d.dropna(subset=["team_elo", "opp_elo", "ga", "saves"]).reset_index(drop=True)


def load(season: str) -> pd.DataFrame:
    files = sorted(glob.glob(_os.path.join(config.repo(season), SEASONS[season])))
    if not files:
        raise FileNotFoundError(f"no matches.csv under {config.repo(season)}")
    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return _team_rows(m, season)


# ----------------------------------------------------------------------------- models
def _x(d):
    return np.column_stack([np.ones(len(d)), (d["opp_elo"] - d["team_elo"]) / 100.0,
                            d["is_home"].to_numpy(float)])


def fit_lambda(train: pd.DataFrame) -> np.ndarray:
    """Poisson GLM coefficients for ga on Elo difference and venue."""
    X, y = _x(train), train["ga"].to_numpy(float)
    nll = lambda b: -(y * (X @ b) - np.exp(X @ b)).sum()
    grad = lambda b: -(X.T @ (y - np.exp(X @ b)))
    return optimize.minimize(nll, np.array([np.log(max(y.mean(), 1e-3)), 0, 0]),
                             jac=grad, method="BFGS").x


def lam_hat(beta, d):
    return np.exp(_x(d) @ beta)


def _nb(mu, alpha):
    n = 1.0 / alpha
    return n, n / (n + mu)


def _mean(arm, theta, lam):
    if arm == "A0":
        return np.full_like(lam, np.exp(theta[0]))
    if arm == "A1":
        return np.exp(theta[0]) * lam
    return np.exp(theta[0] + theta[1] * np.log(lam))


N_MEAN = {"A0": 1, "A1": 1, "A2": 2}


def fit_arm(arm, y, lam):
    k = N_MEAN[arm]

    def nll(th):
        mu = _mean(arm, th[:k], lam)
        alpha = np.exp(np.clip(th[k], -12, 3))
        n, p = _nb(mu, alpha)
        return -stats.nbinom.logpmf(y, n, p).sum()
    th0 = np.r_[np.log(y.mean()), np.zeros(k - 1), np.log(0.05)]
    if arm == "A1":
        th0[0] = np.log(y.mean() / lam.mean())
    return optimize.minimize(nll, th0, method="Nelder-Mead",
                             options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 20000}).x


def predictive(arm, th, lam):
    k = N_MEAN[arm]
    mu = _mean(arm, th[:k], lam)
    alpha = np.exp(np.clip(th[k], -12, 3))
    return _nb(mu, alpha) + (mu,)


def save_pts_logscore(y, n, p):
    """log P(floor(saves/3) = floor(y/3)) under NB(n, p)."""
    k = np.floor(y / 3.0)
    pr = stats.nbinom.cdf(3 * k + 2, n, p) - stats.nbinom.cdf(3 * k - 1, n, p)
    return np.log(np.maximum(pr, 1e-300))


def exp_save_pts(n, p, smax=60):
    s = np.arange(smax + 1)[:, None]
    return (np.floor(s / 3.0) * stats.nbinom.pmf(s, n, p)).sum(0)


def pit(y, n, p, rng):
    lo = stats.nbinom.cdf(y - 1, n, p)
    return lo + rng.random(len(y)) * stats.nbinom.pmf(y, n, p)


# ----------------------------------------------------------------------------- study
def run_fold(train, test, rng):
    beta = fit_lambda(train)
    lam_tr, lam_te = lam_hat(beta, train), lam_hat(beta, test)
    y_tr, y_te = train["saves"].to_numpy(float), test["saves"].to_numpy(float)
    res = {"beta": beta, "arms": {}}
    for arm in ("A0", "A1", "A2"):
        th = fit_arm(arm, y_tr, lam_tr)
        n, p, mu = predictive(arm, th, lam_te)
        res["arms"][arm] = {
            "theta": th, "logscore": save_pts_logscore(y_te, n, p).mean(),
            "pit": pit(y_te, n, p, rng), "mu": mu,
            "sd": np.sqrt(mu + mu ** 2 * np.exp(np.clip(th[N_MEAN[arm]], -12, 3))),
            "esp": exp_save_pts(n, p)}
    res["lam_te"], res["y_te"], res["ga_te"] = lam_te, y_te, test["ga"].to_numpy(float)
    return res


def decide(folds):
    ls = {a: [f["arms"][a]["logscore"] for f in folds] for a in ("A0", "A1", "A2")}
    cond = [a for a in ("A1", "A2") if all(ls[a][i] > ls["A0"][i] for i in range(len(folds)))]
    if not cond:
        arm = "A0"
    elif "A2" in cond and all(ls["A2"][i] - ls["A1"][i] > MARGIN_A2 for i in range(len(folds))):
        arm = "A2"
    elif "A1" in cond:
        arm = "A1"
    else:
        arm = "A2"   # only A2 beat A0 in both folds
    return arm, ls


def fpl_vs_fotmob(season="2026-2027"):
    """R5: FPL-scored saves against FotMob keeper_saves, per (club, gw), single-match weeks."""
    base = config.repo(season)
    out = []
    for gwdir in sorted(glob.glob(_os.path.join(base, "By Gameweek", "GW*"))):
        try:
            pg = pd.read_csv(_os.path.join(gwdir, "player_gameweek_stats.csv"))
            pl = pd.read_csv(_os.path.join(gwdir, "players.csv"))
            mt = pd.read_csv(_os.path.join(gwdir, "matches.csv"))
        except FileNotFoundError:
            continue
        tr = _team_rows(mt, season)
        if tr.empty:
            continue
        gw = int(_os.path.basename(gwdir)[2:])
        tr = tr[tr["gameweek"] == gw]
        one = tr.groupby("team").filter(lambda g: len(g) == 1)
        pl = pl[pl["position"] == "Goalkeeper"][["player_id", "team_code"]]
        pg = pg[pg["gw"] == gw] if "gw" in pg.columns else pg
        g = pg.merge(pl, left_on="id", right_on="player_id", how="inner")
        fpl = g.groupby("team_code")["saves"].sum()
        one = one.assign(fpl_saves=one["team"].map(fpl)).dropna(subset=["fpl_saves"])
        out.append(one)
    if not out:
        return None
    d = pd.concat(out, ignore_index=True)
    return d, d["fpl_saves"].sum() / max(d["saves"].sum(), 1)


def main():
    rng = np.random.default_rng(20260928)
    s24, s25 = load("2024-2025"), load("2025-2026")
    print(f"team-matches: 24/25 {len(s24)}  25/26 {len(s25)}")
    folds = [run_fold(s24, s25, rng), run_fold(s25, s24, rng)]
    arm, ls = decide(folds)
    print("\nPRIMARY — held-out log score of save points per team-match (higher is better)")
    for a in ("A0", "A1", "A2"):
        print(f"  {a}: 24->25 {ls[a][0]:+.4f}   25->24 {ls[a][1]:+.4f}")
    print(f"  R1/R2 -> shipped arm: {arm}")

    # R3 dispersion
    u = np.concatenate([f["arms"][arm]["pit"] for f in folds])
    cov = float(((u > 0.1) & (u < 0.9)).mean())
    print(f"\nR3 coverage of central 80% (saves, {arm}): {cov:.3f}  "
          f"-> {'PASS' if 0.75 <= cov <= 0.85 else 'FAIL: right mean, wrong spread'}")

    # R4 coupling
    rs = np.concatenate([(f["y_te"] - f["arms"][arm]["mu"]) / f["arms"][arm]["sd"] for f in folds])
    rg = np.concatenate([(f["ga_te"] - f["lam_te"]) / np.sqrt(f["lam_te"]) for f in folds])
    rho = float(np.corrcoef(rs, rg)[0, 1])
    print(f"R4 residual corr(saves, ga | lam_hat): {rho:+.3f}  "
          f"-> {'independent draw' if abs(rho) <= RHO_MAX else 'SHARED SHOCK REQUIRED'}")

    # calibration by lam_hat quintile (secondary)
    lam = np.concatenate([f["lam_te"] for f in folds])
    esp = np.concatenate([f["arms"][arm]["esp"] for f in folds])
    obs = np.floor(np.concatenate([f["y_te"] for f in folds]) / 3.0)
    q = pd.qcut(lam, 5, labels=False)
    cal = pd.DataFrame({"q": q, "lam": lam, "pred": esp, "obs": obs}).groupby("q").mean()
    print("\ncalibration of save pts by lam_hat quintile (secondary):")
    print(cal.round(3).to_string())

    # pooled refit for shipping
    pool = pd.concat([s24, s25], ignore_index=True)
    beta = fit_lambda(pool)
    th = fit_arm(arm, pool["saves"].to_numpy(float), lam_hat(beta, pool))
    alpha = float(np.exp(np.clip(th[N_MEAN[arm]], -12, 3)))
    print(f"\nshipped ({arm}, pooled 24/25+25/26): mean params {np.round(th[:N_MEAN[arm]], 5)}"
          f"  exp={np.round(np.exp(th[:N_MEAN[arm]]), 5)}  alpha={alpha:.5f}")
    print(f"pooled mean saves/match {pool['saves'].mean():.3f}; "
          f"mean save pts/match {np.floor(pool['saves'] / 3).mean():.3f}")

    # R5 definition check
    r5 = fpl_vs_fotmob()
    ratio = None
    if r5 is not None:
        d5, ratio = r5
        print(f"\nR5 FPL/FotMob saves ratio, 26/27 single-match club-weeks (n={len(d5)}): "
              f"{ratio:.3f} -> {'no rescale' if 0.95 <= ratio <= 1.05 else 'RESCALE rate'}")

    # 26/27 declared-secondary replication
    try:
        s26 = load("2026-2027")
        f26 = run_fold(pool, s26, rng)
        print(f"\n26/27 GW1-5 replication (n={len(s26)}): " + "  ".join(
            f"{a} {f26['arms'][a]['logscore']:+.4f}" for a in ("A0", "A1", "A2")))
    except FileNotFoundError:
        pass

    rows = [{"fold": fi, "arm": a, "logscore": folds[fi]["arms"][a]["logscore"]}
            for fi in range(2) for a in ("A0", "A1", "A2")]
    rows += [{"fold": "decision", "arm": arm, "logscore": np.nan, "coverage80": cov,
              "rho": rho, "fpl_ratio": ratio, "alpha": alpha,
              **{f"theta{i}": v for i, v in enumerate(th[:N_MEAN[arm]])}}]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


def selftest():
    """Synthetic: saves generated by thinning must select A1 and recover r and alpha."""
    rng = np.random.default_rng(0)
    n = 1500

    def fake(seed):
        r = np.random.default_rng(seed)
        d = pd.DataFrame({"team_elo": r.normal(1800, 120, n), "opp_elo": r.normal(1800, 120, n),
                          "is_home": r.integers(0, 2, n)})
        lam = np.exp(np.log(1.35) + 0.25 * (d.opp_elo - d.team_elo) / 100 + -0.1 * d.is_home)
        d["ga"] = r.poisson(lam)
        mu = 2.1 * lam
        d["saves"] = r.poisson(r.gamma(1 / 0.04, 0.04 * mu))
        return d
    tr, te = fake(1), fake(2)
    folds = [run_fold(tr, te, rng), run_fold(te, tr, rng)]
    arm, ls = decide(folds)
    assert arm == "A1", (arm, ls)
    assert ls["A1"][0] > ls["A0"][0]
    th = fit_arm("A1", tr["saves"].to_numpy(float), lam_hat(fit_lambda(tr), tr))
    assert abs(np.exp(th[0]) - 2.1) < 0.15, np.exp(th[0])
    # PIT of a correctly specified model is uniform -> coverage near 0.8
    u = folds[0]["arms"]["A1"]["pit"]
    assert 0.75 < ((u > 0.1) & (u < 0.9)).mean() < 0.85
    # the save-points log score is a proper probability
    n_, p_ = _nb(np.array([3.0]), 0.05)
    tot = sum(np.exp(save_pts_logscore(np.array([3.0 * k]), n_, p_))[0] for k in range(40))
    assert abs(tot - 1) < 1e-6, tot
    print("gk_saves selftest OK")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        selftest()
    else:
        main()
