from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
defcon_threshold_calibration.py — is a defender's DefCon count over-dispersed at a known rate?
==============================================================================================
PRE-REGISTERED in docs/DEFCON_THRESHOLD_CALIBRATION_PREREG_2026-09-16.md (commit d7b1771),
before any real-data coefficient was read. Every threshold below is copied from that file;
none may be changed after this script has been run on real data.

THE QUESTION
------------
bayes_model.project() pays DefCon when Poisson(rate x m90) >= 10 for a defender. The rate is
one Gamma posterior draw per path, so across paths the count is already negative binomial,
but that spread is POSTERIOR RATE UNCERTAINTY: for a regular defender alpha ~ 184 and the
per-match var/mean is ~1.04. What the model has no term for is per-match variation at a
KNOWN rate. This estimates it as a pooled gamma-process frailty phi,

    Y | lam, G ~ Poisson(lam G),   G ~ Gamma(shape m90/phi, scale phi),   Var = lam m (1 + lam phi)

and asks whether adding it makes held-out P(>= 10) better.

WHY IT IS BUILT THIS WAY
------------------------
* phi is estimated WITHIN player (Pearson moment about the player's own rate), so the
  player's level is conditioned out and banding artefacts (regression to the mean) cannot
  enter it. The quintile table in DEFCON_SOURCE_CORRECTION §6 could not separate the two.
* Its null is SIMULATED: true Poisson at the fitted per-player rates on the identical rows
  and exposures, pushed through the identical estimator (the team_explosiveness pattern).
* Held-out arms form a 2x2 (plug-in vs Gamma-integrated rate, x Poisson vs frailty) plus the
  as-deployed pair, so posterior-uncertainty (Jensen) and frailty are not credited to each other.
* SEs are the larger of club-match and player clustering, inflated by a simulated factor c:
  in the pre-registration synthetic runs club-match clustering alone understated the
  replicate SD by 15-27%.
* Brier is NOT powered to show improvement for r > ~10, so it gates as non-inferiority.
* style_matchup.beats_the_market is NOT the gate: it is a score test on a team-level mean,
  and this is a player-level second-moment claim.

GUARDS: DefCon through defcon_series.fpl_defcon (via defcon_team.build_panel); null counts are
DROPPED, never filled; exposure is measured minutes only; 25/26 only (fpl_defcon refuses
24/25); keyed on player_code; halves split on the explicit gameweek column.

The synthetic selftest (pre-registration §2, checks 1-7) runs FIRST on every invocation and
the real-data analysis refuses to run if it fails.

Run:  python studies/defcon_threshold_calibration.py              (selftest, then study)
      python studies/defcon_threshold_calibration.py --selftest   (selftest only)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import special, stats, optimize

OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "defcon_threshold_calibration.csv")

THR = 10                    # DEF threshold (CBIT)
K0, REVERT = 3.0, 0.70      # multiseason_priors.to_priors defaults, as deployed
POS_MIN_DEF, MIN_K = 87.5, 5.7   # multiseason_priors POS_MINUTES['DEF'], MIN_K
TRAIN_GW, TEST_GW = (1, 19), (20, 38)

# ---- decision-rule constants, pre-registration §3 -------------------------------------
M_MIN = 0.005               # materiality: mean |P_D2 - P_D1|
NI_MARGIN = 0.001           # non-inferiority margin on Brier (D2 - D1)
Z_VALID = 1.645             # ELPD(E2-E1) z for VALIDATED
Z_SHAPE = -1.0              # ELPD z at or below this -> frailty shape wrong -> NULL
G0_BAND = (0.90, 1.10)      # G0 CI entirely outside -> open a separate k0 item
N_NULL = 1000               # simulated true-Poisson replicates for the phi band
K_QUAD = 1600               # quantile-midpoint nodes over G (400 missed 2e-5 on P)


# =========================================================================================
# 1. Estimators
# =========================================================================================
def phi_moment(y, e, pid):
    """Pooled within-player Pearson moment estimator of the frailty.

    `e` is the per-row exposure-scaled offset BEFORE the player's own rate (m90, or m90 x an
    opponent factor). Player rate lam_i = sum y / sum e. Returns phi-hat; players with fewer
    than two rows or a zero total carry no within-player information and are skipped.
    """
    d = pd.DataFrame({"y": np.asarray(y, float), "e": np.asarray(e, float), "p": np.asarray(pid)})
    g = d.groupby("p")
    lam = g["y"].transform("sum") / g["e"].transform("sum")
    mu = lam * d["e"]
    ok = mu > 0
    x2 = ((d["y"] - mu) ** 2 / mu.where(ok)).groupby(d["p"]).sum()
    n = g.size()
    li = (g["y"].sum() / g["e"].sum())
    keep = (n >= 2) & (li > 0)
    den = float(((n - 1) * li)[keep].sum())
    return float((x2 - (n - 1))[keep].sum() / den) if den > 0 else np.nan


def phi_moment_matrix(Y, e, pid_codes, n_players):
    """phi_moment for many simulated outcome vectors at once. Y is (R, n)."""
    E = np.bincount(pid_codes, weights=e, minlength=n_players)
    n = np.bincount(pid_codes, minlength=n_players)
    out = np.empty(Y.shape[0])
    for r in range(Y.shape[0]):
        T = np.bincount(pid_codes, weights=Y[r], minlength=n_players)
        lam = np.divide(T, E, out=np.zeros_like(T), where=E > 0)
        mu = lam[pid_codes] * e
        x2 = np.bincount(pid_codes, weights=np.divide((Y[r] - mu) ** 2, mu,
                                                      out=np.zeros_like(mu), where=mu > 0),
                         minlength=n_players)
        keep = (n >= 2) & (lam > 0)
        den = ((n - 1) * lam)[keep].sum()
        out[r] = ((x2 - (n - 1))[keep].sum() / den) if den > 0 else np.nan
    return out


def poisson_null_band(y, e, pid, R=N_NULL, seed=0):
    """Simulated true-Poisson distribution of phi_moment on this exact design."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(pid))
    e = np.asarray(e, float); y = np.asarray(y, float)
    E = np.bincount(codes, weights=e); T = np.bincount(codes, weights=y)
    lam = np.divide(T, E, out=np.zeros_like(T), where=E > 0)
    Y = rng.poisson(np.broadcast_to(lam[codes] * e, (R, len(e))))
    sims = phi_moment_matrix(Y.astype(float), e, codes, len(uniq))
    return np.nanpercentile(sims, [2.5, 97.5]), sims


def phi_dm_ml(y, m, pid):
    """Dirichlet-multinomial conditional ML cross-check. Given a player's total, gamma-process
    frailty makes the counts DM with alpha_j = m_j / phi, free of the player's rate."""
    d = pd.DataFrame({"y": np.asarray(y, float), "m": np.asarray(m, float), "p": np.asarray(pid)})
    d = d[d.groupby("p")["y"].transform("size") >= 2]
    T = d.groupby("p")["y"].transform("sum"); M = d.groupby("p")["m"].transform("sum")
    first = ~d["p"].duplicated()

    def nll(lphi):
        phi = np.exp(lphi)
        a = d["m"] / phi; A = M / phi
        ll = (special.gammaln(d["y"] + a) - special.gammaln(a)).sum()
        ll += (special.gammaln(A[first]) - special.gammaln(T[first] + A[first])).sum()
        return -ll
    r = optimize.minimize_scalar(nll, bounds=(np.log(1e-6), np.log(2.0)), method="bounded")
    return float(np.exp(r.x))


# =========================================================================================
# 2. Predictive distributions: P(Y >= THR) and log p(y) for each arm
# =========================================================================================
def _nb_cdf_logpmf(y, size, p):
    """NB(size, p) with P(Y=k) = C(k+size-1,k) p^size (1-p)^k. Returns P(Y <= THR-1), log p(y)."""
    size = np.asarray(size, float); p = np.clip(np.asarray(p, float), 1e-300, 1.0)
    cdf = special.betainc(size, THR, p)
    lp = (special.gammaln(y + size) - special.gammaln(size) - special.gammaln(y + 1)
          + size * np.log(p) + y * np.log1p(-np.minimum(p, 1 - 1e-16)))
    return cdf, lp


def pred_poisson(y, mean):
    mean = np.maximum(np.asarray(mean, float), 1e-12)
    return 1 - stats.poisson.cdf(THR - 1, mean), stats.poisson.logpmf(y, mean)


def pred_plugin_frailty(y, lam, m, phi):
    """N0: Poisson(lam G), G ~ Gamma(m/phi, phi) integrates to NB(size m/phi, mean lam m)."""
    if not phi > 0:
        return pred_poisson(y, lam * m)
    size = m / phi
    cdf, lp = _nb_cdf_logpmf(y, size, size / (size + lam * m))
    return 1 - cdf, lp


def pred_gamma(y, alpha, beta, m):
    """E1/D1: Poisson(lam m), lam ~ Gamma(alpha, rate beta) -> NB(alpha, beta/(beta+m))."""
    cdf, lp = _nb_cdf_logpmf(y, alpha, beta / (beta + m))
    return 1 - cdf, lp


def pred_gamma_frailty(y, alpha, beta, m, phi, K=K_QUAD):
    """E2/D2: integrate NB(alpha, beta/(beta+G)) over G ~ Gamma(m/phi, phi) by quantile
    midpoints. Rows are grouped by m so the G grid is built once per exposure value."""
    y = np.asarray(y, float); alpha = np.asarray(alpha, float)
    beta = np.asarray(beta, float); m = np.asarray(m, float)
    if not phi > 0:
        return pred_gamma(y, alpha, beta, m)
    P = np.empty(len(y)); LP = np.empty(len(y))
    u = (np.arange(K) + 0.5) / K
    mk = np.round(m, 6)
    for val in np.unique(mk):
        idx = np.where(mk == val)[0]
        G = stats.gamma.ppf(u, a=val / phi, scale=phi)[None, :]
        pp = beta[idx, None] / (beta[idx, None] + G)
        cdf, lp = _nb_cdf_logpmf(y[idx, None], alpha[idx, None], pp)
        P[idx] = 1 - cdf.mean(axis=1)
        LP[idx] = special.logsumexp(lp, axis=1) - np.log(K)
    return P, LP


# =========================================================================================
# 3. Arms
# =========================================================================================
def role_means(train60, role):
    """Pooled per-90 rate by role on the TRAIN half (60+ rows) — refits RATE_CB/FB/POOLED so
    no test-half information enters the prior mean."""
    t = train60.assign(role=train60["player_code"].map(role).fillna("ALL"))
    allr = t["y"].sum() / t["m"].sum()
    out = {"ALL": float(allr)}
    for r_, g in t.groupby("role"):
        if r_ != "ALL" and g["m"].sum() > 0:
            out[r_] = float(g["y"].sum() / g["m"].sum())
    return out


def eb_k(train60, mu_of, phi):
    """Gamma prior strength k (in 90s) by method of moments with frailty-inclusive noise:
    between-player variance of raw rates about the role mean, net of the expected sampling
    variance lam(1 + lam phi)/M under the frailty process. Prior var = mu/k."""
    g = train60.groupby("player_code").agg(y=("y", "sum"), M=("m", "sum"))
    g = g[g["M"] >= 1.0]
    mu = g.index.map(mu_of).to_numpy(float)
    r = g["y"] / g["M"]
    ph = max(phi, 0.0)
    S = float(np.mean((r - mu) ** 2 - mu * (1 + mu * ph) / g["M"]))
    tau2 = max(S, 1e-6)
    return float(np.clip(np.mean(mu) / tau2, 0.5, 200.0))


def build_arms(train_all, train60, test, role, phi, k_scale=1.0):
    """Predictions for the six arms on `test` (60+ rows). Returns a frame of P_* and LP_*."""
    rm = role_means(train60, role)
    mu_of = lambda pc: rm.get(role.get(pc, "ALL"), rm["ALL"])
    ph = max(phi, 0.0) if np.isfinite(phi) else 0.0
    # --- EB integrated (E) and plug-in (P0/N0): 60+ train rows, revert = 1 -------------
    k = eb_k(train60, mu_of, ph) * k_scale
    ev = train60.groupby("player_code").agg(y=("y", "sum"), M=("m", "sum"))
    mu_t = test["player_code"].map(mu_of).to_numpy(float)
    ys = test["player_code"].map(ev["y"]).fillna(0.0).to_numpy(float)
    Ms = test["player_code"].map(ev["M"]).fillna(0.0).to_numpy(float)
    aE, bE = mu_t * k + ys, k + Ms
    # --- as deployed (D): all measured train minutes, k0 = 3, revert = 0.70 ---------------
    ea = train_all.groupby("player_code").agg(y=("y", "sum"), M=("m", "sum"))
    s60 = train60.groupby("player_code").agg(n=("m", "size"), cm=("mins", "sum"))
    yD = test["player_code"].map(ea["y"]).fillna(0.0).to_numpy(float)
    MD = test["player_code"].map(ea["M"]).fillna(0.0).to_numpy(float)
    aD, bD = mu_t * K0 + REVERT * yD, K0 + REVERT * MD
    n = test["player_code"].map(s60["n"]).fillna(0.0).to_numpy(float)
    cm = test["player_code"].map(s60["cm"]).fillna(0.0).to_numpy(float)
    own = np.divide(cm, n, out=np.full_like(cm, POS_MIN_DEF), where=n > 0)
    w = n / (n + MIN_K)
    expm = np.where(n > 0, np.clip(w * own + (1 - w) * POS_MIN_DEF, 60.0, 90.0), POS_MIN_DEF)
    mD = expm / 90.0

    y = test["y"].to_numpy(float); m = test["m"].to_numpy(float)
    lam = aE / bE
    out = pd.DataFrame(index=test.index)
    for name, (P, LP) in {
        "P0": pred_poisson(y, lam * m),
        "N0": pred_plugin_frailty(y, lam, m, ph),
        "E1": pred_gamma(y, aE, bE, m),
        "E2": pred_gamma_frailty(y, aE, bE, m, ph),
        "D1": pred_gamma(y, aD, bD, mD),
        "D2": pred_gamma_frailty(y, aD, bD, mD, ph),
    }.items():
        out[f"P_{name}"] = P; out[f"LP_{name}"] = LP
    out["lam_E"] = lam; out["lam_D"] = aD / bD; out["mD"] = mD
    out.attrs.update(k=k, role_means=rm)
    return out


# =========================================================================================
# 4. Inference: clustered SEs, G0, contrasts
# =========================================================================================
def cluster_se(d, groups):
    """SE of mean(d) clustered on `groups` (one label per row)."""
    d = np.asarray(d, float); n = len(d)
    s = pd.Series(d - d.mean()).groupby(np.asarray(groups)).sum()
    return float(np.sqrt((s ** 2).sum()) / n)


def row_se(d):
    d = np.asarray(d, float)
    return float(d.std(ddof=1) / np.sqrt(len(d)))


def se_pair(d, cm, pl):
    """(max of club-match and player clustered SE, two-way pigeonhole SE)."""
    a, b = cluster_se(d, cm), cluster_se(d, pl)
    two = np.sqrt(max(a ** 2 + b ** 2 - row_se(d) ** 2, 0.0))   # intersection = row
    return max(a, b), float(two)


def g0_slope(y, m, lam, cm, pl):
    """Slope of count per 90 on the train-half posterior-mean rate (weights m). A frailty
    leaves the mean unchanged, so a slope off 1 is shrinkage/drift, never (b)."""
    y = np.asarray(y, float); m = np.asarray(m, float); x = np.asarray(lam, float)
    X = np.column_stack([np.ones_like(x), x]); W = m
    XtWX = X.T @ (X * W[:, None]); beta = np.linalg.solve(XtWX, X.T @ (W * (y / m)))
    res = (y / m) - X @ beta
    inv = np.linalg.inv(XtWX)
    ses = []
    for grp in (cm, pl):
        sc = pd.DataFrame(X * (W * res)[:, None]).groupby(np.asarray(grp)).sum().to_numpy()
        V = inv @ (sc.T @ sc) @ inv
        ses.append(np.sqrt(V[1, 1]))
    return float(beta[1]), float(max(ses))


def contrasts(test, arms):
    """Row-level differences and their headline means."""
    H = (test["y"].to_numpy() >= THR).astype(float)
    br = {a: (arms[f"P_{a}"].to_numpy() - H) ** 2 for a in ("P0", "N0", "E1", "E2", "D1", "D2")}
    return {
        "dB_E": br["E2"] - br["E1"], "dB_D": br["D2"] - br["D1"],
        "elpd_E": arms["LP_E2"].to_numpy() - arms["LP_E1"].to_numpy(),
        "M": float(np.mean(np.abs(arms["P_D2"] - arms["P_D1"]))),
        "brier": {a: float(v.mean()) for a, v in br.items()}, "H": H,
    }


# =========================================================================================
# 5. The pipeline on one split (used for real data AND synthetic replicates)
# =========================================================================================
def run_split(rows, role, train_mask, test_mask, phi=None, band=True, k_scale=1.0, seed=0):
    """rows: player_code, gameweek, club, match_id, opp, mins, y (measured, any minutes)."""
    r = rows.copy()
    r["m"] = r["mins"] / 90.0
    r["cm"] = r["club"].astype(str) + "|" + r["match_id"].astype(str)
    tr_all = r[train_mask]
    tr60 = tr_all[tr_all["mins"] >= 60]
    te = r[test_mask & (r["mins"] >= 60)]
    phi_tr = phi_moment(tr60["y"], tr60["m"], tr60["player_code"]) if phi is None else phi
    res = {"phi_tr": phi_tr, "n_train60": len(tr60), "n_test": len(te)}
    if band:
        res["band_tr"], _ = poisson_null_band(tr60["y"], tr60["m"], tr60["player_code"], seed=seed)
    arms = build_arms(tr_all, tr60, te, role, phi_tr, k_scale=k_scale)
    c = contrasts(te, arms)
    res.update(te=te, arms=arms, c=c, k=arms.attrs["k"], role_means=arms.attrs["role_means"])
    return res


# =========================================================================================
# 6. Synthetic design (selftest and the SE-inflation factor c)
# =========================================================================================
def synth_rows(seed, phi=0.0, shock=0.0, opp_sd=0.0, drift_sd=0.0, thin=False,
               n_clubs=20, per_club=7, gws=38):
    """Matches the pre-registration design: ~140 defenders, ~3.8 per club-match, CB 55% at
    8.93/90, FB 45% at 6.12/90, within-role log-SD 0.20, 82% full matches."""
    rng = np.random.default_rng(seed)
    P = n_clubs * per_club
    club = np.repeat(np.arange(n_clubs), per_club)
    is_cb = rng.random(P) < 0.55
    base = np.where(is_cb, 8.928, 6.116) * np.exp(rng.normal(0, 0.20, P) - 0.02)
    avail = rng.beta(2.2, 1.8, P)
    drift = rng.normal(0, drift_sd, P)
    opp_eff = np.exp(rng.normal(0, opp_sd, n_clubs)) if opp_sd > 0 else np.ones(n_clubs)
    rows = []
    for gw in range(1, gws + 1):
        perm = rng.permutation(n_clubs)
        opp_of = np.empty(n_clubs, int)
        opp_of[perm[0::2]] = perm[1::2]; opp_of[perm[1::2]] = perm[0::2]
        cm_shock = np.exp(rng.normal(-shock ** 2 / 2, shock, n_clubs)) if shock > 0 else np.ones(n_clubs)
        play = rng.random(P) < avail
        idx = np.where(play)[0]
        mins = np.where(rng.random(len(idx)) < 0.82, 90.0, rng.integers(60, 90, len(idx)).astype(float))
        m = mins / 90.0
        lam = base[idx] * np.exp(drift[idx] * (gw - 19.5) / 19.0) * opp_eff[opp_of[club[idx]]] \
            * cm_shock[club[idx]]
        if thin:
            y = rng.binomial(np.round(2 * lam * m).astype(int), 0.5)
        elif phi > 0:
            y = rng.poisson(lam * rng.gamma(m / phi, phi))
        else:
            y = rng.poisson(lam * m)
        for j, i in enumerate(idx):
            rows.append((i, gw, club[i], f"g{gw}c{min(club[i], opp_of[club[i]])}", opp_of[club[i]],
                         mins[j], float(y[j])))
    d = pd.DataFrame(rows, columns=["player_code", "gameweek", "club", "match_id", "opp", "mins", "y"])
    role = {i: ("CB" if is_cb[i] else "FB") for i in range(P)}
    return d, role


def phi_opp(rows60, seed=0, band=True):
    """Diagnostic (c): phi net of a block-EB opponent rating fitted on the same rows."""
    d = rows60.copy()
    d["m"] = d["mins"] / 90.0
    d["cm"] = d["club"].astype(str) + "|" + d["match_id"].astype(str)
    o = pd.Series(1.0, index=pd.Index(d["opp"].unique()))
    for _ in range(3):
        e = d["m"] * d["opp"].map(o)
        lam = d.groupby("player_code")["y"].transform("sum") / e.groupby(d["player_code"]).transform("sum")
        exp_ = lam * d["m"]
        g = d.assign(ex=exp_).groupby("opp").agg(y=("y", "sum"), ex=("ex", "sum"))
        lo = np.log(g["y"] / g["ex"])
        resid = d["y"] - exp_ * d["opp"].map(np.exp(lo))
        blk = resid.groupby([d["opp"], d["cm"]]).sum()
        se2 = float(((blk ** 2).groupby(level=0).sum() / g["ex"] ** 2).mean())
        tau2 = max(float(lo.var(ddof=1)) - se2, 0.0)
        kk = tau2 / (tau2 + se2) if tau2 + se2 > 0 else 0.0
        o = np.exp(kk * (lo - lo.mean()))
    e = d["m"] * d["opp"].map(o)
    ph = phi_moment(d["y"], e, d["player_code"])
    bnd = poisson_null_band(d["y"], e, d["player_code"], R=300, seed=seed)[0] if band else None
    return ph, bnd


def odd_even(rows):
    gw = rows["gameweek"]
    return (gw % 2 == 1), (gw % 2 == 0)


def halves(rows):
    gw = rows["gameweek"]
    return gw.between(*TRAIN_GW), gw.between(*TEST_GW)


# =========================================================================================
# 7. Selftest — pre-registration §2, checks 1-7, plus quadrature accuracy
# =========================================================================================
def selftest(verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    fails = []

    def check(ok, msg):
        say(("  PASS  " if ok else "  FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    # 0. quadrature: K_QUAD agrees with 10x as many nodes
    y = np.array([3.0, 9.0, 12.0]); a = np.array([30.0, 180.0, 60.0]); b = np.array([4.0, 24.0, 6.0])
    m = np.array([1.0, 1.0, 0.8])
    p1, l1 = pred_gamma_frailty(y, a, b, m, 0.05)
    p2, l2 = pred_gamma_frailty(y, a, b, m, 0.05, K=16000)
    check(np.max(np.abs(p1 - p2)) < 2e-5 and np.max(np.abs(l1 - l2)) < 2e-4,
          f"quadrature K={K_QUAD} vs 16000: max |dP| {np.max(np.abs(p1 - p2)):.1e}")
    # closed forms agree with the quadrature when the other variance is switched off
    pN, _ = pred_plugin_frailty(y, a / b, m, 0.05)
    pQ, _ = pred_gamma_frailty(y, a * 1e5, b * 1e5, m, 0.05)
    check(np.max(np.abs(pN - pQ)) < 5e-4, "N0 closed form == E2 quadrature at a point-mass rate")

    # 1. recovery of planted phi
    for phi in (0.0, 0.02, 0.05, 0.10):
        est = []
        for s in range(25):
            d, _ = synth_rows(1000 + s, phi=phi)
            d60 = d[d["mins"] >= 60]
            est.append(phi_moment(d60["y"], d60["mins"] / 90, d60["player_code"]))
        est = np.array(est)
        check(abs(est.mean() - phi) < 3 * est.std(ddof=1) / np.sqrt(len(est)) + 0.003,
              f"recover phi={phi:.2f}: mean {est.mean():.4f} sd {est.std(ddof=1):.4f}")
    est = []
    for s in range(15):
        d, _ = synth_rows(1100 + s, phi=0.0, shock=0.15)
        d60 = d[d["mins"] >= 60]
        est.append(phi_moment(d60["y"], d60["mins"] / 90, d60["player_code"]))
    check(np.mean(est) > 0.008, f"shared club-match shock (0.15, phi=0) is absorbed as phi: {np.mean(est):.4f}")

    # 2 & 3. true Poisson, 200 replicates: S1 false rejection <= 7%; full rule fires <= 5%
    R2 = 200
    rej, ships = 0, 0
    for s in range(R2):
        d, role = synth_rows(2000 + s)
        tr, te = halves(d)
        res = run_split(d, role, tr, te, band=False)
        d60 = d[tr & (d["mins"] >= 60)]
        band, _ = poisson_null_band(d60["y"], d60["mins"] / 90, d60["player_code"], R=200, seed=s)
        s1 = res["phi_tr"] > band[1]
        rej += int(s1 or res["phi_tr"] < band[0])
        c = res["c"]
        cmv = res["te"]["cm"].to_numpy(); plv = res["te"]["player_code"].to_numpy()
        se_d = se_pair(c["dB_D"], cmv, plv)[0]; se_e = se_pair(c["elpd_E"], cmv, plv)[0]
        ships += int(s1 and c["M"] > M_MIN and c["dB_D"].mean() + 1.645 * se_d < NI_MARGIN
                     and c["elpd_E"].mean() / se_e > Z_VALID and c["dB_E"].mean() < 0)
    check(rej / R2 <= 0.07, f"true Poisson: S1 two-sided false rejection {rej}/{R2} (<= 7%)")
    check(ships / R2 <= 0.05, f"true Poisson: full VALIDATED rule fires {ships}/{R2} (<= 5%)")
    d, _ = synth_rows(3000, phi=0.0); d2, _ = synth_rows(3000, phi=0.10)
    se = np.sqrt(d["y"].var() / len(d) + d2["y"].var() / len(d2))
    check(abs(d["y"].mean() - d2["y"].mean()) < 4 * se,
          f"frailty leaves E[count] unchanged: {d['y'].mean():.3f} vs {d2['y'].mean():.3f}")

    # 4. under-shrunk plug-in (k/3), phi = 0: reproduces the quintile pattern; phi stays in band
    lo_gap, hi_gap, inband = [], [], 0
    for s in range(15):
        d, role = synth_rows(4000 + s)
        tr, te = halves(d)
        res = run_split(d, role, tr, te, band=False, k_scale=1 / 3)
        P = res["arms"]["P_P0"].to_numpy(); H = res["c"]["H"]
        q = pd.qcut(P, 5, labels=False, duplicates="drop")
        lo_gap.append(H[q == 0].mean() - P[q == 0].mean())
        hi_gap.append(H[q == q.max()].mean() - P[q == q.max()].mean())
        d60 = d[tr & (d["mins"] >= 60)]
        band, _ = poisson_null_band(d60["y"], d60["mins"] / 90, d60["player_code"], R=200, seed=s)
        inband += int(band[0] <= res["phi_tr"] <= band[1])
    check(np.mean(lo_gap) > 0 and np.mean(hi_gap) < 0,
          f"k/3 plug-in: low band under-predicted {np.mean(lo_gap):+.3f}, high over-predicted {np.mean(hi_gap):+.3f}")
    check(inband >= 12, f"k/3 plug-in: phi stays inside its Poisson band {inband}/15")

    # 5. planted opponent effect: unconditional phi leaves band, opponent-conditioned returns
    out_u, out_o = 0, 0
    for s in range(12):
        d, _ = synth_rows(5000 + s, opp_sd=0.20)
        d60 = d[d["mins"] >= 60]
        ph = phi_moment(d60["y"], d60["mins"] / 90, d60["player_code"])
        band, _ = poisson_null_band(d60["y"], d60["mins"] / 90, d60["player_code"], R=200, seed=s)
        out_u += int(ph > band[1])
        po, bo = phi_opp(d60, seed=s)
        out_o += int(po > bo[1])
    check(out_u >= 8 and out_o <= 3,
          f"opponent effect (log-SD 0.20, phi=0): unconditional out of band {out_u}/12, conditioned {out_o}/12")

    # 6. planted drift: half split and odd/even disagree on G0
    sh, so = [], []
    for s in range(10):
        d, role = synth_rows(6000 + s, drift_sd=0.35)
        for split, acc in ((halves(d), sh), (odd_even(d), so)):
            res = run_split(d, role, split[0], split[1], band=False)
            te = res["te"]
            acc.append(g0_slope(te["y"], te["m"], res["arms"]["lam_D"], te["cm"], te["player_code"])[0])
    check(np.mean(so) - np.mean(sh) > 0.05,
          f"drift: G0 slope half-split {np.mean(sh):.3f} below odd/even {np.mean(so):.3f}")

    # 7. planted under-dispersion: phi below band
    below = 0
    for s in range(10):
        d, _ = synth_rows(7000 + s, thin=True)
        d60 = d[d["mins"] >= 60]
        ph = phi_moment(d60["y"], d60["mins"] / 90, d60["player_code"])
        band, _ = poisson_null_band(d60["y"], d60["mins"] / 90, d60["player_code"], R=200, seed=s)
        below += int(ph < band[0])
    check(below >= 8, f"binomial thinning: phi below band {below}/10")

    say(f"selftest: {'OK' if not fails else f'{len(fails)} FAILED'}")
    return not fails


# =========================================================================================
# 8. Real data
# =========================================================================================
def load_rows():
    """All measured 25/26 DEF appearances (any minutes), CBIT, with club/opp/match."""
    import defcon_team as dct
    old = dct.MIN_MINUTES
    dct.MIN_MINUTES = 1
    try:
        p = dct.build_panel("2025-2026")
    finally:
        dct.MIN_MINUTES = old
    p = p[p["position"] == "Defender"].copy()
    p["y"] = p["dc"].astype(float)
    assert p["y"].notna().all(), "a null DefCon reached the study; nulls must be dropped, not filled"
    return p[["player_code", "gameweek", "club", "match_id", "opp", "mins", "y"]].reset_index(drop=True)


def se_inflation(rows, role, phi, k, rm, R=100):
    """c = replicate SD / mean clustered SE, simulated at the fitted phi on the REAL design
    (same rows, minutes, clubs, matches), rates drawn from the fitted prior. Floored at 1."""
    P_codes = rows["player_code"].unique()
    stats_ = {"dB_E": ([], []), "dB_D": ([], []), "elpd_E": ([], [])}
    tr, te = halves(rows)
    for s in range(R):
        rng = np.random.default_rng(9000 + s)
        mu = np.array([rm.get(role.get(pc, "ALL"), rm["ALL"]) for pc in P_codes])
        lam_p = dict(zip(P_codes, rng.gamma(mu * k, 1 / k)))
        lam = rows["player_code"].map(lam_p).to_numpy()
        m = rows["mins"].to_numpy() / 90.0
        y = rng.poisson(lam * rng.gamma(m / phi, phi)) if phi > 0 else rng.poisson(lam * m)
        res = run_split(rows.assign(y=y.astype(float)), role, tr, te, phi=None, band=False)
        cmv = res["te"]["cm"].to_numpy(); plv = res["te"]["player_code"].to_numpy()
        for key, (means, ses) in stats_.items():
            means.append(res["c"][key].mean()); ses.append(se_pair(res["c"][key], cmv, plv)[0])
    return {key: max(1.0, float(np.std(mn, ddof=1) / np.mean(se))) for key, (mn, se) in stats_.items()}


def main():
    print("=" * 88)
    print("SELFTEST (synthetic; the real analysis does not run unless this passes)")
    print("=" * 88)
    if not selftest():
        raise SystemExit("selftest failed: real-data analysis NOT run")

    import defcon_roles as dcr
    role = dcr.role_map()
    rows = load_rows()
    out = []
    rec = lambda sec, stat, val, lo=np.nan, hi=np.nan, note="": out.append(
        {"section": sec, "stat": stat, "value": val, "lo": lo, "hi": hi, "note": note})

    tr, te = halves(rows)
    print("\n" + "=" * 88 + "\nREAL DATA: 25/26 DEF, CBIT, measured rows\n" + "=" * 88)
    print(f"rows {len(rows)}  train 60+ {int((tr & (rows['mins'] >= 60)).sum())}  "
          f"test 60+ {int((te & (rows['mins'] >= 60)).sum())}  players {rows['player_code'].nunique()}")

    # ---- S1 and S1-rep -----------------------------------------------------------------
    main_ = run_split(rows, role, tr, te, band=True, seed=1)
    phi_tr, band_tr = main_["phi_tr"], main_["band_tr"]
    te60 = rows[te & (rows["mins"] >= 60)]
    phi_te = phi_moment(te60["y"], te60["mins"] / 90, te60["player_code"])
    band_te, _ = poisson_null_band(te60["y"], te60["mins"] / 90, te60["player_code"], seed=2)
    S1, S1rep = phi_tr > band_tr[1], phi_te > band_te[1]
    below = phi_tr < band_tr[0]
    rec("phi", "phi_GW1-19", phi_tr, *band_tr, "moment; lo/hi = true-Poisson 2.5/97.5%")
    rec("phi", "phi_GW20-38", phi_te, *band_te, "replication")
    tr60 = rows[tr & (rows["mins"] >= 60)]
    rec("phi", "phi_DM_ML_GW1-19", phi_dm_ml(tr60["y"], tr60["mins"] / 90, tr60["player_code"]),
        note="Dirichlet-multinomial conditional ML cross-check")
    print(f"phi GW1-19 {phi_tr:+.4f}  band [{band_tr[0]:+.4f}, {band_tr[1]:+.4f}]  S1={S1}")
    print(f"phi GW20-38 {phi_te:+.4f}  band [{band_te[0]:+.4f}, {band_te[1]:+.4f}]  S1-rep={S1rep}")
    # The SHIPPING constant (pre-registration §3.4.5: phi fitted on all of 25/26). Reported,
    # never gating; defcon_frailty.PHI_DEF must equal this value.
    all60 = rows[rows["mins"] >= 60]
    phi_all = phi_moment(all60["y"], all60["mins"] / 90, all60["player_code"])
    band_all, _ = poisson_null_band(all60["y"], all60["mins"] / 90, all60["player_code"], seed=4)
    rec("phi", "phi_full_2526", phi_all, *band_all, "shipping constant for bayes_model")
    print(f"phi full 25/26 {phi_all:+.4f}  band [{band_all[0]:+.4f}, {band_all[1]:+.4f}]  (shipping constant)")

    # ---- held-out contrasts --------------------------------------------------------------
    c, arms, T = main_["c"], main_["arms"], main_["te"]
    cmv, plv = T["cm"].to_numpy(), T["player_code"].to_numpy()
    ph_use = max(phi_tr, 0.0)
    cfac = se_inflation(rows, role, ph_use, main_["k"], main_["role_means"])
    res = {}
    for key in ("dB_E", "dB_D", "elpd_E"):
        se0, two = se_pair(c[key], cmv, plv)
        res[key] = (float(c[key].mean()), se0 * cfac[key], two)
        rec("heldout", key, res[key][0], note=f"SE={res[key][1]:.6f} (c={cfac[key]:.2f}); two-way SE={two:.6f}")
    zE = res["elpd_E"][0] / res["elpd_E"][1]
    ni_ub = res["dB_D"][0] + 1.645 * res["dB_D"][1]
    M = c["M"]
    rec("heldout", "M_mean_abs_dP_D", M)
    rec("heldout", "z_elpd_E", zE)
    rec("heldout", "NI_upper_dBrier_D", ni_ub, note=f"margin {NI_MARGIN}")
    for a, v in c["brier"].items():
        rec("brier", a, v)
    rec("prior", "k_EB", main_["k"]); [rec("prior", f"rate_{k_}", v) for k_, v in main_["role_means"].items()]
    print(f"k_EB {main_['k']:.2f}  role means {main_['role_means']}  c {cfac}")
    print("Brier: " + "  ".join(f"{a} {v:.5f}" for a, v in c["brier"].items()))
    print(f"dBrier E2-E1 {res['dB_E'][0]:+.6f} (SE {res['dB_E'][1]:.6f})")
    print(f"dBrier D2-D1 {res['dB_D'][0]:+.6f} (SE {res['dB_D'][1]:.6f})  NI upper {ni_ub:+.6f}")
    print(f"ELPD E2-E1 {res['elpd_E'][0]:+.5f}/row  z {zE:+.2f}   M {M:.4f}")

    # ---- odd/even replication ------------------------------------------------------------
    o_tr, o_te = odd_even(rows)
    oe = run_split(rows, role, o_tr, o_te, band=False)
    oe_elpd = float(oe["c"]["elpd_E"].mean())
    oe_same = np.sign(oe_elpd) == np.sign(res["elpd_E"][0])
    rec("oddeven", "phi_odd", oe["phi_tr"]); rec("oddeven", "elpd_E_even", oe_elpd)
    print(f"odd/even: phi {oe['phi_tr']:+.4f}  ELPD {oe_elpd:+.5f}  same sign {oe_same}")

    # ---- G0 (shrinkage/drift; does not gate frailty) ---------------------------------------
    g0, g0se = g0_slope(T["y"], T["m"], arms["lam_D"], cmv, plv)
    g0e, g0ese = g0_slope(T["y"], T["m"], arms["lam_E"], cmv, plv)
    oT = oe["te"]
    g0o, g0ose = g0_slope(oT["y"], oT["m"], oe["arms"]["lam_D"], oT["cm"], oT["player_code"])
    g0_flag = (g0 + 1.96 * g0se < G0_BAND[0]) or (g0 - 1.96 * g0se > G0_BAND[1])
    rec("G0", "slope_D1_half", g0, g0 - 1.96 * g0se, g0 + 1.96 * g0se)
    rec("G0", "slope_E1_half", g0e, g0e - 1.96 * g0ese, g0e + 1.96 * g0ese)
    rec("G0", "slope_D1_oddeven", g0o, g0o - 1.96 * g0ose, g0o + 1.96 * g0ose)
    print(f"G0 D1 slope {g0:.3f} [{g0 - 1.96 * g0se:.3f}, {g0 + 1.96 * g0se:.3f}]  "
          f"E1 {g0e:.3f}  odd/even D1 {g0o:.3f}  -> open k0 item: {g0_flag}")

    # ---- band reliability (report only) -----------------------------------------------------
    q = pd.qcut(arms["P_D1"], 5, labels=False, duplicates="drop")
    print("\nband (D1 quintile)   n    obs    " + "  ".join(f"{a:>6}" for a in ("P0", "N0", "E1", "E2", "D1", "D2")))
    for b in sorted(q.unique()):
        s = q == b
        obs = c["H"][s.to_numpy()].mean()
        preds = [arms.loc[s, f"P_{a}"].mean() for a in ("P0", "N0", "E1", "E2", "D1", "D2")]
        print(f"  Q{b + 1}               {int(s.sum()):4d}  {obs:.3f}  " + "  ".join(f"{p_:.3f}" for p_ in preds))
        for a, p_ in zip(("P0", "N0", "E1", "E2", "D1", "D2"), preds):
            rec("band", f"Q{b + 1}_{a}", p_, note=f"obs {obs:.4f} n {int(s.sum())}")
    for a in ("D1", "D2", "E1", "E2"):
        P = np.clip(arms[f"P_{a}"].to_numpy(), 1e-6, 1 - 1e-6)
        X = np.column_stack([np.ones(len(P)), np.log(P / (1 - P))])
        try:
            import statsmodels.api as sm  # optional
            slope = float(sm.Logit(c["H"], X).fit(disp=0).params[1])
        except Exception:
            f = lambda b_: -np.sum(c["H"] * (X @ b_) - np.logaddexp(0, X @ b_))
            slope = float(optimize.minimize(f, np.array([0.0, 1.0])).x[1])
        rec("calib_slope", a, slope)
        print(f"logistic calibration slope {a}: {slope:.3f}")

    # ---- diagnostics (report only) -----------------------------------------------------------
    rm_ = dict(role)
    for r_ in ("CB", "FB"):
        s = tr60[tr60["player_code"].map(rm_) == r_]
        rec("diag", f"phi_{r_}_GW1-19", phi_moment(s["y"], s["mins"] / 90, s["player_code"]))
    po, bo = phi_opp(tr60, seed=3)
    rec("diag", "phi_opp_GW1-19", po, *bo, "net of block-EB opponent rating")
    mean_m = tr60.groupby("player_code")["mins"].transform("mean") / 90
    rec("diag", "phi_meanminutes_GW1-19", phi_moment(tr60["y"], mean_m, tr60["player_code"]),
        note="(d): exposure = player's mean minutes")
    lam_i = tr60.groupby("player_code")["y"].transform("sum") / (tr60["mins"] / 90).groupby(tr60["player_code"]).transform("sum")
    mu = lam_i * tr60["mins"] / 90
    pr = (tr60["y"] - mu) / np.sqrt(mu.where(mu > 0))
    cmk = tr60["club"].astype(str) + "|" + tr60["match_id"].astype(str)
    g = pr.groupby(cmk)
    num = float(((g.sum() ** 2) - (pr ** 2).groupby(cmk).sum()).sum())
    den = float(((g.size() - 1) * (pr ** 2).groupby(cmk).sum()).sum())
    rec("diag", "within_clubmatch_resid_corr", num / den if den > 0 else np.nan)
    r_tr = tr60.groupby("player_code").agg(y=("y", "sum"), M=("mins", "sum"))
    r_tr["rate"] = r_tr["y"] / (r_tr["M"] / 90)
    terc = pd.qcut(r_tr["rate"], 3, labels=["low", "mid", "high"])
    for t in ("low", "mid", "high"):
        s = tr60[tr60["player_code"].isin(terc[terc == t].index)]
        ph = phi_moment(s["y"], s["mins"] / 90, s["player_code"])
        lr = float(s["y"].sum() / (s["mins"].sum() / 90))
        rec("diag", f"NB1_D_{t}_rate_tercile", 1 + lr * ph, note=f"rate {lr:.2f}, phi {ph:.4f}")
    print(f"diag: phi_opp {po:+.4f} band {bo}; within club-match resid corr "
          f"{num / den if den > 0 else np.nan:+.3f}")

    # ---- decision (pre-registration §3; ties go to the status quo) ---------------------------
    if below:
        verdict, reason = "FINDING-UNDERDISPERSED", "phi below its Poisson band; gamma frailty cannot express it; nothing ships"
    else:
        valid = (S1 and S1rep and M > M_MIN and zE > Z_VALID and res["dB_E"][0] < 0
                 and ni_ub < NI_MARGIN and bool(oe_same))
        off = S1 and M > M_MIN and ni_ub < NI_MARGIN and zE > Z_SHAPE
        if valid:
            verdict, reason = "VALIDATED", "all validated conditions hold"
        elif off:
            verdict, reason = "OFF-BY-DEFAULT", "phi real, non-inferior, validated conditions not all met"
        else:
            why = []
            if not S1: why.append("phi inside its Poisson band")
            if M <= M_MIN: why.append(f"immaterial (M={M:.4f} <= {M_MIN})")
            if ni_ub >= NI_MARGIN: why.append(f"non-inferiority fails (upper {ni_ub:+.5f} >= {NI_MARGIN})")
            if zE <= Z_SHAPE: why.append(f"frailty shape wrong (ELPD z {zE:+.2f} <= {Z_SHAPE})")
            verdict, reason = "NULL", "; ".join(why) or "conditions for OFF not met"
    rec("decision", verdict, np.nan, note=reason)
    rec("decision", "G0_open_k0_item", float(g0_flag))
    print("\n" + "=" * 88 + f"\nDECISION: {verdict} — {reason}\nG0 opens a separate k0 item: {g0_flag}\n" + "=" * 88)
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        raise SystemExit(0 if selftest() else 1)
    main()
