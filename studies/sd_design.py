from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
"""
sd_design.py -- pre-registration arithmetic for the soccerdata integration.

Three questions that determine whether the proposed tests are worth running:
  A. Is rho_SP = 0.204 a low-persistence FINDING or a measurement-error ARTEFACT?
  B. How much Fisher information does the Beta-Binomial DefCon threshold discard
     relative to a count model on the same data?
  C. What is the MDE on the rho_OP vs rho_SP difference at realistic panel size?

Nothing here touches real data. It sizes the tests before they are run.

CAVEAT ON PART A [VERIFIED, see docs/SOCCERDATA_FINDINGS.md]
------------------------------------------------------------
`attenuation_study` reports `rho_disattenuated = rho_measured / reliability`. That
correction is BIASED UPWARD for low-count components: feeding rho_true=0.50 in, this
script returns 0.634 for set_piece against 0.505 for well-measured open_play. The
identity `plim rho_hat = rho_true * reliability` assumes additive homoskedastic error,
which log rates built from a handful of shots do not have.

Part A is retained as the pre-registration record, and its RELIABILITY estimates are
sound. Do not use its `rho_disattenuated` column as an estimate — use
`setpiece.corrected_ar1()`, which inverts the attenuation by simulation instead
(bias +0.003 vs +0.037 for the ratio, over a 12-seed study).
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)


# ---------------------------------------------------------------- A. attenuation
def attenuation_study(n_players=350, matches=34, rho_true=0.50,
                      sp_xg_per_match=0.035, op_xg_per_match=0.22,
                      n_sims=400):
    """A season set-piece xG rate is a low-count average, so it is measured with
    error. AR(1) slopes are attenuated by the reliability of the REGRESSOR:

        plim rho_hat = rho_true * reliability(x_t)

    Simulate both components as compound Poisson (shot count x xG per shot) and
    report measured rho, split-half reliability, and the disattenuated estimate.
    """
    out = {}
    for name, rate in (("set_piece", sp_xg_per_match), ("open_play", op_xg_per_match)):
        rhos, rels = [], []
        for _ in range(n_sims):
            # latent per-player true rate, season t
            mu_t = np.exp(rng.normal(np.log(rate), 0.55, n_players))
            # true AR(1) in the LATENT rate
            mu_t1 = np.exp(np.log(rate) + rho_true * (np.log(mu_t) - np.log(rate))
                           + rng.normal(0, 0.55 * np.sqrt(1 - rho_true**2), n_players))

            def observe(mu, m):
                # shots ~ Poisson(lambda), xG per shot ~ Beta-ish; xG = sum
                lam = mu / 0.09                       # ~0.09 xG per shot
                k = rng.poisson(lam * m)
                xg = rng.gamma(shape=np.maximum(k, 1e-9) * 2.0, scale=0.045)
                return xg / m

            x_t, x_t1 = observe(mu_t, matches), observe(mu_t1, matches)
            h1, h2 = observe(mu_t, matches // 2), observe(mu_t, matches // 2)

            rhos.append(np.polyfit(np.log(x_t + 1e-6), np.log(x_t1 + 1e-6), 1)[0])
            r = np.corrcoef(np.log(h1 + 1e-6), np.log(h2 + 1e-6))[0, 1]
            rels.append(2 * r / (1 + r))              # Spearman-Brown to full length

        rho_m, rel = float(np.mean(rhos)), float(np.mean(rels))
        out[name] = dict(rho_measured=rho_m, reliability=rel,
                         rho_disattenuated=rho_m / max(rel, 1e-6))
    return out


def implied_true_rho(rho_measured=0.204, reliability_grid=(0.25, 0.40, 0.55, 0.70, 0.85)):
    """Given the project's MEASURED rho_SP, what does true persistence look like
    across plausible reliabilities? Drives the size of the mean-reversion haircut."""
    return {r: dict(rho_true=rho_measured / r,
                    reversion_pct=100 * (1 - rho_measured / r))
            for r in reliability_grid}


# ---------------------------------------------------------------- B. information loss
def _nb_pmf(y, mu, k):
    return stats.nbinom.pmf(y, k, k / (k + mu))


def fisher_count_vs_threshold(mu_grid=(6, 9, 12, 15), k=4.0, thresholds=(10, 12)):
    """Fisher information about log(mu) from the full count Y ~ NB(mu, k)
    versus from the binarised Z = 1{Y >= T}.

    I_count(log mu)  = mu * k / (mu + k)                      [NB, exact]
    I_binary(log mu) = (dp/dlog mu)^2 / (p (1-p)),  p = P(Y >= T)
    The ratio is the effective sample-size multiplier from modelling counts.
    """
    rows = []
    for T in thresholds:
        for mu in mu_grid:
            I_count = mu * k / (mu + k)
            p = 1.0 - stats.nbinom.cdf(T - 1, k, k / (k + mu))
            h = 1e-5
            pu = 1.0 - stats.nbinom.cdf(T - 1, k, k / (k + mu * np.exp(h)))
            pd_ = 1.0 - stats.nbinom.cdf(T - 1, k, k / (k + mu * np.exp(-h)))
            dp = (pu - pd_) / (2 * h)
            I_bin = dp**2 / max(p * (1 - p), 1e-12)
            rows.append(dict(threshold=T, mu=mu, p_hit=p,
                             I_count=I_count, I_binary=I_bin,
                             eff_n_multiplier=I_count / max(I_bin, 1e-12)))
    return rows


# ---------------------------------------------------------------- C. MDE
def mde_rho_difference(n=350, sd_x=0.55, n_sims=600, alpha=0.05, power=0.80):
    """Two independent AR(1) slopes (open-play, set-piece) on the same players.
    Test H0: rho_OP - rho_SP = 0. SE of each slope ~ sqrt((1-rho^2)/n) inflated
    by measurement error; the difference SE is the root-sum-square (the two
    residuals are only weakly correlated across components).
    Returns the detectable gap at 80% power, 5% two-sided.
    """
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    res = {}
    for rel_sp in (0.30, 0.45, 0.60, 0.80):
        se_op = np.sqrt((1 - 0.5**2) / n) / 0.85      # open-play well measured
        se_sp = np.sqrt((1 - 0.2**2) / n) / rel_sp
        se_d = np.sqrt(se_op**2 + se_sp**2)
        res[rel_sp] = dict(se_diff=se_d, mde=(z_a + z_b) * se_d)
    return res


if __name__ == "__main__":
    print("=" * 74)
    print("A. ATTENUATION -- is rho_SP=0.204 a finding or an artefact?")
    print("=" * 74)
    att = attenuation_study()
    for k_, v in att.items():
        print(f"  {k_:<10} rho_measured={v['rho_measured']:.3f}  "
              f"reliability={v['reliability']:.3f}  "
              f"rho_disattenuated={v['rho_disattenuated']:.3f}")
    print("\n  Project's MEASURED rho_SP = 0.204. Implied TRUE persistence:")
    for r, v in implied_true_rho().items():
        print(f"    reliability {r:.2f} -> rho_true {v['rho_true']:.3f}  "
              f"=> mean reversion {v['reversion_pct']:.0f}% (not 80%)")

    print("\n" + "=" * 74)
    print("B. INFORMATION LOSS -- Beta-Binomial threshold vs count model")
    print("=" * 74)
    print(f"  {'T':>3} {'mu':>5} {'P(hit)':>8} {'I_count':>9} {'I_binary':>9} {'eff n x':>9}")
    for r in fisher_count_vs_threshold():
        print(f"  {r['threshold']:>3} {r['mu']:>5} {r['p_hit']:>8.3f} "
              f"{r['I_count']:>9.3f} {r['I_binary']:>9.3f} "
              f"{r['eff_n_multiplier']:>9.2f}")

    print("\n" + "=" * 74)
    print("C. MDE on rho_OP - rho_SP  (n=350 players, 80% power, 5% two-sided)")
    print("=" * 74)
    for rel, v in mde_rho_difference().items():
        print(f"  set-piece reliability {rel:.2f} -> SE(diff)={v['se_diff']:.3f}  "
              f"MDE={v['mde']:.3f}")
