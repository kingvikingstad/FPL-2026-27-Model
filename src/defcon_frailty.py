from __future__ import annotations
"""
defcon_frailty.py — per-match overdispersion of a DEFENDER's DefCon count.  OFF BY DEFAULT.
===========================================================================================
bayes_model.project() pays DefCon when the match count reaches the threshold. It draws the
count as Poisson(rate x m90), with `rate` one Gamma posterior draw per path. Across paths that
is already negative binomial, but the extra spread is POSTERIOR RATE UNCERTAINTY: it shrinks
as a player accumulates minutes (alpha ~ 184 for a regular defender, var/mean ~ 1.04). Nothing
represents variation from match to match at a KNOWN rate. This adds it as a gamma-process
frailty:

    count | rate, G ~ Poisson(rate x G),   G ~ Gamma(shape m90 / phi, scale phi)

so E[G] = m90 (the mean is untouched) and Var = rate m90 (1 + rate phi).

EVIDENCE  [VERIFIED 2026-09-17, studies/defcon_threshold_calibration.py/.csv]
------------------------------------------------------------------------------
Pre-registered in docs/DEFCON_THRESHOLD_CALIBRATION_PREREG_2026-09-16.md before any fit.
25/26 DEF CBIT, 60+ min, within-player moment estimator against a simulated true-Poisson band:
phi GW1-19 +0.053 and GW20-38 +0.042, each outside a band of about +/-0.01; +0.047 net of
opponent ratings, so opponent supply does not explain it. Held out on GW20-38: count log score
+0.029/row (z +4.1), odd/even split the same sign; as-deployed Brier D2-D1 -0.00044, upper bound
+0.00055 inside the +0.001 non-inferiority margin. It is NOT validated because on the
Gamma-integrated arm Brier went the wrong way (+0.00028). The pre-registered verdict is
therefore OFF-BY-DEFAULT, pending a 26/27 replication.

SCOPE
-----
DEF only. MID/FWD score CBIRT against 12, and this phi must not be transferred to them; they
need their own registration. The draw uses a child stream (bayes_model._defcon_rng), so turning
it on changes the DefCon component and nothing else in a same-seed A/B.

Switch: FPL_DEFCON_FRAILTY=on   (anything else, or unset, is off)
Run:    python src/defcon_frailty.py --selftest
"""
import os
import numpy as np

# phi fitted on all 25/26 DEF 60+ min rows: `phi_full_2526` in studies/defcon_threshold_calibration.csv
PHI_DEF = 0.0494
ENV = "FPL_DEFCON_FRAILTY"


def enabled():
    return os.environ.get(ENV, "").strip().lower() in ("on", "1", "true", "yes")


def phi_for(pos):
    """The frailty to apply at `pos`: PHI_DEF for a defender when switched on, else 0."""
    return PHI_DEF if (pos == "DEF" and enabled()) else 0.0


def draw_count(rng, rate, m90, phi):
    """One DefCon count per path. phi = 0 is exactly rng.poisson(rate x m90), so the switch-off
    path consumes the same draws as before. Paths with m90 = 0 get G = 0 (no minutes, no count)."""
    rate = np.asarray(rate, float); m90 = np.asarray(m90, float)
    if not phi > 0:
        return rng.poisson(np.maximum(rate * m90, 0))
    G = np.zeros_like(m90 * rate)
    on = np.broadcast_to(m90, G.shape) > 0
    G[on] = rng.gamma(np.broadcast_to(m90, G.shape)[on] / phi, phi)
    return rng.poisson(np.maximum(rate * G, 0))


def selftest():
    fails = []

    def check(ok, msg):
        print(("  PASS  " if ok else "  FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    S = 400_000
    rate = np.full(S, 7.5); m90 = np.where(np.arange(S) % 5 == 0, 0.0, 1.0)
    # off == plain Poisson, bit-identical draws from the same seed
    a = draw_count(np.random.default_rng(1), rate, m90, 0.0)
    b = np.random.default_rng(1).poisson(np.maximum(rate * m90, 0))
    check(np.array_equal(a, b), "phi=0 reproduces rng.poisson bit-for-bit")
    y = draw_count(np.random.default_rng(2), rate, m90, 0.05)
    played = m90 > 0
    check(np.all(y[~played] == 0), "no minutes -> no count")
    mu, var = y[played].mean(), y[played].var()
    check(abs(mu - 7.5) < 0.02, f"mean unchanged: {mu:.3f} vs 7.5")
    check(abs(var / mu - (1 + 7.5 * 0.05)) < 0.02, f"var/mean {var / mu:.3f} vs {1 + 7.5 * 0.05:.3f}")
    # closed form: G ~ Gamma(m/phi, phi) -> NB(size m/phi, mean rate m)
    from scipy import stats
    size = 1.0 / 0.05
    p_nb = 1 - stats.nbinom.cdf(9, size, size / (size + 7.5))
    check(abs((y[played] >= 10).mean() - p_nb) < 0.003,
          f"P(>=10) {(y[played] >= 10).mean():.4f} vs NB closed form {p_nb:.4f}")
    # switch
    old = os.environ.get(ENV)
    try:
        os.environ.pop(ENV, None)
        check(phi_for("DEF") == 0.0, "off by default")
        os.environ[ENV] = "on"
        check(phi_for("DEF") == PHI_DEF and phi_for("MID") == 0.0 and phi_for("FWD") == 0.0
              and phi_for("GK") == 0.0, "on: DEF only")
    finally:
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old
    print(f"selftest: {'OK' if not fails else f'{len(fails)} FAILED'}")
    return not fails


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)
    print(f"PHI_DEF={PHI_DEF}  enabled={enabled()}  ({ENV})")
