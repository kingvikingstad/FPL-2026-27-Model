import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
elo_pin_sensitivity.py — how much can the stale Elo pin be wrong by?
====================================================================

`teams.csv` ships an `elo` column that went entirely NULL upstream and has stayed
NULL — verified against `origin/main` on 2026-09-08, so this is not a stale-clone
artifact and will not self-heal. `core_insights._resolve_elo` falls back to the pinned
snapshot in `data/team_elo_2627.csv`, which is dated 2026-08-14 and therefore contains
no 26/27 result at all. `scripts/doctor.py` reports that.

The question doctor cannot answer is whether it MATTERS, and the answer was being
asserted rather than measured. This measures it.

PRE-REGISTERED, before any number was read:
  Elo enters at `bayes_model.fit` (bayes_model.py:166) as a blend into the PRIOR MEANS
  m_att / m_def at ELO_WEIGHT=0.45. It is not a likelihood term. The E0 reconstruction
  (380 match rows) plus the in-season update therefore sit on top of it, so the prior's
  influence should be modest and should shrink as evidence accumulates.

  DECISION RULE, fixed in advance: the pin is a documented non-issue if removing the
  Elo channel ENTIRELY — the upper bound on any error a stale pin can introduce, since
  a wrong Elo cannot be worse than no Elo plus a wrong nudge — moves net team strength
  by less than 0.5 posterior SD at the maximum and leaves the club ranking at Spearman
  rho > 0.98. Otherwise refreshing it is urgent and blocks the next deadline lock.

  Note the asymmetry that makes this a bound rather than an estimate: A-vs-B is the
  whole channel. The actual staleness error is the difference between the August pin
  and the true current Elo, which is far smaller than the difference between the August
  pin and nothing — August's strong clubs are still August's strong clubs three
  gameweeks in. Reporting the bound is deliberate: it is the conservative direction.

RESULT (2026-09-08, ELO_WEIGHT=0.45, S=1500): mean |delta net| 0.031, max 0.087 =
0.30 posterior SD, Spearman rho = 0.9955. Rule cleared. The pin is a documented
non-issue; refreshing it is housekeeping, not a blocker. Re-run this if ELO_WEIGHT
rises, or if the E0 evidence base ever shrinks.

Run:  .\fpl.ps1 run studies/elo_pin_sensitivity.py
      .\fpl.ps1 run studies/elo_pin_sensitivity.py --selftest
"""
import warnings

warnings.filterwarnings("ignore")

OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "elo_pin_sensitivity.csv")

# Fixed in advance. See the docstring.
MAX_SD_ALLOWED = 0.5
MIN_RHO_ALLOWED = 0.98


def strength(pclub, e0, clubelo, w, seed=0):
    """Posterior team strength under one Elo configuration."""
    import numpy as np
    import pandas as pd
    from bayes_model import TeamModel
    np.random.seed(seed)                       # same draws across arms; the contrast is the point
    tm = TeamModel(promoted_per_club=pclub).fit(e0_path=e0, clubelo=clubelo,
                                                clubelo_weight=w)
    s = tm.sample_2627(S=1500)
    A, D = s["att"], s["dfn"]
    out = pd.DataFrame({"team": s["teams"], "attack": A.mean(0), "defence": D.mean(0),
                        "att_sd": A.std(0), "def_sd": D.std(0)})
    out["net"] = out["attack"] + out["defence"]
    out["clubelo_used"] = int(getattr(tm, "clubelo_used", 0))
    return out.set_index("team").sort_index()


def run(elo_weight=None):
    import numpy as np
    import pandas as pd
    import core_insights as ci

    w = float(elo_weight if elo_weight is not None
              else _os.environ.get("ELO_WEIGHT", "0.45"))
    _d, t26, _gw = ci.load()
    elo = ci.to_elo_frame(t26)
    pclub = ci.promoted_prior_from_elo(t26)["per_club"]
    # The board fits on the in-season E0 when it exists; using the static file instead
    # would measure the sensitivity of a model nobody runs.
    e0 = _os.path.join(config.SCRATCH, "E0_recon_inseason.csv")
    if not _os.path.exists(e0):
        e0 = config.E0_RECON
    print(f"E0 source: {e0}")
    print(f"ELO_WEIGHT: {w}   Elo rows: {int(elo['Elo'].notna().sum())}/{len(elo)}")

    A = strength(pclub, e0, elo, w)            # live
    B = strength(pclub, e0, None, w)           # the bound: no Elo at all
    C = strength(pclub, e0, elo, w / 2)        # is the response ~linear in weight?

    sd = float(np.sqrt((A["att_sd"] ** 2 + A["def_sd"] ** 2).mean()))
    rows = []
    for label, X in (("no_elo", B), ("half_weight", C)):
        d = X["net"] - A["net"]
        rho = float(A["net"].corr(X["net"], method="spearman"))
        rows.append({"arm": label, "mean_abs_delta_net": round(float(d.abs().mean()), 5),
                     "max_abs_delta_net": round(float(d.abs().max()), 5),
                     "max_in_posterior_sd": round(float(d.abs().max() / sd), 4),
                     "spearman_rho": round(rho, 5), "posterior_sd": round(sd, 5),
                     "elo_weight": w})
        print(f"\n{label}: mean |delta net| {d.abs().mean():.4f}  max {d.abs().max():.4f} "
              f"({d.abs().max() / sd:.2f} posterior SD)  rho {rho:.4f}")
        for team, v in d.abs().sort_values(ascending=False).head(5).items():
            print(f"    {team:18s} {A.loc[team, 'net']:+.4f} -> {X.loc[team, 'net']:+.4f}"
                  f"  ({X.loc[team, 'net'] - A.loc[team, 'net']:+.4f})")

    ev = pd.DataFrame(rows)
    ev.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")

    bound = ev[ev["arm"] == "no_elo"].iloc[0]
    ok = (bound["max_in_posterior_sd"] < MAX_SD_ALLOWED
          and bound["spearman_rho"] > MIN_RHO_ALLOWED)
    print(f"\nRULE ({MAX_SD_ALLOWED} SD and rho > {MIN_RHO_ALLOWED}): "
          f"{'CLEARED — the pin is a documented non-issue' if ok else 'FAILED — refresh the pin'}"
          f"  [{bound['max_in_posterior_sd']:.2f} SD, rho {bound['spearman_rho']:.4f}]")
    return 0


def selftest():
    """Offline. Exercises the contrast arithmetic on synthetic posteriors; does not fit
    a model, because the harness must not spend two minutes on three MCMC runs."""
    import numpy as np
    import pandas as pd
    teams = [f"T{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"attack": rng.normal(0, .3, 20), "defence": rng.normal(0, .3, 20),
                         "att_sd": .2, "def_sd": .2}, index=teams)
    base["net"] = base["attack"] + base["defence"]
    sd = float(np.sqrt((base["att_sd"] ** 2 + base["def_sd"] ** 2).mean()))
    assert abs(sd - np.sqrt(0.08)) < 1e-9, sd
    # an identical arm is a zero contrast at rho 1
    d = base["net"] - base["net"]
    assert d.abs().max() == 0
    assert abs(base["net"].corr(base["net"], method="spearman") - 1.0) < 1e-12
    # a pure monotone shift leaves the RANKING intact but not the magnitude — the two
    # criteria in the rule are independent, and a study that reported only rho would
    # pass a uniform bias that moves every club
    shifted = base.assign(net=base["net"] + 1.0)
    assert abs(shifted["net"].corr(base["net"], method="spearman") - 1.0) < 1e-12
    assert (shifted["net"] - base["net"]).abs().max() > MAX_SD_ALLOWED * sd
    # the rule is a conjunction, so that case must FAIL
    ok = ((shifted["net"] - base["net"]).abs().max() / sd < MAX_SD_ALLOWED
          and shifted["net"].corr(base["net"], method="spearman") > MIN_RHO_ALLOWED)
    assert not ok, "a uniform shift must fail the rule even at rho = 1"
    print("SELFTEST OK: contrast is scaled by the pooled posterior SD, and the decision "
          "rule is a conjunction that a rank-preserving uniform shift still fails.")
    return 0


if __name__ == "__main__":
    _sys.exit(selftest() if "--selftest" in _sys.argv else run())
