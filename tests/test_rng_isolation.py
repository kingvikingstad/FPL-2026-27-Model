import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
test_rng_isolation.py — one player's randomness must not depend on another's.
==============================================================================
`project()` used to draw every player from one shared sequential generator. numpy's
beta/gamma/poisson use rejection sampling, so the number of raw bits a draw consumes
depends on the PARAMETER VALUES — changing one player's start_a re-randomised every
player simulated after him. Measured before the fix: perturbing a single Arsenal player
moved 48 projections, 19 of them at other clubs.

The board was still reproducible given identical input, so the determinism check passed
and nothing looked wrong. What was wrong was every A/B: a measured difference was the
real effect plus a re-randomisation term of roughly the same size as the small effects
this project spends its time trying to detect.

These four properties are what "clean A/B" actually means, and none of them held before:

  1. LOCAL      perturbing one player moves that player and nobody else
  2. DETERMINISM identical input reproduces bitwise
  3. ORDER      shuffling the player frame changes no projection
  4. DELETION   removing a player changes no surviving projection

Run:  python tests/test_rng_isolation.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import bayes_model
from bayes_model import TeamModel, project

S = 400
TOL = 0.0          # these are exact properties, not statistical ones


def _frame(n=60):
    """Synthetic players across several clubs, with deliberately varied priors.

    Varied on purpose: identical parameters would consume identical numbers of random
    bits and the old shared-stream code would pass a test built on them.
    """
    rs = np.random.default_rng(11)
    teams = ["Arsenal", "Man City", "Liverpool", "Everton", "Brentford", "Hull"]
    rows = []
    for i in range(n):
        pos = ["GK", "DEF", "MID", "FWD"][i % 4]
        rows.append({
            "id": 1000 + i, "player_code": 500000 + i, "web_name": f"P{i}",
            "pos": pos, "team": teams[i % len(teams)],
            "own": float(rs.uniform(0.1, 40)), "cost": float(rs.integers(40, 130)) / 10,
            "npxgi_alpha": float(rs.uniform(0.2, 4.0)), "npxgi_beta": float(rs.uniform(1, 12)),
            "xa_alpha": float(rs.uniform(0.1, 2.0)), "xa_beta": float(rs.uniform(1, 12)),
            "defcon_alpha": float(rs.uniform(2, 40)), "defcon_beta": float(rs.uniform(1, 8)),
            "start_a": float(rs.uniform(1, 30)), "start_b": float(rs.uniform(1, 30)),
            "sub_app_rate": float(rs.uniform(0, 0.6)),
            "exp_minutes": float(rs.uniform(60, 90)),
            # a penalty taker consumes an EXTRA draw per fixture; under the old shared
            # stream that alone shifted everyone after him
            "pen_xg90": 0.12 if i % 17 == 0 else 0.0,
            "minutes": 900.0, "cold_start": False,
        })
    return pd.DataFrame(rows)


def _ts(tm):
    bayes_model.rng = np.random.default_rng(7)
    return tm.sample_2627(S=S)


def main():
    tm = TeamModel().fit(e0_path=config.E0_RECON)
    ts = _ts(tm)
    pl = _frame()
    base = project(pl, tm, ts, 2, 2, S=S).set_index("id")["mean"]
    print(f"baseline: {len(base)} players projected, S={S}")
    ok = True

    def cmp(other, label, expect_moved=None):
        nonlocal ok
        m = pd.concat([base.rename("a"), other.rename("b")], axis=1, join="inner")
        moved = m.index[(m["a"] - m["b"]).abs() > TOL].tolist()
        good = (moved == expect_moved) if expect_moved is not None else (len(moved) == 0)
        print(f"  {'ok  ' if good else 'FAIL'} {label:52s} moved {len(moved)}/{len(m)}"
              + (f"  {moved[:5]}" if moved and not good else ""))
        ok &= good

    # 1. LOCAL — perturb one player's start prior
    p2 = pl.copy()
    tgt = int(p2.loc[3, "id"])
    p2.loc[3, "start_a"] = float(p2.loc[3, "start_a"]) + 7.0
    cmp(project(p2, tm, ts, 2, 2, S=S).set_index("id")["mean"],
        "perturbing one player moves only that player", expect_moved=[tgt])

    # 1b. and the same for a penalty flag, which changes the DRAW COUNT per fixture
    p3 = pl.copy()
    tgt3 = int(p3.loc[5, "id"])
    p3.loc[5, "pen_xg90"] = 0.25
    cmp(project(p3, tm, ts, 2, 2, S=S).set_index("id")["mean"],
        "granting a penalty duty moves only that player", expect_moved=[tgt3])

    # 2. DETERMINISM
    cmp(project(pl, tm, ts, 2, 2, S=S).set_index("id")["mean"],
        "identical input reproduces bitwise")

    # 3. ORDER
    cmp(project(pl.sample(frac=1.0, random_state=5), tm, ts, 2, 2, S=S)
        .set_index("id")["mean"], "shuffling the player frame changes nothing")

    # 4. DELETION
    cmp(project(pl.drop(index=[7, 19]), tm, ts, 2, 2, S=S).set_index("id")["mean"],
        "dropping players leaves survivors untouched")

    # 5. the seed is a real knob: a different seed must actually re-randomise
    alt = project(pl, tm, ts, 2, 2, S=S, seed=99).set_index("id")["mean"]
    n_moved = int((base - alt).abs().gt(0).sum())
    good = n_moved > 0.9 * len(base)
    print(f"  {'ok  ' if good else 'FAIL'} {'a different seed re-randomises everyone':52s} "
          f"moved {n_moved}/{len(base)}")
    ok &= good
    # ... without moving the distribution
    shift = abs(float(alt.mean() - base.mean()))
    good = shift < 0.25
    print(f"  {'ok  ' if good else 'FAIL'} {'and leaves the mean where it was':52s} "
          f"|delta mean| {shift:.4f} (tol 0.25, MC at S={S})")
    ok &= good

    print("\n" + ("ALL RNG ISOLATION PROPERTIES HOLD" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    _sys.exit(main())
