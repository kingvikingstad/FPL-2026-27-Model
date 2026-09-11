import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
test_shared_concessions.py — team-mates concede the same goals.
================================================================

Acceptance test for `bayes_model._team_rng` (2026-09-11).

THE DEFECT
----------
`project()` drew goals conceded INSIDE the player loop, on each player's own stream, so
two defenders at one club received two independent realisations of the same match.
Measured on the GW4 draws dump:

    same-club nailed-defender pairs    r = +0.033
    simulated, concessions independent r = +0.000
    simulated, concessions SHARED      r = +0.275     <- what it should be

Every per-player marginal was correct — mean, sd, p5/p95 — so neither the board
invariants nor the determinism re-run could see it. Only a quantity built from SEVERAL
players' draws at once exposes it, and those are exactly what the draws are retained for:
a two-defender stack's sd was understated by ~11%, a three-defender stack's by ~21%,
in the direction that makes stacking a defence look safer than it is.

WHAT THIS PINS
--------------
1. IDENTITY. Two nailed defenders at one club, with every other scoring channel switched
   off, score only through the goals their club concedes. Under a shared draw their
   points must be BITWISE equal in every path — not merely correlated. This is the crisp
   form of the property and it cannot pass by accident.
2. The same two players at DIFFERENT clubs must not be equal — otherwise test 1 could be
   passing because concessions were degenerate rather than shared.
3. LOCALITY. A club's concessions come from a team-keyed stream, so they must not depend
   on which other players are in the frame or in what order. Drawing them from, say, the
   first player at each club would re-randomise a whole defence whenever the squad list
   changed — the defect per-player streams were introduced to remove.
4. One club's concessions do not leak into another club's.
"""
import numpy as np
import pandas as pd
import bayes_model
from bayes_model import TeamModel, project

S = 2000
GW = 4


def _defender(pid, code, name, team):
    """A defender whose points can ONLY come from appearance and concessions.

    Nailed (p_start ~ 1, 90 minutes, never a substitute) so `played60` holds in every
    path, and attacking / DefCon rates set to effectively zero so no other channel adds
    per-player randomness. What remains is 2 + clean sheet + concession penalty, all of
    it a function of the club's goals conceded.
    """
    return {"id": pid, "player_code": code, "web_name": name, "pos": "DEF", "team": team,
            "own": 5.0, "cost": 5.0,
            "npxgi_alpha": 1e-9, "npxgi_beta": 1e6, "xa_alpha": 1e-9, "xa_beta": 1e6,
            "defcon_alpha": 1e-9, "defcon_beta": 1e6,
            "start_a": 1e7, "start_b": 1e-3, "sub_app_rate": 0.0,
            "exp_minutes": 90.0, "pen_xg90": 0.0, "minutes": 900.0, "cold_start": False}


def _noise(pid, code, name, team, pos):
    """An ordinary player with live attacking and DefCon channels — frame filler whose
    only job is to change the frame's contents and order around the defenders."""
    return {"id": pid, "player_code": code, "web_name": name, "pos": pos, "team": team,
            "own": 10.0, "cost": 6.0,
            "npxgi_alpha": 2.0, "npxgi_beta": 5.0, "xa_alpha": 1.0, "xa_beta": 6.0,
            "defcon_alpha": 12.0, "defcon_beta": 2.0,
            "start_a": 12.0, "start_b": 4.0, "sub_app_rate": 0.3,
            "exp_minutes": 80.0, "pen_xg90": 0.0, "minutes": 900.0, "cold_start": False}


def _draws(frame, tm, ts):
    """{player id: draw vector} — `project` sorts its rows by mean, so key by id rather
    than trusting position."""
    res, D = project(pd.DataFrame(frame), tm, ts, GW, GW, S=S, seed=7, return_draws=True)
    return {int(i): D[k] for k, i in enumerate(res["id"])}


def main():
    tm = TeamModel().fit(e0_path=config.E0_RECON)
    bayes_model.rng = np.random.default_rng(7)
    ts = tm.sample_2627(S=S)
    ok = True

    def check(good, label, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if good else 'FAIL'} {label:60s} {detail}")
        ok &= bool(good)

    # 1. IDENTITY — same club, identical draws
    a = _defender(1, 900001, "DefA", "Arsenal")
    b = _defender(2, 900002, "DefB", "Arsenal")
    d = _draws([a, b], tm, ts)
    diff = float(np.max(np.abs(d[1] - d[2])))
    check(diff == 0.0, "two nailed defenders at one club score identically",
          f"max|diff| = {diff:.6f}")
    # guard: the identity must not come from a degenerate (always-zero) concession draw
    spread = float(np.std(d[1]))
    check(spread > 0.5, "  ... and their points actually vary across paths",
          f"sd = {spread:.3f}")

    # 2. DIFFERENT clubs must NOT be identical
    c = _defender(3, 900003, "DefC", "Everton")
    d2 = _draws([a, c], tm, ts)
    same = float(np.max(np.abs(d2[1] - d2[3]))) == 0.0
    r = float(np.corrcoef(d2[1], d2[3])[0, 1])
    check(not same, "defenders at different clubs do not score identically",
          f"r = {r:+.3f}")

    # 3. LOCALITY — frame contents and order do not move a club's concessions
    alone = _draws([a], tm, ts)[1]
    crowd = [_noise(10 + k, 800000 + k, f"N{k}", t, p)
             for k, (t, p) in enumerate([("Arsenal", "MID"), ("Hull", "FWD"),
                                         ("Arsenal", "GK"), ("Everton", "DEF"),
                                         ("Man City", "MID")])]
    fwd = _draws(crowd + [a], tm, ts)[1]
    rev = _draws([a] + crowd[::-1], tm, ts)[1]
    check(np.array_equal(alone, fwd) and np.array_equal(alone, rev),
          "a club's concessions ignore who else is in the frame, and order",
          "alone == crowded == reversed")

    # 4. no leakage across clubs: adding an Everton player leaves Arsenal untouched, and
    #    Arsenal's defender still matches his own team-mate rather than anyone else
    d4 = _draws([a, b, c], tm, ts)
    check(np.array_equal(d4[1], d4[2]) and not np.array_equal(d4[1], d4[3]),
          "team-mates stay identical; another club stays distinct", "")

    print("\n" + ("ALL SHARED-CONCESSION PROPERTIES HOLD" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    _sys.exit(main())
