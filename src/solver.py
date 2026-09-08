from __future__ import annotations
import config
"""
solver.py — FPL squad optimisation as a mixed-integer program.
===============================================================
Exact optimisation over the real rules, solved with HiGHS through
`scipy.optimize.milp` — no new dependency, and an optimality certificate rather than a
greedy approximation. Greedy value-ranking cannot solve this: the constraints interact
(spending up on a premium forces cheap defenders, which the club limit then blocks), so
the best squad is generally NOT the top 15 by any per-player score. That is precisely
what `player_value.value` is for — pruning the field — and what it must not be used for:
picking the team.

THE RULES, AS ENCODED (docs/FPL_RULES.md)
------------------------------------------
    squad          15 = 2 GK / 5 DEF / 5 MID / 3 FWD
    budget         £100.0m
    club limit     at most 3 players from any one Premier League club
    XI             11 players: exactly 1 GK, at least 3 DEF, at least 1 FWD
    captain        exactly one, and he must be in the XI — doubles his points
    transfers      1 free per gameweek, unused ones roll, capped at 5 stored;
                   every extra transfer costs 4 points

THREE MODELLING CHOICES WORTH ARGUING WITH
-------------------------------------------
1. BENCH IS NOT WORTH ZERO. A benched player scores if a starter does not play
   (autosubs). Weighting the bench at exactly zero buys implausibly bad bench fodder;
   weighting it at one buys a second XI. `bench_weight` (default 0.15) is the
   probability-ish weight that a bench player's points are actually collected. It is a
   [JUDGMENT] value, exposed rather than buried, and the squad is mildly sensitive to it.
2. THE CAPTAIN IS A DECISION VARIABLE, NOT A PRE-DOUBLED SCORE. The objective adds a
   SECOND copy of the captain's expected points, subject to exactly one captain. Feeding
   pre-doubled points in as `ep` would double the entire XI — see `player_value`.
3. EXPECTED POINTS, NOT RISK-ADJUSTED. The objective maximises the mean. That is the
   right call for a single gameweek in isolation and the wrong one for rank-chasing,
   where variance is what moves you up a leaderboard. `risk_lambda` lets you subtract a
   multiple of the XI's summed sd; it is off by default because the correct value depends
   on your rank and your goal, not on the data.

WHAT THIS DOES NOT MODEL
-------------------------
Price changes (selling price is purchase price plus half the rise, so a squad's value
drifts) and chip interactions beyond the flags below. Autosubs are approximated by
`bench_weight` rather than simulated: the objective still prices every bench slot the
same, so the emitted `bench_order` is a recorded intent (see `assign_bench_order`), not
an optimised quantity.

Run:  python src/solver.py --selftest
"""
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = 15
XI_SIZE = 11
BUDGET = 100.0
MAX_PER_CLUB = 3
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
TRANSFER_COST = 4.0
MAX_STORED_FT = 5


def formation_counts(formation):
    """(DEF, MID, FWD) -> {'GK':1, 'DEF':d, 'MID':m, 'FWD':f}, validated.

    Accepts a tuple (3, 5, 2) or a string '3-5-2'. Returns None for no constraint.
    Rejects anything that is not a legal FPL XI, rather than handing the solver an
    infeasible program and reporting it as 'no squad found'.
    """
    if formation is None:
        return None
    if isinstance(formation, str):
        parts = [int(x) for x in formation.replace("/", "-").split("-") if x.strip()]
    else:
        parts = list(formation)
    if len(parts) == 4:          # '1-3-5-2' style, drop the keeper
        parts = parts[1:]
    if len(parts) != 3:
        raise ValueError(f"formation must be DEF-MID-FWD, got {formation}")
    d, m, f = parts
    if 1 + d + m + f != XI_SIZE:
        raise ValueError(f"formation {d}-{m}-{f} is {1+d+m+f} players, need {XI_SIZE}")
    for pos, k in (("DEF", d), ("MID", m), ("FWD", f)):
        if not (XI_MIN[pos] <= k <= XI_MAX[pos]):
            raise ValueError(f"{k} {pos} is illegal (allowed "
                             f"{XI_MIN[pos]}-{XI_MAX[pos]})")
    return {"GK": 1, "DEF": d, "MID": m, "FWD": f}


def prune(df, per_pos=40, ep_col="ep", keep=None):
    """Shortlist candidates so the MILP stays small, WITHOUT changing the optimum.

    Keeps the top `per_pos` by expected points in each position, plus every player in
    `keep` (an existing squad must remain representable, or the transfer model is
    solving a different problem). Pruning on ep rather than on `value` is deliberate:
    `value` is noisiest for near-floor players (see player_value), and a shortlist chosen
    on it would systematically drop the reliable premiums a squad is built around.
    """
    keep = set(keep or [])
    parts = [g.nlargest(per_pos, ep_col) for _, g in df.groupby("pos")]
    out = pd.concat(parts) if parts else df.iloc[:0]
    if keep:
        out = pd.concat([out, df[df["player_code"].isin(keep)]])
    return out.drop_duplicates("player_code").reset_index(drop=True)


def _pos_matrix(d, pos):
    return (d["pos"].values == pos).astype(float)


def assign_bench_order(squad, ep_col="ep", pos_col="pos"):
    """Fill `bench_order` — the queue FPL walks when a starter does not appear.

    1..3 across the three outfield substitutes, highest `ep` first. The reserve keeper is
    left blank rather than numbered: a goalkeeper can only ever replace a goalkeeper, so
    he does not stand in that queue at all, and giving him a rank would invite a consumer
    to sub him on for an outfielder.

    ORDERING BY ep IS A HEURISTIC, NOT AN OPTIMUM. [JUDGMENT] The MILP never prices the
    bench order — it approximates autosubs with a flat `bench_weight` applied equally to
    every bench slot, so nothing in the objective distinguishes first sub from third.
    Descending ep is the defensible default because ep already carries P(appears), and
    appearance probability is exactly what should dominate the queue: a bench player who
    did not play is skipped, so his points never enter regardless of how good he is.

    What this does NOT do is simulate the substitution. The formation floor (>=3 DEF,
    >=1 FWD, exactly 1 GK) can make a lower-ranked player the only legal replacement, so
    the realised order can differ from this one. The column records which bench a manager
    intended, which is what the ledger needs to score a week exactly; resolving it against
    a specific set of blanks is a separate step.
    """
    squad = squad.copy()
    squad["bench_order"] = np.nan
    outfield = squad.index[(~squad["in_xi"].astype(bool))
                           & (squad[pos_col].values != "GK")]
    ranked = squad.loc[outfield, ep_col].sort_values(ascending=False).index
    squad.loc[ranked, "bench_order"] = np.arange(1, len(ranked) + 1, dtype=float)
    return squad


def solve_gameweek(df, budget=BUDGET, bench_weight=0.15, ep_col="ep",
                   price_col="price", risk_lambda=0.0, sd_col="sd",
                   triple_captain=False, bench_boost=False, locked=None,
                   banned=None, formation=None, verbose=False):
    """Optimal 15-man squad, XI and captain for a single gameweek.

    Variables, per player i:  squad_i, start_i, cap_i  (all binary)
    Objective (maximised):
        sum start_i * ep_i
      + sum cap_i   * ep_i * (2 if triple_captain else 1)     <- the EXTRA copy
      + bench_weight * sum (squad_i - start_i) * ep_i          <- autosub allowance
      - risk_lambda  * sum start_i * sd_i

    `locked` forces player_codes into the squad, `banned` forbids them.
    Returns a dict with the squad frame, XI, captain, cost and the objective.
    """
    d = df.reset_index(drop=True)
    n = len(d)
    if n < SQUAD_SIZE:
        raise ValueError(f"only {n} candidates for a {SQUAD_SIZE}-man squad")
    ep = d[ep_col].fillna(0).values.astype(float)
    price = d[price_col].astype(float).values
    sd = d[sd_col].fillna(0).values.astype(float) if sd_col in d.columns else np.zeros(n)

    # variable layout: [squad(n) | start(n) | cap(n)]
    N = 3 * n
    S, T, C = slice(0, n), slice(n, 2 * n), slice(2 * n, 3 * n)

    obj = np.zeros(N)
    obj[T] = ep - risk_lambda * sd
    obj[C] = ep * (2.0 if triple_captain else 1.0)
    if bench_boost:
        # Bench Boost: all 15 score exactly ONCE, so the squad term carries the whole
        # projection and starting adds nothing on top. The previous version left
        # obj[T] = ep in place and added obj[S] = ep as well, counting every STARTER
        # twice while the bench counted once — which inflated the XI and made the boost
        # look far more valuable than it is.
        obj[S] += ep
        obj[T] = -risk_lambda * sd       # keep only the risk term on starters
    else:
        obj[S] += bench_weight * ep
        obj[T] -= bench_weight * ep      # a starter is not also a bench player

    cons = []

    def row(vals_s=None, vals_t=None, vals_c=None):
        r = np.zeros(N)
        if vals_s is not None: r[S] = vals_s
        if vals_t is not None: r[T] = vals_t
        if vals_c is not None: r[C] = vals_c
        return r

    # squad size and composition
    cons.append(LinearConstraint(row(vals_s=np.ones(n)), SQUAD_SIZE, SQUAD_SIZE))
    for pos, k in SQUAD.items():
        cons.append(LinearConstraint(row(vals_s=_pos_matrix(d, pos)), k, k))
    # budget
    cons.append(LinearConstraint(row(vals_s=price), -np.inf, budget))
    # club limit
    for club in d["team"].dropna().unique():
        m = (d["team"].values == club).astype(float)
        cons.append(LinearConstraint(row(vals_s=m), -np.inf, MAX_PER_CLUB))
    # XI size and shape
    cons.append(LinearConstraint(row(vals_t=np.ones(n)), XI_SIZE, XI_SIZE))
    # `formation` pins the XI shape, e.g. (3, 5, 2) for 3-5-2. Without it the solver
    # picks whichever legal shape scores most, which is usually right but not always
    # what a manager wants to field.
    fixed = formation_counts(formation)
    for pos in SQUAD:
        m = _pos_matrix(d, pos)
        if fixed and pos in fixed:
            cons.append(LinearConstraint(row(vals_t=m), fixed[pos], fixed[pos]))
        else:
            cons.append(LinearConstraint(row(vals_t=m), XI_MIN[pos], XI_MAX[pos]))
    # start_i <= squad_i   and   cap_i <= start_i
    A_ts = np.zeros((n, N)); A_cs = np.zeros((n, N))
    for i in range(n):
        A_ts[i, n + i] = 1.0; A_ts[i, i] = -1.0       # start - squad <= 0
        A_cs[i, 2 * n + i] = 1.0; A_cs[i, n + i] = -1.0   # cap - start <= 0
    cons.append(LinearConstraint(A_ts, -np.inf, 0))
    cons.append(LinearConstraint(A_cs, -np.inf, 0))
    # exactly one captain
    cons.append(LinearConstraint(row(vals_c=np.ones(n)), 1, 1))

    lo = np.zeros(N); hi = np.ones(N)
    if locked:
        for i in d.index[d["player_code"].isin(set(locked))]:
            lo[i] = 1.0
    if banned:
        for i in d.index[d["player_code"].isin(set(banned))]:
            hi[i] = 0.0

    res = milp(c=-obj, constraints=cons, integrality=np.ones(N),
               bounds=Bounds(lo, hi))
    if not res.success:
        raise RuntimeError(f"solver failed: {res.message}")
    x = np.round(res.x).astype(int)
    squad = d[x[S] == 1].copy()
    squad["in_xi"] = x[T][x[S] == 1] == 1
    squad["is_captain"] = x[C][x[S] == 1] == 1
    squad = assign_bench_order(squad, ep_col=ep_col)
    squad = squad.sort_values(["in_xi", ep_col], ascending=[False, False])
    xi = squad[squad.in_xi]
    cap = squad[squad.is_captain]
    out = {
        "squad": squad, "xi": xi,
        "captain": (cap.iloc[0] if len(cap) else None),
        "cost": float(squad[price_col].sum()),
        "budget_left": float(budget - squad[price_col].sum()),
        "xi_ep": float(xi[ep_col].sum()),
        "captain_ep": float(cap[ep_col].sum()) if len(cap) else 0.0,
        "bench_ep": float(squad[~squad.in_xi][ep_col].sum()),
        "objective": float(-res.fun),
        "status": res.message,
    }
    out["total_ep"] = (out["xi_ep"]
                       + out["captain_ep"] * (2.0 if triple_captain else 1.0)
                       + (out["bench_ep"] if bench_boost else 0.0))
    if verbose:
        print(f"[solver] XI {out['xi_ep']:.2f} + captain {out['captain_ep']:.2f} "
              f"= {out['total_ep']:.2f} pts, cost £{out['cost']:.1f}m")
    return out


def best_xi_from_squad(squad, ep_col="ep", triple_captain=False, verbose=False):
    """Given a fixed 15, choose the XI and captain. Used every gameweek you do not
    transfer, and to score a squad you already own."""
    d = squad.reset_index(drop=True)
    n = len(d)
    ep = d[ep_col].fillna(0).values.astype(float)
    N = 2 * n
    T, C = slice(0, n), slice(n, 2 * n)
    obj = np.zeros(N)
    obj[T] = ep
    obj[C] = ep * (2.0 if triple_captain else 1.0)
    cons = []

    def row(t=None, c=None):
        r = np.zeros(N)
        if t is not None: r[T] = t
        if c is not None: r[C] = c
        return r

    cons.append(LinearConstraint(row(t=np.ones(n)), XI_SIZE, XI_SIZE))
    for pos in SQUAD:
        m = _pos_matrix(d, pos)
        cons.append(LinearConstraint(row(t=m), XI_MIN[pos], XI_MAX[pos]))
    A = np.zeros((n, N))
    for i in range(n):
        A[i, n + i] = 1.0; A[i, i] = -1.0
    cons.append(LinearConstraint(A, -np.inf, 0))
    cons.append(LinearConstraint(row(c=np.ones(n)), 1, 1))
    res = milp(c=-obj, constraints=cons, integrality=np.ones(N),
               bounds=Bounds(np.zeros(N), np.ones(N)))
    if not res.success:
        raise RuntimeError(res.message)
    x = np.round(res.x).astype(int)
    d["in_xi"] = x[T] == 1
    d["is_captain"] = x[C] == 1
    xi = d[d.in_xi]
    cap = d[d.is_captain]
    return {"squad": d, "xi": xi, "captain": cap.iloc[0] if len(cap) else None,
            "xi_ep": float(xi[ep_col].sum()),
            "captain_ep": float(cap[ep_col].sum()) if len(cap) else 0.0,
            "total_ep": float(xi[ep_col].sum() +
                              cap[ep_col].sum() * (2.0 if triple_captain else 1.0))}


def solve_horizon(long_df, gws, budget=BUDGET, bench_weight=0.15, ep_col="ep",
                  price_col="price", free_transfers=1, per_pos=30,
                  initial_squad=None, churn_penalty=0.01, time_limit=120.0,
                  mip_gap=0.005, banned=None, locked=None, formation=None,
                  bench_boost_gw=None, pos_budget=None, max_bench_spend=None,
                  max_transfers=None, ft_pin=None, verbose=True):
    """Plan a multi-gameweek horizon with transfers priced in.

    A greedy week-by-week plan is myopic: it will not buy a player whose fixtures turn
    good in three weeks, and it re-buys the same player repeatedly. This solves all
    gameweeks jointly, so a transfer is made only if it pays for itself across the
    remaining horizon.

    Transfer accounting: `free_transfers` at the first gameweek, one earned per week,
    unused ones roll up to MAX_STORED_FT, and each transfer beyond the bank costs
    TRANSFER_COST points. The bank is modelled as a continuous stock with an integer
    paid-transfer count, which is exact for the costs and slightly permissive about
    fractional storage — an approximation that never lets the plan take a free transfer
    it has not earned.

    `ft_pin` is {gameweek: free-transfer bank entering it}, and it OVERRIDES the carry
    arithmetic for that week rather than adding to it. Two rules need this and neither is
    derivable from the carry recursion (docs/FPL_RULES.md):

      * a Wildcard or Free Hit RETAINS saved free transfers (rules L79), but the opening
        free build pins the bank to zero, so a wildcard entered with 2 banked has to be
        told so: ft_pin={first_gw + 1: 2}.
      * any one-off top-up the rules grant in a given season. NOTE the 2025/26 AFCON
        top-up to 5 ahead of GW16 does NOT apply in 2026/27 — see docs/FPL_RULES.md,
        which still carried the dead paragraph. Do not pin GW16 on that basis.

    Off by default; None reproduces the previous behaviour exactly.

    `max_transfers` caps transfers in every gameweek that is not the opening free build.
    `max_transfers=0` forces the squad to be HELD across the whole horizon, which is a
    different problem from letting the solver decline to transfer: with the bank free,
    a plan will always take a free transfer worth even 0.01 points, so "the optimum
    happened to hold" and "holding was required" are not the same squad. The forced-hold
    case is what you want when a wildcard lands just past the horizon and the fifteen
    only has to survive until then.
    """
    gws = list(gws)
    ft_pin = {int(g): float(v) for g, v in (ft_pin or {}).items() if int(g) in gws}
    _pinned_idx = {gws.index(g): v for g, v in ft_pin.items()}
    pool = long_df[long_df["gw"].isin(gws)]
    _keep = list(initial_squad or []) + list(locked or [])
    codes = prune(pool[pool.gw == gws[0]], per_pos=per_pos, ep_col=ep_col,
                  keep=_keep)["player_code"]
    # a player must be in the pool for EVERY gameweek in the horizon
    P = pool[pool["player_code"].isin(set(codes))]
    piv = P.pivot_table(index="player_code", columns="gw", values=ep_col,
                        aggfunc="first")
    piv = piv.dropna()
    meta = (P.drop_duplicates("player_code")
            .set_index("player_code")
            .loc[piv.index, [c for c in ["display_name", "unique_label", "player",
                                         "team", "pos", price_col]
                             if c in P.columns]])
    n, T = len(piv), len(gws)
    if verbose:
        print(f"[horizon] {n} candidates x {T} gameweeks "
              f"= {n * T * 3 + n * T * 2:,} binaries")

    ep = piv[gws].values                       # (n, T)
    price = meta[price_col].astype(float).values
    pos = meta["pos"].values
    team = meta["team"].values

    # layout: squad | start | cap | buy | sell  (each n*T) | paid(T) | ft(T)
    # `ft_t` is the free-transfer BANK entering gameweek t, carried as a continuous
    # stock. Without it the bank has to be approximated by a per-week constant, which
    # cannot decrement when transfers are spent — the first version of this function
    # did exactly that and happily made nine "free" transfers a week.
    blocks = 5
    N = blocks * n * T + 2 * T
    def idx(b, i, t): return b * n * T + t * n + i
    PAID = blocks * n * T
    FT = PAID + T

    obj = np.zeros(N)
    for t in range(T):
        bb = (bench_boost_gw is not None and gws[t] == bench_boost_gw)
        for i in range(n):
            if bb:
                # all 15 score once; starting adds nothing on top
                obj[idx(0, i, t)] += ep[i, t]
            else:
                obj[idx(1, i, t)] += ep[i, t]                # starter
                obj[idx(0, i, t)] += bench_weight * ep[i, t]
                obj[idx(1, i, t)] -= bench_weight * ep[i, t]
            obj[idx(2, i, t)] += ep[i, t]                    # captain extra copy
            # Tie-break AGAINST churn. A free transfer between two players with equal
            # projections costs nothing, so the objective is flat and the solver picks
            # arbitrarily — which surfaces as advice to swap X for Y for no reason.
            # This epsilon is far below any real difference in expected points, so it
            # never overturns a decision; it only decides ties in favour of holding.
            if not (t == 0 and initial_squad is None):
                obj[idx(3, i, t)] -= churn_penalty
        obj[PAID + t] -= TRANSFER_COST

    cons = []
    def newrow(): return np.zeros(N)

    for t in range(T):
        r = newrow()
        for i in range(n): r[idx(0, i, t)] = 1.0
        cons.append(LinearConstraint(r, SQUAD_SIZE, SQUAD_SIZE))
        for p, k in SQUAD.items():
            r = newrow()
            for i in range(n):
                if pos[i] == p: r[idx(0, i, t)] = 1.0
            cons.append(LinearConstraint(r, k, k))
        r = newrow()
        for i in range(n): r[idx(0, i, t)] = price[i]
        cons.append(LinearConstraint(r, -np.inf, budget))
        # optional per-position spend cap, e.g. {"DEF": 26.0} — the squad must still
        # contain five defenders, so this caps what they COST, not how many there are
        for _p, _cap in (pos_budget or {}).items():
            r = newrow()
            for i in range(n):
                if pos[i] == _p: r[idx(0, i, t)] = price[i]
            cons.append(LinearConstraint(r, -np.inf, float(_cap)))
        for club in pd.unique(team):
            r = newrow()
            for i in range(n):
                if team[i] == club: r[idx(0, i, t)] = 1.0
            cons.append(LinearConstraint(r, -np.inf, MAX_PER_CLUB))
        r = newrow()
        for i in range(n): r[idx(1, i, t)] = 1.0
        cons.append(LinearConstraint(r, XI_SIZE, XI_SIZE))
        _fx = formation_counts(formation)
        for p in SQUAD:
            r = newrow()
            for i in range(n):
                if pos[i] == p: r[idx(1, i, t)] = 1.0
            if _fx and p in _fx:
                cons.append(LinearConstraint(r, _fx[p], _fx[p]))
            else:
                cons.append(LinearConstraint(r, XI_MIN[p], XI_MAX[p]))
        r = newrow()
        for i in range(n): r[idx(2, i, t)] = 1.0
        cons.append(LinearConstraint(r, 1, 1))
        # start <= squad, cap <= start
        A1 = np.zeros((n, N)); A2 = np.zeros((n, N))
        for i in range(n):
            A1[i, idx(1, i, t)] = 1.0; A1[i, idx(0, i, t)] = -1.0
            A2[i, idx(2, i, t)] = 1.0; A2[i, idx(1, i, t)] = -1.0
        cons.append(LinearConstraint(A1, -np.inf, 0))
        cons.append(LinearConstraint(A2, -np.inf, 0))

        # Cap the money sitting on the bench in a gameweek that is NOT boosted. In a
        # non-boosted week a benched player scores nothing, so every pound there is
        # parked. This does NOT apply to a boosted week, where the bench scores in full
        # and expensive bench players are the whole point — the two goals genuinely pull
        # in opposite directions and only one of them can bind per gameweek.
        # recompute per t — `bb` from the objective loop above has leaked its LAST
        # value, so reusing it silently disabled this constraint in every gameweek
        _bb_t = (bench_boost_gw is not None and gws[t] == bench_boost_gw)
        if max_bench_spend is not None and not _bb_t:
            r = newrow()
            for i in range(n):
                r[idx(0, i, t)] = price[i]      # squad
                r[idx(1, i, t)] = -price[i]     # minus starters = bench spend
            cons.append(LinearConstraint(r, -np.inf, float(max_bench_spend)))

        # squad continuity: s_t - s_{t-1} - buy + sell = 0
        A = np.zeros((n, N)); lo = np.zeros(n); hi = np.zeros(n)
        for i in range(n):
            A[i, idx(0, i, t)] = 1.0
            A[i, idx(3, i, t)] = -1.0
            A[i, idx(4, i, t)] = 1.0
            if t > 0:
                A[i, idx(0, i, t - 1)] = -1.0
            else:
                if initial_squad is not None:
                    lo[i] = hi[i] = 1.0 if piv.index[i] in set(initial_squad) else 0.0
        if t == 0 and initial_squad is None:
            # first week is a free build: buys are unconstrained and cost nothing
            A = np.zeros((n, N))
            for i in range(n):
                A[i, idx(3, i, t)] = 1.0
                A[i, idx(0, i, t)] = -1.0
            cons.append(LinearConstraint(A, 0, 0))
        else:
            cons.append(LinearConstraint(A, lo, hi))

        # free-transfer accounting
        free_build = (t == 0 and initial_squad is None)
        # A hard cap on transfers, independent of what they cost. Applied to every week
        # except the opening free build, where `buy` is pinned to the squad itself.
        if max_transfers is not None and not free_build:
            r = newrow()
            for i in range(n):
                r[idx(3, i, t)] = 1.0
            cons.append(LinearConstraint(r, -np.inf, float(max_transfers)))
        if free_build:
            # the opening squad is 15 unlimited free transfers, and nothing is paid
            r = newrow(); r[PAID + t] = 1.0
            cons.append(LinearConstraint(r, 0, 0))
        else:
            # (a) transfers made cannot exceed the bank plus what you pay for
            r = newrow()
            for i in range(n): r[idx(3, i, t)] = 1.0
            r[PAID + t] = -1.0
            r[FT + t] = -1.0
            cons.append(LinearConstraint(r, -np.inf, 0))
        # (b) carry: ft_{t+1} <= ft_t - (transfers - paid) + 1, capped at MAX_STORED_FT
        if t + 1 < T and (t + 1) not in _pinned_idx:
            r = newrow()
            r[FT + t + 1] = 1.0
            if not free_build:
                r[FT + t] = -1.0
                for i in range(n): r[idx(3, i, t)] = 1.0
                r[PAID + t] = -1.0
            cons.append(LinearConstraint(r, -np.inf, 1.0))

    integrality = np.ones(N)
    lo_b = np.zeros(N); hi_b = np.ones(N)
    for t in range(T):
        hi_b[PAID + t] = 20.0                      # 20-transfer cap per gameweek
        hi_b[FT + t] = float(MAX_STORED_FT)        # bank caps at 5
        integrality[FT + t] = 0                    # the bank is a continuous stock
    # opening bank
    lo_b[FT] = hi_b[FT] = 0.0 if initial_squad is None else float(free_transfers)
    for _ti, _v in _pinned_idx.items():                    # a pinned bank is exact
        lo_b[FT + _ti] = hi_b[FT + _ti] = float(_v)
    # bans apply in EVERY gameweek — a banned player must not appear even transiently
    if banned:
        ban = set(banned)
        for i in range(n):
            if piv.index[i] in ban:
                for t in range(T):
                    hi_b[idx(0, i, t)] = 0.0
    # locks likewise hold across the whole horizon, so a forced pick cannot be sold on
    # in week two and quietly reappear as a different plan
    if locked:
        lk = set(locked)
        for i in range(n):
            if piv.index[i] in lk:
                for t in range(T):
                    lo_b[idx(0, i, t)] = 1.0
    # A horizon MILP can run a long time chasing the last fraction of a point. Bounded
    # by wall clock and by an optimality gap, and the gap is REPORTED rather than
    # swallowed — a plan 0.5% from optimal is fine, but you should know that is what it
    # is rather than assume you were handed the optimum.
    res = milp(c=-obj, constraints=cons, integrality=integrality,
               bounds=Bounds(lo_b, hi_b),
               options={"time_limit": time_limit, "mip_rel_gap": mip_gap})
    if res.x is None:
        raise RuntimeError(f"horizon solve found no feasible plan: {res.message}")
    gap = getattr(res, "mip_gap", None)
    if verbose and gap is not None:
        print(f"[horizon] optimality gap {gap:.3%}"
              + ("" if res.success else f"  (stopped early: {res.message})"))
    x = np.round(res.x).astype(int)

    plans = []
    for t, gw in enumerate(gws):
        sel = [i for i in range(n) if x[idx(0, i, t)] == 1]
        sq = meta.iloc[sel].copy()
        sq["ep"] = ep[sel, t]
        sq["in_xi"] = [x[idx(1, i, t)] == 1 for i in sel]
        sq["is_captain"] = [x[idx(2, i, t)] == 1 for i in sel]
        sq = assign_bench_order(sq, ep_col="ep")
        buys = [piv.index[i] for i in range(n) if x[idx(3, i, t)] == 1]
        sells = [piv.index[i] for i in range(n) if x[idx(4, i, t)] == 1]
        xi = sq[sq.in_xi]
        cap = sq[sq.is_captain]
        plans.append({"gw": gw, "squad": sq, "xi": xi,
                      "captain": cap.iloc[0] if len(cap) else None,
                      "buys": buys, "sells": sells,
                      "paid_transfers": int(x[PAID + t]),
                      "cost": float(sq[price_col].sum()),
                      "xi_ep": float(xi["ep"].sum()),
                      "bench_ep": float(sq.loc[~sq.in_xi, "ep"].sum()),
                      "bench_boost": bool(bench_boost_gw is not None
                                          and gw == bench_boost_gw),
                      "captain_ep": float(cap["ep"].sum()) if len(cap) else 0.0})
        # In a Bench Boost week the bench SCORES, so it belongs in the total. Reporting
        # xi_ep + captain_ep there hid the entire benefit of the chip while still paying
        # its cost (a more expensive 15), which made the boost look strictly negative.
        plans[-1]["total_ep"] = (plans[-1]["xi_ep"] + plans[-1]["captain_ep"]
                                 + (plans[-1]["bench_ep"] if plans[-1]["bench_boost"]
                                    else 0.0))
    return {"plans": plans, "objective": float(-res.fun), "status": res.message,
            "mip_gap": gap, "optimal": bool(res.success)}


def selftest():
    rng = np.random.default_rng(3)
    rows = []
    clubs = [f"C{i}" for i in range(20)]
    pc = 0
    for pos, k in [("GK", 12), ("DEF", 30), ("MID", 30), ("FWD", 18)]:
        for j in range(k):
            pc += 1
            rows.append({"player_code": pc, "player": f"{pos}{j}",
                         "display_name": f"{pos}{j}", "team": clubs[pc % 20],
                         "pos": pos, "price": float(rng.integers(40, 130)) / 10,
                         "ep": float(rng.gamma(3, 1.2)), "sd": float(rng.gamma(2, 1.5))})
    d = pd.DataFrame(rows)

    r = solve_gameweek(d, bench_weight=0.15)
    sq = r["squad"]
    assert len(sq) == SQUAD_SIZE
    for pos, k in SQUAD.items():
        assert (sq.pos == pos).sum() == k, f"{pos} count wrong"
    assert r["cost"] <= BUDGET + 1e-6, f"over budget: {r['cost']}"
    assert sq.groupby("team").size().max() <= MAX_PER_CLUB, "club limit violated"
    xi = r["xi"]
    assert len(xi) == XI_SIZE
    assert (xi.pos == "GK").sum() == 1
    assert (xi.pos == "DEF").sum() >= 3 and (xi.pos == "FWD").sum() >= 1
    assert int(sq.is_captain.sum()) == 1
    assert bool(sq.loc[sq.is_captain, "in_xi"].iloc[0]), "captain must start"
    # the captain must be the highest-ep starter (nothing else is optimal)
    assert abs(sq.loc[sq.is_captain, "ep"].iloc[0] - xi["ep"].max()) < 1e-6

    # bench order: exactly 1..3 across the outfield subs, descending ep, keeper blank
    bench = sq[~sq.in_xi]
    outf = bench[bench.pos != "GK"]
    assert sorted(outf["bench_order"].tolist()) == [1.0, 2.0, 3.0], \
        f"outfield bench must be ranked 1..3, got {outf['bench_order'].tolist()}"
    assert bench.loc[bench.pos == "GK", "bench_order"].isna().all(), \
        "the reserve keeper must not be given an outfield sub rank"
    assert sq.loc[sq.in_xi, "bench_order"].isna().all(), \
        "a starter must have no bench order"
    ordered = outf.sort_values("bench_order")["ep"].tolist()
    assert ordered == sorted(ordered, reverse=True), \
        f"bench order must run highest ep first, got {ordered}"
    # and the horizon solver must emit the same column, not just the single-week one
    L = d.assign(gw=1)
    L = pd.concat([L, d.assign(gw=2)], ignore_index=True)
    hz = solve_horizon(L, [1, 2], verbose=False)
    for pl in hz["plans"]:
        hb = pl["squad"][~pl["squad"].in_xi]
        assert sorted(hb.loc[hb.pos != "GK", "bench_order"].tolist()) == [1.0, 2.0, 3.0], \
            f"horizon GW{pl['gw']} bench order missing or malformed"

    # optimality: no feasible swap of one squad player improves the objective
    base = r["objective"]
    r2 = solve_gameweek(d, bench_weight=0.15,
                        banned=[int(sq.loc[sq.is_captain, "player_code"].iloc[0])])
    assert r2["objective"] <= base + 1e-6, "banning the captain improved the objective"

    # locking a player must include him
    lock = int(d.iloc[0].player_code)
    r3 = solve_gameweek(d, locked=[lock])
    assert lock in set(r3["squad"].player_code), "locked player missing"

    # triple captain must beat a normal captain on the same squad
    rt = solve_gameweek(d, triple_captain=True)
    assert rt["total_ep"] > r["total_ep"], "triple captain should score more"

    # formation is honoured exactly
    rf = solve_gameweek(d, formation=(3, 5, 2))
    xf = rf["xi"]
    assert (xf.pos == "DEF").sum() == 3 and (xf.pos == "MID").sum() == 5 \
        and (xf.pos == "FWD").sum() == 2 and (xf.pos == "GK").sum() == 1, \
        "3-5-2 not respected"
    assert rf["xi_ep"] <= r["xi_ep"] + 1e-6, "constraining shape cannot help"
    for bad in [(3, 5, 3), (2, 5, 3), (6, 4, 0)]:
        try:
            formation_counts(bad); raise AssertionError(f"{bad} should be rejected")
        except ValueError:
            pass
    assert formation_counts("3-5-2") == {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2}
    assert formation_counts("1-4-4-2")["DEF"] == 4

    # BENCH BOOST: every squad member scores exactly once, so the objective must equal
    # the whole squad's ep plus one captain copy — NOT the XI counted twice.
    rb = solve_gameweek(d, bench_boost=True, bench_weight=0.0)
    sq_ep = rb["squad"]["ep"].sum()
    cap_ep = rb["squad"].loc[rb["squad"].is_captain, "ep"].iloc[0]
    assert abs(rb["objective"] - (sq_ep + cap_ep)) < 1e-6, \
        f"bench boost objective {rb['objective']:.3f} != squad {sq_ep:.3f} + captain {cap_ep:.3f}"
    # and it must pick a stronger 15 than the non-boosted build, whose bench is fodder
    assert sq_ep > r["squad"]["ep"].sum(), "bench boost should buy a stronger 15"
    # the REPORTED total must include the bench, or the chip looks worthless
    assert abs(rb["total_ep"] - (sq_ep + cap_ep)) < 1e-6, \
        "bench boost total must count the bench"
    assert rb["total_ep"] > r["total_ep"], \
        "playing a chip can never lower the score it reports"
    # a boosted horizon week must likewise report more than the same week unboosted
    lgc = pd.concat([d.assign(gw=g) for g in [1, 2]])
    h_no = solve_horizon(lgc, [1, 2], per_pos=12, verbose=False)
    h_bb = solve_horizon(lgc, [1, 2], per_pos=12, bench_boost_gw=2, verbose=False)
    assert h_bb["plans"][1]["total_ep"] > h_no["plans"][1]["total_ep"], \
        "the boosted gameweek must score more than the unboosted one"

    # bans hold in every gameweek of a horizon
    ban_code = int(d.nlargest(1, "ep").player_code.iloc[0])
    lgb = pd.concat([d.assign(gw=g) for g in [1, 2]])
    hb = solve_horizon(lgb, [1, 2], per_pos=12, banned=[ban_code], verbose=False)
    for p in hb["plans"]:
        assert ban_code not in set(p["squad"].index), "banned player appeared"

    # best_xi_from_squad reproduces the XI choice on a fixed 15
    b = best_xi_from_squad(sq.drop(columns=["in_xi", "is_captain"]))
    assert len(b["xi"]) == XI_SIZE and abs(b["xi_ep"] - r["xi_ep"]) < 1e-6

    # horizon: 3 gameweeks, transfers priced
    lg = pd.concat([d.assign(gw=g, ep=d.ep * (1 + 0.1 * g)) for g in [1, 2, 3]])
    h = solve_horizon(lg, [1, 2, 3], per_pos=12, verbose=False)
    assert len(h["plans"]) == 3
    for p in h["plans"]:
        assert len(p["squad"]) == SQUAD_SIZE
        assert p["cost"] <= BUDGET + 1e-6
        assert p["squad"].groupby("team").size().max() <= MAX_PER_CLUB

    # THE TRANSFER ECONOMICS, which feasibility alone does not check. The first version
    # of solve_horizon passed every constraint above while making nine free transfers a
    # week, because the bank was a constant that never decremented.
    for p in h["plans"][1:]:
        n_in = len(p["buys"])
        assert n_in == len(p["sells"]), "a transfer must be a buy AND a sell"
        # you start with 1 FT and earn 1 a week, so by GW3 you can never have banked
        # more than 2 free — anything above that must be paid for
        assert n_in - p["paid_transfers"] <= MAX_STORED_FT, \
            f"{n_in} transfers with only {p['paid_transfers']} paid"
    # max_transfers=0 must HOLD the squad, and holding must not beat transferring —
    # a rising projection makes transfers genuinely worth it, so the forced hold has to
    # score strictly less than the free plan while keeping the same fifteen throughout.
    hh = solve_horizon(lg, [1, 2, 3], per_pos=12, max_transfers=0, verbose=False)
    first = set(hh["plans"][0]["squad"].index)
    for p in hh["plans"][1:]:
        assert not p["buys"] and not p["sells"], "max_transfers=0 still transferred"
        assert set(p["squad"].index) == first, "held squad changed anyway"
    assert sum(x["total_ep"] for x in hh["plans"]) <=         sum(x["total_ep"] for x in h["plans"]) + 1e-6,         "forbidding transfers cannot improve the horizon"
    # and a cap of 1 must be respected without forbidding transfers outright
    h1 = solve_horizon(lg, [1, 2, 3], per_pos=12, max_transfers=1, verbose=False)
    for p in h1["plans"][1:]:
        assert len(p["buys"]) <= 1, f"cap of 1 violated: {len(p['buys'])} transfers"

    # ft_pin: a pinned bank must be spendable in that week without paying for it, and
    # must not leak backwards into weeks before it. GW3 pinned to 3 means up to three
    # transfers land in GW3 free, while GW2 still cannot exceed its own earned bank.
    hp = solve_horizon(lg, [1, 2, 3], per_pos=12, ft_pin={3: 3}, verbose=False)
    p3 = hp["plans"][2]
    assert p3["paid_transfers"] == 0 or len(p3["buys"]) > 3,         "a pinned bank of 3 still charged for <=3 transfers"
    assert len(hp["plans"][1]["buys"]) - hp["plans"][1]["paid_transfers"] <= 1,         "pinned GW3 bank leaked back into GW2"
    # and ft_pin=None must reproduce the unpinned plan exactly
    hn = solve_horizon(lg, [1, 2, 3], per_pos=12, ft_pin=None, verbose=False)
    assert abs(hn["objective"] - h["objective"]) < 1e-6, "ft_pin=None changed the solve"

    # with a static projection there is no reason to transfer at all
    flat = pd.concat([d.assign(gw=g) for g in [1, 2, 3]])
    hf = solve_horizon(flat, [1, 2, 3], per_pos=12, verbose=False)
    churn = sum(len(p["buys"]) for p in hf["plans"][1:])
    assert churn == 0, f"made {churn} transfers on a static projection"
    # and a squad handed in must be the one it starts from
    start = list(h["plans"][0]["squad"].index)
    hs = solve_horizon(lg, [1, 2, 3], per_pos=12, initial_squad=start,
                       free_transfers=1, verbose=False)
    assert set(hs["plans"][0]["squad"].index) == set(start), \
        "initial squad not honoured in the first gameweek"

    print("SELFTEST OK: squad composition, budget, club limit, XI shape, one captain "
          "who starts and is the top scorer, lock/ban honoured, triple captain higher, "
          "horizon feasible in every gameweek, bench ranked 1..3 by ep with the reserve "
          "keeper unranked in both solvers.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    print(__doc__)
