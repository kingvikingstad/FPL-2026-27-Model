from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
def_rotation.py — best 5-defender squad to rotate, playing 3 a week
===================================================================
Same idea as gk_rotation but a far harder problem. You own FIVE defenders and start
THREE, so a squad is worth

    sum over gameweeks of  (the best three of the five that week)

which is neither separable nor linear, so it cannot be solved by picking the five best
players or by a linear program. It rewards squads whose good weeks are spread out: three
defenders who are all strong in the same gameweeks waste each other, while a fourth and
fifth who cover the weeks the top three are away are worth real points.

CONSTRAINTS
  * total cost of the five <= budget (10 budgets from 22.0 to 35.0)
  * at most 3 from any one club (FPL's squad rule, binding here)
  * only genuine starters are eligible — a squad that looks good on a backup's projection
    is a fiction, so anyone below a 50% start probability is dropped

WHY PRUNE, AND HOW IT IS CHECKED
---------------------------------
There are 187 defenders on the board, and C(187,5) is 2.1 billion — not enumerable. The
pool is cut to the best few at each price point (per horizon, since a squad optimised for
GW1-2 differs from one for GW1-10), which makes exhaustive enumeration cheap.

Pruning can in principle drop a player who would have made the optimum through fixture
complementarity rather than raw total. So every enumerated answer is then HILL-CLIMBED
against the FULL 187-player pool with single-player swaps. If the climb improves on the
enumerated squad, the pruning was too tight and the improvement is reported and used.
That converts a heuristic into a checked one.

Run:  python studies/def_rotation.py
"""
import warnings; warnings.filterwarnings("ignore")
from itertools import combinations
import numpy as np, pandas as pd

HORIZONS = (2, 3, 6, 10)
BUDGETS = (22, 23, 24, 25, 26, 27, 28, 29, 30, 35)
SQUAD, PLAY = 5, 3
MAX_PER_CLUB = 3
MIN_START = 0.50
TOP_PER_PRICE = 8
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "def_rotation.csv")


def load():
    d = pd.read_csv(_os.path.join(config.OUTPUTS, "gw_board_long.csv"))
    g = d[d["pos"] == "DEF"].copy()
    wide = g.pivot_table(index=["player", "team", "cost"], columns="gw",
                         values="mean", aggfunc="first").fillna(0.0).reset_index()
    pri = pd.read_pickle(config.MS_PRIORS)
    pri["p_start"] = pri["start_a"] / (pri["start_a"] + pri["start_b"])
    names = pd.read_csv(_os.path.join(config.repo("2026-2027"), "players.csv"))
    teams = pd.read_csv(_os.path.join(config.repo("2026-2027"), "teams.csv"))
    names["team"] = names["team_code"].map(dict(zip(teams["code"], teams["name"]))).replace(
        {"Man Utd": "Man United", "Spurs": "Tottenham"})
    m = (names[["player_code", "web_name", "team"]]
         .merge(pri[["player_code", "p_start"]], on="player_code", how="inner")
         .rename(columns={"web_name": "player"})
         .drop_duplicates(["player", "team"]))
    before = len(wide)
    wide = wide.merge(m[["player", "team", "p_start"]], on=["player", "team"], how="left")
    assert len(wide) == before, "p_start merge fanned out"
    wide["p_start"] = wide["p_start"].fillna(0.0)
    return wide


def value(P, idx, gws):
    """Sum over gameweeks of the best PLAY of the chosen SQUAD. idx: (B, SQUAD)."""
    sub = P[idx][:, :, :gws]                      # (B, SQUAD, gws)
    part = np.partition(sub, SQUAD - PLAY, axis=1)[:, SQUAD - PLAY:, :]
    return part.sum(axis=(1, 2))


def enumerate_pool(pool, P, gws, budgets):
    n = len(pool)
    cost = pool["cost"].values
    club = pd.factorize(pool["team"])[0]
    combos = np.array(list(combinations(range(n), SQUAD)), dtype=np.int16)
    csum = cost[combos].sum(axis=1)
    # at most MAX_PER_CLUB from one club
    cl = club[combos]
    cl.sort(axis=1)
    runs = np.ones(len(cl), dtype=np.int8)
    ok = np.ones(len(cl), dtype=bool)
    for k in range(SQUAD - MAX_PER_CLUB):
        ok &= ~(cl[:, k] == cl[:, k + MAX_PER_CLUB])
    combos, csum = combos[ok], csum[ok]
    vals = np.concatenate([value(P, combos[i:i + 200000], gws)
                           for i in range(0, len(combos), 200000)])
    out = {}
    for b in budgets:
        m = csum <= b + 1e-9
        if not m.any():
            out[b] = None; continue
        i = np.argmax(np.where(m, vals, -np.inf))
        out[b] = (combos[i], float(vals[i]), float(csum[i]))
    return out


def hill_climb(full, P_full, gws, budget, start_idx, rounds=40):
    """Single-player swaps against the FULL pool; the check on the pruning."""
    cost = full["cost"].values
    club = pd.factorize(full["team"])[0]
    cur = list(start_idx)
    best = value(P_full, np.array([cur]), gws)[0]
    for _ in range(rounds):
        improved = False
        for slot in range(SQUAD):
            base_cost = sum(cost[c] for i, c in enumerate(cur) if i != slot)
            cand = np.where(cost <= budget - base_cost + 1e-9)[0]
            cand = np.array([c for c in cand if c not in cur])
            if len(cand) == 0:
                continue
            trial = np.tile(np.array(cur), (len(cand), 1))
            trial[:, slot] = cand
            cl = club[trial]; cl.sort(axis=1)
            ok = np.ones(len(cl), dtype=bool)
            for k in range(SQUAD - MAX_PER_CLUB):
                ok &= ~(cl[:, k] == cl[:, k + MAX_PER_CLUB])
            if not ok.any():
                continue
            trial = trial[ok]
            v = value(P_full, trial, gws)
            j = int(np.argmax(v))
            if v[j] > best + 1e-9:
                best, cur, improved = float(v[j]), list(trial[j]), True
        if not improved:
            break
    return cur, best


def main():
    w = load()
    print(f"[study] {len(w)} defenders on the board")
    elig = w[w["p_start"] >= MIN_START].reset_index(drop=True)
    print(f"        {len(elig)} with start probability >= {MIN_START:.0%}")
    print(f"        squad of {SQUAD}, play best {PLAY} each week, "
          f"max {MAX_PER_CLUB} per club")
    gw_cols = [c for c in elig.columns if isinstance(c, (int, np.integer))]
    P_full = elig[sorted(gw_cols)].values.astype(np.float32)

    rows = []
    for h in HORIZONS:
        tot = P_full[:, :h].sum(axis=1)
        keep = set()
        for c, g in elig.groupby("cost"):
            order = g.index[np.argsort(-tot[g.index])][:TOP_PER_PRICE]
            keep.update(order.tolist())
        pool_idx = sorted(keep)
        pool = elig.loc[pool_idx].reset_index(drop=True)
        P = P_full[pool_idx]
        print(f"\n{'=' * 78}\nGW1-{h}  (pool {len(pool)} of {len(elig)}; "
              f"{len(list(combinations(range(len(pool)), SQUAD))):,} squads)\n{'=' * 78}")
        res = enumerate_pool(pool, P, h, BUDGETS)
        print(f"  {'budget':>7s} {'spend':>6s} {'pts':>7s} {'squad'}")
        for b in BUDGETS:
            r = res[b]
            if r is None:
                print(f"  {b:7.1f}    — no valid squad"); continue
            idx, val, spend = r
            names = pool.loc[list(idx)]
            # validate the pruning by climbing against the full pool
            full_start = [int(elig.index[(elig["player"] == p) & (elig["team"] == t)][0])
                          for p, t in zip(names["player"], names["team"])]
            climbed, cval = hill_climb(elig, P_full, h, b, full_start)
            tag = ""
            if cval > val + 1e-6:
                tag = f"  <- climb found +{cval - val:.2f}"
                names = elig.loc[climbed]; val, spend = cval, names["cost"].sum()
            sq = ", ".join(f"{r_.player} {r_.team[:3]} {r_.cost:.1f}"
                           for r_ in names.itertuples())
            print(f"  {b:7.1f} {spend:6.1f} {val:7.2f}  {sq}{tag}")
            rows.append({"horizon": h, "budget": b, "spend": spend, "pts": val,
                         "squad": sq})
            if h == 10 and b in (24, 26, 30):
                Pn = P_full[[int(elig.index[(elig["player"] == r_.player)
                                            & (elig["team"] == r_.team)][0])
                             for r_ in names.itertuples()]]
                picks = []
                for g in range(h):
                    top = np.argsort(-Pn[:, g])[:PLAY]
                    picks.append("/".join(sorted(names.iloc[t].player for t in top)))
                print(f"          weekly XI: " + "  ".join(
                    f"GW{i+1} {p}" for i, p in enumerate(picks)))

    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")

    print(f"\n{'=' * 78}\nWHAT DOES EXTRA BUDGET BUY?  (GW1-10)\n{'=' * 78}")
    t = R[R.horizon == 10].sort_values("budget")
    prev = None
    print(f"  {'budget':>7s} {'pts':>7s} {'gain':>7s} {'pts per extra 1.0m':>20s}")
    for _, r in t.iterrows():
        if prev is None:
            print(f"  {r.budget:7.1f} {r.pts:7.2f} {'—':>7s} {'—':>20s}")
        else:
            db = r.budget - prev.budget
            print(f"  {r.budget:7.1f} {r.pts:7.2f} {r.pts - prev.pts:+7.2f} "
                  f"{(r.pts - prev.pts) / db:20.2f}")
        prev = r


if __name__ == "__main__":
    main()
