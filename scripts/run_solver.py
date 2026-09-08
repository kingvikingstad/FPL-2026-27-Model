import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
run_solver.py — optimal squad for a gameweek, and a transfer plan across a horizon.
====================================================================================
Drives `src/solver.py` on the board this repo produces. Two modes:

  single    the best legal 15, XI and captain for one gameweek, from scratch
  horizon   a joint plan across several gameweeks with transfers priced at 4 points

Both report the optimality gap, because a squad that is 0.3 points from optimal and one
that is provably optimal are different objects and the difference should be visible.

Run:  python scripts/run_solver.py                 (GW1 squad + GW1-6 horizon)
Env:  SOLVER_GW=1  SOLVER_HORIZON=1,2,3,4,5,6  BENCH_WEIGHT=0.15  POOL=40
      RISK_LAMBDA=0.0  EP_COL=model_pts  TRIPLE_CAPTAIN=off  BENCH_BOOST=off
      BENCH_BOOST_GW=2   FREE_TRANSFERS=1   MAX_TRANSFERS=0
Out:  outputs/plans/solver_gw{N}_squad.csv, outputs/plans/solver_horizon_plan.csv

CHIPS ON A HORIZON
------------------
`BENCH_BOOST=on` boosts the SINGLE-gameweek solve only. A chip you intend to play in a
LATER week is a different problem: knowing the bench scores in GW2 changes what you buy
in GW1, because the fifteenth man stops being fodder. `BENCH_BOOST_GW=2` passes that
through to `solve_horizon`, which prices it jointly. `solver.solve_horizon` has supported
this since it was written; the runner simply never exposed it.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import solver as sv

SRC = _os.path.join(config.OUTPUTS, "fpl_2627_solver_inputs.csv")
EP_COL = _os.environ.get("EP_COL", "model_pts")
GW = int(_os.environ.get("SOLVER_GW", "1"))
HORIZON = [int(g) for g in _os.environ.get("SOLVER_HORIZON", "1,2,3,4,5,6").split(",")]
_bbg = _os.environ.get("BENCH_BOOST_GW", "").strip()
BENCH_BOOST_GW = int(_bbg) if _bbg else None
FREE_TRANSFERS = int(_os.environ.get("FREE_TRANSFERS", "1"))
_mt = _os.environ.get("MAX_TRANSFERS", "").strip()
MAX_TRANSFERS = int(_mt) if _mt else None
BENCH_W = float(_os.environ.get("BENCH_WEIGHT", "0.15"))
POOL = int(_os.environ.get("POOL", "40"))
RISK = float(_os.environ.get("RISK_LAMBDA", "0.0"))
ON = lambda k: _os.environ.get(k, "off").lower() in ("on", "1", "true")


def load():
    d = pd.read_csv(SRC)
    need = {"player_code", "display_name", "team", "pos", "cost", "gw", EP_COL}
    missing = need - set(d.columns)
    if missing:
        raise RuntimeError(f"{SRC} missing {sorted(missing)} — re-run export_workbook.py")
    d = d.rename(columns={EP_COL: "ep", "cost": "price"})
    # A player with no projection for a gameweek cannot be selected in it.
    d = d.dropna(subset=["ep", "price", "pos", "team"])
    return d


def show_squad(sq, ep_col="ep", title=""):
    if title:
        print(f"\n{title}")
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    s = sq.assign(_o=sq["pos"].map(order)).sort_values(
        ["in_xi", "_o", ep_col], ascending=[False, True, False])
    for _, r in s.iterrows():
        mark = "C" if r.get("is_captain") else (" " if r["in_xi"] else "b")
        print(f"  {mark} {r['pos']:3s} {r['display_name']:22s} {r['team']:15s} "
              f"£{r['price']:4.1f}m  {r[ep_col]:5.2f}")


def main():
    d = load()
    print(f"[input] {len(d):,} player-gameweeks, ep column '{EP_COL}'")

    # ---------- single gameweek ----------
    g = d[d.gw == GW].copy()
    pool = sv.prune(g, per_pos=POOL, ep_col="ep")
    print(f"\n[gw{GW}] pruned {len(g)} -> {len(pool)} candidates "
          f"(top {POOL} by ep per position)")
    r = sv.solve_gameweek(pool, bench_weight=BENCH_W, risk_lambda=RISK,
                          triple_captain=ON("TRIPLE_CAPTAIN"),
                          bench_boost=ON("BENCH_BOOST"))
    print("=" * 92)
    print(f"OPTIMAL SQUAD — GW{GW}   ({r['status']})")
    print("=" * 92)
    show_squad(r["squad"])
    cap = r["captain"]
    print(f"\n  captain      {cap['display_name']} ({cap['ep']:.2f} -> "
          f"{cap['ep'] * 2:.2f})")
    print(f"  XI expected  {r['xi_ep']:.2f}")
    print(f"  + captain    {r['captain_ep']:.2f}")
    print(f"  = total      {r['total_ep']:.2f} points")
    print(f"  bench        {r['bench_ep']:.2f} raw ({BENCH_W:.0%} weighted in objective)")
    print(f"  cost         £{r['cost']:.1f}m, £{r['budget_left']:.1f}m unspent")
    out = r["squad"][["player_code", "display_name", "team", "pos", "price", "ep",
                      "in_xi", "is_captain"]]
    p = _os.path.join(config.PLANS, f"solver_gw{GW}_squad.csv")
    out.to_csv(p, index=False)
    print(f"  wrote {p}")

    # sensitivity to the one judgment parameter in the objective
    print("\n  bench_weight sensitivity (how much does the bench assumption move it?):")
    for bw in [0.0, 0.15, 0.3]:
        rr = sv.solve_gameweek(pool, bench_weight=bw)
        same = len(set(rr["squad"].player_code) & set(r["squad"].player_code))
        print(f"    bw={bw:.2f}  XI {rr['xi_ep']:6.2f}  cost £{rr['cost']:5.1f}m  "
              f"{same}/15 squad overlap with bw={BENCH_W}")

    # ---------- horizon ----------
    print("\n" + "=" * 92)
    print(f"HORIZON PLAN — GW{HORIZON[0]}-{HORIZON[-1]}, transfers priced at "
          f"{sv.TRANSFER_COST:.0f} points")
    print("=" * 92)
    if BENCH_BOOST_GW is not None and BENCH_BOOST_GW not in HORIZON:
        raise SystemExit(f"BENCH_BOOST_GW={BENCH_BOOST_GW} is not in the horizon "
                         f"{HORIZON} — the chip cannot be priced outside the window")
    if BENCH_BOOST_GW is not None:
        print(f"  BENCH BOOST planned for GW{BENCH_BOOST_GW}: all 15 score that week, so "
              f"the bench is priced in full rather than at {BENCH_W:.0%}")
    if MAX_TRANSFERS == 0:
        print(f"  HOLD: no transfers permitted after GW{HORIZON[0]} — one fifteen has to "
              f"cover the whole window")
    elif MAX_TRANSFERS is not None:
        print(f"  at most {MAX_TRANSFERS} transfer(s) per gameweek")
    h = sv.solve_horizon(d, HORIZON, bench_weight=BENCH_W, per_pos=POOL,
                         ep_col="ep", price_col="price", verbose=True,
                         bench_boost_gw=BENCH_BOOST_GW,
                         free_transfers=FREE_TRANSFERS,
                         max_transfers=MAX_TRANSFERS)
    NAMES = dict(zip(d["player_code"], d["display_name"]))
    rows = []
    for pl in h["plans"]:
        capn = pl["captain"]["display_name"] if pl["captain"] is not None else "?"
        # sold players are by definition NOT in this week's squad, so the lookup has to
        # span the whole candidate pool rather than the current 15
        buys = [NAMES.get(c, str(c)) for c in pl["buys"]]
        sells = [NAMES.get(c, str(c)) for c in pl["sells"]]
        _bb = ("  + BENCH %.2f [BOOST]" % pl["bench_ep"]) if pl["bench_boost"] else ""
        print(f"\n  GW{pl['gw']}  XI {pl['xi_ep']:.2f} + C {pl['captain_ep']:.2f}{_bb} "
              f"= {pl['total_ep']:.2f}   £{pl['cost']:.1f}m   captain {capn}")
        if pl["gw"] != HORIZON[0]:
            print(f"        transfers: {pl['paid_transfers']} paid"
                  + (f"  IN {', '.join(buys)}" if buys else "")
                  + (f"  OUT {', '.join(sells)}" if sells else ""))
        for _, pr in pl["squad"].sort_values("ep", ascending=False).head(4).iterrows():
            pass
        rows.append({"gw": pl["gw"], "xi_ep": pl["xi_ep"],
                     "bench_ep": pl["bench_ep"], "bench_boost": int(pl["bench_boost"]),
                     "captain": capn, "captain_ep": pl["captain_ep"],
                     "total_ep": pl["total_ep"], "cost": pl["cost"],
                     "paid_transfers": pl["paid_transfers"],
                     "buys": "|".join(map(str, buys)),
                     "sells": "|".join(map(str, sells))})
    # The squads themselves. Written for every gameweek, printed for the first one (the
    # squad you actually enter) and for the boosted one (where the bench is the point).
    sq_rows = []
    for pl in h["plans"]:
        q = pl["squad"].reset_index().rename(columns={"index": "player_code"})
        q["gw"] = pl["gw"]
        sq_rows.append(q[["gw", "player_code", "display_name", "team", "pos", "price",
                          "ep", "in_xi", "is_captain"]])
    SQ = pd.concat(sq_rows, ignore_index=True)
    sp_path = _os.path.join(config.PLANS, "solver_horizon_squads.csv")
    SQ.to_csv(sp_path, index=False)

    for gw in dict.fromkeys([HORIZON[0]] + ([BENCH_BOOST_GW] if BENCH_BOOST_GW else [])):
        pl = next(x for x in h["plans"] if x["gw"] == gw)
        tag = "  [BENCH BOOST — all 15 score]" if pl["bench_boost"] else ""
        show_squad(pl["squad"], title=f"HORIZON SQUAD — GW{gw}{tag}")
        if pl["bench_boost"]:
            print(f"      bench contributes {pl['bench_ep']:.2f} of the "
                  f"{pl['total_ep']:.2f}")

    H = pd.DataFrame(rows)
    hp = _os.path.join(config.PLANS, "solver_horizon_plan.csv")
    H.to_csv(hp, index=False)
    print(f"  wrote {sp_path}")
    print(f"\n  horizon total {H['total_ep'].sum():.2f} points "
          f"({H['paid_transfers'].sum()} paid transfers, "
          f"-{H['paid_transfers'].sum() * sv.TRANSFER_COST:.0f} pts)")
    print(f"  wrote {hp}")

    # a myopic baseline, to show the horizon is doing something
    # The baseline has to play the SAME chips, or it is not a baseline. Solving every
    # week unboosted and comparing against a boosted horizon made the transfer constraint
    # look NEGATIVE — as though obeying the transfer rules earned points — when all it
    # measured was that one side of the comparison got a chip and the other did not.
    myopic = 0.0
    for gw in HORIZON:
        gg = sv.prune(d[d.gw == gw], per_pos=POOL, ep_col="ep")
        myopic += sv.solve_gameweek(
            gg, bench_weight=BENCH_W,
            bench_boost=(BENCH_BOOST_GW is not None and gw == BENCH_BOOST_GW),
        )["total_ep"]
    print(f"\n  unconstrained per-gameweek optimum (ignores transfers): "
          f"{myopic:.2f} pts")
    print(f"  horizon plan under real transfer rules:              "
          f"{H['total_ep'].sum():.2f} pts")
    print(f"  cost of the transfer constraint:                      "
          f"{myopic - H['total_ep'].sum():.2f} pts")


if __name__ == "__main__":
    main()
