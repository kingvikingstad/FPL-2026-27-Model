import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
plan_gw1_3.py — opening squad for a GW1-3 window that ends in a reset.
=======================================================================
The situation this solves: the whole squad is still free to choose (unlimited transfers
before the first deadline), the objective is points in GW1-3 only, and a Wildcard or
Free Hit is under consideration at GW4.

WHY THAT LAST CLAUSE CHANGES THE ANSWER
----------------------------------------
A normal opening squad is a compromise between scoring now and remaining viable later —
you keep a balanced spread, avoid one-week punts, and protect squad value so transfers
stay cheap. A squad that will be torn up at GW4 has NO TERMINAL VALUE, so all three of
those considerations are worth zero:

  * fixtures after GW3 are irrelevant, so a player with a brilliant GW1-3 run and an
    ugly GW4-10 run is strictly better than a steadier alternative;
  * squad value and price rises are irrelevant over three weeks;
  * ending GW3 with an unbalanced squad costs nothing, because it is about to be
    replaced wholesale.

So this optimises the three gameweeks in isolation and deliberately ignores everything
past them. That is correct ONLY if the reset actually happens — the sensitivity at the
bottom prices what it costs if you change your mind and keep the squad.

FREE HIT vs WILDCARD, WHICH IS NOT THE SAME DECISION
-----------------------------------------------------
  Free Hit  replaces the squad for ONE gameweek, then reverts. Worth playing on a
            single exceptional week (a blank/double), and it leaves the GW1-3 squad
            intact afterwards — so the GW1-3 squad still has to be viable at GW5+.
  Wildcard  replaces the squad PERMANENTLY from that point. This is the one that makes
            the GW1-3 squad genuinely disposable.

The report below prices both: what the GW1-3 squad is worth if carried into GW4-10, and
what an unrestricted GW4-10 squad is worth. The difference is the wildcard's value.

Run:  python scripts/plan_gw1_3.py
Env:  POOL=40  BENCH_WEIGHT=0.15  EP_COL=model_pts
Out:  outputs/plan_gw1_3_squad.csv, outputs/plan_gw1_3_transfers.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import solver as sv

SRC = _os.path.join(config.OUTPUTS, "fpl_2627_solver_inputs.csv")
EP_COL = _os.environ.get("EP_COL", "model_pts")
POOL = int(_os.environ.get("POOL", "40"))
BENCH_W = float(_os.environ.get("BENCH_WEIGHT", "0.15"))
WINDOW = [1, 2, 3]
LATER = [4, 5, 6, 7, 8, 9, 10]


def load():
    d = pd.read_csv(SRC).rename(columns={EP_COL: "ep", "cost": "price"})
    return d.dropna(subset=["ep", "price", "pos", "team"])


def show(sq, title):
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    s = sq.assign(_o=sq["pos"].map(order)).sort_values(
        ["in_xi", "_o", "ep"], ascending=[False, True, False])
    print(f"\n{title}")
    for _, r in s.iterrows():
        mark = "C" if r.get("is_captain") else (" " if r["in_xi"] else "b")
        print(f"  {mark} {r['pos']:3s} {r['display_name']:22s} {r['team']:15s} "
              f"£{r['price']:4.1f}m  {r['ep']:5.2f}")


def main():
    d = load()
    NAMES = dict(zip(d["player_code"], d["display_name"]))
    PRICE = dict(zip(d["player_code"], d["price"]))

    print("=" * 96)
    print("OPENING SQUAD OPTIMISED FOR GW1-3 ONLY  (reset assumed at GW4)")
    print("=" * 96)
    h = sv.solve_horizon(d, WINDOW, bench_weight=BENCH_W, per_pos=POOL,
                         ep_col="ep", price_col="price", initial_squad=None,
                         verbose=True)
    plans = h["plans"]
    total = sum(p["total_ep"] for p in plans)

    for p in plans:
        cap = p["captain"]["display_name"] if p["captain"] is not None else "?"
        sq = p["squad"].rename(columns={"price": "price"})
        show(sq, f"GW{p['gw']}  —  XI {p['xi_ep']:.2f} + captain {p['captain_ep']:.2f} "
                 f"= {p['total_ep']:.2f} pts   £{p['cost']:.1f}m   captain {cap}")
        if p["gw"] != WINDOW[0]:
            ins = [NAMES.get(c, c) for c in p["buys"]]
            outs = [NAMES.get(c, c) for c in p["sells"]]
            if ins:
                print(f"      transfer: OUT {', '.join(map(str, outs))}  ->  "
                      f"IN {', '.join(map(str, ins))}   "
                      f"({p['paid_transfers']} paid, -{p['paid_transfers']*4} pts)")
            else:
                print("      no transfer (free transfer banked)")

    print(f"\n  GW1-3 TOTAL: {total:.2f} points")

    gw1 = plans[0]["squad"]
    out = gw1.reset_index()[["player_code", "display_name", "team", "pos", "price",
                             "ep", "in_xi", "is_captain"]]
    out.to_csv(_os.path.join(config.PLANS, "plan_gw1_3_squad.csv"), index=False)
    tr = pd.DataFrame([{
        "gw": p["gw"], "captain": (p["captain"]["display_name"]
                                   if p["captain"] is not None else "?"),
        "xi_ep": p["xi_ep"], "captain_ep": p["captain_ep"], "total_ep": p["total_ep"],
        "cost": p["cost"], "paid_transfers": p["paid_transfers"],
        "in": "|".join(str(NAMES.get(c, c)) for c in p["buys"]),
        "out": "|".join(str(NAMES.get(c, c)) for c in p["sells"])} for p in plans])
    tr.to_csv(_os.path.join(config.PLANS, "plan_gw1_3_transfers.csv"), index=False)

    # ---------- is optimising for 3 weeks actually different? ----------
    print("\n" + "=" * 96)
    print("DOES THE THREE-WEEK VIEW CHANGE THE SQUAD? (vs building for GW1 alone)")
    print("=" * 96)
    g1pool = sv.prune(d[d.gw == 1], per_pos=POOL, ep_col="ep")
    myopic = sv.solve_gameweek(g1pool, bench_weight=BENCH_W)
    ov = len(set(myopic["squad"].player_code) & set(gw1.index))
    print(f"  GW1-only optimal squad scores {myopic['total_ep']:.2f} in GW1")
    print(f"  GW1-3 optimal squad scores    {plans[0]['total_ep']:.2f} in GW1  "
          f"({plans[0]['total_ep'] - myopic['total_ep']:+.2f})")
    print(f"  squad overlap: {ov}/15 — the three-week view moves "
          f"{15 - ov} player(s)")

    # Score the myopic squad across the window. free_transfers=0 is deliberate and the
    # comparison is wrong without it: handing solve_horizon an initial squad lets it
    # transfer IN that same gameweek, so with free_transfers=1 the myopic branch gets a
    # GW1 transfer the fresh build never had, and beats the joint optimum by construction.
    # Both sides must be locked at GW1 and earn their first free transfer for GW2.
    myo_codes = list(myopic["squad"].player_code)
    hm = sv.solve_horizon(d, WINDOW, bench_weight=BENCH_W, per_pos=POOL, ep_col="ep",
                          price_col="price", initial_squad=myo_codes,
                          free_transfers=0, verbose=False)
    myo_total = sum(p["total_ep"] for p in hm["plans"])
    print(f"\n  starting from the GW1-only squad, best GW1-3 total: {myo_total:.2f}")
    print(f"  starting from the GW1-3 optimal squad:               {total:.2f}")
    print(f"  cost of building myopically:                         "
          f"{total - myo_total:+.2f} points")

    # ---------- what is the reset worth at GW4? ----------
    print("\n" + "=" * 96)
    print("THE GW4 DECISION — what is a wildcard actually worth?")
    print("=" * 96)
    end_squad = list(plans[-1]["squad"].index)
    keep = sv.solve_horizon(d, LATER, bench_weight=BENCH_W, per_pos=POOL, ep_col="ep",
                            price_col="price", initial_squad=end_squad,
                            free_transfers=1, verbose=False)
    keep_total = sum(p["total_ep"] for p in keep["plans"])
    fresh = sv.solve_horizon(d, LATER, bench_weight=BENCH_W, per_pos=POOL, ep_col="ep",
                             price_col="price", initial_squad=None, verbose=False)
    fresh_total = sum(p["total_ep"] for p in fresh["plans"])
    print(f"  carry the GW1-3 squad into GW4-10 (1 FT/week): {keep_total:.2f} pts")
    print(f"  unrestricted rebuild at GW4 (wildcard):        {fresh_total:.2f} pts")
    print(f"  WILDCARD VALUE over GW4-10:                    {fresh_total - keep_total:+.2f} pts")
    print("\n  Note: this is an UPPER bound on the wildcard. The rebuild is priced with")
    print("  the same projections that chose the GW1-3 squad, so it inherits their")
    print("  errors; and by GW4 you will have three gameweeks of real results, which is")
    print("  exactly the information a wildcard should be waiting for.")

    # single-week Free Hit value, gameweek by gameweek
    print("\n  FREE HIT, priced per single gameweek (squad reverts afterwards):")
    for gw in LATER[:4]:
        gpool = sv.prune(d[d.gw == gw], per_pos=POOL, ep_col="ep",
                         keep=end_squad)
        best = sv.solve_gameweek(gpool, bench_weight=BENCH_W)
        held = d[(d.gw == gw) & (d.player_code.isin(end_squad))]
        if len(held) >= 15:
            hb = sv.best_xi_from_squad(held.copy())
            print(f"    GW{gw}: hold {hb['total_ep']:6.2f}   free hit "
                  f"{best['total_ep']:6.2f}   gain {best['total_ep'] - hb['total_ep']:+.2f}")


if __name__ == "__main__":
    main()
