import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
export_wildcard_xi.py — the best legal fifteen for EVERY gameweek, independently.
==================================================================================
"What would I field this week if I could rebuild from scratch?" — the answer for each
gameweek in turn, each solved on its own with no reference to the week before it.

WHAT THIS IS, AND WHAT IT IS NOT
---------------------------------
It is a CEILING and a benchmark, not a plan. You hold one wildcard per half-season; a
sequence of independently optimal squads is not reachable, and the gap between it and
your own XI is not points you left on the table — it is the price of having a squad at
all. Read it two ways that ARE actionable:

  * the GAP each week, as a running check on whether your side is drifting off the pace;
  * how OFTEN a player appears across the weeks. A player optimal in eleven weeks of
    twelve is a hold whatever this week's number says, and one who appears once is a
    fixture punt. That frequency is the column worth reading.

The optimisation is `solver.solve_gameweek` — the same MILP the rest of the project
uses, through HiGHS. It is not reimplemented here and it is not reimplemented in the
explorer page either; the page reads what this writes.

BUDGET
------
Your real budget when a live squad is on file (`squad value + bank`), because a wildcard
is spent against what you actually have, not against a notional 100.0. Without one it
falls back to 100.0 and says so.

PRUNING, AND THE ONE PLACE IT BITES
------------------------------------
`solver.prune` keeps the top N per position by expected points. Its docstring says that
does not change the optimum, which holds while the budget is slack and fails when it
binds: a fifteen needs cheap fodder, and £4.0m fodder rarely makes a top-40-by-points
shortlist. So the cheapest few per position are added to `keep` explicitly. Without that
the solver can be handed a shortlist in which no legal fifteen fits the budget at all.

Run:  python scripts/export_wildcard_xi.py            # the next 12 gameweeks
      python scripts/export_wildcard_xi.py --gws 3-20
Out:  outputs/wildcard_xi.csv
"""
import argparse
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import solver as sv

EP_COL = _os.environ.get("EP_COL", "blended")
POOL = int(_os.environ.get("POOL", "50"))
FODDER = int(_os.environ.get("FODDER", "8"))      # cheapest per position, always available
BENCH_W = float(_os.environ.get("BENCH_WEIGHT", "0.15"))
OUT = _os.path.join(config.OUTPUTS, "wildcard_xi.csv")


def board():
    d = pd.read_csv(_os.path.join(config.OUTPUTS, "gw_board_long.csv"))
    d = d.rename(columns={EP_COL: "ep", "cost": "price"})
    d = d.dropna(subset=["ep", "price", "pos", "team", "player_code"])
    d["display_name"] = d["player"]
    return d


def budget_and_squad():
    """(budget, owned player_codes, label). Falls back to a clean 100.0."""
    try:
        import fpl_entry as fe
        sq, meta = fe.load()
        if sq is not None and len(sq) and meta:
            b = float((meta.get("squad_value") or 0) + (meta.get("bank") or 0))
            if b > 50:
                return b, set(int(c) for c in sq["player_code"]), \
                       f"{meta.get('entry_name','your squad')} ({b:.1f})"
    except Exception:
        pass
    return sv.BUDGET, set(), f"no live squad on file, using {sv.BUDGET:.1f}"


def parse_gws(spec, lo, hi):
    if not spec:
        return list(range(lo, hi + 1))
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gws", default=None, help="e.g. 3-14 or 3,4,5 (default: next 12)")
    a = ap.parse_args()

    d = board()
    budget, owned, blabel = budget_and_squad()
    # The first gameweek NOTHING has been played in — the explorer's `first_clean`.
    #
    # This previously excluded only gameweeks that were FULLY played, which is the
    # explorer's `first_live`, a different marker despite the comment here claiming
    # otherwise. On 2026-09-06 that had the solver open its window on GW3, whose
    # deadline had passed and which was settled for 18 of the 20 clubs: it was
    # optimising a fifteen for a gameweek nobody could still pick a team for, and the
    # printed headline read "GW3 optimal 15" two days after most of GW3 was over.
    #
    # A partly-played week is neither past nor future, and for squad selection it is
    # past: you cannot buy into it. So both fully- and partly-played weeks are skipped,
    # which is exactly what the explorer does and for the reason it documents — a window
    # opening on a settled week is summing history for almost everyone in it.
    try:
        import gw_explorer as gx
        done = gx.played_fixtures()
        allgw = sorted(int(g) for g in d["gw"].unique())
        clubs_in = {g: d.loc[d["gw"] == g, "team"].unique() for g in allgw}
        full = {g for g in allgw if all((t, g) in done for t in clubs_in[g])}
        partial = {g for g in allgw
                   if any((t, g) in done for t in clubs_in[g])} - full
        first = min([g for g in allgw if g not in full and g not in partial] or [1])
    except Exception:
        first = 1
    gws = parse_gws(a.gws, first, min(first + 11, int(d["gw"].max())))

    print("=" * 78)
    print(f"WILDCARD XI — the best legal fifteen for each gameweek, solved independently")
    print("=" * 78)
    print(f"budget: {blabel}   ep column: {EP_COL}   pool: top {POOL}/pos + {FODDER} cheapest")
    print(f"gameweeks: GW{gws[0]}-GW{gws[-1]}\n")

    rows, summary = [], []
    for gw in gws:
        g = d[d["gw"] == gw].drop_duplicates("player_code")
        if len(g) < 40:
            print(f"  GW{gw}: only {len(g)} candidates — skipped"); continue
        # cheap fodder is kept explicitly; see the module docstring
        fodder = set()
        for _, gp in g.groupby("pos"):
            fodder |= set(gp.nsmallest(FODDER, "price")["player_code"])
        pool = sv.prune(g, per_pos=POOL, ep_col="ep", keep=fodder)
        try:
            r = sv.solve_gameweek(pool, budget=budget, bench_weight=BENCH_W, ep_col="ep",
                                  price_col="price")
        except Exception as e:
            print(f"  GW{gw}: solve failed ({type(e).__name__}: {e})"); continue
        sq = r["squad"]
        cap = r["captain"]
        for _, p in sq.iterrows():
            rows.append({
                "gw": gw, "player_code": int(p["player_code"]), "player": p["display_name"],
                "pos": p["pos"], "team": p["team"], "price": float(p["price"]),
                "ep": float(p["ep"]), "in_xi": bool(p["in_xi"]),
                "is_captain": bool(p["is_captain"]),
                "bench_order": (None if p.get("bench_order") != p.get("bench_order")
                                else p.get("bench_order")),
                "owned": int(p["player_code"]) in owned,
            })
        summary.append({"gw": gw, "xi_ep": r["xi_ep"], "captain_ep": r["captain_ep"],
                        "total_ep": r["total_ep"], "cost": r["cost"],
                        "budget_left": r["budget_left"],
                        "n_owned": int(sum(1 for _, p in sq.iterrows()
                                           if int(p["player_code"]) in owned))})
        print(f"  GW{gw:<3} XI {r['xi_ep']:6.2f} + cap {r['captain_ep']:5.2f} "
              f"= {r['total_ep']:6.2f}   £{r['cost']:5.1f}m (left {r['budget_left']:4.1f})"
              f"   captain {cap['display_name'] if cap is not None else '?'}"
              + (f"   {summary[-1]['n_owned']}/15 already yours" if owned else ""))

    if not rows:
        print("nothing solved"); return 0
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    # who is optimal WEEK AFTER WEEK — the column that is actually actionable
    freq = (out[out["in_xi"]].groupby(["player_code", "player", "pos", "team"])
               .size().reset_index(name="weeks_in_xi")
               .sort_values("weeks_in_xi", ascending=False))
    n = len(summary)
    print(f"\nmost-selected in the XI across {n} gameweeks:")
    for _, r in freq.head(12).iterrows():
        mark = " (yours)" if int(r["player_code"]) in owned else ""
        print(f"    {r['player']:16s} {r['pos']:3s} {r['team']:14s} "
              f"{int(r['weeks_in_xi']):2d}/{n}{mark}")
    if owned:
        mine = freq[freq["player_code"].isin(owned)]["weeks_in_xi"].sum()
        print(f"\n  your players account for {int(mine)} of {n * 11} optimal XI slots")
    print(f"\nwrote {OUT} ({len(out)} rows over {n} gameweeks)")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
