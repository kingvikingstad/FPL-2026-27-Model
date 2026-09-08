import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
plan_constrained.py — GW1-3 plan under manager constraints, with a Bench Boost.
================================================================================
Runs the GW1-3 window with:
  * named players excluded (BAN)
  * a pinned XI shape (FORMATION, default 3-5-2 — three at the back, two forwards)
  * Bench Boost played in a chosen gameweek, chosen by comparing them

WHY BENCH BOOST CHANGES THE SQUAD, NOT JUST THE SCORE
------------------------------------------------------
Without the chip a bench is dead weight: the optimiser buys the cheapest legal filler
and spends everything else on eleven starters. With the chip all fifteen score, so the
last four places stop being fodder and start competing for budget on the same terms as
the XI. That is a different optimisation, and the right squad for a boosted week is not
the ordinary squad with the bench upgraded afterwards — it is built that way from the
start. So the chip week is passed INTO the horizon solve rather than scored on top of a
plan built without it.

The cost of pinning a formation is reported. A shape constraint can only reduce the
objective, and it is worth knowing by how much before accepting it.

Run:  python scripts/plan_constrained.py
Env:  BAN="Guéhi,Amad"  FORMATION=3-5-2  POOL=35  BB_GWS=1,2
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import solver as sv

SRC = _os.path.join(config.OUTPUTS, "fpl_2627_solver_inputs.csv")
EP = _os.environ.get("EP_COL", "model_pts")
POOL = int(_os.environ.get("POOL", "35"))

# BENCH_WEIGHT PINNED TO 0.0 — the objective and the reported total must mean the same
# thing. `total_ep` counts only the XI plus the captain in a non-boosted gameweek, so a
# non-zero bench weight had the optimiser valuing GW1 bench points that the reported
# score then ignored. Two definitions of "a gameweek" in one plan is a bug waiting to
# mislead, even though it happened to select the same 15 here (verified: identical squad
# and identical 139.57 total at 0.15 and 0.0).
#
# The cost of pinning it: autosubs are no longer modelled, so a bench player who comes on
# for a non-starter scores nothing in the plan's arithmetic. That is the conservative
# direction — it under-counts rather than over-counts — and it is the right default for a
# window that ends in a chip or a wildcard, where the bench is either boosted (and fully
# counted) or about to be replaced.
BENCH_W = float(_os.environ.get("BENCH_WEIGHT", "0.0"))

FORMATION = _os.environ.get("FORMATION", "3-5-2")
BAN = [b.strip() for b in
       _os.environ.get("BAN", "Guéhi,Amad,Thiaw,Hume").split(",") if b.strip()]

# Per-position spend caps. DEF pinned to £28.0m: the squad must still field five
# defenders, this caps what they cost. Measured against the uncapped build it gives up
# 0.54 points and frees £3.0m for the front line, and it cuts exposure to the DefCon
# channel — which drives most of the defensive tilt (removing DefCon entirely drops
# defence spend £6.5m on its own) and whose player-level RANKING is still only 0.320
# correlated with Solio even though its level is well calibrated.
POS_BUDGET = {}
if _os.environ.get("DEF_BUDGET", "28.0").lower() not in ("", "off", "none"):
    POS_BUDGET["DEF"] = float(_os.environ.get("DEF_BUDGET", "28.0"))
for _p in ("GK", "MID", "FWD"):
    _v = _os.environ.get(f"{_p}_BUDGET")
    if _v:
        POS_BUDGET[_p] = float(_v)
BB_GWS = [int(g) for g in _os.environ.get("BB_GWS", "2").split(",")]
WINDOW = [int(g) for g in _os.environ.get("WINDOW", "1,2").split(",")]
# 1000 makes any transfer strictly worse than holding; the opening build is exempt.
CHURN = float(_os.environ.get("CHURN_PENALTY", "1000.0"))

# Cap on money sitting on the bench in a NON-boosted gameweek. A benched player scores
# nothing that week, so this is money that does not play. Applies to GW1 only here —
# the boosted week wants an expensive bench and is deliberately exempt.
#
# The two goals conflict, and the sweep prices it: going from no cap to £18m moves GW1
# from 68.18 to 70.42 and GW2 from 70.45 to 67.73, for a net loss of just 0.49 over the
# window. £20m is the pick — it lifts GW1 by 1.67, puts £3.5m more into the XI, and the
# GW2 bench actually scores slightly MORE (16.09 vs 15.91) because the cheaper bench
# players it buys happen to have better second fixtures. Below £18m it is infeasible:
# a legal 3-5-2 must bench a keeper, two defenders and a forward, and they cost
# something.
BENCH_CAP = _os.environ.get("BENCH_SPEND_CAP", "20.0")
BENCH_CAP = None if BENCH_CAP.lower() in ("", "off", "none") else float(BENCH_CAP)


def load():
    d = pd.read_csv(SRC).rename(columns={EP: "ep", "cost": "price"})
    return d.dropna(subset=["ep", "price", "pos", "team"])


def resolve_bans(d, names):
    """Names -> player_codes, reported explicitly. A ban that matches nobody is a typo
    the manager needs to know about, not something to swallow."""
    # EXACT first, then whole-word. Substring matching is not safe here: 'Amad' is a
    # substring of 'Kamada' and 'Al-Hamadi', so a contains() ban silently removed three
    # players instead of one. A ban is a hard constraint the manager stated — it must hit
    # exactly who they named, and an ambiguous name must be reported, not guessed.
    import re
    u = d.drop_duplicates("player_code")
    codes, report = [], []
    for nm in names:
        key = nm.lower().strip()
        exact = u[u["display_name"].str.lower().eq(key) | u["player"].str.lower().eq(key)]
        if len(exact) == 0:
            pat = rf"(?:^|[\s.\-']){re.escape(key)}(?:$|[\s.\-'])"
            exact = u[u["display_name"].str.lower().str.contains(pat, regex=True, na=False)
                      | u["player"].str.lower().str.contains(pat, regex=True, na=False)]
        if len(exact) == 0:
            report.append(f"  !! '{nm}' matched NO player — ban NOT applied")
        elif len(exact) > 1:
            who = ", ".join(f"{r.display_name} ({r.team})" for r in exact.itertuples())
            report.append(f"  !! '{nm}' is ambiguous ({who}) — ban NOT applied, "
                          f"name it exactly")
        else:
            r = exact.iloc[0]
            codes.append(int(r["player_code"]))
            report.append(f"  banned {r['display_name']} ({r['team']}, {r['pos']})")
    return codes, report


def show(sq, title, boosted=False):
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    s = sq.assign(_o=sq["pos"].map(order)).sort_values(
        ["in_xi", "_o", "ep"], ascending=[False, True, False])
    print(f"\n{title}")
    for _, r in s.iterrows():
        mark = "C" if r.get("is_captain") else (" " if r["in_xi"] else "b")
        note = "" if (r["in_xi"] or not boosted) else "  <- scores (boost)"
        print(f"  {mark} {r['pos']:3s} {r['display_name']:22s} {r['team']:15s} "
              f"£{r['price']:4.1f}m  {r['ep']:5.2f}{note}")


def main():
    d = load()
    NAMES = dict(zip(d["player_code"], d["display_name"]))
    ban_codes, rep = resolve_bans(d, BAN)
    print("=" * 92)
    print(f"CONSTRAINED PLAN — GW{WINDOW}, formation {FORMATION}, "
          f"Bench Boost GW{BB_GWS}")
    print("=" * 92)
    print(f"  bench_weight {BENCH_W}  (0 = a non-boosted gameweek is exactly 11 players)")
    print(f"  position budgets {POS_BUDGET}")
    print(f"  churn_penalty {CHURN:.0f}  "
          f"({'no transfers after the opening build' if CHURN >= 100 else 'transfers allowed'})")
    for line in rep:
        print(line)

    # ---------- where to play the chip ----------
    print(f"\n{'-'*92}\nCHOOSING THE BENCH BOOST WEEK\n{'-'*92}")
    results = {}
    for bb in BB_GWS + [None]:
        h = sv.solve_horizon(d, WINDOW, bench_weight=BENCH_W, per_pos=POOL,
                             ep_col="ep", price_col="price", banned=ban_codes,
                             formation=FORMATION, bench_boost_gw=bb,
                             churn_penalty=CHURN, pos_budget=POS_BUDGET, max_bench_spend=BENCH_CAP,
                             verbose=False)
        tot = sum(p["total_ep"] for p in h["plans"])
        results[bb] = (h, tot)
        lab = f"Bench Boost in GW{bb}" if bb else "no Bench Boost"
        print(f"  {lab:24s} GW1-3 total {tot:7.2f}")
    base = results[None][1]
    for bb in BB_GWS:
        print(f"  boost gain, GW{bb}: {results[bb][1] - base:+.2f} pts")
    best_bb = max(BB_GWS, key=lambda g: results[g][1])
    print(f"\n  -> play it in GW{best_bb}")

    h, tot = results[best_bb]
    for pl in h["plans"]:
        cap = pl["captain"]["display_name"] if pl["captain"] is not None else "?"
        boosted = pl["gw"] == best_bb
        tag = "  [BENCH BOOST]" if boosted else ""
        show(pl["squad"], f"GW{pl['gw']}  —  {pl['total_ep']:.2f} pts   "
                          f"£{pl['cost']:.1f}m   captain {cap}{tag}", boosted)
        if pl["gw"] != WINDOW[0]:
            ins = [NAMES.get(c, c) for c in pl["buys"]]
            outs = [NAMES.get(c, c) for c in pl["sells"]]
            print(f"      {'transfer: OUT ' + ', '.join(map(str, outs)) + '  ->  IN ' + ', '.join(map(str, ins)) if ins else 'no transfer'}"
                  f"   ({pl['paid_transfers']} paid)")
    print(f"\n  GW1-3 TOTAL: {tot:.2f} points")

    # ---------- what the constraints cost ----------
    print(f"\n{'-'*92}\nWHAT THE CONSTRAINTS COST\n{'-'*92}")
    free = sv.solve_horizon(d, WINDOW, bench_weight=BENCH_W, per_pos=POOL, ep_col="ep",
                            price_col="price", bench_boost_gw=best_bb,
                            churn_penalty=CHURN, pos_budget=POS_BUDGET, max_bench_spend=BENCH_CAP,
                            verbose=False)
    free_tot = sum(p["total_ep"] for p in free["plans"])
    # Persist the unconstrained build so it can be tracked alongside the others. Same
    # chip and transfer plan (boost week, no transfers, same window) — only the manager's
    # PREFERENCES are removed, so the comparison isolates what those preferences cost
    # rather than confounding them with a different strategy.
    free["plans"][0]["squad"].reset_index().to_csv(
        _os.path.join(config.PLANS, "plan_unconstrained_squad.csv"), index=False)
    noform = sv.solve_horizon(d, WINDOW, bench_weight=BENCH_W, per_pos=POOL, ep_col="ep",
                              price_col="price", banned=ban_codes,
                              bench_boost_gw=best_bb, churn_penalty=CHURN,
                              pos_budget=POS_BUDGET, max_bench_spend=BENCH_CAP, verbose=False)
    nf_tot = sum(p["total_ep"] for p in noform["plans"])
    print(f"  unconstrained (with boost)          : {free_tot:7.2f}")
    print(f"  + bans only                         : {nf_tot:7.2f}  "
          f"({nf_tot - free_tot:+.2f})")
    print(f"  + bans and {FORMATION}                  : {tot:7.2f}  "
          f"({tot - nf_tot:+.2f})")
    print(f"  total cost of your constraints      : {tot - free_tot:+.2f} pts")

    out = h["plans"][0]["squad"].reset_index()
    out.to_csv(_os.path.join(config.PLANS, "plan_constrained_squad.csv"), index=False)
    print(f"\n  wrote {_os.path.join(config.PLANS, 'plan_constrained_squad.csv')}")


if __name__ == "__main__":
    main()
