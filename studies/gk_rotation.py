from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
gk_rotation.py — best goalkeeper pair to rotate over GW1-2, 1-3, 1-6 and 1-10
=============================================================================
Only one keeper can start in a gameweek, so a PAIR is worth

    sum over gameweeks of  max(keeper A that week, keeper B that week)

not the sum of their averages. That distinction is the whole exercise: the best pair is
not generally the two best keepers, it is the two whose fixtures are most COMPLEMENTARY —
one at home while the other travels, one facing a promoted side while the other faces a
title contender. Two excellent keepers who are strong in the same weeks waste the slot.

The horizon matters too. A pair chosen for GW1-2 can be a poor GW1-10 pair, so each
horizon is optimised separately and the table shows what the choice costs at the others.

WHAT IS CHECKED BEFORE RECOMMENDING
------------------------------------
A rotation needs two players who actually START. The projection already prices
availability in — a backup's weekly number is small because his start probability is —
but a pair can still look good on paper while resting on a keeper who plays 30% of the
time, so the implied start probability is reported alongside every recommendation.

Uses the model-only `mean` column, not `blended`: `blended` mixes an external feed into
GW1 alone, which would make the first gameweek incomparable with the rest.

Run:  python studies/gk_rotation.py
"""
import warnings; warnings.filterwarnings("ignore")
from itertools import combinations
import numpy as np, pandas as pd

HORIZONS = (2, 3, 6, 10)
TIERS = (("both <= 4.5", 4.5), ("both <= 5.0", 5.0))
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "gk_rotation.csv")


def load():
    d = pd.read_csv(_os.path.join(config.OUTPUTS, "gw_board_long.csv"))
    g = d[d["pos"] == "GK"].copy()
    wide = g.pivot_table(index=["player", "team", "cost", "own"], columns="gw",
                         values="mean", aggfunc="first").fillna(0.0)
    wide = wide.reset_index()
    # implied start probability: the projection embeds availability, so a keeper whose
    # weekly number is a fraction of a starting keeper's is a backup
    try:
        pri = pd.read_pickle(config.MS_PRIORS)
        pri["p_start"] = pri["start_a"] / (pri["start_a"] + pri["start_b"])
        names = pd.read_csv(_os.path.join(config.repo("2026-2027"), "players.csv"))
        teams = pd.read_csv(_os.path.join(config.repo("2026-2027"), "teams.csv"))
        tname = dict(zip(teams["code"], teams["name"]))
        names["team"] = names["team_code"].map(tname).replace(
            {"Man Utd": "Man United", "Spurs": "Tottenham"})
        m = names[["player_code", "web_name", "team"]].merge(
            pri[["player_code", "p_start"]], on="player_code", how="inner")
        # join on name AND club: web_name is not unique (two different Martinez), and a
        # name-only merge fans out, duplicating keepers and producing identical "pairs"
        m = m.rename(columns={"web_name": "player"}).drop_duplicates(["player", "team"])
        before = len(wide)
        wide = wide.merge(m[["player", "team", "p_start"]], on=["player", "team"],
                          how="left")
        if len(wide) != before:
            raise RuntimeError(f"p_start merge changed row count {before} -> {len(wide)}")
    except Exception as e:
        print(f"[warn] start probabilities unavailable ({type(e).__name__}: {e})")
        wide["p_start"] = np.nan
    return wide


def best_pairs(w, cap, horizon, top=6):
    elig = w[w["cost"] <= cap].reset_index(drop=True)
    gws = list(range(1, horizon + 1))
    P = elig[gws].values
    rows = []
    for i, j in combinations(range(len(elig)), 2):
        rot = np.maximum(P[i], P[j]).sum()
        solo = max(P[i].sum(), P[j].sum())
        rows.append({
            "a": elig.at[i, "player"], "a_team": elig.at[i, "team"],
            "a_cost": elig.at[i, "cost"],
            "b": elig.at[j, "player"], "b_team": elig.at[j, "team"],
            "b_cost": elig.at[j, "cost"],
            "combined_cost": elig.at[i, "cost"] + elig.at[j, "cost"],
            "rotation_pts": rot, "best_single": solo, "gain": rot - solo,
            "a_ps": elig.at[i, "p_start"], "b_ps": elig.at[j, "p_start"],
            "i": i, "j": j,
        })
    R = pd.DataFrame(rows)
    # Over a short horizon many pairs TIE, because one keeper starts every week and the
    # partner never gets picked — the partner is then arbitrary. Break ties on the
    # partner's own standalone value so the recommendation is the one that still holds up
    # if the horizon extends or the fixture read changes.
    R["tiebreak"] = [min(P[i].sum(), P[j].sum()) for i, j in zip(R["i"], R["j"])]
    R = R.sort_values(["rotation_pts", "tiebreak"], ascending=False)
    return R.head(top), elig


def weekly_plan(elig, i, j, horizon):
    gws = list(range(1, horizon + 1))
    a, b = elig.loc[i], elig.loc[j]
    out = []
    for gw in gws:
        pick = a["player"] if a[gw] >= b[gw] else b["player"]
        out.append((gw, pick, max(a[gw], b[gw]), min(a[gw], b[gw])))
    return out


def main():
    w = load()
    print(f"[study] {len(w)} goalkeepers on the board; "
          f"{(w.cost <= 4.5).sum()} at <=4.5, {(w.cost <= 5.0).sum()} at <=5.0")
    print("        pair value = sum over gameweeks of the BETTER keeper that week")
    print("        (model-only `mean` column, so every gameweek is comparable)")

    all_rows = []
    for tier_name, cap in TIERS:
        print("\n" + "=" * 78)
        print(f"TIER: {tier_name}")
        print("=" * 78)
        for h in HORIZONS:
            R, elig = best_pairs(w, cap, h)
            best = R.iloc[0]
            print(f"\n  --- GW1-{h} ---")
            print(f"  {'pair':44s} {'£':>5s} {'rot':>7s} {'best 1':>7s} {'gain':>6s}")
            for _, r in R.head(4).iterrows():
                pair = f"{r['a']} ({r['a_team']} {r['a_cost']:.1f}) + {r['b']} ({r['b_team']} {r['b_cost']:.1f})"
                print(f"  {pair:44s} {r['combined_cost']:5.1f} {r['rotation_pts']:7.2f} "
                      f"{r['best_single']:7.2f} {r['gain']:+6.2f}")
            ps = f"start prob {best['a_ps']:.2f} / {best['b_ps']:.2f}" \
                if np.isfinite(best.get("a_ps", np.nan)) else "start prob n/a"
            print(f"  best: {best['a']} + {best['b']}  ({ps})")
            plan = weekly_plan(elig, int(best["i"]), int(best["j"]), h)
            print("  weekly: " + "  ".join(f"GW{gw}:{who}" for gw, who, _, _ in plan))
            all_rows.append({"tier": tier_name, "horizon": h, "a": best["a"],
                             "b": best["b"], "combined_cost": best["combined_cost"],
                             "rotation_pts": best["rotation_pts"],
                             "best_single": best["best_single"], "gain": best["gain"]})

    # does the GW1-10 winner hold up at the shorter horizons, and vice versa?
    print("\n" + "=" * 78)
    print("HORIZON SENSITIVITY — is one pair good at every length?")
    print("=" * 78)
    for tier_name, cap in TIERS:
        R10, elig = best_pairs(w, cap, 10, top=1)
        i, j = int(R10.iloc[0]["i"]), int(R10.iloc[0]["j"])
        print(f"\n  {tier_name}: GW1-10 winner = {R10.iloc[0]['a']} + {R10.iloc[0]['b']}")
        print(f"  {'horizon':>9s} {'this pair':>11s} {'best pair':>11s} {'shortfall':>10s}")
        for h in HORIZONS:
            gws = list(range(1, h + 1))
            P = elig[gws].values
            here = np.maximum(P[i], P[j]).sum()
            Rh, _ = best_pairs(w, cap, h, top=1)
            print(f"  {'GW1-' + str(h):>9s} {here:11.2f} {Rh.iloc[0]['rotation_pts']:11.2f} "
                  f"{here - Rh.iloc[0]['rotation_pts']:+10.2f}")

    pd.DataFrame(all_rows).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
