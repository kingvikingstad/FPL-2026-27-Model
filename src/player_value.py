from __future__ import annotations
import config
"""
player_value.py — value above replacement, per player per gameweek.
====================================================================
A solver needs points and price as separate inputs; it does NOT need a ratio. What a
ratio is for is RANKING and PRUNING — deciding which few hundred of 584 players are worth
handing to the optimiser, and understanding why one is better value than another. This
module builds that ranking correctly, which means avoiding two mistakes.

MISTAKE 1: points / price
-------------------------
You cannot spend zero on a goalkeeper. The squad is 2 GK / 5 DEF / 5 MID / 3 FWD, so
every slot must be filled, and the cheapest fillable price at each position is a cost you
pay no matter who you pick. Only the money ABOVE that floor is a real choice, and only the
points ABOVE what the floor buys are a real gain. So:

    marginal_cost = price - floor_price[pos]
    par           = ep    - replacement_ep[pos]        ("points above replacement")
    value         = par / marginal_cost

`replacement_ep` is the BEST expected points available AT the floor price, not the worst.
The opportunity cost of not buying a player is what you would otherwise field in that
slot, and nobody deliberately fields the worst available option.

This reverses conclusions. On raw points-per-million a premium looks efficient because his
price divides into a big number. On value above replacement he is often not, because the
first ~£4.0m of his price bought nothing a free slot would not have given you anyway.

MISTAKE 2: doubling everybody for captaincy
--------------------------------------------
Captaincy doubles ONE player per gameweek. Multiplying every player's points by two and
ranking on that changes nothing (it is a monotone transform of the same ordering) while
quietly implying the whole squad is captained. The captaincy columns here are explicitly
CONDITIONAL — `ep_cap` is what this player is worth IF he wears the armband — and the
armband constraint belongs in the solver:

    maximise   sum_i  x_i * ep_i  +  sum_i  c_i * ep_i
    subject to sum_i  c_i = 1,   c_i <= x_i,   x_i in {0,1}

`c_i` adds a second copy of that player's points. Feeding a pre-doubled `ep` into `x_i`
would double the entire XI. `cap_par` and `cap_value` are therefore upper bounds, useful
for asking "which player most repays the armband", never for summing across a squad.

THE RATIO IS NOISIEST EXACTLY WHERE IT IS LARGEST  [read before ranking on it]
-------------------------------------------------------------------------------
`value` divides by `marginal_cost`, so a player £0.5m above the floor has his sampling
error multiplied by two, while a player £11m above it has it divided by eleven. The
variance of `value` is therefore inversely proportional to marginal cost, and the top of
any `value` ranking is populated by near-floor players whose position there is partly
noise.

Measured on this board: Shaw (£4.5m, £0.5m above floor) scores value +3.22 in GW1 — the
best defender in the league — and −1.02 the following week, averaging −0.13 par across
GW1-10. He is not a good pick that became bad; he is a player sitting AT replacement
level whose ratio is being amplified 2x in both directions. Haaland, £11m above the
floor, moves between 0.36 and 0.56 all season.

Practical consequence: use `value` to PRUNE (drop everyone below replacement, keep the
plausible field) and `par` to COMPARE near-floor players, because `par` is on the points
scale and is not divided by anything. Never hand a solver a shortlist chosen on `value`
alone from the near-floor band.

WHY THE TAIL COLUMNS ARE HERE
------------------------------
Captaincy is a tail problem, not a mean problem (`captaincy.py`, `edge_study.py`): you are
doubling, so the spread matters as much as the centre. `ep_cap` maximises expected score,
but two players with equal means and different `sd` are not equally good armbands, so
`p95` and `sd` travel alongside rather than being collapsed away.

Run:  python src/player_value.py --selftest
"""
import numpy as np
import pandas as pd

# FPL squad composition and budget (docs/FPL_RULES.md)
SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET = 100.0
# A slot is only "fillable" by someone who might actually play. Without this the floor
# price is set by a fourth-choice keeper who will never appear, which makes every
# replacement level ~0 points and every player look like brilliant value.
MIN_START_PROB = 0.10


def position_floors(players, price_col="cost", pos_col="pos", ep_col="ep",
                    start_col=None, min_start=MIN_START_PROB):
    """floor price and replacement EP for each position.

    floor_price     cheapest price at which the position can be filled by a player who
                    might play
    replacement_ep  the BEST expected points available at (or below) that floor price —
                    the opportunity cost of the slot
    """
    p = players.copy()
    if start_col and start_col in p.columns:
        elig = p[p[start_col].fillna(0) >= min_start]
        if elig.empty:
            elig = p
    else:
        elig = p
    rows = []
    for pos, g in elig.groupby(pos_col):
        floor = float(g[price_col].min())
        at_floor = g[g[price_col] <= floor + 1e-9]
        rows.append({"pos": pos, "floor_price": floor,
                     "replacement_ep": float(at_floor[ep_col].max()),
                     "n_at_floor": int(len(at_floor)),
                     "replacement_player": (at_floor.loc[at_floor[ep_col].idxmax()]
                                            .get("display_name",
                                                 at_floor.loc[at_floor[ep_col].idxmax()]
                                                 .get("player", "?"))
                                            if len(at_floor) else "?")})
    return pd.DataFrame(rows)


def committed_spend(floors, squad=None, budget=BUDGET):
    """The money you cannot choose how to spend, and what is left over.

    Filling 15 slots at the floor price is unavoidable, so that portion of the £100m is
    not a decision. `discretionary` is the budget the solver is actually allocating.
    """
    squad = squad or SQUAD
    f = floors.set_index("pos")["floor_price"].to_dict()
    committed = sum(n * f.get(pos, 0.0) for pos, n in squad.items())
    return {"committed": float(committed),
            "discretionary": float(budget - committed),
            "per_position": {pos: float(n * f.get(pos, 0.0))
                             for pos, n in squad.items()}}


def add_value(players, ep_col="ep", price_col="cost", pos_col="pos",
              start_col=None, floors=None, eps=0.05):
    """Attach the value construct. One row per player (per gameweek, if the frame is
    already filtered to one).

    Columns added
      floor_price       position floor
      replacement_ep    best EP obtainable at the floor
      marginal_cost     price - floor_price   (0.0 for a floor-priced player)
      par               ep - replacement_ep
      value             par / marginal_cost           [points per marginal £m]
      ep_cap            2 * ep                        [IF captained — conditional]
      cap_par           2 * ep - replacement_ep
      cap_value         cap_par / marginal_cost
      is_replacement    True where this player IS the floor option
      below_replacement True where par < 0 — `value` is a do-not-buy signal, and its
                        magnitude is NOT a ranking
      captain_candidate top-decile ep; `cap_value` is only populated for these
      value_rank_pos    rank of `value` within position among positive-par players

    Floor-priced players have marginal_cost == 0, so `value` is undefined rather than
    infinite: they are the baseline the ratio is measured against, and dividing by zero
    would put them at the top of every ranking by construction. `par` still carries their
    (usually small, sometimes positive) edge over the specific replacement player, and
    `is_replacement` flags them so a solver can treat them as free slot-fillers.
    """
    p = players.copy()
    if floors is None:
        floors = position_floors(p, price_col, pos_col, ep_col, start_col)
    p = p.merge(floors[["pos", "floor_price", "replacement_ep"]],
                left_on=pos_col, right_on="pos", how="left",
                suffixes=("", "_f"))

    p["marginal_cost"] = (p[price_col] - p["floor_price"]).clip(lower=0.0)
    p["par"] = p[ep_col] - p["replacement_ep"]
    p["ep_cap"] = 2.0 * p[ep_col]
    p["cap_par"] = p["ep_cap"] - p["replacement_ep"]
    p["is_replacement"] = p["marginal_cost"] <= 1e-9

    mc = p["marginal_cost"].where(p["marginal_cost"] > eps, np.nan)
    p["value"] = p["par"] / mc
    p["below_replacement"] = p["par"] < 0

    # A ratio is only interpretable when the numerator is positive. For a below-
    # replacement player `value` is negative and its MAGNITUDE is meaningless — it says
    # "worse than a free slot", which is a valid do-not-buy signal but not a ranking.
    # Ranking is therefore computed on positive-par players only.
    pos_only = p["value"].where(~p["below_replacement"])
    p["value_rank_pos"] = pos_only.groupby(p[pos_col]).rank(ascending=False,
                                                            method="min")

    # Captaincy is not a budget decision in the way squad construction is — you captain
    # someone you already own — so `cap_value` answers only the narrow question "if I am
    # spending up specifically to own a captain, who repays the marginal pound most".
    # For a cheap player 2*ep is still small, and dividing by a tiny marginal cost
    # inflates it into nonsense (a £4.5m keeper topping the list). Restricted to players
    # anyone would actually consider for the armband.
    thresh = p[ep_col].quantile(0.90)
    p["captain_candidate"] = p[ep_col] >= thresh
    p["cap_value"] = (p["cap_par"] / mc).where(p["captain_candidate"])
    return p


def per_gameweek(board, ep_col="mean", price_col="cost", pos_col="pos",
                 gw_col="gw", start_col=None):
    """Apply the construct independently WITHIN each gameweek.

    Floors are recomputed per gameweek on purpose. A cheap defender whose team has a
    blank, or who is flagged out, stops being a viable replacement that week, which
    raises the effective floor and changes every value in the position. A single
    season-long floor would hide exactly the week-to-week variation a per-gameweek
    solver exists to exploit.
    """
    out = []
    for gw, g in board.groupby(gw_col):
        f = position_floors(g, price_col, pos_col, ep_col, start_col)
        v = add_value(g, ep_col, price_col, pos_col, start_col, floors=f)
        v[gw_col] = gw
        out.append(v)
    return pd.concat(out, ignore_index=True)


def solver_inputs(valued, gw, ep_col="mean", price_col="cost"):
    """The minimal frame an optimiser needs for one gameweek, plus the budget split.

    Returns (frame, meta). The frame carries raw `ep` and `price` — NOT a ratio and NOT a
    doubled ep — because the optimiser must apply the captaincy constraint itself.
    """
    g = valued[valued["gw"] == gw].copy()
    floors = g.drop_duplicates("pos")[["pos", "floor_price", "replacement_ep"]]
    meta = committed_spend(floors)
    meta["gw"] = gw
    cols = [c for c in ["player_code", "display_name", "unique_label", "player", "team",
                        "pos", price_col, ep_col, "sd", "p95", "par", "value",
                        "marginal_cost", "is_replacement"] if c in g.columns]
    return g[cols].rename(columns={ep_col: "ep", price_col: "price"}), meta


def selftest():
    df = pd.DataFrame({
        "player": ["cheapGK", "goodGK", "cheapDEF", "midDEF", "premDEF", "benchDEF"],
        "pos": ["GK", "GK", "DEF", "DEF", "DEF", "DEF"],
        "cost": [4.0, 5.5, 4.0, 4.5, 8.0, 4.0],
        "ep": [2.0, 4.0, 1.0, 3.0, 7.0, 0.2],
        "p_start": [0.9, 0.9, 0.9, 0.9, 0.9, 0.02],
    })
    f = position_floors(df, ep_col="ep", start_col="p_start")
    gk = f[f.pos == "GK"].iloc[0]
    de = f[f.pos == "DEF"].iloc[0]
    assert gk["floor_price"] == 4.0 and gk["replacement_ep"] == 2.0
    # benchDEF is cheaper-or-equal but filtered out by start prob, so replacement is the
    # BEST £4.0 defender who might play, not the worst
    assert de["floor_price"] == 4.0, de["floor_price"]
    assert de["replacement_ep"] == 1.0, f"replacement should be best at floor: {de['replacement_ep']}"

    v = add_value(df, ep_col="ep", start_col="p_start").set_index("player")
    assert abs(v.loc["premDEF", "par"] - 6.0) < 1e-9
    assert abs(v.loc["premDEF", "marginal_cost"] - 4.0) < 1e-9
    assert abs(v.loc["premDEF", "value"] - 1.5) < 1e-9
    assert abs(v.loc["midDEF", "value"] - 4.0) < 1e-9, v.loc["midDEF", "value"]
    # the cheap enabler beats the premium on value above replacement, which raw
    # points-per-million gets backwards
    assert v.loc["midDEF", "value"] > v.loc["premDEF", "value"]
    assert (df.set_index("player").loc["premDEF", "ep"] / 8.0 >
            df.set_index("player").loc["midDEF", "ep"] / 4.5), "ppm should disagree"
    # floor-priced players are not infinitely valuable
    assert bool(v.loc["cheapDEF", "is_replacement"]) and pd.isna(v.loc["cheapDEF", "value"])
    # captaincy is conditional and additive on ep only
    assert abs(v.loc["premDEF", "ep_cap"] - 14.0) < 1e-9
    assert abs(v.loc["premDEF", "cap_par"] - 13.0) < 1e-9
    # below-replacement players are flagged and excluded from the value ranking
    assert bool(v.loc["benchDEF", "below_replacement"]), "benchDEF is below replacement"
    assert pd.isna(v.loc["benchDEF", "value_rank_pos"]), \
        "below-replacement players must not be ranked"
    # cap_value only populated for plausible armbands
    assert bool(v.loc["premDEF", "captain_candidate"])
    assert pd.isna(v.loc["cheapDEF", "cap_value"]), \
        "cap_value must not be computed for a player nobody would captain"

    meta = committed_spend(f)
    # 2 GK * 4.0 + 5 DEF * 4.0 = 28.0 of the 100 is not a decision
    assert abs(meta["per_position"]["GK"] - 8.0) < 1e-9
    assert abs(meta["per_position"]["DEF"] - 20.0) < 1e-9

    b = pd.concat([df.assign(gw=1), df.assign(gw=2, ep=df.ep * 0.5)])
    pg = per_gameweek(b, ep_col="ep", start_col="p_start")
    assert set(pg.gw) == {1, 2} and len(pg) == 12
    si, m = solver_inputs(pg, gw=1, ep_col="ep")
    assert "ep" in si.columns and "price" in si.columns and m["gw"] == 1

    print("SELFTEST OK: floors set by the best playable option at the cheapest price, "
          "value measured above replacement per marginal £m, cheap enabler correctly "
          "beats the premium, floor players not infinite, captaincy kept conditional.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    print(__doc__)
