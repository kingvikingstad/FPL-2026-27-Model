from __future__ import annotations
import os as _os, sys as _sys, glob as _glob
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
penalty_assignment.py — is the declared penalty order worth trusting?
=====================================================================
The model computes `pen_xg90_measured` in multiseason_priors.to_priors and then THROWS IT
AWAY. Both runners instead use FPL's declared order:

    gw_board.py / decision_v2.py:   pen_order == 1  ->  clip(pen_xg90, lower=0.10)

Penalties are the single largest per-player lever in the game. Understat prices one at
0.7612 xG and they convert at 83%; a settled taker gets 5-10 a season, so getting the
assignment right is worth several goals — more than most players' entire non-penalty
output. If the declared order is unreliable, or if measured history beats it, that is a
large and cheap gain sitting in code the project already wrote.

THREE QUESTIONS
---------------
1. PERSISTENCE. How much signal is in measured history at all? Does last season's taker
   take them again? Understat shots, 12 seasons, no crosswalk needed — both sides of the
   comparison are Understat.
2. HORSE RACE. Declared order at the start of a season versus measured attempts from the
   previous season, predicting who actually takes them. Both live in player_code space —
   the repo's playermatchstats carries penalties_scored/missed keyed on player_id, and
   vaastav's players_raw carries penalties_order keyed on `code`, which IS player_code —
   so this needs no fuzzy matching either.
3. WHAT IS AT STAKE. Convert to FPL points so the size of the prize is explicit.

Run:  python studies/penalty_assignment.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest
import fpl_history as fh

SEASONS = ["%02d%02d" % (y % 100, (y + 1) % 100) for y in range(2014, 2026)]
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "penalty_assignment.csv")
PEN_XG = 0.7612          # Understat's constant for a penalty


# ------------------------------------------------------------------ 1. persistence
def understat_penalties():
    sh = sd_ingest.understat_shots(SEASONS)
    pen = sh[sh["situation"] == "Penalty"].copy()
    pen["team"] = pen["team"].map(sd_ingest.normalise_team)
    return (pen.groupby(["season", "team", "understat_player_id", "player_name"],
                        as_index=False).size().rename(columns={"size": "pens"}))


def persistence(P):
    """For each club-season, who took the most penalties last season, and did they take
    the most this season? Restricted to clubs with >= 3 penalties in BOTH seasons so the
    question is well posed."""
    order = SEASONS
    rows = []
    for i in range(1, len(order)):
        p, c = order[i - 1], order[i]
        for team, g in P[P["season"] == p].groupby("team"):
            cur = P[(P["season"] == c) & (P["team"] == team)]
            if g["pens"].sum() < 3 or cur["pens"].sum() < 3:
                continue
            prev_top = g.sort_values("pens", ascending=False).iloc[0]
            cur_top = cur.sort_values("pens", ascending=False).iloc[0]
            stayed = prev_top["understat_player_id"] in set(cur["understat_player_id"])
            took_share = 0.0
            if stayed:
                took_share = float(
                    cur.loc[cur["understat_player_id"] == prev_top["understat_player_id"],
                            "pens"].sum() / cur["pens"].sum())
            rows.append({
                "season": c, "team": team,
                "prev_top_id": prev_top["understat_player_id"],
                "prev_top_name": prev_top["player_name"],
                "prev_top_pens": int(prev_top["pens"]),
                "prev_share": float(prev_top["pens"] / g["pens"].sum()),
                "still_at_club": bool(stayed),
                "share_this_season": took_share,
                "retained_role": bool(stayed and cur_top["understat_player_id"]
                                      == prev_top["understat_player_id"]),
                "club_pens_this": int(cur["pens"].sum()),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 2. horse race
def repo_penalties(season_dir, gw_glob):
    files = sorted(_glob.glob(_os.path.join(config.repo(season_dir), gw_glob)))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "match_id" in d.columns:
        d = d[d["match_id"].astype(str).str.contains("-prem-", na=False)]
    for c in ("penalties_scored", "penalties_missed"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce").fillna(0.0)
    d["pen_att"] = d["penalties_scored"] + d["penalties_missed"]
    players = pd.read_csv(_os.path.join(config.repo(season_dir), "players.csv")) \
        if _os.path.exists(_os.path.join(config.repo(season_dir), "players.csv")) \
        else pd.read_csv(_os.path.join(config.repo(season_dir), "players", "players.csv"))
    d = d.merge(players[["player_id", "player_code"]], on="player_id", how="left")
    return d.groupby("player_code", as_index=False)["pen_att"].sum()


def declared_order(season):
    raw = fh._read_csv(_os.path.join(config.history(season), "players_raw.csv"))
    if "penalties_order" not in raw.columns:
        return None
    r = raw[["code", "penalties_order"]].rename(columns={"code": "player_code"})
    return r[r["penalties_order"].notna()]


def main():
    print("=" * 78)
    print("1. PERSISTENCE — does last season's penalty taker take them again?")
    print("=" * 78)
    P = understat_penalties()
    print(f"  {P.pens.sum():.0f} penalties across {P.season.nunique()} seasons, "
          f"{P.understat_player_id.nunique()} distinct takers")
    R = persistence(P)
    print(f"  {len(R)} club-season transitions with >=3 penalties on both sides\n")
    print(f"  previous top taker still at the club : {R.still_at_club.mean():6.1%}")
    stayed = R[R.still_at_club]
    print(f"  ... and still the top taker          : {stayed.retained_role.mean():6.1%} "
          f"(of those who stayed, n={len(stayed)})")
    print(f"  ... their share of the club's pens   : {stayed.share_this_season.mean():6.1%} "
          f"(was {stayed.prev_share.mean():.1%} the previous season)")
    print(f"  unconditional: previous top taker takes "
          f"{R.share_this_season.mean():.1%} of this season's penalties")
    print("\n  Penalty duty is a ROLE, and the question is whether it survives the summer.")

    # concentration: how lumpy is it?
    conc = (P.groupby(["season", "team"])
             .apply(lambda g: g["pens"].max() / g["pens"].sum())
             .rename("top_share").reset_index())
    print(f"\n  concentration: the top taker takes {conc.top_share.mean():.1%} of his "
          f"club's penalties (median {conc.top_share.median():.1%})")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. HORSE RACE — declared order vs measured history, predicting 25/26")
    print("=" * 78)
    prev = repo_penalties("2024-2025", _os.path.join("playermatchstats", "GW*",
                                                     "playermatchstats.csv"))
    cur = repo_penalties("2025-2026", _os.path.join("By Gameweek", "GW*",
                                                    "playermatchstats.csv"))
    dec = declared_order("2025-26")
    if prev is None or cur is None or dec is None:
        print("  repo or declared-order data unavailable — skipped")
        return
    m = cur.rename(columns={"pen_att": "pen_now"}).merge(
        prev.rename(columns={"pen_att": "pen_prev"}), on="player_code", how="outer")
    m = m.merge(dec, on="player_code", how="left").fillna(
        {"pen_now": 0, "pen_prev": 0})
    m["declared_first"] = (m["penalties_order"] == 1).astype(float)
    m["prev_taker"] = (m["pen_prev"] >= 2).astype(float)
    took = m[m["pen_now"] > 0]
    print(f"  {int(m.pen_now.sum())} penalties taken in 25/26 by {len(took)} players")
    print(f"  players flagged penalties_order == 1 : {int(m.declared_first.sum())}")
    print(f"  players with >=2 attempts in 24/25   : {int(m.prev_taker.sum())}")

    def score(flag, label):
        tp = float(((m[flag] == 1) & (m["pen_now"] > 0)).sum())
        fp = float(((m[flag] == 1) & (m["pen_now"] == 0)).sum())
        fn = float(((m[flag] == 0) & (m["pen_now"] > 0)).sum())
        cov = m.loc[m[flag] == 1, "pen_now"].sum() / max(m["pen_now"].sum(), 1)
        prec = tp / max(tp + fp, 1)
        print(f"  {label:28s} precision {prec:5.1%}  flagged {int(tp+fp):3d}  "
              f"missed takers {int(fn):3d}  covers {cov:5.1%} of penalties actually taken")
        return cov

    print()
    c1 = score("declared_first", "declared order == 1")
    c2 = score("prev_taker", "took >=2 pens last season")
    m["either"] = ((m["declared_first"] == 1) | (m["prev_taker"] == 1)).astype(float)
    c3 = score("either", "either signal")
    print("\n  'covers' is the share of actual 25/26 penalties taken by flagged players —")
    print("  the number that matters, because a missed taker is a missed 0.76 xG per pen.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("3. WHAT IS AT STAKE")
    print("=" * 78)
    pens_per_team = P.groupby(["season", "team"])["pens"].sum().mean()
    print(f"  penalties per club-season: {pens_per_team:.1f}")
    print(f"  each is {PEN_XG:.3f} xG, converting at ~83%")
    print(f"  a settled taker on a average club is worth ~{pens_per_team*0.83:.1f} goals")
    print(f"  at 4 pts/goal for a midfielder that is ~{pens_per_team*0.83*4:.0f} FPL points")
    print("  a season, before assists or bonus — larger than most players' entire")
    print("  non-penalty return. Misassigning it is one of the costliest single errors")
    print("  the projection can make.")

    m.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
