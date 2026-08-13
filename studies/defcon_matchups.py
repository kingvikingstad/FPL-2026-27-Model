from __future__ import annotations
import os as _os, sys as _sys, glob as _glob
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
defcon_matchups.py — when do defenders actually hit the DefCon threshold?
=========================================================================
DefCon pays a defender 2 points for 10+ defensive actions (clearances, blocks,
interceptions, tackles, recoveries). It drives the cheap-defender strategy the board
surfaces, and `defcon_env` currently conditions it on PPDA alone.

The obvious question the model does not ask: **defensive actions are supply-driven.** You
cannot make a tackle if your opponent never has the ball. So the threshold should be
easier to hit against strong opponents and harder against weak ones — which puts DefCon in
direct TENSION with the clean sheet, because the same fixture that generates defensive
volume also concedes goals. Any cheap-defender pick is trading one against the other, and
the board should know the exchange rate.

Three questions:
  1. How does DefCon probability move with OPPONENT strength, and with the defender's OWN
     team strength?
  2. Where is the crossover — at what fixture difficulty does DefCon EV stop compensating
     for lost clean-sheet EV?
  3. Do centre-backs and full-backs behave differently? They collect different actions
     (CBs clear and block, FBs tackle and recover higher up), so the threshold may bind
     differently for each.

CB vs FB WITHOUT AN EXTERNAL JOIN
----------------------------------
FPL's position is only DEF. Rather than route through the unverified Understat crosswalk,
defenders are split on their own action PROFILE: headed clearances and aerial duels mark a
centre-back, crosses mark a full-back. The split is validated by checking the resulting
groups look like what they claim to be, and the classification is per player-season, so a
player who changes role is allowed to change group.

Data: repo playermatchstats, 24/25 + 25/26 (the seasons carrying defensive_contributions).

Run:  python studies/defcon_matchups.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import late_form_carryover as lfc

OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "defcon_matchups.csv")
DEF_THRESHOLD = 10          # defenders need 10 defensive contributions


def load():
    specs = [("2025-2026", _os.path.join("By Gameweek", "GW*", "playermatchstats.csv"),
              "players.csv", _os.path.join("By Gameweek", "GW*", "matches.csv")),
             ("2024-2025", _os.path.join("playermatchstats", "GW*", "playermatchstats.csv"),
              _os.path.join("players", "players.csv"),
              _os.path.join("matches", "GW*", "matches.csv"))]
    frames = []
    for season, gwglob, pfile, mglob in specs:
        files = sorted(_glob.glob(_os.path.join(config.repo(season), gwglob)))
        if not files:
            continue
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        d = d[d["match_id"].astype(str).str.contains("-prem-", na=False)]
        p = pd.read_csv(_os.path.join(config.repo(season), pfile))
        d = d.merge(p[[c for c in ("player_id", "player_code", "position", "team_code")
                       if c in p.columns]], on="player_id", how="left")
        mf = sorted(_glob.glob(_os.path.join(config.repo(season), mglob)))
        if mf:
            M = pd.concat([pd.read_csv(f) for f in mf], ignore_index=True)
            M = M[M["match_id"].astype(str).str.contains("-prem-", na=False)]
            keep = [c for c in ("match_id", "home_team", "away_team", "home_score",
                                "away_score", "home_expected_goals_xg",
                                "away_expected_goals_xg") if c in M.columns]
            d = d.merge(M[keep].drop_duplicates("match_id"), on="match_id", how="left")
        d["season"] = season
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def classify_defenders(d):
    """CB vs FB from the action profile, per player-season."""
    dd = d[d["position"] == "Defender"].copy()
    for c in ("headed_clearances", "accurate_crosses", "aerial_duels_won", "clearances",
              "minutes_played", "defensive_contributions"):
        dd[c] = pd.to_numeric(dd.get(c), errors="coerce").fillna(0.0)
    g = dd.groupby(["player_code", "season"], as_index=False).agg(
        mins=("minutes_played", "sum"), hc=("headed_clearances", "sum"),
        cr=("accurate_crosses", "sum"), ad=("aerial_duels_won", "sum"),
        cl=("clearances", "sum"))
    g = g[g["mins"] >= 450]
    n90 = g["mins"] / 90.0
    # per-90 rates, then a single axis: aerial/clearing work minus crossing work
    g["hc90"], g["cr90"] = g["hc"] / n90, g["cr"] / n90
    g["axis"] = (g["hc90"] + 0.5 * g["ad"] / n90) - 2.0 * g["cr90"]
    g["role"] = np.where(g["axis"] >= g["axis"].median(), "CB", "FB")
    return g[["player_code", "season", "role", "hc90", "cr90", "mins"]]


def main():
    d = load()
    print(f"[study] {len(d)} player-match rows")
    # role classification uses aerial/crossing volume, which exists in both seasons, so it
    # is fitted on everything available; only the DefCon analysis is restricted below
    roles = classify_defenders(d)
    print(f"        {len(roles)} defender-seasons with >=450 minutes classified")

    print("\n" + "=" * 74)
    print("0. DOES THE CB/FB SPLIT LOOK LIKE WHAT IT CLAIMS TO BE?")
    print("=" * 74)
    print(roles.groupby("role")[["hc90", "cr90", "mins"]].agg(
        ["count", "mean"]).round(2).to_string())
    print("\n  CB should show high headed clearances and low crosses; FB the reverse.")

    D = d[d["position"] == "Defender"].merge(roles[["player_code", "season", "role"]],
                                             on=["player_code", "season"], how="inner")
    # `defensive_contributions` is 100% NULL in 2024-25 — the stat did not exist before
    # FPL introduced DefCon scoring in 2025/26. Filling those with 0 (the first version of
    # this study did) silently makes half the sample look like defenders who never touch
    # the ball, and it produced a non-monotone opponent-strength curve that looked like a
    # finding. DROP the nulls and say which seasons survive.
    D["dc"] = pd.to_numeric(D.get("defensive_contributions"), errors="coerce")
    have = D.groupby("season")["dc"].apply(lambda s: s.notna().mean())
    print("\n  defensive_contributions coverage by season: "
          + ", ".join(f"{s} {v:.0%}" for s, v in have.items()))
    D = D[D["dc"].notna()].copy()
    if D.empty:
        raise RuntimeError("no season carries defensive_contributions")
    D["mins"] = pd.to_numeric(D["minutes_played"], errors="coerce").fillna(0.0)
    D = D[D["mins"] >= 60]
    D["hit"] = (D["dc"] >= DEF_THRESHOLD).astype(float)
    D["conceded"] = np.where(D["team_code"] == D.get("home_team"),
                             D.get("away_score"), D.get("home_score"))

    # opponent strength: season-long xG created by the opponent, from the match table
    D["is_home"] = D["team_code"] == D.get("home_team")
    D["opp"] = np.where(D["is_home"], D.get("away_team"), D.get("home_team"))
    D["opp_xg"] = pd.to_numeric(
        np.where(D["is_home"], D.get("away_expected_goals_xg"),
                 D.get("home_expected_goals_xg")), errors="coerce")
    D["own_xg"] = pd.to_numeric(
        np.where(D["is_home"], D.get("home_expected_goals_xg"),
                 D.get("away_expected_goals_xg")), errors="coerce")
    opp_strength = D.groupby(["season", "opp"])["opp_xg"].transform("mean")
    own_strength = D.groupby(["season", "team_code"])["own_xg"].transform("mean")
    D["opp_str"] = opp_strength
    D["own_str"] = own_strength
    D = D.dropna(subset=["opp_str", "own_str"])
    print(f"\n        {len(D)} defender-appearances of 60+ minutes with a matchup measure")
    print(f"        overall DefCon hit rate {D.hit.mean():.1%}")

    # ------------------------------------------------------------------ Q1
    print("\n" + "=" * 74)
    print("1. DEFCON vs OPPONENT STRENGTH — is the threshold supply-driven?")
    print("=" * 74)
    D["opp_q"] = pd.qcut(D["opp_str"], 4, labels=["weakest opp", "Q2", "Q3", "strongest opp"])
    D["own_q"] = pd.qcut(D["own_str"], 4, labels=["weakest team", "Q2", "Q3", "strongest team"])
    t = D.groupby("opp_q").agg(n=("hit", "size"), hit=("hit", "mean"),
                               dc=("dc", "mean"), cs=("conceded", lambda s: (s == 0).mean()))
    t.columns = ["n", "defcon_rate", "mean_actions", "clean_sheet_rate"]
    print(t.round(3).to_string())
    print("\n  ^ if DefCon rises and clean sheets fall across this table, the two are in")
    print("    direct tension and a cheap defender is a trade, not a free lunch.")

    print("\n  by the defender's OWN team strength:")
    t2 = D.groupby("own_q").agg(n=("hit", "size"), hit=("hit", "mean"),
                                cs=("conceded", lambda s: (s == 0).mean()))
    t2.columns = ["n", "defcon_rate", "clean_sheet_rate"]
    print(t2.round(3).to_string())

    print("\n  joint — DefCon rate by own team (rows) x opponent (cols):")
    j = D.pivot_table(index="own_q", columns="opp_q", values="hit", aggfunc="mean")
    print(j.round(3).to_string())

    # ------------------------------------------------------------------ Q2
    print("\n" + "=" * 74)
    print("2. THE EXCHANGE RATE — DefCon EV against clean-sheet EV")
    print("=" * 74)
    print(f"  {'opponent':16s} {'DefCon pts':>11s} {'CS pts':>8s} {'total':>8s}")
    for q in ["weakest opp", "Q2", "Q3", "strongest opp"]:
        s = D[D["opp_q"] == q]
        dc_pts = s["hit"].mean() * 2.0
        cs_pts = (s["conceded"] == 0).mean() * 4.0
        print(f"  {q:16s} {dc_pts:11.3f} {cs_pts:8.3f} {dc_pts+cs_pts:8.3f}")
    print("\n  DefCon pays 2, a defender's clean sheet pays 4 — so clean sheets dominate")
    print("  unless the DefCon rate difference is large. This is the number the")
    print("  cheap-defender strategy actually rests on.")

    # ------------------------------------------------------------------ Q3
    print("\n" + "=" * 74)
    print("3. CENTRE-BACKS vs FULL-BACKS")
    print("=" * 74)
    print(f"  {'role':5s} {'n':>7s} {'DefCon rate':>12s} {'mean actions':>13s} "
          f"{'CS rate':>8s}")
    for r in ("CB", "FB"):
        s = D[D["role"] == r]
        print(f"  {r:5s} {len(s):7d} {s['hit'].mean():12.3f} {s['dc'].mean():13.2f} "
              f"{(s['conceded'] == 0).mean():8.3f}")
    print("\n  DefCon rate by role and opponent strength:")
    print(D.pivot_table(index="role", columns="opp_q", values="hit",
                        aggfunc="mean").round(3).to_string())
    print("\n  mean defensive actions by role and opponent strength:")
    print(D.pivot_table(index="role", columns="opp_q", values="dc",
                        aggfunc="mean").round(2).to_string())

    # is the role difference significant, clustered by player?
    rng = np.random.default_rng(0)
    codes = D["player_code"].dropna().unique()
    obs = D[D.role == "CB"]["hit"].mean() - D[D.role == "FB"]["hit"].mean()
    bs = []
    for _ in range(2000):
        pick = rng.choice(codes, len(codes), replace=True)
        s = D[D["player_code"].isin(pick)]
        if s.role.nunique() == 2:
            bs.append(s[s.role == "CB"]["hit"].mean() - s[s.role == "FB"]["hit"].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n  CB minus FB DefCon rate: {obs:+.3f}  95% CI ({lo:+.3f}, {hi:+.3f})"
          f"{'  *' if not (lo <= 0 <= hi) else '  (not significant)'}")

    D[["season", "player_code", "role", "dc", "hit", "mins", "opp_str", "own_str",
       "conceded", "is_home"]].to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
