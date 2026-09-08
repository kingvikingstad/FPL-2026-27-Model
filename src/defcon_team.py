from __future__ import annotations
import config
"""
defcon_team.py — the TEAM layer of DefCon: who concedes defensive actions, and to whom.
========================================================================================
`defcon_env` already conditions a player's DefCon rate on his OWN team's environment
(xGA for DEF, press for MID/FWD). Nothing conditions on the OPPONENT, and
`defcon_matchups` found the DefCon rate flat across opponent STRENGTH quartiles, which
was read as "opponent does not matter".

That reading confuses a scalar with an identity. Strength is one number; the amount of
defending an opponent forces is a different quantity, and two equally strong sides can
differ sharply in it. Measured on 25/26, they do: facing the most permissive opponent
versus the most restrictive moves a defender's DefCon hit rate by about 0.22 raw, and
the ordering is not the strength ordering — Liverpool sit near the top, Chelsea near the
bottom.

WHAT IS AND IS NOT IDENTIFIED  [JUDGMENT]
------------------------------------------
OPPONENT effects ARE identified. Every player faces many different opponents inside one
season, so the opponent term is estimated WITHIN player and is not confounded with who
happens to be on the pitch. That is the quantity this module ships.

OWN-TEAM effects are NOT identified within a single season. A player belongs to exactly
one club, so club and player are nested and demeaning by player annihilates the club
term. A club's raw DefCon mean is highly repeatable (split-half r ~ 0.70-0.88) but that
number is mostly "does this club field the same players", not "this club generates
defensive actions". `own_team_profile()` therefore returns a DESCRIPTIVE profile,
explicitly labelled composition-confounded, and nothing in the model consumes it. The
structural route — conditioning on the mechanism (xGA, press) rather than on a club
label — is what `defcon_env` already does, and it remains the right approach.

SHRINKAGE, NOT RAW MEANS
------------------------
A raw per-opponent mean over ~150 defender-appearances carries real sampling noise.
Ratings are shrunk empirical-Bayes toward zero by tau2/(tau2+se2), where tau2 is the
between-opponent variance net of mean sampling variance. On 25/26 that factor is ~0.7
for DEF, which independently reproduces the split-half reliability — a good sign the
noise model is right.

RELIABILITY GATE
----------------
Per position, a rating is only marked usable if its split-half reliability clears
`MIN_RELIABILITY`. On 25/26 DEF passes and MID does not, so the MID ratings are computed
and returned but flagged `usable=False`. Do not consume a rating whose `usable` is False.

Keyed on `player_code` throughout. Club attribution comes from the per-match lineup
sheet, never from the `players.csv` snapshot, which carries a player's CURRENT club and
would misattribute anyone who transferred mid-season.
"""
import glob
import os
import numpy as np
import pandas as pd

# FPL DefCon thresholds. GK cannot score DefCon.
THRESHOLD = {"Defender": 10, "Midfielder": 12, "Forward": 12}
MIN_MINUTES = 60          # a DefCon-eligible appearance
MIN_RELIABILITY = 0.5     # split-half r required before a rating is `usable`
CATEGORIES = ("suppressing", "neutral", "permissive")


def _norm(name):
    """Core-Insights club naming -> the model frame's naming.

    25/26 `teams.csv` says "Man Utd" and "Spurs"; the 26/27 frame the board is built on
    says "Man United" and "Tottenham". Joining without this silently drops exactly those
    two clubs to NaN, which reads as "no rating available" rather than as a bug. Route
    through the repo's one normaliser so the failure cannot happen quietly.
    """
    import core_insights as ci
    return ci.norm_team(name)


def _match_table(season, base=None):
    root = base or config.repo(season)
    fs = sorted(glob.glob(os.path.join(root, "By Gameweek", "GW*", "matches.csv")))
    if not fs:
        return pd.DataFrame()
    m = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    m = m[m["match_id"].astype(str).str.contains("-prem-", na=False)]
    m = m.drop_duplicates("match_id")
    t = pd.read_csv(os.path.join(root, "teams.csv"))
    # home_team/away_team hold the team CODE, not the id — see repo_events
    c2n = dict(zip(pd.to_numeric(t["code"], errors="coerce"), t["name"]))
    m["home"] = pd.to_numeric(m["home_team"], errors="coerce").map(c2n).map(_norm)
    m["away"] = pd.to_numeric(m["away_team"], errors="coerce").map(c2n).map(_norm)
    return m


def build_panel(season="2025-2026", base=None):
    """One row per DefCon-eligible player-match: who, for which club, against whom,
    how many defensive actions, and whether the positional threshold was hit.

    `defensive_contributions` is 100% NULL before 25/26 — the stat did not exist until
    FPL introduced DefCon scoring. Nulls are DROPPED, never filled with zero: filling
    makes a player who was simply not measured look like one who never touched the ball.
    """
    root = base or config.repo(season)
    fs = sorted(glob.glob(os.path.join(root, "By Gameweek", "GW*",
                                       "playermatchstats.csv")))
    if not fs:
        return pd.DataFrame()
    rows = []
    for f in fs:
        d = pd.read_csv(f)
        pf = os.path.join(os.path.dirname(f), "players.csv")
        if os.path.exists(pf):
            p = pd.read_csv(pf)
            cols = [c for c in ("player_id", "player_code", "position") if c in p.columns]
            d = d.merge(p[cols], on="player_id", how="left")
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d = d[d["match_id"].astype(str).str.contains("-prem-", na=False)]

    d["dc"] = pd.to_numeric(d.get("defensive_contributions"), errors="coerce")
    d["mins"] = pd.to_numeric(d.get("minutes_played"), errors="coerce")
    d = d[d["dc"].notna() & (d["mins"] >= MIN_MINUTES)]
    if d.empty:
        return pd.DataFrame()

    # club at match time, from the lineup sheet
    import repo_events as re_
    sheets = re_.squad_sheets(season, base=root)
    sheets = sheets[sheets["tournament"] == "prem"][["player_code", "match_id", "club"]]
    sheets = sheets.copy()
    sheets["club"] = sheets["club"].map(_norm)
    d = d.merge(sheets.drop_duplicates(["player_code", "match_id"]),
                on=["player_code", "match_id"], how="left")

    m = _match_table(season, base=root)
    d = d.merge(m[["match_id", "home", "away", "gameweek"]], on="match_id", how="left")
    d["opp"] = np.where(d["club"] == d["home"], d["away"], d["home"])
    d["is_home"] = d["club"] == d["home"]
    d = d.dropna(subset=["club", "opp", "player_code"])

    d["thr"] = d["position"].map(THRESHOLD)
    d = d[d["thr"].notna()].copy()
    d["hit"] = (d["dc"] >= d["thr"]).astype(float)
    d["season"] = season
    return d[["season", "gameweek", "match_id", "player_code", "position", "club",
              "opp", "is_home", "mins", "dc", "thr", "hit"]].reset_index(drop=True)


def _split_half(panel, key, value):
    """Reliability of a per-`key` effect across the two halves of the season."""
    med = panel["gameweek"].median()
    a = panel[panel["gameweek"] <= med].groupby(key)[value].mean()
    b = panel[panel["gameweek"] > med].groupby(key)[value].mean()
    j = pd.concat([a.rename("h1"), b.rename("h2")], axis=1).dropna()
    if len(j) < 5:
        return np.nan
    return float(j["h1"].corr(j["h2"]))


def opponent_defcon_ratings(panel, position="Defender"):
    """How many defensive actions each club concedes to the opposition, WITHIN player.

    Demeaning by player is what makes this an opponent effect rather than a statement
    about which players happened to face that club. Returns one row per opponent with
    the raw and empirical-Bayes-shrunk effect in ACTIONS, the effect on hit
    probability, a tercile category, and the reliability that gates use.
    """
    s = panel[panel["position"] == position].copy()
    if s.empty:
        return pd.DataFrame()
    s["dc_dm"] = s["dc"] - s.groupby("player_code")["dc"].transform("mean")
    s["hit_dm"] = s["hit"] - s.groupby("player_code")["hit"].transform("mean")

    g = s.groupby("opp").agg(n=("dc_dm", "size"), raw=("dc_dm", "mean"),
                             sd=("dc_dm", "std"), hit_raw=("hit_dm", "mean"))
    # empirical Bayes: between-opponent variance net of mean sampling variance
    se2 = (g["sd"] ** 2 / g["n"]).mean()
    tau2 = max(float(g["raw"].var(ddof=1) - se2), 0.0)
    k = tau2 / (tau2 + se2) if (tau2 + se2) > 0 else 0.0
    g["shrunk"] = g["raw"] * k
    g["hit_shrunk"] = g["hit_raw"] * k

    rel = _split_half(s.assign(_v=s["dc_dm"]), "opp", "_v")
    g["reliability"] = rel
    g["shrink_factor"] = k
    g["usable"] = bool(np.isfinite(rel) and rel >= MIN_RELIABILITY)
    g["position"] = position

    # Tercile category on the shrunk effect: more actions conceded = permissive.
    # When there is no signal, tau2 collapses to zero, every shrunk effect is zero and
    # the terciles are degenerate. That is the correct outcome, not an error — label
    # everything neutral rather than manufacturing a ranking out of nothing.
    if len(g) >= 3 and g["shrunk"].nunique() >= 3:
        g["category"] = pd.qcut(g["shrunk"], 3, labels=list(CATEGORIES),
                                duplicates="drop")
    else:
        g["category"] = "neutral"
    return g.reset_index().rename(columns={"opp": "team"}).sort_values(
        "shrunk", ascending=False).reset_index(drop=True)


def own_team_profile(panel, position="Defender"):
    """DESCRIPTIVE only. A club's own DefCon output is NOT separable from the players
    it fields — within one season club and player are nested, so this cannot be read
    as a team effect and nothing in the model consumes it. Kept because it is the
    natural thing to look at, and because saying why it is not usable is the point.
    """
    s = panel[panel["position"] == position]
    if s.empty:
        return pd.DataFrame()
    g = s.groupby("club").agg(n=("dc", "size"), mean_actions=("dc", "mean"),
                              hit_rate=("hit", "mean")).reset_index()
    g["reliability_raw"] = _split_half(s, "club", "dc")
    g["identified"] = False
    g["caveat"] = "club and player are nested within a season; composition-confounded"
    return g.sort_values("mean_actions", ascending=False).reset_index(drop=True)


def ratings_for_board(season="2025-2026", base=None, positions=("Defender",)):
    """The shippable artefact: per-team opponent DefCon category, usable positions only.

    Returns [team, position, shrunk, hit_shrunk, category, reliability, usable, n].
    A caller should filter on `usable` — the frame carries unusable rows so the reason
    is visible rather than silently missing.
    """
    panel = build_panel(season, base=base)
    if panel.empty:
        return pd.DataFrame(columns=["team", "position", "shrunk", "hit_shrunk",
                                     "category", "reliability", "usable", "n"])
    out = [opponent_defcon_ratings(panel, p) for p in positions]
    out = [o for o in out if not o.empty]
    if not out:
        return pd.DataFrame()
    cols = ["team", "position", "n", "raw", "shrunk", "hit_raw", "hit_shrunk",
            "category", "reliability", "shrink_factor", "usable"]
    return pd.concat(out, ignore_index=True)[cols]


def selftest():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(20)]
    # planted: opponent effect on actions, players differ in their own baseline
    opp_eff = {t: v for t, v in zip(teams, np.linspace(-1.5, 1.5, 20))}
    rows = []
    for p in range(120):
        club = teams[p % 20]
        base_rate = rng.normal(8, 2.0)
        for gw in range(1, 39):
            opp = teams[(p * 7 + gw) % 20]
            if opp == club:
                continue
            dc = base_rate + opp_eff[opp] + rng.normal(0, 2.0)
            rows.append(dict(season="S", gameweek=gw, match_id=f"m{p}_{gw}",
                             player_code=p, position="Defender", club=club, opp=opp,
                             is_home=True, mins=90, dc=max(dc, 0), thr=10,
                             hit=float(dc >= 10)))
    panel = pd.DataFrame(rows)

    r = opponent_defcon_ratings(panel, "Defender")
    assert len(r) == 20, len(r)
    truth = pd.Series(opp_eff)
    got = r.set_index("team")["shrunk"].reindex(truth.index)
    corr = float(np.corrcoef(truth.values, got.values)[0, 1])
    assert corr > 0.9, f"planted opponent effect not recovered: r={corr:.3f}"
    assert 0.0 <= r["shrink_factor"].iloc[0] <= 1.0
    assert r["reliability"].iloc[0] > MIN_RELIABILITY, r["reliability"].iloc[0]
    assert bool(r["usable"].iloc[0]) is True
    # the shrunk effect must never exceed the raw one in magnitude
    assert (r["shrunk"].abs() <= r["raw"].abs() + 1e-9).all()

    # a planted NULL must shrink to ~nothing and fail the reliability gate
    rows2 = []
    for p in range(120):
        club = teams[p % 20]
        base_rate = rng.normal(8, 2.0)
        for gw in range(1, 39):
            opp = teams[(p * 7 + gw) % 20]
            if opp == club:
                continue
            rows2.append(dict(season="S", gameweek=gw, match_id=f"n{p}_{gw}",
                              player_code=p, position="Defender", club=club, opp=opp,
                              is_home=True, mins=90, dc=max(base_rate + rng.normal(0, 2.0), 0),
                              thr=10, hit=0.0))
    r2 = opponent_defcon_ratings(pd.DataFrame(rows2), "Defender")
    assert abs(r2["shrunk"]).max() < 0.35, (
        f"null opponent effects should shrink to nothing, got {abs(r2['shrunk']).max():.3f}")

    op = own_team_profile(panel, "Defender")
    assert (~op["identified"]).all(), "own-team profile must be flagged unidentified"
    print("SELFTEST OK: opponent effect recovered within player (r=%.3f), shrinkage "
          "bounded, null shrinks away, own-team profile flagged unidentified." % corr)


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    if "--check" in sys.argv:
        import warnings; warnings.filterwarnings("ignore")
        panel = build_panel()
        print(f"25/26 DefCon-eligible appearances: {len(panel)}")
        for pos in ("Defender", "Midfielder"):
            r = opponent_defcon_ratings(panel, pos)
            if r.empty:
                continue
            print(f"\n{pos}: reliability {r['reliability'].iloc[0]:+.3f}, "
                  f"shrink {r['shrink_factor'].iloc[0]:.2f}, "
                  f"usable={bool(r['usable'].iloc[0])}")
            print(r[["team", "n", "raw", "shrunk", "hit_shrunk", "category"]]
                  .round(3).to_string(index=False))
        sys.exit(0)
    print(__doc__)
