from __future__ import annotations
import config
"""
repo_events.py — the event data already sitting in the data repo.
==================================================================
The project scraped Understat to get player-level xG, and built `crosswalk.py` to map
Understat ids onto `player_code` by name. That work is **redundant for xG**: the same
quantities ship inside FPL-Core-Insights, already keyed on the FPL `player_id`, with a
`players.csv` in every folder carrying `player_code`. No name matching, no fuzzy tier,
no unverified rows.

Measured on the 26/27 squad (584 players), 2026-08-19:

    local repo player-match xG   423 / 584  (72.4%)   exact player_code join
    scraped Understat crosswalk  373 / 584  (63.9%)   fuzzy name matching
    in local but not Understat   50
    in Understat but not local   0

Understat is a strict subset. It keeps exactly one advantage the repo cannot match —
shot-level history back to 2014/15, which `setpiece.py` needs and the repo (24/25
onward) does not have. For anything player-season shaped, read it from here.

THREE READERS, THREE DISTINCT USES
----------------------------------
`pl_player_season`   competitive PL history, keyed on player_code. FILTERS ON `-prem-`:
                     the By-Gameweek folders carry every competition, so an unfiltered
                     read mixes 1,047 Champions League and 835 EFL Cup rows into what
                     looks like a league season (Haaland reads 48 appearances, not 38).
`preseason_minutes`  26/27 pre-season friendlies + Super Cup. This is the only direct
                     observation of the new manager's intent before GW1, and it reaches
                     133 of the 209 cold-start players who otherwise have nothing but
                     ownership behind their start prior. See the health warning below.
`confirmed_lineups`  real XIs with `is_starting` and `lineup_status`, which is what
                     `lineups.py` was built to ingest and has been waiting on an
                     API-Football key to supply.

HEALTH WARNING ON PRE-SEASON MINUTES  [JUDGMENT]
------------------------------------------------
These are NOT competitive minutes and must not be treated as such:
  * opposition quality is uncontrolled and often non-league — the fixture list includes
    York City, and a 60-minute run-out against them is not evidence of a PL starting spot
  * managers deliberately rotate whole XIs at half time, so minutes are compressed
    toward ~45-60 for everyone and the top of the distribution is flat
  * there is no 25/26 pre-season in this repo, so the natural validation — do pre-season
    minutes predict GW1-6 starts, controlling for ownership? — cannot be run here
Consequently this module EXPOSES the feature and does not wire it into any prior. It is
data for the analyst, not a fitted signal, until the validation above can be run.
"""
import glob
import os
import numpy as np
import pandas as pd

PL_TOKEN = "-prem-"


def _players_lookup(folder):
    """player_id -> player_code from the players.csv beside a stats file."""
    p = os.path.join(folder, "players.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    if "player_code" not in d.columns or "player_id" not in d.columns:
        return None
    return d[["player_id", "player_code"]].drop_duplicates("player_id")


def _read_stats(pattern, pl_only):
    frames = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if d.empty:
            continue
        look = _players_lookup(os.path.dirname(f))
        if look is not None and "player_code" not in d.columns:
            d = d.merge(look, on="player_id", how="left")
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "match_id" in out.columns:
        out = out.drop_duplicates(["player_id", "match_id"])
        if pl_only:
            out = out[out["match_id"].astype(str).str.contains(PL_TOKEN, na=False)]
    return out


def pl_player_season(season="2025-2026", base=None):
    """Competitive Premier League totals per `player_code` for one season.

    Returns [player_code, pl_apps, pl_minutes, pl_xg, pl_xa, pl_shots, pl_goals,
    pl_assists, pl_xg90, pl_xa90] — league only, cups and Europe excluded.
    """
    root = base or config.repo(season)
    d = _read_stats(os.path.join(root, "By Gameweek", "GW*", "playermatchstats.csv"),
                    pl_only=True)
    if d.empty:
        return pd.DataFrame(columns=["player_code"])
    d = d.dropna(subset=["player_code"])
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    d = d.assign(_min=num("minutes_played"), _xg=num("xg"), _xa=num("xa"),
                 _sh=num("total_shots"), _g=num("goals"), _a=num("assists"))
    g = d.groupby("player_code").agg(
        pl_apps=("match_id", "nunique"), pl_minutes=("_min", "sum"),
        pl_xg=("_xg", "sum"), pl_xa=("_xa", "sum"), pl_shots=("_sh", "sum"),
        pl_goals=("_g", "sum"), pl_assists=("_a", "sum")).reset_index()
    per90 = g["pl_minutes"].replace(0, np.nan) / 90.0
    g["pl_xg90"] = g["pl_xg"] / per90
    g["pl_xa90"] = g["pl_xa"] / per90
    return g


def preseason_minutes(season="2026-2027", base=None,
                      tournaments=("Friendlies", "Uefa Super Cup", "Community Shield")):
    """26/27 pre-season minutes per `player_code`. See the health warning in the module
    docstring — this is exposed as an analyst feature, not wired into any prior.

    Returns [player_code, pre_minutes, pre_apps, pre_xg, pre_xa, pre_mean_minutes].
    """
    root = base or config.repo(season)
    frames = []
    for t in tournaments:
        d = _read_stats(os.path.join(root, "By Tournament", t, "GW*",
                                     "playermatchstats.csv"), pl_only=False)
        if not d.empty:
            d["tournament"] = t
            frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["player_code"])
    d = pd.concat(frames, ignore_index=True).dropna(subset=["player_code"])
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    d = d.assign(_min=num("minutes_played"), _xg=num("xg"), _xa=num("xa"))
    g = d.groupby("player_code").agg(
        pre_minutes=("_min", "sum"), pre_apps=("match_id", "nunique"),
        pre_xg=("_xg", "sum"), pre_xa=("_xa", "sum"),
        pre_mean_minutes=("_min", "mean")).reset_index()
    return g


def confirmed_lineups(season="2025-2026", gw=None, base=None):
    """Real XIs from the repo's `lineups.csv` (is_starting, formation, lineup_status).

    This is the shape `lineups.py` expects and has been waiting on an external API for.
    Note the 26/27 folders carry no lineups.csv yet — nothing has kicked off — so this
    reads history, and returns empty rather than raising when the file is absent.
    """
    root = base or config.repo(season)
    pat = os.path.join(root, "By Gameweek", f"GW{gw}" if gw else "GW*", "lineups.csv")
    d = _read_stats(pat, pl_only=False)
    if d.empty:
        return d
    if "match_id" in d.columns:
        d = d[d["match_id"].astype(str).str.contains(PL_TOKEN, na=False)]
    return d


def start_rate_from_lineups(season="2025-2026", base=None):
    """Per `player_code`: how often he was in the XI, from real lineup sheets rather
    than inferred from minutes. Returns [player_code, xi_starts, xi_sheets, xi_rate]."""
    d = confirmed_lineups(season, base=base)
    if d.empty or "is_starting" not in d.columns:
        return pd.DataFrame(columns=["player_code"])
    d = d.dropna(subset=["player_code"])
    d["_s"] = d["is_starting"].astype(str).str.lower().isin(["true", "1", "yes"])
    g = d.groupby("player_code").agg(xi_starts=("_s", "sum"),
                                     xi_sheets=("_s", "size")).reset_index()
    g["xi_rate"] = g["xi_starts"] / g["xi_sheets"].replace(0, np.nan)
    return g


# --------------------------------------------------- fixture congestion layer
# The By-Gameweek folders carry EVERY competition, tagged both by a `tournament`
# column and by a token inside `match_id`. `pl_player_season` above uses that to
# exclude cups; the congestion work needs the opposite view — the cup rows are the
# treatment and the league rows are the outcome.
#
# TWO TRAPS, both of which produce wrong-but-plausible output rather than an error:
#
#  1. `matches.csv` `home_team`/`away_team` hold the club's `code` from teams.csv,
#     NOT its `id`. Both are small integers and the ranges overlap, so mapping
#     through `id` silently resolves about half the rows to the WRONG club — Burnley
#     reads as a Champions League regular. The rest of this repo already joins on
#     `code`; so does this.
#  2. `players.csv` inside a gameweek folder is a whole-league SNAPSHOT, not that
#     match's teamsheet, so its `team_code` is a player's CURRENT club rather than
#     his club at that match. Attributing a cup appearance through it hands a
#     January transfer's autumn European matches to his new club. Club attribution
#     therefore comes from the match row; `players.csv` supplies only the
#     `player_id -> player_code` map, which is stable.
CUP_TOKENS = ("champions-league", "europa-league", "conference-league", "efl-cup")
PREM = "prem"


def _tournament(match_id):
    """Competition token out of a match_id, or None."""
    s = str(match_id)
    for t in CUP_TOKENS + (PREM,):
        if "-" + t + "-" in s:
            return t
    return None


def _team_code_map(season, base=None):
    """teams.csv `code` -> canonical club name, the join the whole repo uses."""
    root = base or config.repo(season)
    t = pd.read_csv(os.path.join(root, "teams.csv"))
    return dict(zip(pd.to_numeric(t["code"], errors="coerce"), t["name"]))


def competitive_calendar(season="2025-2026", base=None):
    """Every competitive match a PL club played, all competitions, one row per club.

    Returns [club, tournament, gameweek, match_id, kickoff, is_home]. Foreign
    opponents carry no `code` and simply do not generate a row, which is what we
    want — only the PL club's own fixture load is being measured.
    """
    root = base or config.repo(season)
    code = _team_code_map(season, base=root)
    frames = []
    for f in sorted(glob.glob(os.path.join(root, "By Gameweek", "GW*",
                                           "matches.csv"))):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    cols = ["club", "tournament", "gameweek", "match_id", "kickoff", "is_home"]
    if not frames:
        return pd.DataFrame(columns=cols)
    m = pd.concat(frames, ignore_index=True).drop_duplicates("match_id")
    m["tournament"] = m["match_id"].map(_tournament)
    m["kickoff"] = pd.to_datetime(m.get("kickoff_time"), errors="coerce", utc=True)
    rows = []
    for col, ish in (("home_team", True), ("away_team", False)):
        v = m.copy()
        v["club"] = pd.to_numeric(v[col], errors="coerce").map(code)
        v["is_home"] = ish
        rows.append(v[cols])
    out = pd.concat(rows, ignore_index=True).dropna(subset=["club", "tournament"])
    return out.sort_values(["club", "kickoff"]).reset_index(drop=True)


def recovery_panel(season="2025-2026", base=None):
    """One row per club-PL-match with days since that club's previous COMPETITIVE
    fixture in any competition, and which competition that fixture was in.

    `rec_all`   days since the previous match in ANY competition  <- the instrument
    `rec_pl`    days since the previous LEAGUE match              <- what the earlier
                congestion null was restricted to, kept so the two measures can be
                contrasted directly rather than argued about
    `prev_tour` competition of the preceding fixture
    """
    cal = competitive_calendar(season, base=base)
    cal = cal.dropna(subset=["kickoff"]).sort_values(["club", "kickoff"])
    cal["prev_ko"] = cal.groupby("club")["kickoff"].shift(1)
    cal["prev_tour"] = cal.groupby("club")["tournament"].shift(1)
    cal["rec_all"] = (cal["kickoff"] - cal["prev_ko"]).dt.total_seconds() / 86400.0

    lg = cal[cal["tournament"] == PREM].copy().sort_values(["club", "kickoff"])
    lg["prev_pl_ko"] = lg.groupby("club")["kickoff"].shift(1)
    lg["rec_pl"] = (lg["kickoff"] - lg["prev_pl_ko"]).dt.total_seconds() / 86400.0
    return lg.drop(columns=["prev_pl_ko"]).reset_index(drop=True)


def all_comp_player_matches(season="2025-2026", base=None):
    """player_code x match minutes across EVERY competition.

    Club attribution is deliberately absent — see trap 2 in the section comment.
    Join club through the match row (`competitive_calendar`) where it is needed.
    """
    root = base or config.repo(season)
    d = _read_stats(os.path.join(root, "By Gameweek", "GW*",
                                 "playermatchstats.csv"), pl_only=False)
    if d.empty:
        return pd.DataFrame(columns=["player_code", "match_id", "tournament",
                                     "minutes_played"])
    d = d.dropna(subset=["player_code"])
    d["tournament"] = d["match_id"].map(_tournament)
    d["minutes_played"] = pd.to_numeric(d["minutes_played"], errors="coerce")
    keep = ["player_code", "match_id", "tournament", "minutes_played"]
    for c in ("distance_covered", "sprinting_distance", "number_of_sprints",
              "start_min", "finish_min"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
            keep.append(c)
    return d[keep].reset_index(drop=True)


def squad_sheets(season="2025-2026", base=None):
    """The matchday SQUAD for every match in every competition — the risk set.

    This is the difference between measuring rotation and not measuring it.
    `playermatchstats` holds only players who actually got on the pitch: its minimum
    minutes value is 1 and it contains no zero rows, so a panel built from it is
    conditioned on appearing and a rested player simply vanishes from the sample
    rather than showing up as a zero. `lineups.csv` carries the full ~20-man sheet
    including unused substitutes, which is the population a manager chooses from.

    `team_code` here is a per-match lineup row, so unlike the `players.csv` snapshot
    it is match-time truth and is safe to attribute a club with.

    Returns [player_code, match_id, tournament, club, is_starting].
    """
    root = base or config.repo(season)
    d = _read_stats(os.path.join(root, "By Gameweek", "GW*", "lineups.csv"),
                    pl_only=False)
    if d.empty:
        return pd.DataFrame(columns=["player_code", "match_id", "tournament", "club",
                                     "is_starting"])
    d = d.dropna(subset=["player_code"])
    d["tournament"] = d["match_id"].map(_tournament)
    code = _team_code_map(season, base=root)
    d["club"] = pd.to_numeric(d.get("team_code"), errors="coerce").map(code)
    d["is_starting"] = d["is_starting"].astype(str).str.lower().isin(
        ["true", "1", "yes"])
    return d[["player_code", "match_id", "tournament", "club",
              "is_starting"]].reset_index(drop=True)


def selftest():
    import tempfile
    root = tempfile.mkdtemp(prefix="repo_events_")
    gw = os.path.join(root, "By Gameweek", "GW1")
    os.makedirs(gw)
    pd.DataFrame({"player_id": [1, 2], "player_code": [111, 222]}).to_csv(
        os.path.join(gw, "players.csv"), index=False)
    # one league match and one cup match for the same player
    pd.DataFrame({
        "player_id": [1, 1, 2],
        "match_id": ["25-26-prem-a-vs-b", "25-26-efl-cup-a-vs-c", "25-26-prem-a-vs-b"],
        "minutes_played": [90, 90, 45], "xg": [0.5, 9.9, 0.1], "xa": [0.2, 9.9, 0.0],
        "total_shots": [3, 9, 1], "goals": [1, 9, 0], "assists": [0, 9, 0],
    }).to_csv(os.path.join(gw, "playermatchstats.csv"), index=False)

    s = pl_player_season(base=root)
    r = s[s.player_code == 111].iloc[0]
    assert r["pl_apps"] == 1, "cup match leaked into the league season"
    assert abs(r["pl_minutes"] - 90) < 1e-9 and abs(r["pl_xg"] - 0.5) < 1e-9
    assert abs(r["pl_xg90"] - 0.5) < 1e-9

    pd.DataFrame({"player_id": [1, 2], "match_id": ["m", "m"],
                  "is_starting": [True, False], "player_code": [111, 222]}).to_csv(
        os.path.join(gw, "lineups.csv"), index=False)
    # lineups carry no -prem- token in this fixture, so the filter should empty it
    lr = start_rate_from_lineups(base=root)
    assert lr.empty or lr["xi_starts"].sum() == 0

    pre = os.path.join(root, "By Tournament", "Friendlies", "GW0")
    os.makedirs(pre)
    pd.DataFrame({"player_id": [1], "player_code": [111]}).to_csv(
        os.path.join(pre, "players.csv"), index=False)
    pd.DataFrame({"player_id": [1], "match_id": ["26-27-friendly-x"],
                  "minutes_played": [60], "xg": [0.3], "xa": [0.1]}).to_csv(
        os.path.join(pre, "playermatchstats.csv"), index=False)
    p = preseason_minutes(base=root)
    assert abs(p.iloc[0]["pre_minutes"] - 60) < 1e-9, "pre-season minutes not read"

    # --- congestion layer: both traps must be caught ---------------------------
    cg = os.path.join(root, "By Gameweek", "GW2")
    os.makedirs(cg)
    pd.DataFrame({"player_id": [1, 2], "player_code": [111, 222],
                  "team_code": [7, 7]}).to_csv(
        os.path.join(cg, "players.csv"), index=False)
    # code 3 = Alpha, code 7 = Beta, but their *ids* are 7 and 3 — so a consumer that
    # maps through `id` swaps the two clubs and still returns a full, plausible frame.
    pd.DataFrame({"code": [3, 7], "id": [7, 3], "name": ["Alpha", "Beta"],
                  "short_name": ["ALP", "BET"]}).to_csv(
        os.path.join(root, "teams.csv"), index=False)
    pd.DataFrame({
        "gameweek": [2, 2],
        "kickoff_time": ["2025-08-19T19:00:00+00:00", "2025-08-22T19:00:00+00:00"],
        "home_team": [3, 7], "away_team": [7, 3],
        "match_id": ["25-26-efl-cup-alpha-vs-beta", "25-26-prem-beta-vs-alpha"],
    }).to_csv(os.path.join(cg, "matches.csv"), index=False)
    pd.DataFrame({"player_id": [1, 1],
                  "match_id": ["25-26-efl-cup-alpha-vs-beta",
                               "25-26-prem-beta-vs-alpha"],
                  "minutes_played": [90, 20]}).to_csv(
        os.path.join(cg, "playermatchstats.csv"), index=False)

    cal = competitive_calendar(base=root)
    assert set(cal["club"]) == {"Alpha", "Beta"}, (
        "club attribution went through `id` instead of `code`: "
        + str(sorted(set(cal["club"]))))
    assert set(cal["tournament"]) == {"efl-cup", "prem"}

    rp = recovery_panel(base=root)
    assert (rp["tournament"] == "prem").all(), "cup rows leaked into the outcome panel"
    r = rp.iloc[0]
    assert abs(r["rec_all"] - 3.0) < 1e-6, (
        "all-competition recovery should be 3 days, got " + str(r["rec_all"]))
    assert pd.isna(r["rec_pl"]), "no previous LEAGUE match exists, so rec_pl is undefined"
    assert r["prev_tour"] == "efl-cup"

    # lineups carry the unused subs that playermatchstats drops
    pd.DataFrame({"match_id": ["25-26-prem-beta-vs-alpha"] * 2,
                  "player_id": [1, 2], "team_code": [7, 7],
                  "is_starting": [True, False]}).to_csv(
        os.path.join(cg, "lineups.csv"), index=False)
    ss = squad_sheets(base=root)
    ss = ss[ss["match_id"] == "25-26-prem-beta-vs-alpha"]
    assert len(ss) == 2, "unused substitute dropped from the risk set"
    assert set(ss["club"]) == {"Beta"}, sorted(set(ss["club"]))
    assert ss["is_starting"].tolist() == [True, False]

    apm = all_comp_player_matches(base=root)
    assert set(apm["tournament"]) == {"efl-cup", "prem"}
    gw2 = apm[apm["match_id"].str.contains("alpha")]
    assert gw2["minutes_played"].sum() == 110, gw2.to_dict("records")
    assert "team_code" not in apm.columns, (
        "club must not be attributed through the players.csv snapshot")

    print("SELFTEST OK: cup and European rows excluded from the league season, "
          "per-90s correct, pre-season minutes read, lineups filtered, "
          "congestion layer joins on team `code`, keeps cup rows as treatment, and "
          "builds the risk set from lineups rather than appearances.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    if "--check" in sys.argv:
        import warnings; warnings.filterwarnings("ignore")
        s = pl_player_season()
        print(f"25/26 PL player-seasons: {len(s)}")
        print(s.nlargest(8, "pl_xg")[["player_code", "pl_apps", "pl_minutes",
                                      "pl_xg", "pl_xa", "pl_xg90"]].round(2).to_string(index=False))
        p = preseason_minutes()
        print(f"\n26/27 pre-season rows: {len(p)}")
        print(p.nlargest(8, "pre_minutes").round(1).to_string(index=False))
        sys.exit(0)
    print(__doc__)
