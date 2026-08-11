from __future__ import annotations
import config
"""
sd_ingest.py — thin, cached wrapper over `soccerdata` (Understat / WhoScored).
==============================================================================
No modelling lives here. Three readers, one cache, one schema contract:

  understat_shots(seasons)       one row per shot
  understat_team_match(seasons)  one row per team-match (PPDA, deep, xG/xGA)
  whoscored_events(seasons)      one row per event (Selenium-backed, slow, fragile)

Everything checks disk before network. The cache is under `config.SD_CACHE`; it lives
in SCRATCH so it is gitignored, but it is NOT cheaply regenerable — a full WhoScored
pull is hours of sequential requests. Do not delete it casually, and never re-scrape
inside a fit loop.

WhoScored is cached PER MATCH, so an interrupted pull resumes without re-fetching
completed matches. Understat is cached per season.

Every row carries `scraped_at`. Scraped sources drift and break silently; the timestamp
is the only way to diagnose a schema change after the fact.

`--selftest` runs against synthetic fixtures written into a temp cache. It never touches
the network, so it is safe in CI and on an offline box.

**ToS**: these are scrapers. Cache aggressively, run infrequently, keep volume low.

Provenance of the numbers this feeds: [VERIFIED] only once the calibration in
`xg_calibrate.py` has passed its assertions. Raw Understat xG is [CHECK] until then —
it is a different xG model from the Opta series `TeamModel` is fitted on.
"""
import argparse, os, sys, time, shutil, tempfile
import numpy as np, pandas as pd

LEAGUE = "ENG-Premier League"

# Schema contracts. A scraper that stops returning one of these has drifted, and we
# want that to fail here rather than surface as a quiet NaN column downstream.
#
# Verified against soccerdata 1.9.1 / Understat, 2026-08-09. Two columns the spec asked
# for are NOT available and have been dropped from the contract rather than carried as
# permanent NaN:
#   * `last_action` — absent from read_shot_events(); nothing in §2 uses it.
#   * `shots` / `shots_against` — absent from read_team_match_stats(); needed only by
#     §5, which is out of scope. Derive from the shot table if §5 is ever built.
REQUIRED_SHOTS = ("match_id", "date", "team", "understat_player_id", "player_name",
                  "minute", "xg", "situation", "shot_type", "result", "x", "y")
REQUIRED_TEAM_MATCH = ("match_id", "date", "team", "opp", "is_home", "ppda",
                       "ppda_allowed", "deep", "deep_allowed", "xg", "xga",
                       "npxg", "npxga", "goals")
REQUIRED_PLAYER_MATCH = ("match_id", "date", "team", "understat_player_id",
                         "player_name", "position", "minutes", "shots", "xg", "xa")
REQUIRED_PLAYER_SEASON = ("team", "understat_player_id", "player_name", "minutes",
                          "np_xg", "xa", "shots")
REQUIRED_EVENTS = ("match_id", "date", "team", "player_name", "minute", "type",
                   "x", "y")

SITUATIONS = ("OpenPlay", "FromCorner", "SetPiece", "Penalty", "DirectFreekick")
# Understat's own spelling -> the canonical tokens above. soccerdata 1.9.1 returns
# these space-separated, and does NOT return a Penalty label at all (see
# _recover_penalties).
SITUATION_MAP = {"Open Play": "OpenPlay", "From Corner": "FromCorner",
                 "Set Piece": "SetPiece", "Direct Freekick": "DirectFreekick",
                 "Penalty": "Penalty"}

# The penalty spot in Understat's normalised pitch coordinates, and the constant xG
# their model assigns it. Used to re-label the shots soccerdata leaves unlabelled.
PENALTY_XY = (0.885, 0.5)
PENALTY_XG = (0.74, 0.79)
# the event types §4.1 needs to reconstruct CBIT / CBIRT
EVENT_TYPES = ("Tackle", "Interception", "Clearance", "BlockedPass", "Aerial",
               "BallRecovery")


# Understat club names -> the frame naming the rest of the repo uses (E0_recon,
# press_index, regime_panel all agree on this spelling). Anything not listed passes
# through unchanged, so a new promoted side surfaces as an unmatched name in the
# calibration join rather than silently mapping to the wrong club.
UNDERSTAT_TO_FRAME = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Tottenham": "Tottenham", "Wolverhampton Wanderers": "Wolves",
    "Leicester": "Leicester", "Leeds": "Leeds", "West Ham": "West Ham",
    "Brighton": "Brighton", "Sheffield United": "Sheffield United",
    "Ipswich": "Ipswich", "Luton": "Luton",
    # Clubs no longer in the division. Needed once the studies reach back to 2014/15 —
    # the map above only ever covered teams in the current season.
    "Queens Park Rangers": "QPR", "West Bromwich Albion": "West Brom",
}


def normalise_team(name):
    """Understat spelling -> repo frame spelling. Idempotent."""
    return UNDERSTAT_TO_FRAME.get(name, name)


# --------------------------------------------------------------------- cache
def _cache_dir(kind, cache_root=None):
    d = os.path.join(cache_root or config.SD_CACHE, kind)
    os.makedirs(d, exist_ok=True)
    return d


def _season_path(kind, season, cache_root=None):
    return os.path.join(_cache_dir(kind, cache_root), f"{season}.csv.gz")


def _match_path(season, match_id, cache_root=None):
    d = os.path.join(_cache_dir("whoscored", cache_root), str(season))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{match_id}.csv.gz")


def _validate(df, required, what):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{what}: source schema drifted — missing {missing}. "
            f"Got {sorted(df.columns)}. Do not paper over this; the upstream scraper "
            f"changed and every quantity built on it is suspect.")
    return df


def _stamp(df):
    d = df.copy()
    d["scraped_at"] = pd.Timestamp.now("UTC").isoformat()
    return d


def _read_season_cache(kind, season, cache_root=None):
    p = _season_path(kind, season, cache_root)
    return pd.read_csv(p) if os.path.exists(p) else None


def _write_season_cache(df, kind, season, cache_root=None):
    df.to_csv(_season_path(kind, season, cache_root), index=False,
              compression="gzip")
    return df


# ------------------------------------------------------------------- network
def _retry(fn, tries=4, base_delay=2.0, what="request"):
    """Exponential backoff. These are someone else's servers; be polite and give up
    rather than hammer."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                                    # noqa: BLE001
            last = e
            if i == tries - 1:
                break
            delay = base_delay * (2 ** i)
            print(f"[sd-ingest] {what} failed ({e}); retrying in {delay:.0f}s")
            time.sleep(delay)
    raise RuntimeError(f"{what} failed after {tries} attempts") from last


def _soccerdata():
    try:
        import soccerdata as sd
    except ImportError as e:
        raise RuntimeError(
            "soccerdata is not installed and the cache is empty. On a networked box:\n"
            "  pip install soccerdata\n"
            "Then re-run. Offline, populate the cache elsewhere and copy "
            f"{config.SD_CACHE} across.") from e
    return sd


# ------------------------------------------------------------------ Understat
def understat_shots(seasons, league=LEAGUE, cache_root=None, force=False):
    """One row per shot.

    Columns: match_id, date, team, understat_player_id, player_name, minute, xg,
    situation, shot_type, result, x, y, last_action (+ scraped_at).

    `situation` is one of SITUATIONS — that split is the whole point of the reader
    (spec §2 needs open-play separated from set-piece and penalty).

    NOTE: `xg` here is Understat's model, NOT the Opta series the team layer is fitted
    on. Run it through `xg_calibrate.apply_calibration` before it touches the player
    layer, or you introduce a level mismatch into every projection.
    """
    frames = []
    for s in seasons:
        cached = None if force else _read_season_cache("understat_shots", s, cache_root)
        if cached is None:
            sd = _soccerdata()
            reader = sd.Understat(leagues=league, seasons=s)
            raw = _retry(lambda: reader.read_shot_events(),
                         what=f"Understat shots {s}")
            cached = _write_season_cache(_stamp(_normalise_shots(raw)),
                                         "understat_shots", s, cache_root)
        cached = _validate(cached, REQUIRED_SHOTS, f"understat_shots[{s}]")
        cached["season"] = s
        frames.append(cached)
    return pd.concat(frames, ignore_index=True)


def understat_team_match(seasons, league=LEAGUE, cache_root=None, force=False):
    """One row per team-match: ppda, ppda_allowed, deep, deep_allowed, xg, xga,
    npxg, npxga, shots, shots_against (+ scraped_at).

    `ppda` is the measured pressing intensity §3 substitutes for the manager-aware
    estimates currently hardcoded in press_index.py. Season-aggregate PPDA measures
    results as much as intent (a team 1-0 up stops pressing) — see §3.2; filter to
    level-score minutes where the source allows.
    """
    frames = []
    for s in seasons:
        cached = None if force else _read_season_cache("understat_team", s, cache_root)
        if cached is None:
            sd = _soccerdata()
            reader = sd.Understat(leagues=league, seasons=s)
            raw = _retry(lambda: reader.read_team_match_stats(),
                         what=f"Understat team-match {s}")
            cached = _write_season_cache(_stamp(_normalise_team_match(raw)),
                                         "understat_team", s, cache_root)
        cached = _validate(cached, REQUIRED_TEAM_MATCH, f"understat_team_match[{s}]")
        cached["season"] = s
        frames.append(cached)
    return pd.concat(frames, ignore_index=True)


def _recover_penalties(d):
    """soccerdata 1.9.1 does not map Understat's `Penalty` situation — those shots come
    back with a null situation instead. [VERIFIED 2026-08-09] on 25/26: all 92 unlabelled
    shots sit at exactly the penalty spot (0.885, 0.5) with a constant xG of 0.7612 and
    an 83.7% conversion rate. They are penalties.

    Recovered under a guard rather than filled blindly. If an unlabelled shot does NOT
    carry the penalty signature, the parser has drifted somewhere new and we want to
    hear about it — a mislabelled penalty would corrupt both the penalty component and
    whichever component absorbed it.
    """
    na = d["situation"].isna()
    if not na.any():
        return d
    sub = d[na]
    at_spot = (np.isclose(sub["x"], PENALTY_XY[0], atol=1e-6)
               & np.isclose(sub["y"], PENALTY_XY[1], atol=1e-6))
    in_band = sub["xg"].between(*PENALTY_XG)
    ok = at_spot & in_band
    if not ok.all():
        bad = sub[~ok]
        raise RuntimeError(
            f"{len(bad)} unlabelled shots do not carry the penalty signature "
            f"(expected x={PENALTY_XY[0]}, y={PENALTY_XY[1]}, xg in {PENALTY_XG}); "
            f"got e.g. x={bad['x'].iloc[0]}, y={bad['y'].iloc[0]}, "
            f"xg={bad['xg'].iloc[0]}. The Understat parser has drifted — do not guess "
            f"at the situation, fix the mapping.")
    d = d.copy()
    d.loc[na, "situation"] = "Penalty"
    print(f"[sd-ingest] recovered {int(na.sum())} unlabelled shots as penalties "
          f"(verified at the spot)")
    return d


def _normalise_shots(raw):
    """soccerdata returns a MultiIndex frame whose column names have moved between
    releases. Map to the contract, then let _validate catch the rest."""
    d = raw.reset_index()
    ren = {"game_id": "match_id", "player_id": "understat_player_id",
           "player": "player_name", "body_part": "shot_type",
           "location_x": "x", "location_y": "y", "xG": "xg"}
    d = d.rename(columns={k: v for k, v in ren.items() if k in d.columns})
    for c in ("xg", "x", "y", "minute"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = _recover_penalties(d)
    d["situation"] = d["situation"].map(lambda s: SITUATION_MAP.get(s, s))
    unknown = set(d["situation"].dropna().unique()) - set(SITUATIONS)
    if unknown:
        raise RuntimeError(
            f"unrecognised situation values {sorted(unknown)}; extend SITUATION_MAP "
            f"rather than letting them fall through to a component silently.")
    return d


def _normalise_team_match(raw):
    """read_team_match_stats() is WIDE — one row per game with home_*/away_* columns.
    Melt to one row per team-match, which is what every downstream consumer wants and
    what the spec's contract describes."""
    d = raw.reset_index()
    ren = {"game_id": "match_id"}
    d = d.rename(columns={k: v for k, v in ren.items() if k in d.columns})

    def side(us, them, is_home):
        return pd.DataFrame({
            "match_id": d["match_id"], "date": d["date"],
            "team": d[f"{us}_team"], "opp": d[f"{them}_team"], "is_home": is_home,
            "ppda": d[f"{us}_ppda"], "ppda_allowed": d[f"{them}_ppda"],
            "deep": d[f"{us}_deep_completions"],
            "deep_allowed": d[f"{them}_deep_completions"],
            "xg": d[f"{us}_xg"], "xga": d[f"{them}_xg"],
            "npxg": d[f"{us}_np_xg"], "npxga": d[f"{them}_np_xg"],
            "goals": d[f"{us}_goals"]})

    out = pd.concat([side("home", "away", True), side("away", "home", False)],
                    ignore_index=True)
    for c in ("ppda", "ppda_allowed", "deep", "deep_allowed", "xg", "xga",
              "npxg", "npxga", "goals"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _normalise_player_match(raw):
    d = raw.reset_index()
    ren = {"game_id": "match_id", "player_id": "understat_player_id",
           "player": "player_name"}
    d = d.rename(columns={k: v for k, v in ren.items() if k in d.columns})
    if "date" not in d.columns:
        # player_match_stats carries no date; recover it from the game label
        d["date"] = pd.to_datetime(d["game"].astype(str).str.slice(0, 10),
                                   errors="coerce")
    for c in ("minutes", "shots", "xg", "xa", "goals", "assists"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def understat_player_match(seasons, league=LEAGUE, cache_root=None, force=False):
    """One row per player-match: minutes, shots, xg, xa, position (+ scraped_at).

    `minutes` is the exposure the per-90 rates in `setpiece` divide by, so this reader
    is what makes §2 runnable without the FPL data repo. Note the id here is Understat's
    — crossing to `player_code` (G5) is only required at the point the result feeds the
    FPL model, not for measuring reliability or persistence.
    """
    frames = []
    for s in seasons:
        cached = None if force else _read_season_cache("understat_player", s, cache_root)
        if cached is None:
            sd = _soccerdata()
            reader = sd.Understat(leagues=league, seasons=s)
            raw = _retry(lambda: reader.read_player_match_stats(),
                         what=f"Understat player-match {s}")
            cached = _write_season_cache(_stamp(_normalise_player_match(raw)),
                                         "understat_player", s, cache_root)
        cached = _validate(cached, REQUIRED_PLAYER_MATCH, f"understat_player_match[{s}]")
        cached["season"] = s
        frames.append(cached)
    return pd.concat(frames, ignore_index=True)


def understat_player_season(seasons, league=LEAGUE, cache_root=None, force=False):
    """One row per (season, club, player): minutes, np_xg, xa, shots (+ scraped_at).

    Prefer this over `understat_player_match` whenever season totals are enough. The
    match-level reader fetches every match individually — ~380 requests per season, about
    40 minutes for a full history — while this is ONE request per season for the same
    aggregates. A player who moves mid-season appears under each club he played for, which
    is what any "who left this club" question needs.
    """
    frames = []
    for s in seasons:
        cached = None if force else _read_season_cache("understat_pseason", s, cache_root)
        if cached is None:
            sd = _soccerdata()
            reader = sd.Understat(leagues=league, seasons=s)
            raw = _retry(lambda: reader.read_player_season_stats(),
                         what=f"Understat player-season {s}")
            d = raw.reset_index().rename(columns={"player_id": "understat_player_id",
                                                  "player": "player_name"})
            for c in ("minutes", "np_xg", "xg", "xa", "shots", "goals", "np_goals"):
                if c in d.columns:
                    d[c] = pd.to_numeric(d[c], errors="coerce")
            cached = _write_season_cache(_stamp(d), "understat_pseason", s, cache_root)
        cached = _validate(cached, REQUIRED_PLAYER_SEASON, f"understat_player_season[{s}]")
        cached["season"] = s
        frames.append(cached)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------- WhoScored
def whoscored_events(seasons, league=LEAGUE, cache_root=None, match_ids=None,
                     delay=3.0, limit=None):
    """One row per event, cached PER MATCH so a partial pull is resumable.

    Slow and fragile: Selenium-backed, minutes per match, breaks when WhoScored
    reshuffles its page. Never call this inside a fit loop — pull once, then read
    the cache.

    Returns whatever is cached plus whatever this call managed to fetch. Check
    `whoscored_cached_matches()` to see coverage before modelling on it.
    """
    frames, fetched = [], 0
    for s in seasons:
        want = match_ids if match_ids is not None else _whoscored_schedule(s, league)
        for mid in want:
            p = _match_path(s, mid, cache_root)
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
                continue
            if limit is not None and fetched >= limit:
                continue
            sd = _soccerdata()
            reader = sd.WhoScored(leagues=league, seasons=s)
            raw = _retry(lambda: reader.read_events(match_id=mid),
                         what=f"WhoScored events {s}/{mid}")
            df = _stamp(_normalise_events(raw))
            df.to_csv(p, index=False, compression="gzip")
            frames.append(df)
            fetched += 1
            time.sleep(delay)                     # sequential and polite, by design
    if not frames:
        return pd.DataFrame(columns=list(REQUIRED_EVENTS) + ["scraped_at"])
    out = pd.concat(frames, ignore_index=True)
    return _validate(out, REQUIRED_EVENTS, "whoscored_events")


def _whoscored_schedule(season, league):
    sd = _soccerdata()
    reader = sd.WhoScored(leagues=league, seasons=season)
    sched = _retry(lambda: reader.read_schedule(), what=f"WhoScored schedule {season}")
    return list(sched.reset_index()["game_id"].unique())


def _normalise_events(raw):
    d = raw.reset_index()
    ren = {"game_id": "match_id", "player": "player_name", "event_type": "type",
           "expanded_minute": "minute"}
    d = d.rename(columns={k: v for k, v in ren.items() if k in d.columns})
    for c in ("x", "y", "minute"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def whoscored_cached_matches(season, cache_root=None):
    """Which matches are already on disk — coverage check before modelling."""
    d = os.path.join(_cache_dir("whoscored", cache_root), str(season))
    if not os.path.isdir(d):
        return []
    return sorted(f[:-len(".csv.gz")] for f in os.listdir(d) if f.endswith(".csv.gz"))


# ------------------------------------------------------------------- selftest
def _fake_shots(n=600, seed=0):
    rng = np.random.default_rng(seed)
    sit = rng.choice(SITUATIONS, n, p=[0.74, 0.12, 0.05, 0.03, 0.06])
    pen = sit == "Penalty"
    xg = np.where(pen, 0.7612, rng.gamma(1.6, 0.055, n))
    x = np.where(pen, PENALTY_XY[0], rng.random(n))
    y = np.where(pen, PENALTY_XY[1], rng.random(n))
    return pd.DataFrame({
        "match_id": rng.integers(1, 60, n), "date": "2025-08-16",
        "team": rng.choice(["Arsenal", "Liverpool", "Everton"], n),
        "understat_player_id": rng.integers(1000, 1040, n),
        "player_name": "p", "minute": rng.integers(1, 95, n), "xg": xg,
        "situation": sit, "shot_type": "Right Foot",
        "result": rng.choice(["Goal", "Missed Shot", "Saved Shot"], n, p=[.1, .5, .4]),
        "x": x, "y": y})


def _fake_team_match(n=120, seed=1):
    rng = np.random.default_rng(seed)
    teams = rng.choice(["Arsenal", "Liverpool", "Everton"], n)
    return pd.DataFrame({
        "match_id": np.arange(n) // 2, "date": "2025-08-16",
        "team": teams, "opp": teams[::-1], "is_home": np.arange(n) % 2 == 0,
        "ppda": rng.gamma(30, 0.4, n), "ppda_allowed": rng.gamma(30, 0.4, n),
        "deep": rng.poisson(7, n), "deep_allowed": rng.poisson(7, n),
        "xg": rng.gamma(4, 0.35, n), "xga": rng.gamma(4, 0.35, n),
        "npxg": rng.gamma(4, 0.32, n), "npxga": rng.gamma(4, 0.32, n),
        "goals": rng.poisson(1.4, n)})


def selftest():
    tmp = tempfile.mkdtemp(prefix="sd_ingest_selftest_")
    try:
        # seed the cache, then read through the public API: no network is reachable
        # from the selftest path, so a cache miss here would raise.
        _write_season_cache(_stamp(_fake_shots()), "understat_shots", "2526", tmp)
        _write_season_cache(_stamp(_fake_team_match()), "understat_team", "2526", tmp)

        sh = understat_shots(["2526"], cache_root=tmp)
        assert len(sh) == 600, "cache round-trip lost rows"
        assert set(sh["situation"]) <= set(SITUATIONS), "unexpected situation value"
        assert sh["scraped_at"].notna().all(), "scraped_at must be present on every row"
        assert (sh["season"] == "2526").all()

        tm = understat_team_match(["2526"], cache_root=tmp)
        assert len(tm) == 120 and tm["ppda"].notna().all()

        # the live parser leaves penalties unlabelled; recovery is guarded, not blind
        raw = _fake_shots()
        raw.loc[raw["situation"] == "Penalty", "situation"] = np.nan
        rec = _recover_penalties(raw)
        assert (rec["situation"] == "Penalty").sum() == (
            _fake_shots()["situation"] == "Penalty").sum()
        drifted = raw.copy()
        drifted.loc[drifted["situation"].isna(), "x"] = 0.5      # not the penalty spot
        try:
            _recover_penalties(drifted)
        except RuntimeError as e:
            assert "drifted" in str(e)
        else:
            raise AssertionError("an unlabelled non-penalty shot must raise, not be guessed")

        # a drifted source must fail loudly, not quietly return a thinner frame
        bad = _fake_shots().drop(columns=["situation"])
        _write_season_cache(_stamp(bad), "understat_shots", "2425", tmp)
        try:
            understat_shots(["2425"], cache_root=tmp)
        except RuntimeError as e:
            assert "situation" in str(e)
        else:
            raise AssertionError("missing required column must raise")

        # whoscored: per-match shards, resumable, coverage reportable
        assert whoscored_cached_matches("2526", tmp) == []
        ev = pd.DataFrame({"match_id": [1, 1], "date": "2025-08-16",
                           "team": "Arsenal", "player_name": "p", "minute": [4, 9],
                           "type": ["Tackle", "Clearance"], "x": [.3, .1], "y": [.5, .5]})
        _stamp(ev).to_csv(_match_path("2526", 1, tmp), index=False, compression="gzip")
        assert whoscored_cached_matches("2526", tmp) == ["1"]
        got = whoscored_events(["2526"], cache_root=tmp, match_ids=[1])
        assert len(got) == 2 and set(got["type"]) <= set(EVENT_TYPES)

        print(f"SELFTEST OK: {len(sh)} shots, {len(tm)} team-matches, "
              f"{len(got)} events round-tripped from cache; schema drift raises.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seasons", nargs="+", default=["2526"])
    ap.add_argument("--what", choices=["shots", "team", "events"], default="team")
    ap.add_argument("--limit", type=int, default=None,
                    help="whoscored: stop after N newly-fetched matches (resumable)")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    fn = {"shots": understat_shots, "team": understat_team_match}.get(a.what)
    d = (whoscored_events(a.seasons, limit=a.limit) if fn is None
         else fn(a.seasons))
    print(d.head().to_string())
    print(f"\n{len(d)} rows; cache at {config.SD_CACHE}")
