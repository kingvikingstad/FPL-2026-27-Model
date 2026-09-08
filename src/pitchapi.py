from __future__ import annotations
import config
"""
pitchapi.py — client for the PitchAPI event-derived analytics feed.
==================================================================
`https://api.pitchapi.dev/v1`, read-only, `X-API-KEY` header. Supplies the derived
metrics this repo cannot compute from FPL-Core-Insights: Opta-methodology PPDA with its
numerator and denominator published separately, progressive passes and carries, field
tilt, sequence tempo, and possession value (xT / VAEP / PV).

WHAT IT IS AND IS NOT
---------------------
It is NOT an independent check on FPL-Core-Insights for everything. `momentum.csv` in
the repo already carries what looks like the identical FotMob series (minute, signed
value in -100..100), and the repo's `matches.csv` carries `fotmob_id` and FotMob match
URLs, so at least part of the upstream is shared.

But the aggregates are NOT a copy, and this matters when reading any agreement between
the two as corroboration. [VERIFIED 2026-08-31] on Arsenal 3-0 Coventry, 26/27 GW1:

    field              repo    PitchAPI
    interceptions         5          11
    clearances           22          24
    passes attempted    613         619
    pass accuracy        92%       88.9%
    possession         64.0        64.7

PitchAPI recomputes from a raw event feed under published Opta definitions; the repo
takes FotMob's displayed match stats. Treat correlation between them as a measurement of
definitional overlap, not as validation of either.

KEY HANDLING
------------
The key is read from `PITCHAPI_KEY` and is never written to disk, logged, or embedded in
a cache path. Nothing in this module should ever be given a literal key as a default.

CACHING
-------
Every response is cached under `config.PITCH_CACHE` keyed by endpoint path. The service
is free and unmetered today, which is precisely why a research instrument must not
depend on it staying that way: a cached season rebuilds offline. `force=True` refetches.

COVERAGE — THE TRAP
-------------------
`/v1/leagues` lists the Premier League as 2021/2022..2026/2027 and the site says "full
history back to 2021". That is TRUE OF FIXTURES AND RESULTS AND FALSE OF THE ADVANCED
ANALYTICS, and the season list gives no hint of the difference. [VERIFIED 2026-08-31] by
pulling every match of five seasons:

    endpoint                                              PL seasons available
    /shots (expected_goals, xGOT, x/y, situation)         2021/22 ->
    /players, /stats, /lineups, /events, /momentum        2021/22 ->
    /advanced, /advanced/players  (PPDA, progressive      2025/26 ->  ONLY
      passes and carries, xT/VAEP/PV, field tilt, tempo)

    matches with /advanced:  21/22 0/380 | 22/23 0/380 | 23/24 23/380
                             24/25 0/380 | 25/26 380/380 | 26/27 19/19

Confirmed against the live API without the cache, so it is the service and not a cached
failure. The consequence that matters: there is exactly ONE complete season of PPDA from
this feed, so anything needing a multi-season press panel cannot be fitted on it — see
studies/press_k_multiseason.py, which is a recorded null for that reason.

Shot xG going back to 2021/22 is the useful half: the field is `expected_goals`, NOT
`xg`, and reading the wrong key returns None on every shot in every season, which looks
exactly like missing data.

RATE
----
The docs advertise no per-second or per-minute limit but keep "a single unadvertised
burst guard". `SLEEP` throttles anyway — a study pulling a full season is 380 requests
and there is no reason to be rude about it.
"""
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd

BASE = "https://api.pitchapi.dev/v1"
SLEEP = 0.12
USER_AGENT = "FPL-Project/1.0 (research; +https://pitchapi.dev)"
TIMEOUT = 30
RETRIES = 3

EPL = "l_4WFCIZ"          # English Premier League; 2021/2022 .. 2026/2027

# PitchAPI club spelling -> the frame spelling the rest of the repo uses (E0_recon,
# press_index, regime_panel all agree on it). Anything unlisted passes through, so a new
# club surfaces as an unmatched name in a join rather than silently mapping to nothing.
PITCH_TO_FRAME = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham", "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton", "AFC Bournemouth": "Bournemouth",
    "Leeds United": "Leeds", "West Ham United": "West Ham",
    "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester", "Sheffield United": "Sheffield United",
    "Luton Town": "Luton", "Norwich City": "Norwich", "Watford": "Watford",
    "West Bromwich Albion": "West Brom", "Queens Park Rangers": "QPR",
    "Stoke City": "Stoke", "Swansea City": "Swansea", "Cardiff City": "Cardiff",
}


def normalise_team(name):
    """PitchAPI spelling -> repo frame spelling. Idempotent."""
    return PITCH_TO_FRAME.get(name, name)


# ------------------------------------------------------------------ transport
def _key():
    k = os.environ.get("PITCHAPI_KEY", "").strip()
    if not k:
        raise RuntimeError(
            "PITCHAPI_KEY is not set. Get a key at https://pitchapi.dev/get-api-key "
            "and export it; it is deliberately not stored in the repo.")
    return k


def _cache_path(path, params=None):
    slug = path.strip("/").replace("/", "__")
    if params:
        slug += "__" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    slug = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in slug)
    return os.path.join(config.PITCH_CACHE, slug + ".json")


def get(path, params=None, force=False, cache=True, verbose=False):
    """GET one endpoint, cached. Returns the parsed `data` payload.

    A 404 ANALYTICS_UNAVAILABLE is NOT an error — the docs are explicit that event-feed
    coverage is not universal, so it is an ordinary answer meaning "this match was never
    rated". It is returned as None and cached as such, so a season pull does not refetch
    the same gaps on every run.
    """
    cp = _cache_path(path, params)
    if cache and not force and os.path.exists(cp):
        with open(cp, encoding="utf-8") as fh:
            return json.load(fh).get("data")

    url = BASE + path
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
    # A User-Agent is required, not optional: the default `Python-urllib/3.x` is refused
    # with a bare 403 while the identical request from curl succeeds. [VERIFIED 2026-08-31]
    req = urllib.request.Request(url, headers={"X-API-KEY": _key(),
                                               "Accept": "application/json",
                                               "User-Agent": USER_AGENT})
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                body = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            if e.code == 404:
                try:
                    code = json.loads(raw)["error"]["code"]
                except Exception:                                     # noqa: BLE001
                    code = ""
                if code == "ANALYTICS_UNAVAILABLE":
                    body = {"data": None}
                    break
                raise
            if e.code in (429, 500, 502, 503, 504):
                last = e
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"pitchapi GET {path} failed after {RETRIES}: {last}")

    if cache:
        os.makedirs(config.PITCH_CACHE, exist_ok=True)
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump(body, fh)
    time.sleep(SLEEP)
    if verbose:
        print(f"[pitchapi] {path} {'(no analytics)' if body.get('data') is None else 'ok'}")
    return body.get("data")


# ------------------------------------------------------------------ resources
def leagues(force=False):
    return get("/leagues", force=force)["leagues"]


def league_matches(league_id=EPL, season=None, status=None, force=False):
    p = {}
    if season:
        p["season"] = season
    if status:
        p["status"] = status
    return get(f"/leagues/{league_id}/matches", p or None, force=force)


def advanced(match_id, force=False):
    return get(f"/matches/{match_id}/advanced", force=force)


def advanced_players(match_id, force=False):
    return get(f"/matches/{match_id}/advanced/players", force=force)


def lineups(match_id, force=False):
    return get(f"/matches/{match_id}/lineups", force=force)


# ------------------------------------------------------------------ tidy frames
_GROUPS = ("possession_value", "passing", "carrying", "creation", "defending",
           "territory", "tempo", "goalkeeping", "shooting")


def _flatten(team_block, match, opp_name):
    row = {"match_id": match["id"], "date": match["date"],
           "team": normalise_team(team_block["team"]["name"]),
           "opp": normalise_team(opp_name),
           "actions": team_block.get("actions")}
    for g in _GROUPS:
        for k, v in (team_block.get(g) or {}).items():
            if isinstance(v, dict):                 # sca_breakdown and friends
                for k2, v2 in v.items():
                    row[f"{g}.{k}.{k2}"] = v2
            else:
                row[f"{g}.{k}"] = v
    return row


def team_match_advanced(season="2026/2027", league_id=EPL, matches=None,
                        force=False, verbose=True):
    """One row per team-match, every advanced group flattened to `group.metric`.

    Matches whose analytics are unavailable are skipped and counted, never filled — a
    zero here would be a real PPDA of zero, which is not a thing.
    """
    if matches is None:
        matches = league_matches(league_id, season, force=force)["matches"]
    rows, missing = [], []
    for i, m in enumerate(matches, 1):
        a = advanced(m["id"], force=force)
        if not a or not a.get("teams"):
            missing.append(m["id"])
            continue
        names = [t["team"]["name"] for t in a["teams"]]
        for j, tb in enumerate(a["teams"]):
            rows.append(_flatten(tb, m, names[1 - j] if len(names) == 2 else ""))
        if verbose and i % 50 == 0:
            print(f"[pitchapi] {i}/{len(matches)} matches")
    if verbose:
        print(f"[pitchapi] {season}: {len(rows)} team-matches from "
              f"{len(matches) - len(missing)}/{len(matches)} matches"
              + (f"; {len(missing)} without analytics" if missing else ""))
    d = pd.DataFrame(rows)
    if len(d):
        d["season"] = season
    return d


def selftest():
    """Offline. Exercises the flattening and the name map on a synthetic payload —
    the transport is not tested here because a selftest must not need the network."""
    match = {"id": "m_test", "date": "2026-08-21"}
    block = {"team": {"id": "t_1", "name": "Nottingham Forest"}, "actions": 800,
             "defending": {"ppda": 9.72, "opponent_passes": 243,
                           "defensive_actions": 25, "avg_defensive_action_x": 39},
             "passing": {"passes": 619, "progressive_passes": 28},
             "creation": {"sca": 40, "sca_breakdown": {"pass_live": 36, "shot": 0}},
             "territory": {"field_tilt": 68.9}}
    r = _flatten(block, match, "Coventry City")
    assert r["team"] == "Nott'm Forest", r["team"]
    assert r["opp"] == "Coventry", r["opp"]
    assert r["defending.ppda"] == 9.72
    assert r["passing.progressive_passes"] == 28
    assert r["creation.sca_breakdown.pass_live"] == 36, "nested breakdown must flatten"
    assert r["territory.field_tilt"] == 68.9
    # the numerator/denominator must survive: they are why this feed beats the proxy
    assert r["defending.opponent_passes"] / r["defending.defensive_actions"] == 9.72

    assert normalise_team("Brighton & Hove Albion") == "Brighton"
    assert normalise_team("Brighton") == "Brighton", "must be idempotent"
    assert normalise_team("Arsenal") == "Arsenal", "unlisted names pass through"

    cp = _cache_path("/matches/m_1/advanced", {"season": "2026/2027"})
    assert cp.endswith(".json") and "matches__m_1__advanced" in cp, cp
    assert "pk_" not in cp, "a key must never reach a cache path"
    print("pitchapi selftest ok")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--season", default="2026/2027")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        d = team_match_advanced(a.season)
        if len(d):
            cols = ["team", "opp", "defending.ppda", "territory.field_tilt",
                    "passing.progressive_passes", "carrying.progressive_carries"]
            print(d[[c for c in cols if c in d.columns]].head(12).to_string(index=False))
