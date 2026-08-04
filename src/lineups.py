"""
lineups.py — source-agnostic XI ingestion for apply_availability
================================================================
apply_availability collapses the start prior to near-certainty once an XI is
known: pass lineups={team: {"start":[names], "bench":[names]}}. This module
produces that dict from whichever source is available, in priority order:

  1. API-Football confirmed XI (~1h pre-KO)   -> from_apifootball(key, fixture_ids)
  2. A manual / predicted-XI file you maintain  -> from_file(path)   [CSV or JSON]

In pre-season (no confirmed XIs yet) the file path is the usable one: drop in
predicted XIs (e.g. from a team-news source) and the same machinery applies.
Names must match the model's web_name; team names must match the schedule's
short names (Man United, Tottenham, Nott'm Forest, Coventry, Hull, Ipswich...).

CSV schema:   team,player,role      role in {start,bench}
JSON schema:  {"Man City": {"start": [...], "bench": [...]}, ...}
"""
from __future__ import annotations
import json, os
import pandas as pd


def from_file(path: str) -> dict:
    """Load a manual/predicted XI file (CSV team,player,role  or  JSON)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        return {t: {"start": list(v.get("start", [])),
                    "bench": list(v.get("bench", []))} for t, v in d.items()}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tcol, pcol, rcol = cols["team"], cols["player"], cols["role"]
    out: dict = {}
    for _, r in df.iterrows():
        t = str(r[tcol]).strip()
        role = "start" if str(r[rcol]).strip().lower().startswith("s") else "bench"
        out.setdefault(t, {"start": [], "bench": []})[role].append(str(r[pcol]).strip())
    return out


def from_apifootball(key: str, fixture_ids, budget: int = 20) -> dict:
    """Confirmed XIs via API-Football (KEY REQUIRED, 1 request/fixture).
    fixture_ids: iterable of fixture ids for the target gameweek (from
    apifootball.fixtures()). Merges all fixtures into one {team:{...}} dict."""
    import apifootball as af
    client = af.Client(key=key, budget=budget)
    merged: dict = {}
    for fid in fixture_ids:
        try:
            merged.update(af.lineups(client, fid))
        except Exception as e:                       # skip fixtures without an XI yet
            print(f"  lineups: fixture {fid} unavailable ({e})")
    return merged


def gameweek_fixture_ids(key: str, gameweek: int, season: int = 2026,
                         budget: int = 20):
    """Resolve the API-Football fixture ids for a gameweek (1 request)."""
    import apifootball as af
    client = af.Client(key=key, budget=budget)
    fx = af.fixtures(client, season=season)
    if "gameweek" in fx.columns:
        fx = fx[fx.gameweek == gameweek]
    return list(fx["fixture_id"]) if "fixture_id" in fx.columns else []


def load_lineups(source: str = "none", *, path: str = None, key: str = None,
                 gameweek: int = None, fixture_ids=None, season: int = 2026) -> dict:
    """Dispatcher. source in {'none','file','apifootball'}.
    Returns {} when nothing is available (apply_availability then no-ops on XIs)."""
    if source == "file" and path:
        return from_file(path)
    if source == "apifootball" and key:
        if fixture_ids is None and gameweek is not None:
            fixture_ids = gameweek_fixture_ids(key, gameweek, season=season)
        return from_apifootball(key, fixture_ids or [])
    return {}


def coverage(lineups: dict) -> str:
    if not lineups:
        return "no lineups (start priors unchanged)"
    named = sum(1 for v in lineups.values() if v.get("start"))
    return f"{named} teams with a named XI, {len(lineups)} teams total"
