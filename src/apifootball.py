"""
apifootball.py — API-Football (api-sports.io) loader   [FREE TIER, KEY REQUIRED]
================================================================================
Fills the single biggest remaining gaps in the model: INJURIES and LINEUPS
(plus transfers and dated fixtures). Free tier = 100 requests/day, all endpoints.
Sign up at api-sports.io, then set APIFOOTBALL_KEY or pass key=.

    https://v3.football.api-sports.io   header: x-apisports-key

QUOTA DISCIPLINE (100/day is tight, so this module is built around it):
  * every response is cached to disk with a TTL; repeat runs cost 0 requests
  * bulk endpoints only — one call per league/season/matchday, never per player
  * `Client.budget` caps requests per run and refuses to exceed it
  * `status()` reports remaining quota from the API's own headers

Typical weekly cost: fixtures 1 (cached all season) + injuries 1 + lineups 1 per
fixture you care about. Well inside the free tier if you cache.

NOTE: for competitions that have not kicked off, API-Football's coverage flags
are false and endpoints return empty — expect little for 26/27 until August.
"""
from __future__ import annotations
import json, os, time, urllib.request, urllib.parse
import numpy as np, pandas as pd

BASE = "https://v3.football.api-sports.io"
PL_LEAGUE_ID = 39
CACHE_DIR = os.path.expanduser("~/.cache/apifootball")

# API-Football club names -> model short names
TEAM_NORM = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Tottenham": "Tottenham", "Newcastle": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Brighton": "Brighton",
    "Bournemouth": "Bournemouth", "Leeds": "Leeds", "Coventry": "Coventry",
    "Hull City": "Hull", "Ipswich": "Ipswich", "Wolves": "Wolves",
}
def norm_team(t): return TEAM_NORM.get(str(t).strip(), str(t).strip())


class Client:
    def __init__(self, key=None, budget=20, cache_dir=CACHE_DIR, ttl_hours=12):
        self.key = key or os.environ.get("APIFOOTBALL_KEY")
        if not self.key:
            raise ValueError("No API-Football key. Set APIFOOTBALL_KEY or pass key=.")
        self.budget = budget; self.used = 0
        self.cache_dir = cache_dir; self.ttl = ttl_hours * 3600
        self.remaining = None
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, endpoint, params):
        k = endpoint.replace("/", "_") + "_" + urllib.parse.urlencode(sorted(params.items()))
        safe = "".join(c if c.isalnum() or c in "_-=" else "_" for c in k)[:180]
        return os.path.join(self.cache_dir, safe + ".json")

    def get(self, endpoint, force=False, **params):
        """Cached GET. Returns the `response` list. Costs a request only on miss."""
        path = self._cache_path(endpoint, params)
        if not force and os.path.exists(path) and time.time() - os.path.getmtime(path) < self.ttl:
            with open(path) as f:
                return json.load(f)
        if self.used >= self.budget:
            raise RuntimeError(f"request budget ({self.budget}) exhausted this run")
        url = f"{BASE}/{endpoint.lstrip('/')}?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"x-apisports-key": self.key})
        with urllib.request.urlopen(req, timeout=25) as r:
            self.remaining = r.headers.get("x-ratelimit-requests-remaining")
            payload = json.loads(r.read())
        self.used += 1
        data = payload.get("response", [])
        with open(path, "w") as f:
            json.dump(data, f)
        return data

    def status(self):
        d = self.get("status")
        return d


# ---------------------------------------------------------------------------
# Endpoint wrappers (all bulk — one request each)
# ---------------------------------------------------------------------------
def fixtures(client, season=2026, league=PL_LEAGUE_ID) -> pd.DataFrame:
    """Dated fixture list: gives KICK-OFF DATES/TIMES the schedule file lacks,
    plus the fixture ids needed for lineups. 1 request, cache all season."""
    rows = []
    for f in client.get("fixtures", league=league, season=season):
        fx, tm = f["fixture"], f["teams"]
        rows.append({
            "fixture_id": fx["id"], "kickoff": fx["date"], "status": fx["status"]["short"],
            "round": f["league"].get("round"),
            "home": norm_team(tm["home"]["name"]), "away": norm_team(tm["away"]["name"]),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["kickoff"] = pd.to_datetime(df["kickoff"], errors="coerce", utc=True)
        # "Regular Season - 7" -> 7
        df["gameweek"] = df["round"].str.extract(r"(\d+)").astype(float)
    return df.sort_values("kickoff").reset_index(drop=True)


def injuries(client, season=2026, league=PL_LEAGUE_ID) -> pd.DataFrame:
    """Current injury/suspension list -> availability prior. 1 request."""
    rows = []
    for i in client.get("injuries", league=league, season=season):
        rows.append({"name": i["player"]["name"], "team": norm_team(i["team"]["name"]),
                     "type": i["player"].get("type"), "reason": i["player"].get("reason")})
    return pd.DataFrame(rows).drop_duplicates(subset=["name"])


def lineups(client, fixture_id) -> dict:
    """Confirmed XI for one fixture (~1h pre-KO) -> collapses the start prior.
    Returns {team: {"start":[names], "bench":[names]}}. 1 request per fixture."""
    out = {}
    for side in client.get("fixtures/lineups", fixture=fixture_id):
        t = norm_team(side["team"]["name"])
        out[t] = {"start": [p["player"]["name"] for p in side.get("startXI", [])],
                  "bench": [p["player"]["name"] for p in side.get("substitutes", [])]}
    return out


def transfers_in(client, team_id) -> pd.DataFrame:
    """Recent transfers for a club -> roster re-pointing. 1 request per club
    (use sparingly on the free tier; prefer the FPL bootstrap for rosters)."""
    rows = []
    for t in client.get("transfers", team=team_id):
        for tr in t.get("transfers", []):
            rows.append({"name": t["player"]["name"], "date": tr.get("date"),
                         "from": norm_team(tr["teams"]["out"]["name"]),
                         "to": norm_team(tr["teams"]["in"]["name"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bridge into the model's signal frame
# ---------------------------------------------------------------------------
# API-Football injury `type` -> availability probability
TYPE_TO_CHANCE = {"Missing Fixture": 0.0, "Questionable": 0.5, "Doubtful": 0.5}

def merge_into_signals(signals: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """Overlay API-Football injuries onto the FPL signal frame. FPL's own
    `chance_of_playing` is authoritative when present; this fills the gaps and
    catches news FPL hasn't flagged yet. Matching is by surname-insensitive key."""
    if signals is None or not len(inj):
        return signals
    s = signals.copy()
    def key(n): return str(n).lower().strip().split()[-1]      # surname
    ikey = {key(r["name"]): r for _, r in inj.iterrows()}
    for i, r in s.iterrows():
        k = key(r["name"])
        if k in ikey and r.get("status", "a") == "a" and r.get("chance_play", 1.0) >= 1.0:
            t = ikey[k].get("type")
            ch = TYPE_TO_CHANCE.get(t, 0.0)
            s.at[i, "chance_play"] = ch
            s.at[i, "status"] = "i" if ch == 0.0 else "d"
            s.at[i, "news"] = f"API-Football: {t} - {ikey[k].get('reason')}"
    return s


def kickoff_dates(fx: pd.DataFrame) -> pd.DataFrame:
    """Gameweek -> first/last kickoff, and per-fixture dates for the schedule."""
    if not len(fx):
        return pd.DataFrame()
    g = fx.dropna(subset=["gameweek"]).groupby("gameweek")["kickoff"]
    return pd.DataFrame({"first_kickoff": g.min(), "last_kickoff": g.max(),
                         "fixtures": g.size()}).reset_index()


if __name__ == "__main__":
    try:
        c = Client(budget=3)
        print("status:", json.dumps(c.status(), indent=2)[:400])
        fx = fixtures(c); print(f"\nfixtures: {len(fx)}"); print(fx.head().to_string(index=False))
        print("\nkickoff dates by GW:"); print(kickoff_dates(fx).head().to_string(index=False))
        inj = injuries(c); print(f"\ninjuries: {len(inj)}"); print(inj.head().to_string(index=False))
        print(f"\nrequests used: {c.used}, remaining today: {c.remaining}")
    except Exception as e:
        print(f"[apifootball] {type(e).__name__}: {e}")
        print("Set APIFOOTBALL_KEY (free signup at api-sports.io) and re-run.")
</content>
