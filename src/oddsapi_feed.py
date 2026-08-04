"""
oddsapi_feed.py — the live half of handoff §7 item #1.

`betting_odds_ingest.py` already converts a saved Odds API payload into an
E0-format frame. What was missing is everything around it: fetching, snapshot
accumulation, and the guard that stops us fitting on an underidentified design.

Four things:

  1. fetch()          -- GET /v4/sports/soccer_epl/odds, key from $ODDS_API_KEY,
                         logs the x-requests-remaining / x-requests-used headers
                         so the free 500-credit budget is never a surprise.
  2. snapshots        -- every fetch is archived to CACHE_DIR as a timestamped
                         JSON. load_snapshots() merges them, keeping the LATEST
                         quote per event id. This matters: the API only prices
                         the next round or two, so a single call can never
                         identify the team model (see 3). Coverage accumulates
                         week over week and the market E0 gets stronger.
  3. identifiability  -- refuses to emit a market-ONLY E0 until the accumulated
                         fixture set has full column rank. One gameweek of odds
                         is 19 columns short of the 39 free parameters; fitting
                         on it yields plausible-looking numbers that are almost
                         entirely prior, not market.
  4. stack_e0()       -- the correct interim answer. Replicates the market rows
                         onto the backward-looking E0_recon so the market
                         reprices what it covers without being asked to identify
                         20 attacks and 20 defences on its own.

Usage (networked box):
    export ODDS_API_KEY=...
    python oddsapi_feed.py --fetch                       # 4 credits, archives snapshot
    python oddsapi_feed.py --build --recon /tmp/E0_recon.csv --out /tmp/E0_blend.csv
    python ab_market_vs_recon.py --recon /tmp/E0_recon.csv --market /tmp/E0_blend.csv

Offline wiring check (no network, no key):
    python oddsapi_feed.py --selftest
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd

import betting_odds_ingest as boi
from identifiability import rank_report, free_params

API_HOST = "https://api.the-odds-api.com"
SPORT = "soccer_epl"
CACHE_DIR = os.environ.get("ODDS_CACHE", os.path.expanduser("~/.fpl_odds_cache"))

# Every 26/27 club, so a naming drift at the source is caught loudly rather than
# silently dropping a team out of the design matrix.
EXPECTED_TEAMS = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea",
    "Coventry", "Crystal Palace", "Everton", "Fulham", "Hull", "Ipswich",
    "Leeds", "Liverpool", "Man City", "Man United", "Newcastle", "Nott'm Forest",
    "Sunderland", "Tottenham",
}


# ----------------------------------------------------------------- 1. fetch
def fetch(regions: str = "uk,eu", markets: str = "h2h,totals",
          sport: str = SPORT, key: str | None = None,
          archive: bool = True) -> list:
    """One /odds call. Cost = len(markets) * len(regions) credits (here: 4).

    `eu` is included deliberately: Pinnacle is an EU book and is the sharpest
    de-vig anchor available. `uk` alone silently excludes it.
    """
    import urllib.request, urllib.error, urllib.parse
    key = key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("Set ODDS_API_KEY (do not hardcode it; keys leak via shell history).")
    qs = urllib.parse.urlencode({"regions": regions, "markets": markets,
                                 "oddsFormat": "decimal", "apiKey": key})
    url = f"{API_HOST}/v4/sports/{sport}/odds/?{qs}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            events = json.loads(r.read().decode())
            used = r.headers.get("x-requests-used")
            left = r.headers.get("x-requests-remaining")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Odds API HTTP {e.code}: {e.read().decode()[:300]}")

    print(f"[fetch] {len(events)} {sport} events | credits used {used}, remaining {left}")
    if left is not None and int(left) < 50:
        print(f"[fetch] WARNING: only {left} credits left this period.")
    if not events:
        print("[fetch] Empty payload. Normal outside the betting window — books "
              "typically price only the next round or two. Retry nearer the deadline.")
    if archive:
        os.makedirs(CACHE_DIR, exist_ok=True)
        path = os.path.join(CACHE_DIR, f"{sport}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
        with open(path, "w") as fh:
            json.dump(events, fh)
        print(f"[fetch] archived -> {path}")
    return events


# ------------------------------------------------------- 2. snapshot merge
def load_snapshots(cache_dir: str = CACHE_DIR, sport: str = SPORT) -> list:
    """Merge every archived snapshot, keeping the most recent quote per event id.

    Coverage is cumulative across the season: GW1 odds fetched today plus GW2
    odds fetched next week give a design matrix neither call could support.
    """
    files = sorted(glob.glob(os.path.join(cache_dir, f"{sport}_*.json")))
    if not files:
        return []
    merged: dict[str, dict] = {}
    for f in files:                       # sorted ascending -> later overwrites
        try:
            for ev in json.load(open(f)):
                merged[ev["id"]] = ev
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"[cache] skipping unreadable snapshot {f}")
    print(f"[cache] {len(files)} snapshots -> {len(merged)} unique fixtures")
    return list(merged.values())


# --------------------------------------------------- 3. identifiability gate
def check_coverage(e0: pd.DataFrame, strict: bool = False) -> dict:
    """Team-name coverage + design-matrix rank of a market E0."""
    seen = set(e0.HomeTeam) | set(e0.AwayTeam)
    unknown = sorted(seen - EXPECTED_TEAMS)
    missing = sorted(EXPECTED_TEAMS - seen)
    teams = sorted(EXPECTED_TEAMS)
    rep = rank_report(e0.rename(columns={"HomeTeam": "HomeTeam", "AwayTeam": "AwayTeam"}), teams)

    print("=" * 74)
    print("MARKET E0 COVERAGE")
    print("=" * 74)
    print(f"  fixtures priced : {len(e0)}")
    print(f"  teams appearing : {len(seen)}/20")
    if unknown:
        print(f"  !! UNMAPPED NAMES (would corrupt the design): {unknown}")
    if missing:
        print(f"  teams not yet priced: {missing}")
    print(f"  design rank {rep['rank']} / {rep['needed']} needed  -> "
          f"{'IDENTIFIED' if rep['identified'] else 'UNDERIDENTIFIED'}")
    if unknown and strict:
        raise SystemExit("Refusing to build: unmapped team names. Add them to "
                         "betting_odds_ingest._ALIAS.")
    if not rep["identified"]:
        print("  -> market-only fit is NOT valid. Stack onto E0_recon (--recon).")
    return {**rep, "unknown": unknown, "missing": missing}


# ------------------------------------------------------------- 4. stacking
def stack_e0(recon: pd.DataFrame, market: pd.DataFrame,
             market_share: float = 0.35) -> pd.DataFrame:
    """Blend by replication: repeat the market rows until they make up
    `market_share` of the stacked design.

    Replication rather than a weight column because TeamModel.fit() reads plain
    E0 rows and has no weights argument -- this is source-agnostic and needs no
    change to the fit. The market rows carry forward-looking information (new
    managers, transfers, promoted sides); the recon rows carry the identification.

    market_share is a genuine bias/identification tradeoff and should be swept,
    not trusted. 0.35 is a starting point, not a calibrated value.
    """
    if market.empty:
        print("[stack] no market rows; returning recon unchanged")
        return recon.copy()
    n_r = len(recon)
    reps = max(1, int(round(market_share * n_r / ((1 - market_share) * len(market)))))
    stacked = pd.concat([recon] + [market] * reps, ignore_index=True)
    got = reps * len(market) / len(stacked)
    print(f"[stack] {n_r} recon + {len(market)}x{reps} market = {len(stacked)} rows "
          f"(market share {got:.1%}, target {market_share:.0%})")
    return stacked


# -------------------------------------------------------------- build path
def build(recon_path: str | None, out_path: str, consensus: str = "avg",
          method: str = "shin", rho: float = 0.0, market_share: float = 0.35,
          cache_dir: str = CACHE_DIR, events: list | None = None) -> pd.DataFrame:
    events = events if events is not None else load_snapshots(cache_dir)
    if not events:
        raise SystemExit("No snapshots. Run --fetch first (needs ODDS_API_KEY).")
    fx = boi.oddsapi_to_fixtures(events)
    if fx.empty:
        raise SystemExit("No fixture had both 1X2 and an O/U 2.5 anchor. The "
                         "inversion needs the total; retry nearer the deadline.")
    mkt = boi.fixtures_to_e0(fx, consensus=consensus, method=method, rho=rho)
    cov = check_coverage(mkt)

    if recon_path:
        recon = pd.read_csv(recon_path)
        cols = [c for c in recon.columns if c in mkt.columns]
        out = stack_e0(recon, mkt[cols].reindex(columns=recon.columns), market_share)
    else:
        if not cov["identified"]:
            raise SystemExit("Market-only E0 is underidentified. Pass --recon.")
        out = mkt
    out.to_csv(out_path, index=False)
    print(f"\n-> {out_path}  ({len(out)} rows)")
    print(mkt[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "mkt_total"]].round(3).to_string(index=False))
    return out


# --------------------------------------------------------------- selftest
def _synthetic_payload() -> list:
    """SYNTHETIC GW1 payload for offline wiring validation ONLY. Real 26/27
    fixtures, invented prices. Never use these numbers for decisions."""
    fixtures = [("Arsenal", "Coventry", 1.25, 6.0, 11.0, 1.80, 2.05),
                ("Manchester City", "Bournemouth", 1.20, 7.0, 13.0, 1.62, 2.30),
                ("Everton", "Crystal Palace", 2.40, 3.30, 3.05, 2.30, 1.62),
                ("Brentford", "Tottenham Hotspur", 2.90, 3.50, 2.45, 1.95, 1.88),
                ("Nottingham Forest", "Leeds United", 2.05, 3.40, 3.70, 2.15, 1.72),
                ("Brighton and Hove Albion", "Aston Villa", 2.45, 3.45, 2.85, 1.90, 1.92),
                ("Ipswich Town", "Sunderland", 2.70, 3.30, 2.70, 2.20, 1.68),
                ("Fulham", "Chelsea", 3.20, 3.50, 2.25, 2.00, 1.82),
                ("Newcastle United", "Liverpool", 2.85, 3.60, 2.40, 1.75, 2.10),
                ("Hull City", "Manchester United", 4.20, 3.70, 1.85, 2.05, 1.78)]
    out = []
    for i, (h, a, oh, od, oa, ov, un) in enumerate(fixtures):
        out.append({
            "id": f"synthetic{i}", "sport_key": SPORT,
            "commence_time": "2026-08-21T19:00:00Z", "home_team": h, "away_team": a,
            "bookmakers": [{
                "key": bk, "title": bk, "markets": [
                    {"key": "h2h", "outcomes": [{"name": h, "price": oh * m},
                                                {"name": a, "price": oa * m},
                                                {"name": "Draw", "price": od * m}]},
                    {"key": "totals", "outcomes": [{"name": "Over", "point": 2.5, "price": ov * m},
                                                   {"name": "Under", "point": 2.5, "price": un * m}]}]}
                for bk, m in [("pinnacle", 1.02), ("betfair_ex_uk", 1.01), ("williamhill", 0.99)]]})
    return out


def selftest():
    print("### SELFTEST — synthetic GW1 payload, no network, no key\n")
    ev = _synthetic_payload()
    fx = boi.oddsapi_to_fixtures(ev)
    assert len(fx) == 10, f"parser dropped fixtures: {len(fx)}/10"
    print(f"[1] parsed {len(fx)}/10 fixtures from Odds API JSON  OK")

    mkt = boi.fixtures_to_e0(fx, method="shin")
    assert len(mkt) == 10 and mkt.FTHG.between(0.1, 5).all()
    print(f"[2] inverted to lambdas, range "
          f"{mkt[['FTHG','FTAG']].values.min():.2f}-{mkt[['FTHG','FTAG']].values.max():.2f}  OK")

    cov = check_coverage(mkt)
    assert not cov["unknown"], f"unmapped names: {cov['unknown']}"
    assert cov["teams_seen"] == 20
    assert not cov["identified"], "one GW should NOT be identified"
    print(f"\n[3] all 20 clubs mapped; correctly flagged underidentified  OK")

    recon = pd.DataFrame({"Div": ["E0"] * 380, "Date": [""] * 380,
                          "HomeTeam": ["Arsenal"] * 380, "AwayTeam": ["Chelsea"] * 380,
                          "FTHG": 1.5, "FTAG": 1.2, "HxG": 1.5, "AxG": 1.2, "FTR": "H",
                          "mkt_pH": np.nan, "mkt_pA": np.nan, "mkt_total": np.nan})
    for share in (0.20, 0.35, 0.50):
        st = stack_e0(recon, mkt, share)
        got = (len(st) - 380) / len(st)
        assert abs(got - share) < 0.08, f"share off: {got:.3f} vs {share}"
    print("[4] stacking hits target market share at 20/35/50%  OK")

    m_prop = boi.fixtures_to_e0(fx, method="multiplicative")
    d = (mkt.FTHG - m_prop.FTHG).abs()
    assert d.max() > 1e-4, "shin and multiplicative identical -> de-vig regression"
    print(f"[5] shin vs multiplicative differ (max dlambda {d.max():.4f})  OK")

    print("\nSELFTEST PASSED — chain is wired. Supply a real payload with --fetch.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--payload", help="a saved Odds API JSON, instead of the cache")
    ap.add_argument("--recon", default=None, help="E0_recon.csv to stack onto")
    ap.add_argument("--out", default="/tmp/E0_blend.csv")
    ap.add_argument("--market-share", type=float, default=0.35)
    ap.add_argument("--consensus", default="avg", choices=["avg", "max", "b365"])
    ap.add_argument("--method", default="shin", choices=["multiplicative", "shin"])
    ap.add_argument("--rho", type=float, default=0.0)
    a = ap.parse_args()

    if a.selftest:
        selftest(); sys.exit(0)
    if a.fetch:
        fetch()
    if a.build or a.payload:
        ev = json.load(open(a.payload)) if a.payload else None
        build(a.recon, a.out, a.consensus, a.method, a.rho, a.market_share, events=ev)
    if not (a.fetch or a.build or a.payload):
        ap.print_help()
