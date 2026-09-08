from __future__ import annotations
import config
"""
solio_market.py — Solio's market-implied team lambda, its movement, and an E0 re-anchor.
========================================================================================
WHAT THIS IS, AND WHAT IT IS NOT
---------------------------------
The ask was "read Solio's betting odds and movement tracking". **Neither exists.**
Verified against the feed's own agent documentation (https://fpl.solioanalytics.com/llms.txt)
and the full JSON schema of /api/data/latest.json on 2026-08-27:

  - No odds fields. No bookmaker prices, no 1X2, no over/under, nothing de-vigged.
  - No movement or history fields. The endpoint publishes `latest` only; there is no
    archive, no `generatedAt` series, no deltas. `generatedAt` stamps ONE snapshot.
  - "Clean sheet odds" in their tables is a PROJECTED PROBABILITY (`csProb`), not a price.

What Solio does publish, and what this module uses, is one step downstream of the market:
`bestCleanSheets` and `bestAttackingFixtures` each carry, per team-fixture,

    prGoalsFor, prGoalsAgainst, csProb

Solio states its model is "built on efficient sports markets" and that "projections update
whenever underlying market data moves". So (prGoalsFor, prGoalsAgainst) is a
**market-derived lambda pair** — the same quantity `betting_odds_ingest.fixtures_to_e0`
reconstructs by de-vigging 1X2 + O/U and inverting a Poisson, except already inverted.
That is a strictly better input than the odds would be: no de-vig assumption, no
Poisson inversion, no bookmaker margin model. It is also a WORSE one in a specific way —
it is Solio's estimate of the market, not the market, so their modelling choices are
baked in and unobservable.

COVERAGE IS INCIDENTAL AND MUST BE CHECKED EVERY RUN
-----------------------------------------------------
Both source lists are TOP-10 truncations. They are not a fixture list. Coverage is
complete only because each record carries BOTH sides' lambda, so a record for
"Arsenal away at AVL" recovers Aston Villa's lambda without Villa appearing anywhere.
On the 2026-08-27 GW2 payload the union of the two lists spans all 10 fixtures and all
20 clubs exactly, with no conflicting duplicates.

**That is luck, not a guarantee.** A double gameweek, a blank, or heavier overlap
between the two top-10s would leave fixtures uncovered, and a partial set is not a
harmless partial set — the lists are ranked by clean-sheet probability and attacking
output, so what drops out is systematically the weak and the badly-fixtured. Feeding a
censored subset into `TeamModel` is selection on the dependent variable and would bias
att[]/dfn[] toward exactly the compression the GW1 scoring doc flagged as [CHECK].
`fixture_lambdas()` therefore returns a coverage report and `stack_e0()` REFUSES to
build unless coverage is complete. Do not relax that into a warning.

THREE REASONS THE E0 PATH IS OFF BY DEFAULT
--------------------------------------------
1. **Double counting.** `market_odds.py` already blends a 7 Aug outright snapshot into
   the ClubElo path at MARKET_WEIGHT=0.6, on by default in gw_board.py. Solio lambda is
   the same market seen through a different window. Stacking both counts it twice.
   `stack_e0` raises unless MARKET_ODDS=off, rather than quietly compounding.
2. **No calibration transfers.** `studies/inseason_weight.py` fitted W_MATCH on REALISED
   early matches — a club's own played evidence against its prior. Solio lambda is a
   forward-looking prior for an UNPLAYED fixture. Appending it as an E0 row is the
   `betting_odds_ingest` convention and needs no new machinery, but the weight is
   uncalibrated. W_FIXTURE below is exposed, defaulted to 1, and explicitly NOT justified
   by that study. Sweep it; do not assert it.
3. **Orthogonality.** The market-orthogonality guard exists so a new team-level signal
   has to beat the market before it enters TeamModel. This signal IS the market, so it
   cannot clear a test defined as beating one. It is a re-anchor of an existing channel,
   not a new channel, and should be argued and swept as such.

WHAT IS UNCONDITIONALLY SAFE, AND IS THE POINT OF THE MODULE
-------------------------------------------------------------
The snapshot store. The feed keeps no history, so every 4h refresh not captured is a
movement observation that cannot be recovered later. `fetch()` writes a timestamped,
deduplicated JSON snapshot under config.SOLIO_SNAPSHOTS; `movement()` diffs consecutive
snapshots into per-team lambda deltas with elapsed hours. That series is the thing the
repo has none of, it costs nothing, and it is the only way to ever test whether market
movement predicts residual model error. Start it now; test it at GW5-6, when
inseason_weight says the weight stops being 0.01.

Run:  python src/solio_market.py --selftest
      python src/solio_market.py --fetch            (store a snapshot, no model change)
      python src/solio_market.py --report
      python src/solio_market.py --movement
"""
import os
import re
import json
import glob
import datetime as _dt
import numpy as np
import pandas as pd
import betting_odds_ingest as boi
import inseason as ins

SOLIO_JSON_URL = "https://fpl.solioanalytics.com/api/data/latest.json"

# [JUDGMENT] Uncalibrated. See reason 2 in the docstring. Row replication, so integer.
W_FIXTURE = 1

# Solio names both ways: `team` is a full-ish name, `opponent` is a short code. One map
# covers both -> our canonical schedule names. Extends solio_ensemble.TEAM_MAP, which
# only handles the short codes.
_FULL = {
    "Man Utd": "Man United", "Manchester Utd": "Man United",
    "Man City": "Man City", "Manchester City": "Man City",
    "Leeds United": "Leeds", "Leeds Utd": "Leeds",
    "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Nottingham Forest": "Nott'm Forest", "Nott'm Forest": "Nott'm Forest",
    "Brighton & Hove Albion": "Brighton", "Brighton": "Brighton",
    "AFC Bournemouth": "Bournemouth", "Bournemouth": "Bournemouth",
    "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Newcastle United": "Newcastle", "West Ham United": "West Ham",
}


def canon_team(name):
    """Solio team label (full name OR short code) -> our canonical name."""
    import solio_ensemble as se
    n = str(name).strip()
    if n in _FULL:
        return _FULL[n]
    if n in se.TEAM_MAP:
        return se.TEAM_MAP[n]
    return n


# ------------------------------------------------------------------ snapshots
def _stamp(generated_at):
    """generatedAt ISO -> a filesystem-safe, sortable snapshot key."""
    s = re.sub(r"[^0-9T]", "", str(generated_at).replace("Z", ""))
    return s[:15] if len(s) >= 15 else s


def snapshot_path(payload, snapdir=None):
    snapdir = snapdir or config.SOLIO_SNAPSHOTS
    gw = payload.get("gameweek", "NA")
    return os.path.join(snapdir, f"solio_gw{gw}_{_stamp(payload.get('generatedAt'))}.json")


def fetch(url=SOLIO_JSON_URL, snapdir=None, timeout=20, write=True, verbose=True):
    """GET the live JSON and store it as a timestamped snapshot.

    Deduplicates on `generatedAt`: the feed refreshes every 4h but nothing stops us
    polling more often, and storing the same payload twice would fabricate a
    zero-movement observation that never happened. Written atomically and UTF-8 for the
    reason documented in solio_ensemble.fetch_solio — a partial write must not destroy
    a good snapshot, and cp1252 cannot encode this feed on Windows.
    """
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "fpl-project/1.0"})
    payload = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8"))
    if not write:
        return payload, None
    path = snapshot_path(payload, snapdir)
    if os.path.exists(path):
        if verbose:
            print(f"[solio] already have {os.path.basename(path)} - not re-storing")
        return payload, path
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    if verbose:
        print(f"[solio] stored {os.path.basename(path)}  (generated {payload.get('generatedAt')})")
    return payload, path


def load_snapshots(snapdir=None, gw=None):
    """Every stored snapshot, oldest first, as (path, payload)."""
    snapdir = snapdir or config.SOLIO_SNAPSHOTS
    pat = f"solio_gw{gw}_*.json" if gw is not None else "solio_gw*_*.json"
    out = []
    for p in sorted(glob.glob(os.path.join(snapdir, pat))):
        with open(p, encoding="utf-8") as fh:
            out.append((p, json.load(fh)))
    out.sort(key=lambda t: str(t[1].get("generatedAt")))
    return out


# ------------------------------------------------------------ team lambda
_LAMBDA_LISTS = ("bestCleanSheets", "bestAttackingFixtures")


def team_lambdas(payload):
    """One row per (team, fixture) from the union of the two ranked lists.

    Rows appearing in both lists are identical by construction; a genuine disagreement
    would mean the payload is internally inconsistent, so it raises rather than picking
    a side.
    """
    rows, seen = [], {}
    for key in _LAMBDA_LISTS:
        for r in payload.get(key, []) or []:
            fixtures = r.get("fixtures") or []
            team = canon_team(r["team"])
            for f in fixtures:
                opp = canon_team(f["opponent"])
                rec = dict(team=team, opponent=opp, is_home=bool(f["isHome"]),
                           lam_for=float(r["prGoalsFor"]),
                           lam_against=float(r["prGoalsAgainst"]),
                           cs_prob=float(r["csProb"]), src=key)
                k = (team, opp, rec["is_home"])
                if k in seen:
                    prev = seen[k]
                    for c in ("lam_for", "lam_against", "cs_prob"):
                        if abs(prev[c] - rec[c]) > 1e-9:
                            raise ValueError(
                                f"solio payload inconsistent for {k}: {c} "
                                f"{prev[c]} vs {rec[c]} across {prev['src']}/{key}")
                    continue
                seen[k] = rec
                rows.append(rec)
    d = pd.DataFrame(rows)
    if len(d):
        d["gw"] = payload.get("gameweek")
        d["generated_at"] = payload.get("generatedAt")
    return d


def fixture_lambdas(payload, expect_fixtures=None):
    """Per-fixture (home, away, lam_home, lam_away) plus a coverage report.

    Each source row carries BOTH sides' lambda, so one listed team recovers the whole
    fixture. `covered` is the count of distinct fixtures recovered; `complete` compares
    it to `expect_fixtures` when given. Read the docstring at the top of this module
    before using an incomplete set for anything.
    """
    d = team_lambdas(payload)
    fx = {}
    for _, r in d.iterrows():
        home, away = (r["team"], r["opponent"]) if r["is_home"] else (r["opponent"], r["team"])
        lam_h, lam_a = ((r["lam_for"], r["lam_against"]) if r["is_home"]
                        else (r["lam_against"], r["lam_for"]))
        k = (home, away)
        if k in fx and (abs(fx[k][0] - lam_h) > 1e-9 or abs(fx[k][1] - lam_a) > 1e-9):
            raise ValueError(f"solio payload gives two different lambda pairs for {k}")
        fx[k] = (lam_h, lam_a)
    out = pd.DataFrame([dict(home=h, away=a, lam_home=lh, lam_away=la)
                        for (h, a), (lh, la) in sorted(fx.items())])
    teams = set(out["home"]) | set(out["away"]) if len(out) else set()
    rep = dict(gw=payload.get("gameweek"), generated_at=payload.get("generatedAt"),
               covered=len(out), teams=len(teams),
               expected=expect_fixtures,
               complete=(None if expect_fixtures is None else len(out) == expect_fixtures))
    return out, rep


# ------------------------------------------------------------------ movement
def movement(snapdir=None, gw=None, snapshots=None):
    """Per-team lambda deltas between consecutive snapshots.

    This is the module's reason to exist. The feed keeps no history, so this series can
    only ever be as long as the snapshots we chose to store.
    """
    snaps = snapshots if snapshots is not None else load_snapshots(snapdir, gw)
    if len(snaps) < 2:
        return pd.DataFrame(columns=["gw", "team", "from", "to", "hours", "d_lam_for",
                                     "d_lam_against", "d_cs_prob"])
    rows = []
    for (_, prev), (_, cur) in zip(snaps[:-1], snaps[1:]):
        if prev.get("gameweek") != cur.get("gameweek"):
            continue                       # deltas across a gameweek boundary are meaningless
        a = team_lambdas(prev).set_index("team")
        b = team_lambdas(cur).set_index("team")
        t0 = _parse_iso(prev.get("generatedAt"))
        t1 = _parse_iso(cur.get("generatedAt"))
        hours = (t1 - t0).total_seconds() / 3600.0 if (t0 and t1) else np.nan
        for team in sorted(set(a.index) & set(b.index)):
            ra, rb = a.loc[team], b.loc[team]
            if ra["opponent"] != rb["opponent"] or ra["is_home"] != rb["is_home"]:
                continue                   # fixture changed under us; not a price move
        # A snapshot backfilled from the published markdown carries 2dp, not full float.
        # Differencing it against a JSON snapshot manufactures up to +-0.005 of movement
        # that never happened, so carry the floor out rather than let it read as signal.
        noise = max(float(prev.get("_precision", 0.0)),
                    float(cur.get("_precision", 0.0))) / 2.0
        for team in sorted(set(a.index) & set(b.index)):
            ra, rb = a.loc[team], b.loc[team]
            if ra["opponent"] != rb["opponent"] or ra["is_home"] != rb["is_home"]:
                continue                   # fixture changed under us; not a price move
            d_for = float(rb["lam_for"] - ra["lam_for"])
            rows.append(dict(gw=cur.get("gameweek"), team=team,
                             **{"from": prev.get("generatedAt"), "to": cur.get("generatedAt")},
                             hours=round(hours, 2),
                             d_lam_for=d_for,
                             d_lam_against=float(rb["lam_against"] - ra["lam_against"]),
                             d_cs_prob=float(rb["cs_prob"] - ra["cs_prob"]),
                             noise_floor=noise,
                             signal=bool(abs(d_for) > noise)))
    return pd.DataFrame(rows)


def _parse_iso(s):
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


# ------------------------------------------------------------------ E0 path
def stack_e0(payload, expect_fixtures=10, e0_path=None, out=None, weight=W_FIXTURE,
             allow_double_count=False, verbose=True):
    """Append Solio's market lambda to the prior-season E0 as forward-looking rows.

    OFF BY DEFAULT at the caller. Two hard refusals, both deliberate:

      * incomplete coverage -> raise. A top-10 truncation is a censored sample, not a
        small one; see the module docstring.
      * MARKET_ODDS not 'off' -> raise. market_odds.py already blends the outright
        market into the ClubElo path at weight 0.6. Pass allow_double_count=True only
        with a reason you have written down.

    Reuses `inseason.e0_rows`, so the odds columns encode lambda through exactly the
    round trip TeamModel.fit expects and no new convention enters the repo.
    """
    fx, rep = fixture_lambdas(payload, expect_fixtures=expect_fixtures)
    if expect_fixtures is not None and not rep["complete"]:
        raise ValueError(
            f"solio coverage incomplete: {rep['covered']} of {expect_fixtures} fixtures, "
            f"{rep['teams']} clubs. The source lists are top-10 truncations ranked by "
            f"clean-sheet and attacking output, so the missing fixtures are not random. "
            f"Refusing to re-anchor the team layer on a censored sample.")
    if not allow_double_count and os.environ.get("MARKET_ODDS", "").lower() != "off":
        raise ValueError(
            "market_odds is active (MARKET_ODDS != 'off'), which already blends the "
            "outright market into ClubElo at weight 0.6. Stacking Solio lambda on top "
            "counts the same market twice. Set MARKET_ODDS=off, or pass "
            "allow_double_count=True deliberately.")
    matches = pd.DataFrame({
        "date": pd.NaT, "home": fx["home"], "away": fx["away"],
        "xg_home": fx["lam_home"], "xg_away": fx["lam_away"],
    })
    path, n = ins.stack_e0(matches, e0_path=e0_path,
                           out=out or os.path.join(config.SCRATCH, "E0_recon_solio.csv"),
                           weight=int(weight), weight_promoted=int(weight),
                           verbose=False)
    if verbose:
        print(f"[solio-market] team layer re-anchored: {len(fx)} GW{rep['gw']} fixtures "
              f"x{int(weight)} -> {n} rows -> {path}")
        print(f"[solio-market] NOTE weight={weight} is UNCALIBRATED "
              f"(inseason_weight fitted realised matches, not forward market lambda)")
    return path, n, rep


# ------------------------------------------------------------------ reporting
def report(payload=None, snapdir=None):
    if payload is None:
        snaps = load_snapshots(snapdir)
        if not snaps:
            print("[solio] no snapshots stored yet - run --fetch")
            return
        payload = snaps[-1][1]
    fx, rep = fixture_lambdas(payload, expect_fixtures=10)
    print(f"Solio GW{rep['gw']}  generated {rep['generated_at']}")
    print(f"coverage: {rep['covered']} fixtures, {rep['teams']} clubs, "
          f"complete={rep['complete']}")
    d = fx.copy()
    d["total"] = d["lam_home"] + d["lam_away"]
    d["sup"] = d["lam_home"] - d["lam_away"]
    print(d.round(3).to_string(index=False))


def main(argv=None):
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")     # piped stdout is cp1252 on Windows
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="Solio market-implied team lambda: snapshot store, movement, E0 re-anchor")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="store a snapshot (no model change)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--movement", action="store_true")
    ap.add_argument("--gw", type=int, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.fetch:
        fetch()
    if a.report:
        report()
    if a.movement:
        m = movement(gw=a.gw)
        if not len(m):
            print("[solio] need at least two snapshots of the same gameweek to show movement")
        else:
            print(m.round(4).to_string(index=False))
    if not (a.fetch or a.report or a.movement):
        ap.print_help()
    return 0


# ------------------------------------------------------------------ selftest
def _synthetic(gw=2, generated="2026-08-27T12:00:00.000Z", bump=0.0):
    """Two-fixture payload shaped exactly like the live feed, including the top-10
    truncation behaviour: one fixture listed from both sides, one from a single side."""
    return {
        "generatedAt": generated, "gameweek": gw,
        "bestCleanSheets": [
            {"team": "Arsenal", "fixtures": [{"opponent": "AVL", "isHome": False}],
             "prGoalsFor": 1.90 + bump, "prGoalsAgainst": 0.85, "csProb": 0.43},
            {"team": "Spurs", "fixtures": [{"opponent": "NEW", "isHome": True}],
             "prGoalsFor": 1.64, "prGoalsAgainst": 1.43, "csProb": 0.24},
        ],
        "bestAttackingFixtures": [
            {"team": "Arsenal", "fixtures": [{"opponent": "AVL", "isHome": False}],
             "prGoalsFor": 1.90 + bump, "prGoalsAgainst": 0.85, "csProb": 0.43},
        ],
    }


def selftest():
    import tempfile
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # --- naming: full names and short codes both canonicalise -------------------
    assert canon_team("Man Utd") == "Man United"
    assert canon_team("Spurs") == "Tottenham"
    assert canon_team("MUN") == "Man United"
    assert canon_team("Arsenal") == "Arsenal"

    p = _synthetic()

    # --- a duplicate across the two lists is absorbed, not double-counted -------
    tl = team_lambdas(p)
    assert len(tl) == 2, tl
    assert set(tl["team"]) == {"Arsenal", "Tottenham"}

    # --- one listed side recovers the whole fixture ------------------------------
    fx, rep = fixture_lambdas(p, expect_fixtures=2)
    assert rep["covered"] == 2 and rep["teams"] == 4 and rep["complete"], rep
    row = fx[fx["home"] == "Aston Villa"].iloc[0]
    assert abs(row["lam_home"] - 0.85) < 1e-9 and abs(row["lam_away"] - 1.90) < 1e-9
    # Villa appears in neither list yet its lambda is exact.

    # --- an inconsistent payload raises rather than picking a side ---------------
    bad = _synthetic()
    bad["bestAttackingFixtures"][0]["prGoalsFor"] = 9.9
    try:
        team_lambdas(bad)
        raise AssertionError("inconsistent payload must raise")
    except ValueError:
        pass

    # --- censored coverage is refused, not warned about --------------------------
    try:
        stack_e0(p, expect_fixtures=10, allow_double_count=True, verbose=False)
        raise AssertionError("incomplete coverage must raise")
    except ValueError as e:
        assert "censored" in str(e)

    # --- double-count guard fires unless MARKET_ODDS=off -------------------------
    _prev = os.environ.get("MARKET_ODDS")
    os.environ["MARKET_ODDS"] = "on"
    try:
        stack_e0(p, expect_fixtures=2, verbose=False)
        raise AssertionError("double-count guard must raise")
    except ValueError as e:
        assert "twice" in str(e)
    finally:
        if _prev is None:
            os.environ.pop("MARKET_ODDS", None)
        else:
            os.environ["MARKET_ODDS"] = _prev

    # --- lambda round-trips through the odds columns into E0 --------------------
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "E0_base.csv")
        pd.DataFrame([{"Date": "01/01/2026", "HomeTeam": "X", "AwayTeam": "Y",
                       "FTHG": 1.0, "FTAG": 1.0, "AvgH": 2.0, "AvgD": 3.4, "AvgA": 3.8,
                       "Avg>2.5": 1.9, "Avg<2.5": 1.9}]).to_csv(base, index=False)
        out = os.path.join(td, "E0_solio.csv")
        path, n, rep2 = stack_e0(p, expect_fixtures=2, e0_path=base, out=out, weight=2,
                                 allow_double_count=True, verbose=False)
        got = pd.read_csv(path)
        assert n == 4 and len(got) == 5, (n, len(got))     # 2 fixtures x weight 2
        r = got[got["HomeTeam"] == "Aston Villa"].iloc[0]
        lam_h, lam_a, _ = boi.implied_lambdas(r["AvgH"], r["AvgD"], r["AvgA"],
                                              r["Avg>2.5"], r["Avg<2.5"])
        assert abs(lam_h - 0.85) < 5e-3 and abs(lam_a - 1.90) < 5e-3, (lam_h, lam_a)
        assert abs(r["FTHG"] - 0.85) < 1e-6 and abs(r["FTAG"] - 1.90) < 1e-6

    # --- snapshot store: dedupes, sorts, and movement reads the delta ------------
    with tempfile.TemporaryDirectory() as td:
        p0 = _synthetic(generated="2026-08-27T00:00:00.000Z", bump=0.0)
        p1 = _synthetic(generated="2026-08-27T12:00:00.000Z", bump=0.25)
        for pay in (p0, p1, p1):                      # p1 twice: must dedupe
            sp = snapshot_path(pay, td)
            if not os.path.exists(sp):
                with open(sp, "w", encoding="utf-8") as fh:
                    json.dump(pay, fh)
        assert len(load_snapshots(td)) == 2
        m = movement(snapdir=td)
        ars = m[m["team"] == "Arsenal"].iloc[0]
        assert abs(ars["d_lam_for"] - 0.25) < 1e-9, ars["d_lam_for"]
        assert abs(ars["hours"] - 12.0) < 1e-6, ars["hours"]
        # a club whose lambda did not move must read exactly zero, not noise
        tot = m[m["team"] == "Tottenham"].iloc[0]
        assert tot["d_lam_for"] == 0.0 and tot["d_cs_prob"] == 0.0

    # --- deltas are not computed across a gameweek boundary ---------------------
    with tempfile.TemporaryDirectory() as td:
        for pay in (_synthetic(gw=2, generated="2026-08-27T00:00:00.000Z"),
                    _synthetic(gw=3, generated="2026-08-31T00:00:00.000Z", bump=1.0)):
            with open(snapshot_path(pay, td), "w", encoding="utf-8") as fh:
                json.dump(pay, fh)
        assert len(movement(snapdir=td)) == 0

    print("SELFTEST OK: solio lambda parses from truncated lists, one side recovers a "
          "fixture, censored coverage and double-counting both refuse, lambda round-trips "
          "through the E0 odds columns, snapshots dedupe and movement diffs them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
