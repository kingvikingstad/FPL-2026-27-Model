"""
fixture_market.py — the market's lambda for every fixture that has one, beside the model's.
===========================================================================================
The per-fixture table (`team_projections_gw1_38.csv`) carries the MODEL's posterior
lambda for each team-fixture. This module supplies the MARKET's, from whichever stored
source priced that fixture, so the two are read side by side at the fixture they
disagree about rather than in a separate file nobody opens.

DISPLAY AND AUDIT ONLY. NOTHING HERE ENTERS TeamModel.
------------------------------------------------------
`market_odds.py` already blends the outright market into the team layer at
MARKET_WEIGHT=0.6. Feeding per-fixture market lambda in as well counts the market
twice, which is the refusal `solio_market.stack_e0` enforces. The columns added here
change no projection, no board and no locked prediction. Agreement between the two is
therefore partly by construction — the model has seen the outrights — and a gap is the
fitted season-long strength disagreeing with this week's price.

SOURCES, IN PRECEDENCE ORDER
----------------------------
  solio   config.SOLIO_SNAPSHOTS. (prGoalsFor, prGoalsAgainst) per team-fixture, already
          market-derived (solio_market.py). Preferred per SOLIO_MARKET_FEED §2: no de-vig
          or inversion choice on our side. 10/10 fixtures on every full JSON snapshot,
          GW2-4 [VERIFIED 2026-09-11]. A double-gameweek record is dropped as ambiguous
          (solio_market.fixture_lambdas).
  books   config.ODDS_SNAPSHOTS. football-data.co.uk fixtures.csv — market-average 1X2
          and O/U 2.5, Shin de-vig, then the two-solve Poisson inversion in
          betting_odds_ingest.implied_lambdas. Free, no key. Fills what Solio did not
          price, and is an independent read of the same market where both exist.

Precedence is PER FIXTURE, not per gameweek: a fixture Solio could not price is filled
from the books, not left empty because its neighbours were priced.

WHICH SNAPSHOT
--------------
The latest observation taken AT OR BEFORE that gameweek's deadline. For the next
gameweek that is just the latest. For a played one it is the last pre-deadline price —
the same information cut as `predictions/` — so a scored week compares the model with
the market as it stood when the decision was made, not with a price that had already
seen the team sheets. Solio carries its own `deadlineIso`; otherwise the feed's
`gameweek_summaries.deadline_time` is used, looked up BY ID because that file is not
sorted.

Only the next gameweek or two is ever priced, so these columns are empty for most of a
38-week horizon. Empty is the honest value — there is no market for GW20 yet — and GW1
has none because the snapshot store began at GW2.

CLEAN SHEETS ARE PLUG-IN
------------------------
`mkt_p_clean_sheet` = exp(-mkt_lam_against), one definition for every source. Solio's
published csProb agrees with it to within 0.003 on every stored snapshot [VERIFIED
2026-09-11], and is published only for the listed side of a fixture anyway. The model's
`p_clean_sheet` is posterior-PREDICTIVE (mean over draws of exp(-lambda)), which Jensen
puts ABOVE the plug-in. Compare the market figure with `p_clean_sheet_plugin`, or
compare lambdas; set against `p_clean_sheet` it reads the Jensen gap as a disagreement.

Run:  python src/fixture_market.py --selftest
      python src/fixture_market.py --fetch     store a football-data snapshot (no model change)
      python src/fixture_market.py --report    what is priced, per gameweek and source
"""
import os
import re
import glob
import datetime as _dt
import numpy as np
import pandas as pd
import config
import betting_odds_ingest as boi
import solio_market as sm

# Appended to a per-team-fixture frame by attach(). Numeric first, provenance last.
MKT_COLS = ["mkt_lam_for", "mkt_lam_against", "mkt_p_clean_sheet", "mkt_source",
            "mkt_as_of"]
SOURCES = ("solio", "books")                 # precedence, highest first
DEVIG = "shin"                               # oddsapi_feed.build's default, for one convention

_FX_COLS = ["gw", "home", "away", "lam_home", "lam_away", "source", "as_of"]
_BOOK_ODDS = ("AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5")
_BOOK_KEEP = ("Date", "Time", "HomeTeam", "AwayTeam") + _BOOK_ODDS + (
    "MaxH", "MaxD", "MaxA", "Max>2.5", "Max<2.5",
    "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5")
_BOOK_RE = re.compile(r"footballdata_E0_(\d{8}T\d{6})Z\.csv$")


def _utc(s):
    """Anything timestamp-like -> tz-aware UTC Timestamp, or None."""
    if s is None:
        return None
    try:
        t = pd.Timestamp(s)
    except (ValueError, TypeError):
        return None
    if pd.isna(t):
        return None
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def deadlines(gw_frame):
    """gameweek id -> deadline (UTC), from a gameweek_summaries frame. Keyed by id,
    never by position: the feed is not sorted."""
    if gw_frame is None or not len(gw_frame):
        return {}
    out = {}
    for i, t in zip(gw_frame["id"], gw_frame["deadline_time"]):
        u = _utc(t)
        if u is not None and not pd.isna(i):
            out[int(i)] = u
    return out


def feed_deadlines():
    """Deadlines from the external feed, or {} if it cannot be read. Solio snapshots
    carry their own, so a missing feed degrades only the books source."""
    try:
        import core_insights as ci
        return deadlines(pd.read_csv(os.path.join(ci.UPLOADS, "gameweek_summaries.csv")))
    except Exception:                                         # noqa: BLE001
        return {}


def _gw_of_pairing():
    """(home, away) -> gameweek, from the 26/27 schedule. Each ordered pairing is played
    once a season, so a book row with only a date resolves to exactly one gameweek."""
    from schedule_2627 import schedule
    df, _ = schedule()
    key = list(zip(df["home"], df["away"]))
    assert len(key) == len(set(key)), "an ordered pairing appears twice in the schedule"
    return dict(zip(key, (int(g) for g in df["gameweek"])))


def _latest(rows):
    """One row per (gw, home, away): the newest observation."""
    d = pd.DataFrame(rows, columns=_FX_COLS)
    if not len(d):
        return d
    d = d.sort_values("as_of", kind="mergesort")
    return d.drop_duplicates(["gw", "home", "away"], keep="last").reset_index(drop=True)


# ------------------------------------------------------------------ solio
def solio_fixtures(snapshots=None, deadlines=None, snapdir=None):
    """Per-fixture Solio lambda, newest pre-deadline observation of each fixture."""
    snaps = snapshots if snapshots is not None else sm.load_snapshots(snapdir)
    rows, late, amb, used = [], 0, 0, 0
    for _path, pay in snaps:
        gw = pay.get("gameweek")
        t = _utc(pay.get("generatedAt"))
        if gw is None or t is None:
            continue
        gw = int(gw)
        dl = _utc(pay.get("deadlineIso")) or (deadlines or {}).get(gw)
        if dl is not None and t > dl:
            late += 1
            continue
        fx, rep = sm.fixture_lambdas(pay)
        amb += int(rep.get("ambiguous") or 0)
        used += 1
        for h, a, lh, la in zip(fx["home"], fx["away"], fx["lam_home"], fx["lam_away"]):
            rows.append(dict(gw=gw, home=h, away=a, lam_home=float(lh),
                             lam_away=float(la), source="solio", as_of=t))
    return _latest(rows), dict(snapshots=len(snaps), used=used, post_deadline=late,
                               ambiguous=amb)


# ------------------------------------------------------------------ books
def book_snapshot_path(stamp, snapdir=None):
    return os.path.join(snapdir or config.ODDS_SNAPSHOTS, f"footballdata_E0_{stamp}Z.csv")


def load_book_snapshots(snapdir=None):
    """Every stored football-data snapshot, oldest first, as (fetched_at, path, frame)."""
    snapdir = snapdir or config.ODDS_SNAPSHOTS
    out = []
    for p in glob.glob(os.path.join(snapdir, "footballdata_E0_*Z.csv")):
        m = _BOOK_RE.search(os.path.basename(p))
        if not m:
            continue
        ts = _utc(_dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"))
        out.append((ts, p, pd.read_csv(p, encoding="utf-8")))
    out.sort(key=lambda x: x[0])
    return out


def fetch_books(snapdir=None, fetcher=None, now=None, verbose=True):
    """Fetch football-data's rolling fixtures.csv and store its E0 rows, timestamped.

    Deduplicates on CONTENT against the newest stored snapshot: the file carries no
    publication stamp, and storing an unchanged price twice would fabricate a
    no-movement observation. Written atomically and as UTF-8, as solio_market.fetch is.
    Returns the stored path, or None when nothing was stored.
    """
    snapdir = snapdir or config.ODDS_SNAPSHOTS
    d = (fetcher or boi.fetch_football_data_fixtures)(verbose=verbose)
    if not len(d):
        return None
    d = d[[c for c in _BOOK_KEEP if c in d.columns]]
    d = d.sort_values(["HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)
    text = d.to_csv(index=False, lineterminator="\n")
    prev = load_book_snapshots(snapdir)
    if prev:
        with open(prev[-1][1], encoding="utf-8") as fh:
            if fh.read() == text:
                if verbose:
                    print(f"[books] unchanged since {os.path.basename(prev[-1][1])} - not re-storing")
                return None
    now = now or _dt.datetime.now(_dt.timezone.utc)
    path = book_snapshot_path(now.strftime("%Y%m%dT%H%M%S"), snapdir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    os.replace(tmp, path)
    if verbose:
        print(f"[books] stored {os.path.basename(path)}  ({len(d)} E0 fixtures)")
    return path


def book_fixtures(snapshots=None, deadlines=None, snapdir=None):
    """Per-fixture de-vigged book lambda, newest pre-deadline observation of each."""
    snaps = snapshots if snapshots is not None else load_book_snapshots(snapdir)
    gw_of = _gw_of_pairing()
    rows, late, unknown = [], 0, set()
    for ts, _path, d in snaps:
        for _, r in d.iterrows():
            home, away = boi.norm_team(r["HomeTeam"]), boi.norm_team(r["AwayTeam"])
            gw = gw_of.get((home, away))
            if gw is None:
                unknown.add((home, away))        # name drift, or not a 26/27 pairing
                continue
            dl = (deadlines or {}).get(gw)
            if dl is not None and ts > dl:
                late += 1
                continue
            try:
                odds = [float(r[c]) for c in _BOOK_ODDS]
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(odds).all():
                continue
            lh, la, _ = boi.implied_lambdas(*odds, method=DEVIG)
            rows.append(dict(gw=gw, home=home, away=away, lam_home=float(lh),
                             lam_away=float(la), source="books", as_of=ts))
    return _latest(rows), dict(snapshots=len(snaps), post_deadline=late,
                               unknown=sorted(unknown))


# ------------------------------------------------------------------ combine
def fixtures(deadlines=None, solio_snaps=None, book_snaps=None, solio_dir=None,
             books_dir=None):
    """Per-fixture market lambda, one row per (gw, home, away), precedence applied.

    Returns (frame, report). The report carries each source's diagnostics and, where
    BOTH priced a fixture, their agreement — Solio is an estimate of the market, so the
    books are the only check that it is still tracking it.
    """
    s, srep = solio_fixtures(solio_snaps, deadlines, solio_dir)
    b, brep = book_fixtures(book_snaps, deadlines, books_dir)
    key = ["gw", "home", "away"]
    both = s.merge(b, on=key, suffixes=("_s", "_b")) if len(s) and len(b) else pd.DataFrame()
    agree = dict(n=len(both))
    if len(both):
        dh = both["lam_home_s"] - both["lam_home_b"]
        da = both["lam_away_s"] - both["lam_away_b"]
        # The two sources are observed at different times, so a gap between them is
        # disagreement PLUS whatever the market did in between. Carrying the median
        # separation stops the MAE being read as pure disagreement: it rose 0.032 ->
        # 0.055 on GW4 the morning after a books fetch, on no new book price at all.
        hrs = (both["as_of_s"] - both["as_of_b"]).abs().dt.total_seconds() / 3600.0
        agree.update(mae=float(np.abs(pd.concat([dh, da])).mean()),
                     bias=float(pd.concat([dh, da]).mean()),
                     max_abs=float(np.abs(pd.concat([dh, da])).max()),
                     median_gap_h=float(hrs.median()))
    parts = [x.assign(_rank=SOURCES.index(src)) for x, src in ((s, "solio"), (b, "books"))
             if len(x)]
    if parts:
        fx = pd.concat(parts, ignore_index=True)
        fx = fx.sort_values(key + ["_rank"], kind="mergesort")
        fx = fx.drop_duplicates(key, keep="first").drop(columns="_rank")
        fx = fx.reset_index(drop=True)
    else:
        fx = pd.DataFrame(columns=_FX_COLS)
    return fx, dict(solio=srep, books=brep, agreement=agree)


def per_team(fx):
    """Per-fixture -> per-team-fixture, both sides, with market columns named for the
    team the row belongs to."""
    base = ["team", "gw", "opponent", "is_home"]
    if not len(fx):
        return pd.DataFrame(columns=base + MKT_COLS)
    home = fx.rename(columns={"home": "team", "away": "opponent",
                              "lam_home": "mkt_lam_for", "lam_away": "mkt_lam_against"})
    away = fx.rename(columns={"away": "team", "home": "opponent",
                              "lam_away": "mkt_lam_for", "lam_home": "mkt_lam_against"})
    t = pd.concat([home.assign(is_home=1), away.assign(is_home=0)], ignore_index=True)
    t["mkt_p_clean_sheet"] = np.exp(-t["mkt_lam_against"].astype(float))
    t["mkt_source"] = t["source"]
    t["mkt_as_of"] = [x.strftime("%Y-%m-%dT%H:%MZ") for x in t["as_of"]]
    t["gw"] = t["gw"].astype(int)
    return t[base + MKT_COLS]


def attach(G, deadlines=None, **kw):
    """Left-join the market columns onto a per-team-fixture frame keyed
    (team, gw, opponent, is_home). Row count is asserted; a market fixture inside G's
    gameweeks that finds no row is COUNTED in the report, never dropped silently —
    a mismatch there means the schedule and the market disagree about a fixture."""
    fx, rep = fixtures(deadlines=deadlines, **kw)
    t = per_team(fx)
    key = ["team", "gw", "opponent", "is_home"]
    left = G.drop(columns=[c for c in MKT_COLS if c in G.columns]).copy()
    left["is_home"] = left["is_home"].astype(int)
    left["gw"] = left["gw"].astype(int)
    n0 = len(left)
    out = left.merge(t, on=key, how="left")
    assert len(out) == n0, f"market join fanned out {n0} -> {len(out)}"
    t_in = t[t["gw"].isin(set(left["gw"]))]
    hit = t_in.merge(left[key].drop_duplicates(), on=key, how="inner")
    priced = out[out["mkt_lam_for"].notna()]
    rep.update(rows=n0, priced=len(priced), unmatched=len(t_in) - len(hit),
               by_gw={int(g): dict(n=int(len(x)), sources=sorted(set(x["mkt_source"])),
                                   as_of=max(x["mkt_as_of"]))
                      for g, x in priced.groupby("gw")})
    return out, rep


def describe(rep):
    """The report as printable lines."""
    s, b, a = rep["solio"], rep["books"], rep["agreement"]
    lines = []
    if "rows" in rep:
        lines.append(f"market lambda on {rep['priced']} of {rep['rows']} team-fixtures"
                     + ("" if rep["priced"] else " — no stored market covers this window"))
        for g, x in sorted(rep["by_gw"].items()):
            lines.append(f"  GW{g:<3d} {x['n']:3d} team-fixtures  {'+'.join(x['sources']):12s}"
                         f" as of {x['as_of']}")
        if rep["unmatched"]:
            lines.append(f"  !! {rep['unmatched']} market team-fixtures matched no scheduled "
                         f"fixture (rescheduled, blank or double gameweek?) - not shown")
    lines.append(f"  solio: {s['used']}/{s['snapshots']} snapshots pre-deadline"
                 + (f", {s['ambiguous']} double-gameweek records dropped" if s["ambiguous"] else ""))
    lines.append(f"  books: {b['snapshots']} snapshot(s)"
                 + (f", {b['post_deadline']} post-deadline rows ignored" if b["post_deadline"] else "")
                 + (f", UNMAPPED {b['unknown']}" if b["unknown"] else ""))
    if a["n"]:
        lines.append(f"  solio vs books on {a['n']} shared fixtures: lambda MAE {a['mae']:.3f}, "
                     f"bias {a['bias']:+.3f} (solio - books), max |d| {a['max_abs']:.3f}"
                     f"  [observed {a['median_gap_h']:.1f}h apart — part of any gap is "
                     f"movement, not disagreement]")
    return lines


# ------------------------------------------------------------------ CLI
def main(argv=None):
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")     # piped stdout is cp1252 on Windows
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Per-fixture market lambda (display only)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="store a football-data snapshot")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.fetch:
        fetch_books()
    if a.report:
        fx, rep = fixtures(deadlines=feed_deadlines())
        for ln in describe(rep):
            print(ln)
        if len(fx):
            d = fx.copy()
            d["as_of"] = [x.strftime("%Y-%m-%d %H:%M") for x in d["as_of"]]
            print(d.round(3).to_string(index=False))
    if not (a.fetch or a.report):
        ap.print_help()
    return 0


# ------------------------------------------------------------------ selftest
def selftest():
    import sys
    import tempfile
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from schedule_2627 import schedule
    sched, long = schedule()

    def solio_pay(gw, generated, deadline, recs):
        return {"generatedAt": generated, "gameweek": gw, "deadlineIso": deadline,
                "bestCleanSheets": recs, "bestAttackingFixtures": []}

    def rec(team, opp, home, gf, ga):
        return {"team": team, "fixtures": [{"opponent": opp, "isHome": home}],
                "prGoalsFor": gf, "prGoalsAgainst": ga, "csProb": float(np.exp(-ga))}

    dl2 = "2026-08-28T17:30:00.000Z"
    # GW2 Aston Villa v Arsenal, priced from Arsenal's side; a later repricing lands
    # AFTER the deadline and must not be used.
    s_early = solio_pay(2, "2026-08-27T12:00:00.000Z", dl2,
                        [rec("Arsenal", "AVL", False, 1.90, 0.85)])
    s_late = solio_pay(2, "2026-08-28T20:00:00.000Z", dl2,
                       [rec("Arsenal", "AVL", False, 2.40, 0.60)])
    snaps = [("a", s_early), ("b", s_late)]

    # a GW3 fixture from the real schedule for the books to fill
    g3 = sched[sched["gameweek"] == 3].sort_values("home", kind="mergesort")
    h3, a3 = str(g3["home"].tolist()[0]), str(g3["away"].tolist()[0])
    dls = {2: _utc(dl2), 3: _utc("2026-09-04T17:30:00Z")}
    odds = dict(AvgH=1.80, AvgD=3.70, AvgA=4.40, **{"Avg>2.5": 1.75, "Avg<2.5": 2.05})
    book = pd.DataFrame([
        dict(HomeTeam=h3, AwayTeam=a3, **odds),                       # only the books have it
        dict(HomeTeam="Aston Villa", AwayTeam="Arsenal", **odds),     # both have it
        dict(HomeTeam="Wolves", AwayTeam="Burnley", **odds),          # not a 26/27 pairing
    ])
    bsnaps = [(_utc("2026-08-27T09:00:00Z"), "x", book)]

    fx, rep = fixtures(deadlines=dls, solio_snaps=snaps, book_snaps=bsnaps)

    # --- post-deadline Solio snapshot is not used -------------------------------
    avl = fx[(fx["home"] == "Aston Villa") & (fx["away"] == "Arsenal")]
    assert len(avl) == 1 and abs(float(avl["lam_away"].tolist()[0]) - 1.90) < 1e-9, avl
    assert rep["solio"]["post_deadline"] == 1, rep

    # --- Solio wins where both priced; the books fill what Solio did not ---------
    assert avl["source"].tolist() == ["solio"], avl
    f3 = fx[(fx["home"] == h3) & (fx["away"] == a3)]
    lh, la, _ = boi.implied_lambdas(*[odds[c] for c in _BOOK_ODDS], method=DEVIG)
    assert f3["source"].tolist() == ["books"] and abs(float(f3["lam_home"].tolist()[0]) - lh) < 1e-9
    # the agreement figure carries how far apart the two were observed, so a gap cannot
    # be read as disagreement when it is partly movement
    assert rep["agreement"]["n"] == 1, rep["agreement"]
    assert abs(rep["agreement"]["median_gap_h"] - 3.0) < 1e-6, rep["agreement"]
    assert ("Wolves", "Burnley") in rep["books"]["unknown"], rep["books"]

    # --- a book snapshot taken after its gameweek's deadline is ignored ----------
    late_book = [(_utc("2026-09-05T09:00:00Z"), "y", book.iloc[:1])]
    fx2, rep2 = fixtures(deadlines=dls, solio_snaps=[], book_snaps=late_book)
    assert len(fx2) == 0 and rep2["books"]["post_deadline"] == 1, (fx2, rep2)

    # --- both sides mirror; CS is plug-in from the market lambda ------------------
    t = per_team(fx)
    v = t[(t["team"] == "Aston Villa") & (t["gw"] == 2)]
    assert int(v["is_home"].tolist()[0]) == 1
    assert abs(float(v["mkt_lam_for"].tolist()[0]) - 0.85) < 1e-9
    assert abs(float(v["mkt_p_clean_sheet"].tolist()[0]) - np.exp(-1.90)) < 1e-12
    ars = t[(t["team"] == "Arsenal") & (t["gw"] == 2)]
    assert abs(float(ars["mkt_lam_for"].tolist()[0]) - 1.90) < 1e-9 and ars["is_home"].tolist() == [0]

    # --- attach: no fan-out, empty where no market, provenance carried -----------
    G = long[long["gameweek"].isin([2, 3, 5])].rename(
        columns={"gameweek": "gw", "opp": "opponent"})[["team", "gw", "opponent", "is_home"]]
    out, rep3 = attach(G, deadlines=dls, solio_snaps=snaps, book_snaps=bsnaps)
    assert len(out) == len(G) == 60, len(out)
    # the model's columns and row order come back untouched — the join only ADDS
    pd.testing.assert_frame_equal(out[list(G.columns)].reset_index(drop=True),
                                  G.reset_index(drop=True), check_dtype=False)
    assert out.loc[out["gw"] == 5, "mkt_lam_for"].isna().all()
    assert rep3["priced"] == 4 and rep3["unmatched"] == 0, rep3
    assert set(out["mkt_source"].dropna()) == {"solio", "books"}

    # --- a market fixture the schedule does not have is counted, not dropped ------
    wrong = solio_pay(2, "2026-08-27T13:00:00.000Z", dl2,
                      [rec("Chelsea", "HUL", True, 2.5, 0.7)])     # not a GW2 fixture
    _, rep4 = attach(G, deadlines=dls, solio_snaps=[("c", wrong)], book_snaps=[])
    assert rep4["unmatched"] == 2 and rep4["priced"] == 0, rep4

    # --- fetch_books: stores, dedupes on content, stores again on a price move ----
    with tempfile.TemporaryDirectory() as td:
        base = pd.DataFrame([dict(Div="E0", Date="12/09/2026", Time="15:00",
                                  HomeTeam="Chelsea", AwayTeam="Hull", **odds)])
        moved = base.assign(AvgH=1.70)
        t0 = _dt.datetime(2026, 9, 11, 10, 0, 0, tzinfo=_dt.timezone.utc)
        p1 = fetch_books(td, fetcher=lambda verbose: base, now=t0, verbose=False)
        p2 = fetch_books(td, fetcher=lambda verbose: base,
                         now=t0 + _dt.timedelta(hours=2), verbose=False)
        p3 = fetch_books(td, fetcher=lambda verbose: moved,
                         now=t0 + _dt.timedelta(hours=4), verbose=False)
        assert p1 and p2 is None and p3, (p1, p2, p3)
        got = load_book_snapshots(td)
        assert len(got) == 2 and got[0][0] < got[1][0]
        assert "Div" not in got[0][2].columns and float(got[1][2]["AvgH"].tolist()[0]) == 1.70
        assert not any(f.endswith(".tmp") for f in os.listdir(td))

    # --- deadlines are read by id, whatever the row order ------------------------
    gws = pd.DataFrame({"id": [15, 3], "deadline_time": ["2026-12-01T18:30:00Z",
                                                         "2026-09-04T17:30:00Z"]})
    assert deadlines(gws)[3] == _utc("2026-09-04T17:30:00Z")

    print("SELFTEST OK: latest pre-deadline price per fixture, Solio before books per "
          "fixture, post-deadline snapshots ignored, both sides mirrored with plug-in CS, "
          "join preserves rows and counts unmatched fixtures, book snapshots dedupe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
