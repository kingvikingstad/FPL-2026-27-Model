"""
travel.py — the away side's trip, as a per-fixture modifier of home advantage.
==============================================================================
ON BY DEFAULT since 2026-09-11 (owner decision, see below). `FPL_TRAVEL=off` disables it.

What it encodes, and the evidence  (studies/travel_distance.py, pre-registered)
-------------------------------------------------------------------------------
Only the away side travels, so distance is not a team attribute: it modulates home
advantage fixture by fixture. The model's home parameter is one constant, i.e. an average
over derbies and 300 km trips that misprices both ends.

31 seasons (11,944 matches), attack and defence fixed effects per TEAM-SEASON, home
advantage per season, SEs clustered on the club pair:

  home goals (= the traveller's goals AGAINST)  +0.0323 per log-km, z = +4.6
      2016-26 alone +0.0315; linear form and holiday-free cut agree.
      P10 -> P90 trip (23 km -> 316 km): home goals +8.8%.
  away goals (= the traveller's goals FOR)      -0.0096, z = -1.1   NULL, and the recent
      decade has the opposite sign. NOT shipped: this module moves home goals only.

It FAILED the market-orthogonality gate (R3): over the pre-match market lambda, score
z = +1.73 and GLM z = +1.75, both under 2.5; the point estimate over the market (+0.015)
is about half the structural one. The pre-registered rule for "real but not shown to beat
the market" shipped it off by default and raised the guard's scope with the owner.

WHY IT IS ON ANYWAY — an explicit, scoped override of the CLAUDE.md market gate, taken by
the owner on 2026-09-11. The gate exists so a team signal does not double-count what the
odds already price. This pipeline's per-fixture lambda is exp(mu + home + att - def), fitted
on xG-derived pseudo-odds plus results; it ingests NO match odds, so a distance effect the
market prices appears in no per-fixture number the model makes, and there is nothing to
double-count. The override covers this signal only; it does not relax the gate for others.
It WOULD bite if per-fixture market lambdas ever enter the fit (gw_board's SOLIO_MARKET=on
path stacks them into E0) — re-examine this if that path is switched on by default.
The GW4 deadline lock (predictions/gw4_board_locked_2026-09-11_deadline.csv) was made with
it OFF; GW5 is the first locked board that carries it.

How it enters
-------------
`bayes_model._home_effect` adds `fixture_shift(...)` to the HOME side's log-lambda only,
centred on the mean trip of the 25/26 league (every ordered pair played once, so the
centre is a function of the 20 clubs alone). The fitted `home` is the 25/26 average, so
centring there leaves average home advantage where the fit put it; a 26/27 league with
longer average trips gets a small genuine increase.

The coefficient's own uncertainty is carried, not dropped: one draw of b per posterior
draw, shared across every fixture in that draw, from N(B_HOME, SE_HOME).

Run:  python src/travel.py              the 26/27 fixtures it moves most
      python src/travel.py --selftest
"""
from __future__ import annotations
import os as _os
import sys as _sys
import numpy as np
import pandas as pd

OFF_VALUES = ("off", "0", "false")
FLOOR_KM = 5.0
# studies/travel_distance.py, full sample 1993/94-2025/26, pair-clustered.
B_HOME = 0.0323
SE_HOME = 0.0070
SEED = 20260910

# ---------------------------------------------------------------------------
# Grounds. (from_date, lat, lon, ground). A club's ground on a match date is the last
# entry whose from_date <= date. Coordinates are [CHECK] to ~0.5 km; the 5 km floor makes
# that immaterial.
# ---------------------------------------------------------------------------
_T0 = "1900-01-01"
GROUNDS = {
    "Arsenal": [(_T0, 51.5580, -0.1027, "Highbury"),
                ("2006-07-01", 51.5549, -0.1084, "Emirates")],
    "Aston Villa": [(_T0, 52.5092, -1.8848, "Villa Park")],
    "Barnsley": [(_T0, 53.5522, -1.4675, "Oakwell")],
    "Birmingham": [(_T0, 52.4758, -1.8680, "St Andrew's")],
    "Blackburn": [(_T0, 53.7286, -2.4892, "Ewood Park")],
    "Blackpool": [(_T0, 53.8047, -3.0481, "Bloomfield Road")],
    "Bolton": [(_T0, 53.5708, -2.4194, "Burnden Park"),
               ("1997-07-01", 53.5806, -2.5354, "Reebok")],
    "Bournemouth": [(_T0, 50.7352, -1.8383, "Dean Court")],
    "Bradford": [(_T0, 53.8042, -1.7590, "Valley Parade")],
    "Brentford": [(_T0, 51.4907, -0.2886, "Gtech Community")],
    "Brighton": [(_T0, 50.8616, -0.0837, "Amex")],
    "Burnley": [(_T0, 53.7890, -2.2302, "Turf Moor")],
    "Cardiff": [(_T0, 51.4728, -3.2030, "Cardiff City Stadium")],
    "Charlton": [(_T0, 51.4865, 0.0365, "The Valley")],
    "Chelsea": [(_T0, 51.4817, -0.1910, "Stamford Bridge")],
    "Coventry": [(_T0, 52.4118, -1.4906, "Highfield Road"),
                 ("2005-08-01", 52.4481, -1.4956, "CBS Arena")],
    "Crystal Palace": [(_T0, 51.3983, -0.0855, "Selhurst Park")],
    "Derby": [(_T0, 52.9061, -1.4717, "Baseball Ground"),
              ("1997-07-01", 52.9150, -1.4472, "Pride Park")],
    "Everton": [(_T0, 53.4388, -2.9664, "Goodison Park"),
                ("2025-07-01", 53.4253, -3.0006, "Hill Dickinson")],
    "Fulham": [(_T0, 51.4749, -0.2217, "Craven Cottage"),
               ("2002-07-01", 51.5093, -0.2322, "Loftus Road (groundshare)"),
               ("2004-07-01", 51.4749, -0.2217, "Craven Cottage")],
    "Huddersfield": [(_T0, 53.6543, -1.7684, "John Smith's")],
    "Hull": [(_T0, 53.7462, -0.3676, "MKM Stadium")],
    "Ipswich": [(_T0, 52.0545, 1.1447, "Portman Road")],
    "Leeds": [(_T0, 53.7778, -1.5722, "Elland Road")],
    "Leicester": [(_T0, 52.6246, -1.1397, "Filbert Street"),
                  ("2002-07-01", 52.6204, -1.1422, "King Power")],
    "Liverpool": [(_T0, 53.4308, -2.9608, "Anfield")],
    "Luton": [(_T0, 51.8843, -0.4316, "Kenilworth Road")],
    "Man City": [(_T0, 53.4452, -2.2211, "Maine Road"),
                 ("2003-07-01", 53.4831, -2.2004, "Etihad")],
    "Man United": [(_T0, 53.4631, -2.2913, "Old Trafford")],
    "Middlesbrough": [(_T0, 54.5782, -1.2169, "Riverside")],
    "Newcastle": [(_T0, 54.9756, -1.6217, "St James' Park")],
    "Norwich": [(_T0, 52.6222, 1.3091, "Carrow Road")],
    "Nott'm Forest": [(_T0, 52.9400, -1.1328, "City Ground")],
    "Oldham": [(_T0, 53.5552, -2.1286, "Boundary Park")],
    "Portsmouth": [(_T0, 50.7964, -1.0639, "Fratton Park")],
    "QPR": [(_T0, 51.5093, -0.2322, "Loftus Road")],
    "Reading": [(_T0, 51.4222, -0.9827, "Madejski")],
    "Sheffield United": [(_T0, 53.3703, -1.4709, "Bramall Lane")],
    "Sheffield Weds": [(_T0, 53.4114, -1.5006, "Hillsborough")],
    "Southampton": [(_T0, 50.9133, -1.4115, "The Dell"),
                    ("2001-07-01", 50.9058, -1.3911, "St Mary's")],
    "Stoke": [(_T0, 52.9884, -2.1755, "bet365 Stadium")],
    "Sunderland": [(_T0, 54.9225, -1.3767, "Roker Park"),
                   ("1997-07-01", 54.9146, -1.3884, "Stadium of Light")],
    "Swansea": [(_T0, 51.6428, -3.9351, "Liberty")],
    "Swindon": [(_T0, 51.5645, -1.7708, "County Ground")],
    "Tottenham": [(_T0, 51.6033, -0.0660, "White Hart Lane"),
                  ("2017-07-01", 51.5560, -0.2795, "Wembley"),
                  ("2019-04-03", 51.6043, -0.0664, "Tottenham Hotspur Stadium")],
    "Watford": [(_T0, 51.6499, -0.4015, "Vicarage Road")],
    "West Brom": [(_T0, 52.5090, -1.9639, "The Hawthorns")],
    "West Ham": [(_T0, 51.5319, 0.0393, "Boleyn Ground"),
                 ("2016-07-01", 51.5386, -0.0165, "London Stadium")],
    "Wigan": [(_T0, 53.5477, -2.6538, "DW Stadium")],
    "Wimbledon": [(_T0, 51.3983, -0.0855, "Selhurst Park (groundshare)")],
    "Wolves": [(_T0, 52.5902, -2.1304, "Molineux")],
}
# The league whose fit supplies `home`: every ordered pair once, so the mean trip is a
# function of these twenty names and nothing else.
LEAGUE_2526 = ("Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Burnley",
               "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds", "Liverpool",
               "Man City", "Man United", "Newcastle", "Nott'm Forest", "Sunderland",
               "Tottenham", "West Ham", "Wolves")
DATE_2526, DATE_2627 = "2025-10-01", "2026-10-01"


def ground(club: str, date) -> tuple[float, float]:
    """(lat, lon) of `club`'s ground on `date`. KeyError on an unknown club — a silent
    default would place it somewhere plausible and read as a mid-length trip."""
    ts = pd.Timestamp(date)
    cur = None
    for frm, lat, lon, _ in GROUNDS[club]:
        if pd.Timestamp(frm) <= ts:
            cur = (lat, lon)
    return cur


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def trip_km(home: str, away: str, date=DATE_2627) -> float:
    a, b = ground(home, date), ground(away, date)
    return float(haversine_km(a[0], a[1], b[0], b[1]))


def trip_z(home: str, away: str, date=DATE_2627) -> float:
    return float(np.log(max(trip_km(home, away, date), FLOOR_KM)))


def league_centre(clubs=LEAGUE_2526, date=DATE_2526) -> float:
    cl = list(clubs)
    return float(np.mean([trip_z(a, b, date) for i, a in enumerate(cl) for b in cl[i + 1:]]))


Z_CENTRE = league_centre()


def enabled() -> bool:
    return _os.environ.get("FPL_TRAVEL", "on").strip().lower() not in OFF_VALUES


_B_DRAWS: dict[int, np.ndarray] = {}


def _b(S: int) -> np.ndarray:
    """One coefficient draw per posterior draw, fixed by SEED so boards are
    reproducible and every fixture in a draw shares the same b."""
    if S not in _B_DRAWS:
        _B_DRAWS[S] = np.random.default_rng(SEED).normal(B_HOME, SE_HOME, S)
    return _B_DRAWS[S]


def fixture_shift(team: str, opp: str, is_home: bool, S: int | None = None):
    """Shift to the HOME side's log-lambda for this fixture: b * (z - Z_CENTRE).
    0.0 when FPL_TRAVEL is off or either club has no ground on record (reported, not
    guessed). `team`/`opp`/`is_home` are the caller's row, so the home side is
    `team` if is_home else `opp`."""
    if not enabled():
        return 0.0
    h, a = (team, opp) if is_home else (opp, team)
    if h not in GROUNDS or a not in GROUNDS:
        print(f"[travel] no ground on record for {h if h not in GROUNDS else a} — "
              f"fixture left unshifted")
        return 0.0
    dz = trip_z(h, a) - Z_CENTRE
    return _b(S) * dz if S else B_HOME * dz


def table_2627() -> pd.DataFrame:
    """Every 26/27 fixture with its trip and the home-goals multiplier at the point
    estimate."""
    from schedule_2627 import schedule
    sched, long = schedule()
    h = long[long["is_home"].astype(bool)]
    out = pd.DataFrame({"gameweek": h["gameweek"].values, "home": h["team"].values,
                        "away": h["opp"].values})
    out["km"] = [trip_km(a, b) for a, b in zip(out["home"], out["away"])]
    out["dz"] = [trip_z(a, b) - Z_CENTRE for a, b in zip(out["home"], out["away"])]
    out["home_goals_x"] = np.exp(B_HOME * out["dz"])
    return out


def selftest():
    km = trip_km("Man United", "Chelsea", "2020-01-01")
    assert 250 < km < 275, km                                  # ~262 km
    assert trip_km("Wimbledon", "Crystal Palace", "1995-01-01") == 0.0
    assert trip_km("Tottenham", "Arsenal", "2018-01-01") > 10  # Wembley year
    assert trip_km("Tottenham", "Arsenal", "2019-05-01") < 7
    assert trip_km("Everton", "Liverpool", "2024-01-01") < 1.5        # Goodison
    assert 2.0 < trip_km("Everton", "Liverpool", "2026-01-01") < 3.5  # Hill Dickinson
    try:
        ground("Nowhere FC", "2020-01-01"); raise AssertionError("unknown club passed")
    except KeyError:
        pass
    assert 4.5 < Z_CENTRE < 5.5, Z_CENTRE
    old = _os.environ.pop("FPL_TRAVEL", None)
    try:
        d = fixture_shift("Man United", "Man City", True)                # on by default
        for v in OFF_VALUES + ("OFF", " off "):
            _os.environ["FPL_TRAVEL"] = v
            assert fixture_shift("Man United", "Man City", True) == 0.0, v
        _os.environ["FPL_TRAVEL"] = "on"
        assert fixture_shift("Man United", "Man City", True) == d
        assert d < -0.05, d                        # a derby: home advantage shrinks
        assert fixture_shift("Man City", "Man United", False) == d       # same fixture
        assert fixture_shift("Sunderland", "Brighton", True) > 0.0       # long trip
        dS = fixture_shift("Man United", "Man City", True, S=4000)
        assert dS.shape == (4000,) and abs(dS.mean() - d) < 0.01 * abs(d) + 0.002
        # centring: across the 25/26 league the average shift is zero by construction
        cl = list(LEAGUE_2526)
        m = np.mean([B_HOME * (trip_z(a, b, DATE_2526) - Z_CENTRE)
                     for a in cl for b in cl if a != b])
        assert abs(m) < 1e-12, m
    finally:
        _os.environ.pop("FPL_TRAVEL", None)
        if old is not None:
            _os.environ["FPL_TRAVEL"] = old
    print(f"SELFTEST OK: distances and dated moves, on by default and off on "
          f"{'/'.join(OFF_VALUES)}, centre "
          f"{Z_CENTRE:.3f} log-km ({np.exp(Z_CENTRE):.0f} km) with zero mean shift over "
          f"25/26, Manchester derby {d:+.3f} on home log-lambda, b drawn per posterior draw.")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        selftest(); _sys.exit(0)
    t = table_2627()
    print(f"centre {Z_CENTRE:.3f} log-km = {np.exp(Z_CENTRE):.0f} km (25/26 league); "
          f"26/27 league mean dz {t['dz'].mean():+.3f} -> home goals "
          f"x{np.exp(B_HOME * t['dz'].mean()):.4f} on average")
    print("\nlargest cuts to home goals (short trips):")
    print(t.nsmallest(8, "dz").round(3).to_string(index=False))
    print("\nlargest boosts (long trips):")
    print(t.nlargest(8, "dz").round(3).to_string(index=False))
