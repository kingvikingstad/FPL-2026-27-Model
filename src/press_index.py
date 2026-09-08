"""
press_index.py — pressing intensity (PPDA) for the CBIRT DefCon channel.
========================================================================
CBIRT (MID/FWD DefCon, threshold 12) is driven by recoveries, which scale with
press intensity, not team xGA — a high-press possession side generates recoveries
in the opposition half even at low xGA. PPDA (passes per defensive action; lower =
more press) is the standard measure. This supplies the press covariate the DefCon
handoff (§4.4) flagged as the missing input for conditioning CBIRT, and is used by
defcon_env to scale the MID/FWD DefCon rate by relative press.

Two tables:
  PPDA_2526 — actual 25/26 pressing (the environment a player's rate was fitted in).
  PPDA_2627 — manager-aware estimate for 26/27. For the nine regime clubs this is
              the new manager's characteristic press (from their prior club), NOT
              the old side's number — flagged low-confidence; sweepable.

Lower PPDA = higher press = more CBIRT. press_factor(club) = league_avg / ppda.

Sources: TotalFootballAnalysis / PremierLeague.com / techlawnews / adiralsport
25/26 PPDA; manager priors from the regime note §5 (Iraola PPDA ~9.9 at Bournemouth,
Maresca possession, Jaissle/Rose high press, etc.). [JUDGMENT] for 26/27.
"""
from __future__ import annotations
import numpy as np, pandas as pd

# actual 25/26 (old managers). ~12 = league-average; <10 intense, >14 passive.
PPDA_2526 = {
    "Liverpool": 9.9, "Arsenal": 10.05, "Tottenham": 10.0, "Bournemouth": 9.9,
    "Newcastle": 10.3, "Brighton": 10.8, "Man United": 11.0, "Chelsea": 11.0,
    "Aston Villa": 12.45, "Man City": 12.29, "Nott'm Forest": 12.5, "Everton": 14.0,
    "Fulham": 12.5, "Crystal Palace": 12.0, "Brentford": 13.0, "Leeds": 12.5,
    "Sunderland": 13.0, "Wolves": 13.5, "West Ham": 13.0, "Burnley": 14.0,
}
LEAGUE_PPDA = 12.0

# manager-aware 26/27 estimate. Regime clubs take the new manager's press style.
PPDA_2627 = {
    "Arsenal": 10.05,           # Arteta, continuity
    "Liverpool": 9.9,           # Iraola — pressed harder than Slot at Bournemouth
    "Man City": 12.3,           # Maresca — possession, moderate press
    "Chelsea": 11.0,            # Alonso — Leverkusen possession+press
    "Nott'm Forest": 11.5,      # Glasner — proactive, back three
    "Crystal Palace": 11.5,     # Sage — proactive
    "Bournemouth": 10.0,        # Rose — Bundesliga high press
    "Fulham": 12.0,             # Arbeloa
    "Newcastle": 10.0,          # Jaissle — Salzburg high press
    "Tottenham": 10.0,          # De Zerbi — high press
    "Man United": 12.0,         # Carrick — controlled, low-event
    "Aston Villa": 12.45, "Brighton": 10.8, "Everton": 14.0, "Brentford": 13.0,
    "Leeds": 12.5, "Sunderland": 13.0,
    "Coventry": 13.0, "Hull": 13.5, "Ipswich": 13.0,   # promoted, mid/low block
}
# clubs whose 26/27 press is a new-manager estimate (low confidence)
REGIME_PRESS_CLUBS = {"Liverpool", "Man City", "Chelsea", "Nott'm Forest",
                      "Crystal Palace", "Bournemouth", "Fulham", "Newcastle",
                      "Tottenham", "Man United"}
# promoted — no PL season at all, so the 26/27 number is a base rate, not a measurement
PROMOTED_PRESS_CLUBS = {"Coventry", "Hull", "Ipswich"}


# --------------------------------------------------------- in-season revision
# The 26/27 table above is a forecast. `press_measured` measures what the clubs actually
# did and blends it in with a weight that grows as w = n/(n+k); at GW1 that is 2%, so the
# judgment prior is not displaced until the sample earns it. See that module for the two
# backtests behind k, and why the conservative k=40 was taken over the optimal-under-one
# k=12. Set PRESS_MEASURED=off to pin the judgment table.
#
# Blending happens for season "2627" only. "2526" is the historical reference leg of the
# `defcon_env` ratio — the environment a player's rate was FITTED in — and revising it
# with 26/27 evidence would be incoherent.
_MEASURED_CACHE = {}


def _measured(gw=None):
    """({club: measured ppda}, {club: n}) for 26/27, memoised. Never raises: a missing
    or malformed feed leaves the judgment prior in place rather than taking the board
    down, but it says so."""
    import os
    if str(os.environ.get("PRESS_MEASURED", "on")).lower() in ("off", "0", "false"):
        return {}, {}
    if gw in _MEASURED_CACHE:
        return _MEASURED_CACHE[gw]
    try:
        import press_measured as pm
        got = pm.measured_table("2026-2027", gw)
    except Exception as e:                                   # noqa: BLE001
        print(f"[press] measured PPDA unavailable ({type(e).__name__}: {e}) "
              f"— judgment table stands")
        got = ({}, {})
    _MEASURED_CACHE[gw] = got
    return got


def resolve_ppda(club, season="2627", gw=None):
    """(ppda, weight, n_matches) — the judgment prior blended with measured press.

    Blended in logs: PPDA is a ratio and enters `defcon_env` multiplicatively, so the
    geometric mean is the coherent average and a club cannot be dragged negative.
    """
    prior = (PPDA_2627 if season == "2627" else PPDA_2526).get(club, LEAGUE_PPDA)
    if season != "2627":
        return prior, 0.0, 0
    import press_measured as pm
    tbl, n = _measured(gw)
    meas = tbl.get(club)
    if meas is None or meas <= 0:
        return prior, 0.0, int(n.get(club, 0))
    weak = club in REGIME_PRESS_CLUBS or club in PROMOTED_PRESS_CLUBS
    w = pm.blend_weight(n.get(club, 0), weak_prior=weak)
    out = float(np.exp((1.0 - w) * np.log(prior) + w * np.log(meas)))
    return out, w, int(n.get(club, 0))


def press_factor(club, season="2627", gw=None):
    """league_avg / ppda -> >1 means more press than average (more CBIRT)."""
    ppda, _, _ = resolve_ppda(club, season, gw)
    return LEAGUE_PPDA / max(ppda, 4.0)


def ppda_2627(club):
    """The unrevised judgment forecast. `resolve_ppda` is what the model uses."""
    return PPDA_2627.get(club, LEAGUE_PPDA)


def ppda_2526(club):
    return PPDA_2526.get(club, LEAGUE_PPDA)


def fetch_live_ppda(api_key=None, season="2025-2026"):
    """Live refresh from an event-data provider (Understat season PPDA is most
    stable; FBref/Opta/StatsBomb also expose it). Needs network; offline use the
    embedded tables. Returns {team: ppda}."""
    raise NotImplementedError("wire to Understat/FBref PPDA on a networked machine")


def selftest():
    """Offline. The blend must be inert without a feed and bounded with one."""
    import os
    _MEASURED_CACHE.clear()
    old = os.environ.get("PRESS_MEASURED")
    try:
        # switched off -> the judgment table, exactly as before this module existed
        os.environ["PRESS_MEASURED"] = "off"
        _MEASURED_CACHE.clear()
        for c in PPDA_2627:
            ppda, w, _ = resolve_ppda(c)
            assert w == 0.0 and ppda == PPDA_2627[c], c
            assert abs(press_factor(c) - LEAGUE_PPDA / PPDA_2627[c]) < 1e-12, c
        # 25/26 is the reference leg and is never revised
        os.environ["PRESS_MEASURED"] = "on"
        _MEASURED_CACHE.clear()
        assert resolve_ppda("Arsenal", "2526")[1] == 0.0

        # with a planted feed: blend is bounded by prior and measurement, monotone in n,
        # and lands on the geometric mean at w=1/2
        import press_measured as pm
        club = "Everton"; prior = PPDA_2627[club]
        for meas in (prior * 0.5, prior * 2.0):
            for n in (1, 5, 19, 38):
                _MEASURED_CACHE.clear(); _MEASURED_CACHE[None] = ({club: meas}, {club: n})
                out, w, got_n = resolve_ppda(club)
                assert got_n == n and abs(w - pm.blend_weight(n)) < 1e-12
                assert min(prior, meas) <= out <= max(prior, meas), (meas, n, out)
        _MEASURED_CACHE.clear(); _MEASURED_CACHE[None] = ({club: prior * 4.0}, {club: 40})
        w = pm.blend_weight(40)
        assert abs(resolve_ppda(club)[0] - prior * 4.0 ** w) < 1e-9

        # a weak-prior club moves further on identical evidence
        _MEASURED_CACHE.clear()
        _MEASURED_CACHE[None] = ({"Everton": 20.0, "Hull": 20.0}, {"Everton": 6, "Hull": 6})
        assert resolve_ppda("Hull")[1] > resolve_ppda("Everton")[1]

        # GW1 must barely move anything: the press table is ~92% noise at n=1
        _MEASURED_CACHE.clear()
        _MEASURED_CACHE[None] = ({c: 20.0 for c in PPDA_2627}, {c: 1 for c in PPDA_2627})
        moved = max(abs(press_factor(c) / (LEAGUE_PPDA / PPDA_2627[c]) - 1)
                    for c in PPDA_2627)
        assert moved < 0.04, f"GW1 moved press_factor by {moved:.1%}, expected <4%"
    finally:
        _MEASURED_CACHE.clear()
        if old is None:
            os.environ.pop("PRESS_MEASURED", None)
        else:
            os.environ["PRESS_MEASURED"] = old
    print("press_index selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    gw = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--gw=")), None)
    hdr = f"{'club':16s}{'prior':>7s}{'meas':>8s}{'n':>4s}{'w':>7s}{'used':>8s}{'factor':>8s}"
    print(hdr); print("-" * len(hdr))
    for c in sorted(PPDA_2627, key=PPDA_2627.get):
        ppda, w, n = resolve_ppda(c, gw=gw)
        meas, _ = _measured(gw)
        m = meas.get(c)
        flag = " *regime" if c in REGIME_PRESS_CLUBS else (
               " *promoted" if c in PROMOTED_PRESS_CLUBS else "")
        print(f"{c:16s}{PPDA_2627[c]:>7.1f}{(f'{m:.1f}' if m else '-'):>8s}"
              f"{n:>4d}{w:>7.2f}{ppda:>8.1f}{LEAGUE_PPDA / max(ppda, 4.0):>8.2f}{flag}")
