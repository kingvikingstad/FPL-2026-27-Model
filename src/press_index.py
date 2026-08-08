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


def press_factor(club, season="2627"):
    """league_avg / ppda -> >1 means more press than average (more CBIRT)."""
    tbl = PPDA_2627 if season == "2627" else PPDA_2526
    ppda = tbl.get(club, LEAGUE_PPDA)
    return LEAGUE_PPDA / max(ppda, 4.0)


def ppda_2627(club):
    return PPDA_2627.get(club, LEAGUE_PPDA)


def ppda_2526(club):
    return PPDA_2526.get(club, LEAGUE_PPDA)


def fetch_live_ppda(api_key=None, season="2025-2026"):
    """Live refresh from an event-data provider (Understat season PPDA is most
    stable; FBref/Opta/StatsBomb also expose it). Needs network; offline use the
    embedded tables. Returns {team: ppda}."""
    raise NotImplementedError("wire to Understat/FBref PPDA on a networked machine")


if __name__ == "__main__":
    for c in sorted(PPDA_2627, key=PPDA_2627.get):
        flag = " *regime-est" if c in REGIME_PRESS_CLUBS else ""
        print(f"{c:16s} PPDA {PPDA_2627[c]:5.1f}  press_factor {press_factor(c):.2f}{flag}")
