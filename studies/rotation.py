import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
rotation.py — European / cup congestion as a rotation signal
============================================================
The dominant irreducible risk in every projection so far has been minutes:
who actually starts. Midweek European and cup football is the main *predictable*
driver of surprise rotation, and the FPL-Core-Insights repo ships full
Champions League / Europa / Conference / EFL Cup coverage keyed to FPL team codes.

This module:
  1. builds, for every (team, PL gameweek), the days between the team's most
     recent non-PL match and its PL kickoff  -> `rest_days`
  2. validates empirically whether short rest actually depresses minutes
  3. exposes a multiplier that feeds the Beta start prior in signals.py

Team references in the tournament files use teams.csv **code** (Liverpool=14,
Arsenal=3, Man City=43); non-PL opponents are NaN.
"""
from __future__ import annotations
import glob, os
import numpy as np, pandas as pd

BASE = config.REPO
NON_PL = ["Champions League", "Europa League", "Conference League", "EFL Cup"]


def load_teams(season="2025-2026"):
    t = pd.read_csv(f"{BASE}/{season}/teams.csv")
    return t[["code", "id", "name", "short_name"]]


def _read_matches(path):
    try:
        m = pd.read_csv(path)
    except Exception:
        return None
    if "kickoff_time" not in m.columns:
        return None
    return m


def european_matches(season="2025-2026"):
    """All non-PL matches involving PL clubs: (team_code, kickoff, competition)."""
    rows = []
    for comp in NON_PL:
        for p in sorted(glob.glob(f"{BASE}/{season}/By Tournament/{comp}/GW*/matches.csv")):
            m = _read_matches(p)
            if m is None:
                continue
            for side in ["home_team", "away_team"]:
                sub = m[["kickoff_time", side, "finished"]].dropna(subset=[side])
                for _, r in sub.iterrows():
                    rows.append({"team_code": int(r[side]),
                                 "kickoff": r["kickoff_time"], "comp": comp})
    d = pd.DataFrame(rows)
    if len(d):
        d["kickoff"] = pd.to_datetime(d["kickoff"], errors="coerce", utc=True)
        d = d.dropna(subset=["kickoff"]).drop_duplicates()
    return d


def pl_matches(season="2025-2026"):
    """PL matches by gameweek with kickoff time, long form per team."""
    rows = []
    for p in sorted(glob.glob(f"{BASE}/{season}/By Gameweek/GW*/matches.csv")):
        m = _read_matches(p)
        if m is None:
            continue
        gw_dir = int(os.path.basename(os.path.dirname(p)).replace("GW", ""))
        for _, r in m.iterrows():
            gw = int(r["gameweek"]) if pd.notna(r.get("gameweek")) else gw_dir
            for side in ["home_team", "away_team"]:
                if pd.notna(r.get(side)):
                    rows.append({"team_code": int(r[side]), "gameweek": gw,
                                 "kickoff": r["kickoff_time"]})
    d = pd.DataFrame(rows).drop_duplicates()
    d["kickoff"] = pd.to_datetime(d["kickoff"], errors="coerce", utc=True)
    return d.dropna(subset=["kickoff"])


def rest_days_table(season="2025-2026"):
    """For each (team_code, gameweek): days since the team's last non-PL match."""
    euro = european_matches(season)
    pl = pl_matches(season)
    if not len(euro) or not len(pl):
        return pd.DataFrame()
    out = []
    for code, g in pl.groupby("team_code"):
        e = euro[euro.team_code == code].sort_values("kickoff")
        for _, r in g.iterrows():
            prior = e[e.kickoff < r.kickoff]
            if len(prior):
                gap = (r.kickoff - prior.kickoff.iloc[-1]).total_seconds() / 86400.0
                comp = prior.comp.iloc[-1]
            else:
                gap, comp = np.nan, None
            out.append({"team_code": code, "gameweek": r.gameweek,
                        "pl_kickoff": r.kickoff, "rest_days": gap, "last_comp": comp})
    d = pd.DataFrame(out)
    # congestion flag: played a midweek non-PL game within 4 days of the PL kickoff
    d["congested"] = (d.rest_days <= 4.0).fillna(False)
    return d


# ---------------------------------------------------------------------------
# Validation: does congestion actually depress minutes?
# ---------------------------------------------------------------------------
def validate(hist_csv=config.FPL_DATA_STATS, season="2025-2026"):
    rest = rest_days_table(season)
    teams = load_teams(season)
    rest = rest.merge(teams[["code", "name"]], left_on="team_code", right_on="code")
    h = pd.read_csv(hist_csv)
    h["pos"] = h.element_type.map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    # align team naming
    h["team"] = h.team_name.replace({"Man Utd": "Man Utd", "Spurs": "Spurs"})
    d = h.merge(rest[["name", "gameweek", "rest_days", "congested", "last_comp"]],
                left_on=["team_name", "gameweek"], right_on=["name", "gameweek"],
                how="left")
    # restrict to established starters so we measure ROTATION, not squad depth
    starters = (d.groupby("id").minutes.transform(lambda s: (s >= 60).mean()) >= 0.5)
    d = d[starters & d.rest_days.notna()].copy()
    d["started"] = (d.minutes >= 60).astype(int)
    d["congested"] = d["congested"].astype(bool)
    return d


def fixed_effects(d):
    """WITHIN-PLAYER comparison. Raw congested-vs-rested is confounded: only the
    big clubs play in Europe, and their starters are better players. Demeaning by
    player removes that selection entirely, so what's left is the causal-ish
    rotation effect for the same player in congested vs rested weeks."""
    d = d.copy()
    for c in ["minutes", "started", "total_points"]:
        d[f"{c}_dm"] = d[c] - d.groupby("id")[c].transform("mean")
    # only players observed in BOTH states can identify the effect
    both = d.groupby("id")["congested"].transform("nunique") > 1
    return d[both]


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    euro = european_matches()
    print(f"non-PL matches involving PL clubs, 25/26: {len(euro)}")
    print(euro.comp.value_counts().to_string())
    rest = rest_days_table()
    print(f"\nteam-gameweek rows: {len(rest)}  | congested (<=4d): {int(rest.congested.sum())}")

    d = validate()
    print(f"\nvalidation sample (established starters): {len(d)} player-gameweeks")
    print("\n=== DOES EUROPEAN CONGESTION DEPRESS MINUTES? ===")
    g = d.groupby("congested").agg(n=("minutes", "size"), mean_minutes=("minutes", "mean"),
                                   start_rate=("started", "mean"),
                                   mean_points=("total_points", "mean"))
    print(g.round(3).to_string())
    from scipy import stats
    print("\n^ NAIVE comparison is CONFOUNDED: only big clubs play in Europe,")
    print("  so 'congested' also means 'better team'. Within-player below.\n")

    fe = fixed_effects(d)
    print("=== WITHIN-PLAYER FIXED EFFECTS (same player, congested vs rested) ===")
    print(f"players identifying the effect: {fe.id.nunique()}  rows: {len(fe)}")
    g2 = fe.groupby("congested").agg(n=("minutes_dm", "size"),
                                     minutes_dm=("minutes_dm", "mean"),
                                     started_dm=("started_dm", "mean"),
                                     points_dm=("total_points_dm", "mean"))
    print(g2.round(3).to_string())
    a = fe[fe.congested]; b = fe[~fe.congested]
    t, p = stats.ttest_ind(a.minutes_dm, b.minutes_dm, equal_var=False)
    t2, p2 = stats.ttest_ind(a.started_dm, b.started_dm, equal_var=False)
    print(f"\nminutes  effect: {a.minutes_dm.mean()-b.minutes_dm.mean():+.2f} min  (t={t:.2f}, p={p:.4f})")
    print(f"start-rate effect: {a.started_dm.mean()-b.started_dm.mean():+.4f}      (t={t2:.2f}, p={p2:.4f})")

    print("\n=== BY REST-DAY BUCKET ===")
    d["bucket"] = pd.cut(d.rest_days, [0, 3, 4, 5, 7, 100],
                         labels=["<=3d", "4d", "5d", "6-7d", "8d+"])
    print(d.groupby("bucket", observed=True).agg(
        n=("minutes", "size"), mean_minutes=("minutes", "mean"),
        start_rate=("started", "mean"), pts=("total_points", "mean")).round(3).to_string())

    print("\n=== BY POSITION (congested vs rested, start rate) ===")
    piv = d.pivot_table(index="pos", columns="congested", values="started", aggfunc="mean")
    piv.columns = ["rested", "congested"]
    piv["delta"] = piv.congested - piv.rested
    print(piv.round(3).to_string())
