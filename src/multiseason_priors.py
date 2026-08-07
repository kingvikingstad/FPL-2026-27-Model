"""
multiseason_priors.py — two-season hierarchical priors for 2026/27
===================================================================
Validated by multiseason.py: prior-season data roughly DOUBLES early-season
predictive power, and the benefit is concentrated exactly where theory says it
should be — when current-season evidence is thin.

    current-season evidence   current-only    blended     gain
    < 1 game                    -0.010         +0.176    +0.186
    1-3 games                   +0.051         +0.081    +0.030
    3-6 games                   -0.011         +0.015    +0.026
    6-10 games                  +0.212         +0.211    -0.001   <- prior stops mattering

So the prior carries the model until roughly 6-10 games of the new season, then
becomes irrelevant. That's the classic empirical-Bayes reliability curve, and it
is precisely the regime a pre-season 26/27 projection sits in.

IMPLEMENTATION
  Both 24/25 and 25/26 become evidence for the 26/27 prior, with 24/25
  down-weighted for recency (`older_weight`, default 0.5). More total evidence
  means tighter Gamma posteriors and less estimation noise, especially for
  players with a short 25/26 (injury, mid-season transfer, late debut).
"""
from __future__ import annotations
import numpy as np, pandas as pd
import sys; sys.path.insert(0, "/home/claude/fpl")
from multiseason import build_2425_panel

BASE = "/home/claude/repo/FPL-Core-Insights-main/data"


def two_season_evidence(older_weight=0.5, min_minutes_total=270):
    """Player-level evidence pooled across 24/25 and 25/26, keyed by player_code.
    Returns summed events and minutes with the older season down-weighted."""
    # --- 25/26 (recent, full weight) ---
    p25 = pd.read_pickle("/tmp/pms_panel.pkl")
    codes25 = pd.read_csv(f"{BASE}/2025-2026/players.csv")[["player_id", "player_code"]]
    p25 = p25.merge(codes25, on="player_id", how="left")
    a25 = p25.groupby(["player_code", "pos"], dropna=False).agg(
        mins=("mins", "sum"), npxg=("npxg", "sum"), xa=("xa_", "sum"),
        defcon=("defcon_raw", "sum"),
        starts=("mins", lambda s: (s >= 60).sum()), games=("mins", "size"),
        apps=("mins", lambda s: (s > 0).sum()),
        pens=("pens_scored", "sum"), pens_miss=("pens_missed", "sum"),
    ).reset_index()

    # --- 24/25 (older, down-weighted) ---
    p24 = build_2425_panel()
    a24 = p24.groupby(["player_code", "pos"], dropna=False).agg(
        mins=("mins", "sum"), npxg=("npxg", "sum"), xa=("xa_", "sum"),
        defcon=("defcon_raw", "sum"),
        starts=("mins", lambda s: (s >= 60).sum()), games=("mins", "size"),
        apps=("mins", lambda s: (s > 0).sum()),
    ).reset_index()
    for c in ["mins", "npxg", "xa", "defcon", "starts", "games", "apps"]:
        a24[c] = a24[c] * older_weight
    a24["pens"] = 0.0; a24["pens_miss"] = 0.0     # 24/25 lacks the penalty split

    both = pd.concat([a25, a24], ignore_index=True)
    ev = both.groupby("player_code", dropna=False).agg(
        pos=("pos", "first"),
        mins=("mins", "sum"), npxg=("npxg", "sum"), xa=("xa", "sum"),
        defcon=("defcon", "sum"), starts=("starts", "sum"),
        games=("games", "sum"), apps=("apps", "sum"),
        pens=("pens", "sum"), pens_miss=("pens_miss", "sum"),
    ).reset_index()
    ev = ev[ev.mins >= min_minutes_total].copy()
    ev["n_seasons"] = ev.player_code.map(
        both.groupby("player_code").size())
    return ev


def to_priors(ev, revert=0.70, k0=3.0, pen_xg=0.79):
    """Gamma/Beta priors from pooled two-season evidence."""
    PRIOR_INV = {"GK": 0.02, "DEF": 0.11, "MID": 0.27, "FWD": 0.42}
    PRIOR_XA = {"GK": 0.01, "DEF": 0.05, "MID": 0.13, "FWD": 0.10}
    PRIOR_DC = {"GK": 0.0, "DEF": 7.6, "MID": 8.4, "FWD": 4.7}
    out = []
    for _, r in ev.iterrows():
        n90 = r.mins / 90.0
        pos = r.pos if r.pos in PRIOR_INV else "MID"
        out.append({
            "player_code": r.player_code, "pos": pos, "minutes": r.mins,
            "npxgi_alpha": PRIOR_INV[pos] * k0 + revert * (r.npxg + r.xa),
            "npxgi_beta": k0 + revert * n90,
            "xa_alpha": PRIOR_XA[pos] * k0 + revert * r.xa,
            "xa_beta": k0 + revert * n90,
            "defcon_alpha": PRIOR_DC[pos] * k0 + revert * r.defcon,
            "defcon_beta": k0 + revert * n90,
            "start_a": 2.0 + revert * r.starts,
            "start_b": 2.0 + revert * max(r.games - r.starts, 0),
            "sub_app_rate": float(np.clip((r.apps - r.starts) / max(r.games, 1), 0, 1)),
            "pen_xg90_measured": (r.pens + r.pens_miss) * pen_xg / max(n90, 1e-6),
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    ev = two_season_evidence()
    pri = to_priors(ev)
    pri.to_pickle("/tmp/ms_priors.pkl")
    print(f"two-season evidence: {len(ev)} players")
    print(f"  mean pooled minutes {ev.mins.mean():.0f} "
          f"(vs single-season typical ~1500)")
    # how much extra evidence does pooling buy?
    p25 = pd.read_pickle("/tmp/pms_panel.pkl")
    codes25 = pd.read_csv(f"{BASE}/2025-2026/players.csv")[["player_id", "player_code"]]
    p25 = p25.merge(codes25, on="player_id", how="left")
    m25 = p25.groupby("player_code").mins.sum()
    j = ev.set_index("player_code").join(m25.rename("mins_2526"), how="left")
    j["extra"] = j.mins - j.mins_2526.fillna(0)
    print(f"  players gaining evidence from 24/25: {(j.extra > 50).sum()}")
    print(f"  median extra weighted minutes: {j.extra.median():.0f}")
    thin = j[j.mins_2526.fillna(0) < 900]
    print(f"  players THIN in 25/26 (<900 min) who gain: {(thin.extra > 50).sum()} "
          f"— these are where it matters most")
    print(f"\nwrote priors for {len(pri)} players -> /tmp/ms_priors.pkl")
