from __future__ import annotations
import config
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
import numpy as np, pandas as pd
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from multiseason import build_2425_panel

BASE = config.REPO


def two_season_evidence(older_weight=0.5, min_minutes_total=270):
    """Player-level evidence pooled across 24/25 and 25/26, keyed by player_code.
    Returns summed events and minutes with the older season down-weighted."""
    # --- 25/26 (recent, full weight) ---
    p25 = pd.read_pickle(config.PMS_PANEL)
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


# Seasons already represented in `two_season_evidence`. Deep history must not
# re-contribute these or the recent evidence is counted twice.
COVERED_BY_REPO = ("2024-25", "2025-26")


DEEP_HALF_LIFE = 1.5        # measured, see studies/deep_history_study.py §3


def deep_start_evidence(half_life=DEEP_HALF_LIFE, exclude=COVERED_BY_REPO,
                        seasons=None, verbose=True):
    """Recency-decayed START evidence from seasons BEFORE the repo's own coverage.

    *** TESTED NULL — DO NOT WIRE THIS INTO THE BOARD. ***
    [VERIFIED 2026-08-09, studies/deep_history_study.py §4] Held out 25/26 and predicted
    each player's start rate from 24/25 alone (what the model already has) versus 24/25
    plus eight decayed earlier seasons, same players, same metric:

        group                       shallow   deep    deep(active-only)
        all (n=533)                  0.2005  0.1988   0.2935
        regulars >=900 min (n=266)   0.2531  0.2519   0.2487
        fringe <900 min (n=267)      0.1480  0.1458   0.3381

    Deep history buys 0.0017 MAE — nothing — and the steelman is worse, not better.
    Two reasons, both structural rather than fixable by tuning:

      * Start rate is a property of a ROLE AT A CLUB, not of a player. Eight seasons
        encode a role a transfer has already invalidated: Kelleher (Liverpool backup
        for years, now first choice) moves 0.94 -> 0.50, Ampadu 0.93 -> 0.41. These are
        confident predictions in the wrong direction.
      * The denominator does not mean what the repo's does. merged_gw.csv carries a row
        per gameweek for every player in the FPL game whether or not they were in a
        squad, so 27.4% of player-seasons have ZERO appearances yet contribute 38 'did
        not start' events each. The repo's panel is per-MATCH, so its `games` means
        squad appearances. Excluding those seasons (`deep(active-only)` above) removes
        exactly the evidence that a fringe player does not play, and MAE nearly doubles
        for that group.

    Coverage compounds it: the players the prior exists to help gain almost nothing.
    Median weighted games gained is 0.44 for players thin in 25/26 versus 23.50 for
    players who already had enough.

    Kept because the loader is sound and reusable, and because a documented null is
    worth more than a deleted branch. `to_priors` ignores it unless explicitly passed.

    Returns one row per player_code with `hist_starts` / `hist_games`, already
    weighted, ready to be added to the Beta counts in `to_priors`.

    Only minutes-derived quantities. Not xG (penalty split unavailable — see
    fpl_history), not defcon (the stat did not exist before 25/26). Weighting
    continues the existing convention: the repo gives 25/26 weight 1.0 and 24/25
    weight 0.5, so season k back from 25/26 gets 0.5**(k-1) at half_life=1, making
    23/24 worth 0.25 and 16/17 worth 0.004. The eight extra seasons together add
    about ONE season-equivalent of evidence, concentrated on players with long
    careers — see studies/deep_history_study.py for what that is actually worth.
    """
    import fpl_history as fh
    seasons = [s for s in (seasons or fh.SEASONS) if s not in set(exclude)]
    if not seasons:
        return pd.DataFrame(columns=["player_code", "hist_starts", "hist_games"])
    panel = fh.load_history(seasons, verbose=verbose)
    ev = fh.season_evidence(panel)

    # k = seasons back from 2025/26, so 2024-25 -> 1, 2023-24 -> 2, ...
    order = list(fh.SEASONS)
    newest = len(order) - 1
    w = {s: 0.5 ** ((newest - order.index(s) - 1) / half_life) for s in seasons}
    ev["w"] = ev["season"].map(w)
    ev["hist_starts"] = ev["starts"] * ev["w"]
    ev["hist_games"] = ev["games"] * ev["w"]
    out = ev.groupby("player_code", as_index=False)[["hist_starts", "hist_games"]].sum()
    if verbose:
        print(f"[deep-history] {len(out)} players across {len(seasons)} pre-repo seasons; "
              f"weights {min(w.values()):.4f}..{max(w.values()):.3f}")
    return out


def to_priors(ev, revert=0.70, k0=3.0, pen_xg=0.79, deep_starts=None):
    """Gamma/Beta priors from pooled two-season evidence.

    `deep_starts`: optional frame from `deep_start_evidence()`. When given, its
    decayed counts are added to the Beta(start_a, start_b) minutes prior ONLY.
    Omitted (the default) the output is unchanged, so the live board does not move
    until the extension is explicitly validated and switched on.
    """
    hs = hg = None
    if deep_starts is not None and len(deep_starts):
        hs = dict(zip(deep_starts["player_code"], deep_starts["hist_starts"]))
        hg = dict(zip(deep_starts["player_code"], deep_starts["hist_games"]))
    PRIOR_INV = {"GK": 0.02, "DEF": 0.11, "MID": 0.27, "FWD": 0.42}
    PRIOR_XA = {"GK": 0.01, "DEF": 0.05, "MID": 0.13, "FWD": 0.10}
    PRIOR_DC = {"GK": 0.0, "DEF": 7.6, "MID": 8.4, "FWD": 4.7}
    out = []
    for _, r in ev.iterrows():
        n90 = r.mins / 90.0
        pos = r.pos if r.pos in PRIOR_INV else "MID"
        # deep history extends the minutes Beta only; the Gamma priors keep the
        # repo's Opta-sourced two-season evidence untouched
        d_st = hs.get(r.player_code, 0.0) if hs else 0.0
        d_gm = hg.get(r.player_code, 0.0) if hg else 0.0
        out.append({
            "player_code": r.player_code, "pos": pos, "minutes": r.mins,
            "npxgi_alpha": PRIOR_INV[pos] * k0 + revert * (r.npxg + r.xa),
            "npxgi_beta": k0 + revert * n90,
            "xa_alpha": PRIOR_XA[pos] * k0 + revert * r.xa,
            "xa_beta": k0 + revert * n90,
            "defcon_alpha": PRIOR_DC[pos] * k0 + revert * r.defcon,
            "defcon_beta": k0 + revert * n90,
            "start_a": 2.0 + revert * (r.starts + d_st),
            "start_b": 2.0 + revert * (max(r.games - r.starts, 0)
                                       + max(d_gm - d_st, 0.0)),
            "sub_app_rate": float(np.clip((r.apps - r.starts) / max(r.games, 1), 0, 1)),
            "pen_xg90_measured": (r.pens + r.pens_miss) * pen_xg / max(n90, 1e-6),
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    ev = two_season_evidence()
    pri = to_priors(ev)
    pri.to_pickle(config.MS_PRIORS)
    print(f"two-season evidence: {len(ev)} players")
    print(f"  mean pooled minutes {ev.mins.mean():.0f} "
          f"(vs single-season typical ~1500)")
    # how much extra evidence does pooling buy?
    p25 = pd.read_pickle(config.PMS_PANEL)
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
