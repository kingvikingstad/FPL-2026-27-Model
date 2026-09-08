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
import defcon_roles as dcr

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
        # chances created = the repo's key-pass equivalent, and the EXPOSURE for the
        # assist-quality term. See to_priors and studies/rate_components.py.
        kp=("chances_created", "sum"),
        starts=("mins", lambda s: (s >= 60).sum()), games=("mins", "size"),
        apps=("mins", lambda s: (s > 0).sum()),
        # minutes accumulated in appearances of 60+, so `cond_min / starts` is the
        # player's own E[minutes | started] — see to_priors
        cond_min=("mins", lambda s: s[s >= 60].sum()),
        pens=("pens_scored", "sum"), pens_miss=("pens_missed", "sum"),
    ).reset_index()

    # --- 24/25 (older, down-weighted) ---
    p24 = build_2425_panel()
    a24 = p24.groupby(["player_code", "pos"], dropna=False).agg(
        mins=("mins", "sum"), npxg=("npxg", "sum"), xa=("xa_", "sum"),
        defcon=("defcon_raw", "sum"),
        kp=("chances_created", "sum"),
        starts=("mins", lambda s: (s >= 60).sum()), games=("mins", "size"),
        apps=("mins", lambda s: (s > 0).sum()),
        cond_min=("mins", lambda s: s[s >= 60].sum()),
    ).reset_index()
    for c in ["mins", "npxg", "xa", "defcon", "kp", "starts", "games", "apps", "cond_min"]:
        a24[c] = a24[c] * older_weight
    a24["pens"] = 0.0; a24["pens_miss"] = 0.0     # 24/25 lacks the penalty split

    # MINUTES THAT CAN ACTUALLY CARRY A DEFCON.
    # `defensive_contributions` does not exist in 24/25 — the column reads as all zeros,
    # so a24 contributes 750,949 minutes to the pooled denominator and exactly 0 to the
    # numerator. Every DefCon rate was therefore diluted by that player's 24/25 share of
    # minutes: measured r(24/25 share, dilution) = -1.000, i.e. arithmetic, not noise.
    # Pooled defender rate came out at 6.41 per 90 against a measured 8.36; restricting
    # the denominator to 25/26 gives 8.40. The other Gamma channels (npxg, xa) are
    # genuinely present in both seasons and keep the pooled denominator.
    a25["mins_dc"] = a25["mins"]
    a24["mins_dc"] = 0.0
    both = pd.concat([a25, a24], ignore_index=True)
    ev = both.groupby("player_code", dropna=False).agg(
        pos=("pos", "first"),
        mins=("mins", "sum"), npxg=("npxg", "sum"), xa=("xa", "sum"),
        defcon=("defcon", "sum"), mins_dc=("mins_dc", "sum"), kp=("kp", "sum"),
        starts=("starts", "sum"),
        games=("games", "sum"), apps=("apps", "sum"),
        cond_min=("cond_min", "sum"),
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


def _shrunk_minutes(r, pos, pos_minutes, k):
    """E[minutes | started] for one player, shrunk toward his position's mean.

    Falls back to the positional constant when the player has no 60+ appearances, which
    is the correct default — it is exactly what bayes_model used before this existed.
    Clipped to [60, 90]: below 60 is not a start by this model's definition, and nobody
    plays more than 90 of regulation.
    """
    n = float(getattr(r, "starts", 0.0) or 0.0)
    tot = float(getattr(r, "cond_min", 0.0) or 0.0)
    base = pos_minutes.get(pos, 85.3)
    if n <= 0 or not np.isfinite(tot) or tot <= 0:
        return base
    own = tot / n
    w = n / (n + k)
    return float(np.clip(w * own + (1 - w) * base, 60.0, 90.0))



def _xa_alpha(pos, r, n90, k0, revert, prior_xa, prior_kp90, prior_q,
              k_vol, k_qual, min_n90):
    """Gamma shape for the assist channel, split into volume x quality where licensed.

    `xa_beta` is left exactly as it was (k0 + revert*n90), so this changes the prior MEAN
    and nothing about its strength. Rewriting beta as well would quietly alter how much
    the simulation's draws spread, which is a separate question from where they centre.
    """
    if n90 < min_n90:
        return prior_xa[pos] * k0 + revert * r.xa          # pooled, as before
    kp = float(getattr(r, "kp", 0.0) or 0.0)
    # (k*prior + n90 * kp/n90) / (k + n90) reduces to (k*prior + kp) / (k + n90)
    vol = (k_vol * prior_kp90[pos] + kp) / (k_vol + n90)
    qual = ((k_qual * prior_q[pos] + float(r.xa)) / (k_qual + kp) if kp > 0
            else prior_q[pos])
    return vol * qual * (k0 + revert * n90)

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
    # E[minutes | started], shrunk toward the positional mean. bayes_model previously
    # applied 90 to everyone, then a positional constant; both ignore that a 90-minute
    # centre-half and a forward hooked on 65 differ by ~15 minutes of exposure every week.
    # [VERIFIED 2026-08-11, studies/minutes_persistence.py, 1,755 consecutive
    # player-season pairs] this quantity has split-half reliability 0.82 and persists at
    # r=0.65 (0.80 disattenuated). Out of sample it beats the positional constant on MAE
    # 1.968 vs 2.649 — a 25.7% gain, clustered CI (+0.579, +0.784) — and removes the
    # constant's +1.2 minute bias, landing at +0.02.
    # k comes from the measured reliability, not from taste: k = (1-rel)/rel * median n.
    POS_MINUTES = {"GK": 89.9, "DEF": 87.5, "MID": 83.1, "FWD": 81.5}
    MIN_K = 5.7

    PRIOR_INV = {"GK": 0.02, "DEF": 0.11, "MID": 0.27, "FWD": 0.42}
    PRIOR_XA = {"GK": 0.01, "DEF": 0.05, "MID": 0.13, "FWD": 0.10}
    # ASSIST CHANNEL: VOLUME x QUALITY, not one pooled rate.
    #
    #     xA/90  =  (chances created/90)  x  (xA/chance)
    #
    # Those factors have very different reliability. Split-half, match level,
    # Spearman-Brown corrected (studies/rate_components.py, Understat 2014-15..2025-26):
    #
    #     chances created/90   0.850          xA/chance   0.218
    #     shots/90             0.897          npxG/shot   0.610
    #
    # and their EXPOSURES differ too: a rate per 90 is measured on minutes, a rate per
    # chance is measured on chances. Shrinking the product on minutes alone cannot
    # express that, so it over-shrinks the stable factor and under-shrinks the noisy one.
    #
    # Out-of-sample, leave-one-season-out over 2,707 consecutive-season pairs, predicting
    # next-season xA/90:  pooled MAE 0.04359 -> split 0.04271, difference -0.00087 with a
    # season-clustered 95% CI of (-0.00151, -0.00024). Spearman 0.775 -> 0.789. The CI
    # excludes zero, which was the pre-registered condition for shipping this.
    #
    # THE SAME TEST REJECTED THE SPLIT FOR npxG: MAE 0.05197 -> 0.05387, CI
    # (-0.00073, +0.00458), which includes zero. So npxG/90 stays pooled below. The
    # reliability gap there is real but smaller (0.897 vs 0.610, against 0.850 vs 0.218
    # for assists) and the pooled estimator already handles it. Splitting both would have
    # been the intuitive move and it would have been wrong.
    K_KP_VOL = 0.5        # fitted on the training seasons; volume is trusted almost at once
    K_KP_QUAL = 25.0      # quality needs ~25 chances before its own rate outweighs the prior
    # Levels measured on THIS repo's scale, not Understat's. The two sources agree closely
    # on volume (chances/90 0.854 here vs key passes/90 0.825) but not on the per-event
    # rate (0.097 vs 0.120), so the shrinkage CONSTANTS are borrowed from Understat — the
    # only source with enough seasons to fit them — while the TARGETS come from here.
    PRIOR_KP90 = {"GK": 0.057, "DEF": 0.569, "MID": 1.287, "FWD": 0.829}
    PRIOR_XA_PER_KP = {"GK": 0.052, "DEF": 0.102, "MID": 0.098, "FWD": 0.076}
    # The calibration sample was players with >= 450 minutes. K_KP_VOL = 0.5 means volume
    # is barely shrunk, which is right for that regime and reckless outside it: a player
    # with two appearances would have his noisy rate taken almost at face value. Below the
    # threshold the pooled prior is used, because that is the regime it was fitted for.
    SPLIT_MIN_N90 = 5.0
    # DEF corrected 7.6 -> 8.590, the measured pooled per-90 rate over 2,934 appearances
    # (studies/defcon_matchups.csv). The old value sat BELOW the measurement, so every
    # thin-history defender was shrunk toward a target that was too low before any
    # question of role arose. GK/MID/FWD are untouched — no equivalent measurement.
    PRIOR_DC = {"GK": 0.0, "DEF": dcr.RATE_DEF_POOLED, "MID": 8.4, "FWD": 4.7}
    # Centre-back / full-back split, applied ONLY to the DefCon prior mean. `pos` stays
    # DEF everywhere else, so the threshold, clean-sheet and goal multipliers are
    # unchanged. Unlabelled defenders fall back to the pooled rate above.
    _roles = dcr.role_map()
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
            "xa_alpha": _xa_alpha(pos, r, n90, k0, revert, PRIOR_XA, PRIOR_KP90,
                                  PRIOR_XA_PER_KP, K_KP_VOL, K_KP_QUAL, SPLIT_MIN_N90),
            "xa_beta": k0 + revert * n90,
            # DefCon exposure is 25/26 minutes only — see two_season_evidence. Using the
            # pooled n90 here halved the implied hit rate for every player who featured
            # in 24/25, and inverted the ordering so cold-start defenders (on the 7.6
            # prior) out-rated established ones (diluted to 6.31).
            "defcon_alpha": (dcr.prior_rate(pos, _roles.get(r.player_code))
                             or PRIOR_DC[pos]) * k0 + revert * r.defcon,
            "defcon_beta": k0 + revert * (getattr(r, "mins_dc", r.mins) / 90.0),
            "start_a": 2.0 + revert * (r.starts + d_st),
            "start_b": 2.0 + revert * (max(r.games - r.starts, 0)
                                       + max(d_gm - d_st, 0.0)),
            "sub_app_rate": float(np.clip((r.apps - r.starts) / max(r.games, 1), 0, 1)),
            "exp_minutes": _shrunk_minutes(r, pos, POS_MINUTES, MIN_K),
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
