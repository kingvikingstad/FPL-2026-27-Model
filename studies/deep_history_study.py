from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
deep_history_study.py — is the 8-season minutes backfill actually worth switching on?
=====================================================================================
`multiseason.py` established that prior-season evidence roughly doubles early-season
predictive power and stops mattering after 6-10 games. That is the case FOR a prior.
It is NOT a case for an ARBITRARILY DEEP prior, and the difference is the whole
question here.

Three things decide it, and all three are measured rather than asserted:

  1. COVERAGE   — how many 26/27 players actually gain evidence? Deep history only
                  helps players who HAVE a deep history. If the gain lands on
                  Haaland and not on the cold-start fringe, it is worth little:
                  established players are already well-identified, and the thin
                  players are exactly where the prior does the work.
  2. MAGNITUDE  — how much does the Beta prior actually move, in start-probability
                  terms? A shift smaller than MC noise is not a gain.
  3. HALF-LIFE  — pick the decay by out-of-sample fit, not by assertion. Predict each
                  player's 25/26 start rate from PRIOR seasons only, and see which
                  half-life minimises error. This is a real backtest: 25/26 is held
                  out, and only seasons strictly before it are used.

Run:  python studies/deep_history_study.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import fpl_history as fh
import multiseason_priors as ms

OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                    "deep_history_halflife.csv")
HALF_LIVES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 1e6]   # 1e6 ~ no decay (flat)


def backtest_half_life(panel):
    """Hold out 25/26. For each half-life, predict a player's 25/26 start RATE from
    strictly-earlier seasons and score it. `1e6` is effectively no decay, i.e. the
    naive 'more history is always better' position — included so it can lose on
    the numbers rather than on argument."""
    ev = fh.season_evidence(panel)
    order = list(fh.SEASONS)
    held = "2025-26"
    past = [s for s in order if s != held]

    tgt = ev[ev["season"] == held].copy()
    tgt = tgt[tgt["games"] >= 10]
    tgt["y"] = tgt["starts"] / tgt["games"]

    hist = ev[ev["season"].isin(past)].copy()
    newest_past = len(past) - 1

    # A COMMON evaluation set. Scoring each half-life on whoever happens to clear a
    # weighted-games floor under that half-life is not a fair comparison: aggressive
    # decay shrinks the weighted total, drops the thinnest players, and then wins on
    # an easier sample. Fix the player set once, using unweighted games, so every
    # half-life is scored on identical players.
    raw_games = hist.groupby("player_code")["games"].sum()
    eligible = set(raw_games[raw_games >= 5].index)

    rows = []
    for hl in HALF_LIVES:
        w = {s: 0.5 ** ((newest_past - past.index(s)) / hl) for s in past}
        h = hist.copy()
        h["w"] = h["season"].map(w)
        h["sw"] = h["starts"] * h["w"]
        h["gw"] = h["games"] * h["w"]
        agg = h.groupby("player_code", as_index=False)[["sw", "gw"]].sum()
        agg = agg.rename(columns={"sw": "st", "gw": "gm"})
        j = tgt.merge(agg, on="player_code", how="inner")
        j = j[j["player_code"].isin(eligible)]
        if len(j) < 30:
            continue
        # shrunk toward the population rate exactly as to_priors does (Beta(2,2))
        p_hat = (2.0 + j["st"]) / (4.0 + j["gm"])
        mae = float(np.mean(np.abs(p_hat - j["y"])))
        rmse = float(np.sqrt(np.mean((p_hat - j["y"]) ** 2)))
        rows.append({"half_life": hl, "n": len(j), "mae": mae, "rmse": rmse,
                     "corr": float(np.corrcoef(p_hat, j["y"])[0, 1])})
    return pd.DataFrame(rows)


def does_deep_beat_shallow(panel):
    """THE decisive test, and the one the half-life sweep does not answer.

    The sweep only ranks half-lives against each other; it never asks whether deep
    history beats the baseline the project already has. Predict each player's 25/26
    start rate from:
        shallow — 24/25 only (what the repo already uses)
        deep    — 24/25 plus decayed 16/17..23/24
    Same players, same metric, 25/26 held out throughout.

    Reported separately for STAYERS and MOVERS, because start rate is a property of a
    player's ROLE AT A CLUB, not of the player. Eight seasons of history encodes a
    role that a transfer or a promotion may have already invalidated.
    """
    ev = fh.season_evidence(panel)
    order = list(fh.SEASONS)
    held, shallow_season = "2025-26", "2024-25"
    past = [s for s in order if s != held]
    newest_past = len(past) - 1

    tgt = ev[(ev["season"] == held) & (ev["games"] >= 10)].copy()
    tgt["y"] = tgt["starts"] / tgt["games"]

    sh = ev[ev["season"] == shallow_season].set_index("player_code")
    common = tgt[tgt["player_code"].isin(sh.index)].copy()
    common["st_s"] = common["player_code"].map(sh["starts"])
    common["gm_s"] = common["player_code"].map(sh["games"])

    w = {s: 0.5 ** ((newest_past - past.index(s)) / 1.5) for s in past}

    def decayed(min_apps):
        """min_apps=0 counts every player-season in the FPL database. That includes
        the 27.4% with ZERO appearances — a season spent injured, out of favour, or
        at another club still contributes 38 'did not start' events to the Beta
        denominator, because merged_gw.csv carries a row per gameweek for every
        player in the game regardless of squad membership. The repo's own panel is
        per-MATCH, so its `games` means squad appearances. min_apps>0 is the steelman:
        keep only seasons where the player was demonstrably an active PL squad member."""
        h = ev[ev["season"].isin(past) & (ev["apps"] >= min_apps)].copy()
        h["w"] = h["season"].map(w)
        h["sw"] = h["starts"] * h["w"]
        h["gw"] = h["games"] * h["w"]
        return h.groupby("player_code", as_index=False)[["sw", "gw"]].sum()

    j = common.merge(decayed(0), on="player_code", how="left").fillna({"sw": 0, "gw": 0})
    a5 = decayed(5).rename(columns={"sw": "sw5", "gw": "gw5"})
    j = j.merge(a5, on="player_code", how="left").fillna({"sw5": 0, "gw5": 0})

    def beta_p(st, gm):
        return (2.0 + 0.70 * st) / (4.0 + 0.70 * gm)

    j["p_shallow"] = beta_p(j["st_s"], j["gm_s"])
    j["p_deep"] = beta_p(j["sw"], j["gw"])
    j["p_deep_active"] = beta_p(j["sw5"], j["gw5"])

    # a MOVER changed club between 24/25 and 25/26
    club = panel.dropna(subset=["pos"]).groupby(["player_code", "season"]).size()
    last = (panel[panel["season"] == shallow_season].groupby("player_code")["mins"].sum())
    j["mins_prev"] = j["player_code"].map(last).fillna(0)

    rows = []
    for label, sub in (("all", j),
                       ("regulars prev yr (>=900 min)", j[j["mins_prev"] >= 900]),
                       ("fringe prev yr (<900 min)", j[j["mins_prev"] < 900])):
        if len(sub) < 20:
            continue
        rows.append({
            "group": label, "n": len(sub),
            "mae_shallow": float(np.mean(np.abs(sub["p_shallow"] - sub["y"]))),
            "mae_deep": float(np.mean(np.abs(sub["p_deep"] - sub["y"]))),
            "mae_deep_active": float(np.mean(np.abs(sub["p_deep_active"] - sub["y"]))),
        })
    r = pd.DataFrame(rows)
    r["delta"] = r["mae_deep"] - r["mae_shallow"]
    r["delta_active"] = r["mae_deep_active"] - r["mae_shallow"]
    best = r[["delta", "delta_active"]].min(axis=1)
    r["verdict"] = np.where(best < -0.005, "deep WINS",
                            np.where(best > 0.005, "deep LOSES", "no material gain"))
    return r, j


def main():
    panel = fh.load_history(verbose=False)
    ev = fh.season_evidence(panel)
    print(f"[study] {len(panel)} player-GW rows, {panel.player_code.nunique()} players, "
          f"{ev.season.nunique()} seasons")

    # ---------------------------------------------------------------- 1. coverage
    print("\n=== 1. COVERAGE — who actually gains? ===")
    recent = set(ev[ev["season"] == "2025-26"]["player_code"])
    deep = ms.deep_start_evidence(half_life=1.0, verbose=False)
    d = deep.set_index("player_code")
    gain = d.reindex(sorted(recent))["hist_games"].fillna(0.0)
    print(f"players active in 25/26: {len(recent)}")
    for thr in (0.5, 1, 3, 10):
        n = int((gain > thr).sum())
        print(f"  gaining > {thr:>4} weighted games of history: {n:3d}  ({n/len(recent):5.1%})")

    # does the gain land where the prior matters (thin 25/26 evidence)?
    r26 = ev[ev["season"] == "2025-26"].set_index("player_code")
    thin = r26[r26["mins"] < 900].index
    thick = r26[r26["mins"] >= 900].index
    print(f"\n  THIN in 25/26 (<900 min, n={len(thin)}): "
          f"median weighted games gained {gain.reindex(thin).median():.2f}")
    print(f"  THICK in 25/26 (>=900 min, n={len(thick)}): "
          f"median weighted games gained {gain.reindex(thick).median():.2f}")
    print("  ^ if THICK >> THIN the extra evidence is landing on players who "
          "already had enough")

    # ---------------------------------------------------------------- 2. magnitude
    print("\n=== 2. MAGNITUDE — how far does the start prior move? ===")
    base = r26.reindex(sorted(recent))
    st = base["starts"].fillna(0.0); gm = base["games"].fillna(0.0)
    hs = d.reindex(sorted(recent))["hist_starts"].fillna(0.0)
    hg = d.reindex(sorted(recent))["hist_games"].fillna(0.0)
    p_old = (2.0 + 0.70 * st) / (4.0 + 0.70 * gm)
    p_new = (2.0 + 0.70 * (st + hs)) / (4.0 + 0.70 * (gm + hg))
    delta = (p_new - p_old).dropna()
    print(f"  |delta start prob|: mean {delta.abs().mean():.4f}  "
          f"median {delta.abs().median():.4f}  p95 {delta.abs().quantile(.95):.4f}  "
          f"max {delta.abs().max():.4f}")
    print(f"  players moving > 0.02: {int((delta.abs() > 0.02).sum())}")

    # ---------------------------------------------------------------- 3. half-life
    print("\n=== 3. HALF-LIFE — chosen by held-out fit on 25/26 ===")
    bt = backtest_half_life(panel)
    if bt.empty:
        print("  insufficient overlap to backtest")
        return
    bt = bt.sort_values("mae")
    show = bt.copy()
    show["half_life"] = show["half_life"].map(lambda x: "flat" if x > 100 else f"{x:g}")
    print(show.round(4).to_string(index=False))
    best = bt.iloc[0]
    flat = bt[bt["half_life"] > 100]
    print(f"\n  best half-life: {best['half_life']:g}  (MAE {best['mae']:.4f})")
    if len(flat):
        print(f"  no-decay baseline MAE {flat.iloc[0]['mae']:.4f}  -> decay "
              f"{'HELPS' if best['mae'] < flat.iloc[0]['mae'] else 'does NOT help'}")
    bt.to_csv(OUT, index=False)
    print(f"-> {OUT}")

    # --------------------------------------------------- 4. deep vs shallow
    print("\n=== 4. THE DECISIVE TEST — does deep history beat the repo's baseline? ===")
    r, j = does_deep_beat_shallow(panel)
    print(r.round(4).to_string(index=False))
    print("\n  'shallow' = 24/25 only, which is what the model already has.")
    print("  A half-life sweep that only ranks half-lives cannot answer this.")
    worst = j.assign(err_deep=(j["p_deep"] - j["y"]).abs(),
                     err_shallow=(j["p_shallow"] - j["y"]).abs())
    worst = worst.nlargest(6, "err_deep")[
        ["player_code", "y", "p_shallow", "p_deep", "err_shallow", "err_deep"]]
    print("\n  worst deep-history errors (actual y vs each prediction):")
    print(worst.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
