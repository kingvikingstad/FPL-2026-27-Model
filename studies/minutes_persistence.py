from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
minutes_persistence.py — should E[minutes | started] be per-PLAYER, not per-position?
=====================================================================================
`minutes_distribution.py` found that starters average 85.3 minutes rather than 90, and
bayes_model now substitutes a POSITIONAL constant (GK 89.9 / DEF 87.5 / MID 83.1 /
FWD 81.5). That is better than 90 for everyone, but it still says a 90-minute centre-half
and a forward who is always hooked on 65 have the same exposure once they start.

This asks whether the player's own history does better. Two things have to hold:

  1. PERSISTENCE — a player's conditional minutes must carry from season to season. If it
     does not, the positional constant is already the right answer and nothing more is
     available.
  2. BEATING THE BASELINE — a shrunk player estimate must predict next season's
     conditional minutes better than the positional constant, out of sample. Persistence
     alone is not enough: the quantity could persist and still be predicted just as well
     by position, if position is what drives it.

Shrinkage weight comes from the measured reliability rather than being picked — the same
empirical-Bayes logic the rest of the project uses.

Data: vaastav player-gameweek minutes, 2016/17-2025/26, keyed on player_code. No scraping.

Run:  python studies/minutes_persistence.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import fpl_history as fh

MIN_APPS = 10          # appearances of 60+ needed for a season mean to mean anything
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                    "minutes_persistence.csv")
# what bayes_model currently uses
POSITIONAL = {"GK": 89.9, "DEF": 87.5, "MID": 83.1, "FWD": 81.5}


def build():
    panel = fh.load_history(verbose=False)
    p = panel[panel["mins"] >= 60].copy()
    g = p.groupby(["player_code", "season"], as_index=False).agg(
        mean_min=("mins", "mean"), n_apps=("mins", "size"),
        pos=("pos", lambda s: s.dropna().iloc[-1] if s.notna().any() else None))
    # split-half within season, for reliability
    rng = np.random.default_rng(0)
    p = p.assign(half=rng.integers(0, 2, len(p)))
    h = (p.groupby(["player_code", "season", "half"])["mins"].mean()
           .unstack().dropna())
    h.columns = ["h1", "h2"]
    return g[g["n_apps"] >= MIN_APPS], h.reset_index()


def main():
    G, H = build()
    print(f"[study] {len(G)} player-seasons with >= {MIN_APPS} appearances of 60+ minutes")
    print(f"        overall mean conditional minutes {G.mean_min.mean():.2f}, "
          f"sd across players {G.mean_min.std():.2f}")
    print("\n  spread by position (is there room for a player-specific number?):")
    print(G.groupby("pos")["mean_min"].agg(["count", "mean", "std"]).round(2).to_string())

    # ------------------------------------------------------- 1. reliability
    HH = H.merge(G[["player_code", "season"]], on=["player_code", "season"])
    r = float(np.corrcoef(HH["h1"], HH["h2"])[0, 1])
    rel = 2 * r / (1 + r)
    print(f"\n  split-half reliability of a season's conditional minutes: {rel:.3f} "
          f"(half-correlation {r:.3f}, n={len(HH)})")

    # ------------------------------------------------------- 2. persistence
    order = list(fh.SEASONS)
    idx = {s: i for i, s in enumerate(order)}
    G["si"] = G["season"].map(idx)
    a = G.rename(columns={"mean_min": "x", "n_apps": "nx"})
    b = G.rename(columns={"mean_min": "y", "n_apps": "ny"})
    b["si"] = b["si"] - 1
    P = a.merge(b[["player_code", "si", "y", "ny"]], on=["player_code", "si"])
    print(f"\n" + "=" * 74)
    print("1. PERSISTENCE — does conditional minutes carry season to season?")
    print("=" * 74)
    slope = np.polyfit(P["x"], P["y"], 1)[0]
    rr = float(np.corrcoef(P["x"], P["y"])[0, 1])
    print(f"  n={len(P)} consecutive player-season pairs")
    print(f"  r = {rr:.3f}   slope = {slope:.3f}   disattenuated r = {rr/rel:.3f}")
    print(f"  sd of the season mean {P.x.std():.2f} min; a 1-sd better player is "
          f"{slope*P.x.std():+.2f} min next season")

    # ------------------------------------------------------- 3. horse race
    print("\n" + "=" * 74)
    print("2. HORSE RACE — positional constant vs shrunk player history, out of sample")
    print("=" * 74)
    P["pos_pred"] = P["pos"].map(POSITIONAL)
    P = P.dropna(subset=["pos_pred"])
    # empirical-Bayes shrink toward the positional mean, weight from reliability
    k = (1 - rel) / max(rel, 1e-6) * P["nx"].median()
    P["w"] = P["nx"] / (P["nx"] + k)
    P["player_pred"] = P["w"] * P["x"] + (1 - P["w"]) * P["pos_pred"]
    P["flat90"] = 90.0
    for lab, col in (("flat 90 (pre-fix)", "flat90"),
                     ("positional constant (current)", "pos_pred"),
                     ("shrunk player history", "player_pred")):
        e = P["y"] - P[col]
        print(f"  {lab:31s} MAE {np.mean(np.abs(e)):6.3f}  RMSE "
              f"{np.sqrt(np.mean(e**2)):6.3f}  bias {e.mean():+6.3f}")
    print(f"\n  shrinkage: k={k:.1f} appearances, median weight on the player's own "
          f"history {P.w.median():.2f}")

    gain = (np.mean(np.abs(P["y"] - P["pos_pred"]))
            - np.mean(np.abs(P["y"] - P["player_pred"])))
    print(f"  player history beats the positional constant by {gain:.3f} min MAE "
          f"({gain/np.mean(np.abs(P['y']-P['pos_pred']))*100:.1f}%)")

    # cluster the comparison by player so repeated seasons do not inflate certainty
    rng = np.random.default_rng(1)
    codes = P["player_code"].unique()
    bs = []
    for _ in range(2000):
        pick = rng.choice(codes, len(codes), replace=True)
        s = P[P["player_code"].isin(pick)]
        bs.append(np.mean(np.abs(s["y"] - s["pos_pred"]))
                  - np.mean(np.abs(s["y"] - s["player_pred"])))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  clustered 95% CI on that gain: ({lo:+.3f}, {hi:+.3f})"
          f"{'  *' if not (lo <= 0 <= hi) else '  (not significant)'}")

    # ------------------------------------------------------- 4. who moves most
    print("\n" + "=" * 74)
    print("3. WHERE IT MATTERS — the tails")
    print("=" * 74)
    P["delta_vs_pos"] = P["player_pred"] - P["pos_pred"]
    ext = P.reindex(P["delta_vs_pos"].abs().sort_values(ascending=False).index).head(8)
    print(f"  {'pos':4s} {'own hist':>9s} {'positional':>11s} {'shrunk':>8s} "
          f"{'actual next':>12s}")
    for _, r_ in ext.iterrows():
        print(f"  {str(r_['pos']):4s} {r_['x']:9.1f} {r_['pos_pred']:11.1f} "
              f"{r_['player_pred']:8.1f} {r_['y']:12.1f}")
    n_big = int((P["delta_vs_pos"].abs() >= 5).sum())
    print(f"\n  {n_big} of {len(P)} player-seasons ({n_big/len(P):.0%}) move by >= 5 "
          f"minutes against the positional constant")

    P.to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
