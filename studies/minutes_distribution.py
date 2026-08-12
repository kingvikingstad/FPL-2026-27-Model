from __future__ import annotations
import os as _os, sys as _sys, glob as _glob
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
minutes_distribution.py — the model assumes every starter plays exactly 90 minutes
==================================================================================
bayes_model.project draws minutes from a TWO-POINT distribution:

    mins = np.where(start, 90.0, np.where(sub, 20.0, 0.0))

`start` is drawn from Beta(start_a, start_b), and those counts were fitted on
`minutes >= 60` — so "start" in the simulation really means "played 60+". That makes
`played60` correct by construction, and clean-sheet eligibility with it. The exposure is
not:

    m90 = mins / 90.0        ->  1.0 for every starter

and `m90` multiplies attacking involvement, penalty xG and DefCon counts. If a player who
clears 60 minutes averages meaningfully less than 90, EVERY starter's attacking return is
inflated by the shortfall, systematically and in the same direction. Minutes is the
project's own documented dominant lever, so a silent bias here is expensive.

This measures the real distribution from the repo's per-match panel (`minutes_played`,
keyed on player_id -> player_code) and prices the error.

Run:  python studies/minutes_distribution.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                    "minutes_distribution.csv")
POS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD"}


def load():
    frames = []
    specs = [("2025-2026", _os.path.join("By Gameweek", "GW*", "playermatchstats.csv"),
              "players.csv"),
             ("2024-2025", _os.path.join("playermatchstats", "GW*",
                                         "playermatchstats.csv"),
              _os.path.join("players", "players.csv"))]
    for season, gwglob, pfile in specs:
        files = sorted(_glob.glob(_os.path.join(config.repo(season), gwglob)))
        if not files:
            continue
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if "match_id" in d.columns:
            d = d[d["match_id"].astype(str).str.contains("-prem-", na=False)]
        p = pd.read_csv(_os.path.join(config.repo(season), pfile))
        keep = [c for c in ("player_id", "player_code", "position") if c in p.columns]
        d = d.merge(p[keep], on="player_id", how="left")
        d["season"] = season
        d["mins"] = pd.to_numeric(d["minutes_played"], errors="coerce").fillna(0.0)
        d["pos"] = d["position"].map(POS) if "position" in d.columns else np.nan
        frames.append(d[["season", "player_code", "pos", "mins"]])
    return pd.concat(frames, ignore_index=True)


def main():
    d = load()
    d = d[d["mins"] > 0]
    print(f"[study] {len(d)} player-appearances across {d.season.nunique()} seasons")

    starters = d[d["mins"] >= 60]
    subs = d[(d["mins"] > 0) & (d["mins"] < 60)]

    print("\n" + "=" * 74)
    print("1. WHAT THE MODEL ASSUMES vs WHAT HAPPENS")
    print("=" * 74)
    print(f"  {'':22s} {'model':>8s} {'actual':>9s} {'error':>9s}")
    print(f"  {'minutes | played 60+':22s} {90.0:8.1f} {starters['mins'].mean():9.2f} "
          f"{starters['mins'].mean()-90:9.2f}")
    print(f"  {'minutes | sub (<60)':22s} {20.0:8.1f} {subs['mins'].mean():9.2f} "
          f"{subs['mins'].mean()-20:9.2f}")
    print(f"\n  implied exposure m90 for a 'starter': model 1.000 vs actual "
          f"{starters['mins'].mean()/90:.3f}")
    infl = 90 / starters["mins"].mean() - 1
    print(f"  => every starter's attacking return, penalty xG and DefCon count is")
    print(f"     inflated by {infl*100:.1f}%")

    print("\n  distribution of minutes among those who cleared 60:")
    q = starters["mins"].quantile([.05, .25, .5, .75, .95]).round(1)
    print("   " + "  ".join(f"p{int(k*100)}={v:.0f}" for k, v in q.items()))
    print(f"   share playing the full 90+: {(starters['mins'] >= 90).mean():.1%}")

    print("\n" + "=" * 74)
    print("2. BY POSITION — the bias is not uniform")
    print("=" * 74)
    print(f"  {'pos':5s} {'n':>7s} {'E[min|60+]':>11s} {'m90':>7s} {'inflation':>10s} "
          f"{'full 90':>8s}")
    rows = []
    for p in ("GK", "DEF", "MID", "FWD"):
        s = starters[starters["pos"] == p]
        if len(s) < 100:
            continue
        mm = s["mins"].mean()
        rows.append({"pos": p, "n": len(s), "mean_min_60plus": mm, "m90": mm / 90,
                     "inflation_pct": (90 / mm - 1) * 100,
                     "full90_share": (s["mins"] >= 90).mean()})
        print(f"  {p:5s} {len(s):7d} {mm:11.2f} {mm/90:7.3f} "
              f"{(90/mm-1)*100:9.1f}% {(s['mins']>=90).mean():8.1%}")
    print("\n  Goalkeepers are ~exactly 90 by nature; forwards are substituted most, so")
    print("  the inflation lands hardest on the players whose attacking return matters")
    print("  most to the projection.")

    print("\n" + "=" * 74)
    print("3. THE SUB BRANCH")
    print("=" * 74)
    print(f"  model gives every sub exactly 20 minutes.")
    print(f"  actual mean {subs['mins'].mean():.1f}, median {subs['mins'].median():.0f}, "
          f"p90 {subs['mins'].quantile(.9):.0f}")
    over60 = (d[(d['mins'] > 0)]['mins'] >= 60).mean()
    print(f"  share of sub appearances reaching 60+: "
          f"{(subs['mins'] >= 60).mean():.1%} by construction 0, but note "
          f"{(d[d.mins>0]['mins'].between(60, 75)).mean():.1%} of ALL appearances land in "
          f"60-75 minutes")
    print("  A sub who comes on before the hour is currently unable to earn the 2-point")
    print("  appearance or clean-sheet eligibility the real player would.")

    print("\n" + "=" * 74)
    print("4. THE FIX, AND WHAT IT IS WORTH")
    print("=" * 74)
    for p in ("DEF", "MID", "FWD"):
        s = starters[starters["pos"] == p]
        if len(s) < 100:
            continue
        print(f"  {p}: replace mins=90 with {s['mins'].mean():.1f} "
              f"(or draw from the empirical distribution)")
    mid = starters[starters["pos"] == "MID"]
    print(f"\n  For a midfielder projected at 5.0 pts/GW with most of it from attacking")
    print(f"  returns, a {(90/mid['mins'].mean()-1)*100:.1f}% exposure inflation is worth")
    print(f"  roughly {5.0*(1-mid['mins'].mean()/90):.2f} pts/GW of over-projection, or "
          f"{5.0*(1-mid['mins'].mean()/90)*10:.1f} pts across a GW1-10 horizon.")
    print("  That is the same order as the entire GW1-3 home-advantage correction, and")
    print("  it applies to EVERY gameweek and EVERY starter.")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
