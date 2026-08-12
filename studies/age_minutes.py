from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
age_minutes.py — does age add anything beyond a player's own minutes history?
=============================================================================
The cheap version of a question that was nearly answered the expensive way.

`minutes_persistence.py` showed that a player's own conditional minutes (E[min | started])
predicts next season's at r=0.65, and bayes_model now uses a shrunk version of it. Age is
an INDIRECT proxy for the same thing — a declining 34-year-old already shows up as
declining minutes. So the only question worth asking is the incremental one:

    does age predict next season's minutes AFTER conditioning on this season's minutes?

That is genuinely non-zero in principle: a 34-year-old and a 27-year-old with identical
records this season have different odds of falling off next season, and no amount of
minutes history contains that. But it is a much smaller quantity than a raw age curve.

WHY THIS RUNS ON TWO SEASONS AND NOT TWELVE
--------------------------------------------
vaastav carries `birth_date` only from 2024/25, giving one clean transition. Reconstructing
age for earlier seasons means either back-filling from the permanent `code` — which covers
only players still registered in 2024/25+, i.e. SURVIVORS, and survivorship correlates
directly with the decline being measured — or scraping an external source. FBref's reader
was tried and pulls a full browser-automation stack (selenium, seleniumbase, PyAutoGUI)
while managing one season per ~25 minutes, so it was abandoned.

One transition is thin. It is enough to answer "is there a signal here worth paying for",
which is the actual decision, and if the answer is no then the twelve-season version was
never justified.

Run:  python studies/age_minutes.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import fpl_history as fh
import late_form_carryover as lfc

MIN_APPS = 8
SEASON_T, SEASON_T1 = "2024-25", "2025-26"


def load_ages():
    frames = []
    for s in ("2024-25", "2025-26", "2026-27"):
        try:
            raw = fh._read_csv(_os.path.join(config.history(s), "players_raw.csv"))
        except Exception:
            continue
        if "birth_date" not in raw.columns:
            continue
        r = raw[["code", "birth_date"]].dropna()
        frames.append(r.rename(columns={"code": "player_code"}))
    if not frames:
        return None
    a = pd.concat(frames).drop_duplicates("player_code")
    a["dob"] = pd.to_datetime(a["birth_date"], errors="coerce")
    return a.dropna(subset=["dob"])[["player_code", "dob"]]


def main():
    ages = load_ages()
    if ages is None or ages.empty:
        print("no birth_date available — cannot run"); return
    print(f"[study] {len(ages)} players with a date of birth")

    panel = fh.load_history([SEASON_T, SEASON_T1], verbose=False)
    p = panel[panel["mins"] >= 60]
    g = p.groupby(["player_code", "season"], as_index=False).agg(
        cond_min=("mins", "mean"), n=("mins", "size"),
        pos=("pos", lambda s: s.dropna().iloc[-1] if s.notna().any() else None))
    a = g[(g.season == SEASON_T) & (g.n >= MIN_APPS)].rename(
        columns={"cond_min": "x", "n": "nx"})
    b = g[(g.season == SEASON_T1) & (g.n >= MIN_APPS)].rename(
        columns={"cond_min": "y"})
    P = a.merge(b[["player_code", "y"]], on="player_code").merge(ages, on="player_code")
    # age at the START of season t+1
    P["age"] = (pd.Timestamp("2025-08-01") - P["dob"]).dt.days / 365.25
    print(f"        {len(P)} players with >= {MIN_APPS} 60+ appearances in BOTH seasons "
          f"and a DOB")
    print(f"        age range {P.age.min():.1f} to {P.age.max():.1f}, "
          f"median {P.age.median():.1f}")

    print("\n" + "=" * 72)
    print("1. RAW — does age relate to conditional minutes at all?")
    print("=" * 72)
    bands = pd.cut(P["age"], [0, 23, 26, 29, 32, 99],
                   labels=["<23", "23-26", "26-29", "29-32", "32+"])
    print(P.groupby(bands).agg(n=("y", "size"), age=("age", "mean"),
                               mins_t=("x", "mean"), mins_t1=("y", "mean"),
                               change=("y", "mean")).round(2).to_string())
    d = P.assign(chg=P["y"] - P["x"]).groupby(bands)["chg"].agg(["size", "mean"])
    print("\n  season-on-season CHANGE in conditional minutes by age band:")
    print(d.round(2).to_string())

    print("\n" + "=" * 72)
    print("2. THE TEST — does age add anything beyond this season's minutes?")
    print("=" * 72)
    b1, s1, r1 = lfc.ols([P["x"]], P["y"].values)
    b2, s2, r2 = lfc.ols([P["x"], P["age"]], P["y"].values)
    print(f"  y ~ minutes_t          : minutes {b1[1]:+.4f} (se {s1[1]:.4f})   "
          f"r2={r1:.4f}")
    print(f"  y ~ minutes_t + age    : minutes {b2[1]:+.4f} (se {s2[1]:.4f}), "
          f"age {b2[2]:+.4f} (se {s2[2]:.4f}, t {b2[2]/s2[2]:+.2f})   r2={r2:.4f}")
    print(f"  incremental r2 from age: {r2-r1:+.5f}")

    rng = np.random.default_rng(0)
    bs = []
    for _ in range(4000):
        idx = rng.integers(0, len(P), len(P))
        s = P.iloc[idx]
        bb, _, _ = lfc.ols([s["x"], s["age"]], s["y"].values)
        bs.append(bb[2])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  age coefficient 95% CI : ({lo:+.4f}, {hi:+.4f})"
          f"{'  *' if not (lo <= 0 <= hi) else '  <- overlaps zero'}")
    print(f"\n  practical size: {abs(b2[2]):.3f} min per year of age means a 34-year-old")
    print(f"  and a 26-year-old with identical records differ by "
          f"{abs(b2[2])*8:.1f} minutes next season.")

    # a quadratic, in case the effect is only at the tail
    P["age2"] = (P["age"] - 27) ** 2
    b3, s3, r3 = lfc.ols([P["x"], P["age"], P["age2"]], P["y"].values)
    print(f"\n  with a quadratic term  : age {b3[2]:+.4f} (se {s3[2]:.4f}), "
          f"age^2 {b3[3]:+.4f} (se {s3[3]:.4f})   r2={r3:.4f}")
    print(f"  incremental r2 over minutes alone: {r3-r1:+.5f}")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    worth = (not (lo <= 0 <= hi)) and (r2 - r1) > 0.005
    print(f"  age adds {'SOMETHING' if worth else 'essentially nothing'} beyond the "
          f"player's own minutes history.")
    print(f"  {'Consider' if worth else 'Do NOT'} pay for a 12-season age scrape.")


if __name__ == "__main__":
    main()
