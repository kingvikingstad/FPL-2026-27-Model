import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
ab_inseason_minutes.py — does the minutes channel actually score better?
=========================================================================
Two calibrations passed their pre-registered rules this session:

  studies/minutes_per_start.py    exp_minutes should follow realised minutes-per-start
                                  (5.5-18.9% RMSE gain on the parameter)
  studies/start_prior_strength.py the Beta start prior is too strong for its evidence
                                  (~11% Brier gain; only the ratio w/kappa is identified)

Both validate a PARAMETER. Neither shows the board scores better on POINTS, which is the
only thing that matters, and points depend on minutes through the 60-minute threshold and
through exposure scaling — neither linear in the parameter. This script is the missing
step: run the board with the channels off and on, same seed, and score both against a
gameweek neither arm was allowed to see.

HOW THE OUT-OF-SAMPLE SPLIT WORKS
----------------------------------
`INSEASON_UPTO=k` truncates every in-season channel to the first k gameweeks. Both arms
are therefore built from GW1..k only and projected forward to GW k+1, which is then
scored against what actually happened. Without that truncation the board would be scored
on gameweeks whose results are already inside its priors, and the "improvement" would be
memorisation.

WHAT IT CANNOT DO YET, STATED PLAINLY
--------------------------------------
Two gameweeks have been played, so k=1 and a single scored gameweek is all the season
affords. One gameweek is a noisy target — a hatful of goals or a red card moves the
metrics more than the channel does — so a result here is DIRECTIONAL and must not be read
as validation. The pre-registered bar below is deliberately set where a single gameweek
can clear it only if the effect is large; anything smaller is reported as inconclusive,
not as a win. Re-run it after every gameweek: `--upto` walks forward as the season does,
and the verdict only becomes meaningful once several gameweeks are in.

DECISION RULE, FIXED BEFORE THE FIRST RUN
------------------------------------------
Adopt a channel as ON-by-default only when, over at least MIN_WEEKS scored gameweeks,
mean absolute error improves and the sign is consistent in a majority of them. A single
week can never satisfy it. Until then the flags stay off and this file is the record.

Run:  python scripts/ab_inseason_minutes.py                 # k=1, scores GW2
      python scripts/ab_inseason_minutes.py --upto 5        # once GW6 exists
      python scripts/ab_inseason_minutes.py --arm exp       # one channel only
Out:  outputs/ab_inseason_minutes.csv
"""
import argparse, glob, subprocess, tempfile, shutil
import numpy as np
import pandas as pd

MIN_WEEKS = 3          # pre-registered: fewer than this cannot decide anything
ARMS = {
    "base": {},
    "exp":   {"INSEASON_EXP_MINUTES": "on"},
    "kappa": {"INSEASON_KAPPA": "4"},
    "both":  {"INSEASON_EXP_MINUTES": "on", "INSEASON_KAPPA": "4"},
}
OUT = _os.path.join(config.OUTPUTS, "ab_inseason_minutes.csv")
PY = _sys.executable


def actuals(gw):
    """Realised FPL points and minutes for one gameweek, keyed on player_code."""
    base = config.repo("2026-2027")
    d = _os.path.join(base, "By Gameweek", f"GW{gw}")
    f, mf = _os.path.join(d, "player_gameweek_stats.csv"), _os.path.join(d, "matches.csv")
    if not (_os.path.exists(f) and _os.path.exists(mf)):
        return None
    mm = pd.read_csv(mf)
    if not len(mm) or not (mm.get("finished") == True).any():        # noqa: E712
        return None
    g = pd.read_csv(f)
    roster = pd.read_csv(_os.path.join(base, "players.csv"))
    g = g.merge(roster[["player_id", "player_code", "web_name"]],
                left_on="id", right_on="player_id", how="left")
    keep = [c for c in ("player_code", "web_name", "total_points", "minutes", "starts")
            if c in g.columns]
    return g[keep].dropna(subset=["player_code"]).drop_duplicates("player_code")


def run_arm(name, env_extra, upto, gw_hi, verbose=True):
    """One board run into a scratch outputs dir. Same seed, same everything else."""
    out = tempfile.mkdtemp(prefix=f"ab_{name}_")
    env = dict(_os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "FPL_OUTPUTS": out,
                "GW_HI": str(gw_hi), "INSEASON_UPTO": str(upto),
                # hold everything that could differ between arms fixed
                "LIVE_FPL": "off", "PRED_XI": "off", "SOLIO": "off"})
    env.update(env_extra)
    if verbose:
        print(f"  [{name}] running board (INSEASON_UPTO={upto}, GW_HI={gw_hi})"
              + (f"  {env_extra}" if env_extra else "  [baseline]"))
    r = subprocess.run([PY, _os.path.join("scripts", "gw_board.py")],
                       cwd=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                       env=env, capture_output=True, text=True, timeout=3600,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        shutil.rmtree(out, ignore_errors=True)
        raise SystemExit(f"arm {name} failed")
    board = pd.read_csv(_os.path.join(out, "gw_board_long.csv"))
    shutil.rmtree(out, ignore_errors=True)
    return board


def score(board, act, gw):
    """MAE / RMSE / Spearman of projected against realised points for one gameweek.

    Scored over players who ACTUALLY FEATURED. Including the 300-odd who did not turn out
    would let both arms bank the same easy zeros and bury the difference in a sea of
    correct predictions of nothing — the channel is about how long a player lasts, so the
    comparison has to be on players who played.
    """
    b = board[board["gw"] == gw][["player_code", "player", "mean"]].dropna(subset=["player_code"])
    m = b.merge(act, on="player_code", how="inner")
    m = m[m["minutes"] > 0]
    if not len(m):
        return None
    e = m["mean"] - m["total_points"]
    return {"n": len(m), "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean()),
            "spearman": float(m["mean"].corr(m["total_points"], method="spearman"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upto", type=int, default=None,
                    help="build from GW1..upto and score GW upto+1 (default: all but the last played)")
    ap.add_argument("--arm", action="append", choices=sorted(ARMS),
                    help="which arms to run (default: all)")
    a = ap.parse_args()

    import inseason as ins
    played = ins.played(upto_gw=38, require_xg=False, verbose=False)
    done = sorted(int(g) for g, n in played.groupby("gameweek").size().items() if n >= 10)
    if len(done) < 2:
        print(f"Only {len(done)} complete gameweek(s). Nothing to score out of sample yet.")
        return 0
    upto = a.upto if a.upto is not None else done[-2]
    target = upto + 1
    act = actuals(target)
    if act is None:
        print(f"GW{target} is not complete — cannot score it.")
        return 0

    print("=" * 78)
    print(f"A/B — in-season minutes channels, built on GW1-{upto}, scored on GW{target}")
    print("=" * 78)
    print(f"complete gameweeks: {done}")
    n_weeks = 1
    if n_weeks < MIN_WEEKS:
        print(f"\n!! {n_weeks} scored gameweek against a pre-registered minimum of "
              f"{MIN_WEEKS}.")
        print(f"!! This run is DIRECTIONAL ONLY and cannot adopt anything. Re-run each week.")

    arms = a.arm or ["base", "exp", "kappa", "both"]
    rows = []
    for name in arms:
        board = run_arm(name, ARMS[name], upto, target)
        sc = score(board, act, target)
        if sc is None:
            print(f"  [{name}] nothing to score"); continue
        sc.update({"arm": name, "upto": upto, "target_gw": target})
        rows.append(sc)

    if not rows:
        print("no arms scored"); return 0
    df = pd.DataFrame(rows).set_index("arm")
    base = df.loc["base"] if "base" in df.index else None
    print("\n" + "-" * 78)
    print(f"{'arm':>7} {'n':>5} {'MAE':>8} {'vs base':>9} {'RMSE':>8} {'bias':>8} {'rho':>7}")
    print("-" * 78)
    for name in df.index:
        r = df.loc[name]
        d = "" if base is None or name == "base" else f"{(base['mae']-r['mae'])/base['mae']:+8.1%}"
        print(f"{name:>7} {int(r['n']):>5} {r['mae']:>8.4f} {d:>9} {r['rmse']:>8.4f} "
              f"{r['bias']:>+8.4f} {r['spearman']:>7.3f}")

    df.reset_index().to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("\n" + "=" * 78)
    print("VERDICT: INCONCLUSIVE BY CONSTRUCTION — one scored gameweek against a")
    print(f"         pre-registered minimum of {MIN_WEEKS}. A gameweek's noise (one red card,")
    print("         one hatful) moves MAE more than either channel does. The flags stay")
    print("         OFF. Re-run after each gameweek; the sign becoming stable across")
    print("         several weeks is the thing to watch, not the size of any single one.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
