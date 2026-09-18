import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
ab_inseason_starts.py — the board A/B for the start-prior cap (PROJECT_KNOWLEDGE §6.9b)
========================================================================================
§6.9(a) is done (`studies/start_prior_production.py`, 17 Sep 2026): on a replica of the
INSTALLED prior the cap beats the weight at every cutoff, 3/3 folds, and kappa = 4 is
pinned. That is a claim about a PARAMETER measured in isolation. This script is (b): does
the board still gain once the update passes through availability, the XI constraint and
the scoring composition — scored against gameweeks neither arm was allowed to see.

WHY NOT `ab_inseason_minutes.py`  (audited 2026-09-17, kept as the record of its arms)
---------------------------------------------------------------------------------------
That harness cannot answer this question, for three reasons, each of which biases it:

  1. IT SCORES ONLY PLAYERS WHO FEATURED (`m[m["minutes"] > 0]`). The cap's whole effect
     is on players the board expects to start who then do not play — Dubravka at 0.83
     having started none of two. Conditioning the sample on the outcome removes exactly
     the cases under test, and it is selection on a collider: the arms differ in who they
     predict will play, so they are scored on different, arm-dependent samples.
  2. IT SCORES A MEAN FORECAST WITH MAE. MAE is minimised by the median, and FPL points
     are heavily right-skewed (median 2, mean ~2.7), so it rewards shading projections
     DOWN. A proper scoring rule for a mean is squared error.
  3. IT READS TODAY'S OWNERSHIP AND INJURY FLAGS. `INSEASON_UPTO` truncates the in-season
     channels but `playerstats.csv` is a per-gameweek panel and `core_insights.load`
     keeps the LATEST row per player, so a board "built on GW1" still saw GW4 ownership
     and status. Both arms see it, so it is not a straight win for either — it attenuates
     the difference, because the ownership shrink is already doing part of the cap's job
     with information from the future.

Fixed here: every listed player at a club that played is scored, the endpoint is Brier on
the probability the channel actually moves, and `playerstats.csv` is truncated to
gw <= upto in a scratch copy of the feed so the frame is as-of the deadline.

WHAT IS STILL NOT AS-OF, STATED PLAINLY: `players.csv` (club membership) and the team-odds
snapshot are current, not historical. Both arms share them and neither is a minutes
channel, so they cannot manufacture a difference between arms — but a board reconstructed
here is not bit-identical to one that was actually run at that deadline.

POWER, COMPUTED BEFORE THE RULE WAS SET (2026-09-17)
-----------------------------------------------------
Replica of the installed prior on 23/24-25/26, kappa=4 against the current corner, scoring
the NEXT match's start at each cutoff k = 1..8 (studies/start_prior_production.py supplies
the prior; the sweep is in the session scratch, not a repo artefact):

    per-gameweek Brier gain   mean 21.4%, sd 3.3%, positive in 24 of 24 season-gameweeks
    pooled over the first 3 GWs of each season   23/24 25.8%   24/25 20.6%   25/26 19.7%

The effect is large and its sign never flipped in any single week of three seasons. So the
binding constraint is NOT sample size on the start endpoint; it is whether the downstream
layers (availability overrides, the XI constraint's renormalisation to eleven) absorb it.
That is what this A/B measures and the simulation cannot.

POINTS ARE A DIFFERENT MATTER. A per-player-gameweek squared-error difference has sd ~1.1
against an expected mean effect of ~0.04 if the cap is fully right [DERIVED]; at ~550
players that needs roughly 5 gameweeks to detect at all, and ~20 if the cap captures half
of what it should. Points are therefore a GUARDRAIL here, not the endpoint: the design is
superiority on starts plus non-inferiority on points.

PRE-REGISTERED, FIXED BEFORE THE FIRST RUN — 2026-09-17
--------------------------------------------------------
Out of sample by construction: arm built with INSEASON_UPTO = t-1, scored on gameweek t.
Scored population: every player in the frame whose club played in gameweek t. No filter on
whether he appeared — that is the quantity under test.

PRIMARY   Brier on the realised native START indicator, arm `kappa` against `base`.
GUARDRAIL Pooled squared error of projected `mean` against realised points, same rows.
ADOPT `INSEASON_KAPPA=4` as ON by default only if, over at least MIN_WEEKS = 3 scored
gameweeks:  (1) pooled Brier improves by >= 5% (a quarter of the simulated effect, the
slack being what the downstream layers may absorb); AND (2) the sign is positive in EVERY
scored gameweek; AND (3) pooled points squared error is no worse than 2% (the guardrail).
Any of the three failing -> NOT adopted, and the pattern is recorded: a start gain that
does not survive to points would locate the loss in the composition, which is a finding
rather than a null.

THE OTHER ARMS CANNOT ADOPT ANYTHING HERE, and that is declared in advance rather than
decided when their numbers arrive:
  `kappa_lam`  INSEASON_LAM=0.75 on top of kappa=4. Its board effect is a tenth of the
               cap's (mean |delta| 0.73 season points against 7.75), no power calculation
               has been done for it, and §6.9 requires it to be landed jointly. Reported.
  `exp`        INSEASON_EXP_MINUTES=on. Its target quantity is minutes-per-start, which
               this endpoint does not score at all. Reported on points only.

CARRIED OVER FROM §6.9(a), AND IT LIMITS WHAT AN ADOPTION MEANS: the installed prior's
denominator is appearances, not matches, so its mean is biased up (0.548 against a
realised 0.424) and part of what kappa=4 buys is correcting that bias. Adopting kappa here
adopts the PACKAGE as installed. It is not evidence that prior strength alone was the
fault, and it would need re-testing if the denominator is ever fixed.

Run:  python scripts/ab_inseason_starts.py                 # every scorable gameweek
      python scripts/ab_inseason_starts.py --target 3      # one gameweek
      python scripts/ab_inseason_starts.py --arm base --arm kappa
      python scripts/ab_inseason_starts.py --smoke         # wiring only, prints no score
Out:  outputs/ab_inseason_starts.csv   (one row per arm per gameweek, accumulated)
"""
import argparse, shutil, subprocess, tempfile
import numpy as np
import pandas as pd

MIN_WEEKS = 3
MIN_BRIER_GAIN = 0.05
MAX_POINTS_LOSS = 0.02
ARMS = {
    "base":      {},
    "kappa":     {"INSEASON_KAPPA": "4"},
    "kappa_lam": {"INSEASON_KAPPA": "4", "INSEASON_LAM": "0.75"},
    "exp":       {"INSEASON_EXP_MINUTES": "on"},
}
ADOPTING_ARM = "kappa"
OUT = _os.path.join(config.OUTPUTS, "ab_inseason_starts.csv")
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
PY = _sys.executable


def league_matches():
    """Finished LEAGUE matches, names resolved. `matches.csv` in the feed also carries cup
    ties (an EFL Cup row sits in GW2 with a blank `home_team`), and its team columns are
    ids, so `inseason.played` — which drops both problems — is the only honest reader."""
    import inseason as ins
    return ins.played(upto_gw=38, require_xg=False, verbose=False)


def complete_gameweeks(m=None):
    """Gameweeks with a full league round finished. A part-played week would score an arm
    on a sample whose composition depends on which fixtures happen to be done."""
    m = league_matches() if m is None else m
    if not len(m):
        return []
    return sorted(int(gw) for gw, g in m.groupby("gameweek") if len(g) >= 10)


def actuals(gw):
    """Realised starts and points for one gameweek, plus the clubs that played."""
    base = config.repo("2026-2027")
    d = _os.path.join(base, "By Gameweek", f"GW{gw}")
    g = pd.read_csv(_os.path.join(d, "player_gameweek_stats.csv"))
    roster = pd.read_csv(_os.path.join(base, "players.csv"))
    g = g.merge(roster[["player_id", "player_code"]], left_on="id", right_on="player_id",
                how="left").dropna(subset=["player_code"]).drop_duplicates("player_code")
    if "starts" in g.columns:
        g["started"] = pd.to_numeric(g["starts"], errors="coerce").fillna(0) > 0
        src = "native starts"
    else:
        g["started"] = pd.to_numeric(g["minutes"], errors="coerce").fillna(0) >= 60
        src = "minutes >= 60 (no native column)"
    g["player_code"] = g["player_code"].astype(int)
    return g[["player_code", "started", "minutes", "total_points"]], src


def played_clubs(m, gw):
    g = m[m["gameweek"] == gw]
    return set(pd.concat([g["home"], g["away"]]).dropna().unique())


def asof_feed(upto, keep):
    """Scratch copy of the feed with playerstats.csv truncated to gw <= upto."""
    dst = tempfile.mkdtemp(prefix=f"asof_gw{upto}_", dir=keep)
    src = config.REPO                      # the feed root; FPL_DATA resolves to it
    shutil.copytree(src, dst, dirs_exist_ok=True)
    f = _os.path.join(dst, "2026-2027", "playerstats.csv")
    s = pd.read_csv(f)
    if "gw" in s.columns:
        n0 = len(s)
        s = s[pd.to_numeric(s["gw"], errors="coerce") <= upto]
        print(f"  [as-of] playerstats.csv truncated to gw <= {upto}: {n0} -> {len(s)} rows")
        s.to_csv(f, index=False)
    return dst


def run_arm(name, env_extra, upto, target, data_dir):
    out = tempfile.mkdtemp(prefix=f"ab_{name}_")
    frame = _os.path.join(out, "frame.csv")
    env = dict(_os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "FPL_OUTPUTS": out, "FPL_DATA": data_dir,
                "GW_HI": str(target), "INSEASON_UPTO": str(upto), "DUMP_FRAME": frame,
                "LIVE_FPL": "off", "PRED_XI": "off", "SOLIO": "off", "DUMP_DRAWS": "none"})
    env.update(env_extra)
    print(f"  [{name}] board INSEASON_UPTO={upto} GW_HI={target}"
          + (f"  {env_extra}" if env_extra else "  [baseline]"))
    r = subprocess.run([PY, _os.path.join("scripts", "gw_board.py")], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=5400,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-2000:])
        shutil.rmtree(out, ignore_errors=True)
        raise SystemExit(f"arm {name} failed")
    fr = pd.read_csv(frame)
    board = pd.read_csv(_os.path.join(out, "gw_board_long.csv"))
    shutil.rmtree(out, ignore_errors=True)
    return fr[fr["gw"] == target], board[board["gw"] == target]


def score(frame, board, act, clubs, gw):
    f = frame.dropna(subset=["player_code", "p_start"]).copy()
    f["player_code"] = f["player_code"].astype(int)
    if clubs:
        f = f[f["team"].isin(clubs)]
    m = f.merge(act, on="player_code", how="inner").merge(
        board[["player_code", "mean"]].dropna(subset=["player_code"]), on="player_code",
        how="left")
    y = m["started"].astype(float).to_numpy()
    p = m["p_start"].to_numpy()
    e = (m["mean"] - m["total_points"]).to_numpy()
    return {"gw": gw, "n": len(m), "start_rate": float(y.mean()),
            "p_mean": float(p.mean()), "brier": float(((p - y) ** 2).mean()),
            "logloss": float(-(y * np.log(np.clip(p, 1e-6, 1)) +
                               (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))).mean()),
            "pts_mse": float(np.nanmean(e ** 2)), "pts_bias": float(np.nanmean(e)),
            "n_expected_not_played": int(((p > 0.5) & (y == 0)).sum())}


def verdict(df):
    """The pre-registered rule, applied to every gameweek scored so far."""
    if ADOPTING_ARM not in set(df["arm"]):
        return "no `kappa` arm scored"
    b = df[df["arm"] == "base"].set_index("gw")
    k = df[df["arm"] == ADOPTING_ARM].set_index("gw")
    gws = sorted(set(b.index) & set(k.index))
    if not gws:
        return "no gameweek has both arms"
    b, k = b.loc[gws], k.loc[gws]
    pooled = lambda x, y: float((x * y["n"]).sum() / y["n"].sum())
    gain = (pooled(b["brier"], b) - pooled(k["brier"], k)) / pooled(b["brier"], b)
    pts = (pooled(b["pts_mse"], b) - pooled(k["pts_mse"], k)) / pooled(b["pts_mse"], b)
    weeks_pos = int((k["brier"].to_numpy() < b["brier"].to_numpy()).sum())
    print(f"\n  gameweeks scored      {gws}")
    print(f"  pooled Brier gain     {gain:+.1%}   (rule: >= {MIN_BRIER_GAIN:.0%})")
    print(f"  weeks improved        {weeks_pos}/{len(gws)}   (rule: every week)")
    print(f"  points MSE change     {pts:+.1%}   (guardrail: >= {-MAX_POINTS_LOSS:.0%})")
    if len(gws) < MIN_WEEKS:
        return (f"INTERIM — {len(gws)} of {MIN_WEEKS} required gameweeks. Nothing is "
                f"adopted and the numbers above are not a verdict. Re-run as weeks land.")
    if gain >= MIN_BRIER_GAIN and weeks_pos == len(gws) and pts >= -MAX_POINTS_LOSS:
        return ("ADOPT INSEASON_KAPPA=4 by default — all three pre-registered conditions "
                "met. Note the §6.9(a) caveat: this adopts the package, not a claim that "
                "prior strength alone was the fault.")
    if gain >= MIN_BRIER_GAIN and pts < -MAX_POINTS_LOSS:
        return ("NOT ADOPTED, and the pattern is the finding: starts improve while points "
                "get worse, so the loss is in the composition below the start prior, not "
                "in the prior. Investigate before re-testing.")
    return ("NOT ADOPTED — the pre-registered rule is not met. The flag stays off and "
            "this run is the record of why.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, action="append",
                    help="gameweek(s) to score (default: every complete one from GW2)")
    ap.add_argument("--arm", action="append", choices=sorted(ARMS))
    ap.add_argument("--smoke", action="store_true",
                    help="run the wiring on one gameweek and print row counts only")
    a = ap.parse_args()

    M = league_matches()
    done = complete_gameweeks(M)
    targets = a.target or [g for g in done if g >= 2]
    targets = [t for t in targets if t in done]
    print("=" * 78)
    print("A/B — in-season start prior on the board (PROJECT_KNOWLEDGE §6.9b)")
    print("=" * 78)
    print(f"complete gameweeks: {done}   scoring: {targets or 'nothing'}")
    if not targets:
        print("No complete gameweek after GW1 — nothing to score out of sample yet.")
        return 0
    arms = a.arm or list(ARMS)
    rows = []
    keep = tempfile.mkdtemp(prefix="ab_feed_")
    try:
        for t in targets:
            act, src = actuals(t)
            clubs = played_clubs(M, t)
            print(f"\nGW{t}: {len(act)} players in the result feed, start from {src}, "
                  f"{len(clubs)} clubs played")
            data_dir = asof_feed(t - 1, keep)
            for name in arms:
                frame, board = run_arm(name, ARMS[name], t - 1, t, data_dir)
                sc = score(frame, board, act, clubs, t)
                sc.update({"arm": name, "upto": t - 1})
                if a.smoke:
                    print(f"  [{name}] scored {sc['n']} players — smoke run, "
                          f"no metrics printed")
                    continue
                rows.append(sc)
    finally:
        shutil.rmtree(keep, ignore_errors=True)
    if a.smoke:
        print("\nSMOKE OK — wiring runs end to end. No result was computed or recorded.")
        return 0

    df = pd.DataFrame(rows)
    if _os.path.exists(OUT):
        old = pd.read_csv(OUT)
        df = pd.concat([old[~old.set_index(["arm", "gw"]).index.isin(
            df.set_index(["arm", "gw"]).index)], df], ignore_index=True)
    df = df.sort_values(["gw", "arm"])
    df.to_csv(OUT, index=False)

    print("\n" + "-" * 78)
    print(f"{'gw':>3} {'arm':>10} {'n':>5} {'P(start)':>9} {'realised':>9} {'Brier':>8} "
          f"{'vs base':>8} {'logloss':>8} {'pts MSE':>8} {'exp-no-show':>12}")
    print("-" * 78)
    for gw, g in df.groupby("gw"):
        base = g[g["arm"] == "base"]
        for _, r in g.iterrows():
            d = "" if base.empty or r["arm"] == "base" else \
                f"{(base['brier'].iloc[0] - r['brier']) / base['brier'].iloc[0]:+7.1%}"
            print(f"{int(r['gw']):>3} {r['arm']:>10} {int(r['n']):>5} {r['p_mean']:>9.3f} "
                  f"{r['start_rate']:>9.3f} {r['brier']:>8.4f} {d:>8} {r['logloss']:>8.4f} "
                  f"{r['pts_mse']:>8.3f} {int(r['n_expected_not_played']):>12}")
    print(f"\nwrote {OUT}")
    print("\n" + "=" * 78)
    print(verdict(df))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
