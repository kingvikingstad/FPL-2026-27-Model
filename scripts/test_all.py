import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
test_all.py — whole-model regression harness
=============================================
Runs everything that can fail and reports one table. Five layers:

  1. IMPORT      every src module imports cleanly (catches syntax and circular imports)
  2. SELFTEST    modules that ship a --selftest run it (offline, synthetic fixtures)
  3. ACCEPTANCE  the four tests/ scripts
  4. PIPELINE    build_all then every board runner, end to end
  5. STUDIES     every studies/ script runs without raising

Then two properties that no individual test covers:
  * DETERMINISM  — the board reproduces within Monte Carlo tolerance on a re-run
  * INVARIANTS   — sanity checks on the produced board (no NaN, no negative points,
                  every projected player has a club and a price, totals in range)

Run:  python scripts/test_all.py            (full)
      python scripts/test_all.py --quick    (skip pipeline and studies)
"""
import glob
import subprocess
import time

PY = _sys.executable
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

SELFTEST = ["sd_ingest", "xg_calibrate", "crosswalk", "setpiece", "fpl_history"]
ACCEPTANCE = ["test_regime", "test_defcon_env", "test_regime_panel", "validate_shrinkage"]
PIPELINE = ["build_all", "gw_board", "run_final_board", "cs_fixtures", "run_solio_ensemble"]
SKIP_STUDIES = {"sd_design"}          # pre-registration arithmetic, needs no run


# Inputs the project documents as OPTIONAL. A script that stops because one of these is
# absent has not failed — it has nothing to run on. Reporting those as failures buries the
# real ones, so they are surfaced as SKIP with the reason.
OPTIONAL_INPUTS = ("fpl-data-stats.csv",)


def run(cmd, timeout=1800):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        out = (r.stderr or "") + (r.stdout or "")
        if r.returncode != 0:
            for opt in OPTIONAL_INPUTS:
                if opt in out and "FileNotFoundError" in out:
                    return "skip", time.time() - t0, f"optional input absent: {opt}"
        return r.returncode == 0, time.time() - t0, out[-400:]
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, "TIMEOUT"


def section(name):
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")


def report(rows):
    ok = sum(1 for r in rows if r[1] is True)
    skip = sum(1 for r in rows if r[1] == "skip")
    for name, good, secs, err in rows:
        mark = {True: "ok  ", "skip": "SKIP"}.get(good, "FAIL")
        note = f"   ({err})" if good == "skip" else ""
        print(f"  {mark} {name:34s} {secs:6.1f}s{note}")
        if good is False:
            for line in err.strip().splitlines()[-4:]:
                print(f"       | {line}")
    tail = f", {skip} skipped" if skip else ""
    print(f"  -> {ok}/{len(rows) - skip} passed{tail}")
    return all(r[1] is not False for r in rows)


def main():
    quick = "--quick" in _sys.argv
    allok = True

    section("1. IMPORTS")
    rows = []
    for f in sorted(glob.glob(_os.path.join(ROOT, "src", "*.py"))):
        mod = _os.path.splitext(_os.path.basename(f))[0]
        good, secs, err = run([PY, "-c", f"import sys; sys.path.insert(0,'src'); import {mod}"], 300)
        rows.append((mod, good, secs, err))
    allok &= report(rows)

    section("2. MODULE SELFTESTS")
    rows = [(m,) + run([PY, _os.path.join("src", f"{m}.py"), "--selftest"], 900)
            for m in SELFTEST]
    allok &= report(rows)

    section("3. ACCEPTANCE TESTS")
    rows = [(t,) + run([PY, _os.path.join("tests", f"{t}.py")], 1800) for t in ACCEPTANCE]
    allok &= report(rows)

    if not quick:
        section("4. PIPELINE (end to end)")
        rows = [(s,) + run([PY, _os.path.join("scripts", f"{s}.py")], 2400) for s in PIPELINE]
        allok &= report(rows)

        section("5. STUDIES")
        rows = []
        for f in sorted(glob.glob(_os.path.join(ROOT, "studies", "*.py"))):
            name = _os.path.splitext(_os.path.basename(f))[0]
            if name in SKIP_STUDIES:
                continue
            rows.append((name,) + run([PY, _os.path.join("studies", f"{name}.py")], 2400))
        allok &= report(rows)

    section("6. BOARD INVARIANTS")
    import pandas as pd
    import numpy as np
    checks = []
    try:
        b = pd.read_csv(_os.path.join(config.OUTPUTS, "gw_board_long.csv"))
        w = pd.read_csv(_os.path.join(config.OUTPUTS, "gw_board_wide.csv"))
        checks.append(("no NaN in projections", not b["mean"].isna().any()))
        checks.append(("no negative projections", (b["mean"] >= 0).all()))
        checks.append(("every row has a club", b["team"].notna().all()))
        checks.append(("every row has a price", b["cost"].notna().all()))
        checks.append(("prices in 3.5-16.0", b["cost"].between(3.5, 16.0).all()))
        checks.append(("10 gameweeks present", sorted(b["gw"].unique()) == list(range(1, 11))))
        checks.append(("one row per player-gameweek",
                       not b.duplicated(["player", "team", "gw"]).any()))
        # `total` sums `blended`, not `mean` — gw_board mixes the external feed into GW1
        # for matched players, so ~30 of them differ. Comparing against `mean` fails for
        # exactly those, which is a wrong check rather than a wrong board.
        checks.append(("wide totals match long sums (blended)", bool(np.allclose(
            b.groupby(["player", "team"])["blended"].sum().sort_index().values,
            w.set_index(["player", "team"])["total"].sort_index().values, atol=0.05))))
        checks.append(("top scorer is plausible (40-120 pts)",
                       40 <= w["total"].max() <= 120))
        checks.append(("all four positions present",
                       set(b["pos"].unique()) == {"GK", "DEF", "MID", "FWD"}))
    except Exception as e:
        checks.append((f"could not load board: {type(e).__name__}: {e}", False))
    for name, good in checks:
        print(f"  {'ok  ' if good else 'FAIL'} {name}")
    allok &= all(g for _, g in checks)
    print(f"  -> {sum(g for _, g in checks)}/{len(checks)} passed")

    if not quick:
        section("7. DETERMINISM (re-run the board, compare)")
        import tempfile
        out = _os.path.join(tempfile.gettempdir(), "fpl_determinism")
        _os.makedirs(out, exist_ok=True)
        env = dict(_os.environ); env["FPL_OUTPUTS"] = out
        r = subprocess.run([PY, _os.path.join("scripts", "gw_board.py")],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=2400)
        if r.returncode != 0:
            print("  FAIL board re-run errored"); allok = False
        else:
            a = pd.read_csv(_os.path.join(config.OUTPUTS, "gw_board_wide.csv"))
            c = pd.read_csv(_os.path.join(out, "gw_board_wide.csv"))
            m = a.merge(c, on=["player", "team"], suffixes=("_a", "_b"))
            d = (m["total_b"] - m["total_a"]).abs()
            good = d.max() < 0.75          # Monte Carlo tolerance, S=1500
            print(f"  {'ok  ' if good else 'FAIL'} max |delta| {d.max():.3f} over "
                  f"{len(m)} players (tolerance 0.75, MC noise at S=1500)")
            print(f"       mean |delta| {d.mean():.4f}")
            allok &= good

    section("RESULT")
    print(f"  {'ALL CHECKS PASSED' if allok else 'FAILURES ABOVE'}")
    return 0 if allok else 1


if __name__ == "__main__":
    _sys.exit(main())
