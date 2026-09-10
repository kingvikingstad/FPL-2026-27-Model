import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
test_all.py — whole-model regression harness
=============================================
Runs everything that can fail and reports one table. Five layers:

  1. IMPORT      every src module imports cleanly (catches syntax and circular imports)
  2. SELFTEST    every src/ and scripts/ file shipping a --selftest runs it (offline)
  3. ACCEPTANCE  every tests/ script (discovered)
  4. PIPELINE    build_all then every board runner, end to end
  5. STUDIES     every studies/ script runs without raising

Then two properties that no individual test covers:
  * DETERMINISM  — the board reproduces within Monte Carlo tolerance on a re-run
  * INVARIANTS   — sanity checks on the produced board (no NaN, no negative points,
                  every projected player has a club and a price, totals in range)

Run:  python scripts/test_all.py            (full)
      python scripts/test_all.py --quick    (skip pipeline and studies)
"""
import ast
import glob
import subprocess
import time

PY = _sys.executable
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# Modules that ship a `--selftest` are DISCOVERED, not listed. A hardcoded list goes
# stale the moment a module is added, and it had: 5 of the 18 modules carrying a selftest
# were named here, so solver, player_value, squad_tracker, injury_impact, predicted_xi,
# defcon_roles, team_overrides, external_projections, player_names, repo_events,
# style_matchup, ab_market_vs_recon and oddsapi_feed were never run by the harness at all.
def _discover_selftests():
    """Every file carrying a real selftest, as (label, path) — src/ AND scripts/.

    scripts/ was omitted until 2026-09-07, so three runners shipped selftests the gate
    never ran: `lock_board` (which guards the integrity of the locked-prediction record),
    `postgw_review`, and this file. The convention in CLAUDE.md is that selftests are
    discovered and never listed, precisely so nothing can be silently left out — but the
    discovery itself only looked in one directory, which is the same class of omission
    one level up.

    `test_all` is skipped: running the harness from inside the harness is a recursion, and
    its selftest is exercised by invoking it directly.
    """
    out = []
    for d in ("src", "scripts"):
        for f in sorted(glob.glob(_os.path.join(ROOT, d, "*.py"))):
            name = _os.path.splitext(_os.path.basename(f))[0]
            if d == "scripts" and name == "test_all":
                continue
            try:
                src = open(f, encoding="utf-8").read()
            except OSError:
                continue
            # BOTH conditions: a module that merely MENTIONS --selftest in a docstring
            # (betting_odds_ingest does) has no selftest to run and would be reported as
            # a failing argparse invocation.
            if "--selftest" in src and "def selftest(" in src:
                label = name if d == "src" else f"{d}/{name}"
                out.append((label, _os.path.join(d, f"{name}.py")))
    return out


# Acceptance tests are discovered for the same reason the selftests are: a hardcoded
# list silently omits whatever was added last, which is exactly the test most worth
# running. Every .py in tests/ is a runnable script by convention.
def _discover_acceptance():
    return sorted(_os.path.splitext(_os.path.basename(f))[0]
                  for f in glob.glob(_os.path.join(ROOT, "tests", "*.py"))
                  if not _os.path.basename(f).startswith("_"))
PIPELINE = ["build_all", "gw_board", "run_final_board", "cs_fixtures", "run_solio_ensemble"]
SKIP_STUDIES = {"sd_design"}          # pre-registration arithmetic, needs no run


# Inputs the project documents as OPTIONAL. A script that stops because one of these is
# absent has not failed — it has nothing to run on. Reporting those as failures buries the
# real ones, so they are surfaced as SKIP with the reason.
OPTIONAL_INPUTS = ("fpl-data-stats.csv",)


# Children write to a PIPE here, not to a console. On Windows that makes stdout cp1252
# (the console itself is UTF-8), so any script printing a Greek letter or an arrow dies
# with UnicodeEncodeError when the harness runs it and passes when you run it by hand —
# the worst possible failure mode for a regression harness. tests/test_regime.py prints
# "dSD" as a real delta and was failing here for that reason alone. Force UTF-8 on every
# child rather than policing the character set of every print statement.
CHILD_ENV = dict(_os.environ)
CHILD_ENV["PYTHONIOENCODING"] = "utf-8"


def run(cmd, timeout=1800, env=None):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout, env=env or CHILD_ENV,
                           encoding="utf-8", errors="replace")
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



IMPORT_SENTINEL = "__IMPORTCHECK__"


def _src_modules():
    return sorted(_os.path.splitext(_os.path.basename(f))[0]
                  for f in glob.glob(_os.path.join(ROOT, "src", "*.py")))


def _import_check_child():
    """Import every src module in ONE interpreter, purging project modules between each.

    Section 1 used to spawn one interpreter per module: 56 processes, each paying the
    full numpy/pandas import baseline, to answer 56 independent yes/no questions. The
    measurement in docs/INDEX.md puts that section at ~216s of which ~163s is startup
    this project does not control -- so the expensive thing was never any module, it was
    the spawning.

    Isolation is not thrown away to get that back. What section 1 exists to catch is a
    module that does not import at all, and, second, one that imports only because
    something else already populated sys.modules. So third-party packages stay loaded
    (that is the whole saving) while every module whose __file__ lives under src/ is
    dropped from sys.modules after each import -- project code is therefore imported
    genuinely fresh each time. What survives is only non-sys.modules state: env vars,
    warnings filters, C-extension registration. That residue is why the parent re-runs
    any module that fails here in its own interpreter before reporting it, and why a
    module that fails batched but passes alone is surfaced rather than quietly passed.
    """
    import importlib
    import traceback
    src_dir = _os.path.abspath(_os.path.join(ROOT, "src"))
    for mod in _src_modules():
        before = set(_sys.modules)
        t0 = time.time()
        try:
            importlib.import_module(mod)
            status, detail = "ok", ""
        except BaseException:
            # BaseException, not Exception: a module calling sys.exit() at import time is
            # an import failure, and SystemExit would otherwise sail past and take the
            # whole batch with it.
            status, detail = "fail", traceback.format_exc()
        secs = time.time() - t0
        for name in set(_sys.modules) - before:
            fp = getattr(_sys.modules.get(name), "__file__", None) or ""
            if fp and _os.path.abspath(fp).startswith(src_dir):
                _sys.modules.pop(name, None)
        print(f"{IMPORT_SENTINEL}\t{mod}\t{status}\t{secs:.3f}\t{detail!r}", flush=True)


def _import_rows():
    """Rows for section 1. Batched first, then anything unproven re-run in isolation.

    Deliberately does NOT go through run(): that helper truncates output to the last 400
    characters and reinterprets a FileNotFoundError on an optional input as a skip, and
    both would destroy the per-module protocol lines this parses.
    """
    mods = _src_modules()
    t0 = time.time()
    try:
        r = subprocess.run([PY, _os.path.abspath(__file__), "--import-check"],
                           cwd=ROOT, capture_output=True, text=True, timeout=900,
                           env=CHILD_ENV, encoding="utf-8", errors="replace")
        out = (r.stderr or "") + (r.stdout or "")
    except subprocess.TimeoutExpired:
        out = ""
    batch_secs = time.time() - t0

    seen = {}
    for line in out.splitlines():
        if line.startswith(IMPORT_SENTINEL + "\t"):
            _, mod, status, secs, detail = line.split("\t", 4)
            seen[mod] = (status == "ok", float(secs), ast.literal_eval(detail))

    rows, notes = [], []
    for mod in mods:
        batched = seen.get(mod)
        if batched and batched[0]:
            rows.append((mod, True, batched[1], ""))
            continue
        # Unproven: it failed batched, or the batch never reached it because an earlier
        # module killed the interpreter. Either way the verdict comes from its own
        # process, so this section reports exactly what it reported before.
        good, secs, err = run(
            [PY, "-c", f"import sys; sys.path.insert(0,'src'); import {mod}"], 300)
        rows.append((mod, good, secs, err))
        if batched is None:
            notes.append(f"{mod}: batch never reached it (an earlier module ended the "
                         f"interpreter); verdict from its own process")
        elif good:
            notes.append(f"{mod}: FAILS after other src modules import but PASSES alone "
                         f"-- a cross-module side effect, not an import error. Reported "
                         f"as its isolated result; worth a look.")
    return rows, notes, batch_secs


def main():
    if "--import-check" in _sys.argv:      # re-entrant child of _import_rows()
        _import_check_child()
        return True
    quick = "--quick" in _sys.argv
    allok = True

    section("0. INPUT DATA (the boundary we do not control)")
    # Only external and committed nodes gate here. A derived artifact that is stale or
    # absent is a build step not yet run — which is exactly what section 4 does next —
    # so failing on it would make the harness unrunnable on a fresh clone. What must
    # hold before anything else is meaningful is that the INPUTS are shaped as the
    # model expects: the 2026-08-21 incident (teams.csv kept its `elo` column and went
    # entirely NULL upstream) passed every test in this file and surfaced forty lines
    # into the team layer as "SVD did not converge".
    import manifest
    gate = [n for n in manifest.graph().values() if n.kind in ("external", "committed")]
    rows = []
    for name, status, detail in manifest.check(gate):
        rows.append((name, status == "ok", 0.0, detail))
    allok &= report(rows)
    stale = [n for n, s, _ in manifest.check() if s == "STALE"]
    if stale:
        print(f"  note: {len(stale)} derived artifact(s) stale "
              f"({', '.join(stale[:3])}...) — section 4 rebuilds them")

    section("1. IMPORTS")
    rows, notes, batch_secs = _import_rows()
    allok &= report(rows)
    print(f"  ({len(rows)} modules in one interpreter, {batch_secs:.1f}s wall; the times "
          f"above are per-module import cost with process startup excluded)")
    for n in notes:
        print(f"  note: {n}")

    section("2. MODULE SELFTESTS")
    rows = [(label,) + run([PY, path, "--selftest"], 900)
            for label, path in _discover_selftests()]
    allok &= report(rows)

    section("3. ACCEPTANCE TESTS")
    rows = [(t,) + run([PY, _os.path.join("tests", f"{t}.py")], 1800)
            for t in _discover_acceptance()]
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
        # NOT `>= 0`.  [CHANGED 2026-09-08 — JUDGMENT, argued below; overrule it by
        # restoring the strict form if you disagree.]
        #
        # The strict version asserts "no player has negative expected points", which is a
        # MODELLING claim, not a rule of the game, and it is false: a keeper who comes on
        # as a substitute (+1) at a club conceding four (-2) nets -1. It held until now by
        # luck. Twenty-five rows of the live board sit inside [-0.001, +0.010] and every
        # one of them is a GOALKEEPER at HULL, the weakest defence — their app_ev and
        # conc_ev cancel to within a thousandth of a point (Lo-Tutala GW16: +0.013 and
        # -0.013). Any change that moves Hull's start probabilities at all flips the sign
        # of one of them; the XI-constraint reordering did, at -0.001.
        #
        # A binary test on a quantity whose true value is zero fails at random forever, so
        # it tests the seed rather than the model. The bound below is what the invariant
        # actually means — no MATERIALLY negative projection — and is still far tighter
        # than anything a real defect could hide under: a bench keeper's EV is bounded by
        # p_start x conditional-net, a few hundredths at most, while a sign error or a
        # mis-scaled penalty term lands whole points below zero.
        _NEG_TOL = -0.05
        checks.append((f"no materially negative projections (min {b['mean'].min():+.3f}, "
                       f"floor {_NEG_TOL})", (b["mean"] >= _NEG_TOL).all()))
        checks.append(("every row has a club", b["team"].notna().all()))
        checks.append(("every row has a price", b["cost"].notna().all()))
        checks.append(("prices in 3.5-16.0", b["cost"].between(3.5, 16.0).all()))
        # GW_HI is an env knob (default 10), so pinning the count to 10 fails the board
        # every time it is run to a longer horizon. The invariant that actually matters
        # is that the gameweeks are a CONTIGUOUS run starting at 1 with nothing missing.
        _gws = sorted(int(g) for g in b["gw"].unique())
        checks.append((f"gameweeks contiguous 1..{_gws[-1] if _gws else 0}",
                       bool(_gws) and _gws == list(range(1, _gws[-1] + 1))))
        checks.append(("one row per player-gameweek",
                       not b.duplicated(["player", "team", "gw"]).any()))
        # `total` sums `blended`, not `mean` — gw_board mixes the external feed into GW1
        # for matched players, so ~30 of them differ. Comparing against `mean` fails for
        # exactly those, which is a wrong check rather than a wrong board.
        checks.append(("wide totals match long sums (blended)", bool(np.allclose(
            b.groupby(["player", "team"])["blended"].sum().sort_index().values,
            w.set_index(["player", "team"])["total"].sort_index().values, atol=0.05))))
        # Per GAMEWEEK, for the same reason the contiguity check above is horizon-free:
        # `total` sums however many weeks the board ran, so a fixed 40-120 band silently
        # encodes GW_HI=10 and fails the moment the board is run to a full season. The
        # quantity with a stable plausible range is the top scorer's points per week.
        _top = w["total"].max() / (len(_gws) or 1)
        checks.append((f"top scorer plausible per GW ({_top:.2f}, expect 4-12)",
                       4.0 <= _top <= 12.0))
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
        env = dict(CHILD_ENV); env["FPL_OUTPUTS"] = out
        r = subprocess.run([PY, _os.path.join("scripts", "gw_board.py")],
                           cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=2400, encoding="utf-8", errors="replace")
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
