import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
profile_pipeline.py — where the time actually goes.
===================================================

`test_all.py` takes ~20 minutes and nobody knows which part. That is the whole
problem with "seek efficiencies" as an instruction: without a measurement, the
plausible-looking candidates (a slow-looking loop, a repeated read) get optimised
while the real cost sits somewhere nobody looked. This measures it first.

Two things are timed, and they are different questions:

  IMPORTS   `test_all` spawns one interpreter per module — 56 of them — and each
            pays for the interpreter plus pandas/scipy before it does anything this
            project wrote. So a raw per-module time is almost entirely baseline and
            ranking on it tells you nothing. What is reported is the cost ABOVE a
            measured `import pandas` baseline: the part that is actually ours, and
            the only part an edit could remove.

  STAGES    each script in the pipeline, end to end (--full only; these are minutes,
            not seconds).

The output is a CSV in SCRATCH, not OUTPUTS: a profile is machine-specific evidence
about this box, and committing it invites comparing numbers measured on different
hardware.

What this does NOT do is suggest fixes. It is an instrument, and the register here
is that a measurement precedes the hypothesis, not the other way round.

Run:  .\fpl.ps1 run scripts/profile_pipeline.py            # imports only (~2-4 min)
      .\fpl.ps1 run scripts/profile_pipeline.py --full     # + every pipeline stage
      .\fpl.ps1 run scripts/profile_pipeline.py --selftest
"""
import glob
import subprocess
import time

PY = _sys.executable
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

# Same reason test_all.py forces it: a child writing to a PIPE gets cp1252 stdout on
# Windows, and a module printing a Greek letter at import time would be timed as a
# failure rather than as a module.
CHILD_ENV = dict(_os.environ)
CHILD_ENV["PYTHONIOENCODING"] = "utf-8"

# Kept in step with test_all.PIPELINE by importing it rather than restating it — a
# second copy of that list is a second thing to forget to update.
def _pipeline():
    import test_all
    return list(test_all.PIPELINE)


def _time(cmd, repeat=1, timeout=2400):
    """Best of `repeat` runs. Best, not mean: we are measuring a floor that an edit
    could move, and the mean is dominated by whatever else the machine was doing."""
    best, err = None, ""
    for _ in range(repeat):
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               timeout=timeout, env=CHILD_ENV,
                               encoding="utf-8", errors="replace")
            dt = time.time() - t0
            if r.returncode != 0:
                err = ((r.stderr or "") + (r.stdout or ""))[-200:]
                return None, err
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"
        best = dt if best is None else min(best, dt)
    return best, err


def baseline(repeat=3):
    """(bare interpreter, interpreter + pandas). Everything else is reported net of
    the second number, because every module in src/ imports pandas transitively."""
    bare, _ = _time([PY, "-c", "pass"], repeat=repeat, timeout=120)
    pan, _ = _time([PY, "-c", "import pandas"], repeat=repeat, timeout=300)
    return bare, pan


def profile_imports(repeat=2, progress=True):
    """Per-module import cost above the pandas baseline, slowest first.

    `progress` exists because this takes minutes: 56 modules x `repeat` interpreter
    spawns, each paying the pandas baseline. A tool that prints nothing for ten
    minutes is indistinguishable from a hung one, which is the same failure this
    project fixed in `doctor` by ending in a plan rather than in silence.
    """
    bare, pan = baseline()
    if progress:
        print(f"  bare interpreter      {bare:6.3f}s")
        print(f"  + import pandas       {pan:6.3f}s   <- every module below pays this")
    files = sorted(glob.glob(_os.path.join(ROOT, "src", "*.py")))
    rows = []
    for i, f in enumerate(files, 1):
        mod = _os.path.splitext(_os.path.basename(f))[0]
        if progress:
            print(f"  [{i:2d}/{len(files)}] {mod}", flush=True)
        t, err = _time([PY, "-c", f"import sys; sys.path.insert(0,'src'); import {mod}"],
                       repeat=repeat, timeout=300)
        if t is None:
            rows.append(("import", mod, None, None, err[:80]))
            continue
        rows.append(("import", mod, round(t, 3), round(max(0.0, t - pan), 3), ""))
    rows.sort(key=lambda r: (r[3] is None, -(r[3] or 0)))
    return bare, pan, rows


def profile_stages(repeat=1):
    """Each pipeline script, end to end. Minutes, so `repeat` defaults to 1."""
    rows = []
    for s in _pipeline():
        t, err = _time([PY, _os.path.join("scripts", f"{s}.py")], repeat=repeat)
        rows.append(("stage", s, None if t is None else round(t, 2), None, err[:80]))
    rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))
    return rows


def write(rows, path=None):
    import pandas as pd
    path = path or config.PROFILE
    d = pd.DataFrame(rows, columns=["kind", "name", "seconds", "seconds_net", "error"])
    d.to_csv(path, index=False)
    return path


def main():
    if "--selftest" in _sys.argv:
        return selftest()
    full = "--full" in _sys.argv
    rows = []

    print("timing baselines ...")
    bare, pan, imp = profile_imports()
    print()
    print("IMPORTS — cost above the pandas baseline, slowest first")
    ours = sum(r[3] for r in imp if r[3] is not None)
    for kind, name, t, net, err in imp[:15]:
        if t is None:
            print(f"  {name:26s}   FAILED  {err}")
        else:
            print(f"  {name:26s} {t:7.3f}s  net {net:7.3f}s")
    print(f"  ... {len(imp) - 15} more")
    print(f"  -> {len(imp)} modules, {ours:.1f}s of OUR import cost, "
          f"{pan * len(imp):.1f}s of unavoidable pandas baseline")
    rows += imp

    if full:
        print()
        print("STAGES — end to end")
        stages = profile_stages()          # once: each of these is minutes
        for kind, name, t, net, err in stages:
            print(f"  {name:26s} {'FAILED  ' + err if t is None else f'{t:8.1f}s'}")
        rows += stages
    else:
        print("\n(stages skipped — pass --full to time the pipeline end to end)")

    p = write(rows)
    print(f"\nwrote {p}")
    return 0


def selftest():
    """Offline and fast. Times only the bare interpreter, so it adds seconds to
    test_all rather than the minutes a real profile takes."""
    import tempfile
    bare, _ = _time([PY, "-c", "pass"], repeat=2, timeout=120)
    assert bare is not None and bare > 0, bare
    # a failing child is reported as None with the error, never as a time
    t, err = _time([PY, "-c", "raise SystemExit(3)"], timeout=60)
    assert t is None and err is not None, (t, err)
    # best-of-N returns the minimum, not the last or the mean
    t1, _ = _time([PY, "-c", "pass"], repeat=3, timeout=120)
    assert t1 <= bare + 1.0, (t1, bare)
    # the stage list is taken from test_all, not restated here
    assert "gw_board" in _pipeline(), _pipeline()
    # write() produces the declared columns, and puts them where config says
    p = _os.path.join(tempfile.mkdtemp(), "profile.csv")
    write([("import", "x", 1.0, 0.5, "")], p)
    import pandas as pd
    d = pd.read_csv(p)
    assert list(d.columns) == ["kind", "name", "seconds", "seconds_net", "error"], list(d.columns)
    print("profile_pipeline selftest ok")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
