import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
import manifest
"""
doctor.py — one call that says whether this repo is in a state you can trust.
=============================================================================

The question asked at the start of every session was some version of "is what I am
looking at current, and if not, what do I run?". Answering it meant: print the
paths, look at a few mtimes, remember which script writes which file, and guess.
Four or five commands, each with its own environment preamble, and the answer was
only as good as the memory of the dependency order.

This prints the answer. Three sections:

  ENVIRONMENT  the resolved paths and Python, plus whether the external feed is
               newer than the last deadline (a feed that stopped updating is the
               one failure that looks exactly like "nothing happened this week")
  ARTIFACTS    manifest.check() — missing, schema-broken, empty, or stale
  PLAN         the ordered list of scripts that makes everything fresh again,
               computed from the graph rather than recalled

Exit code is 0 when nothing is wrong, 1 otherwise, so it works as a gate.

Run:  .\fpl.ps1 doctor            (or: python scripts/doctor.py)
      python scripts/doctor.py --selftest
"""

def next_gameweek(gw, now):
    """(next gw id or None, deadline of the most recent one that has passed).

    Split out of `environment()` because it is the one piece here with a wrong
    answer that looks plausible: gameweek_summaries.csv is NOT in id order — in the
    26/27 file GW15 is row 0 and GW3 is row 29 — so reading `.iloc[0]` of the future
    rows reports whichever gameweek the feed happened to write first. Sorting by the
    deadline is the only safe read. `lock_board.gameweeks()` sorts for the same reason.
    """
    import pandas as pd
    g = gw.copy()
    g["deadline"] = pd.to_datetime(g["deadline_time"], utc=True, errors="coerce")
    g = g.sort_values("deadline")
    past = g.loc[g["deadline"] <= now, "deadline"]
    nxt = g.loc[g["deadline"] > now, "id"]
    return (int(nxt.iloc[0]) if len(nxt) else None,
            past.max() if len(past) else None)


def _fmt_age(seconds):
    if seconds is None:
        return "  n/a"
    h = seconds / 3600.0
    if h < 48:
        return f"{h:5.1f}h"
    return f"{h / 24:5.1f}d"


def environment():
    """Resolved paths, and whether the external feed has moved since the deadline."""
    import time
    import pandas as pd
    rows = []
    rows.append(("python", _sys.version.split()[0]))
    rows.append(("repo", config.ROOT))
    rows.append(("FPL_DATA", config.REPO if _os.path.isdir(config.REPO)
                 else f"{config.REPO}   << NOT A DIRECTORY"))
    rows.append(("FPL_HISTORY", config.HISTORY or "(unset — only fpl_history.py needs it)"))
    rows.append(("outputs", config.OUTPUTS))
    rows.append(("scratch", config.SCRATCH))

    problems = []
    notes = []
    if not _os.path.isdir(config.REPO):
        problems.append("FPL_DATA does not resolve to a directory — nothing will load")
        return rows, problems, notes

    # Feed freshness against the schedule, not against the clock. "playerstats.csv is
    # 30 hours old" means nothing on its own; "playerstats.csv has not moved since
    # before the last deadline" means the projections are running on pre-deadline data.
    gwp = _os.path.join(manifest.EXTERNAL, "gameweek_summaries.csv")
    stats = _os.path.join(manifest.EXTERNAL, "playerstats.csv")
    if _os.path.exists(gwp) and _os.path.exists(stats):
        now = pd.Timestamp.now(tz="UTC")
        nxt, last_deadline = next_gameweek(pd.read_csv(gwp), now)
        rows.append(("next gameweek", str(nxt) if nxt else "(season over)"))
        feed_mtime = pd.Timestamp(_os.path.getmtime(stats), unit="s", tz="UTC")
        rows.append(("feed updated", f"{_fmt_age(time.time() - _os.path.getmtime(stats))} ago"))
        if last_deadline is not None and feed_mtime < last_deadline:
            problems.append(
                f"playerstats.csv has not moved since the GW deadline at "
                f"{last_deadline:%Y-%m-%d %H:%M}Z — the board would be built on "
                f"pre-deadline data. Pull the FPL-Core-Insights repo.")

        # Elo is a silent degradation, not a failure. `teams.csv` keeps its `elo`
        # column and goes entirely NULL upstream; core_insights falls back to the
        # pinned snapshot and says so in a log line nobody reads. The manifest cannot
        # express this: marking `elo` non-null would make the whole gate red on a
        # condition the model handles, and leaving it out — which is what was done —
        # makes the gate green while the team layer runs on a pin of unknown age.
        #
        # The rule is deliberately not an age threshold. A pin is stale when it
        # predates the last COMPLETED gameweek, because then no 26/27 result can be
        # in it and the team layer is running on pre-season strength.
        rows.append(("upstream elo", _elo_state()))
        pin = _elo_pin_stale(last_deadline)
        if pin:
            # A NOTE, not a PROBLEM. `studies/elo_pin_sensitivity.py` measured the
            # bound: removing the Elo channel ENTIRELY — the most a wrong pin can cost,
            # since a stale Elo cannot be worse than no Elo — clears the pre-registered
            # rule with room to spare. Elo is a blend into the prior MEANS at
            # bayes_model.py:166, not a likelihood term, and 380 match rows sit on top
            # of it. Reporting this at PROBLEM severity made the gate permanently red
            # over a documented non-issue, which is how a gate stops being read.
            #
            # The message cites the RULE, not the measurement. It said "0.30 posterior
            # SD" until 2026-09-09, when re-running the study against the pulled feed
            # moved it to 0.3415 and the sentence became quietly false — a hardcoded
            # number that drifts every re-run is the documentation-vs-code gap this
            # tool exists to close, reproduced inside the tool itself.
            notes.append(pin)

        # The loop that is actually broken. PROJECT_KNOWLEDGE §6.6 asks for the model
        # to be scored against real gameweeks, and GW2_REVIEW records that it could
        # not be: "No locked board existed for this gameweek", so the review scored
        # the CURRENT board and is a diagnostic rather than a forecasting result. A
        # board saved after the deadline is not a prediction and never becomes one —
        # this is the one check here whose window closes permanently.
        if nxt is not None:
            lock = _lock_state(nxt)
            rows.append((f"locked for GW{nxt}", lock))
            if lock.startswith("NO"):
                problems.append(
                    f"no locked board for GW{nxt} — score it as a forecast and the "
                    f"window shuts at the deadline. `.\\fpl.ps1 run scripts/lock_board.py "
                    f"--auto` writes one.")
    return rows, problems, notes


def _elo_upstream_dead():
    """True if teams.csv carries an `elo` column that is entirely NULL."""
    import pandas as pd
    p = _os.path.join(manifest.EXTERNAL, "teams.csv")
    if not _os.path.exists(p):
        return None
    t = pd.read_csv(p)
    return "elo" in t.columns and t["elo"].isna().all()


def _pin_as_of():
    """The `as_of` date on the pinned Elo snapshot, or None."""
    import pandas as pd
    if not _os.path.exists(config.TEAM_ELO):
        return None
    d = pd.read_csv(config.TEAM_ELO)
    if "as_of" not in d.columns or not len(d):
        return None
    return pd.to_datetime(d["as_of"].iloc[0], utc=True, errors="coerce")


def _elo_state():
    dead = _elo_upstream_dead()
    if dead is None:
        return "teams.csv absent"
    if not dead:
        return "live (upstream populated)"
    pin = _pin_as_of()
    return f"DEAD upstream — using pin from {pin:%Y-%m-%d}" if pin is not None \
        else "DEAD upstream — and the pin has no as_of"


def _elo_pin_stale(last_deadline):
    """A problem string if the pin predates the last completed gameweek, else None.

    Not an age threshold: the question is whether any 26/27 result can be in the pin.
    If it was taken before the most recent deadline that has passed, the answer is no
    and the team layer is running on pre-season strength.
    """
    if not _elo_upstream_dead():
        return None
    pin = _pin_as_of()
    if pin is None:
        return ("teams.csv Elo is NULL upstream and data/team_elo_2627.csv carries no "
                "`as_of` — the age of the strength the team layer is using is unknown.")
    if last_deadline is not None and pin < last_deadline:
        days = (last_deadline - pin).days
        return (f"teams.csv Elo is NULL upstream and the pinned snapshot is from "
                f"{pin:%Y-%m-%d}, {days}d before the last deadline "
                f"({last_deadline:%Y-%m-%d}), so it contains no 26/27 result. MEASURED "
                f"and not urgent: removing the Elo channel entirely — the upper bound "
                f"on a wrong pin — stays inside the pre-registered rule (<0.5 posterior "
                f"SD, Spearman rho > 0.98). Current numbers: "
                f"studies/elo_pin_sensitivity.csv. Re-run that study if ELO_WEIGHT "
                f"rises above 0.45.")
    return None


def _lock_state(gw):
    """Whether a pre-deadline board exists for `gw`. lock_board names the real thing
    `gw{n}_board_locked_<stamp>_deadline.csv`; an `_early` or `_LATE` file is
    deliberately named so it cannot be mistaken for one."""
    import glob
    hits = glob.glob(_os.path.join(config.PREDICTIONS, f"gw{gw}_board_locked_*_deadline*.csv"))
    real = [h for h in hits if "LATE" not in _os.path.basename(h)]
    if real:
        return _os.path.basename(real[0])
    other = glob.glob(_os.path.join(config.PREDICTIONS, f"gw{gw}_board_locked_*.csv"))
    if other:
        return f"NO deadline lock (only {_os.path.basename(other[0])})"
    return "NO locked board"


def plan(findings):
    """The ordered scripts that fix every STALE / missing derived artifact.

    Built from the graph rather than the finding order: rebuilding the board before
    its priors is the mistake this replaces, and it is silent — the board runs fine
    on stale priors and is simply wrong.
    """
    broken = [name for name, status, _ in findings if status in ("STALE", "skip")]
    if not broken:
        return []
    # Expand to everything downstream of each broken node, then order the union.
    need = list(broken)
    for b in broken:
        for d in manifest.downstream(b):
            if d not in need:
                need.append(d)
    order = [n.name for n in manifest._NODES if n.name in need]
    return manifest.producers_for(order)


def main():
    if "--selftest" in _sys.argv:
        return selftest()
    bad = 0

    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    rows, problems, notes = environment()
    for k, v in rows:
        print(f"  {k:14s} {v}")
    for p in problems:
        print(f"  PROBLEM  {p}")
    for n in notes:
        print(f"  note     {n}")
    bad += len(problems)

    print()
    print("=" * 72)
    print("ARTIFACTS")
    print("=" * 72)
    findings = manifest.check()
    for name, status, detail in findings:
        print(f"  {status:8s} {name:30s} {detail}")
    bad += sum(1 for _, s, _ in findings if s not in ("ok", "skip"))

    print()
    print("=" * 72)
    print("PLAN")
    print("=" * 72)
    steps = plan(findings)
    if not steps:
        print("  everything is current — nothing to rebuild")
    else:
        for i, s in enumerate(steps, 1):
            print(f"  {i}. .\\fpl.ps1 run {s}")

    print()
    print(f"  -> {bad} problem(s)" if bad else "  -> clean")
    return 1 if bad else 0


def selftest():
    """Offline. Exercises the ordering logic on a synthetic set of findings; does
    not touch the real tree, which may legitimately be stale."""
    # plan() must return producers in dependency order, never a consumer first.
    findings = [("ms_priors.pkl", "STALE", ""), ("gw_board_long.csv", "ok", "")]
    steps = plan(findings)
    assert "scripts/build_all.py" in steps, steps
    assert "scripts/gw_board.py" in steps, steps
    assert steps.index("scripts/build_all.py") < steps.index("scripts/gw_board.py"), steps
    # A clean tree plans nothing.
    assert plan([("gw_board_long.csv", "ok", "")]) == []
    # A downstream-only break does not drag the whole pipeline back in.
    steps = plan([("solio_ensemble_demo.csv", "STALE", "")])
    assert steps == ["scripts/run_solio_ensemble.py"], steps
    # next_gameweek must not depend on row order. This frame is shuffled the way the
    # real 26/27 feed is: a mid-season gameweek first, an early one buried.
    import pandas as pd
    g = pd.DataFrame({
        "id": [15, 16, 1, 2, 4, 3],
        "deadline_time": ["2026-12-12T13:30:00+00:00", "2026-12-19T13:30:00+00:00",
                          "2026-08-21T17:30:00+00:00", "2026-08-28T17:30:00+00:00",
                          "2026-09-12T12:30:00+00:00", "2026-09-04T17:30:00+00:00"]})
    now = pd.Timestamp("2026-09-07T00:00:00Z")
    nxt, last = next_gameweek(g, now)
    assert nxt == 4, f"row order leaked into the answer: got {nxt}"
    assert str(last)[:10] == "2026-09-04", last
    # season over -> None, and no exception
    assert next_gameweek(g, pd.Timestamp("2027-06-01T00:00:00Z"))[0] is None
    # _elo_pin_stale is a comparison against the last deadline, not an age threshold.
    import tempfile
    import pandas as pd
    dl = pd.Timestamp("2026-08-28T17:30:00Z")
    real_elo = config.TEAM_ELO
    try:
        tmpd = tempfile.mkdtemp()
        config.TEAM_ELO = _os.path.join(tmpd, "elo.csv")
        # a pin taken BEFORE the last deadline cannot contain any 26/27 result
        pd.DataFrame({"code": [1], "elo": [1500.0],
                      "as_of": ["2026-08-14"]}).to_csv(config.TEAM_ELO, index=False)
        if _elo_upstream_dead():                 # only meaningful when it is
            assert _elo_pin_stale(dl), "a pre-deadline pin must be reported"
            # a pin taken AFTER it is fine
            pd.DataFrame({"code": [1], "elo": [1500.0],
                          "as_of": ["2026-09-01"]}).to_csv(config.TEAM_ELO, index=False)
            assert _elo_pin_stale(dl) is None, "a post-deadline pin is not stale"
            # no as_of at all is its own finding
            pd.DataFrame({"code": [1], "elo": [1500.0]}).to_csv(config.TEAM_ELO, index=False)
            assert _elo_pin_stale(dl), "a pin with no as_of must be reported"
        # never raises when the pin is absent
        config.TEAM_ELO = _os.path.join(tmpd, "gone.csv")
        _elo_state()
        _elo_pin_stale(dl)
    finally:
        config.TEAM_ELO = real_elo

    # _lock_state must not accept an `_early` or `_LATE` file as a real deadline lock:
    # that is the failure GW2_REVIEW records, and a check that passes on the wrong file
    # is worse than none.
    real_pred = config.PREDICTIONS
    try:
        config.PREDICTIONS = tempfile.mkdtemp()
        assert _lock_state(9) == "NO locked board"
        open(_os.path.join(config.PREDICTIONS, "gw9_board_locked_2026-01-01_early.csv"), "w").close()
        assert _lock_state(9).startswith("NO deadline lock"), _lock_state(9)
        open(_os.path.join(config.PREDICTIONS,
                           "gw9_board_locked_2026-01-01_deadline_LATE_not_a_prediction.csv"),
             "w").close()
        assert _lock_state(9).startswith("NO"), _lock_state(9)
        open(_os.path.join(config.PREDICTIONS, "gw9_board_locked_2026-01-02_deadline.csv"), "w").close()
        assert _lock_state(9) == "gw9_board_locked_2026-01-02_deadline.csv", _lock_state(9)
    finally:
        config.PREDICTIONS = real_pred
    # environment() must not raise even when the feed is absent.
    real = manifest.EXTERNAL
    try:
        manifest.EXTERNAL = _os.path.join(config.SCRATCH, "_no_such_feed")
        r, p, n = environment()
        assert isinstance(n, list), "environment must always return the notes channel"
    finally:
        manifest.EXTERNAL = real
    print("doctor selftest ok")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
