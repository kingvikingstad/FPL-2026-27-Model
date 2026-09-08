from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
lock_board.py — preserve the projection for one gameweek BEFORE its deadline.
==============================================================================
`predictions/` is the only directory in this repo whose contents cannot be rebuilt. Once
a deadline passes, the availability, ownership and team-news inputs that produced a
projection are gone, and PROJECT_KNOWLEDGE §6.6 — score the model against real gameweeks
— is unanswerable for that week forever. GW2 has only a reconstruction and GW3 has
nothing, because this was a manual step nobody performed in time. This makes it one
command.

    python scripts/lock_board.py                  # lock the NEXT gameweek, primary
    python scripts/lock_board.py --label early    # an insurance lock, days out
    python scripts/lock_board.py --gw 5 --dry-run # preview, write nothing
    python scripts/lock_board.py --auto           # for a daily scheduler; see below

--auto, AND WHY THE TRIGGER IS A STATE TEST
--------------------------------------------
Deadlines do not fall on a weekday a cron can express: GW4 is a Saturday at 12:30 UTC,
GW18 a Wednesday at 18:30. Anything pinned to a time either fires days early — writing a
PRIMARY lock with none of the team news it exists to capture — or misses the week.

So `--auto` runs daily and acts only when the state says to: the next gameweek's deadline
is inside `DEADLINE_WINDOW_H`, and no lock for it exists yet. Every other day it prints
one line and exits 0, so a scheduler sees success rather than a daily failure. Running it
ten times in a day is a no-op nine times. This is the same trigger `postgw_review --auto`
uses, for the same reason.

`--auto` does two things before locking, in this order:

  1. FETCH TEAM NEWS for that gameweek (`fplpage`), because the predicted XI is the input
     the project's own scoring identifies as its largest error source — correlation
     roughly halves conditioned on appearing, and GW1 lost 52.6 projected points to 13
     availability misses. Best effort: fpl.page publishes the day before a deadline, so a
     404 is the normal answer when this runs early, and a lock without team news still
     beats no lock. The outcome is reported either way.
  2. REBUILD the board, because the failure this guards against is not locking late, it
     is locking a board built before the last data pull. `gw_board` folds the predicted XI
     it just fetched into the projection, which is why the order is fixed.

Both are expensive and therefore happen only on the day the lock is actually taken, never
on the quiet days before it.

WHAT IT REFUSES TO DO, AND WHY EACH REFUSAL EARNS ITS PLACE
------------------------------------------------------------
1. IT WILL NOT LOCK AFTER THE DEADLINE. This is the whole point of the directory. A file
   written at kickoff+1 looks exactly like a real lock and is worthless: the board by
   then may have absorbed team news that was published after the deadline, so scoring it
   measures hindsight. Refusing is not a convenience check, it is the integrity of every
   number that will ever be computed from the file. `--force-late` exists only so the
   refusal can be overridden deliberately and noisily, and it renames the output
   `_LATE_not_a_prediction` so it can never be mistaken for one.

2. IT WILL NOT OVERWRITE AN EXISTING LOCK. "Copied before the deadline and never
   regenerated" is the README's contract. Silently replacing a lock with a later, better
   board is how a prediction becomes a postdiction.

3. IT WARNS LOUDLY IF THE BOARD IS STALE. The most likely way to get a useless lock is
   not locking late, it is locking a board built before the last data pull — you preserve
   a prediction the model had already superseded. The board's mtime is compared against
   the newest match file in the data repo and the gap is printed.

WHICH LOCK GETS SCORED
----------------------
A gameweek may carry more than one: an `early` insurance lock and the `deadline` one.
`score_gw._locked()` always prefers `*_deadline*`, whatever the dates say, so the choice
is made by the code and not by whoever is looking at the results. See predictions/README.

Deadlines come from `gameweek_summaries.csv` in the data repo — offline, no API call, and
the same file the rest of the project reads. If that file is stale the deadline it
reports is stale too, which is why the staleness warning in (3) covers it.

Run:  python scripts/lock_board.py --selftest
"""
import argparse
import datetime as _dt
import glob as _glob
import pandas as pd

SEASON = "2026-2027"
# How close to the deadline a `deadline`-labelled lock has to be taken. The GW1 lock was
# ~6 hours out; a day is the loosest thing still honestly describable as "at the deadline".
#
# 26 and not 24 for one specific reason. The daily `--auto` scheduler catches every 26/27
# deadline SAME DAY, with lead times from 0.9h (the 10:00 UTC slot, GW6 and GW7) to 9.4h.
# But if a same-day run is missed — machine off, app closed — the fallback is the previous
# day's run, which for that earliest slot lands at 24.9h. At a 24h window those two
# gameweeks would be refused as "too early" on the fallback and then be past the deadline
# on the next run: silently never locked, which is the exact failure this whole path
# exists to prevent. 26h buys the fallback and costs at most two extra hours of team news
# in the rare case it is used.
DEADLINE_WINDOW_H = 26.0
KEEP = ["player_code", "player", "pos", "team", "cost", "own", "gw", "mean", "solio",
        "blended", "src", "sd", "app_ev", "att_ev", "defcon_ev", "cs_ev", "conc_ev",
        "par", "p5", "median", "p95"]


def gameweeks(season=SEASON, base=None):
    """id -> deadline (UTC), plus the is_next / is_current flags."""
    root = base or config.repo(season)
    g = pd.read_csv(_os.path.join(root, "gameweek_summaries.csv"))
    g["deadline"] = pd.to_datetime(g["deadline_time"], errors="coerce", utc=True)
    for c in ("is_next", "is_current", "finished"):
        if c in g.columns:
            g[c] = g[c].astype(str).str.lower().isin(["true", "1", "yes"])
    return g.sort_values("id").reset_index(drop=True)


def next_gw(g):
    """The gameweek to lock: the one FPL flags `is_next`, else the first unfinished."""
    if "is_next" in g.columns and g["is_next"].any():
        return int(g.loc[g["is_next"], "id"].iloc[0])
    live = g[~g.get("finished", pd.Series(False, index=g.index))]
    if not len(live):
        raise SystemExit("every gameweek is finished; nothing to lock")
    return int(live["id"].iloc[0])


def board_staleness(board_path, season=SEASON, base=None):
    """(board mtime, newest data mtime, hours the board lags). Negative means fresh."""
    root = base or config.repo(season)
    fs = _glob.glob(_os.path.join(root, "By Gameweek", "GW*", "matches.csv"))
    if not fs or not _os.path.exists(board_path):
        return None, None, None
    b = _dt.datetime.fromtimestamp(_os.path.getmtime(board_path), _dt.timezone.utc)
    d = _dt.datetime.fromtimestamp(max(_os.path.getmtime(f) for f in fs),
                                   _dt.timezone.utc)
    return b, d, (d - b).total_seconds() / 3600.0


def fetch_lineups(gw, verbose=True):
    """Pull the predicted XI / team news for `gw`. BEST EFFORT — never fatal.

    This is the input the project's own scoring says matters most and errs on worst: the
    correlation roughly halves once you condition on players who actually appeared
    (GW1 r=0.53 -> 0.32), and GW1 carried 13 availability misses worth 52.6 projected
    points. A deadline lock taken without it is preserving a prediction missing its most
    valuable ingredient, which is most of the reason a deadline lock beats an early one.

    It is best-effort and not a precondition, because a lock WITHOUT team news still beats
    no lock at all — a gameweek that goes unlocked can never be scored. But the outcome is
    reported loudly either way: whether the lock carries team news changes what it is
    worth, and that has to be visible in the file's provenance rather than inferred later.

    fpl.page publishes the day before a deadline, so a 404 is the NORMAL answer when this
    runs early and is not an error. Returns the number of predicted-XI sources available
    for `gw` afterwards, counting every feed `predicted_xi` can see — a manually saved
    Rotowire or FFS file counts just as much as the scrape.
    """
    try:
        import fplpage
        html = fplpage.fetch(gw)
        d, dropped, probs = fplpage.scrape(gw, html=html, verbose=False)
        if verbose:
            print(f"  team news: scraped {len(d)} players for GW{gw}"
                  + (f", {len(dropped)} unmatched" if len(dropped) else ""))
    except Exception as e:                                        # noqa: BLE001
        if verbose:
            msg = str(e)
            if "404" in msg:
                print(f"  team news: no GW{gw} preview published yet (404) — normal if "
                      f"this is running more than a day out")
            else:
                print(f"  team news: fetch failed ({type(e).__name__}: {msg[:120]})")
    try:
        import predicted_xi as pxi
        n = len(pxi.sources(gw))
    except Exception:
        n = 0
    if verbose:
        print(f"  team news: {n} predicted-XI source(s) on disk for GW{gw}"
              + ("" if n else "  ** the lock will carry NO team news **"))
    return n


def rebuild_board(verbose=True):
    """Re-run the board so the lock preserves a current projection, not a stale one.

    PYTHONIOENCODING is forced because a child writing to a pipe on Windows gets cp1252
    and dies on the first non-ASCII player name — the trap documented in CLAUDE.md and
    handled the same way in `test_all`.
    """
    import subprocess
    script = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "gw_board.py")
    env = dict(_os.environ, PYTHONIOENCODING="utf-8")
    if verbose:
        print("  rebuilding the board first ...")
    r = subprocess.run([_sys.executable, script], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("REFUSING: the board rebuild failed, so the lock would preserve "
                         "a stale projection. Nothing was written.\n"
                         + (r.stderr or r.stdout or "")[-1500:])
    if verbose:
        print("  board rebuilt")
    return True


def auto(out_dir=None, season=SEASON, base=None, now=None, verbose=True):
    """Lock the next gameweek IF its deadline is close and it is not already locked.

    Returns the path written, or None. Never raises on the quiet path: a scheduler must
    see exit 0 on the days there is nothing to do.
    """
    out_dir = out_dir or config.PREDICTIONS
    now = now or _dt.datetime.now(_dt.timezone.utc)
    g = gameweeks(season, base=base)
    try:
        gw = next_gw(g)
    except SystemExit:
        print("[auto] no unfinished gameweek; nothing to lock.")
        return None
    row = g[g["id"] == gw]
    deadline = row["deadline"].iloc[0] if len(row) else pd.NaT
    if pd.isna(deadline):
        print(f"[auto] GW{gw} has no deadline in gameweek_summaries.csv; nothing done.")
        return None
    hrs = (deadline.to_pydatetime() - now).total_seconds() / 3600.0
    if hrs <= 0:
        print(f"[auto] GW{gw}'s deadline passed {abs(hrs):.1f}h ago and it was never "
              f"locked. Nothing can be written now — that gameweek is unscoreable.")
        return None
    if hrs > DEADLINE_WINDOW_H:
        print(f"[auto] GW{gw} deadline in {hrs:.1f}h, outside the "
              f"{DEADLINE_WINDOW_H:.0f}h window. Nothing to do.")
        return None
    existing = _glob.glob(_os.path.join(out_dir, f"gw{gw}_board_locked_*_deadline*.csv"))
    if existing:
        print(f"[auto] GW{gw} already has a deadline lock "
              f"({_os.path.basename(existing[-1])}). Nothing to do.")
        return None
    print(f"[auto] GW{gw} deadline in {hrs:.1f}h and unlocked — locking now.")
    # Order matters: team news first, THEN the board, because `gw_board` folds the
    # predicted XI into the projection it is about to build. Fetching after the rebuild
    # would lock a board that never saw it.
    n_xi = fetch_lineups(gw, verbose=verbose)
    rebuild_board(verbose=verbose)
    path = lock(gw=gw, label="deadline", out_dir=out_dir, season=season, base=base,
                now=now, verbose=verbose)
    if verbose:
        print(f"\n[auto] locked GW{gw} "
              + (f"WITH team news ({n_xi} source(s))" if n_xi
                 else "WITHOUT team news — no preview was available"))
    return path


def lock(gw=None, label="deadline", board_path=None, out_dir=None, season=SEASON,
         base=None, dry_run=False, force_late=False, now=None, verbose=True,
         anyway=False):
    """Write the lock. Returns the path written, or None on a dry run."""
    board_path = board_path or _os.path.join(config.OUTPUTS, "gw_board_long.csv")
    out_dir = out_dir or config.PREDICTIONS
    now = now or _dt.datetime.now(_dt.timezone.utc)

    g = gameweeks(season, base=base)
    gw = int(gw) if gw is not None else next_gw(g)
    row = g[g["id"] == gw]
    if not len(row):
        raise SystemExit(f"GW{gw} is not in gameweek_summaries.csv")
    deadline = row["deadline"].iloc[0]
    late = pd.notna(deadline) and now >= deadline.to_pydatetime()

    if verbose:
        print("=" * 74)
        print(f"LOCK GW{gw}  ({label})")
        print("=" * 74)
        print(f"  deadline  {deadline}")
        print(f"  now       {now:%Y-%m-%d %H:%M:%S%z}")
        if pd.notna(deadline):
            hrs = (deadline.to_pydatetime() - now).total_seconds() / 3600.0
            print(f"  {'PASSED by' if late else 'time remaining'} {abs(hrs):.1f} hours")

    # A `deadline` lock taken days out is a contradiction: it becomes the PRIMARY file
    # `score_gw` scores, while containing none of the team news the deadline lock exists
    # to capture. That is the exact mistake that would quietly undo the pre-declaration,
    # and it is one careless default away, so it is refused rather than warned about.
    if (label == "deadline" and not late and pd.notna(deadline)
            and not anyway):
        hrs = (deadline.to_pydatetime() - now).total_seconds() / 3600.0
        if hrs > DEADLINE_WINDOW_H:
            raise SystemExit(
                f"REFUSING: this would write the PRIMARY lock {hrs:.0f} hours before the "
                f"deadline, so it would be scored without the team news, injuries and "
                f"price moves it exists to capture. Use `--label early` for an insurance "
                f"lock now, and run this again inside {DEADLINE_WINDOW_H:.0f} hours of "
                f"the deadline ({deadline}). Pass --anyway to override.")

    if late and not force_late:
        raise SystemExit(
            f"REFUSING: GW{gw}'s deadline ({deadline}) has passed. A board saved now is "
            f"not a prediction — it may contain team news published after the deadline, "
            f"so scoring it would measure hindsight. Nothing was written. If you truly "
            f"want the file for another purpose, pass --force-late; it will be named so "
            f"it can never be mistaken for a lock.")

    b = pd.read_csv(board_path)
    if "gw" not in b.columns:
        raise SystemExit(f"{board_path} has no `gw` column — is it a gw_board_long.csv?")
    sel = b[b["gw"] == gw].copy()
    if not len(sel):
        raise SystemExit(f"the board carries no GW{gw} rows (it spans "
                         f"GW{int(b['gw'].min())}-{int(b['gw'].max())})")

    bt, dt_, lag = board_staleness(board_path, season, base=base)
    if verbose and lag is not None:
        print(f"  board built {bt:%Y-%m-%d %H:%M}, newest data {dt_:%Y-%m-%d %H:%M}")
        if lag > 1.0:
            print(f"  ** WARNING: the board is {lag:.1f} hours OLDER than the data. You "
                  f"would be preserving a projection the model has already superseded. "
                  f"Re-run scripts/gw_board.py first unless that is deliberate. **")
        else:
            print("  board is current with the data")

    cols = [c for c in KEEP if c in sel.columns]
    missing = [c for c in ("player_code", "player", "blended") if c not in cols]
    if missing:
        raise SystemExit(f"the board is missing required columns {missing}")
    out = sel[cols]

    stamp = now.strftime("%Y-%m-%d")
    name = (f"gw{gw}_board_locked_{stamp}_{label}.csv" if not (late and force_late)
            else f"gw{gw}_board_locked_{stamp}_{label}_LATE_not_a_prediction.csv")
    path = _os.path.join(out_dir, name)
    if _os.path.exists(path):
        raise SystemExit(f"REFUSING: {name} already exists. Locks are never regenerated "
                         f"— see predictions/README.md.")
    existing = sorted(_os.path.basename(p) for p in
                      _glob.glob(_os.path.join(out_dir, f"gw{gw}_board_locked_*.csv")))
    if verbose and existing:
        print(f"  note: GW{gw} already has {len(existing)} lock(s): {existing}")
        print("        `score_gw` scores the *_deadline* one; the rest are secondary.")

    if dry_run:
        if verbose:
            print(f"\n  DRY RUN — would write {name} ({len(out)} players, "
                  f"{len(cols)} columns)")
        return None

    out.to_csv(path, index=False)
    if verbose:
        print(f"\n  wrote {path}")
        print(f"  {len(out)} players, projected total {out['blended'].sum():.1f}")
        print("  top: " + ", ".join(out.nlargest(5, "blended")["player"].astype(str)))
        print("\n  README row:")
        print(f"| `{name}` | {now:%Y-%m-%d %H:%M} UTC, deadline {deadline} | GW{gw} | "
              f"{'PRIMARY' if label == 'deadline' else 'secondary'} |")
    return path


def selftest():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    base = _os.path.join(d, "repo"); _os.makedirs(_os.path.join(base, "By Gameweek", "GW4"))
    pd.DataFrame({"match_id": ["26-27-prem-a-vs-b"], "finished": [False]}).to_csv(
        _os.path.join(base, "By Gameweek", "GW4", "matches.csv"), index=False)
    pd.DataFrame({"id": [3, 4], "name": ["Gameweek 3", "Gameweek 4"],
                  "deadline_time": ["2026-09-05T12:30:00+00:00",
                                    "2026-09-12T12:30:00+00:00"],
                  "finished": [True, False], "is_current": [True, False],
                  "is_next": [False, True]}).to_csv(
        _os.path.join(base, "gameweek_summaries.csv"), index=False)
    bp = _os.path.join(d, "board.csv")
    pd.DataFrame({"player_code": [1, 2], "player": ["A", "B"], "pos": ["MID", "FWD"],
                  "team": ["X", "Y"], "cost": [5.0, 9.0], "gw": [4, 4],
                  "mean": [4.0, 6.0], "blended": [4.2, 6.1], "src": ["model"] * 2,
                  "sd": [3.0, 4.0], "p5": [0.0, 1.0], "median": [4.0, 6.0],
                  "p95": [11.0, 14.0]}).to_csv(bp, index=False)
    out = _os.path.join(d, "pred"); _os.makedirs(out)

    g = gameweeks(base=base)
    assert next_gw(g) == 4, next_gw(g)

    # inside the 24h window, like the GW1 lock which was taken ~6 hours out
    before = _dt.datetime(2026, 9, 12, 6, 0, tzinfo=_dt.timezone.utc)
    after = _dt.datetime(2026, 9, 13, tzinfo=_dt.timezone.utc)

    # a dry run writes nothing
    assert lock(board_path=bp, out_dir=out, base=base, now=before, dry_run=True,
                verbose=False) is None
    assert not _os.listdir(out), "dry run must not write"

    p = lock(board_path=bp, out_dir=out, base=base, now=before, verbose=False)
    assert _os.path.basename(p) == "gw4_board_locked_2026-09-12_deadline.csv", p
    got = pd.read_csv(p)
    assert len(got) == 2 and "player_code" in got.columns
    assert set(got["gw"]) == {4}, "only the locked gameweek may be in the file"

    # never regenerated
    try:
        lock(board_path=bp, out_dir=out, base=base, now=before, verbose=False)
        raise AssertionError("must refuse to overwrite an existing lock")
    except SystemExit:
        pass

    # THE guard: past the deadline it refuses outright
    try:
        lock(board_path=bp, out_dir=out, base=base, now=after, verbose=False)
        raise AssertionError("must refuse to lock after the deadline")
    except SystemExit:
        pass
    # ...and --force-late names the file so it cannot pass as a prediction
    pl = lock(board_path=bp, out_dir=out, base=base, now=after, force_late=True,
              verbose=False)
    assert "LATE_not_a_prediction" in _os.path.basename(pl), pl

    # a `deadline` lock taken days out is refused; `early` at the same moment is fine
    far = _dt.datetime(2026, 9, 8, tzinfo=_dt.timezone.utc)   # ~4.5 days out
    try:
        lock(board_path=bp, out_dir=out, base=base, now=far, verbose=False)
        raise AssertionError("a primary lock days from the deadline must be refused")
    except SystemExit:
        pass
    pe = lock(board_path=bp, out_dir=out, base=base, now=far, label="early",
              verbose=False)
    assert "_early" in _os.path.basename(pe), pe
    # ...and --anyway overrides it
    pa = lock(board_path=bp, out_dir=out, base=base, now=far, anyway=True, verbose=False)
    assert "_deadline" in _os.path.basename(pa), pa

    # a gameweek the board does not carry is refused, not silently written empty
    try:
        lock(gw=5, board_path=bp, out_dir=out, base=base, now=before, verbose=False)
        raise AssertionError("must refuse a gameweek absent from the board")
    except SystemExit:
        pass
    # --- auto: quiet on the days there is nothing to do, and never raises ---
    out2 = _os.path.join(d, "pred2"); _os.makedirs(out2)
    far = _dt.datetime(2026, 9, 8, tzinfo=_dt.timezone.utc)      # outside the window
    assert auto(out_dir=out2, base=base, now=far, verbose=False) is None
    assert not _os.listdir(out2), "auto must write nothing outside the window"
    past = _dt.datetime(2026, 9, 20, tzinfo=_dt.timezone.utc)    # deadline long gone
    assert auto(out_dir=out2, base=base, now=past, verbose=False) is None
    assert not _os.listdir(out2), "auto must never write a post-deadline lock"
    # and it treats an existing deadline lock as done
    open(_os.path.join(out2, "gw4_board_locked_2026-09-12_deadline.csv"), "w").close()
    inside = _dt.datetime(2026, 9, 12, 6, 0, tzinfo=_dt.timezone.utc)
    assert auto(out_dir=out2, base=base, now=inside, verbose=False) is None, (
        "auto must not overwrite an existing deadline lock")

    shutil.rmtree(d, ignore_errors=True)
    print("SELFTEST OK: picks is_next, dry run writes nothing, refuses to overwrite, "
          "refuses to lock after the deadline, refuses a PRIMARY lock taken days early "
          "while allowing `early` at the same moment, --force-late is named so it cannot "
          "pass as a prediction, an absent gameweek is refused not written empty, and --auto is "
          "silent outside the window, after the deadline, and when already locked.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None, help="default: the next gameweek")
    ap.add_argument("--label", default="deadline",
                    help="deadline (primary, scored) or early (secondary insurance)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-late", action="store_true",
                    help="write even though the deadline has passed; the file is renamed "
                         "so it can never be mistaken for a prediction")
    ap.add_argument("--anyway", action="store_true",
                    help="write a `deadline` lock even though the deadline is far off")
    ap.add_argument("--auto", action="store_true",
                    help="for a daily scheduler: lock only if the deadline is close and "
                         "the gameweek is not already locked; exit 0 quietly otherwise")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); _sys.exit(0)
    if a.auto:
        auto(); _sys.exit(0)
    lock(gw=a.gw, label=a.label, dry_run=a.dry_run, force_late=a.force_late,
         anyway=a.anyway)
