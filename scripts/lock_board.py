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

So `--auto` runs several times a day and acts only when the state says to: the next
gameweek's deadline is inside `DEADLINE_WINDOW_H`, and no lock for it exists yet. Every
other run prints one line and exits 0, so a scheduler sees success rather than a daily
failure. This is the same trigger `postgw_review --auto` uses, for the same reason.

INSIDE THE WINDOW IT WAITS FOR TEAM NEWS, UP TO A POINT
--------------------------------------------------------
"Lock on the first in-window run" meant more scheduler slots made the lock EARLIER, not
better-informed: GW4's primary was taken at 11:31 UTC on the Friday, 25h out, by an
on-launch catch-up, while fpl.page still returned 404 for GW4. The one input the lock
exists to capture was the one it reliably missed. So inside the window:

  * a predicted-XI source exists for the gameweek  ->  lock now, WITH team news;
  * none exists, and the deadline is more than `LAST_CHANCE_H` away  ->  SKIP, exit 0,
    write nothing; a later run tries again;
  * none exists, and the deadline is inside `LAST_CHANCE_H`  ->  lock anyway, WITHOUT
    team news, and say so loudly. An unlocked gameweek can never be scored, which is
    strictly worse than a lock missing its best input.

`LAST_CHANCE_H` is derived from the scheduler's slots, see the constant. Every lock
records how many predicted-XI sources it carried in a `team_news_sources` column, so
whether a lock had team news is read off the file, never inferred later.

When it locks it does two things, in this order:

  1. FETCH TEAM NEWS for that gameweek (`fplpage`), because the predicted XI is the input
     the project's own scoring identifies as its largest error source — correlation
     roughly halves conditioned on appearing, and GW1 lost 52.6 projected points to 13
     availability misses. The fetch is also what decides between waiting and locking,
     so it runs on every in-window run until the lock is taken.
  2. REBUILD the board, because the failure this guards against is not locking late, it
     is locking a board built before the last data pull. `gw_board` folds the predicted XI
     it just fetched into the projection, which is why the order is fixed. The rebuild is
     pinned to the gameweek being locked (`PRED_XI_GW`), so the `team_news_sources` count
     describes the XI the board actually folded, not a neighbouring week's.

The rebuild is the expensive half and runs only on the run that actually locks.

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
# Inside the window, `--auto` waits for team news until the deadline is this close, then
# locks without it. It is pinned by the scheduler's slots (fpl-lock-board, cron
# `0 7,9,17,22 * * *` in America/New_York local time), measured against all 38 26/27
# deadlines, and has to sit strictly between two numbers:
#
#   ABOVE 10.5h. 22 of the 38 deadlines fall at 08:30 local on a Saturday, and the last
#   evening slot before them is Friday 22:00 — 10.5h out. Below that, the Friday 22:00 run
#   SKIPS and the only thing left before kick-off is a 07:00 Saturday run; if the machine
#   is off then, nothing locks. On the old 09/17/22 cron an 8-10h cutoff left those 22
#   gameweeks with no unconditional run at all.
#   BELOW 15.5h. Friday and Wednesday deadlines fall at 13:30 local, and the Thursday or
#   Tuesday 22:00 slot before them is 15.5h out. Above that, it locks without news before
#   the morning of deadline day — which is when fpl.page dated GW1, GW2 and GW3's articles.
#
# 14.5 keeps an hour of margin on both sides and gives every deadline at least TWO slots
# inside it (Fri 22:00 + Sat 07:00 for the 08:30s; Fri 17:00 + 22:00 for the 06:00 and
# 07:00 kick-offs; 07:00 + 09:00 on the day for the 13:30s), so one missed run cannot cost
# a gameweek, and the on-launch catch-up is a third chance. The scheduler's jitter (~8 min)
# only ever makes a run later, never earlier. If the cron changes, re-derive this.
LAST_CHANCE_H = 14.5
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

    It never raises, and its count is what `auto` waits on: zero sources means wait, unless
    the deadline is inside `LAST_CHANCE_H`, where a lock WITHOUT team news still beats no
    lock at all — a gameweek that goes unlocked can never be scored. Whether the lock
    carries team news changes what it is worth, so the lock records it in the file.

    fpl.page dated GW1-GW3's articles on deadline day itself, so a 404 is the NORMAL
    answer the day before and is not an error. Returns the number of predicted-XI sources
    available for `gw` afterwards, counting every feed `predicted_xi` can see — a manually
    saved Rotowire or FFS file counts just as much as the scrape.
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
                print(f"  team news: no GW{gw} preview published yet (404) — normal "
                      f"before deadline day")
            else:
                print(f"  team news: fetch failed ({type(e).__name__}: {msg[:120]})")
    try:
        import predicted_xi as pxi
        n = len(pxi.sources(gw))
    except Exception:
        n = 0
    if verbose:
        print(f"  team news: {n} predicted-XI source(s) on disk for GW{gw}")
    return n


def team_news_sources(gw, board_path, xi_dir=None):
    """How many predicted-XI sources for `gw` existed when the board was last built.

    What a manual `lock` records in `team_news_sources` (`auto` passes the count it
    fetched instead, just before its own rebuild). Only files no newer than the board
    count: a preview saved after the board was built is not in it. Overwriting a source
    after a build therefore UNDER-counts, which is the safe direction. None if the
    sources cannot be read, which is written as an empty cell — unknown, not zero.
    """
    try:
        import predicted_xi as pxi
        built = _os.path.getmtime(board_path)
        return sum(1 for s in pxi.sources(gw, data_dir=xi_dir)
                   if _os.path.getmtime(s["path"]) <= built)
    except Exception:                                             # noqa: BLE001
        return None


def rebuild_board(gw=None, verbose=True):
    """Re-run the board so the lock preserves a current projection, not a stale one.

    PYTHONIOENCODING is forced because a child writing to a pipe on Windows gets cp1252
    and dies on the first non-ASCII player name — the trap documented in CLAUDE.md and
    handled the same way in `test_all`. PRED_XI_GW is pinned to the gameweek being
    locked: `gw_board` otherwise derives its own team-news week, and if that ever
    disagreed with `is_next` the lock would claim team news the board never folded.
    """
    import subprocess
    script = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "gw_board.py")
    env = dict(_os.environ, PYTHONIOENCODING="utf-8")
    if gw is not None:
        env["PRED_XI_GW"] = str(int(gw))
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


def auto(out_dir=None, season=SEASON, base=None, now=None, verbose=True,
         board_path=None, fetch=fetch_lineups, rebuild=rebuild_board):
    """Lock the next gameweek IF its deadline is close, it is not already locked, and
    either team news exists or the deadline is inside `LAST_CHANCE_H`.

    Returns the path written, or None. Never raises on the quiet paths — outside the
    window, already locked, waiting for team news: a scheduler must see exit 0 on runs
    where there is nothing to do. A failed rebuild still raises, so nothing is written.
    `fetch` and `rebuild` are injectable so the selftest never touches the network or
    the real board.
    """
    out_dir = out_dir or config.PREDICTIONS
    # With the real clock, `lock` reads the clock again at write time instead of reusing
    # this one: a fetch plus a rebuild takes ~15 minutes, and a run that starts just
    # before the deadline must not write a lock that is timestamped before it.
    live_clock = now is None
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
    existing = _glob.glob(_os.path.join(out_dir, f"gw{gw}_board_locked_*_deadline*.csv"))
    if hrs <= 0:
        # `is_next` only moves on when the feed refreshes, so a run just after a deadline
        # can still see it. Say "missed" only if it really was.
        if existing:
            print(f"[auto] GW{gw}'s deadline passed {abs(hrs):.1f}h ago; it was locked "
                  f"({_os.path.basename(existing[-1])}). Nothing to do.")
        else:
            print(f"[auto] GW{gw}'s deadline passed {abs(hrs):.1f}h ago and it was never "
                  f"locked. Nothing can be written now — that gameweek is unscoreable.")
        return None
    if hrs > DEADLINE_WINDOW_H:
        print(f"[auto] GW{gw} deadline in {hrs:.1f}h, outside the "
              f"{DEADLINE_WINDOW_H:.0f}h window. Nothing to do.")
        return None
    if existing:
        print(f"[auto] GW{gw} already has a deadline lock "
              f"({_os.path.basename(existing[-1])}). Nothing to do.")
        return None
    # Order matters: team news first, THEN the board, because `gw_board` folds the
    # predicted XI into the projection it is about to build. Fetching after the rebuild
    # would lock a board that never saw it.
    n_xi = fetch(gw, verbose=verbose)
    if not n_xi and hrs > LAST_CHANCE_H:
        print(f"[auto] GW{gw} deadline in {hrs:.1f}h, unlocked, no team news yet — "
              f"WAITING. A later run locks with it, or without it once inside "
              f"{LAST_CHANCE_H:g}h. Nothing written.")
        return None
    if n_xi:
        print(f"[auto] GW{gw} deadline in {hrs:.1f}h, unlocked, team news found "
              f"({n_xi} source(s)) — locking now.")
    else:
        print("*" * 74)
        print(f"[auto] LAST CHANCE: GW{gw} deadline in {hrs:.1f}h and STILL NO TEAM NEWS.")
        print("       Locking WITHOUT it, because an unlocked gameweek can never be scored.")
        print("       This primary lock is missing the input worth most to the projection.")
        print("*" * 74)
    rebuild(gw=gw, verbose=verbose)
    path = lock(gw=gw, label="deadline", board_path=board_path, out_dir=out_dir,
                season=season, base=base, now=None if live_clock else now,
                verbose=verbose, team_news=n_xi)
    if verbose:
        print(f"\n[auto] locked GW{gw} "
              + (f"WITH team news ({n_xi} source(s))" if n_xi
                 else f"WITHOUT team news — none published {hrs:.1f}h before the deadline"))
    return path


def lock(gw=None, label="deadline", board_path=None, out_dir=None, season=SEASON,
         base=None, dry_run=False, force_late=False, now=None, verbose=True,
         anyway=False, team_news=None, xi_dir=None):
    """Write the lock. Returns the path written, or None on a dry run.

    `team_news` is the number of predicted-XI sources the board was built with, written
    to every row as `team_news_sources`; when not given it is counted from disk by
    `team_news_sources()` (from `xi_dir`, default the data dir).
    """
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
    if team_news is None:
        team_news = team_news_sources(gw, board_path, xi_dir=xi_dir)
    # Provenance travels IN the file: whether this lock saw team news changes what it is
    # worth, and a column cannot be separated from the board the way a log line can.
    out = sel[cols].assign(team_news_sources=pd.NA if team_news is None
                           else int(team_news))
    if verbose:
        print("  team news " + ("UNKNOWN (sources unreadable)" if team_news is None else
                                f"{int(team_news)} predicted-XI source(s)"
                                + ("" if team_news else "  ** NONE: no team news **")))

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
        news = ("team news unknown" if team_news is None else
                f"team news: {int(team_news)} source(s)" if team_news else "NO team news")
        print(f"| `{name}` | {now:%Y-%m-%d %H:%M} UTC, deadline {deadline} | GW{gw} | "
              f"{'PRIMARY' if label == 'deadline' else 'secondary'}; {news} |")
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
    # predicted-XI files for the test live here, never in the real data dir
    xd = _os.path.join(d, "xi"); _os.makedirs(xd)
    kw = dict(board_path=bp, out_dir=out, base=base, xi_dir=xd, verbose=False)

    g = gameweeks(base=base)
    assert next_gw(g) == 4, next_gw(g)

    # inside the 24h window, like the GW1 lock which was taken ~6 hours out
    before = _dt.datetime(2026, 9, 12, 6, 0, tzinfo=_dt.timezone.utc)
    after = _dt.datetime(2026, 9, 13, tzinfo=_dt.timezone.utc)

    # a dry run writes nothing
    assert lock(now=before, dry_run=True, **kw) is None
    assert not _os.listdir(out), "dry run must not write"

    p = lock(now=before, **kw)
    assert _os.path.basename(p) == "gw4_board_locked_2026-09-12_deadline.csv", p
    got = pd.read_csv(p)
    assert len(got) == 2 and "player_code" in got.columns
    assert set(got["gw"]) == {4}, "only the locked gameweek may be in the file"
    assert set(got["team_news_sources"]) == {0}, "no XI on disk must read 0, not blank"

    # never regenerated
    try:
        lock(now=before, **kw)
        raise AssertionError("must refuse to overwrite an existing lock")
    except SystemExit:
        pass

    # THE guard: past the deadline it refuses outright
    try:
        lock(now=after, **kw)
        raise AssertionError("must refuse to lock after the deadline")
    except SystemExit:
        pass
    # ...and --force-late names the file so it cannot pass as a prediction
    pl = lock(now=after, force_late=True, **kw)
    assert "LATE_not_a_prediction" in _os.path.basename(pl), pl

    # a manual lock counts only the previews that predate the board it copies
    def xi(src, gw=4):
        f = _os.path.join(xd, f"predicted_xi_gw{gw}_{src}.csv")
        pd.DataFrame({"team": ["X"], "player": ["A"], "role": ["start"]}).to_csv(
            f, index=False)
        return f
    built = _os.path.getmtime(bp)
    _os.utime(xi("rotowire"), (built - 60, built - 60))     # saved before the build
    _os.utime(xi("fplpage"), (built + 60, built + 60))      # saved after it: not in it
    _os.utime(xi("rotowire", gw=5), (built - 60, built - 60))
    assert team_news_sources(4, bp, xi_dir=xd) == 1, team_news_sources(4, bp, xi_dir=xd)
    assert team_news_sources(3, bp, xi_dir=xd) == 0, "another week's XI is not this week's"

    # a `deadline` lock taken days out is refused; `early` at the same moment is fine
    far = _dt.datetime(2026, 9, 8, tzinfo=_dt.timezone.utc)   # ~4.5 days out
    try:
        lock(now=far, **kw)
        raise AssertionError("a primary lock days from the deadline must be refused")
    except SystemExit:
        pass
    pe = lock(now=far, label="early", **kw)
    assert "_early" in _os.path.basename(pe), pe
    assert set(pd.read_csv(pe)["team_news_sources"]) == {1}
    # ...and --anyway overrides it
    pa = lock(now=far, anyway=True, **kw)
    assert "_deadline" in _os.path.basename(pa), pa

    # a gameweek the board does not carry is refused, not silently written empty
    try:
        lock(gw=5, now=before, **kw)
        raise AssertionError("must refuse a gameweek absent from the board")
    except SystemExit:
        pass

    # --- auto: stubs stand in for the network fetch and the real rebuild ---
    import contextlib, io
    calls = []

    def no_news(gw, verbose=True):
        calls.append("fetch"); return 0

    def news(gw, verbose=True):
        calls.append("fetch"); return 2

    def rebuilt(gw=None, verbose=True):
        calls.append(f"rebuild GW{gw}"); return True

    def broken(gw=None, verbose=True):
        calls.append(f"rebuild GW{gw}")
        raise SystemExit("REFUSING: the board rebuild failed")

    A = dict(base=base, board_path=bp, verbose=False, rebuild=rebuilt)

    def fresh(name):
        o = _os.path.join(d, name); _os.makedirs(o); calls.clear()
        return o

    # quiet on the runs there is nothing to do, never raises, and does not even fetch
    out2 = fresh("pred2")
    assert auto(out_dir=out2, now=far, fetch=news, **A) is None
    assert not _os.listdir(out2), "auto must write nothing outside the window"
    past = _dt.datetime(2026, 9, 20, tzinfo=_dt.timezone.utc)    # deadline long gone
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert auto(out_dir=out2, now=past, fetch=news, **A) is None
    assert not _os.listdir(out2), "auto must never write a post-deadline lock"
    assert "never locked" in buf.getvalue()
    # and it treats an existing deadline lock as done
    open(_os.path.join(out2, "gw4_board_locked_2026-09-12_deadline.csv"), "w").close()
    inside = _dt.datetime(2026, 9, 12, 6, 0, tzinfo=_dt.timezone.utc)
    assert auto(out_dir=out2, now=inside, fetch=news, **A) is None, (
        "auto must not overwrite an existing deadline lock")
    # ...including after the deadline, where it must not report a miss that did not happen
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        auto(out_dir=out2, now=past, fetch=news, **A)
    assert "never locked" not in buf.getvalue(), buf.getvalue()
    assert calls == [], f"the quiet paths must not fetch or rebuild: {calls}"

    # inside the window: wait for team news, lock with it, or lock without it at the end
    dl = _dt.datetime(2026, 9, 12, 12, 30, tzinfo=_dt.timezone.utc)
    waiting = dl - _dt.timedelta(hours=LAST_CHANCE_H + 6)     # in the window, pre-cutoff
    last = dl - _dt.timedelta(hours=LAST_CHANCE_H - 4)        # inside the cutoff
    assert LAST_CHANCE_H + 6 < DEADLINE_WINDOW_H

    # 1. no team news and the cutoff not reached: SKIP. It fetched; it did not rebuild
    #    or write, and it returns quietly so the scheduler sees exit 0.
    o = fresh("wait")
    assert auto(out_dir=o, now=waiting, fetch=no_news, **A) is None
    assert not _os.listdir(o), "no news before the cutoff must write nothing"
    assert calls == ["fetch"], calls

    # 2. the same moment WITH team news: lock now, fetch before rebuild, rebuild pinned
    #    to the locked week, and the file itself records that it carried team news
    calls.clear()
    p = auto(out_dir=o, now=waiting, fetch=news, **A)
    assert _os.path.basename(p) == "gw4_board_locked_2026-09-11_deadline.csv", p
    assert calls == ["fetch", "rebuild GW4"], calls
    assert set(pd.read_csv(p)["team_news_sources"]) == {2}
    calls.clear()
    assert auto(out_dir=o, now=last, fetch=news, **A) is None and calls == [], (
        "once locked, later runs do nothing")

    # 3. last chance: still no team news inside the cutoff — lock anyway, as the primary,
    #    and the file says it has none
    o = fresh("last")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        p = auto(out_dir=o, now=last, fetch=no_news, **A)
    assert p is not None, "inside LAST_CHANCE_H it must lock even without team news"
    assert _os.path.basename(p) == "gw4_board_locked_2026-09-12_deadline.csv", p
    assert calls == ["fetch", "rebuild GW4"], calls
    assert set(pd.read_csv(p)["team_news_sources"]) == {0}, "a no-news lock must say so"
    assert "LAST CHANCE" in buf.getvalue() and "WITHOUT" in buf.getvalue()

    # 4. a failed rebuild refuses on both locking paths, and writes nothing
    for fetch_, when in ((news, waiting), (no_news, last)):
        o = fresh(f"broken_{fetch_.__name__}")
        try:
            auto(out_dir=o, now=when, fetch=fetch_, **dict(A, rebuild=broken))
            raise AssertionError("a failed rebuild must refuse")
        except SystemExit:
            pass
        assert not _os.listdir(o), "nothing may be written over a failed rebuild"

    shutil.rmtree(d, ignore_errors=True)
    print("SELFTEST OK: picks is_next, dry run writes nothing, refuses to overwrite, "
          "refuses to lock after the deadline, refuses a PRIMARY lock taken days early "
          "while allowing `early` at the same moment, --force-late is named so it cannot "
          "pass as a prediction, an absent gameweek is refused not written empty, "
          "team_news_sources counts only previews that predate the board; --auto is "
          "silent (and does not fetch) outside the window, after the deadline and when "
          "already locked, WAITS without team news before LAST_CHANCE_H, locks WITH it "
          "as soon as it exists, locks WITHOUT it inside LAST_CHANCE_H and says so in "
          "the file, and refuses over a failed rebuild on both paths.")


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
                    help="for a scheduler: lock only if the deadline is close, the "
                         "gameweek is not already locked, and team news exists or the "
                         "deadline is inside LAST_CHANCE_H; exit 0 quietly otherwise")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); _sys.exit(0)
    if a.auto:
        auto(); _sys.exit(0)
    lock(gw=a.gw, label=a.label, dry_run=a.dry_run, force_late=a.force_late,
         anyway=a.anyway)
