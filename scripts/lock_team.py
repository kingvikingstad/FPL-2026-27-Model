from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
lock_team.py — preserve the model's AND the market's per-fixture lambda before a deadline.
=========================================================================================
`lock_board.py` preserves the PLAYER board. It never preserved the team layer, so the
model's pre-deadline per-fixture lambda for GW2-GW4 is gone: today's
`team_projections_gw1_38.csv` rows for a played week are a posterior that has already
absorbed that match. That made the model-vs-market study
(docs/MODEL_VS_MARKET_PREREG_2026-09-16.md) unrunnable for any week not locked here,
and it is why GW5 is the first admissible gameweek.

Two files per lock, written from ONE build, so the means and the draws cannot disagree:

  predictions/gw{N}_team_locked_{date}_{label}.csv
      one row per team-fixture of GW N: model lam_for/against with p5/p95, the
      clean-sheet and match-outcome probabilities, the market's mkt_* columns as they
      stood at lock time (fixture_market: the last price at or before now), and the
      provenance the pre-registration requires — lock time, deadline, repo HEAD / dirty
      / diff hash, feed commit, draws and seed, and the flag vector.
  predictions/team_draws/gw{N}_team_draws_{date}_{label}.csv.gz
      the paired posterior draws (lam_home, lam_away) for every fixture, for the
      posterior-predictive secondary (prereg §9.6).

The name deliberately does NOT match `gw*_board_locked_*`, so `score_gw._locked()`,
`doctor` and `postgw_review` can never pick a team lock up as a board. Draws live in a
subdirectory for the same glob reason `crosswalks/` does.

IT INHERITS EVERY REFUSAL `lock_board` HAS, BY CALLING THE SAME CODE
--------------------------------------------------------------------
Deadlines come from `lock_board.gameweeks`, the `deadline` window from
`lock_board.DEADLINE_WINDOW_H`. It will not lock after the deadline (`--force-late` renames
the file `_LATE_not_a_prediction`), will not overwrite a lock, and will not write a
`deadline`-labelled lock days early (`--label early` is the insurance lock; `--anyway`
overrides). Which team lock the study uses is fixed in advance, like the board's: a
`_deadline` lock wins whenever one exists (prereg §12, 2026-09-16 entry).

FLAGS THAT MAKE A LOCK INADMISSIBLE ARE RECORDED, NOT REFUSED
-------------------------------------------------------------
`export_team_projections` hardcodes clubelo_weight=0.45 and never reads INJURY_IMPACT,
TEAM_OVERRIDES or SOLIO_MARKET, all of which `gw_board` applies to its team samples. With
any of them off-default the locked lambda would quietly differ from the board's (prereg
§0.3). The lock is still written — a lock that exists can be excluded, a lock that does
not exist cannot be recovered — but its `admissible` column says why it is not.
Flag truthiness mirrors `gw_board._flag` exactly.

Run:  python scripts/lock_team.py --gw 5 --label early      # insurance lock, days out
      python scripts/lock_team.py                            # primary, inside the window
      python scripts/lock_team.py --selftest
`lock_board.py --auto` calls this in the same run, straight after its board lock.
"""
import argparse
import datetime as _dt
import hashlib
import subprocess
import pandas as pd

import lock_board as lb

LOCK_COLS = ["gw", "team", "opponent", "is_home", "lam_for", "lam_against",
             "lam_for_p5", "lam_for_p95", "lam_against_p5", "lam_against_p95",
             "p_clean_sheet", "p_clean_sheet_plugin", "p_win", "p_draw", "p_loss",
             "mkt_lam_for", "mkt_lam_against", "mkt_p_clean_sheet", "mkt_source",
             "mkt_as_of"]
# The prereg §10 flag vector. Recorded raw ("" when unset) so nothing is inferred later.
FLAGS = ["MARKET_ODDS", "MARKET_WEIGHT", "ELO_WEIGHT", "INSEASON", "INSEASON_UPTO",
         "INSEASON_W_MATCH", "INSEASON_W_PROMOTED", "FPL_TRAVEL", "INJURY_IMPACT",
         "TEAM_OVERRIDES", "SOLIO_MARKET"]


def draws_dir(out_dir):
    return _os.path.join(out_dir, "team_draws")


def admissibility(env):
    """Reasons this lock's lambda would not match the board's (prereg §4). Empty = ok.

    Truthiness copied from `gw_board`: INJURY_IMPACT / TEAM_OVERRIDES are ON unless
    "off"/"0" (`_flag(name, "off")`), SOLIO_MARKET is ON only for on/1/true, ELO_WEIGHT is
    a float defaulting to 0.45.
    """
    bad = []
    try:
        if float(env.get("ELO_WEIGHT", "0.45")) != 0.45:
            bad.append(f"ELO_WEIGHT={env.get('ELO_WEIGHT')}")
    except ValueError:
        bad.append(f"ELO_WEIGHT={env.get('ELO_WEIGHT')!r} unparseable")
    for name in ("INJURY_IMPACT", "TEAM_OVERRIDES"):
        if str(env.get(name, "off")).lower() not in ("off", "0"):
            bad.append(f"{name}={env.get(name)}")
    if str(env.get("SOLIO_MARKET", "off")).lower() in ("on", "1", "true"):
        bad.append(f"SOLIO_MARKET={env.get('SOLIO_MARKET')}")
    return bad


def provenance(root=None, feed=None):
    """Repo and feed state at lock time. Every field degrades to "" rather than raising:
    a lock missing a hash is still a lock, and the empty cell says it is unknown."""
    root = root or config.ROOT
    feed = feed or config.REPO

    def git(args, cwd):
        try:
            r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, timeout=60)
            return r.stdout if r.returncode == 0 else None
        except Exception:                                         # noqa: BLE001
            return None

    head = git(["rev-parse", "HEAD"], root)
    status = git(["status", "--porcelain"], root)
    diff = git(["diff", "HEAD", "--binary"], root)
    untracked = git(["ls-files", "--others", "--exclude-standard"], root)
    feed_head = git(["rev-parse", "HEAD"], feed)
    digest = ""
    if diff is not None and untracked is not None:
        digest = hashlib.sha256(diff + b"\0" + untracked).hexdigest()
    return {
        "repo_head_sha": head.decode().strip() if head else "",
        "repo_dirty": "" if status is None else int(bool(status.strip())),
        "repo_diff_sha256": digest,
        "feed_commit_sha": feed_head.decode().strip() if feed_head else "",
    }


def build_team(gw):
    """The real build: `export_team_projections.build`, window pinned to cover `gw`."""
    import export_team_projections as etp
    etp.GW_HI = max(int(gw), int(etp.GW_HI))
    _T, G = etp.build(keep_draws_gw=int(gw))
    return G


def lock_team(gw=None, label="deadline", out_dir=None, now=None, season=lb.SEASON,
              base=None, force_late=False, anyway=False, dry_run=False, build=None,
              env=None, prov=None, verbose=True):
    """Write the team lock and its draws. Returns the lock path, or None on a dry run."""
    out_dir = out_dir or config.PREDICTIONS
    now = now or _dt.datetime.now(_dt.timezone.utc)
    env = dict(_os.environ) if env is None else env

    g = lb.gameweeks(season, base=base)
    gw = int(gw) if gw is not None else lb.next_gw(g)
    row = g[g["id"] == gw]
    if not len(row):
        raise SystemExit(f"GW{gw} is not in gameweek_summaries.csv")
    deadline = row["deadline"].iloc[0]
    late = pd.notna(deadline) and now >= deadline.to_pydatetime()
    hrs = ((deadline.to_pydatetime() - now).total_seconds() / 3600.0
           if pd.notna(deadline) else float("nan"))
    if verbose:
        print("=" * 74)
        print(f"TEAM LOCK GW{gw}  ({label})")
        print("=" * 74)
        print(f"  deadline  {deadline}")
        print(f"  now       {now:%Y-%m-%d %H:%M:%S%z}")

    # The same three refusals as lock_board, for the same reasons (see its docstring).
    if label == "deadline" and not late and pd.notna(deadline) and not anyway \
            and hrs > lb.DEADLINE_WINDOW_H:
        raise SystemExit(
            f"REFUSING: a `deadline` team lock {hrs:.0f}h before the deadline would become "
            f"the study's primary lock with a stale market price. Use `--label early` for "
            f"an insurance lock now; the primary is written inside "
            f"{lb.DEADLINE_WINDOW_H:.0f}h. Pass --anyway to override.")
    if late and not force_late:
        raise SystemExit(
            f"REFUSING: GW{gw}'s deadline ({deadline}) has passed. A team lock written now "
            f"is not a forecast. Nothing was written. --force-late writes a file named so "
            f"it can never be mistaken for one.")

    stamp = now.strftime("%Y-%m-%d")
    suffix = f"{label}_LATE_not_a_prediction" if late else label
    name = f"gw{gw}_team_locked_{stamp}_{suffix}.csv"
    dname = f"gw{gw}_team_draws_{stamp}_{suffix}.csv.gz"
    path = _os.path.join(out_dir, name)
    dpath = _os.path.join(draws_dir(out_dir), dname)
    for p in (path, dpath):
        if _os.path.exists(p):
            raise SystemExit(f"REFUSING: {_os.path.basename(p)} already exists. Locks are "
                             f"never regenerated — see predictions/README.md.")

    G = (build or build_team)(gw)
    missing = [c for c in LOCK_COLS if c not in G.columns]
    if missing:
        raise SystemExit(f"REFUSING: the team build lacks {missing}; nothing written.")
    sel = G[G["gw"] == gw][LOCK_COLS].copy()
    if not len(sel):
        raise SystemExit(f"REFUSING: the team build carries no GW{gw} rows; nothing written.")
    draws = G.attrs.get("draws")
    if draws is None or not len(draws):
        raise SystemExit("REFUSING: the team build returned no draws; nothing written.")

    bad = admissibility(env)
    meta = {"locked_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "deadline_utc": "" if pd.isna(deadline) else
            deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "label": label,
            **(prov if prov is not None else provenance()),
            "draws_S": G.attrs.get("draws_S", ""), "seed": G.attrs.get("seed", ""),
            **{f: env.get(f, "") for f in FLAGS},
            "admissible": "ok" if not bad else "inadmissible_flags: " + "; ".join(bad)}
    out = sel.assign(**meta)

    n_mkt = int(out["mkt_lam_for"].notna().sum())
    if verbose:
        print(f"  {len(out)} team-fixtures, market lambda on {n_mkt}"
              + ("" if n_mkt == len(out) else "  ** market missing on some fixtures — "
                                               "those are excluded by prereg §4 **"))
        print(f"  draws: {len(draws)} rows (S={meta['draws_S']})")
        print(f"  admissible: {meta['admissible']}")
        print(f"  repo {meta['repo_head_sha'][:10]} dirty={meta['repo_dirty']}  "
              f"feed {str(meta['feed_commit_sha'])[:10]}")
    if dry_run:
        if verbose:
            print(f"\n  DRY RUN — would write {name} and team_draws/{dname}")
        return None

    _os.makedirs(draws_dir(out_dir), exist_ok=True)
    # Draws first: if the process dies between the two writes, a draws file with no lock
    # is harmless, while a lock with no draws would silently lose the §9.6 secondary.
    draws.round(5).to_csv(dpath, index=False, compression="gzip")
    out.to_csv(path, index=False)
    if verbose:
        print(f"\n  wrote {path}")
        print(f"  wrote {dpath}")
    return path


def selftest():
    import tempfile
    import shutil
    import numpy as np
    try:
        _sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    d = tempfile.mkdtemp()
    base = _os.path.join(d, "feed")
    _os.makedirs(base)
    pd.DataFrame({"id": [5, 4], "deadline_time": ["2026-09-18T17:30:00Z",
                                                  "2026-09-12T12:30:00Z"],
                  "finished": [False, True], "is_next": [True, False],
                  "is_current": [False, True]}).to_csv(
        _os.path.join(base, "gameweek_summaries.csv"), index=False)
    dl = _dt.datetime(2026, 9, 18, 17, 30, tzinfo=_dt.timezone.utc)
    far = dl - _dt.timedelta(hours=54)
    inside = dl - _dt.timedelta(hours=6)
    past = dl + _dt.timedelta(hours=1)
    prov = {"repo_head_sha": "abc", "repo_dirty": 1, "repo_diff_sha256": "f00",
            "feed_commit_sha": "def"}
    calls = []

    def fake_build(gw, market=True):
        calls.append(gw)
        rows = []
        for gw_, h, a in ((5, "Arsenal", "Brighton"), (5, "Chelsea", "Brentford"),
                          (6, "Leeds", "Hull")):
            for team, opp, ih in ((h, a, 1), (a, h, 0)):
                rows.append(dict(gw=gw_, team=team, opponent=opp, is_home=ih,
                                 **{c: 1.2 for c in LOCK_COLS[4:15]},
                                 mkt_lam_for=1.3 if market else np.nan,
                                 mkt_lam_against=1.1 if market else np.nan,
                                 mkt_p_clean_sheet=0.33 if market else np.nan,
                                 mkt_source="solio" if market else np.nan,
                                 mkt_as_of="2026-09-16T11:04Z" if market else np.nan))
        G = pd.DataFrame(rows)
        G.attrs["draws"] = pd.DataFrame({"gw": 5, "home": "Arsenal", "away": "Brighton",
                                         "draw": range(4), "lam_home": 1.5,
                                         "lam_away": 1.0})
        G.attrs["draws_S"], G.attrs["seed"] = 4, 7
        return G

    K = dict(base=base, build=fake_build, prov=prov, verbose=False, env={})
    try:
        # 1. a `deadline` lock days out is refused; `early` at the same moment is written,
        #    with both files, the GW's rows only, and full provenance
        o = _os.path.join(d, "p1")
        try:
            lock_team(gw=5, label="deadline", out_dir=o, now=far, **K)
            raise AssertionError("a deadline team lock 54h out must be refused")
        except SystemExit as e:
            assert "early" in str(e)
        assert calls == [], "a refusal must happen BEFORE the expensive build"
        p = lock_team(gw=5, label="early", out_dir=o, now=far, **K)
        assert _os.path.basename(p) == "gw5_team_locked_2026-09-16_early.csv", p
        t = pd.read_csv(p)
        assert len(t) == 4 and set(t["gw"]) == {5}, t
        for c in LOCK_COLS + ["locked_at_utc", "deadline_utc", "repo_head_sha",
                              "repo_dirty", "repo_diff_sha256", "feed_commit_sha",
                              "draws_S", "seed", "admissible"] + FLAGS:
            assert c in t.columns, c
        assert set(t["admissible"]) == {"ok"} and t["deadline_utc"].iloc[0] == \
            "2026-09-18T17:30:00Z"
        dr = pd.read_csv(_os.path.join(o, "team_draws",
                                       "gw5_team_draws_2026-09-16_early.csv.gz"))
        assert len(dr) == 4 and {"lam_home", "lam_away", "draw"} <= set(dr.columns)
        # it can never be globbed as a board lock
        import glob
        assert not glob.glob(_os.path.join(o, "gw5_board_locked_*.csv"))

        # 2. never overwrites
        try:
            lock_team(gw=5, label="early", out_dir=o, now=far, **K)
            raise AssertionError("must not overwrite a team lock")
        except SystemExit as e:
            assert "already exists" in str(e)

        # 3. inside the window the primary is written; after the deadline it is refused,
        #    and --force-late names the file so it cannot pass as a forecast
        p = lock_team(gw=5, label="deadline", out_dir=o, now=inside, **K)
        assert p.endswith("gw5_team_locked_2026-09-18_deadline.csv"), p
        calls.clear()
        try:
            lock_team(gw=5, out_dir=_os.path.join(d, "p3"), now=past, **K)
            raise AssertionError("must refuse after the deadline")
        except SystemExit as e:
            assert "passed" in str(e)
        assert calls == []
        p = lock_team(gw=5, out_dir=_os.path.join(d, "p3"), now=past, force_late=True, **K)
        assert "_LATE_not_a_prediction" in p

        # 4. off-default flags: written, but marked inadmissible, with gw_board's truthiness
        assert admissibility({}) == []
        assert admissibility({"ELO_WEIGHT": "0.45", "INJURY_IMPACT": "off",
                              "TEAM_OVERRIDES": "0", "SOLIO_MARKET": "off"}) == []
        assert admissibility({"ELO_WEIGHT": "0.5"}) == ["ELO_WEIGHT=0.5"]
        assert admissibility({"INJURY_IMPACT": "yes"}) == ["INJURY_IMPACT=yes"]
        assert admissibility({"SOLIO_MARKET": "yes"}) == [], "gw_board: on/1/true only"
        assert admissibility({"SOLIO_MARKET": "TRUE"}) == ["SOLIO_MARKET=TRUE"]
        p = lock_team(gw=5, label="early", out_dir=_os.path.join(d, "p4"), now=far,
                      **dict(K, env={"ELO_WEIGHT": "0.5", "SOLIO_MARKET": "on"}))
        t = pd.read_csv(p)
        assert t["admissible"].iloc[0].startswith("inadmissible_flags"), t["admissible"]
        assert str(t["ELO_WEIGHT"].iloc[0]) == "0.5"

        # 5. a build that lost its columns, rows or draws writes nothing
        def no_mkt_cols(gw):
            G = fake_build(gw)
            G2 = G.drop(columns=["mkt_source"])
            G2.attrs = G.attrs
            return G2
        o5 = _os.path.join(d, "p5")
        for bad_build in (no_mkt_cols, lambda gw: fake_build(99)[lambda x: x["gw"] == 6]):
            try:
                lock_team(gw=5, label="early", out_dir=o5, now=far, **dict(K, build=bad_build))
                raise AssertionError("a broken build must refuse")
            except SystemExit:
                pass
        assert not _os.path.exists(o5) or not _os.listdir(o5)
        # ...but missing market prices are written (they are excluded later, not now)
        p = lock_team(gw=5, label="early", out_dir=_os.path.join(d, "p6"), now=far,
                      **dict(K, build=lambda gw: fake_build(gw, market=False)))
        assert pd.read_csv(p)["mkt_lam_for"].isna().all()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("SELFTEST OK: refuses a deadline team lock days out before building, writes an "
          "early one with draws and full provenance, never matches the board glob, never "
          "overwrites, refuses after the deadline and names --force-late so it cannot pass, "
          "records off-default flags as inadmissible with gw_board's truthiness, refuses a "
          "broken build, and keeps a lock whose market prices are missing.")
    return 0


if __name__ == "__main__":
    try:
        _sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw", type=int, default=None, help="default: the next gameweek")
    ap.add_argument("--label", default="deadline",
                    help="deadline (primary) or early (insurance)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-late", action="store_true")
    ap.add_argument("--anyway", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _sys.exit(selftest())
    lock_team(gw=a.gw, label=a.label, dry_run=a.dry_run, force_late=a.force_late,
              anyway=a.anyway)
