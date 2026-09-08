"""
manifest.py — the artifact graph: what exists, what made it, what it needs.
==========================================================================

`config.py` names every path. It does not say which script *writes* a path, which
paths that script *reads*, or what the file must contain to be usable. That
knowledge lived in prose comments and in the import structure of 55 modules, so
answering "the board looks wrong, what upstream of it changed?" meant reading the
code. This module makes the dependency graph machine-readable and checkable.

Three things it buys:

  1. STALENESS is derived, not guessed. A derived artifact older than any of its
     inputs is stale — no age threshold to tune, no judgment call. `downstream()`
     then says exactly what to rebuild after touching a node.
  2. DATA QUALITY is checked at the boundary the project does not control. On
     2026-08-21 `teams.csv` kept its `elo` column and went entirely NULL upstream;
     that NaN'd the team layer and surfaced forty lines later as "SVD did not
     converge". A required-columns + non-null-columns + row-floor check on the
     external feed catches that class at the door (see `scripts/doctor.py`).
  3. ORIENTATION is bounded. `--brief` prints the whole pipeline in ~40 lines.
     Reading it costs a fraction of tracing the imports, and it cannot drift from
     the paths because it is built from `config`.

What this is NOT: a knowledge graph, a database, or an agent framework. It is a
declared DAG over files, queried by `doctor.py`, `test_all.py` and by whoever is
working on the repo. A graph nothing queries is overhead; this one has callers.

Node kinds
----------
  external   upstream feed we do not control (FPL-Core-Insights). Checked, never built.
  committed  transcribed or pinned by hand, lives in data/, in git. Checked, never built.
  derived    regenerable from the nodes above by `producer`. Staleness applies.
  cache      a memoisation any consumer regenerates on demand. Its mtime carries no
             information about freshness, so it is never stale and never an input edge.
  orphan     READ by live code, written by NOTHING in the tree. Cannot be refreshed,
             and will silently rot as the season moves. Always a finding.
  locked     predictions/ — NOT regenerable, and never rebuilt. Existence only.

Run:  python src/manifest.py                              # the brief
      python src/manifest.py --check                      # the data-quality gate
      python src/manifest.py --downstream playerstats.csv # what to rebuild
      python src/manifest.py --selftest
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config

# The season the external feed is read for. core_insights.UPLOADS resolves the same
# directory; duplicated here only so the manifest can be inspected without importing
# the whole model layer — a broken module must not break the diagnostic that explains it.
SEASON = "2026-2027"
EXTERNAL = _os.path.join(config.REPO, SEASON)


class Node:
    """One artifact. `inputs` are node names; `producer` is a script path or None."""

    def __init__(self, name, path, kind, producer=None, inputs=(), columns=(),
                 nonnull=(), min_rows=0, note=""):
        self.name = name
        self.path = path
        self.kind = kind
        self.producer = producer
        self.inputs = tuple(inputs)
        self.columns = tuple(columns)      # required columns [VERIFIED against the files]
        self.nonnull = tuple(nonnull)      # columns that must not be entirely NULL
        self.min_rows = min_rows           # floor, not an expectation
        self.note = note

    def exists(self):
        return _os.path.exists(self.path)

    def mtime(self):
        return _os.path.getmtime(self.path) if self.exists() else None

    def __repr__(self):
        return f"<Node {self.name} ({self.kind})>"


def _ext(p):
    return _os.path.join(EXTERNAL, p)


def _out(p):
    return _os.path.join(config.OUTPUTS, p)


def _dat(p):
    return _os.path.join(config.DATA, p)


# ---------------------------------------------------------------------------
# The graph. Columns and row floors are read off the real files, not assumed;
# --selftest re-derives nothing, so re-check them by hand if the feed changes shape.
# ---------------------------------------------------------------------------
_NODES = [
    # --- external feed (olbauday/FPL-Core-Insights) -------------------------
    Node("players.csv", _ext("players.csv"), "external", min_rows=500,
         columns=("player_code", "player_id", "web_name", "team_code", "position"),
         nonnull=("player_code", "team_code"),
         note="the ONLY source of player_code, which every join keys on"),
    Node("playerstats.csv", _ext("playerstats.csv"), "external", min_rows=500,
         columns=("id", "now_cost", "selected_by_percent", "total_points", "minutes"),
         nonnull=("id", "now_cost"),
         note="per-gameweek panel; core_insights takes the newest snapshot per id"),
    Node("teams.csv", _ext("teams.csv"), "external", min_rows=20,
         columns=("code", "id", "name", "short_name", "elo"),
         nonnull=("code", "name"),
         note="elo goes NULL upstream without warning — see team_elo_2627.csv"),
    Node("gameweek_summaries.csv", _ext("gameweek_summaries.csv"), "external", min_rows=38,
         columns=("id", "deadline_time", "finished", "is_next"),
         nonnull=("id", "deadline_time"),
         note="deadlines; drives which gameweek is live"),
    Node("team_history.csv", _ext("team_history.csv"), "external", min_rows=1000,
         columns=("player_id", "gw", "team_code"),
         note="in-season club moves"),

    # --- committed inputs (hand-transcribed or pinned; in git) ---------------
    Node("team_elo_2627.csv", _dat("team_elo_2627.csv"), "committed", min_rows=20,
         columns=("code", "name", "elo", "as_of", "source"), nonnull=("elo",),
         note="pinned last-good Elo; core_insights falls back to it and says so"),
    Node("set_piece_takers.csv", _dat("set_piece_takers.csv"), "committed", min_rows=50,
         columns=("club", "role", "order", "name"),
         note="FFS projected duty, fills clubs where FPL declares no taker"),
    Node("team_hyperparams.json", config.TEAM_HYPERPARAMS, "committed",
         note="TeamModel hyperparameters from 31 seasons; scripts/calibrate_team_history.py"),
    Node("manager_changes.csv", config.MANAGER_CHANGES, "committed",
         note="transfermarkt transcription; caveat in studies/late_form_carryover.py"),
    Node("solio_cache.md", config.SOLIO_CACHE, "committed",
         note="market feed cache; data/solio_snapshots are NOT regenerable"),

    # --- derived: reconstructed inputs --------------------------------------
    Node("E0_recon.csv", config.E0_RECON, "derived", producer="scripts/reconstruct_e0.py",
         inputs=("teams.csv",), min_rows=300,
         columns=("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "AvgH", "AvgD", "AvgA"),
         note="team-model input rebuilt from repo Opta xG"),
    Node("coldstart_hist.csv", config.COLDSTART_HIST, "derived",
         producer="scripts/reconstruct_coldstart.py",
         inputs=("players.csv", "playerstats.csv"), min_rows=400,
         columns=("id", "element_type", "minutes", "now_cost"),
         note="cold-start calibration input"),

    # --- derived: priors (pickles in SCRATCH, gitignored) --------------------
    Node("pms_panel.pkl", config.PMS_PANEL, "derived", producer="scripts/build_all.py",
         inputs=("players.csv", "playerstats.csv", "team_history.csv"),
         note="25/26 per-match panel (build_pms)"),
    Node("ms_priors.pkl", config.MS_PRIORS, "derived", producer="scripts/build_all.py",
         inputs=("pms_panel.pkl",),
         note="two-season pooled priors, older_weight=0.5"),
    # NOT derived, and deliberately not an input edge anywhere. `starter_prior.
    # calibrate_ownership_start()` REWRITES this pickle every time it is called, and it
    # is called by build_all, gw_board (line 107), export_projection_detail (line 70)
    # and decision_v2. So whichever script runs last leaves it newer than the board that
    # ran first, and modelling it as an input to the board made the board permanently
    # STALE against a file the board itself had just rewritten. A check that is always
    # red is worse than no check: it trains you to skip reading it.
    #
    # Nothing is lost by dropping the edge. The pickle is a memoisation of pms_panel.pkl
    # and playerstats.csv, both of which ARE declared inputs to the board, so the real
    # dependency is still tracked — just at the source rather than through the cache.
    Node("own_start_cal.pkl", config.OWN_START_CAL, "cache",
         note="ownership -> start calibration; regenerated on demand by any consumer, "
              "so its mtime says nothing about freshness"),

    # --- derived: the canonical board and its siblings -----------------------
    Node("gw_board_long.csv", _out("gw_board_long.csv"), "derived",
         producer="scripts/gw_board.py",
         inputs=("players.csv", "playerstats.csv", "teams.csv", "gameweek_summaries.csv",
                 "team_history.csv", "E0_recon.csv", "ms_priors.pkl", "pms_panel.pkl",
                 "coldstart_hist.csv", "set_piece_takers.csv"),
         min_rows=1000,
         columns=("player_code", "player", "pos", "team", "cost", "gw", "mean",
                  "blended", "sd", "app_ev", "att_ev", "cs_ev", "defcon_ev"),
         nonnull=("player_code", "mean", "team", "cost"),
         note="CANONICAL. One row per player-gameweek, with the point decomposition"),
    Node("gw_board_wide.csv", _out("gw_board_wide.csv"), "derived",
         producer="scripts/gw_board.py", inputs=("gw_board_long.csv",), min_rows=400,
         columns=("player", "pos", "team", "cost", "total"),
         note="same board pivoted; `total` sums `blended`, not `mean`"),
    Node("projection_detail_gw1_10.csv", _out("projection_detail_gw1_10.csv"), "derived",
         producer="scripts/export_projection_detail.py", inputs=("gw_board_long.csv",),
         note="per-gameweek fixture and prior detail behind each projection"),
    Node("solio_ensemble_demo.csv", _out("solio_ensemble_demo.csv"), "derived",
         producer="scripts/run_solio_ensemble.py",
         inputs=("gw_board_long.csv", "solio_cache.md"),
         note="model x market ensemble"),

    # --- derived: registered 2026-09-08 -------------------------------------
    # These three were durable, tracked in git, and invisible to the graph, so a tree
    # containing month-old copies of them read as clean. Inputs below are the reads
    # the producers actually perform, not the ones their names imply: all three call
    # `ci.load()` and `ci.to_elo_frame()`, so each carries the whole external feed plus
    # the pinned-Elo fallback, and each fits its own TeamModel from E0_recon.
    Node("cs_fixtures_gw1_10.csv", _out("cs_fixtures_gw1_10.csv"), "derived",
         producer="scripts/cs_fixtures.py",
         inputs=("players.csv", "playerstats.csv", "teams.csv", "team_elo_2627.csv",
                 "E0_recon.csv"),
         min_rows=100,
         columns=("gw", "team", "opp", "venue", "xGA", "cs_prob"),
         nonnull=("team", "cs_prob"),
         note="per-fixture clean-sheet table, 20 clubs x 10 gameweeks"),
    Node("decision_gw1_6_defconenv.csv", _out("decision_gw1_6_defconenv.csv"), "derived",
         producer="scripts/run_final_board.py",
         inputs=("players.csv", "playerstats.csv", "teams.csv", "team_elo_2627.csv",
                 "E0_recon.csv", "coldstart_hist.csv", "ms_priors.pkl", "pms_panel.pkl"),
         min_rows=400,
         columns=("id", "player", "pos", "team", "cost", "mean", "sd",
                  "defcon_ev", "cs_ev", "app_ev", "att_ev", "ppm"),
         nonnull=("player", "team", "mean", "cost"),
         note="GW1-6 decision board under the DefCon environment layer"),
    # NOTE: this one is a CALIBRATION SWEEP, not a decision surface. Declaring its real
    # inputs makes it STALE every time the feed moves, which is daily, and a check that
    # is always red is a check nobody reads. It is registered because it is durable and
    # regenerable; if the noise costs more than the coverage, the fix is to move it to
    # studies/ as evidence for older_weight=0.5, not to under-declare its inputs here.
    Node("older_weight_sweep.csv", _out("older_weight_sweep.csv"), "derived",
         producer="scripts/sweep_older_weight.py",
         inputs=("players.csv", "playerstats.csv", "teams.csv", "team_elo_2627.csv",
                 "E0_recon.csv", "coldstart_hist.csv", "pms_panel.pkl"),
         min_rows=200,
         columns=("player", "pos", "team", "cost", "own", "range"),
         nonnull=("player", "range"),
         note="two-season older_weight sensitivity, 0.0-1.0; the evidence for 0.5"),

    # --- orphan: read by live code, produced by nothing -----------------------
    # None currently, and an empty section here is the finding, not an omission.
    # `team_strength_2627.csv` was the one, registered 2026-09-08 and resolved the
    # same day: it was not the explorer's SOURCE but its fallback, taken only when
    # `team_projections_season.csv` was absent, so the explorer was reading current
    # data and would have switched to a 2026-08-06 snapshot without saying so. The
    # read is gone (`gw_explorer.team_frames`); the file is left on disk, referenced
    # by nothing. The `orphan` kind stays because the check that finds the next one
    # is what has value, not the node it happened to find first.

    # --- locked: never regenerated, never checked for staleness --------------
    Node("predictions/", config.PREDICTIONS, "locked",
         note="boards as they stood before kick-off. The inputs that produced them no "
              "longer exist; scoring against a real gameweek needs exactly these"),
]

NODES = {n.name: n for n in _NODES}


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def graph():
    """The nodes, name -> Node."""
    return dict(NODES)


def consumers(name):
    """Nodes that list `name` as a direct input."""
    return [n.name for n in _NODES if name in n.inputs]


def _reach(name, seen):
    for c in consumers(name):
        if c not in seen:
            seen.append(c)
            _reach(c, seen)
    return seen


def downstream(name):
    """Everything that must be rebuilt if `name` changes, in dependency order."""
    seen = _reach(name, [])
    order = []
    for n in seen:
        deps = [d for d in seen if d in NODES[n].inputs]
        pos = max([order.index(d) for d in deps if d in order] + [-1]) + 1
        order.insert(pos, n)
    return order


def producers_for(names):
    """The distinct scripts to run, in order, to rebuild `names`."""
    out = []
    for n in names:
        p = NODES[n].producer
        if p and p not in out:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# The check. Every finding is one of: ok / STALE / EMPTY / SCHEMA / MISSING / skip
# ---------------------------------------------------------------------------
def _read_head(path, nrows=20000):
    """First `nrows` only — the check runs on every invocation of doctor and must stay
    cheap. The consequence: `min_rows` is a floor that a capped read can only ever
    satisfy early (fine, it is a floor), and `nonnull` sees a prefix. A column that is
    NULL for the first 20,000 rows and populated after would be a false positive; no
    artifact here is ordered in a way that makes that reachable, and the failure this
    exists to catch — a column that went entirely NULL upstream — shows up in row one.
    """
    import pandas as pd
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def check(nodes=None):
    """Run the gate. Returns a list of (name, status, detail); status in
    {'ok', 'MISSING', 'SCHEMA', 'EMPTY', 'STALE', 'skip'}.

    Staleness is TRANSITIVE. `gw_board.py` writes long and wide in one pass, so
    wide is always a second newer than long and a purely local mtime comparison
    calls it fresh while the board it was pivoted from is stale — which is the
    exact case the check exists to catch. A node whose input is stale is stale.
    """
    rows = _check_local(nodes)
    by_name = dict((r[0], list(r)) for r in rows)
    changed = True
    while changed:                       # fixpoint; the graph is small and acyclic
        changed = False
        for r in rows:
            n = NODES.get(r[0])
            # by_name, not r: r is the ORIGINAL row and never changes, so testing it
            # here re-marks an already-stale node every pass and the loop never ends.
            if n is None or n.kind != "derived" or by_name[r[0]][1] == "STALE":
                continue
            bad = [i for i in n.inputs
                   if i in by_name and by_name[i][1] == "STALE"]
            if bad:
                by_name[r[0]][1] = "STALE"
                by_name[r[0]][2] = (f"input stale: {', '.join(bad[:3])}"
                                    f" -> run {n.producer}")
                changed = True
    return [tuple(by_name[r[0]]) for r in rows]


def _check_local(nodes=None):
    """Per-node checks, ignoring what upstream neighbours look like."""
    rows = []
    for n in (nodes or _NODES):
        if isinstance(n, str):
            n = NODES[n]
        if not n.exists():
            # A missing derived artifact is a build step not yet run, not a defect.
            sev = "skip" if n.kind == "derived" else "MISSING"
            rows.append((n.name, sev,
                         "not built yet" if sev == "skip" else f"absent: {n.path}"))
            continue

        # A cache is rewritten by whoever needs it, so it is always "current" by
        # definition and its mtime must never enter a staleness comparison.
        if n.kind == "cache":
            rows.append((n.name, "ok", "cache"))
            continue

        # An orphan is a standing finding, not a transient one: it is read by live code
        # and no command can refresh it. Reported before the shape checks because a
        # well-formed orphan is still an orphan.
        if n.kind == "orphan":
            rows.append((n.name, "ORPHAN", "read by live code, no producer — " + n.note))
            continue

        # 1. staleness — derived only, and purely relative. No thresholds to tune.
        if n.kind == "derived" and n.inputs:
            mt = n.mtime()
            newer = [i for i in n.inputs if NODES[i].exists() and NODES[i].mtime() > mt]
            if newer:
                more = f" +{len(newer) - 3} more" if len(newer) > 3 else ""
                rows.append((n.name, "STALE",
                             f"older than {', '.join(newer[:3])}{more}"
                             f" -> run {n.producer}"))
                continue

        # 2. shape and content — CSVs only; a pickle or json is checked by its reader.
        if not n.path.endswith(".csv"):
            rows.append((n.name, "ok", n.kind))
            continue
        try:
            d = _read_head(n.path)
        except Exception as e:
            rows.append((n.name, "SCHEMA", f"unreadable: {type(e).__name__}: {e}"))
            continue
        missing = [c for c in n.columns if c not in d.columns]
        if missing:
            rows.append((n.name, "SCHEMA", f"columns absent: {', '.join(missing)}"))
            continue
        if len(d) < n.min_rows:
            rows.append((n.name, "EMPTY", f"{len(d)} rows, floor {n.min_rows}"))
            continue
        # The 2026-08-21 failure mode: the column is present and entirely NULL.
        dead = [c for c in n.nonnull if c in d.columns and d[c].isna().all()]
        if dead:
            rows.append((n.name, "EMPTY", f"column entirely NULL: {', '.join(dead)}"))
            continue
        rows.append((n.name, "ok", f"{len(d)} rows"))
    return rows


def brief():
    """The whole pipeline, bounded. Built from config, so it cannot drift."""
    lines = [f"FPL artifact graph — external feed: {EXTERNAL}",
             "  ' ' present   '?' absent", ""]
    for kind, title in (("external", "EXTERNAL (upstream, not ours)"),
                        ("committed", "COMMITTED (hand-maintained, in git)"),
                        ("derived", "DERIVED (regenerable)"),
                        ("cache", "CACHE (regenerated on demand; freshness not tracked)"),
                        ("orphan", "ORPHAN (read by live code, produced by nothing)"),
                        ("locked", "LOCKED (never regenerate)")):
        lines.append(title)
        for n in _NODES:
            if n.kind != kind:
                continue
            mark = " " if n.exists() else "?"
            src = f"  <- {n.producer}" if n.producer else ""
            lines.append(f" {mark} {n.name:28s}{src}")
            if n.inputs:
                lines.append(f"     needs: {', '.join(n.inputs)}")
        lines.append("")
    return "\n".join(lines)


def selftest():
    """Offline. Verifies the graph is well-formed and the query logic is right;
    requires no artifact to exist."""
    # every declared input names a real node
    for n in _NODES:
        for i in n.inputs:
            assert i in NODES, f"{n.name} depends on unknown node {i}"
    # no node depends on itself, directly or transitively
    for n in _NODES:
        assert n.name not in downstream(n.name), f"cycle through {n.name}"
    # every derived node names a producer; nothing else does
    for n in _NODES:
        if n.kind == "derived":
            assert n.producer, f"derived node {n.name} has no producer"
        else:
            assert not n.producer, f"{n.kind} node {n.name} must not have a producer"
    # a producer must be a file that exists — the whole point of the PLAN is that its
    # steps are runnable, and on 2026-09-08 it emitted `src/bayes_model.py` for an
    # artifact that module does not write. A named producer nobody checked is how a
    # dependency graph starts lying.
    for n in _NODES:
        if n.producer:
            assert _os.path.exists(_os.path.join(config.ROOT, n.producer)), \
                f"{n.name} names a producer that does not exist: {n.producer}"
    # an orphan is read by live code and produced by nothing; it must never be a
    # dependency of a derived node, or the PLAN would imply it can be refreshed
    # a cache must never be an input edge, or it reintroduces the permanent-STALE bug
    for n in _NODES:
        if n.kind == "cache":
            assert not consumers(n.name),                 f"cache {n.name} is an input to {consumers(n.name)} — its mtime is meaningless"
    for n in _NODES:
        if n.kind == "orphan":
            assert not consumers(n.name), \
                f"orphan {n.name} is an input to {consumers(n.name)} — the plan cannot rebuild it"
    # downstream order is a valid topological order
    d = downstream("playerstats.csv")
    assert "gw_board_long.csv" in d and "gw_board_wide.csv" in d
    assert d.index("gw_board_long.csv") < d.index("gw_board_wide.csv"), d
    assert d.index("ms_priors.pkl") < d.index("gw_board_long.csv"), d
    # producers_for is de-duplicated and order-preserving
    p = producers_for(d)
    assert len(p) == len(set(p)), p
    # check() on a synthetic tree: a stale derived node reports STALE, and a
    # present-but-entirely-NULL required column reports EMPTY.
    import tempfile
    import time
    import pandas as pd
    tmp = tempfile.mkdtemp()
    a = _os.path.join(tmp, "a.csv")
    b = _os.path.join(tmp, "b.csv")
    pd.DataFrame({"x": [1, 2, 3], "elo": [None, None, None]}).to_csv(a, index=False)
    pd.DataFrame({"y": [1]}).to_csv(b, index=False)
    _os.utime(b, (time.time() - 100, time.time() - 100))   # b older than a
    c = _os.path.join(tmp, "c.csv")
    pd.DataFrame({"z": [1]}).to_csv(c, index=False)          # c is NEWER than b
    na = Node("a.csv", a, "external", columns=("x", "elo"), nonnull=("elo",))
    nb = Node("b.csv", b, "derived", producer="p.py", inputs=("a.csv",))
    nc = Node("c.csv", c, "derived", producer="q.py", inputs=("b.csv",))
    NODES["a.csv"], NODES["b.csv"], NODES["c.csv"] = na, nb, nc
    try:
        got = dict((r[0], r[1]) for r in check([na, nb, nc]))
        assert got["a.csv"] == "EMPTY", got
        assert got["b.csv"] == "STALE", got
        # c is newer than b by mtime, so only TRANSITIVE staleness catches it.
        assert got["c.csv"] == "STALE", got
        # an orphan reports ORPHAN even when it is perfectly well formed
        nd = Node("d.csv", c, "orphan", note="no producer")
        assert check([nd])[0][1] == "ORPHAN", check([nd])
    finally:
        del NODES["a.csv"], NODES["b.csv"], NODES["c.csv"]
    print(f"manifest selftest ok — {len(_NODES)} nodes, "
          f"{sum(len(n.inputs) for n in _NODES)} edges, no cycles")
    return 0


def main():
    if "--selftest" in _sys.argv:
        return selftest()
    if "--check" in _sys.argv:
        bad = 0
        for name, status, detail in check():
            print(f"  {status:8s} {name:30s} {detail}")
            bad += status not in ("ok", "skip")
        print(f"  -> {bad} problem(s)")
        return 1 if bad else 0
    if "--downstream" in _sys.argv:
        who = _sys.argv[_sys.argv.index("--downstream") + 1]
        d = downstream(who)
        print(f"changing {who} invalidates: {', '.join(d) or '(nothing)'}")
        print("rebuild with: " + " ; ".join(producers_for(d)))
        return 0
    print(brief())
    return 0


if __name__ == "__main__":
    _sys.exit(main())
