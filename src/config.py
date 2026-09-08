"""
config.py — one place for every path. Cross-platform; override via environment.
==============================================================================
Set nothing and it auto-detects; or set these env vars (Windows: `set VAR=...`,
PowerShell: `$env:VAR="..."`, macOS/Linux: `export VAR=...`):

  FPL_DATA     path to the cloned data repo's `data` dir (the ONE you likely must set):
               e.g.  C:\\Users\\you\\FPL-Core-Insights\\data
  FPL_HISTORY  path to a clone of vaastav/Fantasy-Premier-League's `data` dir, which
               carries player-gameweek history back to 2016/17 (optional; only the
               deep-history backfill in fpl_history.py needs it)
  FPL_SCRATCH  where regenerable pickles live (default: <repo>/.cache)
  FPL_OUTPUTS  where result boards are written (default: <repo>/outputs)

Everything else (E0_recon.csv, coldstart_hist.csv, the Solio cache) lives in
<repo>/data and needs no configuration.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (parent of src/)


def _first_existing(paths):
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None


# --- the external data repo (olbauday/FPL-Core-Insights) ---
REPO = (os.environ.get("FPL_DATA")
        or _first_existing([
            "/home/claude/repo/FPL-Core-Insights-main/data",              # sandbox
            os.path.join(ROOT, "..", "FPL-Core-Insights", "data"),        # sibling clone
            os.path.join(ROOT, "..", "FPL-Core-Insights-main", "data"),
            os.path.join(ROOT, "FPL-Core-Insights-main", "data"),
        ])
        or os.path.join(ROOT, "..", "FPL-Core-Insights", "data"))         # documented default

# --- deep player-gameweek history (vaastav/Fantasy-Premier-League) ---
# Optional. REPO carries only 24/25 onward; this reaches back to 2016/17 and is the
# only source of player-level history the project has (history.py covers TEAM results
# from 1993 but explicitly cannot inform player priors).
HISTORY = (os.environ.get("FPL_HISTORY")
           or _first_existing([
               os.path.join(ROOT, "..", "Fantasy-Premier-League", "data"),
               os.path.join(ROOT, "..", "Fantasy-Premier-League-master", "data"),
           ]))

# --- repo-local dirs (created if missing; work on any OS) ---
DATA = os.path.join(ROOT, "data")
SCRATCH = os.environ.get("FPL_SCRATCH") or os.path.join(ROOT, ".cache")
OUTPUTS = os.environ.get("FPL_OUTPUTS") or os.path.join(ROOT, "outputs")
# Scenario runs (a squad under a constraint, a horizon transfer plan) are answers to a
# question you asked once, not the canonical board. They were sitting alongside
# gw_board_long.csv in OUTPUTS, where nothing distinguished "the model's projection" from
# "what if I keep Haaland" six weeks later. Same directory tree, one level down.
PLANS = os.path.join(OUTPUTS, "plans")
# LOCKED PREDICTIONS — the one directory here that is NOT regenerable.
# Everything else in OUTPUTS can be rebuilt from the current data; a prediction cannot,
# because the moment a deadline passes the inputs that produced it stop existing. Scoring
# the model against a real gameweek (PROJECT_KNOWLEDGE §6.6, still unticked) needs the
# board exactly as it stood before kick-off, so it is copied here and never overwritten.
PREDICTIONS = os.path.join(ROOT, "predictions")
for _d in (SCRATCH, OUTPUTS, PLANS, PREDICTIONS):
    os.makedirs(_d, exist_ok=True)

# --- regenerable scratch (build_all.py writes these) ---
# Dated board snapshots, one per day, for the week-over-week delta the explorer shows.
# In SCRATCH and not OUTPUTS because they are neither regenerable (a past board cannot be
# rebuilt once the data moves) nor precious (losing them costs a convenience view, never a
# validation — that is what predictions/ is for). Gitignored; a cleared cache degrades the
# delta column to blank rather than breaking anything.
BOARD_HISTORY = os.path.join(SCRATCH, "board_history")
os.makedirs(BOARD_HISTORY, exist_ok=True)

# Where scripts/profile_pipeline.py writes its timings. In SCRATCH because a profile is
# machine-specific evidence, not a result: it says what THIS box costs, and committing it
# would invite comparing two numbers measured on different hardware.
PROFILE = os.path.join(SCRATCH, "profile.csv")

PMS_PANEL = os.path.join(SCRATCH, "pms_panel.pkl")
MS_PRIORS = os.path.join(SCRATCH, "ms_priors.pkl")
PMS_PRIORS = os.path.join(SCRATCH, "pms_priors.pkl")
OWN_START_CAL = os.path.join(SCRATCH, "own_start_cal.pkl")

# --- reconstructed inputs (committed in <repo>/data) ---
E0_RECON = os.path.join(DATA, "E0_recon.csv")
# The gameweek explorer's page template. It lives beside its module in src/ rather than
# inside it: it is ~2k lines of HTML/CSS/JS and an embedded string that size makes the
# module unreadable and unsearchable. src/ stays flat and importable — this is a sibling
# asset, not a package.
EXPLORER_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "gw_explorer_view.html")
# TeamModel hyperparameters calibrated from 31 seasons of results.
# Regenerate with scripts/calibrate_team_history.py (needs network).
TEAM_HYPERPARAMS = os.path.join(DATA, "team_hyperparams.json")
# Managerial changes, club + date, transcribed from transfermarkt (see
# studies/late_form_carryover.py for the provenance caveat).
MANAGER_CHANGES = os.path.join(DATA, "manager_changes.csv")
# Per club-season transfer transaction counts, transcribed from transfermarkt
# (see studies/transfer_churn.py for the provenance caveat and the validation gate).
TRANSFER_COUNTS = os.path.join(DATA, "transfer_counts.csv")
# English clubs that played July/August European qualifying ties, per PL season
# (see studies/euro_qualifying_fade.py for provenance).
EUROPEAN_QUALIFYING = os.path.join(DATA, "european_qualifying.csv")
# Projected set-piece duty (Fantasy Football Scout), used to fill clubs where FPL
# declares no taker — see src/set_piece_takers.py.
SET_PIECE_TAKERS = os.path.join(DATA, "set_piece_takers.csv")
# Fantasy Football Scout's GW1-6 projection workbook — a third-party comparator read by
# src/external_projections.py. Lived in the repo ROOT under its download name ("FFS 1-6
# Projection.xlsx", spaces and all) and was opened by a path built inside that module,
# which is the one thing this file exists to prevent.
FFS_PROJECTION = os.path.join(DATA, "ffs_1_6_projection.xlsx")
# Last-good club Elo, pinned. The data repo ships an `elo` column in teams.csv and on
# 2026-08-21 it went entirely NULL upstream while the column itself stayed put — which
# NaN'd the whole team layer and surfaced forty lines later as "SVD did not converge".
# core_insights.load() falls back to this snapshot and says so. Refresh it from the repo
# whenever upstream is populating Elo again.
TEAM_ELO = os.path.join(DATA, "team_elo_2627.csv")
COLDSTART_HIST = os.path.join(DATA, "coldstart_hist.csv")
SOLIO_CACHE = os.path.join(DATA, "solio_cache.md")
# Timestamped Solio JSON snapshots. NOT regenerable: the feed publishes only `latest`
# and keeps no history, so a snapshot not taken is a movement observation lost forever.
# Lives in DATA (committed) for the same reason PREDICTIONS does — see src/solio_market.py.
SOLIO_SNAPSHOTS = os.path.join(DATA, "solio_snapshots")
os.makedirs(SOLIO_SNAPSHOTS, exist_ok=True)

# --- optional legacy input (only used if present) ---
FPL_DATA_STATS = os.environ.get("FPL_DATA_STATS") or os.path.join(DATA, "fpl-data-stats.csv")

# --- soccerdata (Understat / WhoScored) ---
# Scraped sources: cache aggressively, re-scrape rarely. Lives under SCRATCH so it is
# gitignored, but it is NOT cheaply regenerable — a WhoScored pull takes hours.
SD_CACHE = os.environ.get("FPL_SD_CACHE") or os.path.join(SCRATCH, "soccerdata")
os.makedirs(SD_CACHE, exist_ok=True)

# PitchAPI response cache. The service is free and unmetered TODAY, which is exactly why
# a research instrument must not depend on it staying up: every response is written to
# disk so a board can be rebuilt with the network unplugged.
PITCH_CACHE = os.environ.get("FPL_PITCH_CACHE") or os.path.join(SCRATCH, "pitchapi")
os.makedirs(PITCH_CACHE, exist_ok=True)

# hand-verified and committed (see crosswalk.py); the review file is a working artefact
CROSSWALK_UNDERSTAT = os.path.join(DATA, "crosswalk_understat.csv")
CROSSWALK_REVIEW = os.path.join(DATA, "crosswalk_review.csv")


HISTORY_PANEL = os.path.join(SCRATCH, "history_panel.pkl")


def repo(season=None):
    """REPO, or REPO/<season> if given."""
    return os.path.join(REPO, season) if season else REPO


def history(season=None):
    """HISTORY, or HISTORY/<season> if given. Raises if unset — deep history is
    optional, so callers should fail loudly rather than silently skip seasons."""
    if not HISTORY:
        raise RuntimeError(
            "deep history not found. Clone vaastav/Fantasy-Premier-League next to this "
            "repo, or set FPL_HISTORY to its `data` dir.")
    return os.path.join(HISTORY, season) if season else HISTORY


def status():
    print(f"ROOT     {ROOT}")
    print(f"REPO     {REPO}   {'[ok]' if os.path.isdir(REPO) else '[MISSING - set FPL_DATA]'}")
    print(f"HISTORY  {HISTORY or '(not found - optional, set FPL_HISTORY)'}")
    print(f"SCRATCH  {SCRATCH}")
    print(f"OUTPUTS  {OUTPUTS}")
    print(f"PLANS    {PLANS}")
    print(f"PREDICT  {PREDICTIONS}")
    print(f"DATA     {DATA}")
    print(f"SD_CACHE {SD_CACHE}")
    print(f"PITCH    {PITCH_CACHE}")


if __name__ == "__main__":
    status()
