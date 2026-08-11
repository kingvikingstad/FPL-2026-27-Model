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
for _d in (SCRATCH, OUTPUTS):
    os.makedirs(_d, exist_ok=True)

# --- regenerable scratch (build_all.py writes these) ---
PMS_PANEL = os.path.join(SCRATCH, "pms_panel.pkl")
MS_PRIORS = os.path.join(SCRATCH, "ms_priors.pkl")
PMS_PRIORS = os.path.join(SCRATCH, "pms_priors.pkl")
OWN_START_CAL = os.path.join(SCRATCH, "own_start_cal.pkl")

# --- reconstructed inputs (committed in <repo>/data) ---
E0_RECON = os.path.join(DATA, "E0_recon.csv")
# TeamModel hyperparameters calibrated from 31 seasons of results.
# Regenerate with scripts/calibrate_team_history.py (needs network).
TEAM_HYPERPARAMS = os.path.join(DATA, "team_hyperparams.json")
# Managerial changes, club + date, transcribed from transfermarkt (see
# studies/late_form_carryover.py for the provenance caveat).
MANAGER_CHANGES = os.path.join(DATA, "manager_changes.csv")
# Per club-season transfer transaction counts, transcribed from transfermarkt
# (see studies/transfer_churn.py for the provenance caveat and the validation gate).
TRANSFER_COUNTS = os.path.join(DATA, "transfer_counts.csv")
COLDSTART_HIST = os.path.join(DATA, "coldstart_hist.csv")
SOLIO_CACHE = os.path.join(DATA, "solio_cache.md")

# --- optional legacy input (only used if present) ---
FPL_DATA_STATS = os.environ.get("FPL_DATA_STATS") or os.path.join(DATA, "fpl-data-stats.csv")

# --- soccerdata (Understat / WhoScored) ---
# Scraped sources: cache aggressively, re-scrape rarely. Lives under SCRATCH so it is
# gitignored, but it is NOT cheaply regenerable — a WhoScored pull takes hours.
SD_CACHE = os.environ.get("FPL_SD_CACHE") or os.path.join(SCRATCH, "soccerdata")
os.makedirs(SD_CACHE, exist_ok=True)

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
    print(f"DATA     {DATA}")
    print(f"SD_CACHE {SD_CACHE}")


if __name__ == "__main__":
    status()
