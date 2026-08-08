"""
config.py — one place for every path. Cross-platform; override via environment.
==============================================================================
Set nothing and it auto-detects; or set these env vars (Windows: `set VAR=...`,
PowerShell: `$env:VAR="..."`, macOS/Linux: `export VAR=...`):

  FPL_DATA     path to the cloned data repo's `data` dir (the ONE you likely must set):
               e.g.  C:\\Users\\you\\FPL-Core-Insights\\data
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
COLDSTART_HIST = os.path.join(DATA, "coldstart_hist.csv")
SOLIO_CACHE = os.path.join(DATA, "solio_cache.md")

# --- optional legacy input (only used if present) ---
FPL_DATA_STATS = os.environ.get("FPL_DATA_STATS") or os.path.join(DATA, "fpl-data-stats.csv")


def repo(season=None):
    """REPO, or REPO/<season> if given."""
    return os.path.join(REPO, season) if season else REPO


def status():
    print(f"ROOT     {ROOT}")
    print(f"REPO     {REPO}   {'[ok]' if os.path.isdir(REPO) else '[MISSING - set FPL_DATA]'}")
    print(f"SCRATCH  {SCRATCH}")
    print(f"OUTPUTS  {OUTPUTS}")
    print(f"DATA     {DATA}")


if __name__ == "__main__":
    status()
