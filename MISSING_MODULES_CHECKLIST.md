# FPL-Core-Insights — Missing Modules Checklist

**Status as of Aug 5, 2026: RESOLVED.** The `fpl-core-engine` folder (connected this session)
supplied every core engine module this checklist originally flagged, plus previously-unknown
transitive dependencies and 15 supporting/validation scripts. All 29 files have been copied into
`FPL-2026-27-Model/src/`. This document is kept as a historical record of what was missing and
how it got resolved — nothing further needs to be uploaded for the items below.

---

## 1. Core engine modules — ✅ RESOLVED

All ten, previously supplied nowhere in the project cache, git repo, or either update package,
are now present in `FPL-2026-27-Model/src/`.

| Module | What it does | Status |
|---|---|---|
| `bayes_model.py` | Hierarchical Bayesian core — `TeamModel`, `.fit()`, `project(pl, tm, ts, gw_start, gw_end, S=...)` — the function that produces true per-gameweek projections | ✅ copied |
| `core_insights.py` | `load(base=...)` — reads the public [olbauday/FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) dataset into a unified frame; `to_signals()` | ✅ copied |
| `roster.py` | `calibrate_cold_start()`, `_coldstart_row()` — cold-start player calibration | ✅ copied |
| `signals.py` | `apply_availability()` — folds injuries/confirmed XIs into the projection | ✅ copied |
| `build_pms.py` | `build()` — builds the 25/26 per-match panel | ✅ copied |
| `multiseason_priors.py` | `two_season_evidence(older_weight=...)`, `to_priors()` | ✅ copied |
| `betting_features.py` | Builds team ratings from betting odds / xG | ✅ copied |
| `schedule_2627.py` | 2026/27 fixture scheduling logic | ✅ copied |
| `captaincy.py` | `captain_picks()`, `best_differential()` | ✅ copied |
| `apifootball.py` | Optional live injury/confirmed-XI feed via API-Football | ✅ copied |

## 2. Referenced but never located — ✅ RESOLVED

| Item | Status |
|---|---|
| `run_2627.py` — the true per-gameweek projection driver | ✅ copied. Calling `bayes_model.project()` once per single GW (`gw_lo == gw_hi`) produces a genuine per-GW posterior — confirmed by direct code read. This resolves the original ask behind the board's "Gameweek Compare" tab, which until now only approximated per-GW output with a fixture-weighted proportional split. |
| `final_ms.py` — the runner `PATCHES.md`'s edits target | ✅ copied, **with all three patches pre-applied** in the supplied copy (confirmed by reading the file). `docs/PATCHES.md` updated to note it's now historical/reference. |
| A true per-GW posterior file | ✅ resolved — see `run_2627.py` above. Not yet actually run in this environment (sandbox execution has been unavailable all session); needs to be run locally to produce and validate real output. |

## 3. Transitive dependencies — discovered and resolved

Not in the original checklist (they weren't known to be needed until `fpl-core-engine`'s
manifest, `RESOLVES_CHECKLIST.md`, documented them):

| Module | Needed by | Status |
|---|---|---|
| `fpl_xp_model.py` | `bayes_model` (scoring constants, component xP) | ✅ copied |
| `multiseason.py` | `multiseason_priors` (`build_2425_panel`) | ✅ copied |
| `pms_priors.py` | single-season priors (`season_rates`, `to_model_priors`) | ✅ copied |

## 4. Supporting data — action still needed (not code)

| Item | Status |
|---|---|
| A local clone of the public data repo, `olbauday/FPL-Core-Insights` | **Action needed on your machine.** `core_insights.load()`'s `base` argument and the `REPO` constant in `reconstruct_e0.py` / `reconstruct_coldstart.py` / `rotation.py` / `pit_ownership.py` all need to point at wherever you clone it (expects a `.../data/2026-2027` subpath). No upload needed here — just a `git clone` + path edit locally. |
| Original `E0.csv` (odds-based) | Not needed — `reconstruct_e0.py` rebuilds an equivalent from the repo's Opta xG |
| Original `fpl-data-stats.csv` | Not needed — `reconstruct_coldstart.py` rebuilds an equivalent from the per-match panel |

## 5. Bonus — supplied beyond what was asked for

`fpl-core-engine` also included 15 context/runner/validation files not in the original
checklist at all, all now copied into `src/`:

- **Runners:** `final_2627.py` (joins on stable `player_code`), `holds_real.py`, `run_captain.py`
- **New prior sources:** `fm_priors.py` (Football Manager attributes as Bayesian priors for
  cold-start players), `history.py` (30-season hyperparameter calibration from
  football-data.co.uk), `fpl_live_data.py` (captured live FPL API team-strength snapshot),
  `pull_fpl.py` (standalone script to pull fresh FPL data)
- **Validation studies (document validated & null findings):** `rotation.py` (European
  congestion — rotation null once player fixed effects control for squad-quality confounding),
  `pit_ownership.py` (corrected point-in-time ownership signal, ~0.48 vs the earlier leaky 0.51),
  `retest.py` (per-match vs FPL-aggregate rate model retest), `edge_study.py` (haul/P(≥10)
  prediction study), `variance.py` (single-GW variance decomposition), `multihorizon.py` (3/6/12
  GW cumulative projection), `test_history.py` / `test_fm.py` (estimator self-tests on
  synthetic data with known ground truth)

## What's left

1. **Run it.** Nothing in this repo has been executed end-to-end in this session (sandbox
   execution unavailable throughout). Clone `olbauday/FPL-Core-Insights` locally, set the
   `REPO`/`base` paths, and run `python src/run_2627.py --gw-from 1 --gw-to 1` to get and
   validate a real single-gameweek posterior.
2. Consider replacing the model's hard-coded hyperparameters (`revert=0.85`, `season_sd=0.15`,
   promoted-team priors, home advantage) with `history.py`'s 30-season-calibrated estimates —
   see `docs/PROJECT_KNOWLEDGE_2627.md` for how the current guesses were chosen.
3. If you have your own Football Manager save, `fm_priors.py` can sharpen priors for the three
   promoted clubs (Coventry, Hull, Ipswich) and any player with zero PL history.
</content>
