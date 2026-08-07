# FPL-2026-27-Model

Fantasy Premier League 2026/27 projection system — a hierarchical Bayesian, component-based
model (minutes → events → FPL scoring rules) plus an interactive projection board.

**Philosophy:** FPL points are a deterministic function of underlying match events. Estimate
E[minutes], E[goals], E[assists], E[clean sheets], E[bonus] with well-specified models and
compose through the scoring rules — never regress total points directly.

## What's in this repo

```
FPL-2026-27-Model/
├── README.md
├── requirements.txt
├── docs/
│   ├── PROJECT_KNOWLEDGE_2627.md   — full project handoff: methodology, validated findings,
│   │                                  open questions, corrections log
│   ├── FPL_RULES.md                — 2026/27 official scoring & squad rules (reference)
│   └── INTEGRATION.md              — live betting-odds feed integration notes
├── src/
│   ├── CORE ENGINE (now present — previously the project's central gap, see Status below)
│   ├── bayes_model.py              — hierarchical Bayesian core: `TeamModel`, `player_posteriors`,
│   │                                  `project()` — call once per single GW for a TRUE per-gameweek
│   │                                  posterior (not just a GW-range aggregate)
│   ├── core_insights.py            — `load()`/`to_signals()`/`to_roster()` for the public
│   │                                  olbauday/FPL-Core-Insights data repo
│   ├── roster.py                   — cold-start calibration, bootstrap/pull loaders
│   ├── signals.py                  — availability, set-pieces, ClubElo/Understat priors, odds fusion
│   ├── schedule_2627.py            — real 380-fixture 2026/27 schedule + promoted clubs
│   ├── betting_features.py         — odds de-vig → xG inversion, Dixon-Coles team ratings
│   ├── captaincy.py                — haul/blank/regret-based captain & differential picks
│   ├── apifootball.py              — optional live injuries/XI feed
│   ├── build_pms.py                — 25/26 per-match panel builder
│   ├── multiseason_priors.py / multiseason.py — two-season pooled priors
│   ├── pms_priors.py               — single-season per-match priors
│   ├── fpl_xp_model.py             — scoring constants + component xP, OLS helpers
│   ├── run_2627.py                 — CLI entry point / true per-GW projection driver
│   ├── final_ms.py                 — two-season runner (PATCHES.md edits pre-applied)
│   ├── final_2627.py               — joins on stable `player_code` (avoids name-reassignment bugs)
│   ├── holds_real.py / run_captain.py — GW1-3/6 holds + captaincy runners
│   ├── fm_priors.py                — Football Manager attributes as Bayesian priors for cold-start
│   │                                  (no-PL-history) players, ridge-calibrated on the FM x PL overlap
│   ├── history.py                  — 30-season hyperparameter calibration (reversion, promoted-team
│   │                                  prior, home advantage) from football-data.co.uk, replaces
│   │                                  hard-coded guesses
│   ├── fpl_live_data.py            — captured live FPL API snapshot (team strength, scoring config)
│   ├── pull_fpl.py                 — standalone script to pull fresh FPL bootstrap data locally
│   ├── rotation.py / pit_ownership.py / retest.py / edge_study.py / variance.py /
│   │   multihorizon.py / test_history.py / test_fm.py — validation & study scripts (rotation null,
│   │   corrected point-in-time ownership signal, per-match vs aggregate retest, haul-prediction edge
│   │   study, variance decomposition, multi-horizon (3/6/12 GW) projection, estimator self-tests)
│   ├── LAYERS ADDED THIS PROJECT (sit on top of the core engine)
│   ├── betting_odds_ingest.py      — de-vig betting odds → market-implied expected goals
│   ├── oddsapi_feed.py             — The Odds API fetch/snapshot/identifiability/stacking
│   ├── identifiability.py          — checks whether a market-odds fit is under-identified
│   ├── ab_market_vs_recon.py       — A/B team model: backward-looking xG vs market odds
│   ├── starter_prior.py            — ownership-aware cold-start depth prior + minutes shrinkage
│   ├── lineups.py                  — confirmed/predicted XI ingestion (file or API-Football)
│   ├── decision_v2.py              — integrated GW1-6 board builder (priors + injuries + XIs)
│   ├── decision_2627.py            — consolidated board + GW1 captaincy/differential picks
│   ├── cs_fixtures.py              — GW1-10 clean-sheet fixture ranking
│   ├── solio_ensemble.py           — blend/benchmark against Solio Analytics' public feed
│   ├── run_solio_ensemble.py       — end-to-end ensemble run script
│   ├── build_all.py                — regenerate every reconstructed input + prior, in order
│   ├── reconstruct_e0.py           — builds E0_recon.csv from repo 25/26 Opta xG (fair-odds
│   │                                  Poisson inversion — no betting-odds file required)
│   ├── reconstruct_coldstart.py    — builds coldstart_hist.csv from the per-match panel
│   ├── sweep_older_weight.py       — older_weight prior-sensitivity sweep
│   ├── validate_shrinkage.py       — A/B of the minutes-shrinkage fix vs. the Solio benchmark
│   └── style_matchup.py            — OFF BY DEFAULT: experimental style x style interaction
│                                      model (not yet validated — see module docstring)
├── data/                           — reconstructed pipeline inputs (gitignored — see note below)
│   ├── E0_recon.csv                — team-model input, rebuilt from real 25/26 Opta xG
│   ├── coldstart_hist.csv          — cold-start calibration input, rebuilt from the match panel
│   └── solio_cache.md              — cached Solio Analytics GW1 snapshot (2026-07-25) used for
│                                      the ensemble benchmark
└── outputs/
    ├── fpl_projection_model.html    — interactive sortable/filterable projection board
    ├── fpl_2627_team_fixtures_gw1_10_raw.csv
    ├── older_weight_sweep.csv       — older_weight sensitivity sweep results (330 players)
    └── solio_ensemble_demo.csv      — sample our-model-vs-Solio ensemble/benchmark output
```

Not added: `decision_gw1_6_depthprior.csv` (591 rows) — its contents are already embedded
verbatim as the board's player data, so a standalone copy would just duplicate it — and
`decision_gw1_6_realistic.csv` (429 rows), an alternate-assumptions variant not currently
wired into the board. Both are available in the source update package if needed later.

## Status

**As of Aug 5 2026, the core engine is code-complete in this repo.** Every module previously
flagged as missing (`bayes_model.py`, `core_insights.py`, `roster.py`, `signals.py`,
`schedule_2627.py`, `multiseason_priors.py`, `build_pms.py`, `betting_features.py`,
`apifootball.py`, `captaincy.py`) plus the two top-level runners (`run_2627.py`, `final_ms.py`),
their transitive dependencies (`fpl_xp_model.py`, `multiseason.py`, `pms_priors.py`), and 15
supporting/validation scripts have all been copied into `src/` — see the file tree above. The
pipeline (`src/`) is written to run against the public `olbauday/FPL-Core-Insights` data repo
(match-level Opta stats, per-season player panels); point `core_insights.load()`'s `base` arg
and the `REPO` constant in the reconstruction scripts at a local clone of it. `outputs/`
contains the most recent computed board as static snapshots.

**Per-gameweek output is now possible.** `bayes_model.project(players, tm, tsamp, gw_lo, gw_hi,
S=...)` filters internally to `gameweek >= gw_lo & <= gw_hi`, so calling it once per single GW
(`gw_lo == gw_hi`) produces a genuine per-GW posterior — not the fixture-weighted proportional
split the board's "Gameweek Compare" tab currently approximates. `run_2627.py` is the CLI
driver for this. This directly resolves the original ask behind that tab.

**Not yet done in this session (sandbox execution was unavailable throughout):** nothing has
been run end-to-end here — the files are code-complete but unexecuted and unvalidated in this
environment. Run `python src/run_2627.py --gw-from <n> --gw-to <n>` locally to produce and
verify true per-GW output before trusting it over the board's current approximation.

**`data/` is present on disk but gitignored, on purpose.** `E0_recon.csv` is derived from
Opta match data and `solio_cache.md` is a cached scrape of a third-party commercial product
(Solio Analytics) — neither should be republished in a public repo. They're regenerated
locally via `python src/build_all.py` (which also needs a local clone of
`olbauday/FPL-Core-Insights` — see "Paths to set" below) rather than checked in.

Team strength currently comes from reconstructed 25/26 Opta xG + ClubElo blending, with
`fpl_live_data.py` adding FPL's own live team-strength ratings as an additional prior source.
Live 26/27 betting-odds team ratings (the feed-swap machinery in `betting_odds_ingest.py` /
`oddsapi_feed.py`) is built but not yet wired to a live odds source — see
`docs/PROJECT_KNOWLEDGE_2627.md` §7 for the full ranked list of open items.

## Paths to set

**Only needed for step 1 onward below — `run_2627.py` (step 0) needs none of this.** It pulls
everything it needs live over the network (FPL bootstrap-static, football-data.co.uk results,
optionally Understat/ClubElo) and falls back gracefully if a source is unreachable.

`reconstruct_e0.py`, `reconstruct_coldstart.py`, `rotation.py`, and `pit_ownership.py` each
hard-code `REPO`/`BASE = "/home/claude/repo/FPL-Core-Insights-main/data"` at the top; `core_
insights.load()` takes a `base` argument with the same default shape (`.../data/2026-2027`).
Point all of these at wherever you've cloned the public `olbauday/FPL-Core-Insights` data repo
before running `build_all.py`. Scratch pickles are written to a platform temp dir
(`pms_panel.pkl`, `ms_priors.pkl`, `own_start_cal.pkl`).

## Windows setup (PowerShell)

Confirmed working with Python 3.13 on Windows PowerShell. If `pip` alone isn't recognized (a
common PATH gap even when `python` itself works), run pip **as a module through Python**
instead — this sidesteps the PATH issue entirely:

```powershell
cd "C:\path\to\FPL-2026-27-Model"
python -m pip install -r requirements.txt

cd src
python run_2627.py --gw-from 1 --gw-to 1 --top 30
```

`python -m pip ...` (not bare `pip ...`) is the reliable form on Windows — use it for every
install command below too. No repo clone or extra setup needed for this command; see the note
above for what step 1+ additionally requires.

## Quick start

```bash
# 0. true per-gameweek projection (the core engine's main entry point) — no repo clone needed
python src/run_2627.py --gw-from 1 --gw-to 1     # single GW -> real per-GW posterior
python src/run_2627.py --gw-from 1 --gw-to 6     # GW range -> aggregate over the range

# 1. regenerate data/E0_recon.csv, data/coldstart_hist.csv + priors (needs the data repo clone)
python src/build_all.py

# 2. integrated GW1-6 board (depth prior + minutes shrinkage; injuries/XIs override)
python src/decision_v2.py
#    optional: LINEUPS_PATH=xi.csv  APIFOOTBALL_KEY=...  DEPTH_OFF=1

# 3. consolidated board + GW1 captaincy/differential picks
python src/decision_2627.py

# 4. clean-sheet fixture ranking, GW1-10
python src/cs_fixtures.py

# 5. ensemble/benchmark vs Solio (live on your machine; data/solio_cache.md offline)
python src/run_solio_ensemble.py

# 6. parameter robustness
python src/sweep_older_weight.py
python src/validate_shrinkage.py
```

See `docs/PATCHES.md` for the three one-line edits the existing runner needs to point at the
reconstructed inputs instead of the originals.

## Key validated findings (see docs/PROJECT_KNOWLEDGE_2627.md for full detail + evidence)

- Minutes/availability is the dominant single-GW lever (~32% of variance)
- No rotation multiplier — tested null, not modeled
- Trust xG over goals; "he's due" has ~zero predictive value
- GK picked on team defence, not save volume (save volume is *negatively* correlated with points)
- Two-season priors roughly double early-season predictive power vs. single-season
- Ownership is the strongest predictor after minutes (Spearman ~0.48) — used as a start-probability
  input, never as a projection shortcut
- Cold-start players: ownership predicts start-rate at Spearman ~0.68, used as a depth prior
- Evidence-weighted minutes shrinkage improves every agreement metric vs. the Solio benchmark
  (Pearson 0.73→0.78, MAE 0.75→0.70) while leaving rich-history players untouched

## Running the pipeline

Requires Python 3.10+, `numpy`, `pandas`, `scipy`, `sklearn`, and a local clone of the public
`olbauday/FPL-Core-Insights` data repo (update the `REPO`/`base` path at the top of each script
— see "Paths to set" above). See each module's docstring for its exact CLI / usage — most have
a `--selftest` flag that validates the logic offline with synthetic data, no repo needed.

```bash
python -m pip install -r requirements.txt
python src/identifiability.py --selftest      # sanity-check the market-odds identifiability logic
python src/oddsapi_feed.py --selftest         # sanity-check the odds pipeline wiring
```

(On Windows, always prefer `python -m pip` over bare `pip` — see "Windows setup" above.)

See "Quick start" above for the reconstruction + analysis scripts added in the 2026/27
session update.

## Interactive board

Open `outputs/fpl_projection_model.html` in any browser — no server needed. Sortable/filterable
player table, a derived per-gameweek split (flagged as an approximation — see the Notes tab in
the file itself for the exact methodology), CSV export by gameweek, and a GW1-10 team fixture
outlook tab.

## Automatic refresh

A Cowork scheduled task checks this repo every 3 days for a newer board under `outputs/` and
pulls it into the live interactive view; until a fresher board is pushed here it falls back to
checking live public inputs (official FPL API, football-data.co.uk fixtures, Solio Analytics'
public feed) and flags anything material that's changed.
