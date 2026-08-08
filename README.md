# FPL 2026/27 Model

A hierarchical Bayesian Fantasy Premier League projection system for the 2026/27 season.
It estimates the underlying events (minutes, team strength, attacking share, clean sheets,
defensive contributions) and composes them through the FPL scoring rules to produce
posterior-predictive point distributions, then turns those into decisions — who to hold,
who to captain, where the differential edge is.

**Start with [`docs/PROJECT_KNOWLEDGE_2627.md`](docs/PROJECT_KNOWLEDGE_2627.md)** — the
authoritative context (settled decisions, validations, open questions, corrections). This
README is the how-to-run and the map.

---

## Layout

```
src/        all importable modules (flat, so cross-imports resolve)
  core engine   bayes_model, core_insights, roster, signals, betting_features,
                schedule_2627, multiseason_priors, multiseason, build_pms, pms_priors,
                fpl_xp_model, captaincy, apifootball, fm_priors, history, fpl_live_data, pull_fpl
  signal layers starter_prior (depth prior + minutes shrinkage + regime κ/δ),
                regime_panel (appointment split), defcon_env (DefCon environment),
                lineups (XI ingestion), solio_ensemble (benchmark + ensemble)
scripts/    runnable entry points (run_final_board is canonical)
tests/      acceptance/validation tests (regime, panel, defcon, shrinkage)
studies/    validation studies & the null-finding record (rotation, ownership, variance, ...)
data/       reconstructed inputs (E0_recon.csv, coldstart_hist.csv) + a Solio feed snapshot
outputs/    result boards (CSV)
docs/       project knowledge + the design notes (regime, reduced-form, DefCon handoff)
```

`src/` is flat on purpose: the engine modules import each other by name, so they must share
one path. Scripts/tests/studies add `../src` to `sys.path` automatically.

---

## Setup

```bash
pip install -r requirements.txt
# clone the public data repo somewhere and point the scripts at it:
git clone https://github.com/olbauday/FPL-Core-Insights
```

All paths are centralized in `src/config.py` and resolve automatically. The only thing you may
need to set is where the data repo lives, via the **`FPL_DATA`** environment variable.

**Windows (PowerShell):**
```powershell
$env:FPL_DATA = "C:\Users\you\FPL-Core-Insights\data"
py scripts\build_all.py
py scripts\run_final_board.py
```
**Windows (cmd):** `set FPL_DATA=C:\Users\you\FPL-Core-Insights\data` then `py scripts\build_all.py`.
**macOS / Linux:** `export FPL_DATA=~/FPL-Core-Insights/data` then `python scripts/build_all.py`.

Check paths any time with `py src\config.py` (or `python src/config.py`) — it prints where it
reads data and writes outputs, and flags `FPL_DATA` if the data repo isn't found. Regenerable
pickles go to `<repo>\.cache`, boards to `<repo>\outputs`, reconstructed inputs to `<repo>\data`
— all inside the repo, no `/tmp` or absolute paths (override with `FPL_SCRATCH` / `FPL_OUTPUTS`).
Original `E0.csv` and `fpl-data-stats.csv` are **not needed** — `build_all.py` reconstructs them.

> Windows: use `py` (the Python launcher) wherever these docs say `python`. The scripts run
> identically under either and resolve their own imports relative to the repo.

---

## Quick start

```bash
# 1. regenerate inputs + priors (writes /tmp/*.pkl, data/E0_recon.csv, data/coldstart_hist.csv)
python scripts/build_all.py

# 2. WEEK-BY-WEEK per-player predictions (primary output): true per-GW projections,
#    betting-odds team strength + GW1 Solio blend, -> outputs/gw_board_{long,wide}.csv
python scripts/gw_board.py          # env: GW_HI=10 SOLIO_W_OURS=0.5 SOLIO=off ...

# 2b. horizon-aggregate board (GW1-6 totals; defcon_ev / cs_ev surfaced)
python scripts/run_final_board.py
#    flags: DEFCON_ENV=off | REGIME_PANEL=on | REGIME=proposed
#    optional feeds: LINEUPS_PATH=xi.csv  APIFOOTBALL_KEY=...

# 3. clean-sheet fixture ranking, GW1-10
python scripts/cs_fixtures.py

# 4. ensemble / benchmark vs Solio (live on your machine; cached snapshot offline)
python scripts/run_solio_ensemble.py

# 5. robustness / A-B
python scripts/sweep_older_weight.py
python scripts/decision_v2.py           # depth + shrinkage + regime A/B
python tests/test_regime.py             # regime variance acceptance tests
python tests/test_defcon_env.py         # DefCon environment natural experiment
```

---

## Live-data integrations

Two external signals are embedded as static snapshots (so the model runs offline) with
live-refresh hooks for your machine:

- **Betting odds → team strength** (`src/market_odds.py`). The 26/27 outright markets
  (title + relegation, de-vigged) become a per-team market Elo that blends into
  `TeamModel.fit()` through the existing ClubElo path — no model change. This is the
  documented biggest predictive gain (the market prices in transfers, new managers, and
  pre-season form that last-season xG can't). Snapshot 7 Aug 2026; refresh via
  `fetch_live_odds(api_key)` (The Odds API). Flags: `MARKET_ODDS=off`, `MARKET_WEIGHT=0.6`.
- **Press index (PPDA) → CBIRT DefCon** (`src/press_index.py`). Pressing intensity conditions
  the MID/FWD DefCon channel (recoveries scale with press, not xGA). Manager-aware for the
  26/27 regime clubs. This closes the CBIRT gap the DefCon handoff flagged (§4.4): Anderson
  correctly holds up at Man City (Forest's press ≈ City's under Maresca), while midfielders at
  high-press sides (Iraola's Liverpool, De Zerbi's Spurs) get boosted.

Caveats: the odds→strength map is an *overall* team signal (not an attack/defence split — that
needs match/supremacy odds), and new-manager PPDA is a judgment estimate from each manager's
prior club, flagged low-confidence and sweepable. Both are on by default as validated
corrections; disable with the flags above.

## What the model does, in one pass

1. **Team strength** — hierarchical Poisson on 25/26 results, priors from betting-implied xG,
   blended with ClubElo; promoted sides get informative Elo priors. (`bayes_model.TeamModel`)
2. **Player rates** — conjugate Gamma-Poisson per-90 involvement / assists / DefCon, two-season
   pooled by `player_code`, reverted for the new season. (`multiseason_priors`)
3. **Minutes** — the dominant lever. Ownership-aware cold-start depth prior → evidence-weighted
   shrinkage (trust history by sample size, shrink thin cases to the market) → regime variance
   widening for new-manager clubs → injuries/confirmed XIs override. (`starter_prior`,
   `regime_panel`, `lineups`)
4. **DefCon** — conditioned on the team defensive-action environment (xGA), not treated as a
   team-invariant trait. (`defcon_env`)
5. **Composition** — Monte-Carlo through the scoring rules; retains full posterior draws for the
   captaincy tail. Surfaces `mean, sd, defcon_ev, cs_ev, p5..p95`. (`bayes_model.project`)
6. **Validation** — benchmarked against Solio Analytics' public feed, pooled and within-team.
   (`solio_ensemble`)

## Design principles

Empirical over asserted. Reject unvalidated heuristics (rotation and mean-reversion are
tested-null; directional style priors are deliberately not built). New components ship
off-by-default or as validated corrections. Report uncertainty honestly — regime change is a
variance statement, not a mean one.

## Known limitations

- Team strength is reconstructed 25/26 xGA + ClubElo, **not live odds** — the biggest open
  predictive gain (`betting_features` already has the de-vig machinery; it needs a feed).
- `older_weight` (and Isak's projection) is parameter-dependent until 2023-24 data enables a
  walk-forward fit.
- The CBIRT DefCon channel is left unconditioned pending a press index.

See `docs/PROJECT_KNOWLEDGE_2627.md` §6-§7 for the full open-items and not-built lists.
