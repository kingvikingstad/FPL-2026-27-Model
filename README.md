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
│   ├── betting_odds_ingest.py      — de-vig betting odds → market-implied expected goals
│   ├── oddsapi_feed.py             — The Odds API fetch/snapshot/identifiability/stacking
│   ├── identifiability.py          — checks whether a market-odds fit is under-identified
│   ├── ab_market_vs_recon.py       — A/B team model: backward-looking xG vs market odds
│   ├── starter_prior.py            — ownership-aware cold-start depth prior + minutes shrinkage
│   ├── lineups.py                  — confirmed/predicted XI ingestion (file or API-Football)
│   ├── decision_v2.py              — integrated GW1-6 board builder (priors + injuries + XIs)
│   ├── cs_fixtures.py              — GW1-10 clean-sheet fixture ranking
│   ├── solio_ensemble.py           — blend/benchmark against Solio Analytics' public feed
│   ├── run_solio_ensemble.py       — end-to-end ensemble run script
│   └── style_matchup.py            — OFF BY DEFAULT: experimental style x style interaction
│                                      model (not yet validated — see module docstring)
└── outputs/
    ├── fpl_projection_model.html   — interactive sortable/filterable projection board
    └── fpl_2627_team_fixtures_gw1_10_raw.csv
```

## Status

The pipeline (`src/`) is written to run against the private `FPL-Core-Insights` data repo
(match-level Opta stats, per-season player panels) which is **not included here** — these
scripts are the modeling logic, not the data. `outputs/` contains the most recent computed
board as static snapshots.

Team strength currently comes from reconstructed 25/26 Opta xG + ClubElo blending. Live
26/27 betting-odds team ratings (the feed-swap machinery in `betting_odds_ingest.py` /
`oddsapi_feed.py`) is built but not yet wired to a live odds source — see
`docs/PROJECT_KNOWLEDGE_2627.md` §7 for the full ranked list of open items.

## Key validated findings (see docs/PROJECT_KNOWLEDGE_2627.md for full detail + evidence)

- Minutes/availability is the dominant single-GW lever (~32% of variance)
- No rotation multiplier — tested null, not modeled
- Trust xG over goals; "he's due" has ~zero predictive value
- GK picked on team defence, not save volume (save volume is *negatively* correlated with points)
- Two-season priors roughly double early-season predictive power vs. single-season
- Ownership is the strongest predictor after minutes (Spearman ~0.48) — used as a start-probability
  input, never as a projection shortcut

## Running the pipeline

Requires Python 3.10+, `numpy`, `pandas`, and access to the `FPL-Core-Insights` data repo
(update the `REPO` path constant at the top of each script). See each module's docstring for
its exact CLI / usage — most have a `--selftest` flag that validates the logic offline with
synthetic data, no repo needed.

```bash
pip install -r requirements.txt
python src/identifiability.py --selftest      # sanity-check the market-odds identifiability logic
python src/oddsapi_feed.py --selftest         # sanity-check the odds pipeline wiring
```

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
