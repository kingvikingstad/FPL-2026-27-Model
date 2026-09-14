# Live 26/27 team-strength odds — integration (handoff §7 item #1)

Feed swap, not new architecture. Two modules, one entry point.

## Files
- `betting_odds_ingest.py` — `fixtures → de-vig (1X2 + O/U 2.5) → implied (λH, λA) → E0-format`.
  Handles **both** sources behind one call. Caveats baked in: Avg*/Max* consensus only
  (Pinnacle never read); events with no O/U at the line are dropped + counted; Dixon-Coles
  `rho` available but off by default.
- `oddsapi_feed.py` — the live half: fetch, snapshot accumulation, and the rank guard that
  refuses a market-ONLY E0 until the accumulated fixture set identifies all 39 free
  parameters. `--build --recon <E0_recon.csv>` emits the blended E0; that is the only
  market-bearing E0 this repo will produce.

- ~~`ab_market_vs_recon.py`~~ — **DELETED 2026-09-14.** It fitted the market-only E0 that
  `oddsapi_feed` (3) refuses, at HALF the ClubElo weight, and reported the difference from
  `E0_recon` as market repricing. See `INTEGRATION_LOG.md`, "The market-vs-recon A/B was
  fitting the design the rank guard refuses".

## Operational commands
```bash
# Primary weekly snapshot (immediate round). --out defaults to SCRATCH/E0_market.csv:
.\fpl.ps1 run src/betting_odds_ingest.py --source footballdata --fixtures https://www.football-data.co.uk/fixtures.csv
```

```bash
# Forward horizon for an identification-grade fit (needs ODDS_API_KEY):
.\fpl.ps1 run src/oddsapi_feed.py --fetch
```

```bash
# Blend the accumulated market rows onto E0_recon; prints coverage + rank report:
.\fpl.ps1 run src/oddsapi_feed.py --build --recon data/E0_recon.csv
```

There is no "close the loop" A/B step any more. `--build` prints the coverage and rank
report, and refuses to emit a market-only E0 that cannot identify the fit; that check is
the thing the old A/B was reaching for, done before the fit rather than after it.

## Wiring (paths from `config`, never literals)
```python
import os, config
from betting_odds_ingest import build_market_e0

# footballdata (path/URL) OR oddsapi (parsed events list). Confirm goal/xg cols first (below).
_mkt = os.path.join(config.SCRATCH, "E0_market.csv")
build_market_e0("footballdata", FIXTURES_URL, out_path=_mkt,
                goal_cols=("FTHG","FTAG"), xg_cols=("HxG","AxG"))

# STACK, do not swap. A market E0 on its own is 19 columns short of the 39 free
# parameters (identifiability.free_params(20)); fitting it yields numbers that are
# almost entirely prior, and LOWERING clubelo_weight to "let the market drive" makes
# that worse, not better. oddsapi_feed.build() does the stacking and the rank check.
tm = TeamModel(promoted_per_club=pclub).fit(
        e0_path=blended_e0, clubelo=elo, clubelo_weight=0.45)
```

## The one thing to confirm before trusting the swap
`build_market_e0` writes the market λ into **both** `FTHG/FTAG` and `HxG/AxG`. Check which
column `bayes_model.TeamModel.fit` actually reads off `e0_path` (whatever `E0_recon.csv` used)
and set `goal_cols`/`xg_cols` to match. This is the sole integration unknown.

## Decisions (flagged, not asserted — want walk-forward evidence)
1. **Replace vs augment.** The immediate football-data round is 1 fixture/team (~20 λ for ~40
   att/dfn params) → under-identified alone; ClubElo + shrinkage would carry it, defeating the
   purpose. Prefer the Odds API GW1–6 pull (~6/team) so the market genuinely drives att/dfn,
   then drop `clubelo_weight` toward ~0.20. Fitting the single round with `clubelo_weight=0.45`
   is a nudge, not a repricing.
2. **`older_weight` interaction.** Once team strength is market-set, the Isak sensitivity
   (player-prior, not team-prior) is unchanged — this feed does not resolve it.

## Expected signature of a correct run
Biggest positive `d_att` on the heavy-turnover / market-repriced sides (Spurs under De Zerbi,
Chelsea under Alonso, the promoted trio getting real ratings). Spurs' CS-conceded rechecks
should fall (~−0.05 to −0.10 on the synthetic analogue), softening the City+Chelsea rotation's
reliance on Chelsea's GW8-vs-Spurs fixture.

## Refresh cadence
football-data: Fri ~17:00 / Tue ~13:00 BST, ≥2×/week. Odds API: h2h+totals×1 region = 2
credits/pull → twice-daily ≈ 120/month, leaving room for a weekly outrights cross-check.
Snapshot only — won't capture post-team-news line moves; refit weekly, override lineups via
`apply_availability` as now.
