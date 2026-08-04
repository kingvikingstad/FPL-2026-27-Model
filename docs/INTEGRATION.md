# Live 26/27 team-strength odds — integration (handoff §7 item #1)

Feed swap, not new architecture. Two modules, one entry point.

## Files
- `betting_odds_ingest.py` — `fixtures → de-vig (1X2 + O/U 2.5) → implied (λH, λA) → E0-format`.
  Handles **both** sources behind one call. Caveats baked in: Avg*/Max* consensus only
  (Pinnacle never read); events with no O/U at the line are dropped + counted; Dixon-Coles
  `rho` available but off by default.
- `ab_market_vs_recon.py` — fits the model on `E0_recon` vs `E0_market`, reports the att/dfn
  repricing per team, and rechecks the CS fixtures flagged as leaning on the stale Spurs prior
  (Chelsea GW8, Everton GW4). `--selftest` validates its logic with no repo.

## Operational commands
```bash
# Primary weekly snapshot (immediate round):
python betting_odds_ingest.py --source footballdata \
       --fixtures https://www.football-data.co.uk/fixtures.csv --out /tmp/E0_market.csv

# Forward horizon (GW1–6) for an identification-grade fit — save the /odds JSON first:
#   GET /v4/sports/soccer_epl/odds?regions=uk&markets=h2h,totals&oddsFormat=decimal
python betting_odds_ingest.py --source oddsapi --fixtures odds.json --out /tmp/E0_market.csv

# Close the loop:
python ab_market_vs_recon.py --recon /tmp/E0_recon.csv --market /tmp/E0_market.csv \
       --clubelo-recon 0.45 --clubelo-market 0.20
```

## run_2627.py wiring (mirrors cs_fixtures.py:18)
```python
from betting_odds_ingest import build_market_e0

# footballdata (path/URL) OR oddsapi (parsed events list). Confirm goal/xg cols first (below).
build_market_e0("footballdata", FIXTURES_URL, out_path="/tmp/E0_market.csv",
                goal_cols=("FTHG","FTAG"), xg_cols=("HxG","AxG"))

tm = TeamModel(promoted_per_club=pclub).fit(
        e0_path="/tmp/E0_market.csv",   # was E0_recon.csv
        clubelo=elo, clubelo_weight=0.20)   # was 0.45 — see note
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
