# DefCon source correction — 2026-09-10

**Status: shipped as a validated correction.** Four commits: three correction steps plus
the stats-referee's changes. Each has its own board A/B arm, so no movement is credited to
the wrong cause. Code: `src/defcon_series.py`.
Evidence: `studies/defcon_source_audit.py` / `.csv`.

---

## 1. The defect  [VERIFIED 2026-09-10]

FPL scores DefCon on two different sums: **DEF = CBIT** (clearances + blocks +
interceptions + tackles, threshold 10) and **MID/FWD = CBIRT** (the same plus recoveries,
threshold 12). FPL-Core-Insights 2025-26 publishes a pre-summed column, and for
**defenders in GW2-GW10 only** it carries CBIRT.

Against FPL's official value (vaastav `merged_gw.csv`, joined on `player_code` through
`players_raw.csv`, single-fixture player-gameweeks, 11,184 rows):

| | GW1 | GW2-10 | GW11-38 |
|---|---|---|---|
| DEF published ≠ official | 0.000 | **0.854** (0.81-0.88 per GW) | 0.000 |
| MID / FWD published ≠ official | 0.000 | 0.000 | 0.000 |

- 89.9% of the GW2-10 DEF mismatches equal FPL's own CBIRT (official + official recoveries).
- FPL's official DEF value equals its own CBI + tackles in 100% of rows.
- The per-gameweek file `player_gameweek_stats.csv` has the same fault in its pre-summed
  column (DEF agreement 14.6% in GW2-10, 100% elsewhere), but its **components** are
  FPL-exact: `clearances_blocks_interceptions + tackles` = official in 3,845/3,845 rows.
- GK is also wrong in GW2-10 of the published column, and immaterial: GK cannot score DefCon.

**What it did** (defenders):

| | published | corrected (components) | FPL official |
|---|---|---|---|
| P(hit 10 \| 60+ min), all GWs | 0.361 | 0.270 | 0.270 |
| P(hit 10 \| 60+ min), GW2-10 | 0.632 | 0.262 | 0.259 |
| DefCon per 90, 900+ min defenders | 8.66 | 7.68 | 7.68 |

Per-player inflation of published over official: median +13.4%, p90 +25.7%, max +41.2%.

## 2. The choice: components, not FPL's official column

Both are right in expectation. The **per-match components** were chosen:

1. **Same feed and grain.** The prior panel is per match (it carries opponent and venue).
   vaastav is per fixture id, so using it needs an element→code map per season **and** a
   fixture→match_id crosswalk: two new joins that fail quietly, for a gain the audit
   puts at zero bias (mean signed error +0.003 per match; exact in 92.3% of DEF rows; the
   rest is ±1-3 Opta revision noise, symmetric).
2. **One rule for prior and in-season.** In 26/27 the published column **and** the
   per-match components are 100% null through GW3 for every position. The only populated
   DefCon source is `player_gameweek_stats`, whose components are FPL-exact. "Sum the
   components" applies to both files unchanged, so an in-season update cannot drift from
   the prior it updates. `fpl_defcon` already handles both schemas. **No in-season DefCon
   update exists yet**; this is the constraint it must build on.
3. **No new dependency.** The prior build still does not need `FPL_HISTORY`.

MID/FWD keep the published column, which is exact for them.

## 3. What changed

| Commit | Change | Board effect (same seed, GW4-38) |
|---|---|---|
| `dcf9003` | DEF DefCon = CBIT from components everywhere (`build_pms`, `defcon_team`, studies); constants re-measured; panel column `defcon_raw` → `defcon_fpl` so a stale pickle fails; manifest edge `coldstart_hist.csv ← pms_panel.pkl` | DEF likely starters `defcon_ev` 0.683 → 0.528 (−22.8%); FWD/GK bit-identical |
| `3c13bfe` | Evidence counted under the **26/27 scoring position's** rule | exactly six re-listed players move; everyone else bit-identical |
| `3c06395` | DefCon exposure excludes minutes with a null defensive block | DEF likely starters +0.011 (+2.1%); concentrated where the nulls were |
| (referee) | Re-listed players shrink toward their own position's pool in the scoring unit; clustered EB shrink; block permutation null (§5) | exactly the six re-listed players move, ≤0.04 pts/gw each; everything else bit-identical in `mean` |

Arms A (`505c8f7`) through E share one seed, and `project()` seeds an RNG per player, so
an unchanged player is bit-identical and any move is attributable. Comparisons use the
model `mean`, not `blended`: `blended` mixes the live Solio feed, which moved between
arms D and E (22 of 27 rows) and on no earlier pair.

**Net A→E:** DEF likely starters `defcon_ev` **0.683 → 0.539 per gameweek (−0.144, −21%)**.
It is −21% on EV against −11% on the rate because P(≥10) is convex in the rate. DEF GW4-9
rank Spearman 0.998, top-20 overlap 18/20; the top ten are the same players, with
Virgil and Thiaw swapping 5th and 6th. Whole-board GW4-9 Spearman 0.999, top-50 48/50.
The losers are cheap DefCon-reliant defenders over GW4-9: Sessegnon −1.95, Bassey −1.73,
Truffert −1.66, Mitchell −1.50, Dunk −1.43, Murillo −1.39, Porro −1.38. The gainers are
players whose 25/26 DefCon was null: Colwill +1.48, Disasi +1.07, Muñoz +0.31. Mean GW4-9
totals: DEF 10.28 → 9.89, MID 8.635 → 8.676, FWD and GK unchanged to MC noise.

**The scoring-position rule** (`3c13bfe`). FPL re-lists players between seasons, and
the prior counted each under his 25/26 listing. So **Sessegnon and Wieffer (MID→DEF)
carried midfield recoveries into a defender's threshold of 10**: this is the same defect,
reached by a different route, and it predates the upstream one. Sessegnon's `defcon_ev`
falls 0.42 → 0.09 per gameweek (0.13 at commit `3c13bfe`, 0.09 once shrunk toward midfielders' own CBIT pool, §5). Dorgu, Lewis-Potter, Bogarde and Lewis-Skelly (DEF→MID)
had CBIT counts against a CBIRT threshold of 12, and rise by +0.04 to +0.18. The prior
*mean* for a player re-listed across the rules (DEF ↔ MID/FWD) is his own position's pool
converted to the scoring unit (§5). A MID↔FWD re-listing is CBIRT either way and keeps his
own position's prior.

**Null exposure** (`3c06395`). 2.4% of 25/26 league minutes, rising to 16% of rows in
GW29, carry every other stat but a null defensive block, and FPL's official value for
them is non-zero. A NaN-skipping sum over a full-minutes denominator is `fillna(0)` by
another route. It diluted 12% of 450+ minute players, some wholly: Disasi's 14
appearances are all null, and his rate read 1.81/90 against a pooled 7.68.

## 4. The studies: what survives

| Claim (as recorded) | Published series | Corrected | Survives? |
|---|---|---|---|
| DefCon flat in opponent **strength** | 0.339 / 0.328 / 0.416 / 0.329 | **0.245 / 0.240 / 0.334 / 0.246** | **Yes, in the monotone sense.** Level falls ~0.09; the shape is unchanged |
| Q3 bump | +0.08 | **+0.090**; opponent-permutation p = **0.0095** for the largest of four bins (0.0045 had Q3 been named in advance) | **Post hoc and unexplained.** A strength-shuffled null already carries each opponent's identity effect, so "identity falling into one bin" does **not** account for it. That earlier reading is withdrawn. The player-cluster CI first reported (+0.057, +0.127) resampled the wrong unit. Not consumed; needs a pre-registered 26/27 replication on the Q3 band as defined by 25/26 xG |
| Total defender EV falls with fixture difficulty | 2.05 → 1.17 | **1.86 → 1.71 → 1.63 → 1.00** | **Yes.** Pick defenders on fixture ease |
| CB vs FB | 0.480 vs 0.207, **2.3×** | **0.398 vs 0.114, 3.5×** (CI 2.82-4.37); gap +0.284 CI (+0.238, +0.328) | **Yes, and larger.** Recoveries had inflated full-backs proportionally more |
| `PRIOR_DC` DEF 7.6 was "below the measurement", so replaced by 8.590 | 8.590 | **7.678** | **No.** The measurement was what was high; 7.6 was right |
| Opponent **identity** rating ships (D1: split-half r ≥ 0.5) | r = +0.555, shrink 0.68, swing 0.129 | **r = +0.526**; clustered EB shrink **0.68**; swing **0.142** (0.28 pts/match) | **Procedurally yes, informationally thin.** 0.026 over the gate on 20 opponents is ~0.16 SE; the Fisher CI on r is (0.11, 0.79). That is justified **only because the rating is display-only** (explorer and the team table; it never enters points). Promotion into points needs a pre-registered 26/27 replication: do 25/26 ratings predict 26/27 within-player opponent residuals? 18/20 categories are unchanged: Arsenal neutral→permissive, Tottenham permissive→neutral |
| Opponent main effect beats its permutation null | row null (anti-conservative) | block null: R² 0.072 vs 0.025 [0.014, 0.041] | Yes |
| Permissive opponents are the better attacks | corr +0.264 | **+0.346** | Yes, stronger. Still no "hard fixture, DefCon floor" trade |
| MID opponent rating fails D1 | r = +0.241 | +0.241 (bit-identical) | Yes (MID was never contaminated) |
| **Matchup (club × opponent) is a null** | row-null R² 0.170 "clears"; match identity **0.278**; two-meeting r **−0.217** | **block null: 0.503 vs 0.496 [0.440, 0.545], inside**; r **+0.005** | **Verdict survives. Its evidence was replaced.** See below |

**The matchup null, re-founded.** Two faults sat under the original verdict.

1. **The defect built the old legs.** A defect confined to GW2-10 is a *time* effect.
   Match identity partly encodes the gameweek, so under the published series it absorbed
   the inflated block (0.278). A club-opponent pair met once inside GW2-10 and once
   outside it showed opposite residuals in its two meetings (r = −0.217). The referee
   reproduced both legs from a constant GW2-10 shift alone (0.291 / −0.219).
2. **The permutation null was anti-conservative.** `perm_r2` permuted rows, but a club's
   ~4 defenders in one match share that match's shock. So any grouping that nests
   club-matches (opponent, club × opponent, match identity) beats a row-level null by
   construction, and "club × opponent clears its null" was that artefact.
   `perm_r2(..., block=["club", "match_id"])` now permutes club-match blocks. The selftest
   plants a pure shared shock that fools the row null (0.455 > 0.143) and not the block
   null.

On the corrected series, with the block null, the DEF interaction sits **inside** its
null and the two meetings of a pair do not agree (r = +0.005). That is a null on two
legs. The "match identity explains more" comparison is reported but decides nothing,
since it contrasts two groupings that differ only in which shocks they pool.

The `defcon_matchups` CB/FB bootstrap is also fixed. It filtered on `isin(pick)`, which
collapses duplicate draws, so each replicate was a ~63% subsample *without* replacement,
and the CI came out ~24% too narrow. It is now a real player-cluster bootstrap.

## 5. Stats-referee pass (2026-09-10): ship with changes, all applied

The referee's verdict was that the board path is sound and the study record carried biases.

**Confirmed:**
- **Components are unbiased for both uses.** Per-player derived/official − 1 has mean
  +0.05% and sd 1.25%. Threshold flips at 60+ minutes are symmetric (0.41% up, 0.38%
  down). Outside GW2-10 the components add ~3-4% variance to a season rate at zero bias,
  a cost the single-rule argument outweighs.
- **Null exclusion is sound.** Nulls say *who and when*, not *how many*. Against FPL's
  official value on the null rows, a player's null-row rate is 0.957× his own measured
  rate for DEF (0.70-1.10), 1.06× for MID (1.004-1.154), and 0.946× for FWD. Residual
  bias is ≤ ~1.6%, second-order next to the dilution removed.
- **The scoring-position rule is correct** (the threshold is paid in that unit).
- **The defect-built-the-old-legs attribution is correct** (§4).

**Fixed in response:**
1. The block permutation null in `perm_r2` (§4).
2. **EB shrink now clustered by club-match** (`defcon_team.opponent_defcon_ratings`).
   Row-level `se2` ignored a design effect of 1.31, which under-shrank every rating by
   ~11%. The clustered k is 0.68, which agrees with the full-season split-half reliability
   (0.689). The swing is 0.142, not 0.157. Ranks and categories are unchanged, since k is
   common to all opponents.
3. The Q3 bump is re-tested at the opponent level and recorded as post hoc and unexplained (§4).
4. **Re-listed players' prior mean** is now their *own* position's pool converted to the
   scoring unit. That is DEF CBIRT for a DEF→MID listing and MID CBIT for a MID→DEF one,
   measured live by `multiseason_priors._pooled_rate`. Previously it was the new label's
   pool, an unfitted pull on an administrative relabel.
5. `build_pms` attached the legacy published per-GW value as `fpl_defcon`, one
   transposition from `defcon_fpl`. It is renamed `published_defcon_gw`.

## 6. Still open

- **Poisson threshold calibration** `[CHECK]`, pre-existing. P(≥10) is composed as
  Poisson(rate × m90). With leave-one-out own rates it predicts 0.267 against 0.275
  observed, but by rate quintile it under-predicts low-rate defenders (0.043 vs 0.088) and
  over-predicts high-rate ones (0.576 vs 0.516). Var/mean of the residual is 1.45. This
  is part regression-to-the-mean, part overdispersion, and it is a second-moment claim the
  market gate cannot test. It needs a within-player negative-binomial dispersion for DEF
  CBIT and a held-out calibration by shrunk-rate band (fit GW1-19, score GW20-38). The −21%
  board headline passes through this composition.
- **`defcon_env` on prior-only players** — split implemented 2026-09-16 but **HELD, not
  validated**: the stats-referee showed it amplifies an unfitted DEF xGA beta (§7.1).
- **Backfill the null rows from FPL's official value.** Optional and lossless: 275 of 293
  rows are joinable through the audit's own join, and the January arrivals would stop
  falling to the prior.
- **Off-board residues** that dilute if revived: `bayes_model.player_posteriors` and
  `fm_priors` (published column, full-minutes denominator, source file absent);
  `multiseason.season_rates`, `pms_priors` and `studies/retest.py` (null minutes in the
  denominator). None is on the board path.

- **24/25 DEF DefCon from components.** The components exist in 24/25, so CBIT could be
  derived and would roughly double DefCon evidence for returning defenders. This is
  **untested, not done.** It is a new evidence channel, and `fpl_defcon` refuses pre-25/26
  seasons precisely so it cannot happen by accident. It needs its own validation (FPL
  published no 24/25 DefCon to check against) and an A/B.
- **Downstream artifacts in the main checkout** must be rebuilt after merging:
  `build_all` → `gw_board` → `export_team_projections` (the Arsenal and Tottenham
  `defcon_opponent_category` values change) → `export_gw_explorer`.
- `data/defcon_roles.csv` (being frozen in the main checkout) is unaffected: role labels
  come from the aerial/crossing profile and were verified identical, 146 of 146.
- `src/pms_priors.py` has no consumer. Only its column name was updated.
- A GK's DefCon gamma is still drawn (and consumes RNG) although it can never pay. This
  is harmless, but it is why a DefCon-only change moves GK rows by MC noise.

## 7. `defcon_env` references each share of alpha to where it was measured (2026-09-16)

**Status: HELD — do not merge as a validated correction.** See §7.1.

**The bias** `[DERIVED]`. `apply_defcon_environment` scaled the whole DefCon alpha by
xGA_27[club] / xGA_25/26[25/26 club] (press: the same ratio of `press_factor`). Alpha is
prior plus evidence. The evidence was produced at the 25/26 club; the prior (a pooled,
role, position, re-listed or cold-start price-calibrated rate) was pooled over the
league. Referencing the prior to one club over-credits it at a club that conceded less
than average and under-credits it at one that conceded more, by the full club/league
ratio for a player who is all prior. Players with no 25/26 club already used the league
reference, so two identical priors got different factors depending on whether FPL had
ever listed the player at a PL club.

**The fix.** `to_priors` and `roster._coldstart_row` carry `defcon_prior_alpha`
(= prior rate x k0). `defcon_env` scales that share by xGA_27[club] / mean xGA_25/26 and
`press_factor^beta` (the league press factor is 1 by construction), and the remainder by
the club ratio as before; each factor is clipped separately. A frame without the column
raises instead of reverting to whole-alpha scaling. Selftest: a pure prior with a known
high-xGA club equals the same prior with no club; pure evidence at an unchanged club is
unchanged; mixed, press and per-share clip cases.

**Scope is wider than the January arrivals.** The same bias applied to every prior share:
40 DEF in `ms_priors` whose 25/26 DefCon is all null, the prior share of every established
player (board DEF not on a cold start: median 17%, upper quartile 29%), and 111 cold-start rows that FPL had listed at a 25/26
PL club (e.g. Mfuni, listed at Man City, now Coventry: 1.88/1.16 clipped to 1.6 -> 1.33).

**Board A/B** `[VERIFIED]`, same seed, `LIVE_FPL=off`, model `mean`, GW1-38 (next GW 5):
- `ms_priors` identical in every existing column; only `defcon_prior_alpha` is new.
- Isolation: 0 players whose `mean` moved without `defcon_ev` moving; GK bit-identical.
- `defcon_ev` per GW, GW1-6: DEF 0.2439 -> 0.2404 (-1.4%), MID 0.0972 -> 0.0984,
  FWD 0.0040 -> 0.0042. Whole-board GW1-6 total rank rho 0.9998, top-50 50/50; DEF rho
  0.9996, top-20 20/20. New/old factor: DEF median 1.000, range 0.82-1.11.
- Movers, GW1-6 points: Gabriel -0.63, Mfuni -0.57, White -0.44, Vuskovic -0.41, Hincapie
  -0.34 (Arsenal's prior shares were referenced to xGA 0.75); Yalcouye +0.30, A.Garcia
  +0.24, Cook +0.21, Branthwaite +0.20. Disasi +0.03 (factor 0.92 -> 1.02; he barely plays).
- Level check: mean 26/27 projected xGA 1.457 against the 25/26 league 1.414 is
  composition (three promoted for three relegated); on the 17 common clubs it is
  1.348 -> 1.355. The evidence ratio carries the same level, so no new shift.

**Still `[JUDGMENT]`.** The league reference is the unweighted club mean of xGA, while
`RATE_DEF_POOLED` pools appearances (minutes-weighted). The evidence reference is the
player's end-of-season 25/26 club; a January mover with measured DefCon at two clubs is
referenced to one. Both pre-existing in kind.

### 7.1 Stats-referee pass (2026-09-16): biased as shipped

The split is the right *form*, and the A/B verifies the *implementation*: nothing moves
except `defcon_ev`. Nothing in it shows the new numbers are *accurate*. What decides that
is an elasticity that was never fitted:

- **`XGA_BETA["DEF"] = 1.0` is a judgment (regime handoff §4.2), and the data the prior
  was pooled from contradict it** `[VERIFIED, reproduced]`. On `studies/defcon_matchups.csv`
  (2,934 DEF appearances of 60+ minutes), the club CBIT rate regressed on log 25/26 club xGA
  (weighted by exposure) gives beta **0.174**, club bootstrap 95% CI **(-0.002, 0.488)**, and
  **0.221** after adjusting for the CB/FB mix. The rate at Arsenal (xGA 0.75) is 7.04
  against 7.68 pooled. beta=1 predicts 4.1.
- Before the split, beta only acted on club *movers*: a player who stayed had a factor of
  about 1. The split applies it to every prior share as a cross-club elasticity. For
  Arsenal the prior-share error goes from about +17% (old) to **-32%** (new, clipped at
  0.6). It is larger and has changed sign, so the headline movers (Gabriel, White,
  Hincapie) are mostly over-steep beta, not a corrected bias. The prior clip binds on 21 of
  215 DEF rows (Arsenal 8, Hull 13). Movers' evidence share was already over-scaled by
  the same beta.
- **Press league reference is not 1** `[VERIFIED]`. The mean of `press_factor` over the 20
  PPDA_2526 clubs is 1.0234 (1.0117 after the square root), so MID/FWD prior shares sit about
  1.2% high. Fix: divide by the empirical club mean, as DEF does.
- xGA denominator `[VERIFIED]`: unweighted 1.4142 against 1.4195 exposure-weighted (CB
  1.4114, FB 1.4298). Negligible. Including relegated clubs is right.
- Splitting alpha against the exact form r27*(a0+c)/(k0+m*r26) `[DERIVED]`: median
  difference +0.02% for regulars. The large gaps come from the clip, not the split.
- Cold-start `dc90` is an unweighted mean over players with 450+ minutes, not
  price-calibrated. Its seasons are `[CHECK]`. MID 8.4 / FWD 4.7 have no recorded source.

**What turns this into a validated correction.** First a pre-registered fit of DEF beta:
a Poisson GLM `dc ~ role FE + beta*log(xGA_club)`, offset log(mins_dc/90), SEs clustered by
club, plus a within-player version and a planted-beta simulation (0.2, 1.0) on real
exposures. The decision rule is fixed before the fit. Then the press reference fix, then a
same-seed A/B, scored on 26/27 GW1-4 DEF CBIT by club tercile. If beta refits near 0.2,
the split is the correct estimator and its board effect shrinks by roughly 5x.
