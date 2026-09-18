# Integration Log — 2026/27 build session

Chronological record of what was built, validated, and deliberately left out, with the
empirical result for each. Companion to `PROJECT_KNOWLEDGE_2627.md`.

## Foundation de-risking
- Verified prices/ownership/fixtures/transfers/injuries against the live 26/27 launch data.
- Corrected two earlier claims: Salah is absent from the data (not carried); Haaland's 70.3%
  is live pre-GW1 ownership, not a stale estimate.
- Reconstructed the two missing inputs faithfully: `E0_recon.csv` from the repo's 25/26 Opta
  match xG (fair-odds Poisson inversion → existing team model runs unchanged; ratings sound —
  Man City 2.02 attack, Arsenal 0.75 defence); `coldstart_hist.csv` from the per-match panel.
- Full pipeline reproduces prior numbers within MC noise (faithfulness confirmed).

## Signal layers built
1. **Cold-start depth prior** (`starter_prior.apply_coldstart_depth`) — ownership-aware start
   prior for no-history players (ownership predicts start-rate at Spearman ~0.68). Collapses
   cold-start × strong-team inflation (Alleyne 29.6→11.9); established players untouched;
   discriminating by ownership.
2. **Evidence-weighted minutes shrinkage** (`apply_minutes_shrinkage`) — trust history by
   sample size (`w=minutes/(minutes+900)`), shrink thin/role-changed cases to the market. The
   largest validated gain: Spearman vs Solio 0.482→0.609, MAE 0.75→0.70, Mosquera +1.32,
   Gabriel +0.02.
3. **Lineup/injury layer** (`lineups.py`) — source-agnostic XI ingestion feeding
   `apply_availability`; named starter→0.99, benched→sub-rate, excluded→0.05. API path wired.
4. **Solio ensemble** (`solio_ensemble.py`) — fetch/parse/align/blend/benchmark; reproduces
   Solio's leverage ranking; team-aware matching (fixed a Palmer GK/MID collision). Live GW1:
   Pearson 0.72, MAE 0.72. Added `benchmark_within_team` (isolates the share term; flagged
   Arsenal internal ranking rho 0.0 that pooled 0.61 hid).
5. **Regime variance decoupling** (`apply_regime_uncertainty`) — two knobs: δ (mean-pull, no-op
   1.0) and κ (variance-inflation via moment-matched mixture, no-op 0.0). Baseline identity
   exact; horizon signature passes (SD widens +0.32 at H=6, ~0 at H=1) — confirms the
   H(H−1) shared-p term propagates with no change to project(). Off by default
   (`REGIME_2627_PROPOSED`, includes Newcastle as the ninth regime club).
6. **Appointment panel split** (`regime_panel.py`) — for partial-regime clubs (Man Utd/Spurs),
   weight post-appointment 25/26 matches at full strength instead of a flat multiplier. Keyed on
   player_code (a player_id-keyed first cut matched only 2 of 41 incumbents). Market agreement:
   boosted incumbents avg 6.9% own vs cut 1.7%. Isolation passes.
7. **DefCon environment conditioning** (`defcon_env.py`) — scale the DefCon rate by the team
   xGA environment. DEF/CBIT full sensitivity (monotone in xGA); MID-FWD/CBIRT untouched
   (press-driven; Solio's Anderson 56% at City confirms it holds up). Anderson natural
   experiment correctly untouched; promoted-club defenders boosted (Hull/Coventry), City
   transfers cut (Guéhi/Khusanov).

## Bug fixes
- **Cold-start id collision (§6.2)** — `roster._coldstart_row` gave every cold-start player
  `id=-1`, collapsing ~198 into one under `drop_duplicates("id")`. Now unique (-1,-2,-3,...).
- **defcon_ev / cs_ev surfaced** — `bayes_model.project()` now returns per-player DefCon and
  clean-sheet expected points, making DefCon auditable/A-B-able (§4.1a of the DefCon handoff).
- **Solio name collision** — alignment made team-aware (name+team).

## The start-prior cap is ON by default — scored out of sample (§6.9b), 17 Sep 2026
`scripts/ab_inseason_starts.py`, evidence `outputs/ab_inseason_starts.csv`. The rule, the
power calculation and the endpoint were written into the file before the first run.

- **The old harness could not answer this, and its flaws all pointed one way.**
  `ab_inseason_minutes.py` scored only players who FEATURED — removing exactly the case the
  cap fixes, and selecting on a collider, since the arms disagree about who will feature. It
  scored a MEAN forecast with MAE, which is minimised by the median and so rewards shading
  down a right-skewed target. And `INSEASON_UPTO` never truncated `playerstats.csv`, so a
  board "built on GW1" still read GW4 ownership and injury flags. Kept as the record of its
  arms; superseded for this question.
- **Design.** Arm built with `INSEASON_UPTO=t-1`, scored on GW t, with the feed copied to
  scratch and `playerstats.csv` truncated to `gw <= t-1`. Every listed player at a club that
  played is scored — no filter on whether he appeared. Primary endpoint Brier on the realised
  native start; points squared error as a GUARDRAIL, because a per-player-gameweek squared
  error difference has sd ~1.1 against an expected effect of ~0.04 [DERIVED] and cannot be
  decisive in three weeks. Power for the primary came first: on 23/24-25/26, kappa=4 improved
  the next match's start prediction in **24 of 24 season-gameweeks**, mean 21.4%.
- **RESULT, GW2/3/4, all three pre-registered conditions met.** Brier **+15.4%** pooled
  (12.0 / 17.0 / 17.4 by week), better in **3/3** weeks, points squared error **+4.0%**
  (better, against a −2% guardrail). Players projected above 0.5 who did not start fell from
  29/33/35 to 22/26/30. `INSEASON_KAPPA` now defaults to 4 in `gw_board.py` and
  `export_projection_detail.py`; `INSEASON_KAPPA=off` restores the uncapped prior.
- **The level bias from §6.9(a) is absorbed downstream, and the A/B shows it.** In EVERY arm
  the mean projected start probability equals the realised start rate to three decimals
  (0.357 / 0.337 / 0.335) — the XI constraint pins the league to eleven per club, so the
  appearance-denominator bias cannot reach the board's level. What kappa buys is therefore
  DISCRIMINATION between players, not calibration in the large. That also answers the open
  question left in §6.9(a): fixing the denominator is a prior-layer correction, not a board
  one, and it does not confound this adoption.
- **The other two arms, non-adopting by prior declaration.** `INSEASON_LAM=0.75` on top of
  kappa added +1.2pp (18.2% / 19.8% in GW3/GW4) — promising, but it had no power calculation
  and stays OFF. `INSEASON_EXP_MINUTES` moved the start endpoint by 0.0% at every week, as
  expected since it touches minutes-per-start rather than selection; its points effect was
  +0.1%, far inside noise. §6.8 still needs its own design.
- One board change came with this: `DUMP_FRAME=<path>`, off by default, writes each
  gameweek's start prior as projected. Without it the board exposes no p_start at all and an
  A/B could only score points, where this effect is invisible.

## Start-prior strength re-fit on the production prior (§6.9a), 17 Sep 2026
`studies/start_prior_production.py`, evidence `studies/start_prior_production.csv`. It
reruns `start_prior_strength.py`'s sweep (same grids, same Brier, same leave-one-season-out
over 23/24-25/26) on a replica of the prior `gw_board.py` actually installs: two-season
pooled minutes-based starts, revert 0.7 + Beta(2,2), then the ownership shrink with live
ownership at the deadline after match k. Rules were pre-registered in the file before it
was run. No flag changed.

- **Fidelity gate.** The first run FAILED it and scored nothing (median strength 43.9
  replica vs 26.6 installed). The fix was to the replica, not to the gate: see the
  denominator finding below. Second run: r of prior means 0.962, median strength error 0%.
- **Q1: replicates, smaller.** The current setting (w=1, κ=∞) is beaten at every cutoff, 3/3
  folds: **6.3 / 6.7 / 7.3 / 7.2%** at k = 2/3/5/8, against 8.4-11.5% on the stand-in prior.
- **Q2: the lever is now identified, and it is the CAP.** On the stand-in prior, capping the
  prior and up-weighting evidence could not be told apart (S3 reproduces that on the same
  rows: mixed signs). On the installed prior the cap beats the weight by **+0.64% to +1.01%**
  of baseline Brier, 3/3 folds, at all four cutoffs. The margin is small next to the gain
  itself.
- **Q3: κ = 4 pinned** as the `INSEASON_KAPPA` value. It is interior; the per-cutoff LOSO
  optima are 4/4/5/5; κ=4 and κ=5 are 0.1pp apart pooled. `inseason.START_KAPPA` already
  held 4. **The flag stays OFF**: this is §6.9(a) only, and (b), the points A/B, is not done.
- **FINDING, bigger than the question: the installed start prior is biased UP by ~12
  points.** `two_season_evidence` counts `games` as rows of match-sheet panels (24/25: no
  zero-minute rows; 25/26: 9.9%), so `start_b` counts sub appearances, not matches missed.
  The prior estimates P(start | appeared); the in-season update and the board both need
  P(start | club match). Prior mean **0.548** against a realised **0.424**. Rebuilt with
  every listed match as the denominator (S5), the mean is **0.434** and the uncapped
  baseline Brier falls 0.2253 → 0.2109 (−6.4%) before any cap, roughly the cap's whole gain.
  With the denominator fixed, the cap still helps (+4.1-5.7%) and its optimum moves to κ=5-8.
  **So part of what κ=4 "fixes" is this mean bias.** Two caveats: [VERIFIED] at the prior
  level only. The board's XI constraint and availability run after the update and may absorb
  part of the level bias, and this study replicates neither. And S5 was declared
  non-decision-bearing, so it licenses a new pre-registered item, not a change.
- Declared sensitivities, reported only: mins≥60 target (S1) matches the primary; removing
  the ownership shrink (S2) raises baseline Brier to 0.270, i.e. the shrink is already
  correcting much of the denominator bias; a 10-match horizon (S4) prefers κ = 2-3. GK-only
  gain at κ=4: +18.0%, descriptive (H-POS is a recorded null).

## The market's lambda, per fixture, beside the model's, 11 Sep 2026
`src/fixture_market.py` — the per-fixture table now carries what the market says about
the same fixture (`mkt_lam_for`, `mkt_lam_against`, `mkt_p_clean_sheet`, `mkt_source`,
`mkt_as_of`), and the explorer's fixture tooltip shows it with the signed model − market
gap. **Nothing enters `TeamModel`**: the market is already in the team layer at
`MARKET_WEIGHT=0.6`, so an E0 re-anchor would count it twice and `stack_e0` still
refuses. What was missing was not a signal, it was the comparison.
- **Two sources, precedence per FIXTURE.** Solio's λ pair first (no de-vig or inversion
  choice on our side — `SOLIO_MARKET_FEED` §2); football-data.co.uk's market-average 1X2 +
  O/U 2.5, Shin de-vigged, second. Free, no key, stored in `data/odds_snapshots` by
  `python src/fixture_market.py --fetch`. A fixture Solio missed is filled by the books
  rather than left empty because its neighbours were priced.
- **Which price: the last one at or before that gameweek's deadline.** For a played week
  that is the same information cut as `predictions/`, so the column stays honest when the
  week is scored. Deadlines come from the payload's own `deadlineIso`, else from
  `gameweek_summaries` **by id** — that feed is not sorted.
- **Cross-source check, and it is the only one there can be.** Solio is Solio's estimate
  of the market, not the market. Against the independent de-vig on GW4's ten fixtures:
  **λ MAE 0.032, bias +0.011, max |Δ| 0.069**, the two observed 0.3h apart. [VERIFIED] on
  one book snapshot.
  The same statistic read **0.055** the next morning on *no new book price at all* — the
  Solio side had moved 13h. So the report carries the median observation gap beside the
  MAE: part of any cross-source difference is movement, not disagreement, and without the
  gap the number silently mixes the two.
- **Solio's `csProb` is plug-in exp(−λ)** to within 0.003 on all 22 stored snapshots
  [VERIFIED], so `mkt_p_clean_sheet` uses that one definition for every source. Set
  against the engine's posterior-predictive `p_clean_sheet` it would read the Jensen gap
  (mean +0.018) as a disagreement.
- **The tooltip's first draft walked into exactly that trap, and `ux-reviewer` caught it.**
  Printing the market's plug-in P(CS) beside the model's posterior-predictive one made the
  two round to the **same 2dp in 8 of 60 priced cells** — including Arsenal at Sunderland,
  where a real λ disagreement (0.72 vs 0.69 against) and the +0.011 Jensen bump cancelled
  and the pair read as the market confirming the model. Fixed by showing the market's
  figure only against the model's **plug-in** P(CS), on its own line that names what the
  headline number above it is. The signed λ gap now also carries the model's own 90% band
  for that fixture — inside/outside is the scale that says whether a gap exceeds the
  model's uncertainty, and it needed no new machinery: `lam_for_p5`/`p95` were already in
  the export.
- **Model vs market, GW4:** λ MAE 0.121, bias −0.010, plug-in P(CS) MAE 0.026 against
  Solio's 11 Sep 20:18Z price; 0.107 / −0.029 / 0.023 against its 12 Sep 10:51Z one, the
  last before the deadline. Quote the price with the statistic — the market moved more
  between those two snapshots than the two statistics differ. Printed for UPCOMING weeks
  only: on a played one the model column is today's posterior, which with `INSEASON=on`
  has already absorbed that match, so the comparison would flatter the model. The largest
  single gap at the earlier price, Arsenal at Sunderland (model λ 1.52 vs market 1.81),
  had closed to −0.19 and back inside the model's 90% band by the deadline.
- **Double gameweeks refused, not averaged.** A Solio record listing two fixtures carries
  one λ pair for both; whether that is a sum or a mean is undocumented, so
  `fixture_lambdas` drops it and reports `ambiguous` (the opponent's single-fixture record
  still recovers the match). This also tightens the off-by-default E0 path, which would
  previously have written that λ to both fixtures.
- **Both snapshot stores are manifest nodes** and inputs to the per-fixture table, so a
  newly stored price makes `doctor` report it — and the explorer — STALE. It fires only on
  a genuinely new price: both fetchers dedupe, Solio on `generatedAt`, the books on
  content.
- ~~Not scheduled~~ — **scheduled 2026-09-16** on the owner's decision: the books fetch
  runs after the Solio fetch on the existing 4-hourly task, independently of it, and the
  task exits non-zero if either fails (`SOLIO_MARKET_FEED` §9 item 4).

## Travel distance: one real effect, one null, shipped off then switched on, 10-11 Sep 2026
Pre-registered study `studies/travel_distance.py`, outcome-blind design pass first. 31
seasons, team-season attack+defence FE, SEs clustered on the club pair.
- **Traveller goals against: real.** +0.0323 per log-km, z = +4.62, the same in 2016-26
  alone. A derby roughly halves home advantage; the longest trips add ~4% to home goals.
- **Traveller goals for: null.** No code path.
- **Failed the market gate** (score z = +1.73), so `src/travel.py` ships **off by default**
  (`FPL_TRAVEL=on`). It is hooked into `_home_effect`, home side only, centred on the 25/26
  mean trip, with b drawn per posterior draw. The guard's premise (double-counting odds)
  does not hold mechanically here, because the per-fixture λ ingests no match odds. That is
  raised for a decision in `docs/TRAVEL_DISTANCE_2026-09-10.md` §6, not acted on.
- `style_matchup.market_score_test` split out of `beats_the_market`, so any covariate uses
  the one gate implementation; `beats_the_market` behaviour is unchanged.
- A/B, same seed: per-fixture CS moves −1.4pp to +3.6pp; season CS ±0.12 per club.
  Player-level deltas sit at the Monte Carlo floor. A changed λ desynchronises numpy's
  small-λ Poisson stream, so a same-seed A/B of a λ change is not common random numbers.
- **11 Sep 2026: switched ON by default** on the owner's explicit decision. It is a scoped
  override of the market gate, recorded in the CLAUDE.md guard row; `FPL_TRAVEL=off`
  disables it. The board and everything downstream were rebuilt. The GW4 deadline lock
  predates the change.

## The five inline home terms are routed, and there is now one owner, 14 Sep 2026
**Two sessions did this independently on the same day and their numbers agree to the
decimal place.** Both are recorded here as one entry, because it is one change. The
measurement below is the more thorough of the two A/Bs; the structural half beneath it is
from the other. Dated 14 Sep: an earlier draft of this entry and of
`TRAVEL_DISTANCE_2026-09-10.md` said 12 Sep, but there is no 12 Sep commit — both landed
on the 14th (`a48f4fd`, `e105a30`, merged at `b5a3547`).

`captaincy.point_draws`, `cs_fixtures.py`, and the `xga27` loops in `gw_board.py`,
`run_final_board.py` and `tests/test_defcon_env.py` built λ with their own
`home if is_home else 0`. They skipped the GW1-3 discount, and once `FPL_TRAVEL` went on
(11 Sep) the travel term as well, so the captaincy tails, the CS ranking and the DefCon
environment each read a different home term from the board's own mean for the same fixture.
Each now goes through `bayes_model.fixture_home_terms`, which fetches the fixture's trip
once and applies `_home_effect` to both sides with the SAME trip, as `project()` does.
A/B on one frozen tree `[VERIFIED]`:
- **The refactor is exact.** With `FPL_TRAVEL=off`, all 700 GW4-38 team-fixtures give
  bitwise-identical `lam_for` and `lam_against`; `point_draws` over GW4, GW4-6 and GW7-10 is
  bitwise equal; `cs_fixtures_gw1_10.csv` GW4-10 is identical in all 140 rows. So everything
  below is the discount and the travel term arriving, not a changed formula.
- **GW1-3, from the discount alone** (travel off): home sides' xGA ×1.079, away ×0.927
  (e^±0.076 exactly), P(CS) −2.5pp home and +2.4pp away on all 60 rows.
- **GW4-10, from the travel term** (default on): 70 of 140 rows move — every away side, mean
  |ΔP(CS)| 0.35pp, max **+3.3pp** (GW4 Man City at Man United), +3.0pp (GW5 Chelsea at
  Brentford), down to −1.4pp on the longest trips (Brighton at Sunderland). Home sides are
  unchanged by construction: the shift is on the home side's λ_for, which is the away side's
  λ_against. GW1-10 CS totals move ≤0.09 per club (Chelsea); the top 10 fixtures are the same
  set in a slightly different order.
- **The boards are not bitwise unchanged at GW4+, and should not be.** `xga27` is one scalar
  per club averaged over a window starting at GW1, so both terms reach it and then scale every
  DEF's `defcon_alpha` in every gameweek. Ratio new/old, travel on: 0.9917-1.0140 over the
  board's GW1-38 window, 0.932-1.026 over `run_final_board`'s GW1-6. `gw_board` with
  `LIVE_FPL=off`: **only DEF rows change** — 6,681 of 24,852, GW4-38 mean +0.0007/gw, max
  0.132/gw; GK, MID and FWD are bitwise identical at float64 in the GW4 draws (440/440), and
  174 of 214 DEFs move. Season top 30 is the same set. Player-level deltas are mostly the
  desynchronised-stream noise noted above; the systematic part is ≤1.4% of a DEF's DefCon EV.
- A same-seed board A/B needs `LIVE_FPL=off`: with it on, one player's status changed between
  two runs 20 min apart (168 → 169 ruled out) and moved 570 non-DEF rows that had nothing to
  do with the change under test.

**One owner, not five copies.** Routing each site to `_home_effect` directly leaves the
incantation — fetch the trip, remember it belongs to the FIXTURE and not to a side, call
twice — duplicated at six places, which is the duplication that produced this bug. The
merged implementation adds `bayes_model.fixture_home_terms(home, gw, is_home, team, opp)
-> (h, hopp)` and routes `project()` through it too, so the engine and its consumers
cannot diverge again. The board is BYTE-IDENTICAL across that `project()` refactor, so
nothing above is the restructuring. Row access at every touched site is bracket-indexed
per CLAUDE.md; the four `import travel` lines the inlined version needed are gone, the
helper owning that call now.

**Decomposition, on the club-season xGA that feeds DefCon (GW1-10 mean).** Discount alone
mean -0.0022 / max |0.031|; travel alone mean +0.0052 / max |0.035|; together mean +0.0030
/ max |0.051| — comparable size, partly offsetting. The travel half has the predicted
sign: the clubs whose own away trips are longest see xGA rise (Newcastle +0.019,
Ipswich / Sunderland / Hull +0.020), the London clubs see it fall (Chelsea -0.035, Spurs
-0.013). CS ranking cross-check from the second implementation: 71 of 200 fixtures move
P(CS) by more than 0.01, max 0.062, and the top-10 table is unchanged in membership.

`ab_market_vs_recon.py` was the sixth such site. It is not routed — it was deleted the
same day; see the entry below.

## Season-to-date facts labelled, and points-per-£m retired, 7 Sep 2026
Two changes to the explorer's Players tab, both about the same failure: a table that puts
an exact fact and a modelled quantity side by side, unlabelled, invites reading one as the
other.

**The realised 26/27 block is now the source of price and ownership, not just a column
beside them.** `season_to_date` joins `playerstats.csv` at its latest gameweek on
`player_code` — 653 of 653 board players, no misses — and `now_cost`,
`cost_change_event`, `cost_change_start`, `selected_by_percent`, `transfers_in/out_event`,
`total_points` and `minutes` come off it. `now_cost` and `selected_by_percent` now feed the
`Cost` and `Own %` cells directly rather than the board's copy taken at build time. The two
AGREE today — 653 of 653, max |difference| exactly 0.0 in both — so this is a no-op on
current inputs and a guard against reading a stale board's prices as live ones; a
disagreement is counted and printed at export rather than silently reconciled, because a
board built against different prices is a fact worth seeing.

Every one of these columns carries its own tooltip now, on the header and in the column
picker, saying which side of the line it sits on. The distinction matters most exactly
where it is least visible: at three gameweeks `total_points` is mostly variance and the
projection beside it already conditions on everything the number contains, while
`cost_change_start` and `selected_by_percent` are the two facts a transfer decision turns
on that the model does not model at all. They were previously separated only by a comment
in the source. Display precision was wrong too — counts and prices were printed to two
decimals, which reads as model output; minutes and points are integers, transfers carry a
thousands separator, and the price moves carry their sign, since direction is the whole of
what they say. The CSV export writes raw values and is untouched.

**`perM` is gone.** The points-per-£m column divided the window total by the WHOLE price,
so it charged every player for the first ~4.0m that buys nothing a free slot would not have
given you anyway — the near-floor artefact `player_value.py` documents at length. With
`par` on the metric list since 4 Sep, the page carried two columns claiming to answer the
same question, and they disagreed COMPLETELY: over GW3-8 par's top six were Haaland,
B.Fernandes, Thiago, Mbeumo, Semenyo, Raya; points-per-million's were van Ewijk, Guehi,
Thiaw, Thomas, Bassey, Mitchell. Removed as a column, as a filter field and as the
profile's `Value` axis, which is now `par` summed over the window — on the same profile
Haaland moves from near the bottom of the value axis to #1 of 27 forwards. `m:par` takes
its slot in the default column set, so the value question still gets an answer in the
default view, from the construct that answers it. The selftest asserts the string `perM`
appears nowhere in the rendered page, so it cannot come back by accident.

Profile axes are also filtered by availability now: an axis whose column the board does not
## The market-vs-recon A/B was fitting the design the rank guard refuses, 14 Sep 2026

**`src/ab_market_vs_recon.py` deleted.** It could not run on this machine, and routing its
paths through `config` would only have moved the failure later. Recorded here because the
removal is the deliverable.

**What it did.** Fitted `TeamModel` twice - once on `E0_recon`, once on a market-implied
`E0_market` from `betting_odds_ingest` - and reported per-team `d_att` / `d_dfn` plus a
clean-sheet recheck on two fixtures (Chelsea GW8 vs Spurs, Everton GW4 at Spurs), to close
the loop on the pre-season stale-Spurs-prior finding.

**Why it goes, in order of weight.**

1. **It could not tell a rank-deficient fit from an identified one, and its defaults
   assumed the rank-deficient case.** `--market` took any CSV. Fed the market-only E0 that
   `INTEGRATION.md` told you to build (`betting_odds_ingest --out E0_market.csv`), fit B is
   one round: 20 observations against `identifiability.free_params(20)` = **39** free
   parameters, 19 columns short. `oddsapi_feed.py` exists partly to refuse exactly that
   ("refuses to emit a market-ONLY E0 until the accumulated fixture set has full column
   rank... fitting on it yields plausible-looking numbers that are almost entirely prior,
   not market"). Fed a blended E0 instead, fit B is identified but is mostly `E0_recon`
   rows — so the diff against fit A is dominated by the replication ratio, not by the
   market. **The tool ran the same code path either way and never called `rank_report`.**
   Worse, its default `--clubelo-market 0.20` against `--clubelo-recon 0.45` *halves the
   shrinkage on fit B* on the theory that a lower prior weight lets the market drive. With
   the design 19 columns short that is backwards: weakening the prior does not recover
   market information, it lets the unidentified directions move. The headline `d_att` was
   not a repricing estimate under either input.
2. **The reported difference confounded two changes.** Even granting the design, fit A and
   fit B differed in both the E0 *and* `clubelo_weight`. Nothing in the output said so, and
   the two cannot be separated because the weight gap is there precisely to compensate for
   the thin E0.
3. **It was blind to the selection problem its successor refuses on.** `solio_market.stack_e0`
   raises on incomplete coverage, because the source lists are top-10 truncations ranked by
   clean-sheet and attacking output - a censored sample, selected on the dependent variable.
   `ab_market_vs_recon` accepted any E0 and silently diffed whatever teams survived the
   `set(aA) & set(bA)` intersection.
4. **The question is now a calibrated channel, not a one-off A/B.** `market_odds.py`
   (outright market into ClubElo, on at 0.6), `inseason.stack_e0` (realised 26/27 xG, on by
   default at weights fitted in `studies/inseason_weight.py`), and `solio_market.stack_e0`
   (per-fixture market lambda, gated) all answer "does forward-looking information reprice
   att/dfn" on every run, with weights that were fitted rather than asserted.
5. **Its two hardcoded fixtures are spent.** `CS_RECHECK` is GW4 Everton-at-Spurs and GW8
   Chelsea-vs-Spurs. The GW4 deadline passed on 12 Sep (the board was locked 7 Sep,
   `predictions/gw4_board_locked_2026-09-07_early.csv`; GW1-3 are in the scoring ledger and
   GW4 is not yet), so that half is no longer a forward-looking recheck at all. A tool whose
   headline is two hand-picked fixtures chosen pre-season, one of them now behind us, is a
   pre-season artifact.
6. Mechanically: it was a runner living in `src/`, it built `REPO` from a Linux literal, it
   defaulted its inputs to `/tmp/` and wrote to `/mnt/user-data/outputs/`, and its output
   `ab_team_strength.csv` was never a `manifest.py` node - so `doctor` could not have told
   anyone it was stale.

**Not replaced.** Nothing else diffs two fits' team strength, and nothing needs to: the
question it asked is answered before the fit now (coverage + rank report) rather than after
it (a diff of two posteriors). `ab_team_strength.csv` is consequently NOT added to the
manifest and NOT re-homed to `outputs/plans/`; there is no producer left.

**Kept, and fixed while adjacent.** `betting_odds_ingest.py` and `oddsapi_feed.py` stay -
they are the de-vig/inversion and the fetch+rank-guard, both still the right tools. Their
`--out` defaults were the same `/tmp/` violation and now resolve through `config.SCRATCH`.
`oddsapi_feed --recon` deliberately still defaults to `None`, because `None` means
"market-only" and triggers the rank refusal; that is a guard, not a path bug.

**Correcting the record on the home term.** A note carried into this task said the other
five inline home-term sites were routed on 12 Sep and that `ab_market_vs_recon` was the
deliberate exception. The DATE was wrong — there is no 12 Sep commit, and at the time this
deletion was made `docs/TRAVEL_DISTANCE_2026-09-10.md` still read "Not wired, and
pre-dating this change", with all six sites inline. The WORK was real but same-day: a
concurrent session landed it as `a48f4fd` a few hours later, and its draft text carried the
same wrong 12 Sep date, which is the likeliest origin of the note. Both routings are now
merged (see the entry above) and the date is corrected wherever it appeared. So this
deletion removed the sixth inline site, and the other five are closed too — not, as this
entry originally recorded, still open.

## The board is bit-identical, so `d_blended` has no noise floor, 7 Sep 2026
`d_blended` went live when the date rolled and immediately showed six players moving
-0.01 to -0.02, all in GW4, on identical data and code. Either that was Monte Carlo
jitter — in which case the delta column shows phantom movement daily and small values
cannot be read — or it was real.

Chasing it turned up that `test_all`'s DETERMINISM section is `if not quick`, so every
harness run this session skipped it, and its tolerance is `max |delta| < 0.75` — the board
is only asserted deterministic WITHIN Monte Carlo noise, not bit-identical.

Measured directly instead: a second board into a temp `FPL_OUTPUTS`, diffed on
(player_code, gw). **24,814 of 24,814 player-gameweeks exactly equal, max delta
0.000000.** The board is strictly deterministic — `bayes_model._player_rng` did its job
and the 0.75 tolerance is legacy conservatism from before that fix.

So `d_blended` has NO NOISE FLOOR: every non-zero value is a real change. The six movers
were the live FPL availability feed — the board's only non-deterministic input — revising
chance-of-playing overnight between the 22:29 and 06:26 runs, and GW4-only because that
cap applies to the imminent gameweek. The delta column caught a genuine team-news change
on its first day, which is exactly what it is for. Recorded on the metric so a small
value is not dismissed as jitter.

## Squad tab: four-day-old snapshot, and a refresh that would have reverted a transfer, 10 Sep 2026
The squad tab showed a GW3 fetch taken MID-gameweek on 6 Sep: 45 points (final 53),
overall 244 / rank 50,275 (now 248 / 27,933), Sep 6 prices (Groß, Rogers, Isak have each
risen 0.1), "as picked in GW3 with the wildcard chip" over what was really a GW4 squad, and
bank 0.0 where the truth is 0.0-0.3. Nothing on the page said when it was fetched, and the
file was no manifest node, so doctor could not say either.

The obvious fix — re-run `fpl_entry.py` — was a trap. The public picks endpoint publishes a
gameweek only after its deadline and `entry/transfers` lists only confirmed moves, so the
pending Cherki -> Rogers transfer (held as a hand `overlay` inside the meta JSON) would have
been overwritten and silently reverted. Pending transfers now live in
`data/my_squad_pending.json`; `fpl_entry.load()` applies them over every fetch (slot and
armband inherited, bank as a range whose LOW end the transfer search spends, FPL's budget
total preserved) and ignores them once a fetch returns picks for their gameweek. The meta
carries `fetched_at`; the tab prints it, flags a fetch over a day old, names the pending
transfers, and says the XI is the last one FPL published — after a gameweek finishes that
is the lineup AFTER automatic substitutions (O'Reilly benched, Konsa in, for GW3), not a
GW4 lineup, which is private until the deadline. The three squad files are `committed`
manifest inputs of the explorer and the wildcard solve.

**Tested and dropped: per-player selling prices.** Purchase prices are public
(`element_in_cost`), so selling prices looked recoverable. Rebuilt with FPL's
half-the-rise-rounded-down rule they sum to 100.3 against `entry_history.value` 100.7 (and
101.1 at current prices). The method does not reproduce FPL's own number, so it is not
used, and `fpl_entry`'s claim that `value` is the sum of selling prices is now marked
[CHECK]. The tab keeps FPL's aggregate and says £ is not a selling price.

## Stale-inputs audit: three untracked reads closed, one export un-drifted, 10 Sep 2026
Audit of every study and stale artifact against its wiring. **Every tested null is off**
[VERIFIED] — no live module imports a study, and each rejected effect has no code path or a
default-off flag. The findings were all in the other direction: things feeding a decision
that the manifest could not see. Three fixed here; the board is unchanged by them.

**A study file was a live input to the priors.** `defcon_roles.role_map()` read
`studies/defcon_matchups.csv` (literal path, not `config`) inside
`multiseason_priors.to_priors()`, so a study CSV that `test_all` rewrites on every run fed
`ms_priors.pkl` with no manifest edge, and a missing file silently returned `{}` — every
labelled defender moved to the pooled DefCon prior without a word. The labels are now FROZEN
to `data/defcon_roles.csv` (`config.DEFCON_ROLES`, `defcon_roles.py --freeze`), a
`committed` node and an input of `ms_priors.pkl`; `role_map()` raises when it is absent.
Frozen map identical to the live one (146 players, no ties); rebuilt `ms_priors.pkl`
bit-identical to the pre-change pickle.

**The explorer read three files the graph did not know about.** `wildcard_xi.csv` (built
7 Sep, before all four 8 Sep board corrections) and `team_projections_gw1_38.csv` /
`team_projections_season.csv` (8 Sep, stale against the feed) are now derived nodes and
inputs of both explorer outputs, so `doctor` reports the page STALE when any lags. Their
producer defaulted to `GW_HI=10`, which writes a file the explorer refuses (it needs the
board's full window), so the bare run `doctor` prescribes could never have refreshed it:
default is now 38. `export_projection_detail` pins its own 10-week window on the module
rather than relying on the two defaults agreeing.

**`export_projection_detail` described a run that never happened.** Its docstring claims to
mirror the board exactly; it carried three defects the board had corrected: `FPL_SETPIECE`
defaulted to `observed` (the n=1 estimator `penalty_assignment` rejected), `PRED_XI_GW` was
hardcoded to 1 (the board is on GW4), and the XI constraint ran BEFORE availability with no
`hold` (the ordering that cost 220 -> 184.9 starters). All three mirror the board now, plus
`INSEASON_EXP_MINUTES`, the injury ceiling on predicted XIs and the set-piece window. The
open-gameweek derivation moved into `inseason.next_open_gw()` so the two runners call one
function. After rebuild: league `p_start_prior` 220.0, every club 11.0; 20 penalty takers,
the board's 20.

**Board after rebuild.** 91.7% of rows bit-identical to this morning's; every moved row is
at Brighton or Hull, attributable to live availability (Yohanna newly `i`, back 10 Oct;
McNair now available), not to these changes.

**Left open, deliberately.** The Solio blend (`SOLIO` on, `W_OURS=0.5` [JUDGMENT], 30 GW4
players, scored once in GW1 at MAE 1.58 vs 1.59, and `score_gw` does not write the blended
line to the ledger); `gk_rotation` / `def_rotation` are squad planners filed as studies,
with horizons starting at GW1; `export_workbook` / `export_solio_compare` read
`projection_detail` and `team_projections_gw1_10.csv` and are not manifest nodes — the
latter is no longer refreshed by a bare run and will age.

## Captaincy per gameweek, and the armband rotation, 10 Sep 2026
The tail metrics surfaced on 7 Sep answered captaincy for exactly ONE gameweek — the
first unplayed one — because that is the only week `captaincy_tail` was ever called for.
A captain is chosen weekly, so the question has an answer per week, and the question
behind it — "which players do I need to OWN to have a good armband all month" — is a
coverage question over a window that cannot be asked from a single week's numbers at all.

The Players tab now carries a **Captaincy** panel: 1st / 2nd / 3rd per gameweek across the
selected window, `C1`/`C2`/`C3` badges in the per-gameweek grid cells beside the
projection that produced them, and an armband rotation over the window.

THE ROTATION IS `def_rotation`'s ESTIMATOR AT k=1, and nothing new is claimed. Sum over
gameweeks of the best 1 of the N you own. The same argument licenses it: you pick the
captain from the projections before any match is played, so what you get is the score of
the player you actually pick — max(projection) — not E[max], which would assume you knew
the outcome in advance. The baseline is the **best fixed captain over the same window**,
because the thing you would otherwise do is captain your best player every week, so
`gain` is what a second premium buys you. Not to be confused with `rotation multipliers`
(p=0.23), which is a dead claim about MINUTES and stays in the null register; this is the
armband moving between players you own, which is a decision, not a pattern claimed about
a manager.

### Nine of the ten draws dumps in `.cache/` were from a superseded board
`gw_board` dumps ONE gameweek by default and `.cache/draws_gw<N>.npz` has no mtime
relationship to the board that is loaded, so SCRATCH accumulates dumps from every board
ever run. Measured 10 Sep against the current board: **GW4 reconciled to 5e-4** (the CSV's
own 4dp rounding) and **GW1-3 and GW5-10, written eleven days earlier, disagreed by a
median of 0.18-0.33 points and a maximum of 4.21.** Reading those would have ranked
captaincy for GW7 off a superseded model wearing the current board's clothes.

The gate is now `draws.mean(1)` against the board's `mean` for the same gameweek, which is
an identity when the dump came from this board and not otherwise. A dump that fails is
treated as ABSENT — not repaired, not fallen back to a neighbouring week — and the
gameweek ranks on the projection instead, labelled `PROJ` on its own row against `TAIL`.
A tail-ranked week and a mean-ranked week are not two measurements of the same quantity,
and stacking them silently under one heading would make the ordering incomparable across
the very axis the view is built on. `build` names every refused dump and its max
disagreement.

### The top of the ranking is routinely inside the simulation's own error
The finding that changed the design. Both bases are means over S draws and both carry a
standard error — `sd/sqrt(S)` for the projection, `sqrt(p(1-p)/S)` for P(haul) — and at
S=3000:

- **GW8**: 5.739 / 5.566 / 5.517, standard errors ~0.10. C1 to C2 is 0.173 against a
  combined two-sigma of 0.27. **Not separated.**
- **GW4 on the tail basis**: 0.2053 / 0.2050 / 0.2047 against an MC standard error of
  0.0074 — **twenty times the gap that separates them.**
- Over GW4-9, the C1 pick is inseparable from C2 in **2 of 6 weeks**.

The board being bit-identical run to run does NOT rescue this, and I had been reading that
property too generously. Reproducibility is a statement about the seed: it says the same
simulation error is reproduced exactly, not that there is none. Two players a tenth of a
point apart swap places under a different seed and nothing in the board tells you which of
the two orderings you are holding.

So the page prints `±` beside every captaincy number and marks with `≈` any pick it cannot
separate from the one above it, in the table and on the grid badge. S is NOT assumed: it
comes from a reconciled dump — one establishes it for the whole board, since `gw_board`
runs one `DRAWS` for every gameweek — and where no dump reconciles the page makes **no
separation claim at all** rather than guessing at the default. A confidence marker with
nothing behind it is worse than no marker.

### What the panel actually says about this window
Over GW4-9 the best armband rotation is **Haaland + Palmer, worth +1.01 points against
captaining Haaland every week** — one swap, in GW4. Every set of three is that pair plus a
passenger: only 2 of 3 ever take the armband, which the panel says on the row, because a
member who never captains changes the gain by exactly zero and the gain column therefore
cannot show it. The honest reading is that there is **almost no captaincy rotation to do
in this window**, and the panel is built to say so — it names the threshold (1.0 points
over the window) and tells you when the best set is inside it — rather than ranking
something first and letting the sort imply it is worth doing.

The default basis is therefore the PROJECTION, not the tail, even though captaincy is a
tail question: on a board dumped for one gameweek the tail exists for one week of six, and
a control that silently answers a different question for five of them is worse than one
that answers the same question for all six and names it. `DUMP_DRAWS=all` puts the whole
window on the tail, and the page says so where it matters. When no gameweek reconciles at
all, the P(haul) button is disabled and says why, rather than sitting there live and
quietly ranking every week on the projection.

## Captaincy tail metrics surfaced, 7 Sep 2026
`src/captaincy.py` has computed P(haul), ceiling, floor, regret against the template
captain and a differential edge all season. Its only importer was `scripts/legacy/`, so
the board, the explorer and every export ranked captaincy on the MEAN — for a decision
its own docstring calls a tail problem, not a mean problem.

NOT by calling `captain_picks`. That re-derives the draws through its own `point_draws`
with its own RNG, so its `ceiling` would disagree with the `p95` on the same row by pure
Monte Carlo noise — the divergence trap `export_projection_detail` documents. The new
`gw_explorer.captaincy_tail(gw)` reads `.cache/draws_gw<N>.npz`, written by `gw_board`
itself, so every number comes from the same simulation as the rest of the board.
`HAUL_PTS` / `BLANK_PTS` moved to module constants in `captaincy` and are imported rather
than restated, so "haul" cannot come to mean two things.

`DUMP_DRAWS` now DEFAULTS to the projected gameweek instead of nothing. Percentiles do not
add and a joint event — "mine blanks while the template hauls" — cannot be recovered from
summary columns at all, so without the raw draws these metrics are not merely inconvenient
to compute, they are unavailable. ~1MB per gameweek in SCRATCH.

THE KNOWN BIAS, MEASURED RATHER THAN WAVED AT. The engine draws team goals as Poisson and
the league's upper tail is thinner: 4+ goal team-games happen on 4.08% of team-matches
where the model implies 6.54%. That inflates `p_haul` — and NOT uniformly. On GW4,
corr(p_haul, fixture lambda) = +0.34, and the top 25 by `p_haul` sit at mean lambda 1.95
against 1.46 board-wide. The captaincy shortlist is drawn from fixtures **34% above
average lambda**, exactly where the Poisson tail is too fat, so the error concentrates at
the TOP of the ranking and distorts the ordering, not just the level.

Surfaced anyway, uncorrected, with that written on the metric. I had said surfacing it
uncorrected would be worse than not surfacing it; that was wrong. The status quo ranks
captaincy on the mean and ignores the tail completely, and a biased tail metric with its
bias quantified beats no tail metric. `team_explosiveness.apply_tail_calibration` stays
OFF: applying it moves the clean-sheet engine validated at GA r=0.89 / CS r=0.93.

## Week-over-week view, 7 Sep 2026
The weekly decision is MARGINAL — what changed since you last picked — and the explorer
only ever showed levels, so the diff had to be held in your head. Two additions:

**1. `d_blended`, change since the last board.** `gw_board` now writes a compact dated
snapshot to `SCRATCH/board_history/board_YYYY-MM-DD.csv`. The baseline is the newest
snapshot from a PREVIOUS DAY, never the previous run: the board gets rebuilt several times
in an afternoon and diffing against the last run would make the column mean "since I last
pressed go", which is noise dressed as information. Same-day reruns overwrite today's file
and leave yesterday's intact. Additive, so a window sum is the total change to a player's
run. Absent — not zero — when there is no earlier snapshot, because a column of zeros
cannot be told apart from "nothing moved".

Snapshots live in SCRATCH, not OUTPUTS: neither regenerable (a past board cannot be
rebuilt once the data moves) nor precious (losing them costs a convenience view, never a
validation — that is what predictions/ is for).

Verified against a genuine earlier state — the board from before the npxG channel opened.
10,554 of 24,814 player-gameweeks moved, and the largest movers reproduced the npxG
measurement exactly (Yalcouye +0.51, Palacios +0.40, Grimes -0.38 on GW4).

**2. Realised 26/27 facts, five static columns.** Price move this gameweek, price move on
the season, net transfers this gameweek, points, minutes. Straight from `playerstats.csv`,
100% coverage of all 653 board players, exact rather than modelled — and labelled that
way, because at three gameweeks `total_points` is mostly variance while `now_cost` decides
what you can literally afford.

**A UNITS TRAP, caught in the browser.** `now_cost` arrives in millions (12.0) but
`cost_change_event` / `cost_change_start` are in TENTHS. Shown raw beside a price in
millions, Mbeumo's -0.1m read as "-1.00" — an order of magnitude out and plausible enough
to be believed. Converted once at the source.

## The predicted-XI layer had been dark since August, 7 Sep 2026
`PRED_XI_GW` defaulted to a hardcoded `1`. Correct in August; silently wrong from GW2 on.
The board's own log on 6 Sep:

    [pred-xi] GW1 Fantasy Football Scout   2026-08-20
    [pred-xi] applied to GW1 only; GW2+ use the untouched priors

So it was folding in 20 August's team news and running every gameweek anyone was actually
picking on untouched priors. That is the input the project's own scoring calls its largest
error source — correlation roughly halves conditioned on appearing (GW1 r=0.53 -> 0.32),
and GW1 lost 52.6 projected points to 13 availability misses. It failed silently because
the log line looked healthy: it said what it WAS doing, never what it was missing.

`src/fplpage.py` was written specifically to fetch this feed and was **imported by
nothing**.

TWO FIXES:

1. `PRED_XI_GW` is DERIVED, not hardcoded — the first gameweek with nothing played, the
   same marker the explorer and wildcard solver use. A hardcoded default cannot age well
   when the right answer changes weekly. Env var still overrides; falls back to 1 if the
   in-season feed is unreachable, as before.
2. `lock_board --auto` now fetches team news BEFORE rebuilding, since `gw_board` folds the
   predicted XI into the projection it is about to build. Best effort, never fatal: a lock
   WITHOUT team news still beats no lock, but which one happened is printed loudly because
   it changes what the lock is worth.

VERIFIED END TO END. The fplpage path had never run in anger. On GW1 it fetched, OCR'd the
pitch graphic and scraped **220 players**, which `predicted_xi.sources()` then discovered
as a fourth feed (test artefact removed afterwards; GW1 is settled). On GW4 it returns 404
— fpl.page publishes the day before a deadline — handled as the normal early answer and
reported as "the lock will carry NO team news".

## Schedule rank split by end of the pitch, 7 Sep 2026
Team trends showed ONE "Schedule rank". `sosRaw(t, mode)` already knew the difference —
an attacker meets the opponent's DEFENCE, a defender meets its ATTACK, so the two are
different opponents' qualities — and the SoS chart already had the toggle. Only the KPI
and the profile axis hardcoded `"all"`, averaging the two into a number that can be wrong
at both ends at once.

Now two KPIs (`Schedule · attack`, `Schedule · defence`, each with the mean opponent
strength behind it) and two profile axes (`Schedule (att)`, `Schedule (def)`). Home count
moved to its own KPI rather than riding along as a subtitle on a rank it did not describe.

It is not a cosmetic split. Over GW4-9 the two ranks differ by a mean of 3.5 places, up to
7, and only 1 of 20 clubs ranks the same at both ends:

    Chelsea    #3 attack   #10 defence   (old single rank: 5)
    Brentford  #4          #11           (8)
    Arsenal    #15         #8            (12)

The old rank sat between them and misled in both directions — it overstated Chelsea's
defensive run and understated its attacking one, in the same cell.

The profile caption needed no edit: it counts season-level versus window axes from
`team_components` rather than stating a number, so it moved from "the last 4" to "the last
5" on its own. That is the second time that change has paid for itself.

## test_all discovered only half of what it claimed, 7 Sep 2026
`_discover_selftests()` globbed `src/*.py` only, so **three runners shipped selftests the
gate never ran**: `scripts/lock_board.py` (which guards the integrity of the locked
prediction record), `scripts/postgw_review.py`, and `test_all` itself.

CLAUDE.md's convention is that selftests are DISCOVERED and never listed, exactly so that
nothing can be silently left out. The discovery then looked in one directory — the same
class of omission, one level up, and invisible because the count kept rising as `src/`
grew. Found by checking whether the harness that had just passed actually covered the
script it was passing on.

Now globs `src/` and `scripts/`, labelling the latter `scripts/<name>` so the report says
where each came from. `test_all` excludes itself: running the harness inside the harness
is a recursion. 29 selftests -> 31.

## Scheduled lock, 7 Sep 2026 — `lock_board.py --auto`, daily
`--auto` added and wired to a daily scheduled task (`fpl-lock-board`, 05:00 local).

WHY DAILY AND NOT "ON DEADLINE DAY". Deadlines do not fall on a weekday cron can express:
26/27 spreads them across 10:00, 11:00, 12:30, 13:30, 17:30 and 18:30 UTC on several
different days. Anything pinned to a time either fires days early — writing a PRIMARY lock
with none of the team news it exists to capture — or misses the week. So `--auto` is a
STATE test, the same trigger `postgw_review --auto` uses: act only when the next
gameweek's deadline is inside the window and no lock exists. Every other day it prints one
line and exits 0, so the scheduler sees success rather than a daily failure.

`--auto` implies a board rebuild, because the likeliest way to get a useless lock is not
locking late but locking a board built before the last data pull. The rebuild is expensive
and so happens only on the day the lock is actually taken.

TIMING, CHECKED AGAINST ALL 38 GAMEWEEKS. The machine is UTC-4, so the first attempt at
10:00 local (14:08 UTC) would have fired AFTER the 12:30 UTC deadlines and locked most
weeks ~22h early off the day-before run. Moved to 05:00 local (09:08 UTC), which catches
every one of the 38 SAME DAY, lead times 0.9h to 9.4h.

WINDOW WIDENED 24h -> 26h, for one specific hole that the coverage check exposed. The
earliest slot (10:00 UTC, GW6 and GW7) has only 0.9h of same-day lead; if that run is
missed — machine off — the fallback is the previous day's run at 24.9h, which a 24h window
would have REFUSED as too early, and the next run is past the deadline. Those two
gameweeks would have gone silently unlocked, the exact failure this path exists to
prevent. At 26h the fallback covers both tight slots; every other slot has 3.4h+ of
same-day margin and needs none. Cost is at most two extra hours of team news, in the rare
case the fallback is used at all.

## scripts/lock_board.py, 7 Sep 2026 — the missed step, made one command
GW2 has only a reconstruction and GW3 nothing, because locking was a manual copy nobody
performed in time. `python scripts/lock_board.py` now does it, defaulting to the gameweek
FPL flags `is_next` and reading deadlines offline from `gameweek_summaries.csv`.

FOUR REFUSALS, each earning its place:

* **After the deadline.** A board saved at kickoff+1 looks identical to a real lock and is
  worthless — it may carry team news published after the deadline, so scoring it measures
  hindsight. `--force-late` overrides but renames the file `_LATE_not_a_prediction`.
* **A PRIMARY lock taken days early.** This was the sharp one, found by dry-running the
  first version: with `--label deadline` as the default, running it today would have
  written the primary lock 131 hours out — becoming the file `score_gw` scores while
  containing none of the team news a deadline lock exists to capture, quietly undoing the
  pre-declaration made hours earlier. Refused outside `DEADLINE_WINDOW_H = 24`, pointing
  the user at `--label early` instead. `--anyway` overrides.
* **Overwriting an existing lock.** "Never regenerated" is the README's contract.
* **A gameweek the board does not carry** — refused, not written empty.

It also compares the board's mtime against the newest match file and says so: the likeliest
way to get a useless lock is not locking late but locking a board built before the last
data pull, preserving a projection the model had already superseded.

The selftest covers all four plus the dry run, and it caught its own fixture: `before` was
originally set 60 hours from the deadline, which the new primary-lock guard then correctly
refused.

## GW4 locked, 7 Sep 2026 — and the choice removed from scoring time
`predictions/gw4_board_locked_2026-09-07_early.csv`, 653 players, GW4 only. Carries GW3
results and the newly-opened npxG channel. Unlike the GW1 lock it includes `player_code`,
so it joins on the key the rest of the repo mandates rather than on `player` + `team`.

The deadline is 2026-09-12 12:30 UTC — **5.5 days out**, so this lock is missing a week of
team news, injuries, price moves and availability. A second `_deadline` lock will be taken
on the day. GW4 therefore carries TWO, which raises the obvious hazard: choosing between
them once the results are in is choosing the flattering one.

So the choice was removed rather than documented. **The `_deadline` lock is declared
primary as of 2026-09-07, before a ball was kicked, and `score_gw._locked()` enforces it**
— any file matching `*_deadline*.csv` wins regardless of date, verified against a fake
deadline lock deliberately dated eight months EARLIER. A README convention would have
relied on the filenames happening to sort the right way; this does not.

The early lock is not redundant. Scored alongside the deadline one it measures something
never measured here: what a week of team news is worth to the projection.

Why an early lock at all: this step was missed for GW2 (only a reconstruction exists) and
GW3 (nothing), and an unlocked gameweek cannot be scored at all. PROJECT_KNOWLEDGE §6.6
has been the last unticked item all season; a file that exists beats a better file that
does not.

## In-season rate channel: one gate per quantity, 6 Sep 2026
"Turn the current season on and weight it appropriately." The WEIGHT was already
appropriate — `W_RATE = 1.0`, fitted in `studies/inseason_rate_weight.py`, a current
season match worth about one prior-season match, the same answer the team channel reached
independently. What was wrong was the GATE.

`MIN_RATE_MATCHES = 5` applied a single threshold to two quantities that were calibrated
separately and do not clear at the same k:

    npxG/90  k=3  MAE 0.06270 -> 0.05999  CI (-0.00512, -0.00029)   clears
    xA/90    k=3       0.04877 -> 0.04840  CI (-0.00108, +0.00034)   does not

The constant's own comment said it: "xA does not clear the rule at k=3, so the gate sits
at 5". The gate was therefore set by the WEAKER channel, withholding npxG for two
gameweeks in which npxG had been measured to help, because a different quantity was
noisy. Split into `MIN_RATE_MATCHES_NPXG = 3` and `MIN_RATE_MATCHES_XA = 5`, applied per
channel inside `update_rates`. **Neither number is new** — both come straight out of the
table already in the module. `MIN_RATE_MATCHES` stays as an alias for the stricter one,
and an explicit `min_matches=` still forces the old single-gate behaviour.

Live at 3 matches per club: **npxG updated 243 players, xA updated 0**, mean |change| in
npxG/90 = 0.0228. Board effect: mean |change| 0.0255 pts per player-gameweek, 542 of 653
players moved, league total 32453.9 -> 32190.6 (-0.8%). The movement is concentrated in
fringe players whose priors were weakest — biggest risers Palacios +18.6, Yalcouye +14.4
over the season; biggest fallers Jacquet -13.2, Grimes -12.9. A settled XI barely moves:
the live squad's GW4 XI went 48.11 -> 48.12.

`gw_explorer.evidence()` now reports the two gates separately (`npxg_channel`,
`xa_channel`); `rate_channel` is true only when BOTH are open, so no existing reader is
told the channel is on while half of it is still gated. xA opens at GW5.

## GW3 refresh, 6 Sep 2026 — three counting bugs in the scoring/selection path
Pulled the data repo (stale since 1 Sep), rebuilt priors, board and exports. GW3 is
**9/10** finished — Arsenal 2-1 Chelsea was still in play — so it is NOT reviewable yet
and `postgw_review --gw 3` correctly refused. GW2 was reviewed and the ledger refreshed
to final (r=0.622, MAE=1.320, n=612). Three defects surfaced on the way, all in the code
that decides *what counts as a played gameweek*:

1. **`score_gw.played_clubs` counted cup ties as league fixtures.** It read the raw GW
   folder `matches.csv`, which carries cup rows, so GW2 read 20/20 done when the league
   round was 10 fixtures. It also mapped absent EFL team codes to NaN and passed them
   through `norm_team`, which stringifies — putting a phantom club literally named
   `"nan"` in the returned set and a `clubs` count of **21** in the scoring ledger. Now
   filtered on the `-prem-` token like `inseason`/`repo_events`, with unresolved clubs
   dropped as a second guard. The `final` flag was computed on the cup-inflated
   denominator, so a week could have been called final with league games outstanding.

2. **`postgw_review.league_fixtures` had the trap its own docstring warns about.** It
   identified cup ties as "team codes absent from teams.csv", which is true of a tie
   against an EFL club and **false of a cup tie between two Premier League clubs**. GW2
   contained exactly one — `26-27-efl-cup-nottingham-forest-vs-leeds-united` — so the
   round reported **11/11** when the league had 10. Both were finished so the equality
   held by luck; one unfinished all-PL cup tie would have made `--auto` withhold a review
   of a COMPLETE league round indefinitely. Now on the `-prem-` token, and the selftest
   gained that exact case as a regression: an unfinished PL-vs-PL cup tie in the fixture.

3. **`export_wildcard_xi` opened its window on a gameweek nobody could pick for.** Its
   comment claimed "the same marker the explorer uses"; it in fact used the explorer's
   `first_live` (first not-FULLY-played week), not `first_clean` (first week with nothing
   played). On 6 Sep that solved a "GW3 optimal 15" two days after GW3's deadline, for a
   week settled for 18 of 20 clubs. Now skips partly-played weeks too, and solves GW4-15.
   A partly-played week is neither past nor future, but for squad selection it is past.

## Gameweek Explorer, 4 Sep 2026
- **Points above replacement (`par`) wired in** — from `player_value.per_gameweek`, NOT
  reimplemented. Floors recompute inside every gameweek, which is what makes `par`
  additive across a window: each week is measured against that week's own replacement, so
  the sum is total points above replacement over the run. Added to `ADDITIVE`. Eligibility
  for the floor uses `app_ev / 2` as a start-probability proxy so a 4.0m player who will
  never play cannot set it; that proxy is used for the filter only and enters no
  projection.
  The two constructs disagree completely, which is the point — over GW3-8 the top six by
  `par` are Haaland, B.Fernandes, Thiago, Mbeumo, Semenyo, Raya; by points-per-million
  they are van Ewijk, Guehi, Thiaw, Thomas, Bassey, Mitchell, i.e. entirely the near-floor
  artefact `player_value.py` documents. The old `perM` column is left in place rather than
  removed; that is a separate decision. **Taken 7 Sep 2026: removed — see the entry above.**
- **Notebook on the My squad tab** — free-text team strategy plus per-player notes, which
  is the one input the model structurally cannot have. Same storage discipline as the
  existing locks/bans: `localStorage`, keyed on `player_code`, every read and write
  guarded, so a weekly rebuild of the page keeps them. Verified surviving a real
  `export_gw_explorer.py` rebuild.
  Three deliberate choices: a **write probe** at load, because storage that reads but
  cannot write would otherwise fail silently and the status line says so; note text set as
  a DOM property and never interpolated into markup, so a note containing quotes or tags
  cannot break its own row; and import **merges** rather than replaces, so importing on a
  machine that already has notes cannot wipe them. Notes on sold players are kept — the
  reasoning for selling is worth as much as the reasoning for holding.

## Verified, no change needed
- 26/27 position reclassifications are correct in the frame (Wieffer DEF, Sessegnon DEF,
  Lewis-Skelly MID); `DEFCON_THRESHOLD` keys off `pos` — no stale-position scoring bug.
- Team clean-sheet engine already market-calibrated (GA r=0.89, CS% r=0.93) — not recalibrated.

## Deliberately NOT built
- Directional style priors (matchup_design C1/C2/C3) — same class as the null heuristics;
  style unmeasurable for ~55/60 high-leverage GW1-6 fixtures.
- CBIRT xGA downscaling — press-driven, Solio-contradicted; pending a press-index test.
- δ regime mean-pull values — unfitted; wired but off.
- Recalibrating the team model on a single-GW competitor cross-section.

## Provenance
Numbers attributed to Solio are cross-sectional single-GW benchmarks, not ground truth. The
decisive validation is post-GW1 scored data; `older_weight`/Isak needs 2023-24 walk-forward.
