# FPL 2026/27 Projection System — Project Knowledge & Handoff

*Authoritative context document. Supersedes all prior handoffs. Reflects the full build:
live-data de-risking, the signal layers (depth prior, evidence-weighted minutes, regime
variance, appointment panel split, DefCon environment conditioning), the Solio ensemble,
and the validation harness. Paste into a new chat to restore context; the repo + this doc
are the starting point.*

**Persona to restore:** econometrician (Oxford) building FPL projection models. Rigorous;
tests claims empirically rather than asserting; owns mistakes; states confidence and
caveats; distinguishes robust from speculative; free data by default; reports both biased
and corrected estimates when relevant; rejects unvalidated heuristics.

---

## 1. Goal

Hierarchical Bayesian projection for **2026/27** producing posterior-predictive point
distributions per player over any horizon, turned into decisions (hold / captain /
differential). Emphasis on **single-GW accuracy** and **small-sample early season**. Core
philosophy: FPL points are a deterministic function of underlying events — estimate
E[events] with well-specified models, compose through the scoring rules; never regress
total points directly.

---

## 2. Pipeline (frame-build order matters)

```
core_insights.load(base=REPO/2026-2027)
  → two-season pooled priors by player_code (multiseason_priors, older_weight=0.5)
  → apply_coldstart_depth            (ownership-aware start prior for no-history players)
  → apply_regime_panel_split         (optional; appointment-weighted minutes for Man Utd/Spurs)
  → apply_regime_uncertainty         (evidence-weighted minutes shrinkage + regime κ/δ variance)
  → apply_availability               (injuries + confirmed XIs OVERRIDE the priors)
  → pen assignment from penalties_order==1
  → TeamModel.fit(e0_path=E0_recon) → sample_2627
  → apply_defcon_environment         (scale DefCon rate by team xGA environment)
  → project()                        (Monte-Carlo compose; surfaces mean/sd/defcon_ev/cs_ev/p5..p95)
```

Priors set the pre-team-news baseline; injuries and confirmed XIs always override. Team
strength currently comes from **reconstructed 25/26 Opta xG + ClubElo** (not live odds).

**Primary output:** `scripts/gw_board.py` — TRUE week-by-week per-player projections
(project(gw,gw) per GW, not a horizon split), with betting-odds team strength and an offline
GW1 Solio blend (w_ours=0.5 default; inverse-MAE once results land). Writes
gw_board_long.csv / gw_board_wide.csv. Horizon-aggregate runner: `scripts/run_final_board.py` (flags: `DEFCON_ENV` default on,
`REGIME_PANEL=on`, `REGIME=proposed`). A/B harness: `scripts/decision_v2.py`.

---

## 3. Settled decisions (robust)

- **Component model, not points regression.** Estimate events, compose via scoring weights.
- **xG over goals**; "he's due" is a myth (residual has ~zero predictive value).
- **Minutes/availability is the dominant, most-resolvable single-GW lever** (~32% of variance),
  and the single largest *validated* gain in the project (see §5).
- **Rotation multiplier and mean-reversion are tested-null** (p=0.23, p=0.69). Do not build.
- **GK: pick on team defence**; save volume is negatively correlated with points.
- **Per-match priors beat FPL aggregates; two-season priors ~double early-season power**
  (`older_weight=0.5`, folded in).
- **Join on `player_code`** (stable), never `player_id` (reassigned across seasons).
- **26/27 BPS rescale** `{GK:1.019, DEF:0.896, MID:1.045, FWD:1.041}`; DefCon thresholds and
  chips confirmed unchanged for 26/27 (PL, 20 Jul 2026).
- **Captaincy is a tail problem** — full posterior draws, rank on P(haul).
- **Ownership is the strongest predictor after minutes** (Spearman ~0.68 for start-rate) and
  is not otherwise a projection input — clean information for the start-probability layer.
- **Team clean-sheet engine is market-calibrated** (GA r=0.89, CS% r=0.93 vs Solio); do not
  recalibrate it on single-GW competitor numbers.
- **DefCon is repeatable conditional on team environment, not team-invariant** — condition the
  rate on team xGA (§5).
- **Regime change is an ignorance statement** — it belongs in variance (κ), not directional
  style priors.

---

## 4. Modules (this session's additions in **bold**)

**Signal / prior layers (src/):**
- **`starter_prior.py`** — `apply_coldstart_depth` (ownership-aware cold-start start prior),
  `apply_minutes_shrinkage` (evidence-weighted, `w=minutes/(minutes+900)`),
  `apply_regime_uncertainty` (two-knob δ mean-pull + κ variance-inflation via moment-matched
  mixture), `REGIME_2627_PROPOSED` (off-by-default, includes Newcastle).
- **`regime_panel.py`** — `apply_regime_panel_split` (appointment-weighted minutes prior for
  partial-regime clubs Man Utd/Spurs; keyed on player_code; shrinks effective minutes).
- **`defcon_env.py`** — `apply_defcon_environment`: DEF/CBIT scaled by team xGA, MID-FWD/CBIRT
  scaled by press intensity (PPDA via `press_index`). Both channels now conditioned.
- **`market_odds.py`** — betting-odds-implied team strength (26/27 outright markets → market
  Elo, blended into the team model via the ClubElo path). Embedded snapshot + live-fetch stub.
- **`press_index.py`** — manager-aware PPDA; conditions the CBIRT (MID/FWD) DefCon channel.
  `resolve_ppda()` blends the judgment table with measured 26/27 press; `press_factor()` is
  unchanged at every call site.
- **`press_measured.py`** — measures PPDA from 26/27 results and revises the judgment
  forecast at weight `n/(n+40)` (2% at GW1, 20% by GW10). Feed is rebuilt from the FPL repo
  and calibrated onto the Understat scale (r=0.937). Evidence: `studies/press_switchover.py`.
- **`lineups.py`** — source-agnostic XI ingestion (file CSV/JSON or API-Football).
- **`solio_ensemble.py`** — fetch/parse/align/blend/benchmark vs Solio's public feed;
  `benchmark_within_team` (isolates the attacking-share term).

**Core engine (src/):** `bayes_model` (**now surfaces defcon_ev/cs_ev**), `core_insights`,
`roster` (**unique cold-start ids**), `signals`, `betting_features`, `schedule_2627`,
`multiseason_priors`, `multiseason`, `build_pms`, `pms_priors`, `fpl_xp_model`, `captaincy`,
`apifootball`, `fm_priors`, `history`, `fpl_live_data`, `pull_fpl`.

**Scripts:** `run_final_board` (canonical), `decision_v2` (A/B), `cs_fixtures`,
`sweep_older_weight`, `run_solio_ensemble`, `build_all`, `reconstruct_e0`,
`reconstruct_coldstart`, plus legacy runners.

**Tests:** `test_regime`, `test_regime_panel`, `test_defcon_env`, `validate_shrinkage`.

**Studies (null/validation record):** `rotation`, `pit_ownership`, `variance`, `edge_study`,
`multihorizon`, `retest`, `matchup_design` (style constructs — not built, see §7),
`deep_history_study` (8 extra seasons of player-GW minutes — **tested null**, 0.0017 MAE
vs the existing baseline; see docs/DEEP_HISTORY_FINDINGS.md), `setpiece_study`
(component reliability measured; persistence **inconclusive** — see docs/SOCCERDATA_FINDINGS.md),
`late_form_carryover` (late-season surge → next-season start: **tested null**, +0.0000 r²
over full-season strength; new-manager interaction collapses with sample size; a
close-season manager change does NOT reset carryover — see docs/TEAM_FIXTURE_FINDINGS.md),
`early_season_goals` (opening-GW scoring: **no** global early effect — total goals within
1% of the same season's later rate, finishing/CS/dispersion all flat — but home advantage
is only 0.086 goals in GW1-3 vs 0.282 later, CI (−0.374,−0.011), and archetype mismatch
drives goals 3x harder early. **Applied 2026-08-11:** a single-step GW1-3 home discount of
0.152, split symmetrically home/away so the match total is preserved (−0.41%); one step
not a schedule because md4-6 shows no discount and the segment profile is non-monotone.
Board effect confined to GW1-3, home −0.084 / away +0.079, net ~0),
`defcon_matchups` (DefCon flat in opponent strength; CB vs FB 2.3x — see §5),
`defcon_team_matchups` (opponent **identity**, which a strength quartile cannot express.
DefCon hit rate swings **0.129** across opponents, within player — Bournemouth/Leeds/
Liverpool permissive, Fulham/West Ham/Chelsea/Wolves suppressing, and the ordering is NOT
the strength ordering. Split-half r=+0.555, EB shrink 0.68 agreeing independently.
**Shipped** as `defcon_opponent_category` in `team_projections_season.csv`; worth ~0.25
pts/match between extremes vs ~0.87 for the clean-sheet swing, so a **tiebreaker not a
driver**, and it points the SAME way as clean sheets (corr +0.264 with opponent attack) so
there is still no "hard fixture, DefCon floor" trade. MID fails the gate (r=+0.241),
ships flagged unusable. **Matchup (club x opponent) is a null**: it clears a permutation
null but match identity explains more (R2 0.278 vs 0.170) and the two meetings of the same
pair correlate **r=-0.217** — it is match-level shock, not tactics. **Own-team DefCon is
NOT identified** within a season (club nested in player); untested, not dead — needs a
second season of DefCon with transfers),
`team_explosiveness` (are teams differentially explosive? **Two nulls and one reversed
finding.** Per-club dispersion reliability r=+0.006 vs simulated true-Poisson null
(-0.426,+0.480); return concentration r=+0.363 vs shuffled null (-0.415,+0.397),
borderline and inside. But the league's upper tail is **THINNER** than Poisson: P(4+)
observed 4.08% vs implied 6.54%, at the 0.2nd percentile of a simulated null that already
carries the convexity bias. **The engine OVERSTATES blowouts by ~60% relative**, exactly
where captaincy concentrates. `apply_tail_calibration` implements it **OFF BY DEFAULT** —
wiring it in moves the CS engine validated at GA r=0.89/CS r=0.93. Two traps that each
flipped a result: in-sample lambda deflated dispersion to 0.887 vs 1.056 cross-fitted, and
P(4+) convexity in lambda needs a simulated null, not a z-test),
`early_dispersion` (does the model under-disperse strength in GW1-6? residual slope +0.101
CI (+0.002,+0.206) early vs −0.023 CI (−0.056,+0.009) later, difference +0.127
CI (+0.018,+0.235). **Measured, NOT applied** — barely clears zero, third test of the same
question, and worth only ~4% of a team-match. Settle it with an out-of-sample board
backtest, not a fourth in-sample slope),
`minutes_distribution` (starters average 85.3 min not 90 — **bias found and FIXED**),
`minutes_persistence` (conditional minutes are a PLAYER trait: reliability 0.82, persistence
r=0.65; shrunk player history beats the positional constant on out-of-sample MAE 1.968 vs
2.649, CI (+0.579,+0.784), and removes its +1.2 min bias — **shipped as `exp_minutes`**;
all four Solio metrics improve),
`age_minutes` (does age add anything beyond minutes history? **null** — incremental r2
+0.0027, age coef CI (−0.068,+0.203); a 34- and 26-year-old with identical records differ
by 0.5 min. **Do not scrape historical ages**),
`rest_congestion` (fixture congestion: **null**, +0.00000 xG per day of rest advantage,
clustered CI (−0.0043,+0.0040); short-turnaround cut −0.070 xG CI (−0.170,+0.033) n=208),
`fixture_congestion` (the same question re-opened with the instrument the 19 Aug audit
named — actual cup and European fixtures by recovery day, not PL-only rest. The
attenuation was real: **14.3% of club-PL-matches sit in the wrong recovery bucket** when
cups are invisible. Correcting it does not change the answer. GW1-26, established
starters: P(start) at a 3-day turnaround **+0.001, CI (−0.014,+0.019)**; team residual xG
**+0.122, CI (−0.046,+0.287)** — both fail their pre-registered rules and both point the
wrong way for fatigue. **Results too**: points +0.03, CI (−0.238,+0.280); win rate −0.003,
CI (−0.110,+0.099) — though note the MDE on points is 0.40/match, larger than home
advantage, so this is a weaker null than the xG one. **Trap recorded**: the rest
DIFFERENTIAL against a RAW outcome manufactures a large backwards effect (tired side +0.32
points) because the short-rested side is the European side — the quality gap by bucket is a
symmetric ±0.255 ppg. Symmetry of the regressor is not exchangeability of the units;
`rest_congestion` was right to pair the differential with an opponent-adjusted outcome.
Adjusted, it is +0.064 CI (−0.229,+0.352). Mechanism found instead: rotation is spent on the **cup team**, and
that is where competition matters — share of 60+ minute players who are PL regulars is
84.3% in the league, 74.3% CL, 58.2% EL, 46.9% EFL Cup. ≤2-day turnarounds essentially do
not exist (n=1). **Competition main effects are club identity** — the Conference League is
Crystal Palace — and no competition parameter was adopted. GW27+ untested: knockout kickoff
times are 6/50 and the FA Cup is absent from the source entirely. See
docs/FIXTURE_CONGESTION_2026-09-01.md),
`penalty_assignment` (declared `penalties_order` BEATS measured history — 85.7% precision
covering 39.1% vs 45.0%/34.8%; the runners are right to use it, correcting an implication in
SOCCERDATA_FINDINGS §6.4. Real gap is coverage: ~6 clubs have no declared taker),
`midtable_fade` (why prior-7-12 clubs attack worse in GW1-6: effect is stable to
leave-one-season-out but **fails Bonferroni** across the 4 archetypes tested, p=0.039 vs
0.0125 required. Schedule **ruled out** — opponent quality and venue are flat, and the
residual gap is if anything larger. Churn ruled out separately. July-Aug European
qualifying eliminated too — see `euro_qualifying_fade`. **Closed as a probable false
positive; no model change**),
`euro_qualifying_fade` (real July/Aug European participation, not rank as a proxy: within
prior-7-12, treated clubs fade −0.065 and untreated −0.069, difference +0.005 CI
(−0.202,+0.198). Top-6 who played Aug CL play-offs started BETTER (+0.153 vs +0.055).
Fourth and last mechanism eliminated. **NB: leave-one-season-out stability is not evidence
against noise** — a pooled-sample fluke is also LOO-stable),
`sale_hypothesis` (does losing your best attacker cause it? **null at every step** — prior
7-12 clubs are mid-pack on departed xG share (0.198 vs 0.161 top-6, 0.243 for 13-20), the
orderings of departures and fade contradict each other, and controlling for departures
moves the 7-12 coefficient by 0.1%. Three of four mechanisms now eliminated; only European
qualifying survives),
`transfer_churn` (transaction counts 2014-2026 vs the table by month: **null** — +0.98 pts
per SD of churn controlling for prior strength, t=1.08, and no month effect. Directly
measured squad disruption does carry a small season-long cost, clustered CI (−0.113,
−0.005), but it is NOT front-loaded (Aug-Sep interaction t=−0.73) and is confounded by
January signings. **No gameweek-specific churn adjustment**),
`tournament_summers` (WC/Euro summers vs GW1-6: **null on every metric**, exact permutation
p 0.25-0.99, and the mechanism test points the wrong way — but n=5 vs 7 seasons so the
minimum detectable effect is 0.34 goals/match; **no tournament adjustment for 26/27**,
and the null is "not detectable", not "zero"),
`new_manager_debut` (offseason hires vs prior-season strength: +0.081/+0.075/+0.124 over
first 3/6/12, only the 12-match window borderline and the sign test disagrees; **no early
bump inside the GW1-6 horizon and no reset** — price them at prior strength).

---

## 5. Validations (empirical, with confidence)

- **Pipeline faithfulness:** rebuild reproduces prior numbers within MC noise (Haaland 42.3
  vs 42.97; Mbeumo 4.21 ppm @ 13.6% exact).
- **Cold-start depth prior:** established players untouched (|Δ|=0.33 = MC noise); fringe
  cold-start collapses (Alleyne 29.6→11.9); discriminating by ownership (Wilson/van Ewijk held).
- **Evidence-weighted minutes shrinkage — largest validated gain:** vs Solio Pearson
  0.727→0.779, Spearman 0.482→0.609, MAE 0.75→0.70, DEF gap −1.07→−0.86; Mosquera +1.32,
  Gabriel (rich history) +0.02.
- **Regime variance (κ):** baseline identity exact (`regime={}` bitwise); **horizon signature
  passes** — regime SD widens +0.32 at H=6, ~0 (−0.005) at H=1, non-regime flat. Confirms the
  H(H−1) shared-p term propagates (project() draws p once per path, holds across horizon).
  Widening is disagreement-driven `(p_h−p_o)²`, not price.
- **Appointment panel split:** isolation passes (non-partial clubs bitwise unchanged);
  player_code keying matched 41 incumbents (was 2 under a player_id bug); **market agreement** —
  boosted incumbents avg 6.9% own vs cut 1.7% (moves in the direction the market confirms).
- **Team CS engine:** GA r=0.89, CS% r=0.93 vs Solio, bias ≈ 0.
- **Projected set-piece duty (2026-08-11):** FFS 26/27 takers for all 20 clubs, resolved
  within-club (126/130; the 4 misses are players absent from the FPL squad). Overrides FPL's
  `penalties_order` by default since that is a carryover at pre-season —
  `FPL_SETPIECE=fill|off` reverts. Sources agree on 14/20 clubs, disagree on 5. Board:
  Szoboszlai +4.91, Kluivert +3.06 / Robinson −4.51, Hirst −2.04; mean ≈ 0.
- **DefCon matchups (2026-08-11):** the supply hypothesis is WRONG — DefCon rate is flat in
  opponent strength (0.339/0.328/0.416/0.329) while clean sheets collapse 0.344→0.127, so
  total defender EV falls monotonically 2.05→1.17 and hard fixtures pay nothing back. Pick
  defenders on fixture ease. **CBs hit DefCon 2.3× as often as FBs** (0.480 vs 0.207, CI
  +0.239/+0.308) with identical CS value — `PRIOR_DC` is a single 7.6 for all defenders,
  wrong for cold-start CBs and FBs alike, but the fix is circular without a positional
  source. NB `defensive_contributions` is 100% null in 24/25 — never fillna(0).
- **Minutes exposure fix (2026-08-11) — largest correctness gain in the deep-history pass:**
  `project()` assumed every starter plays exactly 90 minutes. Measured over 23,059
  appearances, mean minutes GIVEN 60+ is 85.3, and positional: GK 89.9 / DEF 87.5 /
  MID 83.1 / FWD 81.5, with only 52% of MID and 43% of FWD finishing the match. `m90`
  scales attacking involvement, penalty xG and DefCon, so every starter was inflated.
  Fixed via `MINUTES_IF_START` (`FPL_MINUTES_MODEL=flat` reverts). Board: GK −0.06%,
  DEF −1.36%, MID −3.42%, FWD −4.17%; clean sheets untouched (−0.06%, MC noise);
  **DefCon −13.1%** through threshold convexity. Solio rank agreement improves
  (Spearman 0.540→0.567); level agreement worsens but that comparison is already
  season-mismatched. See docs/PLAYER_LAYER_FINDINGS.md.
- **Calibrated team hyperparameters (2026-08-11):** `revert` 0.85→0.963, `home_prior`
  0.26→0.184, promoted sds 0.30→~0.20, from 31 seasons (`data/team_hyperparams.json`,
  `FPL_TEAM_HYPER=guess` reverts). Team-strength spread widened 1.11× (attack); board
  Spearman 0.9993, mean |Δ| 0.224 over GW1-10, Man City +0.51 / Crystal Palace −0.17.
  Solio agreement improved on all four metrics (Pearson 0.741→0.753, MAE 0.840→0.808),
  n=29 so directional only. See docs/TEAM_FIXTURE_FINDINGS.md.
- **Solio ensemble (live GW1):** two independent sharp models agree Pearson 0.72, MAE 0.72.
  Within-team check flagged Arsenal internal ranking (rho 0.0) that pooled 0.61 hid.
- **DefCon environment conditioning:** Anderson (Forest→City natural experiment) correctly
  untouched on the press-driven CBIRT channel (Solio 56% confirms it holds up); DEF corrections
  sensible (Hull/Coventry defenders up — richest CBIT environments; Guéhi/Khusanov at City down).
  `defcon_ev` now auditable on the board.
- **`older_weight` sweep:** board robust (top-15 overlap ≥12/15); **Isak the single most
  sensitive** (16.9→28.5, range 11.5) — parameter-dependent, unresolvable offline.
- **Market odds blend:** market Elo sensible (Arsenal 1839 top, Hull 1390 bottom); shifts team
  strength toward the market, strengthening promoted-opponent clean-sheet fixtures.
- **Press-conditioned CBIRT:** Anderson held at City (Solio-aligned 56%); Munoz/Fernandes/Touré
  (high-press regime clubs) boosted; deep-block Everton midfielders cut.
- **Minutes-per-start is an in-season signal, and the channel is missing (2026-09-01):**
  `inseason.update_minutes` does a Beta update on realised STARTS and stops there —
  `exp_minutes`, the minutes a player is expected to last GIVEN a start, is touched by no
  in-season path (grep: no hits). So a player withdrawn at half time every week is
  indistinguishable from one who plays every minute; both register as "started" and both keep
  their prior `exp_minutes`. This is the player-specific counterpart of the positional
  **Minutes exposure fix (2026-08-11)** above: `MINUTES_IF_START` corrected the constant, the
  per-player in-season update was never built. It biases twice, since `exp_minutes` scales
  attacking exposure as well as appearance points.
  **Pre-registered and met at every cutoff** (`studies/minutes_per_start.py`, evidence
  `studies/minutes_per_start.csv`). Leave-one-season-out over 2023-24/24-25/25-26, 398 players,
  4,205 player-season-cutoff rows; blend `(1-w)*prior + w*early` against a rest-of-season
  minutes-per-start target. Rule fixed in advance: ship at k only if RMSE falls ≥2% against
  w=0 AND the sign holds in ≥3 held-out seasons.

  | k | 1 | 2 | 3 | 5 | 8 | 10 |
  |---|---|---|---|---|---|---|
  | w | 0.18 | 0.24 | 0.32 | 0.40 | 0.54 | 0.63 |
  | RMSE gain | 5.5% | 7.2% | 9.5% | 12.4% | 15.5% | 18.9% |

- **The start prior is too strong for its evidence — direction verified, constants NOT
  identified (2026-09-01):** `update_minutes` adds ONE pseudo-observation per realised match
  to a Beta prior carrying ~one per prior-season match, so a long-career player arrives with
  ~34 pseudo-matches and two weeks of not being picked move him almost nowhere. Spurs after
  GW2: **Dubravka 0.880 on 34.1 pseudo-matches, 0 starts from 2 → 0.831; Kinsky 0.652 on 13.8,
  2 starts from 2 → 0.696** — the keeper who started both ranked BELOW two who had not played
  a minute, and Dubravka was being offered as a rotation option. `apply_availability` cannot
  catch it: these players are FIT, just not selected. Board-wide 372 players had started none
  of their club's two matches and 139 still carried `app_ev > 0.5`.
  Pre-registered, leave-one-season-out over 2023-24/24-25/25-26, **Brier on 204,682 individual
  remaining matches** (`studies/start_prior_strength.py`, evidence
  `studies/start_prior_strength.csv`). The current corner `w=1, κ=∞` is beaten at every
  cutoff, 3/3 folds: **8.4% / 9.5% / 11.1% / 11.5%** at k = 2/3/5/8.
  **`[CHECK]`, NOT `[VERIFIED]`, on any constant.** The Brier surface is a RIDGE, not a peak —
  at k=5 ten of forty-eight grid points sit within 0.5pp of the optimum, along a diagonal.
  Once a prior is capped at κ, scaling κ and `w` together leaves the prior-mass-to-evidence
  balance unchanged, so **only the ratio w/κ ≈ 0.15–0.40 is identified**. Two endpoints score
  identically and the study cannot choose between them: keep `w=1` and **cap the prior near 5
  pseudo-matches**, or leave it uncapped and **weight a realised match ~8×**. The per-cutoff
  pairs the grid search emits are arbitrary points on that ridge; `ridge()` reports the
  near-optimal set on every run so no precise-looking constant escapes unqualified.
  **Position ordering is a recorded NULL** — see §7. What survives: **goalkeepers are where the
  fault is worst**, gaining 14.9–19.7% against DEF's 5.6–8.6% at every cutoff.
  **Scope**: fits a previous-season prior, not the two-season pooled production prior after
  cold-start fill, depth and regime shrinkage — it validates the weighting PRINCIPLE, not a
  drop-in constant. Cold-start players never enter (no prior season); their start prior comes
  from the ownership calibration, which this does not test. Tzolis is one of those.
  **Interim guard, presentation only:** `gw_explorer` holds players who have started none of
  their club's completed matches out of the likely-starter pool, which feeds the percentile
  reference, the shading scale, the shortlist bands and the rotation enumeration. That is a
  view-layer patch over a model fault, and it is labelled as such on the page.

  All 3/3 folds positive at every k. The weights are far larger than the team channel's
  (0.01–0.25) because minutes-per-start is a direct observation of a player's own role, not a
  noisy proxy for team strength. **It is a tail correction:** at k=2 the median player moves
  1.15 minutes and 0.7% move more than ten; in 26/27 after GW2, 16 of 183 players with 2+
  starts average under 70 min/start. `[VERIFIED]` on the parameter — **`[CHECK]` on points**:
  the study establishes that early minutes-per-start predicts later minutes-per-start, NOT
  that the board scores better, because points depend on minutes through the 60-minute
  threshold and through exposure scaling, neither linear in `exp_minutes`. A points secondary
  was drafted, found to blend a minutes prior against a points target — a scale mismatch that
  answers nothing — and deleted rather than left in looking like evidence. Prompted by Tzolis:
  cold start, p_start 0.97 from the ownership calibration, 2 starts from 2, 120 minutes across
  them, still projecting top-10 for his position.

- **The minutes likelihood is exchangeable and should not be** `[VERIFIED 2026-09-07]`
  (`studies/start_persistence.py`, `studies/start_recency.py`; full writeup
  `docs/START_PERSISTENCE_2026-09-07.md`). 113,571 player-matches, 22/23–25/26, native `starts`.
  **Lag-1:** P(start | started) 0.798 vs P(start | benched) 0.076, pooled gap +0.722 — but the
  *within-player-season* gap is **+0.493** (proxy seasons +0.405). Two-thirds real state
  dependence, one-third frailty. **Duration:** against a permutation null that holds each
  player's own start count fixed and destroys only the ORDER, the excess in P(start at t+1 |
  k consecutive starts) is +0.190 at k=1, +0.153 at k=3, +0.080 at k=6, +0.030 at k=8, and
  indistinguishable from zero from k=10 on (k=20: obs 0.959, null 0.936, CI [0.926, 0.980]).
  Raw h(k) rises to 0.96 by k=20 **and so does the null** — a long streak is a filter that
  selects high-p players, so `h(k) = const` is the WRONG null and testing against it would
  "find" a streak effect that is entirely frailty. **Asymmetry, the usable half:** P(start next
  | k consecutive non-starts) = 0.276 / 0.114 / 0.046 / 0.014 at k = 1 / 3 / 8 / 20. Being
  dropped is far more informative than being picked and saturates far more slowly.
  *Consequence:* NOT a streak covariate — the excess is already zero past k≈8, so one would
  re-encode the base rate the Beta prior holds. The indictment is of the LIKELIHOOD: a
  conjugate Beta update reads a count, so start-start-bench and bench-start-start give an
  identical posterior. Fix is a geometric recency weight `u_d = λ^d` on the estimator's own
  observations, renormalised to the raw match count so it moves only the order and does not
  confound with the `W_MINUTES`/κ ridge. **λ is a function of FORECAST HORIZON** (LOSO, κ=4):
  λ* = 0.40/0.50/0.55/0.65/**0.75**/0.82 at h = 1/2/3/5/**10**/rest, gains +7.9%/+5.6%/+4.0%/
  +2.6%/**+1.2%**/+0.3%, positive in 4/4 folds to h=10. Short horizon, short memory. The
  pre-registered gate (h=1) fires ADOPT at +3.5% but **cannot pin λ** — the gain is monotone to
  the grid floor there, because the last match nearly suffices for the next; that limit is
  right at h=1 and ruinous for a ten-week board. Rest-of-season fails the 1% gate. Shipped OFF
  (`INSEASON_LAM=0.75` with `INSEASON_KAPPA=4`); `[CHECK]` on points, see §6.9.
  *Board effect, GW1-38, decomposed so neither flag is credited with the other's movement:*
  κ=4 alone moves 497/653 players (mean |Δ| 7.75 season points, max 38.1); λ=0.75 **on top of
  κ=4** moves 102 (mean 0.73, max 8.8). κ is by far the bigger lever and the two must never be
  reported jointly. The recency-only moves have the right shape — same-club rotation pairs move
  in OPPOSITE directions (Enzo +8.8 vs O'Reilly −8.8, Foden −8.4 at Man City) — which is the
  exchangeability failure being corrected, visible directly on the board.

---

## 6. Open questions / next steps (ranked by predictive value)

1. **Betting-odds team strength — ADDED** (`market_odds.py`): 26/27 outright markets blended in
   as market Elo. Remaining upgrade: an attack/defence *split* from match/supremacy odds (the
   outright markets give overall strength only), and a live cron on The Odds API.
2. **Live lineups/injuries feed** — layer built (`lineups.py`); needs API-Football key or team news.
3. **2023-24 data** — for walk-forward joint fit of `older_weight` × shrinkage K (resolves Isak).
4. **CBIRT DefCon — press index ADDED** (`press_index.py`): the MID/FWD channel is now
   conditioned on PPDA (Anderson correctly holds at City; high-press-club midfielders boosted).
   The judgment PPDA table now **self-revises against 26/27 results** (`press_measured.py`,
   evidence `studies/press_switchover.py`) — the open item there is closed. Remaining: the
   formal `manager_FE jointly zero` test (§4.4) still needs the fitted regression with the
   press covariate, now available.
   *Partial null recorded:* the switchover weight the evidence supports is `k=40`, not the
   `k=12` that is optimal when the feed is Understat itself. On the feed actually available
   offline, `k=12` **loses** 3.6%. `k=40` is the value non-negative under both backtests; the
   cost of not having Understat 26/27 is roughly two-thirds of the available gain. Refreshing
   the `sd_ingest` Understat cache in-season would let `k` move toward 12 — the single highest-
   value follow-up on this channel.
5. **`market_share × κ` joint sweep** (grid, not coordinate-wise) once GW1 results land.
6. **Fix remaining dedup** to key on `player_code` end-to-end (cold-start ids now unique; the
   `player+team` dedup in some runners should move to id/player_code).
7. **Transfer/chip optimisation solver** — the capability Solio has and we don't.
8. **Build `inseason.update_exp_minutes`, then A/B it on scored gameweeks.** The calibration
   is done and passed (§5, `studies/minutes_per_start.py`): weights 0.18 → 0.63 as k runs 1 → 10.
   What is NOT done is the code path or the proof it helps POINTS. Ship it off by default,
   then run the board A/B — same seed, channel off against on, scored against real gameweeks —
   because the study validates the parameter and not the projection built on it. Blocked on
   nothing; the denominator it needs (`inseason.appearances().start_minutes`, minutes accrued
   in started matches only) already exists.
9. **Re-fit the start-prior strength against the REAL prior, then A/B it.** §5 shows the
   current `W_MINUTES = 1.0` against an uncapped Beta is dominated by ~11% on Brier, but the
   study fits a previous-season prior and only the ratio w/κ is identified. Two jobs, in
   order: (a) re-run the sweep against the actual two-season pooled prior as installed, which
   is what decides whether the fix is a `W_MINUTES` change or a cap on prior strength — they
   are indistinguishable on the approximation and need not be on the real one; (b) board A/B
   on scored gameweeks, same seed. Land this with §6.8 if both are ready, since they touch the
   same channel and a joint A/B avoids attributing one's gain to the other. Highest-value item
   on the minutes layer, which §3 already calls the dominant single-GW lever.
   **Land `INSEASON_LAM` in this same A/B (§5, 2026-09-07).** Recency weighting was fitted at
   κ=4 and reallocates the very evidence whose total weight (a) is re-fitting, so run apart
   each would be credited with the other's gain — the same argument that already ties §6.8 to
   this item. Three levers, one A/B: `INSEASON_W_MIN`, `INSEASON_KAPPA`, `INSEASON_LAM`.
   The prerequisite that used to sit here — a `gw_panel` correction the λ leg needed — is
   **CLOSED [2026-09-08]**, and the diagnosis first recorded for it was WRONG, which is
   worth carrying because the wrong version is the plausible one. The 22 starts that
   `recency_starts` dropped and the raw `starts` column kept were not cup ties: `gw_panel`
   admitted a whole gameweek as soon as ANY match in it had finished, so GW3 carried 22
   half-time starts from an unfinished Arsenal-Chelsea that `played` correctly excluded
   (16 players with `starts > club_matches`, masked all along by `update_minutes`
   clamping). `gw_panel` now applies the finished guard PER MATCH, so raw and
   recency-weighted counts agree at source and λ=1 is a true identity on the raw count.
   The λ-on and λ-off arms of the A/B now differ by recency ALONE, which is what makes the
   A/B clean. Two properties came in with that fix and the λ leg depends on both:
   `recency_starts` keys on `team_now` while placing each start on the club he played for
   that week, so a mid-season mover stays ONE row; and `club_matches` is summed per
   gameweek over the club he was at that week.
10. BPS/DEF refinement: concentrate the −10% haircut on `BONUS_PER_CS[DEF]`; full-back vs
   centre-back CBI split. GK save-metric recompute (low priority).

---

## 7. Deliberately NOT built (with reasons)

- **Directional style priors** (matchup_design C1/C2/C3; "Iraola presses → buy Liverpool";
  "Glasner → buy Forest CBs") — same class as the rotation/mean-reversion nulls. Style is
  collinear with strength in the mismatched fixtures that carry leverage, and per-team state
  proclivity is not identifiable from PL data. The leverage CSV flags ~55/60 GW1-6 fixtures as
  "style unmeasurable". Regime belief lives in κ (ignorance), not directional priors.
- **CBIRT (MID/FWD) xGA downscaling** — press-driven, and Solio's Anderson read (56% DefCon at
  City) contradicts a naive xGA cut. Left untouched pending the press-index test (§6.4).
- **δ regime mean-pull values** — unfitted `[JUDGMENT]`; wired but off. Turning them on shifts
  point estimates on assertion. Sweep against live data first.
- **Recalibrating the team model on Solio** — it's already market-calibrated; that would overfit.
- **Position-specific weight on realised starts** — pre-registered as H-POS in
  `studies/start_prior_strength.py`: goalkeepers should need realised evidence weighted more
  heavily than forwards, because a forward left out of two is rotation while a keeper left out
  of two has been dropped. Tested as "GK's fitted `w` strictly above FWD's in every held-out
  season" and it held in **3 of 12** — NOT SUPPORTED, recorded as a null. Do not reintroduce
  it as "keepers are more nailed" or any other phrasing without a fresh registration.
  The ratio w/κ does look ordered (GK ~0.30–0.40 against DEF ~0.15–0.20) and it would be easy
  to call that a win; it is not one. The comparison was not pre-registered, and
  re-parameterising after seeing the result to rescue a failed test is the exact move this
  register exists to refuse. It is a hypothesis a future study may register, not a finding.
  Note this null does NOT weaken the main result: the overall direction passed its own rule,
  and goalkeepers still show the largest gain of any position — just not via `w`.
- **A `streak_k` covariate on the start prior** — NOT built, and the study that could have
  motivated one is the reason (§5, 2026-09-07). Measured against a frailty-preserving
  permutation null, the streak excess is zero from k=10 on, so a streak term would mostly
  re-encode the player's own base rate — which the Beta prior already holds. That is double
  counting, and it is why the raw h(k) curve rising to 0.96 by k=20 is a **diagnosis and not
  a finding**. `INSEASON_LAM` is not this under another name: it re-weights the ORDER of the
  estimator's own observations and adds no predictor. Nor is it a rotation multiplier — it
  carries no fixture-conditional term, and congestion stays dead three ways over.

---

## 8. Corrections & data facts to carry

- **Salah left the PL** — absent from 26/27 data; do not carry him.
- **Prices/ownership match the official launch** (Haaland £15.5m/70.3%, Bruno £12.0m, Isak £9.0m,
  Gabriel £8.0m, Semenyo £8.5m at City). Ownership is live pre-GW1, not an estimate.
- **Managers:** Iraola→Liverpool, Alonso→Chelsea, Jaissle→Newcastle (5 Aug, ninth regime club),
  Carrick (Man Utd, ~GW20 25/26), De Zerbi (Spurs, ~GW30). Nine full + two partial regime changes.
- **Promoted:** Coventry (Elo 1661), Ipswich (1640), Hull (1533, weakest). Leeds/Sunderland are
  second-season survivors, not promoted.
- **Position reclassifications verified in-frame:** Wieffer MID→DEF, Sessegnon MID→DEF,
  Lewis-Skelly DEF→MID, Kroupi FWD→MID. `pos` drives the DefCon threshold (10 vs 12) and the
  goal/CS multipliers — confirmed correct, not stale.
- **Penalties:** drive from `penalties_order==1` (Isak, Bruno, Saka, Palmer, Haaland order-1).
- **Saliba injured** (status i) — promotes Mosquera/Calafiori into Arsenal's XI.
- **`E0.csv` and `fpl-data-stats.csv` not required** — reconstructed from the repo
  (`reconstruct_e0.py`, `reconstruct_coldstart.py`).
- **PL match filter:** keep only `match_id` containing `-prem-`. **Prices in millions.**
- **Minutes per start needs `start_minutes`, not `minutes`.** Total minutes over starts is wrong
  for anyone who both started some matches and came off the bench in others — the substitute
  minutes land in the numerator with no start in the denominator and the ratio can exceed 90.
  `inseason.appearances` now returns `start_minutes` (minutes accrued in started matches only)
  for exactly this denominator. The per-gameweek read is extracted as `inseason.gw_panel`, with
  `inseason.last_appearance` giving each player his own CLUB's most recent finished gameweek —
  not the league's, which diverges the moment a week splits.
- **FPL only publishes a real `starts` column from 2022-23.** Earlier seasons in `FPL_HISTORY`
  have it inferred from a minutes threshold, which miscounts a 60-minute substitute as a
  starter. Any study conditioning on starting must drop those seasons rather than caveat them —
  `studies/minutes_per_start.py` does, which is why it evaluates three seasons and not eight.
- **Data source:** public **olbauday/FPL-Core-Insights** repo; `core_insights.load` at
  `<repo>/data/2026-2027`.

---

## 9. Environment / provenance notes

- Team strength is reconstructed-from-last-season Opta xG + ClubElo, **not live odds** — the
  standing stale-prior risk on heavy-turnover clubs until the odds feed lands.
- Findings validated against Solio are cross-sectional single-GW checks, not ground truth; the
  decisive validation is post-GW1 scored data and (for `older_weight`) 2023-24 walk-forward.
- All new components ship off-by-default or as validated corrections; unfitted judgment values
  (δ, appointment GWs, w_pre, κ) are sweepable and flagged as such.
