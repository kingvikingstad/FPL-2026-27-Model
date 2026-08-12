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

---

## 6. Open questions / next steps (ranked by predictive value)

1. **Betting-odds team strength — ADDED** (`market_odds.py`): 26/27 outright markets blended in
   as market Elo. Remaining upgrade: an attack/defence *split* from match/supremacy odds (the
   outright markets give overall strength only), and a live cron on The Odds API.
2. **Live lineups/injuries feed** — layer built (`lineups.py`); needs API-Football key or team news.
3. **2023-24 data** — for walk-forward joint fit of `older_weight` × shrinkage K (resolves Isak).
4. **CBIRT DefCon — press index ADDED** (`press_index.py`): the MID/FWD channel is now
   conditioned on PPDA (Anderson correctly holds at City; high-press-club midfielders boosted).
   Remaining: the formal `manager_FE jointly zero` test (§4.4) still needs the fitted regression
   with the press covariate, now available.
5. **`market_share × κ` joint sweep** (grid, not coordinate-wise) once GW1 results land.
6. **Fix remaining dedup** to key on `player_code` end-to-end (cold-start ids now unique; the
   `player+team` dedup in some runners should move to id/player_code).
7. **Transfer/chip optimisation solver** — the capability Solio has and we don't.
8. BPS/DEF refinement: concentrate the −10% haircut on `BONUS_PER_CS[DEF]`; full-back vs
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
