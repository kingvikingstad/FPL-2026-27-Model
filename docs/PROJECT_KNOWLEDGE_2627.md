# FPL 2026/27 Projection System — Project Knowledge & Handoff

*Authoritative context document. Supersedes the prior handoff by folding in the live-data
de-risking, the new signal layers, the Solio ensemble, and the validated team model.
Paste into a new chat to restore full context; re-upload the repo zip + module set + this file.*

**Persona to restore:** econometrician (Oxford) who builds FPL projection models. Rigorous,
tests claims empirically rather than asserting, owns mistakes directly, gives honest caveats
and confidence levels, distinguishes robust findings from speculative ones. Free data by
default; report both biased and corrected estimates when relevant.

---

## 1. Goal (unchanged)

Hierarchical Bayesian projection system for **2026/27** producing posterior-predictive point
distributions for every player over any gameweek horizon, turned into decisions: who to hold,
who to captain, where the differential edge is. Emphasis on **single-gameweek accuracy** and
**small-sample (early-season)** performance. Core philosophy: FPL points are a deterministic
function of underlying events — estimate E[events] with well-specified models and compose
through the scoring rules; never regress total points directly.

---

## 2. What this session did (update log)

1. **De-risked the foundation against live 26/27 launch data** (game went live ~22 Jul 2026;
   GW1 deadline 21 Aug). Verified prices, ownership, transfers, promoted teams, injuries, and
   the 26/27 BPS rule changes directly against the repo data and public sources.
2. **Corrected two earlier claims of mine** (see §9). Salah is *absent* from the 26/27 data;
   Haaland's 70.3% is *live* ownership, not a pre-launch estimate.
3. **Reconstructed the two missing inputs** faithfully so the pipeline runs end-to-end without
   the un-uploaded files: `E0.csv` from the repo's real 25/26 Opta match xG (fair-odds Poisson
   inversion), and the cold-start calibration from the per-match panel.
4. **Ran the full pipeline and reproduced the prior handoff numbers within Monte-Carlo noise**
   — confirming the reconstructions are faithful, not a divergent rebuild.
5. **Built an ownership-aware cold-start depth prior** (`starter_prior.py`) — collapses
   cold-start × strong-team inflation.
6. **Built a source-agnostic lineup/injury layer** (`lineups.py`) feeding `apply_availability`.
7. **Ran the `older_weight` sensitivity sweep** — board robust except Isak.
8. **Surveyed the FPL analytics landscape** (Solio, FPL Review, OpenFPL) and **built a Solio
   ensemble + benchmark layer** (`solio_ensemble.py`).
9. **Diagnosed and fixed a minutes-prior weakness** surfaced by the ensemble, **validated the
   team model's clean-sheet engine against the market**, and added **evidence-weighted minutes
   shrinkage**.
10. **Produced the GW1–10 clean-sheet fixture ranking** on the validated model.

---

## 3. Current state

The system runs end-to-end on **real 2026/27 data** (555 players, 20 teams, nothing played yet).
`run_2627.py` / `final_ms.py` build the frame the same way: `core_insights.load()` → two-season
priors by `player_code` → cold-start depth prior → **minutes shrinkage** → `apply_availability`
(injuries / lineups) → pen assignment from `penalties_order==1` → `TeamModel.fit` → `sample_2627`
→ `project`. Team strength currently comes from **reconstructed 25/26 Opta xG + ClubElo**
(betting-odds path exists but is not yet fed live 26/27 odds).

**Current best board (GW1–6, two-season priors, depth prior + minutes shrinkage):**
Haaland ~41.8, B.Fernandes ~33.9, Mbeumo ~33.4, Semenyo ~29.0, Saka ~28.9, Palmer ~28.8,
Gabriel ~27.6, Thiago ~27.2, Guéhi ~27.1, O'Reilly ~26.5, Isak ~25.1 *(parameter-dependent — see §7)*.

**GW1 captaincy:** Haaland EV 14.85 / haul 29.2% (70–73% owned → rank-neutral); Mbeumo best
value premium (4.21 pts/£m @ 13.6%). Matches both the prior handoff and Solio's live read.

---

## 4. Settled decisions (carry forward)

All prior decisions stand. The essential ones, plus this session's additions:

- **Component model, not points regression.** Estimate events, compose via scoring weights.
- **Trust xG over goals**; rolling xG+xA is the best haul predictor. "He's due" is a myth
  (goals-minus-xG has ~zero predictive value; underperformers don't bounce).
- **Minutes/availability is the dominant, most-resolvable single-GW lever** (~32% of variance).
- **Rotation multiplier does NOT exist** (tested null). Do not build one.
- **GK correction:** save volume is *negatively* correlated with points; xGOT-faced is the real
  signal. Pick keepers on team defence.
- **Ownership is the strongest predictor after minutes** (Spearman ~0.48 point-in-time; ~0.68
  for start-rate specifically). It is *not* otherwise a projection input, so it is clean
  information for the start-probability prior (§5).
- **Per-match data beats FPL aggregates**; **two-season priors ~double early-season predictive
  power** (folded in with `older_weight=0.5`).
- **Join on `player_code`** (stable), never `player_id` (reassigned) or name (collisions).
- **26/27 BPS rescale** by position (`BPS_2627_MULT = {GK:1.019, DEF:0.896, MID:1.045, FWD:1.041}`).
  **NEW — validated & extended this session** (§6).
- **Captaincy is a tail problem** — full posterior draws, rank on P(haul), risk profiles.
- **Free-data default, paid as opt-in hooks.**
- **NEW — cold-start start probability is ownership-aware** (depth prior).
- **NEW — the minutes prior is evidence-weighted**: trust history in proportion to sample size,
  shrink toward the market-implied (ownership) prior when history is thin.
- **NEW — team model clean-sheet engine is validated against the live market** (§6); do not
  recalibrate it on single-gameweek competitor numbers (overfitting risk).

---

## 5. New modules built this session

| Module | Purpose |
| --- | --- |
| `starter_prior.py` | Ownership-aware cold-start depth prior (`apply_coldstart_depth`) **and** evidence-weighted minutes shrinkage for thin-history established players (`apply_minutes_shrinkage`). Calibration fit from 25/26: logit P(start) ~ log(own) + price, per position. |
| `lineups.py` | Source-agnostic XI ingestion → `{team:{start,bench}}` for `apply_availability`. `from_file` (CSV/JSON of predicted or manual XIs), `from_apifootball` (confirmed XIs when keyed), dispatcher `load_lineups`. |
| `decision_v2.py` | Integrated runner: API-Football injury overlay + lineup override + depth prior + minutes shrinkage, in the correct order (priors set baseline; injuries/XIs override). Env: `LINEUPS_PATH`, `APIFOOTBALL_KEY`, `DEPTH_OFF=1`. A/B built in. |
| `solio_ensemble.py` | Fetch/parse/align/blend/benchmark against Solio's public feed (`.md`, free, no-auth, 4h refresh). Team-aware name matching. Reproduces Solio's own leverage ranking as a correctness check. |
| `sweep_older_weight.py` | `older_weight` sensitivity sweep (fixed team draws + RNG → isolates the prior). |
| `cs_fixtures.py` | GW1–10 clean-sheet fixture ranking on the validated team model. |
| `validate_shrinkage.py` | A/B of the minutes-shrinkage update against the Solio benchmark. |

**Reconstruction helpers (this session, one-off):** `E0_recon.csv` from repo 25/26 `matches.csv`
Opta xG via fair-odds Poisson inversion; `coldstart_hist.csv` from the per-match panel
(`npxg`, `xa_`, `defcon_raw`). Both feed the existing pipeline unchanged via path/arg swaps.

**Order of operations in the frame build:** cold-start depth prior → minutes shrinkage →
`apply_availability` (injuries + confirmed XIs) → pen assignment. Confirmed XIs and injuries
always win; the priors only set the pre-team-news baseline.

---

## 6. Validations & findings (empirical, this session)

**Pipeline faithfulness.** Rebuild reproduces the prior handoff within MC noise: Haaland 42.30
vs 42.97, Bruno 34.30 vs 33.75, Mbeumo 33.31 vs 33.68, Saka 29.08 vs 28.95, Palmer 28.53 vs
28.54, Gabriel 27.21 vs 27.88. GW1 captain EV 14.85 vs 14.72; Mbeumo 4.21 ppm @ 13.6% (exact).

**BPS 26/27 — confirmed and extended.** The three assumed changes are real: being-tackled −1
removed, CBI 1-per-2 → 1-per-3, +1 big-chance save. **Two GK-only changes the prior handoff
missed:** save-outside-box +2 *removed*, "any other save" +2 *added* (pen save 8→7, net 8 with
big-chance) — GK multiplier 1.019 is directionally right but computed without this swap; low
priority. DefCon thresholds unchanged; chips unchanged (2 sets). **Magnitude anchor:** Gabriel
30→20 bonus (−33%, worst case) validates DEF avg −10.4%. **Refinement (open):** the DEF loss is
bimodal — full-backs *gain* from tackle-penalty removal, centre-backs lose; concentrate the
haircut on `BONUS_PER_CS[DEF]`, not uniformly, and consider a per-player CBI-propensity split.

**Cold-start depth prior — validated.** Calibrated on Spearman(own, start_rate)=0.68 (0.62 among
≤£5.5m). A/B: established players mean |Δ|=0.33 pts (untouched); cold-start fringe collapses
(Alleyne 29.6→11.9, Vitor Reis 28.0→11.7); discriminating, not blunt (Wilson/Leeds 10.7%,
van Ewijk/Coventry 14.5% held up). It penalises *fringe* (revealed by ownership), not *newness*.

**Lineup override — demonstrated end-to-end (offline, via file loader).** Named starter → 0.99;
benched → sub-rate; XI-excluded → 0.05. API-Football path wired and ready (not executed here —
no sandbox network).

**`older_weight` sweep.** Board robust: top-15 overlap with w=0.5 never < 12/15 (13–14 in the
0.25–0.75 band). Two archetypes drive all sensitivity — players who rise with more 24/25 credit
(Isak, Wissa, Madueke, Havertz) and those who fall (Wieffer, Anderson, Tarkowski). **Isak is the
single most sensitive player:** EV 16.9 (w=0) → 25.6 (0.5) → 28.5 (1.0), range 11.5 — from
outside the top 12 to 7th overall. Not resolvable offline without 2023-24 (walk-forward) or live
26/27 games.

**Solio ensemble — live GW1 head-to-head.** Two independent sharp models agree strongly:
Pearson 0.72, MAE 0.72 pts across 30 GW1 players; our top four match Solio's. Cross-checks
confirmed our foundation (prices, fixtures, club assignments, Haaland captain 14.85 vs 14.48).

**Team model clean-sheet engine — validated against the market.** Our GW1 per-team goals-against
and CS% match Solio's: GA correlation **0.89**, CS% correlation **0.93**, bias near zero (−0.05
GA, +0.03 CS%). The team model needed **no** recalibration.

**Minutes shrinkage — the diagnosed fix, validated.** The ensemble's ~0.4 uniform + extra ~0.6
DEF gap traced *not* to the team model but to the backward-looking minutes prior (Mosquera at
P(start)=0.38 despite Saliba's injury making him nailed). Evidence-weighted shrinkage
(`w_hist = minutes/(minutes+900)`) improved every benchmark metric: Pearson 0.727→0.779,
Spearman 0.482→0.609, MAE 0.75→0.70, DEF gap −1.07→−0.86; Mosquera +1.32, Gabriel +0.02
(rich history untouched). Residual Mosquera gap needs a confirmed XI.

**Notable live differentials (same-fixture, vs Solio):** Mbeumo — we rate +1.55 *above* Solio
(our signature edge, or an over-projection to pressure-test). Mosquera — we rated −2.97 *below*
(now corrected; residual is the Saliba-injury lineup effect). Murillo — we rate below Solio on an
FPL fitness flag Solio appears to miss (a case where we're likely right).

---

## 7. Open questions / next steps (ranked by predictive value)

1. **Live 26/27 betting-odds team ratings — the biggest remaining gain.** Currently team strength
   comes from *last-season* Opta xG + ClubElo. Live match odds (1X2 + O/U 2.5, de-vigged) price
   in new managers (Iraola→Liverpool, Alonso→Chelsea), Salah's exit, and the promoted sides that
   have no PL data. `betting_features.build()` already contains the de-vig→xG machinery; this is a
   **feed swap, not new architecture**. Needs an odds source.
2. **Live lineups/injuries feed.** Layer is built (`lineups.py`, `apply_availability`); needs the
   feed (API-Football key, or a team-news source). Closes the Mosquera-type residual directly.
3. **2023-24 season data** to jointly fit `older_weight` and shrinkage K via walk-forward — the
   only way to resolve the Isak parameter dependence offline.
4. **Live FPL API refresh** once GW1 plays (prices/ownership/availability drift immediately).
5. **Transfer/chip optimisation solver** — the capability Solio has and we don't (MILP over the
   branching GW tree: budget, 15-man, 3-per-club, hits, chip timing). Turns projections into plans.
6. **BPS/DEF refinement:** concentrate the −10% haircut on `BONUS_PER_CS[DEF]`; full-back/
   centre-back CBI split. GK save-metric recompute (low priority).
7. **Player-prop odds** — *lower priority than previously assumed*: FPL Review's backtests
   indicate a well-built component goalscoring model beats player prop odds, so props are a
   validation/ensemble input, not a strict upgrade. Team-level odds (item 1) remain valuable.

---

## 8. Landscape notes (what the sharp public tools do)

- **Solio Analytics** — market odds + stochastic component model (our methodological twin), plus
  a full transfer/chip optimisation solver and a free public data endpoint
  (`https://fpl.solioanalytics.com/api/data/latest.md`, 4h refresh). Now a live benchmark and
  ensemble partner. Rolled to 26/27 GW1 on 2026-07-25.
- **FPL Review "Massive Data"** — multi-season weighted goalscoring model (weight fresher/larger
  samples, condition on team quality + role/set-piece hierarchy) that *beats bookmaker player
  odds* at player level. Validates our approach; deprioritises prop odds.
- **OpenFPL** (open-source, May 2026) — a free-data ensemble rivals commercial services. Lesson:
  **ensembling reliably reduces single-GW error** — hence the Solio blend.

---

## 9. Corrections & data facts (must carry forward)

**Corrections to earlier claims made this session:**
- **Salah is absent from the 26/27 data** — he left the Premier League. The frame does *not*
  carry him. (Earlier warning that it did was wrong for this snapshot.)
- **Haaland's 70.3% is live pre-GW1 ownership**, not a stale pre-launch estimate. Ownership
  populates immediately at launch.

**Verified 26/27 facts (live launch data):**
- Prices/ownership match the official launch: Haaland £15.5m (70–73%), B.Fernandes £12.0m,
  Saka £9.5m, Gabriel £8.0m (big riser), Isak £9.0m (−£1.5m, injury-hit 25/26), Semenyo £8.5m
  (Man City), Palmer £9.5m, Szoboszlai £7.0m (39% — the templated Liverpool pick), Guéhi £6.0m,
  O'Reilly £6.5m (all Man City).
- **Penalties:** drive assignment from the `penalties_order` column. Order-1 takers include
  Isak (LIV), B.Fernandes (MUN), Saka (ARS), Palmer (CHE), Haaland (MCI). Szoboszlai is LIV
  order-2 + free-kicks.
- **Managers:** Iraola → Liverpool (high press; bullish for LIV attack, magnitude speculative);
  Alonso → Chelsea (Palmer revival narrative).
- **Promoted 26/27:** Coventry (Elo 1661), Ipswich (1640), Hull (1533 — materially weakest).
  **Leeds and Sunderland are second-season survivors, not promoted** — they keep established
  priors. **Relegated 25/26:** Burnley, West Ham, Wolves.
- **Saliba injured** (status `i`, 0%) — this promotes Mosquera/Calafiori into the Arsenal XI
  (the reason the minutes-shrinkage fix matters).

**Data/environment facts:**
- Repo (`FPL-Core-Insights-main.zip`) is ~32MB, 3 seasons. `data/2026-2027/` has GW1–38 folders
  but `playermatchstats` are empty (season not started). 24/25 uses the flat
  `playermatchstats/GW{n}/` layout; 25/26 uses `By Gameweek/GW{n}/` + `By Tournament/`.
- `core_insights.load(base=...)` must point at `.../data/2026-2027` (loose CSVs live in the repo,
  not uploads).
- **`E0.csv` is not required** — reconstruct from repo 25/26 `matches.csv` Opta xG.
- **`fpl-data-stats.csv` is not required for priors** — reconstruct the cold-start input from the
  per-match panel; its use in `build_pms`/`roster` is `try/except` or replaceable.
- **PL match filter:** keep only `match_id` containing `-prem-` (cup/Europa rows leak otherwise).
- **Prices are in millions** everywhere; guard `np.where(cost>30, cost/10, cost)`.
- **Sandbox constraint (this session):** bash has **no network egress**; `web_fetch`/`web_search`
  work (Claude-side). The Solio module fetches live on your machine and falls back to a cache in
  the sandbox.
- **Known minor bug:** the projection dedups on `player`+`team`; it should key on `id`
  (surfaces as a duplicate "Palmer" row). One-line fix.

---

## 10. Deliverables produced this session (in `/mnt/user-data/outputs/`)

**Code:** `starter_prior.py`, `lineups.py`, `decision_v2.py`, `solio_ensemble.py`,
`run_solio_ensemble.py`, `sweep_older_weight.py`, `cs_fixtures.py`, `validate_shrinkage.py`.

**Data/boards:** `decision_gw1_6_depthprior.csv` (corrected board), `older_weight_sweep.csv`,
`solio_ensemble_demo.csv`, `cs_fixtures_gw1_10.csv`, `solio_cache.md` (26/27 GW1 feed snapshot).

**Headline analytical output — Top 10 clean-sheet fixtures for defenders, GW1–10:**

| GW | Fixture | Exp. GA | P(CS) |
| --- | --- | --- | --- |
| 10 | Arsenal vs Hull (H) | 0.50 | 62% |
| 1 | Arsenal vs Coventry (H) | 0.61 | 56% |
| 8 | Arsenal vs Everton (H) | 0.70 | 51% |
| 7 | Man City vs Ipswich (H) | 0.73 | 50% |
| 6 | Arsenal vs Leeds (H) | 0.73 | 49% |
| 3 | Man City vs Coventry (H) | 0.75 | 49% |
| 7 | Fulham vs Hull (H) | 0.79 | 47% |
| 5 | Man City vs Sunderland (H) | 0.78 | 47% |
| 4 | Chelsea vs Hull (H) | 0.81 | 46% |
| 4 | Arsenal vs Sunderland (A) | 0.80 | 46% |

**Best defences to target over the run:** Arsenal (mean CS 46%, ~4.6 expected clean sheets,
9/10 fixtures strong) is the standout; Man City second (~3.8); then a clear drop to Fulham /
Man Utd / Everton (~2.9–3.0). Actionable read: double up on Arsenal defenders (Gabriel, plus the
now-nailed Mosquera/Calafiori with Saliba out) and Man City cover. Caveat: our elite-defence CS
runs marginally *under* the market (Arsenal-Coventry 56% vs Solio 61%), so the ranking order is
robust and the top numbers are, if anything, conservative.
