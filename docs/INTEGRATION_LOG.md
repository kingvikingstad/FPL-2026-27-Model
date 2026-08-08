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
