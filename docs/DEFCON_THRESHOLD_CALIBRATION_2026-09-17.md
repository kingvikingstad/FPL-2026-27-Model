# DefCon threshold calibration: per-match overdispersion — 2026-09-17

**Status: OFF BY DEFAULT** (the pre-registered verdict). Switch: `FPL_DEFCON_FRAILTY=on`.
Closes the `[CHECK]` in `DEFCON_SOURCE_CORRECTION_2026-09-10.md` §6, bullet 1.

- Pre-registration: `DEFCON_THRESHOLD_CALIBRATION_PREREG_2026-09-16.md` (commit `d7b1771`, written before any real-data fit).
- Study and evidence: `studies/defcon_threshold_calibration.py` / `.csv`.
- Code: `src/defcon_frailty.py`, `bayes_model._defcon_rng`, `captaincy.point_draws`.

---

## 1. The question

`bayes_model.project()` pays DefCon when Poisson(rate × m90) ≥ 10 for a defender. The rate is a
single Gamma posterior draw per path. So across paths the count is already negative binomial, but
that spread is uncertainty about the rate. It shrinks with minutes: a regular defender has α ≈ 184
and a per-match var/mean of about 1.04. The model has no term for match-to-match variation at a
known rate. The candidate term is a gamma-process frailty:

    count | rate, G ~ Poisson(rate · G),   G ~ Gamma(m90/φ, φ)   ⇒   Var = rate·m90·(1 + rate·φ)

## 2. What the §6 quintile table was actually showing  [VERIFIED]

The table in §6 (low-rate defenders under-predicted 0.043 vs 0.088, high-rate over-predicted
0.576 vs 0.516) banded defenders on raw leave-one-out rates. Two things change once the banding is
fixed:

- Banded instead on held-out predictions (fit GW1-19, score GW20-38), the empirical-Bayes
  Poisson arm **E1 is close to calibrated**: logistic calibration slope **0.98**, lowest quintile
  0.066 predicted vs 0.061 observed.
- So **most of the §6 pattern was regression to the mean** from sorting on noisy rates. What
  survives is over-prediction in the top quintile: 0.530 vs 0.491.

## 3. Result against the pre-registered rule

| Condition | Value | Rule | |
|---|---|---|---|
| S1: φ GW1-19 outside true-Poisson band | **+0.0528**, band [−0.0103, +0.0098] | > 97.5% | yes |
| S1-rep: φ GW20-38 | **+0.0419**, band [−0.0098, +0.0109] | > 97.5% | yes |
| M: mean \|P_D2 − P_D1\| | 0.0210 | > 0.005 | yes |
| ELPD(E2 − E1), count log score | +0.0295/row, **z +4.06** | > 1.645 | yes |
| ΔBrier(E2 − E1) | **+0.00028** (SE 0.00062) | < 0 | **no** |
| ΔBrier(D2 − D1), one-sided 95% upper bound | −0.00044, upper **+0.00055** | < +0.001 | yes |
| Odd/even: ELPD sign | +0.0288 (φ_odd +0.0431) | same sign | yes |

Every VALIDATED condition holds except Brier on the integrated arm. That is the pre-registered
case "φ clearly finite, Brier does not improve", so the verdict is **OFF BY DEFAULT**.

- **Cross-checks on φ.** Dirichlet-multinomial conditional ML gives 0.0543. Net of a block-EB
  opponent rating φ is **0.0472** (band [−0.0087, +0.0099]), so opponent supply does not explain it.
- **φ by group.** CB 0.054, FB 0.048. By rate tercile φ is 0.046, 0.057 and 0.053, roughly flat, so
  variance grows with the square of the rate and the NB2 parameterisation fits. It is also
  0.058 when exposure is each player's mean minutes.
- **Shipping constant.** Fitted on all of 25/26: **φ = 0.0494**, band [−0.0065, +0.0069].

### Why the count improves and the threshold does not  [DERIVED]

Reliability by quintile of D1's predicted P(hit), GW20-38:

| Quintile | n | observed | E1 (Poisson) | E2 (frailty) | D1 (deployed) | D2 (deployed + frailty) |
|---|---|---|---|---|---|---|
| Q1 | 297 | 0.061 | 0.066 | 0.089 | 0.063 | 0.085 |
| Q2 | 297 | 0.135 | 0.146 | 0.173 | 0.144 | 0.171 |
| Q3 | 289 | 0.249 | 0.240 | 0.259 | 0.245 | 0.265 |
| Q4 | 292 | 0.387 | 0.372 | 0.375 | 0.364 | 0.368 |
| Q5 | 293 | 0.491 | 0.530 | 0.506 | 0.563 | 0.533 |

- Counts really are over-dispersed: the log score gains at z = 4 and φ replicates on three splits.
- **Gamma frailty puts the extra spread in the wrong place for low-rate defenders.** It adds
  probability above 10 that the data do not show: Q1 0.089 vs 0.061, Q2 0.173 vs 0.135.
- It does correct the top quintile (Q5 0.530 → 0.506).
- Calibration slopes: E2 1.18, D2 1.06.
- `[JUDGMENT]` The excess variance is probably asymmetric, sitting more in low-count matches than
  in high ones. A gamma frailty cannot express that. A different shape needs its own registration.

## 4. Board A/B  [VERIFIED; pre-registration §3.4]

Setup: same seed, `LIVE_FPL=off`, model `mean`, `FPL_DATA` `b6521ec` in every arm, GW1-38, 25,004 rows.

| Arm | Change | Result |
|---|---|---|
| A0 → R (`3e806da`) | DefCon count moved to a child RNG stream | GK bit-identical; no row of any position with \|z\| > 3; GW5+ app/att/cs means unchanged to 4 dp. **Noise only.** |
| R → Roff | frailty code present, switch off | `gw_board_long.csv` **byte-identical** to R |
| Roff → F | `FPL_DEFCON_FRAILTY=on` | see below |

Roff → F, against the pre-registered checks:

- GK/MID/FWD: all 10 numeric columns bit-identical. DEF app/att/cs/conc bit-identical.
- 65 of 65 DEF likely starters whose analytic move exceeds 2 MC SE moved in the analytic
  direction (rule ≥ 95%). "Likely starter" means E[p_start] ≥ 0.60 `[JUDGMENT]`; 69 players, GW5-38.
- Mean Δ`defcon_ev` per likely-starter gameweek: board +0.00027 vs analytic +0.00015, z +0.29 (rule |z| ≤ 2).
- The analytic reproduces each arm's level: 0.5068 board vs 0.5067 analytic, row correlation 0.9993.

**Effect: redistribution, not level.** DEF likely-starter `defcon_ev` goes 0.5068 → 0.5070 per
gameweek (+0.1%).

- Over GW5-38, high-rate centre-backs lose: Senesi −2.59, Lacroix −2.43, Egan −2.39, Ajayi −2.37, Virgil −2.11.
- Low-rate full-backs gain: Mykolenko +1.74, Mitchell +1.69, Guéhi +1.65, Truffert +1.53.
- Board GW5-38 Spearman 0.9999, top-50 49/50; DEF Spearman 0.9996, top-20 19/20.
- The pre-registration's §0.11 arithmetic expected frailty to "partly reverse the −21%". In
  aggregate it does not: the gains and losses cancel across likely starters.

## 5. Reported, not gating

- **The deployed prior is the larger miscalibration** `[VERIFIED, not decomposed]`.
  - As deployed (D1: k0 = 3, revert 0.70, `exp_minutes`, evidence from all minutes), Brier is
    **0.1695** against **0.1638** for the EB-integrated arm E1. The gap of 0.0057 is about **ten times** the frailty effect.
  - D1's calibration slope is 0.89, and its top quintile predicts 0.563 against 0.491 observed.
  - The pre-registered G0 trigger did **not** fire: slope 0.93, CI [0.80, 1.07], not entirely outside [0.9, 1.1].
  - Two cautions:
    - (i) `revert = 0.70` is a between-season tempering, so a within-season split is not its design target.
    - (ii) D2's non-inferiority "win" partly **compensates** for this gap, since frailty pulls the
      over-predicted top down. **Do not promote the frailty on D2 evidence alone.**
  - The gap needs its own registration: which of k0, revert, exposure and minutes produces it.
- **Drift signature.** The G0 slope is 0.93 on the half split and **1.12 [1.01, 1.23]** on odd/even.
  Selftest check 6 shows planted within-season rate drift produces exactly this ordering.
- **Shared club-match shock.** Pearson residuals correlate **+0.13** within a club-match. `project()`
  draws teammates independently, so the joint tail of stacking two defenders from one club is
  understated whether or not the switch is on. This is outside the scope of this study.
- **Selftest.** Every check passed. False rejection of S1 under true Poisson was 14/200 = 7.0%, exactly
  on the ≤ 7% limit. The selftest bands use 200 simulations, and percentile noise inflates that
  rate; the real-data bands use 1,000.

## 6. What happens next

- **Registered replication (forecast-scorer).**
  - φ = 0.0494 (all of 25/26), scored on 26/27 single-fixture DEF player-gameweeks of 60+ minutes.
  - Counts from `player_gameweek_stats` components, double gameweeks excluded.
  - Run once there are ≥ 1,500 rows.
  - The switch may default on only if D2 − D1 is non-inferior there **and** the integrated-arm Brier
    does not get worse. Otherwise it stays off. If it is still off at the end of 26/27, delete the code path.
- **MID/FWD** (CBIRT, threshold 12) are out of scope. The DEF φ must not be transferred.
- **Does not ship.** Per-player, per-club or per-role φ: that would revive the team-explosiveness null
  at player level (pre-registration §5).
- **Opened, not registered.** The deployed-prior calibration gap (§5), which is larger than anything measured here.
