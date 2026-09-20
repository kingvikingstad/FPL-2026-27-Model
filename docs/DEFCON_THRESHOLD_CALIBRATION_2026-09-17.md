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

**One caveat on that slope** `[CHECK]`. The CB/FB role labels come from `defcon_roles.role_map()`,
which reads `studies/defcon_matchups.csv`, where a defender is split CB/FB on a median of
aerial/crossing volume measured over the **whole** 25/26 season. Headed clearances are part of
CBIT, so test-half outcomes helped choose which prior mean (CB 8.99 or FB 6.29) a player is shrunk
toward. That prior mean is common to all six arms, so it cannot tilt E2−E1 or D2−D1 (the gating
contrasts) and never touches the φ bands. It does make this calibration slope, and the D1-vs-E1
gap in §5, look better than a clean out-of-sample fit would.

## 3. Result against the pre-registered rule

| Condition | Value | Rule | |
|---|---|---|---|
| S1: φ GW1-19 outside true-Poisson band | **+0.0528**, band [−0.0103, +0.0098] | > 97.5% | yes |
| S1-rep: φ GW20-38 | **+0.0419**, band [−0.0098, +0.0109] | > 97.5% | yes |
| M: mean \|P_D2 − P_D1\| | 0.0210 | > 0.005 | yes |
| ELPD(E2 − E1), count log score | +0.0295/row, **z +3.62** | > 1.645 | yes |
| ΔBrier(E2 − E1) | **+0.00028** (SE 0.00073) | < 0 | **no** |
| ΔBrier(D2 − D1), one-sided 95% upper bound | −0.00044, upper **+0.00072** | < +0.001 | yes |
| Odd/even: ELPD sign | +0.0288 (φ_odd +0.0431) | same sign | yes |

**Standard errors.** Clustered on the larger of club-match and player, then multiplied by a
simulated inflation factor c. c is taken as the **larger** of two simulations on the real design,
both holding total dispersion at φ so no source is counted twice:

| | ΔBrier(E2−E1) | ΔBrier(D2−D1) | ELPD |
|---|---|---|---|
| club-match shock only (shock variance 0.0241 from ρ = 0.130) | **1.31** | **1.18** | 1.06 |
| shock + opponent (0.0056) + per-player drift (log-SD 0.15) | 1.28 | 1.06 | **1.13** |
| used | 1.31 | 1.18 | 1.13 |

The first version of this study simulated c with none of that structure and got 1.10 / 1.01 / 1.01,
understating every SE. Adding the opponent and drift terms does **not** raise c further on the
Brier contrasts: taking those sources out of the residual frailty shrinks the per-match term the
contrasts respond to. The non-inferiority bound reaches the +0.001 margin only at c ≈ 1.47, against
1.18 used.

Every VALIDATED condition holds except Brier on the integrated arm. That is the pre-registered
case "φ clearly finite, Brier does not improve", so the verdict is **OFF BY DEFAULT**.

**Design deviation, and the sensitivity that bounds it.** The pre-registration powered the study on
the 143 defenders present in both halves; the run scored all 1,468 test rows. Rows whose player has
no train-half evidence sit on the role prior alone with maximal posterior spread, which is exactly
where a frailty over-predicts the tail — so the widened set biases ΔBrier(E2−E1) **positive**, i.e.
toward OFF. Restricted to the 134 players with ≥ 2 train-half 60+ rows (1,392 rows):

| | full sample | registered sub-sample |
|---|---|---|
| ΔBrier(E2 − E1) | +0.000283 | **+0.000248** (SE 0.000764) |
| ΔBrier(D2 − D1) | −0.000444 | −0.000531 (SE 0.000738) |
| ELPD(E2 − E1) | +0.0295 | +0.0286 (SE 0.00836) |

Every sign is unchanged, so the deviation did not decide the verdict. Reported as a sensitivity, not
a re-decision: the rule was run as written.

- **Cross-checks on φ.** Dirichlet-multinomial conditional ML gives 0.0543. Net of a block-EB
  opponent rating φ is **0.0472** (band [−0.0087, +0.0099]), so opponent supply does not explain it.
- **φ by group.** CB 0.054, FB 0.048. By rate tercile φ is 0.046, 0.057 and 0.053, roughly flat, so
  variance grows with the square of the rate and the NB2 parameterisation fits. It is also
  0.058 when exposure is each player's mean minutes.
- **Shipping constant.** Fitted on all of 25/26: **φ = 0.0494**, band [−0.0065, +0.0069].

### Why the count improves and the threshold does not  [DERIVED]

Reliability by quintile of D1's predicted P(hit), GW20-38:

| Quintile | n | observed (clustered SE) | E1 (Poisson) | E2 (frailty) | D1 (deployed) | D2 (deployed + frailty) |
|---|---|---|---|---|---|---|
| Q1 | 297 | 0.061 (0.014) | 0.066 | 0.089 | 0.063 | 0.085 |
| Q2 | 297 | 0.135 (0.019) | 0.146 | 0.173 | 0.144 | 0.171 |
| Q3 | 289 | 0.249 (0.026) | 0.240 | 0.259 | 0.245 | 0.265 |
| Q4 | 292 | 0.387 (0.029) | 0.372 | 0.375 | 0.364 | 0.368 |
| Q5 | 293 | 0.491 (0.029) | 0.530 | 0.506 | 0.563 | 0.533 |

The E2 gaps in Q1 and Q2 are about 2 SE each, and D1's Q5 gap about 2.5 SE. Nothing here is
established on its own; the table is corroboration for a count-level result, not a finding.

- Counts really are over-dispersed: the log score gains at z = 4 and φ replicates on three splits.
- **Gamma frailty appears to put the extra spread in the wrong place for low-rate defenders**
  `[JUDGMENT]`. It adds probability above 10 that the data do not show: Q1 0.089 vs 0.061, Q2
  0.173 vs 0.135. Read the clustered SEs in the table before leaning on this: each quintile gap is
  only 1-2 SE, the pre-registration put band reliability in report-only precisely because it is
  underpowered, and this is a post-hoc inspection of a table. It is the only threshold-level
  account of why the count gain does not carry, not an established mechanism.
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

- **The deployed COMPOSITION is the larger miscalibration** `[VERIFIED, not decomposed]`.
  Composition, not prior: the gap below is prior **and** expected minutes together.
  - As deployed (D1: k0 = 3, revert 0.70, `exp_minutes`, evidence from all minutes), Brier is
    **0.1695** against **0.1638** for the EB-integrated arm E1: a gap of **+0.00574 (SE 0.00180)**,
    about 3.2 SE and roughly **ten times** the frailty effect.
  - **The E arms get the realised test-half minutes; D must forecast them** from the train half.
    A minutes oracle is therefore the first candidate for the gap, ahead of k0 or revert, and the
    ratio "ten times" compares a confounded estimate against a well-bounded one.
  - D1's calibration slope is 0.89, and its top quintile predicts 0.563 against 0.491 observed.
  - The pre-registered G0 trigger did **not** fire: slope 0.93, CI [0.80, 1.07], not entirely outside [0.9, 1.1].
  - Two cautions:
    - (i) `revert = 0.70` is a between-season tempering, so a within-season split is not its design target.
    - (ii) D2's non-inferiority "win" partly **compensates** for this gap, since frailty pulls the
      over-predicted top down. **Do not promote the frailty on D2 evidence alone.**
  - The gap needs its own registration: which of realised-vs-forecast minutes, k0, revert and the
    evidence window produces it. Start with minutes, by scoring E at `exp_minutes` too.
- **Drift does not contaminate the shipping constant** `[DERIVED]`. Within-window rate drift
  inflates φ̂, and drift over 38 gameweeks exceeds drift over 19, so a drift-inflated estimate
  would put φ_full **above** both halves. It does not: 0.0494 sits between 0.0528 and 0.0419,
  near their mean. That bounds the drift contribution to φ at essentially nil, whatever the G0
  slopes say about the deployed prior's shrinkage.
- **Drift signature.** The G0 slope is 0.93 on the half split and **1.12 [1.01, 1.23]** on odd/even.
  Selftest check 6 shows planted within-season rate drift produces exactly this ordering.
- **Shared club-match shock.** Pearson residuals correlate **+0.130** within a club-match.
  - It is **inside the standard errors**: `shock_split` divides the fitted dispersion into a shock
    variance of 0.0241 and a residual frailty of 0.0287, holding the total at φ so the shock cannot
    double-count, and the SE-inflation simulation carries it (§3).
  - It is **not** in the model: `project()` draws teammates independently, so the joint tail of
    stacking two defenders from one club is understated whether the switch is on or off. Out of
    scope here, and unaffected by this change, since a frailty on each player separately cannot
    create the correlation between them.
- **Selftest.** Every check passed. False rejection of S1 under true Poisson was 14/200 = 7.0%, exactly
  on the ≤ 7% limit. The selftest bands use 200 simulations, and percentile noise inflates that
  rate; the real-data bands use 1,000.

## 6. What happens next

- **Registered replication (forecast-scorer).** Fixed now, so nothing is decided after the fact.
  1. **Sample.** 26/27 single-fixture DEF player-gameweeks of 60+ minutes, counts summed from
     `player_gameweek_stats` components through `defcon_series.fpl_defcon`, double gameweeks
     excluded, nulls dropped (exposure = `defcon_series.exposure`).
  2. **When.** Evaluated **exactly once**, at the first gameweek whose cumulative row count reaches
     **1,500**, and not re-run afterwards. No looking before that.
  3. **Arms.** φ = **0.0494** held fixed from 25/26 (not refitted). The prior is fitted on 25/26
     as deployed; D1/D2 are the deployed composition, E1/E2 the EB-integrated one, exactly as in §3.
  4. **Statistics.** ΔBrier(D2−D1) one-sided 95% upper bound against the **+0.001** margin, and
     ΔBrier(E2−E1) as a point estimate; SE = max(club-match, player) clustered × c, with c
     re-simulated on the 26/27 design through `shock_split` and the opponent/drift terms.
  5. **Outcome, fixed in advance.**
     - φ̂(26/27) **inside** its own simulated null band → **delete the code path**, whatever the
       Brier gates say. The dispersion did not replicate.
     - φ̂ outside the band, D2−D1 upper bound < +0.001, **and** ΔBrier(E2−E1) ≤ 0 → default it **on**.
     - φ̂ outside the band but either Brier gate fails → it stays **off**.
     - Still off at the end of 26/27 → **delete the code path**.
- **MID/FWD** (CBIRT, threshold 12) are out of scope. The DEF φ must not be transferred.
- **Does not ship.** Per-player, per-club or per-role φ: that would revive the team-explosiveness null
  at player level (pre-registration §5).
- **Opened, not registered.** The deployed-composition calibration gap (§5), which is larger than
  anything measured here.
- **A null of power, recorded so it is not re-attempted** `[VERIFIED]`. Threshold-level Brier
  *superiority* for a per-match frailty is not measurable at one season's n. The pre-registration
  put the minimum detectable difference at 0.0013-0.0028, reached only near φ ≈ 0.11, and computed
  that 80% power at r = 20 needs about 9,800 player-matches, roughly three seasons. The realised
  SE confirms it: the observed ΔBrier(E2−E1) of +0.00028 is about 0.4 SE. So OFF was reached by the
  pre-registered tie-break to the status quo on a statistic the design already said could not give
  an informative answer. **It is not evidence that the frailty harms threshold prediction.** Any
  future study must treat this gate as non-inferiority, never superiority, on a single season.
