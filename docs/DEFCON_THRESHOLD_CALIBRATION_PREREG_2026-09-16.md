<!-- Pre-registration written by research-preregistrar 2026-09-16, BEFORE any real-data fit.
Saved verbatim. Code claims spot-checked by the main session: captaincy.py:107 duplicate
composition [VERIFIED]; defcon_env scales alpha only [VERIFIED]; to_priors k0=3.0,
revert=0.70 [VERIFIED]. Synthetic power scripts live in the session scratchpad only. -->

# PRE-REGISTRATION: DEF DefCon threshold calibration (per-match overdispersion)

Opens `docs/DEFCON_SOURCE_CORRECTION_2026-09-10.md` §6 bullet 1 `[CHECK]`. No real data was fitted to write this. Every number below is from the repo's code or docs, is a design count you supplied, or comes from synthetic simulation and is marked as such.

```
PROPOSAL: Estimate a pooled per-match gamma frailty φ for DEF CBIT counts within
  each player, and test whether adding it to the deployed Gamma-Poisson makes held-out
  DefCon threshold predictions (P(>=10)) more accurate.
NULL-REGISTER CHECK: untested-adjacent to team explosiveness (E1 per-club dispersion
  null; E2 league tail, VERIFIED thinner). Not a revival (see §5).
ESTIMAND: φ in Var(Y_ij | λ_i) = λ_i m_ij (1 + λ_i φ), on 25/26 DEF, CBIT, 60+ min,
  measured rows. Ship target is ΔBrier = E[(P_D2−H)² − (P_D1−H)²] on GW20-38.
INSTRUMENT: within-player Pearson moment estimator with player fixed effects, fitted on
  GW1-19 (build_pms defcon_fpl, exposure from defcon_series.exposure). Confounds:
  club-match shared shock, opponent supply, minutes variation, drift within a half.
IDENTIFICATION: λ_i is removed by conditioning on the player, so φ̂ is immune to
  regression to the mean. The held-out contrasts use a 2×2 of arms (§2).
  check_identification does not apply; beats_the_market is the wrong gate (§6).
BASELINE: simulated true-Poisson null for φ̂ with the design held fixed. Arm D1 (as
  deployed) for held-out scores.
DECISION RULE: §3. Validated, off-by-default and null are each fixed numerically.
POWER: φ is well powered (80% at φ≈0.013, r≈75). Count log score is powered at
  φ≳0.02. Brier is NOT powered for r>~10, so it gates only as non-inferiority.
IF IT FAILS: delete the candidate frailty helper and flag. Nothing in project()
  exists yet (§8). Record in docs/DEFCON_THRESHOLD_CALIBRATION_<date>.md + INDEX nulls.
COST: no new data. Runtime of minutes. The board A/B needs a refactor arm first (RNG
  substream) and must not share an arm with the §6.11 GK saves A/B.
RANK: none of PROJECT_KNOWLEDGE §6. It closes a [CHECK] from the DefCon correction and
  sits well below §6.11 (at most about 0.05–0.08 pts/match against GK saves at about 1.0 pt/gw).
```

---

## 0. Wrong or stale premises

1. **Where the model's existing NB-ness comes from is right in form, but the effect is almost nothing for regular starters.** `[DERIVED]` from `multiseason_priors.to_priors`: α = prior·3 + 0.70·count and β = 3 + 0.70·n90. For a defender with about 30 90s at 7.68 per 90, α ≈ 23 + 161 ≈ 184. Integrated over the Gamma, the per-match count is NB with size α, so its dispersion is D = 1 + μ/α ≈ **1.04**. The NB-ness only matters for thin players (α of 25–40, D ≈ 1.2–1.3). So "already NB across paths" is true, but it does very little for the players who carry `defcon_ev`.
2. **Holding the rate fixed across fixtures does not affect `defcon_ev`.** `defcon_ev` is a mean over paths of per-fixture sums, so only the per-match marginal matters. The shared draw only changes the window's `sd`, `p5` and `p95`. Rate uncertainty is permanent: it is correlated across a path's fixtures. Frailty is transient. They are different variance components and must not be traded against each other.
3. **`defcon_env` scales α but not β.** It moves the mean and also changes the marginal NB size by the factor f (clipped 0.6–1.6). The within-season study has no environment change, so arm D1 leaves it out. The board A/B includes it.
4. **The quintile table does not describe the deployed composition.** It is a Poisson plug-in at a raw leave-one-out rate: no prior, no revert, no Gamma integration, and actual minutes rather than `exp_minutes`. Its residual var/mean of 1.45 includes the estimation variance of that raw rate, so it is an upper bound polluted by (a), not an estimate of dispersion.
5. **Cause (c) mixes two different things.** The CB/FB role difference is between players. It sits in each player's own rate and prior (`defcon_roles.prior_rate`), and a within-player estimator removes it. Only opponent supply, minutes, game state and mid-season role changes are within-player variation that a per-match NB would absorb.
6. **Banding on out-of-sample shrunk rates still leaks.**
   - The deployed shrinkage is not a proper within-season posterior. k0 = 3 is not fitted by empirical Bayes, revert = 0.70 is a between-season tempering, and `RATE_CB`, `RATE_FB` and `RATE_DEF_POOLED` were measured on the full season, so they leak the test half.
   - Even a proper posterior mean, used as a Poisson plug-in, leaks a Jensen term from posterior rate uncertainty. It pushes the tails in the same directions as the quintile pattern: up where P is convex (low rates), down where it is concave (high rates).
   - Fix: refit the prior on GW1-19 only, fit k by empirical Bayes, set revert = 1, integrate the Gamma, and put every arm into the same outcome-free bands.
7. **Your three-arm comparison mixes up two changes.** "NB at shrunk rate" against "as deployed" differs in both frailty and posterior integration. It needs a 2×2 (plug-in or integrated, crossed with Poisson or frailty); see §2.
8. **"Cluster = club-match for every SE" is not enough.** In synthetic runs the club-match bootstrap understated the true replicate SD of ΔBrier by 15–27% at φ ≥ 0.05. A player's test rows share his prediction error, and club-match clusters miss that.
9. **`captaincy.py:107` is a second copy of the DefCon composition** (`rng.poisson(dc*m90) >= thr`). Any change must reach both, or the captaincy tail and the board will disagree.
10. **`prng.poisson` uses a variable number of uniforms depending on λ** `[CHECK]`. So frailty in the main stream would shift every later draw for that player, and the "unchanged components are bit-identical" A/B method breaks (§3.4).
11. **Frailty raises DEF `defcon_ev` for most defenders.** At the pooled 7.68 per 90 (μ ≈ 7.47 per match) the threshold is above the mean. Synthetic arithmetic at φ = 0.05: P(≥10) goes from 0.220 to 0.245. The crossover is near μ ≈ 9 per match. So the fix would partly reverse the −21% for typical defenders and deepen it only for high-rate centre-backs.

## 1. Hypotheses, and the estimator that separates them

| Cause | Signature | Separating estimator |
|---|---|---|
| (a) Regression to the mean from banding on noisy rates | band-level miscalibration of the **mean** count | **G0**: slope of test-half count per 90 on the train-half posterior-mean rate, by band. Frailty keeps the mean unchanged, so a mean miscalibration cannot be (b). The within-player φ̂ is immune to (a) by construction. |
| (a′) Jensen from posterior rate uncertainty | P0 differs from E1 | P0 → E1 contrast |
| (b) Per-match overdispersion at a known rate | φ̂ > 0 while the mean is calibrated | φ̂ against the simulated Poisson null; E1 → E2 |
| (c) Within-player structure (opponent, game state) | φ̂ drops once structure is conditioned on | Diagnostic φ̂_opp: offset by a GW1-19 block-EB opponent rating (`block=["club","match_id"]`) |
| (d) Minutes varying around `exp_minutes` | φ̂ with m = the player's mean minutes minus φ̂ with actual minutes | diagnostic |
| (e) Rate drift between halves | a G0 slope different from 1 that disappears on an odd/even-GW split | odd/even replication |

**Leave (c) in the shipped φ.** `project()` has no per-match opponent term for DefCon (the opponent rating is display-only), so opponent variation is real per-match variance from the board's point of view. Coupling rule, fixed now: if an opponent DefCon term is ever promoted into points, φ must be re-estimated conditional on it, or the variance is counted twice.

**Club-match shared shocks are also left in.** In the synthetic run a shock with log-SD 0.15 and φ = 0 produced φ̂ ≈ 0.022. That is right for each player's marginal distribution. It is wrong for the joint distribution of teammates, because `project()` draws players independently. Report the within-club-match correlation of Pearson residuals as a diagnostic only. Squad-sum tails are out of scope.

## 2. Instrument

- **Population.** 25/26 rows published as DEF; CBIT counts from `defcon_series.fpl_defcon` (check `assert_no_recoveries`); `mins >= 60`; `defcon_fpl` not null.
  - Exposure = `defcon_series.exposure` (unmeasured minutes are dropped, never filled with 0).
  - Split on the explicit `gameweek` column, never by row position.
  - Key on `player_code`.
  - `[CHECK]`: confirm whether `build_pms` counts re-listed players under the 25/26 or 26/27 rule. This study uses the 25/26 rule.
- **Parameterisation: gamma-process frailty.** Y | λ, G ~ Poisson(λG), with G ~ Gamma(shape = m90/φ, scale = φ), so E[G] = m90 and D = 1 + λφ whatever m is.
  - At the fixed m90 of `project()` this is the same as NB2 with size m90/φ.
  - Chosen because the moment estimator below is exactly unbiased under it, and sub-appearance frailty shrinks consistently.
  - NB1 (constant D across rates) is reported only, as within-player D by rate tercile. It cannot ship from this study.
- **Estimator (primary).** φ̂ = Σ_i [X²_i − (n_i − 1)] / Σ_i (n_i − 1) λ̂_i, where X²_i is player i's Pearson χ² about λ̂_i = Σy/Σm, over players with n_i ≥ 2. Pooled across DEF, one parameter, fitted on **GW1-19**. The Dirichlet-multinomial conditional maximum likelihood is a reported cross-check.
- **Replication.** The same φ̂ on GW20-38 against its own null band.
- **Not shippable here.** φ split by CB/FB: diagnostic only. Any per-player or per-club φ: forbidden (§5).
- **Arms.** All are scored on the GW20-38 rows. Bands are the quintile edges of D1's predicted P on the test rows, which uses no outcomes.

| Arm | Rate | Per-match |
|---|---|---|
| P0 | EB posterior mean, plug-in | Poisson |
| N0 | EB posterior mean, plug-in | frailty φ̂ |
| E1 | EB Gamma, integrated (k fitted on GW1-19 by method of moments with frailty-inclusive noise, revert = 1, role prior from GW1-19) | Poisson |
| E2 | as E1 | frailty φ̂ |
| D1 | as deployed: k0 = 3, revert = 0.70, role prior refit on GW1-19, `exp_minutes` via `_shrunk_minutes` on GW1-19 | Poisson |
| D2 | as D1 | frailty φ̂ |

Predictives are computed by 48-node Gamma quadrature over G and the closed-form NB over λ.

- **Science contrast:** E1 → E2.
- **Ship contrast:** D1 → D2.
- **Reported only:** P0 → E1, P0 → N0, E1 → D1.

- **Standard errors.** Use SE = max(club-match cluster SE, player cluster SE) × c.
  - c = the ratio of replicate SD to that SE, from the selftest simulation at the fitted φ̂, floored at 1.
  - Pigeonhole two-way SE is reported. It was 35–60% conservative in the synthetic runs.
- **Selftest (synthetic, must pass before the real run).**
  1. Planted φ ∈ {0, .02, .05, .10} × shared shock ∈ {0, .15}: φ̂ recovered to within 2 SD.
  2. True Poisson: φ̂ false-rejection ≤ 7%, and the full rule ships in ≤ 5% of 200 replicates.
  3. Frailty leaves E[count] unchanged, to MC error.
  4. k0 mis-set by ×3 with φ = 0: the plug-in band table reproduces the quintile pattern, and φ̂ stays inside its null band.
  5. A planted opponent effect (log-SD 0.15, φ = 0): unconditional φ̂ leaves the band, φ̂_opp comes back into it.
  6. Planted drift: the half split and the odd/even split disagree on G0.
  7. Planted underdispersion (binomial thinning): φ̂ falls below the band.

  A synthetic prototype of 1–2 already ran (`scratchpad/power_sim.py`, `power_sim2.py`, temporary). Planted φ was recovered as 0.0193–0.0197, 0.0495–0.0499 and 0.0983–0.0986. Under true Poisson, φ̂ averaged −0.0003 and false-rejected 4%.

## 3. Decision rule (fixed now)

Definitions:
- **S1:** φ̂(GW1-19) is above the 97.5th percentile of its simulated true-Poisson band. The band is simulated from the GW1-19 per-player rates with the same rows and exposures.
- **S1-rep:** the same test on GW20-38.
- **M:** mean over test rows of |P_D2 − P_D1|. It uses no outcomes.
- **ELPD:** log predictive density of the count for E2 minus E1, per row.

**Before anything else:**
- If φ̂ is **below** the 2.5th percentile, record "DEF CBIT under-dispersed at a known rate" as a finding. Gamma frailty cannot express underdispersion, so nothing ships; a new study is needed. The E2 precedent (team goals thinner than Poisson) is why this test is two-sided.
- **G0** (reported, not gating the frailty decision): if D1's mean-calibration slope CI excludes 1 by more than 0.10, open a separate registered k0 item. It is shrinkage calibration, not a directional pull, and it gets its own A/B.

| Outcome | Condition (all must hold) |
|---|---|
| **VALIDATED CORRECTION** (on by default after the board A/B) | S1 **and** S1-rep; M ≥ 0.005; ELPD(E2−E1) z ≥ 1.645; ΔBrier(E2−E1) ≤ 0; D2−D1 non-inferior: one-sided 95% upper bound of ΔBrier < **+0.001**; odd/even split gives ELPD of the same sign |
| **OFF BY DEFAULT** (flag, awaiting 26/27 replication) | S1; M ≥ 0.005; D2−D1 non-inferior (< +0.001); ELPD z > −1; the validated conditions not all met. **This includes r clearly finite but Brier not improving:** a finite φ with a positive ΔBrier point inside the margin ships off. |
| **NULL: delete path** | φ̂ inside the band; **or** M < 0.005 ("finite but immaterial"); **or** the non-inferiority bound ≥ +0.001 ("overdispersion present, not a points correction"; φ̂ recorded, path deleted); **or** ELPD z ≤ −1 (frailty shape is wrong) |

- **Tie-break.** A statistic exactly on a threshold, or Brier and ELPD disagreeing across a row boundary, takes the more conservative row (status quo).
- **Report-only.** Band reliability and logistic calibration slope. Their power is marginal: band SE is about 0.03 against shifts of 0.02–0.04.
- **Margin.** +0.001 Brier is about the same loss as a uniform 0.032 bias in P(hit), roughly 0.6% of the base Brier.

**3.4 Board A/B pass** (only after VALIDATED or OFF):
1. **Refactor arm first.** Move the DefCon Poisson (and the frailty draw) to a dedicated child RNG per player, φ off, in both `bayes_model.project()` and `captaincy.py`. GK/MID/FWD and all non-DefCon DEF components change by MC noise only. This arm then becomes the reference.
2. **Frailty arm, same seed.**
   - GK/MID/FWD and every non-DefCon DEF component bit-identical in `mean`.
   - Compare `mean`, not `blended`, with `LIVE_FPL=off`.
   - Every DEF likely starter whose analytic |ΔP| exceeds 2 MC SE moves in the analytic direction: ≥ 95% of them.
   - Mean Δ`defcon_ev` for DEF likely starters within ±2 MC SE of the quadrature expectation.
3. `test_all --quick` green, except the GW-save invariant that is known to be red (§6.11).
4. Spearman and top-20 overlap are reported, not gating.
5. **Registered 26/27 replication (forecast-scorer).** φ fitted on all of 25/26, scored on 26/27 single-fixture DEF player-gameweeks with 60+ minutes, from `player_gameweek_stats` components (DGWs excluded), once there are ≥ 1,500 rows. A VALIDATED result falls back to OFF if the D2−D1 non-inferiority bound fails there.

## 4. Power (synthetic design, matched to your counts)

The synthetic design: 143 defenders, 20 clubs, train and test matches per player drawn to your quantiles, about 3.8 rows per club-match, 55% CBs at 8.93 and 45% FBs at 6.12 per 90, within-role log-SD 0.20 `[JUDGMENT]`, 82% full matches. 120 replicates, plus 60 for the SE comparison.

| Planted | φ̂ (mean ± rep SD) | P(S1) | ΔBrier E2−E1 (rep SD) | ELPD per row (rep SD) |
|---|---|---|---|---|
| Poisson | −0.0003 ± 0.0047 | 0.04 | +0.00001 (0.00006) | +0.0002 (0.0006) |
| φ = .02 (r = 50) | 0.0193 ± 0.0063 | 1.00 | −0.00005…−0.0001 (0.00025) | +0.0065 (0.0031) |
| φ = .05 (r = 20) | 0.0495 ± 0.0075 | 1.00 | −0.0005 (0.0006) | +0.031 (0.008) |
| φ = .10 (r = 10) | 0.0986 ± 0.0104 | 0.99 | −0.0023 (0.0011) | +0.101 (0.013) |
| Shock 0.15, φ = 0 | 0.022 ± 0.007 | 1.00 | −0.0001 (0.0003) | +0.008 (0.003) |

- **φ.** The null band's upper edge is about +0.008. 80% power at φ ≈ 0.013 (r ≈ 75). **Can produce an informative null.** An upper CI below 0.015 bounds the P(hit) shift at the pooled rate to about 0.008, roughly 0.016 pts/match.
- **ELPD.** z ≈ 2.1 at φ = .02 and ≈ 3.8 at φ = .05. Powered.
- **Brier.** The minimum detectable effect is about 2.49 × SE ≈ 0.0013–0.0028, which is only reached at φ ≈ 0.11 (r ≈ 9). **It cannot produce an informative null for r > 10.** Detecting r = 20 at 80% would need about 9,800 player-matches, around 3 seasons. That is why Brier is a non-inferiority gate here and not a superiority gate.
- **No spurious win for frailty in a true-Poisson world:** ΔBrier +0.00001, ELPD +0.0002, about 0.3 SD.
- **Size of the effect `[DERIVED, synthetic arithmetic]`.** At φ = .05, P(≥10) moves +0.016 at 5 per 90, +0.025 at 7.68 per 90 and −0.040 at 11 per 90. That is ≤ about 0.08 pts/match.
- **Not simulated `[CHECK]`.** G0 slope power and φ̂ bootstrap coverage. The selftest must report both before the real run.

## 5. Null register

- **Team explosiveness E1.** E1 asked whether dispersion **differs by unit**: split-half reliability of per-club dispersion in team goals. This study asks for a **single pooled** φ for a different count, at a scoring threshold. The closest analogue is E2, the league tail test, which came back verified *thinner* than Poisson, not a null. That precedent is why the test is two-sided.
- **The revival boundary is explicit.** A per-player, per-club or per-role φ shipped as a term would be E1 moved down to player level. That is forbidden here. The CB/FB φ split is diagnostic only.
- **`early_dispersion`.** It measured the **first-moment** slope of xG on the strength difference in GW1-6. It was measured and not applied, and it is not a null. The shared word "dispersion" is the only overlap.
- **Mean-reversion (p = 0.69).** Frailty leaves the mean unchanged (E[G] = m90), so it makes no directional claim. A G0 failure would lead to a separately registered k0 item, never a directional pull.
- **Congestion and rotation.** No fixture-conditional term. Not in play.

## 6. Guards in play

- **`beats_the_market` is the wrong gate, twice over.** It is a Poisson score test on a team-level mean, and this is a player-level second-moment claim that never enters `TeamModel`. The gate that applies is the explosiveness pattern: a simulated true-Poisson null plus held-out calibration of counts and threshold hits (S1 and §3).
- **`check_identification`:** not in play (no style regression).
- **`perm_r2`:** any permutation diagnostic (φ̂_opp) uses `block=["club","match_id"]`.
- **No `fillna(0)`:** a null `defcon_fpl` row is dropped from both the count and the exposure (4,450 → 4,385 rows). Exposure = `defcon_series.exposure`.
- **Pre-25/26 refusal:** `fpl_defcon` refuses 24/25, so there is no 24/25 fitting half. More power must **not** be bought by summing 24/25 components; that is its own untested channel.
- **Published DefCon column:** never read for DEF. Join on `player_code`. Gameweeks split by the explicit column.
- **Regime change is a variance statement:** consistent here, since frailty is variance only.
- **New components ship off by default:** a study CSV plus a `src/manifest.py` node, `--selftest` offline, and one flag (name it when the code is written).
- **MID/FWD are out of scope, but they probably matter more in relative terms.** CBIRT includes high-volume recoveries, and the threshold of 12 sits well above the MID mean of 8.4, so frailty would raise P(hit) proportionally more. The DEF φ must **not** be transferred. MID needs its own registration under this protocol.

## 7. Cost and sequencing

- **Data:** none new (the `build_pms` panel).
- **Runtime:** estimator and quadrature take seconds; bootstraps and 200-replicate selftest simulations take about 10 minutes.
- **Sequencing:** the §6.11 GK saves work also edits `project()`, so run it as a separate A/B arm. The refactor arm (§3.4.1) should land alone first.

## 8. If it fails

- **Order of work:** write the study, then run it. Write the model path only if the outcome is not NULL.
- **On NULL:** delete the candidate frailty helper (study-local). Neither `project()` nor `captaincy.py` gains a flag.
  - The refactor arm's RNG substream may stay only if it was already merged and passed its own A/B.
- **Record the null in:**
  - a new `docs/DEFCON_THRESHOLD_CALIBRATION_<date>.md`, with φ̂ and its CI and the reason: band, immaterial, non-inferiority or shape;
  - a row in `docs/INDEX.md` "The nulls, in one place";
  - closing the `[CHECK]` in `docs/DEFCON_SOURCE_CORRECTION_2026-09-10.md` §6.
- **Keep** `studies/defcon_threshold_calibration.py` and its `.csv` as evidence.

Files: `C:\Users\mjone\FPL-Project\.claude\worktrees\zen-robinson-59f08d\src\bayes_model.py` (lines 465-507), `...\src\captaincy.py` (line 107), `...\src\multiseason_priors.py` (lines 360-361), `...\src\defcon_env.py` (line 91), `...\src\defcon_series.py`. The synthetic power scripts are only in the session scratchpad (`power_sim.py`, `power_sim2.py`), not the repo.
