# soccerdata Integration — Findings & Corrections

**Opened:** 2026-08-09 · **Status:** foundation landed + §2.1 measured; §2.2 is a recorded null
**Source spec:** `SOCCERDATA_INTEGRATION_SPEC.md` (external, 2026-08-09) + `studies/sd_design.py`
**Data:** Understat 24/25 + 25/26, scraped 2026-08-09 via soccerdata 1.9.1 — 19,402 shots,
1,520 team-matches, 23,057 player-matches, cached under `config.SD_CACHE`

Per project convention this file records **nulls and corrections**, not just wins.

**Bottom line:** §2.1 (reliability) is measured and well-powered — that is the deliverable.
§2.2 (persistence) does **not** identify on this panel and is recorded below as a null; the
two-component prior in §2.3 must **not** be built on it. Nothing in the live board was touched.

---

## 1. What landed

| Module | Purpose | Selftest |
|---|---|---|
| `src/sd_ingest.py` | cached Understat/WhoScored readers, resumable, `scraped_at` per row | ✅ offline |
| `src/xg_calibrate.py` | Understat xG → the model's scale, with the join gate | ✅ offline |
| `src/crosswalk.py` | Understat id → `player_code` (G5), collisions never auto-resolved | ✅ offline |
| `src/setpiece.py` | §2.1 reliability, §2.2 persistence, the re-derived decision rule | ✅ offline |
| `studies/setpiece_study.py` | the §2 run, writes both deliverable CSVs | ✅ runs on real data |
| `studies/sd_design.py` | the pre-registration arithmetic, reproduced | ✅ runs |

Deliverables written: `studies/setpiece_reliability.csv`, `studies/component_ar1.csv`.

---

## 2. `[VERIFIED]` §2.1 — component reliability, measured

200 random match-splits per player-season, Spearman-Brown to full length, ~500 players per
season, goalkeepers excluded (role exclusion — see §6.2).

| component | 24/25 | 25/26 | spec §6.A predicted |
|---|---|---|---|
| `open_play` | 0.896 | 0.880 | 0.933 |
| `penalty` | 0.659 | 0.727 | — |
| `from_corner` | 0.598 | 0.557 | — |
| `set_piece` (direct FK) | 0.496 | 0.462 | **0.491** |
| `set_piece_all` (all dead-ball ex-pen) | 0.652 | 0.605 | — |

**The spec's 0.491 was right — for the wrong component.** It matches the narrow direct-free-kick
margin (measured 0.462–0.496) almost exactly. But the quantity §2 actually needs to shrink is
all non-penalty dead-ball xG, and *that* is measured at **≈0.63**, not 0.49. Corners are the bulk
of set-piece xG and are better measured than direct free kicks.

Consequence: the attenuation the whole of §2 is premised on correcting is **substantially
smaller than the design assumed**. At reliability 0.63 rather than 0.49, there is materially
less room between measured and true persistence, and correspondingly less headroom for the
two-component prior to pay for itself.

This measurement is well-powered and stable (SE across splits ≤ 0.045) and stands on its own,
independent of the §2.2 failure below.

---

## 3. `[NULL]` §2.2 — persistence does not identify on this panel

**Pre-committed rule outcome: no verdict for the decision component. §2.3 is not authorised.**

| component | n pairs | ρ measured | reliability | ρ ratio-corrected | ρ simulation | saturated |
|---|---|---|---|---|---|---|
| `open_play` | 187 | 0.776 | 0.888 | 0.873 | 0.905 (0.821–0.970) | no |
| `set_piece` | 187 | 0.389 | 0.479 | 0.813 | 0.809 (0.547–0.970) | no |
| `from_corner` | 187 | 0.710 | 0.578 | **1.229** | 0.970 | **yes** |
| `set_piece_all` | 187 | 0.679 | 0.629 | **1.080** | 0.970 | **yes** |
| `penalty` | 187 | 0.525 | 0.693 | 0.758 | 0.810 | **yes** |

Three independent symptoms, all pointing the same way:

1. **The ratio correction returns ρ > 1** for `from_corner` (1.229) and `set_piece_all` (1.080).
   Persistence above 1 is impossible for a stationary process. This is the upward bias predicted
   in §5 below, showing up on real data rather than in simulation.
2. **The simulation inversion saturates** at the grid ceiling for the same components: the
   measured slope is larger than the simulated count process produces at *any* ρ_true.
3. **The penalty CI collapses to zero width** — a pinned estimate wearing an interval.

**Diagnosis.** The simulation redraws every player's shot count at random each season. The real
panel has *persistent structural zeros*: a centre-back takes no direct free kicks in either
season, and that pair contributes perfect persistence. Real ρ is therefore inflated by role
composition, which the simulation cannot reproduce, so the inversion runs off the top of its
grid. The AR(1) as specified is measuring **role stability**, not **set-piece-taking-skill
stability**, and those are not the same quantity.

`setpiece.decide()` now returns `inconclusive_saturated` rather than laundering a pinned
estimate into a verdict.

**Only clean result:** `open_play` is not saturated, and its CI (0.821–0.970) **covers** the
pooled prior's implied w0 = 0.835. Per the pre-committed rule that is `keep_pooled_prior`.

**What would fix it** (not attempted here — it changes the estimand and needs re-pre-registering):
condition the AR(1) on position, or restrict to players with non-zero set-piece xG in season *t*
and accept the resulting selection explicitly, or model the zero mass rather than flooring it.
Each is a different question from the one §2 asked.

---

## 4. `[CHECK]` §1.2 — the calibration gate fails on a verified-correct join

Understat 25/26 team-match xG against the data repo's raw Opta `expected_goals_xg`, joined on
the fixture pair, **100% matched** (760/760):

```
per match:      xg_opta = +0.142 + 0.818 * xg_understat    r2 = 0.843   n = 760
season totals:  xg_opta = +2.496 + 0.868 * xg_understat    r2 = 0.965   n = 20
```

- `0.8 < b < 1.2` → **PASS** (0.818, near the edge)
- `r2 > 0.85` → **FAIL** (0.843)

The spec says a failure here means "the join is wrong, not that the models differ". **The join
is verifiably right** — every team-match matched, every club name mapped. So the threshold
itself is the thing that does not survive contact with the data: per-match Understat-vs-Opta
agreement is r = 0.918, i.e. r² = 0.843, and that is simply what two different xG models on the
same matches look like. Under G7 the rule is **not** being softened — it is recorded here as a
failed pre-committed gate, with the cause diagnosed.

**The level correction is large and worth having regardless.** Understat runs ~9.8% hot:
season xG totals 1162.4 vs Opta 1059.0 (ratio 0.911). Feeding raw Understat xG into the player
layer would inflate every derived projection by that margin. This is the concrete payoff of
§1.2 and the reason the spec was right to call it mandatory.

**§2 is unaffected either way.** Split-half reliability is a correlation and an AR(1) on log
rates absorbs a scale factor into the intercept, so both §2 quantities are invariant to the
affine map. The gate binds on anything feeding *levels* — §2.3's prior, §2.4's penalty xG.

### 4.1 Correction to an earlier correction

An earlier draft of this file claimed the team layer is not fitted on Opta xG at all. That was
half right and is superseded. `E0_recon.csv` genuinely holds no xG column — but
`reconstruct_e0.py` builds its odds *from* the data repo's Opta `expected_goals_xg`, and
`betting_features.build()` de-vigs them back to λ. So the λ the team layer consumes **is** Opta
xG, round-tripped.

The round trip is **lossy**, which the module docstring does not say:

```
corr(Understat, Opta)       0.9182
corr(Opta,      E0 lambda)  0.9331     <- would be 1.000 if the round-trip were faithful
corr(Understat, E0 lambda)  0.8521
```

`reconstruct_e0.py`'s docstring asserts "the round-trip is faithful". It loses ~7% of
correlation with its own input, because the fair-odds → de-vig → solve-λ inversion is numerical
and approximate. Anything calibrating against the model's scale should use the raw Opta column
from the data repo, not E0_recon's λ.

---

## 5. `[VERIFIED]` Ratio-disattenuation — a claim made, then partly retracted

The spec's correction is `ρ_disattenuated = ρ̂ / reliability`, from `plim ρ̂ = ρ_true × reliability`.

**What an earlier draft of this file claimed, and why it was wrong.** A 12-seed simulation
initially put the ratio's bias at +0.037 with 92% CI coverage against +0.003 and 100% for a
simulation-based inversion, and concluded the identity itself was at fault. Re-running the
identical study after `_floor_log` was changed (from half the *minimum* positive rate to half
the 5th percentile) gives:

| estimator | mean | bias | 95% CI coverage |
|---|---|---|---|
| ratio, `ρ̂ / reliability` | 0.567 | +0.017 | 100% |
| simulation inversion | 0.566 | +0.016 | 100% |

**Indistinguishable.** Most of the gap I attributed to the estimator was caused by my own floor
choice: half the minimum positive rate lets a single outlier — one heavy-minutes player with one
grazing low-xG shot — sit ~8 log units below the typical rate, inflating var(log) to 2.06 where
the real spread is ~0.6, which is exactly what breaks a variance-ratio identity. The quantile
floor repaired it for both estimators. **The +0.037/+0.003 figures are superseded; do not cite
them.**

**What survives.** On real data (§3) the ratio returns ρ = 1.229 and ρ = 1.080 — impossible for
a stationary process — for the two lowest-reliability components. The simulated regime does not
reproduce that failure because it lacks the real panel's persistent structural zeros. So the
case against the ratio rests on real-data evidence, not on simulation.

**Why the inversion is kept anyway.** Not because it is less biased — in a clean regime it is
not. Because it **fails visibly**: when the observed slope exceeds anything the count process
can produce, it pins at the grid edge and raises `saturated`, and `decide()` then refuses to
rule. The ratio has no such signal; it returns 1.229 and lets a prior be built on it. That
diagnostic is what turned §3 into an honest null instead of a false green light.

`studies/sd_design.py`'s reliability estimates are sound; its `rho_disattenuated` column should
not be used as an estimate.

---

## 6. Premise corrections to the spec

### 6.1 There is no "80% set-piece haircut" and no ρ = 0.204 `[VERIFIED by inspection]`

§2 motivates itself as re-sizing "a global ~80% set-piece mean-reversion haircut derived from a
measured AR(1) of ρ = 0.204". Neither exists. The npxGI prior is a single pooled Gamma-Poisson
(`multiseason_priors.to_priors`) with no component split at all; ρ = 0.204 appears nowhere; and
the tested-null mean-reversion result (p = 0.69, `studies/edge_study.py` H3) is a *directional*
over/under-performance test, a different quantity.

The real status quo is one shrinkage weight applied uniformly to every component:

```
posterior = w * own_rate + (1 - w) * position_prior
w(n90)    = revert * n90 / (k0 + revert * n90)        revert = 0.70, k0 = 3.0
```

At the estimation sample's median exposure (21.6 nineties) that is **w0 = 0.835**.
`setpiece.implied_pooled_persistence()` reads the constants from `multiseason_priors` at
runtime, so the rule tracks the code rather than a copied number.

### 6.2 Two population choices, both recorded

- **Goalkeepers excluded.** A role exclusion, not an outcome one: 48 keepers sat in the
  ≥900-minute panel with structurally zero open-play xG in both seasons, contributing ~7% of
  pairs with perfect persistence. Outfield players with genuine zeros are kept — dropping those
  would be selection on the outcome.
- **Keyed on the Understat player id, not `player_code`.** G5 governs anything feeding the FPL
  model; this study feeds a decision. Routing through the crosswalk would inject match error
  into a persistence estimate for no benefit. §2.3, if ever built, must join on `player_code`.

### 6.3 §3.1's action item is already done `[VERIFIED by inspection]`

`NEW_MANAGER_CLUBS_2627` does not exist. `press_index.REGIME_PRESS_CLUBS` already carries ten
clubs including Newcastle, with Jaissle at PPDA 10.0, and `PROJECT_KNOWLEDGE_2627.md` §8 already
records nine full plus two partial regime changes.

### 6.4 §2.4's penalty measurement already exists `[VERIFIED by inspection]`

`pen_xg90_measured` is computed in `multiseason_priors.to_priors` and
`pms_priors.to_model_priors`. The runners discard it and apply the declared-order heuristic
instead (`gw_board.py`, `decision_v2.py`: `pen_order == 1 → clip(pen_xg90, lower=0.10)`). §2.4
is plumbing plus an A/B, not new measurement. Untouched — it changes the live board.

Supporting data now available: 175 penalties across the two seasons, Understat xG a constant
0.7612, realised conversion 0.834.

> **Tested 2026-08-11 — the declared order is the RIGHT choice, not a gap.** The wording
> above implies the runners are wrong to discard `pen_xg90_measured`. They are not.
> `studies/penalty_assignment.py` raced the two signals on 25/26: declared
> `penalties_order == 1` gives 85.7% precision covering 39.1% of penalties actually taken,
> against 45.0% and 34.8% for "took ≥2 penalties last season". The declared order wins on
> both. The real gap is **coverage** — only ~39% of penalty EV is captured, because roughly
> six clubs have no declared first-choice taker — and that is a data problem, not a
> modelling one. See docs/PLAYER_LAYER_FINDINGS.md §3.

---

## 7. `[CHECK]` §3 — measured PPDA vs the hardcoded estimates

Measured 25/26 season-mean PPDA against `press_index.PPDA_2526`, all 20 clubs:

```
correlation 0.800   rank correlation 0.770   mean |error| 0.94
mean signed error (estimate - measured) -0.52   i.e. the estimates systematically
                                                 overstate how hard clubs pressed
repo LEAGUE_PPDA constant 12.0  vs  measured league mean 12.40
```

Worst misses: Crystal Palace (est 12.0, measured 14.98), Nott'm Forest (12.5 → 14.89), West Ham
(13.0 → 14.94), Brighton (10.8 → **9.07** — actually the league's most intense press, ranked
6th by the estimates), Brentford (13.0 → 11.64).

The estimates get the broad ordering right (ρ = 0.77) but misplace individual clubs by up to 3
PPDA. **Not substituted** — §3 was out of scope for this pass and the substitution touches
`defcon_env`'s CBIRT channel and therefore the live board. G4 also requires routing PPDA to the
DefCon channel only. Recorded here as the evidence base for doing it.

> **DONE [2026-08-28]** — `src/press_measured.py`, evidence `studies/press_switchover.py`.
> Not a substitution in the end: a straight swap is wrong because a season-aggregate PPDA is
> not available in-season, and an early-season one is mostly noise (single-match reliability
> 0.168 over 12 Understat seasons). The judgment table is instead *revised* toward measured
> press at weight `n/(n+40)` — 2% after one match, 20% by GW10. Two things this section could
> not have known:
> - **Understat 26/27 is not obtainable offline**, so the in-season feed is rebuilt from the
>   FPL repo and calibrated onto the Understat scale (r = 0.937, 20 clubs, 25/26).
> - **That proxy is worth much less than Understat itself.** With Understat as the feed the
>   optimal weight is `k = 11.5` (−27.5% MSE, 220 team-seasons); with the proxy the optimum is
>   `k ≈ 45` and `k = 12` loses. `k = 40` is the value non-negative under both. Refreshing this
>   cache in-season is what would unlock the rest of the gain.

---

## 8. Data-quality findings in the existing repo

Both found while building the joins; neither affects current model output.

- **`E0_recon.csv` dates are a sentinel, not data.** `reconstruct_e0.py` stamps any match with a
  missing kickoff time as the literal `"01/01/2026"`. Result: 20 teams collapse onto 2 dates,
  with Arsenal appearing in 5 fixtures on 2026-01-01. Harmless to the model (it only sorts by
  date) but fatal to any date-keyed join. `xg_calibrate` joins on the fixture pair instead.
- **`reconstruct_e0.py`'s round trip is not faithful** — see §4.1. Its docstring says it is.

---

## 9. Environment

- `soccerdata` 1.9.1 installed 2026-08-09. Deliberately **not** added to `requirements.txt`: it
  is a scraping dependency needed only to refill the cache, not to run the model. It also pulls
  Selenium/SeleniumBase for the WhoScored path.
- **soccerdata 1.9.1 does not label penalties.** Understat's `Penalty` situation comes back
  null. `sd_ingest._recover_penalties` re-labels them under a guard — all 175 sit at exactly the
  penalty spot (0.885, 0.5) with constant xG 0.7612 — and raises if an unlabelled shot does not
  carry that signature rather than guessing.
- Other live-schema deviations from the spec, handled in `sd_ingest`: `read_team_match_stats`
  is wide (home_*/away_*) and is melted to long; situations are space-separated
  (`"Open Play"`); `last_action` and `shots`/`shots_against` do not exist and were dropped from
  the contracts rather than carried as permanent NaN.
- Cache at `config.SD_CACHE` (`<repo>/.cache/soccerdata`, override `FPL_SD_CACHE`). `.gitignore`
  previously covered only `*.pkl`, so `.cache/` was added — otherwise the scraped shards would
  have been committed.
- Data repo cloned to `C:\Users\mjone\FPL-Core-Insights`; set `FPL_DATA` to its `data` dir.

Selftests, all offline:

```bash
python src/sd_ingest.py --selftest && python src/xg_calibrate.py --selftest && python src/crosswalk.py --selftest && python src/setpiece.py --selftest
```

---

## 10. Outstanding

1. **Do not build §2.3.** The persistence estimate that would justify it is a null (§3).
   Re-specify the AR(1) against role composition first, and re-pre-register the rule.
2. Decide what to do about the §1.2 gate (§4): the threshold failed on a correct join. Either
   re-pre-register a threshold calibrated to real Understat-vs-Opta agreement, or accept the
   affine map on the slope evidence alone (b = 0.818 is comfortably inside its band).
3. Build the crosswalk and hand-verify `data/crosswalk_review.csv` — needed before any of this
   reaches the FPL model, not before more analysis. Fuzzy rows are written `verified=False` and
   `load_crosswalk(strict=True)` hides them until a human signs off.
4. §3 PPDA substitution — evidence in §7; `press_index.fetch_live_ppda()` is a stub waiting for it.
5. §4 WhoScored counts, gated on the §4.1 reconstruction check against `defcon_raw`. Untouched.
6. §5 shot volume × conversion — the spec gates this on §2 landing cleanly. It did not.
