# Start persistence: what a start tells you about the next one

*2026-09-07; **corrected 2026-09-16** after a stats-referee review. Evidence:
`studies/start_persistence.py` / `.csv`, `studies/start_recency.py` / `.csv`. Code:
`src/inseason.py` (`recency_starts`, `appearances(lam=)`), `scripts/gw_board.py`
(`INSEASON_LAM`).*

---

## Corrections, 2026-09-16 — read these first

Every number below is the corrected one. What changed, and why:

1. **The 2022/23 `starts` column was a literal 0 for GW1-15** `[VERIFIED]`. FPL began
   publishing it at GW16; 8,491 rows, 2,818 of them players on 60+ minutes, read as
   non-starts. Zero, not null, so the loader's presence check passed it. Regular starters
   got a fake 14-gameweek zero block followed by an unbroken run, which pushed every
   persistence number **up**. `fpl_history.empty_native_gws` now finds such blocks from the
   data; this study treats 22/23 as observed from GW16. Effect: within-player lag-1
   +0.493 → **+0.436**; streak excess at k=1 +0.190 → **+0.164**, at k=8 +0.030 →
   **+0.015**; recency gain at h=10, κ=4 +1.18% → **+1.64%**. The shape survives, and the
   proxy seasons (untouched by the defect) reproduce unchanged.
2. **Several claims were tagged above what the estimators identify.** Retagged below:
   the lag-1 gap is *persistence*, not identified state dependence; "a `streak_k` term
   would be redundant" and "momentum is dead — precisely Result 2" are **withdrawn**
   (untested, not null); the benching asymmetry has no null and is descriptive only; the
   Enzo/O'Reilly "right shape" check was mechanical and is removed as evidence.
3. **"λ moves only the order" was wrong** `[DERIVED]`. See *What renormalisation does not do*.
4. **Production renormalised over the club's history instead of the player's own
   window.** Fixed. It moves 0 of 658 players on 26/27 data to date — see *How it ships*.

---

## The question, and why the obvious answer is wrong

"How much does starting in one gameweek raise the odds of starting in the next, and how
does that change after 3, 6, 10, 20 consecutive starts?"

Both halves are confounded, in the same direction, by the same thing.

- **Lag-1.** Nailed-on players supply almost all the 1→1 transitions and fringe players
  almost all the 0→0 ones. The pooled contrast therefore measures *who the players are*,
  not what last week did.
- **Duration.** A long streak is a *filter* that selects high-p players. Raw `h(k)` rises
  with `k` under a pure mover-stayer process with **zero** true duration dependence. So
  the null is not `h(k) = const`, and a study that tests against a flat line will
  "discover" a streak effect that is entirely frailty.

Two estimators, accordingly:

| | estimator | what it removes | what it does NOT remove |
|---|---|---|---|
| lag-1 | within-player-season contrast, player-seasons with ≥5 of each state | between-player frailty | within-season drift in a player's rate (role change, sacking, injury spell) |
| duration | permutation of the outcome sequence **within** player-season | order, holding each player-season's start *count* fixed | — but see the hindsight caveat under Result 2 |

Per the repo's permutation guard, the **outcome** is permuted, not labels.

**Data.** 105,080 player-matches, 1,589 players, 2022/23 (from GW16) – 2025/26, native
`starts` column. Unit is the player-**match** ordered by kickoff (a double gameweek is two
ordered matches). Streak resets each season. Replicated on 2016/17–2021/22 under the
`mins ≥ 60` proxy, reported separately because the proxy spuriously *breaks* streaks
whenever a starter is hooked before the hour.

---

## Result 1 — one start on the next start: persistence `[DERIVED]`

| scope | P(start \| started) | P(start \| benched) | pooled gap | within-player gap |
|---|---|---|---|---|
| native 22/23 GW16 – 25/26 | 0.798 | 0.082 | +0.717 | **+0.436** (n=1,255 player-seasons) |
| proxy 16/17–21/22 | 0.784 | 0.111 | +0.673 | **+0.405** (n=1,979) |

The within-player gap is large and replicates across definitions. **What it is not** is
identified state dependence, and the earlier line "two-thirds genuine state dependence,
one-third frailty" is withdrawn:

- It is estimated only on *rotation-zone* player-seasons (≥5 of each state, weighted by
  min(n₁, n₀)), so dividing it by the population-wide pooled gap compares two different
  populations.
- Two biases pull in opposite directions. Under exchangeability a finite sequence with a
  fixed count has E[gap] = −1/(n−1) ≈ −0.03 (**down**). Within-season drift in a
  player's rate adds persistence with no state dependence at all (**up**), and the net is
  judged up.

Read it as **persistence = state dependence + within-season drift + availability**
`[JUDGMENT]`. Useful for forecasting; not a causal quantity. Separating them needs the
within-sequence permutation mean subtracted and a lag-2-given-lag-1 test — not done.

## Result 2 — duration dependence fades by k ≈ 8 `[DERIVED]`

Native scope. `h_null` is the within-player-season permutation null; CIs are a cluster
bootstrap over `player_code`.

| k | n | h_obs | 95% CI | h_null | excess | z |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 6,600 | 0.656 | [0.643, 0.671] | 0.492 | **+0.164** | 40.6 |
| 2 | 4,240 | 0.761 | [0.746, 0.776] | 0.606 | **+0.155** | 26.3 |
| 3 | 3,147 | 0.800 | [0.785, 0.816] | 0.678 | **+0.122** | 15.8 |
| 4 | 2,456 | 0.813 | [0.797, 0.830] | 0.731 | +0.082 | 9.6 |
| 5 | 1,943 | 0.847 | [0.828, 0.863] | 0.770 | +0.076 | 7.1 |
| 6 | 1,609 | 0.861 | [0.845, 0.881] | 0.801 | +0.060 | 5.6 |
| 8 | 1,133 | 0.859 | [0.840, 0.882] | 0.844 | +0.015 | 1.2 |
| 10 | 809 | 0.874 | [0.849, 0.895] | 0.872 | +0.002 | 0.2 |
| 15 | 395 | 0.896 | [0.869, 0.924] | 0.912 | −0.016 | −0.8 |
| 20 | 195 | 0.959 | [0.931, 0.985] | 0.936 | +0.023 | 1.1 |
| 25 | 102 | 0.931 | [0.881, 0.979] | 0.956 | −0.025 | −0.9 |

Raw `h(k)` climbs to 0.96 by k = 20 — **and so does the null.** The excess is clearly
positive to k ≈ 6 and indistinguishable from zero by k ≈ 8–10. Proxy seasons: +0.144 at
k=1, +0.032 at k=8, +0.028 at k=10 (z=2.1), noise from k=15.

Read it narrowly, because of two caveats:

- **The null conditions on hindsight.** It holds each player-season's *full-season* start
  count fixed, including matches after t that a forecaster cannot see. So "excess ≈ 0
  past k≈8" means *a streak adds nothing beyond the player's own full-season rate*. It
  does **not** mean a streak adds nothing beyond the forecaster's prior (last season plus
  the season to date). Whether a `streak_k` term improves a real forecast is
  **untested** — not a null.
- **z is overstated.** It uses the permutation SD only and ignores the sampling variance
  of `h_obs` (stats-referee's estimate: z at k=8 roughly 1.9 → 1.5 on the pre-fix
  numbers; not re-derived here). Bootstrapping the excess with the null recomputed inside
  each resample would fix it; not done.

`h_obs` and `n` are exact. `h_null`, the CI, `excess` and `z` are resampled (N_PERM=200,
N_BOOT=300) and move by ~0.003 between runs.

**Power.** At k=20 the null SD is 0.024. That rules out an excess ≳ +0.05 there; it
does not rule out ±0.03.

## Result 3 — runs of non-starts: descriptive only `[CHECK]`

P(start next | *k* consecutive **non**-starts), native, raw:

| k | 1 | 2 | 3 | 5 | 8 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| h | 0.290 | 0.165 | 0.126 | 0.091 | 0.054 | 0.042 | 0.010 |

This curve has **no null**. It is the raw hazard, and so exactly the frailty-confounded
kind of curve this doc opens by warning against: long non-start runs select players who
rarely start. "Being dropped is far more informative than being picked" was tagged
`[VERIFIED]` and is withdrawn until the permutation null is run for the non-start state.

---

## What this licenses, and what it does not

What Results 1–2 indict is the **likelihood**. `inseason.update_minutes` is a conjugate
Beta update; a Beta update reads a *count* and is therefore **exchangeable**, so
start-start-bench and bench-start-start produce an identical posterior. Ordering carries
information over a window of several matches, and a count is blind to it.

The response built was a geometric recency weight on the estimator's own observations,
`u_d = λ^d`, renormalised so the weights sum to the match count. It is not a predictor
and not fixture-conditional (see *What stays dead*).

### What renormalisation does not do `[DERIVED]`

This doc first said renormalising means "λ moves **only** the order", keeping recency
separate from the `W_MINUTES`/κ ridge. **That was wrong.** Holding the weights' *sum* at
k holds the nominal evidence count fixed, not the information in it:

- The effective sample size of geometric weights levels off at (1+λ)/(1−λ): **7 at
  λ = 0.75**. At k = 3, 12 and 38 it is 2.85, 6.6 and 7.0, while the Beta is still charged
  k observations. The posterior is too concentrated, and because it is drawn once per
  player and shared across the projection window, the variance of multi-gameweek starts
  and totals is **understated**. The understatement grows with k (stats-referee's
  estimate: posterior SD ~1.7× too small by k≈30).
- λ therefore trades off against κ and W. The study shows it: at h=10, λ* is **0.25** with
  an uncapped prior and **0.75** at κ=4. If λ moved only order, λ* would not depend on
  prior strength.
- It also cannot "pick up the benching asymmetry for free", as first claimed: the weights
  do not depend on whether a match was a start.

The estimator with the right concentration is a discounted Beta-Bernoulli filter,
a_t = λ·a_{t−1} + (1−λ)·κ·m₀ + y_t (likewise b), fitted jointly with κ on the board's real
endpoint, with cutoffs out to k≈30. **Not built**; it needs its own pre-registration.

## Result 4 — λ depends on horizon and on κ `[DERIVED]`

`studies/start_recency.py`, leave-one-season-out over 22/23 (from GW16) – 25/26 at κ=4,
Brier on the next *h* matches:

| horizon (matches) | 1 | 2 | 3 | 5 | 10 | rest |
|---|---|---|---|---|---|---|
| λ* | 0.35 | 0.45 | 0.55 | 0.60 | **0.75** | 0.82 |
| LOSO Brier gain | +10.9% | +7.8% | +5.5% | +3.4% | **+1.6%** | +0.5% |
| folds positive | 4/4 | 4/4 | 4/4 | 4/4 | **4/4** | 3/4 |

**The pre-registered rule holds.** Primary endpoint Brier on the next match, gate ≥1%
and ≥3 folds: ADOPT at +10.9% (κ=4) and +5.5% (uncapped), 4/4 folds each. The
pre-registered endpoint **cannot pin λ**: uncapped, the gain is monotone down to the grid
floor, because the last match nearly suffices for the next one.

**What λ = 0.75 is.** It is `[DERIVED]` inside the study's conditions — h=10, κ=4, W=1,
previous-season prior, k ≤ 12, match grain — and on a fold set that includes the mid-season
22/23 fold (see the sensitivity below; on the three season-start folds alone the value is
0.70). Using it as a board constant is `[JUDGMENT]`. The post-hoc cost was choosing the endpoint (h=10 and κ=4 from 12
horizon-by-κ cells after seeing the first run), not the λ value. At a fixed λ=0.75,
κ=4, with no fitting:

| held-out season | 22/23 | 23/24 | 24/25 | 25/26 |
|---|---|---|---|---|
| h = 10 | +1.60% | +3.29% | +1.20% | **+0.81%** |
| rest of season | +0.51% | +1.98% | +0.04% | **−0.42%** |

The most recent season is under the 1% bar at h=10 and negative over the rest of the
season. The board now defaults to `GW_HI=38`, which is nearer the rest-of-season
endpoint. κ was fitted on rest-of-season while λ was taken from h=10, so the pair is
not a joint optimum. And by cutoff, λ* at h=1 rises from 0.15 to 0.40 as k runs 3 → 12,
so a constant λ is not supported late in the season.

**Sensitivity: the three season-start folds alone** `[VERIFIED 2026-09-17]`. Since the
22/23 fix, that fold begins at GW16 against a 21/22 proxy prior. Its first step spans the
World Cup break, and its rest-of-season horizon is at most 23−k matches. A regime break
and a shorter horizon both favour short memory, so the fold pulls λ* **down** and pooled
gains **up**, most at the rest-of-season endpoint. It stays in the reported results
because it is part of the pre-registered fold set; dropping it now would be the post-hoc
move. Refitting without it (LOSO, κ=4):

| endpoint | λ* | mean gain | folds positive |
|---|---|---|---|
| next match (the gate) | 0.35 | +10.3% | 3/3 — **ADOPT holds** |
| h = 10 | **0.70** | +1.5% | 3/3 |
| rest of season | 0.80 | +0.45% | **2/3** |

The gate is unaffected (uncapped: +5.3%, 3/3). At h=10 λ* moves one grid step. Rest of
season drops to 2/3 folds positive, so the "3/4" above partly reflects 22/23's short
horizon. Rest-of-season was already below the 1% gate and stays a null.

Two things were added *after* the first run and are labelled post-hoc in the study: the
λ grid was extended downward (every fold had pinned to the old floor of 0.50), and the
horizon sweep itself.

---

## How it ships

**OFF by default, and it should stay off.** `INSEASON_LAM=0.75` with `INSEASON_KAPPA=4`
reproduces the study's conditions:

```bash
INSEASON_KAPPA=4 INSEASON_LAM=0.75 python scripts/gw_board.py
```

Nothing here licenses it for the default 38-gameweek board (Result 4), for production's
two-season pooled prior, for k > 12, or for the Beta's dispersion (*What renormalisation
does not do*). κ=4 itself survived the 22/23 fix: re-run, `start_prior_strength` keeps
ADOPT at every cutoff and its single-lever optimum at w=1 is κ = 3/4/4/4 across
k = 2/3/5/8 (was 3/4/4/5; the k=8 move is along a flat ridge — 17 grid points within 0.5pp).
The 23/24 fold's prior now comes from 22/23 GW16-38 only (median ~24 pseudo-matches against
38 elsewhere). At κ ≤ 20 the cap binds for anyone with 5+ prior matches, so the κ=4 corner is
unaffected by prior length: per-fold gain at w=1/κ=4 against the fitted pair is 23/24
+6.1%/+7.3%, 24/25 +9.1%/+11.6%, 25/26 +13.3%/+16.2% at k=3/8 (stats-referee, 2026-09-17).
The shorter prior matters only at the uncapped end of the ridge, where it biases the pooled
gain DOWN — so the ADOPT is conservative.

**Renormalisation window, fixed 2026-09-16.** `recency_starts` first renormalised over
the *club's* whole run of gameweeks and then summed a player's rows. That is only right
when his charged window equals the club's run. When it is shorter, recent weights exceed
1 and his starts are over-counted (GW4, λ=0.75, charged GW3-4 only, started GW4: 1.46 of 2
against 1.14 over his own window). It now renormalises over exactly the slots
`appearances` sums into his `club_matches`, so the bound holds by construction.
Measured effect on 26/27 through GW4: **0 of 658 players move** `[VERIFIED]`, because
`team_at_gw` carries every player's club forward and backward, so every charged window
already equals a club's full run. The versions diverge when a blank or double gameweek
sits either side of a transfer, and on the panel fallback.

**Board movement, measured 2026-09-08 on 26/27 through GW3** (GW1-38 season totals, 653
players, pre-fix code — unaffected by the window fix above, which moves nobody), reported
per flag so neither is credited with the other's movement:

| arm | players moved | mean \|Δ\| | max \|Δ\| |
|---|---|---|---|
| κ=4 alone (off → κ=4) | 497 | 7.75 | 38.1 |
| λ=0.75 on top of κ=4 | 102 | 0.73 | 8.8 |

κ is by far the bigger lever. The equal-and-opposite moves of same-club rotation pairs
(Enzo +8.8, O'Reilly −8.8) were first offered as validation; they are not. Two players
sharing a slot must move that way under any reweighting of complementary sequences. Only
a points A/B on scored gameweeks is evidence (PROJECT_KNOWLEDGE §6.9).

`appearances()` attaches `w_starts` **only** when λ ≠ 1, so the flag-off board is
byte-identical to before this existed.

**Correction, 2026-09-08.** The 22 starts that `recency_starts` once dropped and the raw
`starts` column kept were first attributed to EFL Cup ties. They were half-time starts
from an unfinished 26/27 GW3 Arsenal–Chelsea: `gw_panel` admitted a whole gameweek as
soon as *any* match in it had finished. `gw_panel` now applies its finished guard **per
match**, so raw and recency-weighted counts agree at source and λ = 1 is a true identity.

## What stays dead

- A **rotation multiplier** predicts *which* gameweeks a player is rested, from fixture
  congestion. Dead three ways over (rest differential, European participation, midweek
  fixtures by recovery day; P(start) +0.001, CI ±0.02). Nothing here is
  fixture-conditional.
- A **streak / momentum term** is not built, and stays out under this module's standing
  rule (no form, streaks, momentum or confidence terms). It is **not** a tested null: see
  the hindsight caveat under Result 2. The earlier line "momentum is dead — precisely
  Result 2" is withdrawn; Result 2 shows positive excess up to k≈6.

## Limitations

1. **The excess bundles channels this data cannot separate**: manager state dependence
   (a settled XI), availability persistence (an injury is a *block* of zeros, and
   clustered zeros mechanically cluster the ones), and within-season drift in role. A
   `nogap` scope (no 4+ consecutive non-appearance block; 738 player-seasons) keeps the
   pattern at roughly half the size — excess +0.062 at k=1, +0.009 at k=8. That filter
   conditions on the outcome path, dropping anyone benched four weeks running rather than
   only the injured, so it is **suggestive, not clean**. A proper control needs an
   in-squad availability flag, which this data does not carry.
2. **Informative censoring.** A January exit from the league is preceded by losing your
   place, which biases `h` slightly upward at low `k`.
3. **22/23 enters from GW16.** Its fold in `start_recency` starts mid-season against a
   prior from 21/22 (proxy), across the World Cup break, with at most 23−k matches of
   rest-of-season. That pulls pooled λ* DOWN and pooled gains UP, most at rest-of-season
   (see the three-fold sensitivity under Result 4: gate unchanged, h=10 λ* 0.70, rest 2/3
   folds). In `start_persistence` the shorter 22/23 sequences deepen the fixed-count bias on
   the within-player lag-1 (about −1/(n−1)), so that figure is slightly LOW — roughly −0.004
   on +0.436 (stats-referee estimate). The streak excess is unaffected, because the
   permutation null preserves sequence length.
4. **Not yet validated on points.** The endpoint is a start probability, and points depend
   on minutes through the 60-minute threshold and exposure scaling, neither linear in
   `p_start`. The board A/B on scored gameweeks (§6.9) is the outstanding work, and should
   land jointly with the `W_MINUTES`/κ re-fit — they touch one channel, and run apart each
   would be credited with the other's gain.
