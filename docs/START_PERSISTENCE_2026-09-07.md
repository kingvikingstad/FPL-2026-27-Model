# Start persistence: what a start tells you about the next one

*2026-09-07. Evidence: `studies/start_persistence.py` / `.csv`, `studies/start_recency.py` / `.csv`.
Code: `src/inseason.py` (`recency_starts`, `appearances(lam=)`), `scripts/gw_board.py` (`INSEASON_LAM`).*

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

| | estimator | what it removes |
|---|---|---|
| lag-1 | within-player-season fixed effects — each player is his own control | all between-player frailty |
| duration | permutation of the outcome sequence **within** player-season | ordering only; each player's start *count* is held exactly |

The permutation null preserves every player's own start rate and destroys nothing but
the order, so `h_obs(k) − h_perm(k)` is the part of the streak profile that ordering
actually buys. Per the repo's permutation guard, the **outcome** is permuted, not labels.

**Data.** 113,571 player-matches, 1,589 players, 2022/23–2025/26, native `starts` column.
Unit is the player-**match** ordered by kickoff (a double gameweek is two ordered matches).
Streak resets each season. Replicated on 2016/17–2021/22 under the `mins ≥ 60` proxy,
which is reported separately rather than pooled because it spuriously *breaks* streaks
whenever a starter is hooked before the hour.

---

## Result 1 — one start on the next start `[VERIFIED]`

| scope | P(start \| started) | P(start \| benched) | pooled gap | **within-player gap** |
|---|---|---|---|---|
| native 22/23–25/26 | 0.798 | 0.076 | +0.722 | **+0.493** |
| proxy 16/17–21/22 | 0.784 | 0.111 | +0.673 | **+0.405** |

**About two-thirds of the raw gap is genuine state dependence; one-third is frailty.**
Still the largest single-lag effect on the minutes layer.

## Result 2 — duration dependence dies at k ≈ 6–8 `[VERIFIED]`

Native scope. `h_null` is the frailty-preserving permutation null; CIs are a cluster
bootstrap over `player_code`.

| k | n | h_obs | 95% CI | h_null | excess | z |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 6,600 | 0.656 | [0.642, 0.670] | 0.467 | **+0.190** | 47.6 |
| 2 | 4,240 | 0.761 | [0.745, 0.775] | 0.574 | **+0.187** | 30.4 |
| 3 | 3,147 | 0.800 | [0.783, 0.818] | 0.648 | **+0.153** | 19.4 |
| 4 | 2,456 | 0.813 | [0.798, 0.828] | 0.703 | +0.109 | 10.2 |
| 5 | 1,943 | 0.847 | [0.827, 0.865] | 0.745 | +0.101 | 9.6 |
| 6 | 1,609 | 0.861 | [0.845, 0.877] | 0.781 | +0.080 | 5.9 |
| 8 | 1,133 | 0.859 | [0.837, 0.880] | 0.829 | +0.030 | 1.9 |
| 10 | 809 | 0.874 | [0.850, 0.897] | 0.863 | +0.011 | 0.7 |
| 15 | 395 | 0.896 | [0.866, 0.924] | 0.906 | −0.010 | −0.5 |
| 20 | 195 | 0.959 | [0.926, 0.980] | 0.936 | +0.023 | 0.9 |
| 25 | 102 | 0.931 | [0.870, 0.973] | 0.955 | −0.023 | −1.0 |

Raw `h(k)` climbs to 0.96 by k = 20 — **and so does the null.** Past about six consecutive
starts, everything the streak says is already in the player's own base rate. Replicated in
the proxy scope (+0.145 at k=1, +0.031 by k=8, noise at 15/20/25).

`h_obs` and `n` are exact; `h_null`, the CI and therefore `excess`/`z` are resampled
(N_PERM=200, N_BOOT=300) and move by ~0.003 between runs. Read the third decimal of those
columns as noise — it does not touch any conclusion here, where the live comparisons are
+0.19 against a null SD of 0.004 at one end and +0.01 against 0.016 at the other.

**Power.** At k=20 the null SD is 0.024 and the bootstrap CI contains the null. This rules
out an effect ≳ +0.05; it does not rule out ±0.03. A null for decision-relevant
magnitudes, not a proof of exact zero.

## Result 3 — the asymmetry is the usable half `[VERIFIED]`

P(start next | *k* consecutive **non**-starts), native:

| k | 1 | 2 | 3 | 5 | 8 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| h | 0.276 | 0.154 | 0.114 | 0.080 | 0.046 | 0.036 | 0.014 |

**Being dropped is far more informative than being picked, and it saturates much more
slowly.** Three benchings cut the next-start probability to ~11%; three starts move a
player only from 0.66 to 0.80.

---

## What this licenses, and what it does not

The finding is **not** a streak covariate. The excess is already zero past k ≈ 8, so a
`streak_k` term would mostly re-encode the player's base rate, which the Beta prior
already holds — that is double counting, and it is why the streak curve is the *diagnosis*
and not the fix.

What it indicts is the **likelihood**. `inseason.update_minutes` is a conjugate Beta
update; a Beta update reads a *count* and is therefore **exchangeable**, so
start-start-bench and bench-start-start produce an identical posterior. The measured
ordering information is invisible to it by construction.

The fix is a geometric recency weight on the estimator's own observations, `u_d = λ^d`,
renormalised so total evidence mass equals the raw match count. Renormalising matters for
identification: without it, λ < 1 would both shorten memory *and* shrink the in-season
weight, confounding this with the `W_MINUTES` / `START_KAPPA` ridge that
`start_prior_strength.py` already fitted. As specified, λ moves **only** the order.

## Result 4 — λ is a function of forecast horizon `[VERIFIED]`

`studies/start_recency.py`, leave-one-season-out over 22/23–25/26 at κ=4, Brier on the
next *h* matches:

| horizon (matches) | 1 | 2 | 3 | 5 | 10 | rest |
|---|---|---|---|---|---|---|
| λ* | 0.40 | 0.50 | 0.55 | 0.65 | **0.75** | 0.82 |
| half-life (matches) | 0.8 | 1.0 | 1.2 | 1.6 | **2.4** | 3.6 |
| LOSO Brier gain | +7.9% | +5.6% | +4.0% | +2.6% | **+1.2%** | +0.3% |
| folds positive | 4/4 | 4/4 | 4/4 | 4/4 | **4/4** | 3/4 |

**Short horizon, short memory.** The pre-registered gate (≥1% LOSO gain, positive in ≥3
folds, primary endpoint = next match) fires ADOPT decisively at +3.5%. But the
pre-registered endpoint **cannot pin λ**: at h=1 the gain is monotone all the way down to
the grid floor, because the last match is very nearly a sufficient statistic for the next
one. That limit — "ignore everything but the last match" — is correct at h=1 and
catastrophic for a ten-week board, which is exactly why the horizon sweep exists.

The board projects ~10 gameweeks, so **λ = 0.75** is its value. Rest-of-season fails the
1% gate, so nothing here licenses recency weighting for a season-long projection.

Two things were added *after* the first run and are labelled post-hoc in the study: the
λ grid was extended downward (every fold had pinned to the old floor of 0.50, so nothing
was actually being fitted — the same correction `start_prior_strength` made to its κ
grid), and the horizon sweep itself.

---

## How it ships

**OFF by default.** `INSEASON_LAM=0.75`, set together with `INSEASON_KAPPA=4`:

```bash
INSEASON_KAPPA=4 INSEASON_LAM=0.75 python scripts/gw_board.py
```

Off by default for two reasons, both about what was actually fitted: λ was fitted at κ=4,
which `INSEASON_KAPPA` itself leaves off, and against the *uncapped* production prior λ*
pins to a grid boundary rather than fitting. It also inherits the standing caveat on that
channel — validated on the **parameter**, not yet on **points** (PROJECT_KNOWLEDGE §6.9).
**Measured on the GW1-38 board, 26/27 data through GW3**, decomposed so the two flags are
not credited with each other's movement (season-total points per player, 653 players):

| arm | players moved | mean \|Δ\| | max \|Δ\| |
|---|---|---|---|
| κ=4 alone (off → κ=4) | 497 | 7.75 | 38.1 |
| **λ=0.75 on top of κ=4** | **102** | **0.73** | **8.8** |
| both | 497 | 7.99 | 38.1 |

**κ is by far the bigger lever** — capping a 34-pseudo-match prior at 4 is a large change,
and reporting the two together would have credited recency with it. Recency's own
contribution is modest after three gameweeks, which is expected: with `k=3` the
reweighting can only bite where a player's start pattern is non-constant.

The recency-only moves have the right *shape*, which is the more informative check.
Rotation pairs at the same club move in **opposite directions** — Enzo +8.8 against
O'Reilly −8.8 and Foden −8.4 at Man City — i.e. the board now separates players with
identical season start counts by *when* those starts happened. That is the exchangeability
failure being corrected, visible directly.

`appearances()` attaches the column **only** when λ ≠ 1, so the flag-off board is
byte-identical to before this existed.

**Correction, 2026-09-08.** That gate originally carried a second job, and the reason I
first gave for it was wrong. `recency_starts` places each start by the league fixture it
belongs to, so it drops any start with no finished league fixture behind it — and
`gw_panel` was feeding it 22 such starts, which I attributed to EFL Cup ties. It was not
cup ties: `gw_panel` admitted a whole gameweek as soon as *any* match in it had finished,
so 26/27 GW3 carried 22 half-time starts from an unfinished Arsenal–Chelsea that `played`
correctly excluded. That is why the raw `starts` column disagreed with its own denominator
for 16 players — a discrepancy `update_minutes` had always masked by clamping to
`club_matches`. `gw_panel` now applies its finished guard **per match**, so the two agree
at source and λ = 1 is a true identity on the raw count. The gate stays because the raw
path is cheaper and byte-identity is worth keeping, not because the answers still differ.

Two consequences of that fix worth carrying: `recency_starts` keys its output on
`team_now` while placing each start on the club he played for that week, so a player who
moved mid-season stays one row and keeps his whole record; and `club_matches` is now
summed per gameweek over the club he was at that week, which is the honest denominator for
the Beta update the moment a blank or double gameweek straddles a transfer.

## What stays dead

This is not a rotation multiplier and not a momentum term, and the distinction is
measured, not asserted:

- A **rotation multiplier** predicts *which* gameweeks a player is rested, from fixture
  congestion. Dead three ways over (rest differential, European participation, midweek
  fixtures by recovery day; P(start) +0.001, CI ±0.02). Nothing here is
  fixture-conditional.
- A **momentum term** asserts a run of starts predicts *more than* the player's own rate.
  Also dead — that is precisely Result 2, and it is why there is no `streak_k` covariate.

What survives is narrower than either: the *order* of a player's own realised matches is
informative over a window of roughly six, and a Beta update on a count is blind to it.
The fix is a weight on the likelihood's observations, not a new predictor.

## Limitations

1. **The excess bundles two channels this data cannot separate**: manager state dependence
   (a settled XI) and availability persistence (an injury is a *block* of zeros, and
   clustered zeros mechanically cluster the ones). Both are real for forecasting; only the
   first is "trust". A `nogap` scope (no 4+ consecutive non-appearance block) reproduces
   the pattern — excess +0.112 at k=1, +0.020 by k=8 — but that filter conditions on the
   outcome path, dropping anyone benched four weeks running rather than only the injured,
   so it over-selects nailed-on players and is **suggestive, not clean**. A proper control
   needs an availability (in-squad) flag, which this data does not carry.
2. **Informative censoring.** A January exit from the league is preceded by losing your
   place, which biases `h` slightly upward at low `k`.
3. **Not yet validated on points.** Same caveat as every other channel on this layer: the
   endpoint is a start probability, and points depend on minutes through the 60-minute
   threshold and exposure scaling, neither linear in `p_start`. The board A/B on scored
   gameweeks (§6.9) is the outstanding work, and this should land in the same A/B as the
   `W_MINUTES`/κ re-fit rather than separately — they touch one channel, and run apart
   each would be credited with the other's gain.
