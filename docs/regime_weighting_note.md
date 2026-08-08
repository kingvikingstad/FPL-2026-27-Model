# Regime Weighting: Flexibility and Variance Propagation

**Context note for the model-building conversation — FPL-Core-Insights, 2026/27**
Prepared 6 Aug 2026. Status of claims flagged throughout: `[VERIFIED]` = read from
code/source, `[DERIVED]` = analytic result, `[JUDGMENT]` = not fitted, needs a sweep.

---

## 1. The problem in one sentence

The regime discount is documented as a *variance* statement ("trust history less here")
but is implemented as a *mean-only* operation — it moves the point estimate and leaves
the posterior width untouched. Every downstream SD, credible interval, P(haul) tail and
captaincy risk profile is therefore currently blind to managerial regime change.

---

## 2. What the code actually does `[VERIFIED]`

`starter_prior.apply_minutes_shrinkage_with_regime()`:

```python
N        = r.start_a + r.start_b            # Beta concentration
p_hist   = r.start_a / N
p_own    = coldstart_start_prob(own, cost, pos, cal)
eff_min  = minutes * regime_discount(club)  # <-- the discount
w        = eff_min / (eff_min + k_min)
p_new    = w * p_hist + (1 - w) * p_own
p.at[i, "start_a"] = p_new       * N        # <-- N PRESERVED
p.at[i, "start_b"] = (1 - p_new) * N        # <-- N PRESERVED
```

`Var[p] = p(1-p) / (N + 1)`. Because `N` is carried through unchanged, the discount
cannot widen the posterior. It can only slide the mean toward the ownership-implied
prior. The same concentration-preserving convention is used deliberately in
`apply_transfer_fee_floor()` (documented there as intentional, and correct in that
context — a floor genuinely is a mean statement).

Two consequences worth stating plainly:

1. The regime discount is currently a **directional** operation, which is the exact
   thing the module docstring says it is not.
2. Sweeping `REGIME_DISCOUNT_2627` today can only ever change point estimates. It
   cannot produce the "wider intervals at Liverpool/Newcastle/Chelsea" behaviour the
   design intends.

---

## 3. Required parameterisation — decouple mean-pull from variance-inflation

Replace the single per-club scalar with **two independent knobs**, both config-driven
and sweepable, defaulting to a no-op:

| Symbol | Name | Range | Meaning | No-op value |
|---|---|---|---|---|
| `δ_club` | minutes discount | (0, 1] | scales effective minutes → drives `w` → pulls the mean toward `p_own` | `1.0` |
| `κ_club` | disagreement inflation | [0, 1] | weight on the between-component variance term → widens the posterior | `0.0` |

`δ = 1.0, κ = 0.0` must reproduce the current validated baseline **bitwise**.

Keeping these separate matters because they encode different claims. `δ` asserts the
market's ownership signal is more reliable than history at this club. `κ` asserts only
that we don't know — no direction. It is legitimate to end up with `δ = 1.0, κ > 0`,
i.e. don't move the mean at all, just widen. That configuration is currently
unreachable, and it may well be the honest one.

Neither should stay hard-coded in a module-level dict. Move to a config surface
(YAML/env/CLI) so the sweep harness can drive them, and log the resolved values in the
run manifest alongside `market_share` and `older_weight`.

---

## 4. The variance mechanics `[DERIVED]`

### 4.1 Moment-matched mixture

Treat the shrunk prior as a genuine two-component mixture rather than a mean blend.
With history component `Beta(a_h, b_h)`, `N_h = a_h + b_h`, and ownership component
`Beta(a_o, b_o)`, `N_o = a_o + b_o` (use `kstart = 6.0` as the existing fallback):

```
p̄  = w·p_h + (1-w)·p_o
v_h = p_h(1-p_h) / (N_h + 1)
v_o = p_o(1-p_o) / (N_o + 1)

v  = w·v_h + (1-w)·v_o  +  κ · w(1-w)·(p_h - p_o)²
                            ^^^^^^^^^^^^^^^^^^^^^^
                            between-component term
N_eff = p̄(1-p̄)/v - 1
a = p̄·N_eff,   b = (1-p̄)·N_eff
```

The between-component term is the whole point. It is maximised at `w = 0.5` — i.e.
precisely where the regime discount pushes players at affected clubs — and it scales
with the **squared disagreement** between what history says and what the market says.

A Newcastle regular whose ownership has collapsed after four first-XI departures has a
large `(p_h - p_o)²` and gets a genuinely wide interval. A Man City player whose
history and ownership agree gets almost no widening even at `δ = 0.5`. That asymmetry
is the correct behaviour and it falls out of the algebra rather than being asserted.

**Guard required:** if `v ≥ p̄(1-p̄)` the moment match returns `N_eff ≤ 0`. Clamp to a
floor (`N_eff ≥ 1.0`, i.e. Uniform) and count clamp events in the run log — a high
clamp rate means `κ` is set too aggressively.

### 4.2 Propagation to points — the horizon result

This is the part that determines whether any of the above shows up on the projection
board.

Let `S_h` be the start indicator in gameweek `h`, `S_h | p ~ Bern(p)` i.i.d. given `p`,
and `Y` points-given-start with mean `μ`, variance `σ²`. Over an `H`-gameweek horizon:

```
E[X_H]   = H·p̄·μ

Var[X_H] = H·[ p̄σ² + p̄(1-p̄)μ² ]   +   H(H-1)·μ²·v_p
           ^^^^^^^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^
           within-GW (H = 1 term)        shared-p term
```

Three implications:

1. **At `H = 1` the second term vanishes.** Uncertainty in `p` is fully absorbed into
   the Bernoulli marginal for a single gameweek. So the regime discount correctly has
   *no* effect on single-GW SD — and anyone testing the change on a GW1-only run will
   see nothing and conclude it's inert.
2. **At `H = 6` the shared-p term carries a factor of 30 vs 6.** For the GW1–6 board
   (`decision_gw1_6_*.csv`) it dominates. This is where regime uncertainty belongs.
3. **It scales with `μ²`.** Premium assets at regime-change clubs take the widening
   quadratically harder than fodder. Salah under Iraola and Palmer under Alonso should
   end up with visibly fatter intervals than a £4.5m defender at the same clubs. If
   they don't, the propagation isn't wired up.

### 4.3 The implementation detail that makes or breaks it

Given the pipeline already draws full posteriors for captaincy: **`p` must be drawn
once per player per simulation path and held fixed across the horizon.** If the
simulator redraws `p` each gameweek, or substitutes the point estimate `p̄`, the
`H(H-1)` term is destroyed and the entire mixture widening is silently discarded.

Concretely: the draw order must be `p ~ Beta(a,b)` outer, `S_h ~ Bern(p)` inner. Not
the reverse, and not `S_h ~ Bern(p̄)`.

This is the single highest-risk line item in the change. It should get an explicit
test (see §6, test 4).

---

## 5. Proposed club settings `[JUDGMENT]`

Nine full regime changes for 26/27 — **Newcastle was missing from the current dict**,
which was frozen 2026-07-28. Jaissle was appointed 5 Aug 2026, ~18 days before the
opener, with Isak, Trippier, Gordon and Tonali gone and Guimarães reportedly departing.

| Club | Manager | δ (current) | δ (proposed) | κ (proposed) | Rationale |
|---|---|---|---|---|---|
| Newcastle | Jaissle | *absent* | 0.35 | 0.8 | No pre-season with squad; four first-XI departures; worst information state in the league |
| Liverpool | Iraola | 0.50 | 0.50 | 0.8 | Only same-league first-10 evidence in the set (3 pts from 9 at Bournemouth), and it's negative |
| Chelsea | Alonso | 0.50 | 0.50 | 0.7 | Fast-start history but heavy squad churn |
| Nott'm Forest | Glasner | 0.50 | 0.50 | 0.7 | Documented slow starter; 5th HC in ~12 months |
| Crystal Palace | Sage | 0.50 | 0.50 | 0.7 | No PL data at all |
| Bournemouth | Rose | 0.50 | 0.50 | 0.7 | No PL data; post-Semenyo squad |
| Fulham | Arbeloa | 0.50 | 0.50 | 0.7 | Near-uninformative prior (5 months of inherited elite personnel) |
| Ipswich | O'Neil | 0.50 | 0.55 | 0.6 | Known PL quantity; least disruptive |
| Man City | Maresca | 0.50 | 0.55 | 0.6 | Strongest positive first-10 evidence; low continuity risk |
| Tottenham | De Zerbi | 0.55 | 0.45 | 0.7 | ~9 GWs *and* a relegation scrap — sample is small **and** unrepresentative of his system |
| Man United | Carrick | 0.70 | 0.80 | 0.4 | Now 17 games and permanent; 0.70 was set while interim |

None of these are fitted. They are starting points for a sweep, and the note should be
read as "here is the axis to sweep along", not "here are the values".

**Cleaner alternative, flagged previously and still preferred:** split the per-match
panel by appointment date for Spurs and Man United rather than applying a flat
multiplier. The flat multiplier discards the information that ~19 of Man United's 25/26
matches *were* under the current regime. If the date split is cheap, do that instead of
tuning `δ_ManUtd`.

---

## 6. Acceptance tests

1. **Baseline identity.** `δ=1.0, κ=0.0` reproduces `decision_gw1_6_full.csv` bitwise.
2. **Isolation.** Rows for clubs not in the regime dict are bitwise unchanged under any
   `(δ, κ)`.
3. **Monotonicity.** For affected players, `∂SD/∂κ > 0` and `∂SD/∂δ < 0` over the GW1–6
   horizon. Assert, don't eyeball.
4. **Horizon signature.** For a fixed player, plot SD against `H` for `H ∈ {1..6}` at
   `κ=0.8`. The `H=1` SD must equal the baseline `H=1` SD (§4.2 point 1); the gap must
   grow superlinearly thereafter. **If the `H=1` values differ, the simulator is
   redrawing `p` per gameweek and the propagation is wrong.**
5. **Decoupling.** A player with `p_h ≈ p_o` at `δ=0.5, κ=0.8` must show ≈0 mean shift
   but non-zero SD widening. Confirms the two knobs are genuinely independent.
6. **`μ²` scaling.** Rank affected players by SD inflation; it should correlate with
   projected EP, not be flat across price tiers.
7. **Benchmark guard.** Rank correlation against the Solio GW1 benchmark must not
   degrade. Widening SDs should not move the *ordering* much — if it does, `δ` is
   doing directional work it shouldn't be.

---

## 7. Interaction with `market_share`

Nine regime changes means bookmaker disagreement on team ratings will be unusually wide
in GW1–3. That cuts both ways and the sweep needs to consider them jointly:

- Wider market disagreement is an argument for *lower* `market_share` (the market is
  itself noisy right now), **and**
- Regime change is an argument for *higher* `market_share` (our historical priors are
  the thing that's stale).

These pull in opposite directions and the current 0.35 is asserted, not calibrated.
Sweep `market_share × κ` on a grid rather than one at a time — the optimum for one is
almost certainly conditional on the other, and a coordinate-wise sweep will find a
local answer and present it as the answer.

---

## 8. What not to do

The temptation with nine new managers is to add directional style priors — "Iraola
presses high, bullish Liverpool attackers", "Glasner is defensively elite, buy Forest
CBs". These are the same class of heuristic as the rotation multipliers (null, p=0.23)
and mean-reversion (rejected, p=0.69). The Glasner slow-start pattern is the only one
with even a suggestion of a repeatable signature, and n=2 seasons is not a test.

`κ` is the correct home for everything we believe about regime change, because
everything we believe about regime change is a statement about ignorance.
