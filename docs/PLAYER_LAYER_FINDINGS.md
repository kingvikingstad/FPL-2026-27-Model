# Player-Layer Findings — minutes exposure, congestion, penalties

**Opened:** 2026-08-11 · Three hunches pursued for predictive gain. One is a real bias that
has been fixed, one is a clean null, one corrects a claim made earlier in this project.

---

## 1. `[FIXED]` The model assumed every starter plays exactly 90 minutes

`studies/minutes_distribution.py`. The single most valuable thing found in this pass.

`bayes_model.project` drew minutes from a **two-point distribution**:

```python
mins = np.where(start, 90.0, np.where(sub, 20.0, 0.0))
```

`start` comes from `Beta(start_a, start_b)`, whose counts were fitted on `minutes >= 60` —
so the "start" branch means *played 60+*, which makes `played60` (and with it clean-sheet
eligibility and the 2-point appearance) correct by construction. **The exposure was not.**
`m90 = mins/90` multiplies attacking involvement, penalty xG and DefCon counts, and
assuming a full 90 for everyone who clears the hour inflates all three.

Measured over 23,059 appearances across 24/25 and 25/26:

| pos | E[minutes given 60+] | m90 | inflation | plays the full 90 |
|---|---|---|---|---|
| GK | 89.9 | 0.999 | 0.1% | 99.5% |
| DEF | 87.5 | 0.972 | 2.9% | 81.8% |
| **MID** | **83.1** | **0.923** | **8.4%** | 52.2% |
| **FWD** | **81.5** | **0.905** | **10.5%** | 42.9% |

Only **52% of midfielders and 43% of forwards** who clear the hour finish the match. The
bias is systematic, one-directional, applies to every gameweek, and lands hardest on
exactly the players whose attacking return drives the projection.

### The fix

`MINUTES_IF_START` (per position) and `MINUTES_IF_SUB = 22.0` replace the constants;
`FPL_MINUTES_MODEL=flat` restores 90/20 for A/B. The conditional **mean** is the right
substitution rather than a sampled distribution: exposure enters linearly
(`E[Poisson(λ·m90)] = λ·E[m90]`) and the 60-minute threshold has already been passed on
that branch.

### Board effect

```
GK   -0.06%     DEF  -1.36%     MID  -3.42%     FWD  -4.17%
```

Ordering matches the measured inflation. The board effect is smaller than the raw
inflation because appearance points and clean sheets are unaffected — only exposure-scaled
components move.

**Clean sheets are untouched**, as intended: `cs_ev` moves −0.06% in total, pure Monte
Carlo noise from the RNG stream, and `played60` is unchanged by construction.

**DefCon falls 13.1%** — far more than the 3–8% exposure change, and this is the most
consequential part of the fix. DefCon points are a *threshold* (10 actions for defenders,
12 for midfielders), so `P(Poisson(λ) ≥ 10)` is convex in λ and an 8% cut in exposure
removes much more than 8% of the probability mass. The DefCon channel — which drives the
cheap-defender strategy the board surfaces — was materially over-valued.

### External check

| | Pearson | Spearman | MAE | bias |
|---|---|---|---|---|
| flat 90 | 0.7496 | 0.5401 | 0.8282 | −0.641 |
| measured | **0.7600** | **0.5669** | 0.9018 | −0.823 |

**Rank agreement improves; level agreement worsens.** The level comparison is already
known-invalid — `run_solio_ensemble.py` states that Solio's feed is 25/26 GW35 against our
26/27 GW1 — so Spearman, which is invariant to a level shift, is the trustworthy metric
here, and it improves. n = 29, so this is weak either way; the case for the change rests
on correctness, not on Solio.

All four acceptance tests pass unchanged.

### 1b. `[FIXED]` Refined again — per PLAYER, not per position

`studies/minutes_persistence.py`. The positional constant still says a 90-minute
centre-half and a forward always hooked on 65 have identical exposure once they start.
They do not, and the difference persists:

- split-half reliability of a season's conditional minutes: **0.819**
- season-to-season persistence: r = 0.652, **0.796 disattenuated**, n = 1,755 pairs
- spread within position is real: MID sd 4.16 min, FWD sd 4.38

Out-of-sample horse race predicting next season's conditional minutes:

| predictor | MAE | RMSE | bias |
|---|---|---|---|
| flat 90 (original) | 3.614 | 5.271 | −3.614 |
| positional constant | 2.649 | 3.406 | +1.202 |
| **shrunk player history** | **1.968** | **2.876** | **+0.019** |

A **25.7% MAE gain** over the positional constant, clustered CI (+0.579, +0.784), and it
removes the constant's +1.2 minute bias almost exactly. Shrinkage weight is empirical —
`k = (1−rel)/rel × median n = 5.7` appearances — not chosen by taste.

Implemented as `exp_minutes` in `multiseason_priors.to_priors`, consumed by
`_minutes_if_start`, with the positional constant as the fallback for players with no
60+ appearances. `FPL_MINUTES_MODEL=positional` pins to the constant for A/B;
`flat` restores the original 90/20.

**A wiring trap worth recording:** the runners copy priors into the player frame through
an explicit column whitelist. The first A/B returned *exactly zero difference* on every
row, because `exp_minutes` was computed, stored, and then silently dropped at that
boundary. A change that appears to do nothing is more likely unplumbed than ineffective —
the whitelist appears in six runners and four tests, all now updated.

**Board A/B** (same seed, both arms on the same Solio cache):

```
overall mean +0.0024   mean |d| 0.053   max |d| 0.684
GK +0.05%   DEF +0.15%   MID +0.03%   FWD +0.40%
```

Small in aggregate, meaningful per player, and directionally sensible: Haaland **+4.16**
over GW1–10 (he finishes matches, so the FWD constant of 81.5 under-rated him), Thiago
+2.05, B.Fernandes +2.02; rotation-prone squad players fall — Okafor −1.49, Estêvão −1.20,
Brooks −1.18.

Against Solio, **all four metrics improve** — unlike the positional step, which improved
rank but not level:

| | Pearson | Spearman | MAE | bias |
|---|---|---|---|---|
| positional | 0.7936 | 0.6055 | 0.9465 | −0.882 |
| **player history** | **0.7951** | **0.6166** | **0.9139** | **−0.828** |

(The Solio baseline differs from the section above because the cached feed refreshed
between runs; both arms here share one cache, so the comparison is internally valid.)

---

## 1c. `[NULL]` Age adds nothing beyond a player's own minutes history

`studies/age_minutes.py`. Asked because minutes is the dominant lever and the Beta prior
has no age term — a 35-year-old with a strong record gets a strong prior.

The right question is the **incremental** one: does age predict next season's minutes
*after* conditioning on this season's? A declining 34-year-old already shows up as
declining minutes, so age is an indirect proxy for something now measured directly.

209 players with ≥8 appearances of 60+ in both 24/25 and 25/26 and a known date of birth:

```
y ~ minutes_t         r2 = 0.4719
y ~ minutes_t + age   age +0.0669 (se 0.0651, t +1.03)   r2 = 0.4746
incremental r2 from age: +0.0027      95% CI (-0.068, +0.203)
```

A 34-year-old and a 26-year-old with identical records differ by **0.5 minutes** next
season. The raw age bands show no gradient either — season-on-season change in conditional
minutes runs −0.27, +0.35, +0.43, −0.43, −0.27 across <23 / 23-26 / 26-29 / 29-32 / 32+.
A quadratic adds +0.012 r², with age² at −0.026 (se 0.014); not enough to change anything.

**Why this ran on one transition rather than twelve.** `birth_date` exists in vaastav only
from 2024/25. The alternatives were back-filling via the permanent `code` — which covers
only survivors, and survivorship correlates directly with the decline being measured — or
an external source. FBref's soccerdata reader was tried and **abandoned**: it installs a
full browser-automation stack (`selenium`, `seleniumbase`, `PyAutoGUI`, `PyGetWindow`) and
managed one season per ~25 minutes with connection errors.

The cheap test costs minutes and answers the actual decision — *is there signal here worth
paying for?* There is not. **Do not scrape historical ages.** The lesson is to run the
cheap version of an expensive question first; the expensive path was started before its
cost was checked.

---

## 2. `[NULL]` Fixture congestion does nothing measurable

`studies/rest_congestion.py`. The model has no congestion term at all, so if rest mattered
it would be a per-fixture adjustment available in every gameweek.

Estimand is the **rest differential** (own days since last PL match minus the opponent's),
because raw rest is confounded — the clubs on short rest are disproportionately the good
ones, via cup runs and TV picks — while the differential is symmetric by construction.
Outcome is opponent- and venue-adjusted residual xG.

```
resid_xg ~ rest_diff : +0.00000 xG per extra day of rest advantage
                       clustered 95% CI (-0.00434, +0.00402)
restricted to |rest_diff| >= 2 days (n=1908): +0.00021, CI (-0.00413, +0.00504)
```

Nothing. Even a four-day swing sits inside ±0.017 xG, about 1% of a team-match. The
sharper cut — playing on ≤3 days' rest against an opponent on 5+ — gives −0.070 xG with
CI (−0.170, +0.033), directionally sensible but not significant on 208 fixtures.

**Measurement limit, stated because it weakens the null:** only Premier League matches are
visible, so a club playing a Thursday Europa tie looks fully rested by Sunday. That is real
error and it attenuates toward zero. The CI is nonetheless tight enough to exclude anything
worth a model parameter.

---

## 3. `[CHECK]` Penalties — the declared order is the better signal, correcting an earlier claim

`studies/penalty_assignment.py`. Both sides live in `player_code` space — the repo's
`playermatchstats` carries `penalties_scored`/`missed` keyed on `player_id`, and vaastav's
`players_raw` carries `penalties_order` keyed on `code`, which **is** `player_code` — so no
crosswalk is needed.

### How persistent is penalty duty?

1,164 penalties over 12 Understat seasons, 141 club-season transitions with ≥3 penalties on
both sides:

- previous top taker still at the club: **67.4%**
- of those who stayed, still the top taker: **75.8%**
- their share of this season's penalties: 63.5% (was 68.0%)
- unconditional, the previous top taker takes **42.8%** of this season's penalties
- concentration: the top taker takes **70%** of his club's penalties

Penalty duty is a sticky role, but a third of takers leave each summer, which is what caps
the value of measured history.

### The horse race, predicting 25/26

| signal | precision | flagged | covers |
|---|---|---|---|
| **declared `penalties_order == 1`** | **85.7%** | 14 | **39.1%** |
| took ≥2 penalties last season | 45.0% | 20 | 34.8% |
| either signal | 56.0% | 25 | 45.7% |

**The declared order wins on both precision and coverage.** This corrects
`docs/SOCCERDATA_FINDINGS.md` §6.4, which noted that `pen_xg90_measured` is computed and
then discarded in favour of the declared-order heuristic and implied that was a gap. It is
not — on this test the declared order is the better predictor, and the runners are right to
use it.

### What the real gap is

Coverage, not choice of signal. The declared order identifies only **39% of the penalties
actually taken**: 14 players are flagged, 39 took one, and roughly six clubs have no
declared first-choice taker at all. Combining both signals lifts coverage to 45.7% at a
cost in precision.

At 5.0 penalties per club-season, 0.761 xG each and ~83% conversion, a settled taker is
worth about 4.1 goals — roughly 16 points for a midfielder before assists or bonus, more
than many players' entire non-penalty return. Missing 55–60% of that EV is material, but
closing it needs a better source for penalty duty (in-season reassignment, press
confirmation) rather than a different way of combining the two signals already available.

**Not changed.** The current heuristic is validated; the residual opportunity is a data
problem, not a modelling one.

---

## Age curves — since tested and null, see §1c above
