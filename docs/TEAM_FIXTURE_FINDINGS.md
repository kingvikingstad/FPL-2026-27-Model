# Team Archetypes as a Fixture Predictor — Findings

**Opened:** 2026-08-09 · **Data:** Understat 2014/15–2025/26 (9,120 team-matches, 12 seasons)
plus football-data 1993–2025 (11,944 matches, 31 seasons) for the hyperparameters
**Study:** `studies/team_archetype_study.py` · **Calibrators:** `src/history.py`

Two rules the study obeys, because they decide whether any of this is tradeable:

1. **No lookahead.** Every archetype uses only what is knowable *before* a ball is kicked:
   whether a club was promoted, and where it finished *last* season. Bucketing by how a
   team ends up doing this season would "predict" GW3 from the final table.
2. **Season-relative units.** League scoring drifted from 2.57 goals/game (14/15) to 3.28
   (23/24). Everything is a ratio to that season's league mean, so `1.20` means "20% above
   the league *that year*".

xG rather than goals throughout — 38 matches is far too few for goals to measure quality.

---

## Headline, with the correction

The eye-catching pattern is that prior-top-6 sides look strongest in GW1–6 (attack index
1.325, falling to 1.232 by midwinter). **That effect is not statistically significant** — a
cluster bootstrap over team-seasons puts it at +0.063 with a 95% CI of (−0.027, +0.137).

The one early-season effect that *does* survive is the opposite and less obvious:
**prior-7th-to-12th teams attack ~7% worse in GW1–6 than their own season average**,
−0.068 with CI (−0.126, −0.007), and they improve monotonically all year.

| archetype | early effect on attack | 95% CI | team-seasons | verdict |
|---|---|---|---|---|
| prior top-6 | +0.063 | (−0.027, +0.137) | 66 | not significant |
| **prior 7–12** | **−0.068** | **(−0.126, −0.007)** | 66 | **real** |
| prior 13–20 | −0.029 | (−0.067, +0.027) | 55 | not significant |
| promoted | −0.040 | (−0.110, +0.037) | 33 | not significant |

---

## Q1 — How do promoted sides and top-six fare across the season?

**xG created** (1.00 = league mean that season):

| archetype | GW1–6 | GW7–19 | GW20–30 | GW31–38 |
|---|---|---|---|---|
| prior top-6 | 1.325 | 1.285 | 1.232 | 1.270 |
| prior 7–12 | 0.905 | 0.931 | 0.980 | 1.007 |
| prior 13–20 | 0.842 | 0.858 | 0.863 | 0.892 |
| promoted | 0.735 | 0.760 | 0.780 | 0.784 |

**xG conceded** (lower = better):

| archetype | GW1–6 | GW7–19 | GW20–30 | GW31–38 |
|---|---|---|---|---|
| prior top-6 | 0.773 | 0.809 | 0.815 | 0.857 |
| prior 7–12 | 1.034 | 1.011 | 1.021 | 1.017 |
| prior 13–20 | 1.082 | 1.062 | 1.069 | 1.116 |
| promoted | 1.187 | 1.214 | 1.173 | 1.231 |

Points per game and clean-sheet rate:

| archetype | ppg GW1–6 | ppg GW31–38 | CS GW1–6 | CS GW31–38 |
|---|---|---|---|---|
| prior top-6 | 1.99 | 1.80 | 0.366 | 0.337 |
| prior 7–12 | 1.22 | 1.44 | 0.258 | 0.286 |
| prior 13–20 | 1.07 | 1.14 | 0.191 | 0.202 |
| promoted | 0.95 | 0.91 | 0.207 | 0.174 |

Two readings that matter:

- **Promoted sides do not "adapt".** Attack creeps from 0.735 to 0.784 while their defence
  *worsens* (1.187 → 1.231) and points per game *fall* (0.95 → 0.91). Whatever early
  competitiveness they have erodes. There is no point waiting for promoted attackers to
  come good.
- **Mid-table is a rising asset class.** Prior 7–12 go 0.905 → 1.007 on attack and 1.22 →
  1.44 ppg. This is the same population as the significant early-season fade above: they
  are genuinely worse in the opening six than they will be later.

---

## Q2 — Does offensive or defensive quality carry over?

Season *t* → *t+1*, surviving teams only, n = 170 team-pairs:

| trait | slope | r | r² | reliability | disattenuated r |
|---|---|---|---|---|---|
| **attack** (npxG created) | 0.779 | **0.765** | 0.585 | 0.891 | **0.859** |
| **defence** (npxG conceded) | 0.664 | **0.656** | 0.430 | 0.826 | **0.794** |

**Attack carries over more than defence, and the gap is real.** The obvious objection is
that defence might merely be *measured* worse — so I split each team-season's matches in
half and computed each trait's reliability. Defence is indeed noisier (0.826 vs 0.891), but
correcting for it only narrows the gap from 0.109 to 0.065 rather than closing it.

Practical consequence: last season's attack numbers deserve more weight than last season's
defensive numbers. Clean-sheet projection is intrinsically the harder half of the problem,
which is worth knowing before over-trusting a defensive fixture run.

**Caveat:** measured on surviving teams only. Relegation censors the worst performers, so
these slopes understate persistence at the bottom of the table.

---

## Q3 — What types of teams do well in early fixtures?

Answered above: the only significant effect is mid-table (prior 7–12) **under**performing
early. The intuitive "top-six start fast" story does not clear the significance bar, though
the point estimate leans that way.

The genuinely large early-season structure is not a phase effect at all, it is the
**spread**: in GW1–6 the gap between prior top-6 and prior 7–12 attack is 1.325 vs 0.905 —
a ratio of 1.46, the widest of any phase. So the *ordering* is at its most reliable early
even though no individual archetype's own deviation is significant.

Home advantage is also archetype-dependent, which matters for a six-week fixture plan:

| archetype | away | home | home edge |
|---|---|---|---|
| prior top-6 | 1.138 | 1.407 | **+0.269** |
| prior 7–12 | 0.866 | 1.048 | +0.183 |
| prior 13–20 | 0.786 | 0.942 | +0.156 |
| promoted | 0.674 | 0.860 | +0.186 |

Top-six sides gain roughly **50% more from playing at home** than everyone else. A top-six
home fixture is worth materially more than the league-average home bump the model assumes.

---

## Q4 — Which opponents drive high and low scoring?

This is the fixture lookup. **xG created by row team against column opponent:**

| ↓team \ opp→ | prior top-6 | prior 7–12 | prior 13–20 | promoted |
|---|---|---|---|---|
| prior top-6 | 1.026 | 1.302 | 1.343 | **1.510** |
| prior 7–12 | 0.813 | 0.936 | 1.041 | 1.141 |
| prior 13–20 | 0.713 | 0.910 | 0.913 | 1.009 |
| promoted | **0.640** | 0.771 | 0.846 | 0.937 |

**Clean-sheet rate, row team against column opponent:**

| ↓team \ opp→ | prior top-6 | prior 7–12 | prior 13–20 | promoted |
|---|---|---|---|---|
| prior top-6 | 0.247 | 0.359 | 0.386 | **0.424** |
| prior 7–12 | 0.164 | 0.274 | 0.289 | 0.399 |
| prior 13–20 | 0.138 | 0.226 | 0.268 | 0.361 |
| promoted | **0.091** | 0.210 | 0.248 | 0.250 |

**Total match xG by opponent archetype** (both teams, season-relative):

| opponent archetype | total xG index | n |
|---|---|---|
| prior top-6 | **1.044** | 2508 |
| prior 7–12 | 0.988 | 2508 |
| promoted | 0.984 | 1254 |
| prior 13–20 | **0.971** | 2090 |

The counterintuitive one: **games against top-six are the highest-scoring**, not games
against promoted sides. Top-six teams concede little but create so much that the match
total rises. Promoted opponents concede heavily but create almost nothing (0.735–0.784), so
the totals roughly cancel to league average.

**Lowest-scoring environment is against prior 13–20** — bottom-half-but-established sides.
They neither create nor collapse. For attacking returns these are the fixtures to avoid,
and they are frequently mispriced as "easy" because the opponent is bad.

---

## Applying it to fixtures

- **Best attacking spot:** prior-top-6 attacker at home to a promoted side — 1.510 xG index
  before the +0.269 top-six home edge.
- **Best clean sheet:** prior-top-6 defence against a promoted side, 0.424 — more than four
  times the 0.091 a promoted defence manages against a top-six team.
- **Avoid for attackers:** anything against prior 13–20. Lowest total-xG environment in the
  league (0.971), and the "easy fixture" framing is usually wrong.
- **Avoid early:** prior 7–12 attackers in GW1–6, the one significant phase effect found.
- **Do not wait on promoted sides.** They get worse, not better.

---

## Model constants these replace

`history.py` implements the estimators but was never wired in; `bayes_model.py:59` still
calls the promoted priors "educated guesses". Validated first on simulated data with known
parameters (`studies/test_history.py`), where IV recovers `revert` to within 0.022 while
naive OLS is off by 0.158, then run on 31 real seasons:

| constant | current guess | measured (1993–2025) |
|---|---|---|
| `revert` | 0.85 | **0.963** |
| `season_sd` | 0.15 | 0.152 |
| `promoted_att` | (−0.20, 0.30) | (−0.209, **0.213**) |
| `promoted_def` | (−0.22, 0.30) | (−0.193, **0.182**) |
| `home_prior` | (0.26, 0.08) | (**0.184**, 0.086) |

Three of these are materially wrong:

- **`revert` 0.85 → 0.963.** Team strength carries over far more completely than assumed.
  The model is reverting teams toward the mean roughly four times harder than 31 seasons
  support, which flattens the spread of projected team strength every pre-season.
- **Promoted SDs 0.30 → ~0.20.** Promoted sides are *more* predictable than assumed, not
  less. The means were close; the uncertainty was overstated by about 50%.
- **`home_prior` 0.26 → 0.184**, declining −0.0059/season. Home advantage has fallen
  steadily since the 1990s and stepped down around the crowdless 2020/21 season. The model
  overstates it by ~40%.

### Applied 2026-08-11, with the A/B

Now live. `scripts/calibrate_team_history.py` writes `data/team_hyperparams.json`, which
`bayes_model` loads at import; `GUESSES` remains the fallback and the A/B baseline —
`FPL_TEAM_HYPER=guess` restores the old values without editing code.

**Team-strength spread widened, as predicted by `revert` 0.85 → 0.963:**

| | guess | calibrated | ratio |
|---|---|---|---|
| sd of projected attack | 0.1608 | 0.1782 | **1.11×** |
| sd of projected defence | 0.1681 | 0.1781 | 1.06× |

Direction is coherent: Man City attack +0.041, Arsenal +0.030, Man United +0.029;
Sunderland −0.028, Crystal Palace −0.022. Strong teams keep more of their strength, weak
teams more of their weakness.

**Board impact over GW1–10** — a redistribution, not a level shift:

```
mean delta   +0.012      (essentially zero — nothing inflates)
mean |delta|  0.224
max  |delta|  1.420      (Haaland 71.69 -> 73.11)
Spearman      0.9993     (ranking almost untouched)
```

By club: Man City +0.51, Arsenal +0.42, Man United +0.20, Chelsea +0.10; Crystal Palace
−0.17, Sunderland −0.16, Everton −0.14, Coventry −0.13, Tottenham −0.12. Tottenham falling
is the mechanism working, not a bug — their 25/26 was weak, and less reversion means they
stay weak.

**Why the player-level effect is smaller than a 0.85 → 0.963 change suggests.** Two
reasons, and both are worth knowing: the team layer is already anchored by betting-odds
strength (weight 0.6) and ClubElo (0.45), which absorb much of the change; and home
advantage moved the *other* way (0.26 → 0.184), partly offsetting the widening for home
fixtures.

**External check.** Against Solio's independent feed, agreement improves on all four
metrics:

| | Pearson | Spearman | MAE | bias |
|---|---|---|---|---|
| guess | 0.7412 | 0.5575 | 0.8401 | −0.677 |
| calibrated | **0.7534** | **0.5684** | **0.8076** | **−0.620** |

n = 29 matched players on a single gameweek, and Solio's feed is 25/26 GW35 against our
26/27 GW1, so this is directionally supportive, **not** decisive. Four correlated metrics
are not four independent tests.

**What is and is not established.** The constants are now measured from 31 seasons rather
than self-described guesses, the estimator was validated against known parameters on
simulated data, and the direction of every movement is coherent. Predictive superiority on
26/27 is **not** demonstrated and cannot be until the season provides outcomes. All four
acceptance tests pass unchanged.

---

## `[NULL]` Late-season surges do not carry into the next season

**Tested 2026-08-11 before committing the calibration.** `studies/late_form_carryover.py`,
187 team season-pairs over 11 transitions, surviving teams only.

The question is not "do teams that finish well start well" — of course they do, good teams
do both. The question is whether late form adds anything **beyond** full-season strength,
so the regressor is a residual: how far a team's last 6 matches ran above or below its own
season baseline.

```
early(t+1) ~ full_season(t) + late_resid(t)          n = 187

  full_season(t)   +0.9260   (se 0.0735, t +12.60)   <- significant
  late_resid(t)    -0.0019   (se 0.0816, t  -0.02)

  r2 with late form 0.4632
  r2 without        0.4632      late form buys +0.0000
```

**Exactly nothing.** Not a small effect — a zero, to four decimal places of r².

### The null is bounded, not merely asserted

A 6-match window is mostly noise, and errors-in-variables attenuates a coefficient toward
zero, so "no effect" is also what a broken test produces. Split-half within the window
puts its reliability at **0.483**, so the correction matters and was applied:

| | coefficient | 95% CI |
|---|---|---|
| raw | −0.0019 | (−0.162, +0.158) |
| disattenuated | −0.0039 | (−0.335, +0.327) |

Read as a null *with bounds*. The window is genuinely noisy, so what is excluded is any
effect larger than ~0.33 SD: a team finishing a full standard deviation above its own
baseline moves its next-season start by at most that, with a point estimate of zero. Small
effects are not ruled out. A tradeable one is.

### Managerial change — the hypothesis does not survive

The appealing version: a late surge under a *new* manager is a real regime change and
should persist, unlike a surge under the incumbent. The first cut looked supportive —
interaction −0.655 (se 0.331, t = −1.98). It does not hold up:

| definition of "new manager" | n | interaction | t |
|---|---|---|---|
| change 6–10 matches before end | 6 | −0.832 | −1.41 |
| change 6–12 | 8 | −0.831 | −1.72 |
| change 6–15 | 12 | −0.614 | −1.74 |
| change 6–20 | 22 | −0.115 | −0.38 |
| change 6–99 | 46 | **−0.009** | −0.05 |

The effect exists only at n ≤ 12 and evaporates the moment the sample is large enough to
mean anything. That is the signature of noise, and it arrived after several tests had
already been run on the same data. **Not supported.**

### A new manager does not reset the team either

| | n | carryover of full(t) |
|---|---|---|
| no close-season change | 152 | +0.912 (se 0.080) |
| close-season change | 35 | +1.154 (se 0.218) |

If anything *higher*, though the interval is wide. There is no evidence a summer
managerial change breaks the normal carryover of team strength — so `TeamModel`'s uniform
`revert` needs no manager-specific exception. That is a useful negative: it says the
simpler model is the right one.

### Incidental cross-check on `revert`

The `full_season(t)` coefficient here is **+0.926**, estimated from Understat xG over 12
seasons. The calibration wired in above gives `revert = 0.963`, estimated from
football-data **goals** over 31 seasons by an IV estimator. Two different data sources,
two different metrics, two different methods, ~0.04 apart — and both far from the 0.85
guess they replace. Independent corroboration of the largest constant change made here.

## `[CHECK]` Offseason managerial hires — first 3, 6 and 12 matches

`studies/new_manager_debut.py`. Managers appointed between the end of season *t* and
matchday 1 of season *t+1* — a full pre-season, no mid-season handover. 35 such cases
across 11 transitions.

**The selection problem is the whole difficulty.** Clubs change manager in the summer
*because* the previous season disappointed, and the data confirms it: prior-season
strength averages −0.036 for changers against +0.123 for everyone else. So "new-manager
clubs start below average" is guaranteed and meaningless. Every number below is a residual
from a baseline fitted **on non-changers only** — fitting it on everyone would let the
changers drag the line toward themselves and shrink their own residual.

| window | n | mean residual | 95% CI | beat expectation |
|---|---|---|---|---|
| first 3 | 35 | +0.081 | (−0.085, +0.251) | 51.4% |
| first 6 | 35 | +0.075 | (−0.058, +0.201) | 57.1% |
| first 12 | 35 | **+0.124** | **(+0.017, +0.232)** | 65.7% |

Only the 12-match window clears the bar, and only just.

### How much to believe the 12-match result

| check | result | reading |
|---|---|---|
| bootstrap CI | (+0.017, +0.232) | excludes zero, barely |
| exact sign test | 23/35, p = 0.089 | **does not** clear 0.05 |
| trimmed mean (drop 2 each tail) | +0.121 vs +0.124 | not outlier-driven |
| leave-one-season-out | +0.092 to +0.145 | very stable, no single transition carries it |

The bootstrap and the sign test disagree at this margin. The mean is stable and the
direction is consistent, but this is **suggestive, not established** — and three windows
were tested, so some multiplicity discount applies on top.

Practical size, since a per-match z unit is not intuitive: 1 z = 1.31 npxG difference per
match, so +0.124 z is **+0.163 npxG per match, ≈ +1.96 npxG across the twelve**. Roughly
two goals of expected difference over a third of a season — worth having if real.

**Unavoidable confound:** a summer managerial change normally arrives with squad
investment. This measures "new manager *and* whatever else the club did that summer", not
the manager. Nothing in this data can separate them.

### Trajectory — nothing early, then a step

| segment | gap vs prior strength |
|---|---|
| matches 1–3 | +0.059 |
| matches 4–6 | +0.053 |
| matches 7–12 | **+0.169** |

Flat for six, then a jump. Consistent with a bedding-in period before a new manager's
methods show up. Given the overall effect is only borderline, treat the split as
suggestive — but the shape matters for FPL, because **the board's GW1–6 horizon sits
entirely inside the flat part.**

### A new manager does not reset the team — confirmed a second way

| window | new-manager slope on prior strength | no-change slope |
|---|---|---|
| first 3 | +1.116 (se 0.269) | +0.812 (se 0.107) |
| first 6 | +1.142 (se 0.214) | +0.911 (se 0.079) |
| first 12 | +0.995 (se 0.170) | +0.840 (se 0.065) |

Prior-season strength predicts a new-manager club **at least as well** as it predicts
everyone else — the slopes are if anything steeper. This independently reproduces the
close-season result from the late-form study. The "new manager, blank slate" intuition is
wrong: whatever a new manager changes, he does not detach the club from what it was.

### What this means for the board

For a GW1–6 plan, price an offseason-hire club **at its prior-season strength, with no
adjustment**. There is no early bump to buy and no reset to fear. Any uplift arrives after
roughly six matches, outside the horizon the board projects, and is borderline even then.

(This does not contradict the existing regime work in `press_index`, which adjusts PPDA
*style* under a new manager rather than strength *level* — a different quantity.)

---

### Provenance caveat on the manager data

`data/manager_changes.csv` was transcribed from transfermarkt's per-season manager-change
pages via an LLM summarisation step, and **the transcribed manager names contain known
errors** (2014/15 lists Pardew → McClaren at Newcastle in January; McClaren arrived that
June). Only **club and date** are used — the names sit in a `note` column and feed nothing.
Matchdays are derived by joining dates to actual fixture dates rather than trusting the
source's matchday column, which is also wrong in places (2022/23 lists an April change as
"Matchday 8"). Spot-check before relying on this file for anything else.

---

## Bug found and fixed: season codes do not sort lexicographically

`history.py` ordered seasons with plain `sorted()` on codes like `9394`, `0001`, `2526`.
That puts `0001`–`2526` *before* `9394`–`9900`, so the 1990s land after the 2020s.

`estimate_promoted_prior` zips consecutive pairs off that order, so it compared 1993/94
against 2025/26 and labelled almost the whole 93/94 division "promoted";
`estimate_reversion` built its t−1/t/t+1 triples the same way. These feed exactly the
constants meant to replace the guesses.

Fixed with `season_start_year()` / `sort_seasons()`. Measured impact was smaller than the
severity suggests, because football-data's coverage gaps mean it only mis-pairs at the wrap
boundary — 96 spurious promoted team-seasons versus 88 real, attack mean −0.195 versus
−0.209 (~7%). Real bug, modest damage.

`studies/test_history.py` did not catch it: its simulated seasons carry sortable integer
labels, so lexicographic and chronological order coincide there. A test that never sees the
production key format cannot catch a key-format bug.

---

## Outstanding

1. ~~Wire the calibrated constants into `TeamModel`.~~ **Done 2026-08-11**, see above.
   Re-run `scripts/calibrate_team_history.py` when a season completes.
2. The archetype matrices are descriptive, not yet a model input. The natural use is as a
   prior//sanity check on `TeamModel`'s fixture λ, or as a residual check — do the model's
   own λ reproduce the 1.510 / 0.640 corners?
3. Extend the archetype work to `defcon_env`: the clean-sheet matrix is directly relevant
   to the DefCon channel, which currently conditions on PPDA only.
4. Promoted-side sample is 33 team-seasons of Understat. The football-data path reaches 88
   across 31 seasons but has no xG — worth pooling if the promoted prior gets revisited.
