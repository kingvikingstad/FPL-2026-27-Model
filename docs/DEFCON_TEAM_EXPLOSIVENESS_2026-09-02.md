# DefCon by team and matchup, and team explosiveness — 2 Sep 2026

*`studies/defcon_team_matchups.py`, `studies/team_explosiveness_study.py`,
`src/defcon_team.py`, `src/team_explosiveness.py`. Both pre-registered in their module
docstrings before any coefficient was read.*

**Headline.** One thing ships, three things are nulls, and the explosiveness result runs
the opposite way to the premise.

| question | verdict | key number |
|---|---|---|
| Which **opponents** concede DefCon? | **SHIPS** (DEF only) | hit-rate swing **0.129** across opponents, split-half r=+0.555 |
| Which opponents concede DefCon (MID)? | `[NULL]` | split-half r=+0.241, below the 0.5 gate |
| Is there a **matchup** (club × opponent) effect? | `[NULL]` | two meetings of the same pair correlate **r=−0.217** |
| Do **own teams** differ? | not identified | club and player are nested within a season |
| Are some teams more **explosive**? | `[NULL]` | dispersion reliability r=+0.006 vs null (−0.426, +0.480) |
| Do returns **concentrate** in one player? | `[NULL]` | r=+0.363 vs shuffled null (−0.415, +0.397) |
| Is the league tail Poisson? | `[VERIFIED]` **thinner** | P(4+) observed 4.08% vs implied 6.54%, 0.2nd pctile of null |

---

## 1. `[VERIFIED]` Opponent DefCon permissiveness — the one shippable signal

### Why re-open a settled null

`defcon_matchups` binned opponents into **strength quartiles** and found the DefCon rate
flat (0.339 / 0.328 / 0.416 / 0.329), concluding the supply hypothesis is wrong. That
conclusion stands. The inference drawn from it — "the opponent does not matter" — does
not follow: strength is a *scalar*, and "how much defending this opponent forces" is a
different quantity a strength bin cannot express. Two equally strong sides can differ
sharply.

They do. Estimated **within player** (every player faces many opponents inside a season,
so this is identified and is not a statement about who was on the pitch), on 2,978
defender-appearances of 60+ minutes:

| most permissive | shrunk actions | hit effect | | most suppressing | shrunk | hit |
|---|---|---|---|---|---|---|
| Bournemouth | +0.65 | +0.053 | | Fulham | −0.56 | −0.062 |
| Leeds | +0.45 | +0.026 | | West Ham | −0.53 | −0.055 |
| Liverpool | +0.41 | +0.055 | | Chelsea | −0.52 | −0.045 |
| Everton | +0.35 | +0.054 | | Wolves | −0.50 | −0.071 |
| Crystal Palace | +0.35 | +0.058 | | Burnley | −0.26 | −0.026 |

The ordering is **not** the strength ordering — Liverpool near the top, Chelsea near the
bottom — which is exactly why a strength quartile could not see it.

- **Split-half reliability r=+0.555** (Spearman-Brown 0.714), clearing the pre-registered
  0.5 gate.
- **Empirical-Bayes shrink factor 0.68**, computed independently of the reliability from
  between-opponent variance net of sampling variance. The two agreeing is a check that
  the noise model is right.
- **Size: 0.129 swing in hit probability = 0.26 DefCon points per match** between the
  extremes. Real, modest. For scale, the clean-sheet swing across fixture difficulty is
  ~0.87 points, so this is worth just under a third of that.

**Midfielders fail the gate** (r=+0.241). The ratings are computed and returned but
flagged `usable=False`, and nothing consumes them.

### It does not create a "tough fixture, DefCon floor" trade

corr(permissiveness, opponent attacking output) = **+0.264** for DEF. Permissive
opponents are *better* attacking sides, so DefCon and clean sheets still point the same
way. This reinforces the earlier study's conclusion rather than qualifying it: pick
defenders on fixture ease. There is still no compensation.

## 2. `[NULL]` Matchups are match-level shock, not tactical matchups

A club × opponent grouping "explains" 18.6% of defender DefCon variance. That number is
meaningless on its own: **380 cells on 2,978 rows is under 8 rows per cell**, and such a
grouping explains a large share of *any* variance. Every R² here is therefore reported
against a permutation baseline.

A methodological note worth recording: the obvious permutation — shuffling opponent
*labels* — is **wrong**. The fixture calendar makes club × opponent cells almost exactly
balanced, and shuffling labels destroys that balance, creating more small cells and a
*higher* baseline than the real design. It produced a baseline of 0.163 against an
observed 0.101, i.e. the null exceeded the data. Permuting the **outcome** holds the
design and every cell size fixed and destroys only the association being tested.

On that correct null, for defenders:

| term | R² | permuted null | excess |
|---|---|---|---|
| player identity | 0.219 | — | — |
| opponent | 0.020 | 0.007 [0.003, 0.011] | **+0.013** ✓ |
| club × opponent | 0.186 | 0.127 [0.110, 0.147] | +0.059 |
| interaction \| additive | 0.170 | 0.127 [0.111, 0.146] | clears its null |

The interaction clears its null — and is still not a matchup effect:

- **Match identity explains more** (R² 0.278) than club × opponent (0.170) on the same
  residual. A club × opponent cell is two matches, so the "interaction" is largely match
  identity in disguise.
- **The two meetings of the same pair disagree: r = −0.217** across 378 pairs (MID:
  −0.094). If the matchup were tactical it would repeat. It anti-repeats.

So the excess variance is real and is **match-level shock** — red cards, game state,
routs — which is unpredictable by construction. **D2 verdict: DROP for both positions.**

### A pre-registered statistic that had to be replaced

D2 as written compared the club × opponent excess against the opponent excess. That is
wrong: the club × opponent grouping *contains* the opponent main effect, so it is bound
to be larger and the comparison cannot isolate an interaction. It is replaced by a test
on additive-model residuals plus the two checks above, and the original is still printed
so the change is auditable. Fidelity to a pre-registered spec does not outrank
correctness.

## 3. Own-team DefCon is not identified `[JUDGMENT]`

Raw club means range 6.46 to 9.79 actions with split-half r=+0.701 — which looks like a
strong team effect and is not one. **Within a season a player belongs to exactly one
club**, so club and player are nested; demeaning by player annihilates the club term, and
the raw reliability is largely "does this club field the same players".

Separating them needs players who changed club with DefCon measured on both sides, i.e. a
second season. 26/27 currently has **2 gameweeks**. Recorded as a design limit, not a
null — it is untested, not tested-and-dead.

The structural route stays right: `defcon_env` conditions on a **mechanism** (team xGA
for the CBIT channel, press for CBIRT) rather than on a club label, and a mechanism can
be validated where a label cannot.

## 4. `[NULL]` Teams are not differentially explosive

The engine draws `rng.poisson(lam)`, which pins variance to the mean. If teams differed
in that second moment, `captaincy.p_haul` and `ceiling` would be wrong for the extremes —
and captaincy ranks on exactly the tail. Worth asking; the answer is no.

**Two traps, both of which flipped a result:**

1. **In-sample λ.** 41 attack/defence/home parameters on 760 team-matches overfits, and
   the fitted λ chases the data, *deflating* residual dispersion. The naive in-sample
   Pearson dispersion is **0.887** — which reads as strong under-dispersion and is an
   artefact. Cross-fitted on season halves it is **1.056**, essentially Poisson. The
   per-club spread is the same illusion: 0.63–1.40 in sample, 0.65–2.28 cross-fitted.
2. **Convexity.** P(G≥4) is convex in λ, so a *noisy* λ̂ inflates the implied tail by
   Jensen and manufactures a thin-tail finding from estimation error alone. A z-test
   against a fixed λ cannot see this.

Per-club dispersion split-half **r=+0.006** against a simulated true-Poisson null of
+0.012, 95% band (−0.426, +0.480). Entirely noise — with 38 matches a club's dispersion
is barely estimated at all.

Return **concentration** (Herfindahl of goal+assist shares, "do returns bunch into one
player") gives r=+0.363 against a club-shuffled null band of (−0.415, +0.397). Borderline,
inside the band, **not established**. Per-club HHI spans only 0.343 (Man Utd) to 0.426
(Spurs).

## 5. `[VERIFIED]` The upper tail is THINNER than Poisson — the opposite finding

Because λ must be estimated, the null is **simulated**: generate from a true Poisson with
the fitted λs and push it through the identical cross-fit pipeline.

| k | observed − implied | null mean | null 95% | percentile |
|---|---|---|---|---|
| 3 | −0.0031 | −0.0058 | (−0.0210, +0.0097) | 64.5 |
| **4** | **−0.0247** | −0.0077 | (−0.0185, +0.0047) | **0.2** ✓ |
| **5** | **−0.0144** | −0.0055 | (−0.0133, +0.0029) | **1.5** ✓ |

The null's own mean gap is negative — *that number is the convexity bias, made visible*.
The observed gap has to beat it, not zero. It does.

Measured calibration (observed / Poisson-implied):

| goals | 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| ratio | 0.904 | 1.010 | 1.127 | 1.203 | **0.623** |

**The league produces about a third fewer 4+ goal team-games than the engine's Poisson
draw implies**, and mass piles up at 2–3 instead. The engine *overstates blowouts* — and
those are precisely the fixtures a captaincy pick concentrates on, so the error lands
where it is most expensive.

### The market guard does not apply here, and saying so matters

`style_matchup.beats_the_market()` gates new team signals with a Poisson **score test on
the mean**. It is structurally blind to a second-moment claim: a distribution can match
the market's mean exactly and still have the wrong tail. Invoking it here would be a
category error. The relevant validation is tail calibration against out-of-sample
scorelines, which is what was done.

## 6. What shipped, and what deliberately did not

| artefact | status |
|---|---|
| `src/defcon_team.py` | **ships.** Opponent DefCon ratings, EB-shrunk, reliability-gated |
| `defcon_opponent_category` in `team_projections_season.csv` | **ships.** Tercile category per club — the "team trend" category |
| `opp_defcon` metric in the Gameweek Explorer | **ships.** Per-fixture, defender rows only |
| "DefCon" view on the Explorer's Fixture outlook tab | **ships.** Per club, whole-run |
| "DefCon schedule" axis + "DefCon conceded" KPI on Team trends | **ships.** The two directions, labelled apart |
| `src/team_explosiveness.py` | measurement ships; `apply_tail_calibration` is **OFF BY DEFAULT** |
| per-team explosiveness term | **not built** — E1 and E3 are nulls |
| club × opponent matchup term | **not built** — match-level shock |
| own-team DefCon term | **not built** — not identified |

`apply_tail_calibration` is off by default because changing the goal distribution moves
the clean-sheet engine validated at GA r=0.89 / CS r=0.93. Wiring it in needs its own
revalidation against scored gameweeks — which is open item 6 in PROJECT_KNOWLEDGE, and
nothing in this repo has yet been scored against a real gameweek.

### A join bug caught on the way

25/26 `teams.csv` names two clubs "Man Utd" and "Spurs"; the 26/27 frame the board is
built on says "Man United" and "Tottenham". Joining the ratings without normalising drops
exactly those two clubs to NaN, which reads as "no rating available" rather than as a bug.
Everything now routes through `core_insights.norm_team`. After the fix all 17 returning
clubs carry a rating and only the three promoted sides are NaN, which is correct.

---

## In the Gameweek Explorer

`gw_explorer` carries the rating as the per-gameweek metric **`opp_defcon`** ("Opponent
DefCon (DEF)"), joined on the **opponent** — the rating is *actions conceded to the
opposition*, so the number on a player's row is the club he faces, not his own. Joining it
on `team` would invert the meaning and still produce a full, plausible column, so the
direction is asserted in the export test rather than trusted.

Three deliberate restrictions:

- **Defender rows only.** The midfielder rating failed its reliability gate and is not
  shipped; putting the defender number on a MID row would assert something the study
  explicitly declined to. Goalkeepers cannot score DefCon at all. Blanked in
  `assemble()` rather than in the view, so the CSV export carries the same restriction.
- **Promoted clubs are null, never neutral.** Coventry, Ipswich and Hull have no 25/26
  rating. As an opponent they render blank.
- **Not additive.** It is a probability *deviation*, not an expected count, so its sum
  over a window has no referent. It is kept out of `ADDITIVE`, and the view aggregates by
  mean. It is also kept out of `INVERT`: a more permissive opponent is *better* for the
  defender holding the row, which is the default scale direction.

The Players tab needed no view change — it is driven generically off `D.metrics`, so the
picker, the window columns and the CSV export all picked it up. A consistency check that
fell out of the verification: Gabriel shows 32 values over 38 gameweeks, which is exactly
38 minus the six fixtures against the three unrated promoted clubs.

### The Fixture outlook tab — a fourth "DefCon" view

The fixtures tab is not metric-driven, so this one is a real change: a fourth `fxView`
alongside Attackers / Defenders / Overall, answering "whose defenders have the kindest
DefCon run over this window".

**The relative basis does not apply, and the view says so.** `rel` exists because raw
lambda confounds how good a club is with how kind its run is — Manchester City top an
absolute lambda table in every window ever selected, which is a fact about their attack.
`opp_defcon` carries **no club component at all**: it is a property of the opponent, so a
club's own 38-fixture mean is already zero up to schedule imbalance. `fxBase` therefore
returns exactly 0 for this view, which makes the two bases coincide rather than differ by
that imbalance noise, and the note under the table explains it instead of leaving a
control that looks like it does something.

**A trap that would have produced a half-empty grid.** `_team_gw` takes one arbitrary row
per (team, gameweek) via `drop_duplicates`. The player-facing `opp_defcon` is blanked for
non-defenders, so reading it there would have returned a value or a null depending on
whether that club's first row that week happened to be a defender — an order-dependent
bug. The unblanked value is carried separately as `opp_defcon_fx` and emitted to the view
under the plain name. Verified after the fact: every established club shows 32 non-null
fixtures (38 minus the six against unrated promoted sides) and each promoted club shows
34 (38 minus the four against the other two).

Verified in the browser: all four views render, the three lambda views keep their own
notes and KPI labels, the absolute/relative toggle still moves them (+2.19 vs +0.21 on
Attackers), and spot-checks match the ratings table — Newcastle vs Bournemouth at home
reads 0.05 against Bournemouth's +0.053, Chelsea away at Arsenal reads −0.01 against
Arsenal's −0.011, and Hull renders blank.

### The Team trends tab — two directions, deliberately labelled apart

The same rating answers two different questions for a club, and reading one for the other
inverts the advice. Both are on the tab, named so they cannot be confused:

| where | what it means | good news for |
|---|---|---|
| **"DefCon schedule"** profile axis | mean permissiveness of the OPPONENTS this club faces over the window | owning **this club's** defenders |
| **"DefCon conceded"** KPI | how much this club hands to the OPPOSITION's defenders — its own rating and tercile | owning the **other side's** defenders |

The axis reuses the existing window-mean plumbing (`agg="mean"` over the fixture run, the
same path as Fixture attack and Fixture safety). The KPI comes from `teams_meta`, so
`team_season()`'s column whitelist gained the two DefCon columns.

Two defects fixed on the way:

- **`_teams_meta` would have crashed.** It coerced every column with `float()`, so the
  first categorical column — the DefCon tercile — would have raised `ValueError` and taken
  the whole team dashboard down. Strings now pass through as strings.
- **`sgn` is scoped to `renderFixtures`.** Reusing it in `renderTeams` would have thrown a
  `ReferenceError` and broken the tab. Caught before it shipped; the KPI formats inline.
- **The profile caption was already wrong.** It read "the last two are this club's fixture
  run" when there were three window axes, and would have drifted further with each one
  added. It is now counted from `team_components` rather than written down, and reads
  "the first 5 … the last 4".

Verified in the browser across four clubs: Arsenal *Neutral, −0.011*; Fulham
*Suppressing, −0.062*; Bournemouth *Permissive, +0.053*; Coventry *— , no 25/26 rating*.
Radar and bars both render with nine axes, and there are no console errors.

## How to use the category

`defcon_opponent_category` ∈ {permissive, neutral, suppressing} answers "does this fixture
help my cheap defender hit 10 actions". It is a **tiebreaker, not a driver** — worth ~0.25
points per match between the extremes against ~0.87 for the clean-sheet fixture swing, and
the two point the same way. Use it to separate two defenders whose fixtures are otherwise
equal; do not use it to justify a hard fixture.
