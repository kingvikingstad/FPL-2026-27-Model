# Early-season scoring trends — four hypotheses, four nulls

**Study:** `studies/early_scoring_trends.py` · **Date:** 20 Aug 2026
**Outputs:** `studies/early_scoring_trends.csv`, `studies/early_scoring_panel.csv`

Tests whether four club characteristics predict **early-season scoring totals**. All four
had prior findings in this repo, but against *xG strength residuals* or *league points* —
none against goals or match totals, which is the estimand asked for here.

---

## 1. Design, pre-committed before results

| | |
|---|---|
| **Primary outcome** | match total goals residual, early window **minus** rest of season |
| **Secondary** | team goals-scored residual (support only, not counted in the correction) |
| **Early window** | first 6 matches per club |
| **Panel** | 31 seasons (1993/94–2025/26), 23,888 club-matches, 624 club-seasons |
| **Alpha** | 0.05 / 4 = **0.0125**, two-sided, fixed before any result was seen |
| **Confirmed requires** | p < 0.0125 on primary **AND** same sign on secondary **AND** leave-one-season-out sign stability |
| **SEs** | clustered on season |

Three design choices carry most of the weight:

- **Within-club differencing.** Good clubs score more all season, so a cross-sectional
  comparison is mostly club quality. Every outcome is early *minus* that same club's
  rest-of-season in the same season.
- **Opponent- and venue-adjusted residuals.** Each match is scored against a multiplicative
  season model fitted within season, so "they had easy early fixtures" cannot masquerade as
  an effect — the confound `midtable_fade` had to rule out separately.
- **H1 is a continuous rank, not four archetype buckets.** Bucketing is what generated the
  `midtable_fade` false positive (p=0.039 against the 0.0125 four tests require). A slope is
  one test with more power.

---

## 2. `[VERIFIED]` Positive control — the panel *can* find a real effect

Four nulls mean nothing if the machinery is dead. The control is early-season home
advantage, which `early_season_goals.py` measured at −0.152 on 12 Understat seasons and
which the model **already applies** in `bayes_model._home_effect`.

| segment | home advantage | vs md13–38 |
|---|---|---|
| md1–3 | 0.1989 | **−0.1717** |
| md4–6 | 0.3332 | −0.0374 |
| md7–12 | 0.4284 | +0.0578 |
| md13–38 | 0.3706 | — |

Paired across 31 seasons: **−0.1728 goals, CI (−0.297, −0.049), p = 0.0105, 22 of 31
seasons down.**

This independently replicates the applied −0.152 discount on a panel 2.5× longer and built
from raw results rather than xG. It also confirms the effect is confined to md1–3 and gone
by md4–6, which is exactly why the model applies one step rather than a schedule.

The synthetic selftest closes the other half: pure noise produces no false positive, and a
planted 0.05 effect **is** recovered at the pre-committed alpha.

---

## 3. Results — all four null

| | n | seasons | coef per SD | 95% CI (per unit) | p | verdict |
|---|---|---|---|---|---|---|
| **H1** prior-season final position | 508 | 30 | −0.0444 | (−0.0225, +0.0048) | 0.196 | null |
| **H2** prior-season last-10 form | 508 | 30 | −0.0154 | (−0.263, +0.174) | 0.681 | null |
| **H3** offseason manager change | 240 | 12 | −0.0104 | (−0.306, +0.249) | 0.826 | null |
| **H4** squad turnover | 240 | 12 | +0.0554 | (−0.0037, +0.0116) | 0.278 | null |

Nothing approaches 0.0125. Nothing would survive even an uncorrected 0.05.

**Consistency with prior work.** H1 null on goals agrees with `midtable_fade` closing as a
probable false positive. H2 agrees with `late_form_carryover` (+0.0000 r²). H3 agrees with
`new_manager_debut` (no bump inside GW1–6). H4 agrees with `transfer_churn` (not
front-loaded). Four independent methods, different outcome variable, same answer — that is
about as good as null replication gets.

**H4 is the only one worth a second glance.** It is the sole positive coefficient, and the
secondary outcome reaches p=0.083. Direction: *more* churn, *more* early goals — the
opposite of the disruption story. It does not clear the bar and should not be built on, but
it is the one to re-check when power improves.

---

## 4. `[VERIFIED]` What the nulls do and do not license

The MDEs are the whole story:

| hypothesis | MDE (goals/match per SD, 80% power @ α=0.0125) |
|---|---|
| H1 | 0.1120 |
| H2 | 0.1235 |
| H3 | 0.1540 |
| H4 | 0.1624 |

A typical match total is ~2.7 goals, so these rule out effects larger than **4–6% of a match
total**. That is a real result: no *large* early-season scoring effect exists for any of the
four. Anything of that size would be trivially exploitable and it is not there.

They do **not** rule out effects around 0.03–0.05 goals/match. That range still matters for
FPL — 0.05 goals/match over six gameweeks is ~0.3 goals per club, roughly 1–2 points spread
across a squad — and it sits entirely inside the confidence intervals. **These are
"not detectable", not "zero".**

---

## 5. New information required — and what it would buy

Sample size is the binding constraint, and on the current outcome it cannot be fixed:

> To reach MDE 0.05 on goals needs **127 seasons**. To reach 0.03 needs 354. There are 31.

So the lever is a **lower-variance outcome**, not more history. Measured on identical
matches (12 Understat seasons, 9,120 team-matches):

| | goals | xG | ratio |
|---|---|---|---|
| per-match residual sd | 1.1160 | 0.7141 | **0.640** |
| club-season outcome sd | 0.7215 | 0.4203 | **0.583** |

xG carries ~58% of the noise of goals on the same matches. That is enough that **xG on 12
seasons beats goals on 31**: MDE 0.0949 versus 0.1120. Switching outcome is worth more than
19 extra seasons.

Still short of 0.05. The step that closes it is **pooling the other Understat leagues**,
which multiplies club-seasons without needing new history to exist:

| panel | n | MDE |
|---|---|---|
| EPL only, xG | 240 | 0.0949 |
| + La Liga | 480 | 0.0671 |
| + Bundesliga | 720 | 0.0548 |
| + Serie A | 960 | **0.0475** |
| + Ligue 1 | 1200 | **0.0425** |

### The concrete ask

1. **Scrape Understat team-match xG for La Liga, Bundesliga, Serie A and Ligue 1, 2014/15
   to 2025/26.** `sd_ingest.py` already handles Understat; the readers take a league
   argument. This is the single highest-value data acquisition for this question — it takes
   MDE from 0.112 to ~0.043 and moves all four hypotheses into the range that matters.
2. **Manager changes and transfer counts for those leagues.** H3 and H4 are capped at 12
   seasons *and* one league; without this they stay at n=240 while H1/H2 scale.
3. **Squad-value or minutes-weighted turnover, not transaction counts.** `transfer_counts`
   counts a loaned-out youth player the same as a first-choice striker sold. `transfer_churn`
   already flagged this; minutes-weighted departure share is the measure H4 actually wants,
   and H4 is the hypothesis closest to signal.

### The assumption that ask carries

Pooling leagues assumes the effect is **common across leagues**. That is testable — fit the
league × hypothesis interaction — and it must be tested rather than assumed, because if
effects differ by league the pooled estimate is a weighted average of different things. Test
the interaction first; only pool if it does not reject.

---

## 6. Recommendation

**No model change.** No early-season adjustment for prior position, late form, new manager or
squad turnover is warranted on this evidence, and the four prior studies already reached the
same conclusion by other routes.

The one applied early-season effect — the GW1–3 home discount — is **reconfirmed and
strengthened** by this study: −0.173 on 31 seasons against the −0.152 currently applied, with
the effect correctly confined to md1–3.
