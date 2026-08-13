# Set-Piece Duty, DefCon Matchups, and Early Dispersion

**Opened:** 2026-08-11 · Three requests: reflect projected set-piece duty in player values,
push the early-season and matchup findings into projections, and analyse when defenders
actually earn DefCon.

---

## 1. `[APPLIED]` Projected set-piece duty

`data/set_piece_takers.csv` + `src/set_piece_takers.py`. Source: Fantasy Football Scout's
projected 26/27 set-piece takers, all 20 clubs, penalties + direct free kicks + corners.

This closes the coverage gap measured in `penalty_assignment.py`: FPL's own
`penalties_order` is the more *precise* signal (85.7% precision against 45.0% for measured
history) but covers only ~39% of penalties actually taken, because several clubs carry no
declared taker at all.

### Name resolution — 126/130, and the four failures are correct

Matching is **within club** against the actual 26/27 squad, so a wrong name cannot attach
to a player elsewhere, and anything unresolved is dropped rather than guessed.

Three rounds of fixes were needed, all real:
- **`Ø` and `ß` do not decompose under NFKD** — they are distinct letters, not
  letter+diacritic. Without explicit folding, `Ødegaard` and `Groß` never matched.
- **`.` had to tokenise as a space** for `N.Williams`, `B.Fernandes`, `Kroupi.Jr`.
- **The source carries fuller names than FPL** — "Amad Diallo" against FPL's `Amad`.

The four that remain unresolved are **genuine**: Pino (Crystal Palace) and Simons
(Tottenham, three roles) are not in the 26/27 squad data at all. The projection lists
players the FPL dataset does not have.

**First-choice penalty takers now resolve for 20/20 clubs.**

### The two sources disagree on a quarter of clubs

| | clubs |
|---|---|
| agree | 14 |
| **disagree** | **5** |
| FPL blank | 1 (Chelsea) |

| club | FPL #1 | FFS #1 |
|---|---|---|
| Bournemouth | Kroupi.Jr | **Kluivert** |
| Fulham | Robinson | **Muniz** |
| Hull | Crooks | **McBurnie** |
| Ipswich | Palmer, Hirst | **Clarke** |
| Liverpool | Isak | **Szoboszlai** |

(Ipswich flags *two* players as #1 in the FPL data, which is itself a defect.)

### Precedence: the projection wins, and that is a choice

`mode="override"` is the default. For a **pre-season** board, FPL's `penalties_order` is
largely carried over from last season while the projection is made for the season being
modelled. `FPL_SETPIECE=fill` restores FPL precedence (projection fills blanks only) and
`=off` disables it. Every reassignment is logged, never silent.

### Board effect

Mean +0.006 over GW1–10 — pure reassignment, not inflation — with 18 players moving more
than 0.5:

| gained | Δ | lost | Δ |
|---|---|---|---|
| Szoboszlai | **+4.91** | Robinson | **−4.51** |
| Kluivert | +3.06 | Hirst | −2.04 |
| McBurnie | +2.20 | Kroupi.Jr | −0.67 |
| Clarke | +2.16 | Doku | −0.58 |
| Muniz | +2.08 | Livramento | −0.57 |

Roughly 4–5 points over ten gameweeks per taker, which matches the ~16 points a season
per settled taker computed earlier from first principles.

**Caveat:** this is a projection from a web page read through an LLM, and the five
disagreements above are exactly where it matters. They are worth eyeballing before the
season starts — particularly Liverpool, where Isak → Szoboszlai is a 4.9-point swing on
one editorial judgement.

---

## 2. `[CHECK]` DefCon — when do defenders actually hit the threshold?

`studies/defcon_matchups.py`. 2,934 defender-appearances of 60+ minutes, 25/26 only.

> **A bug caught mid-analysis, worth recording.** `defensive_contributions` is **100% null
> in 24/25** — the stat did not exist before FPL introduced DefCon scoring in 25/26. The
> first run filled those nulls with zero, which made half the sample look like defenders
> who never touch the ball and produced a confident, non-monotone opponent-strength curve
> that looked like a finding. The study now drops nulls and prints per-season coverage.

### The supply hypothesis is wrong

The intuition — you cannot tackle a team that never has the ball, so DefCon should be
easier against strong opponents — **does not hold**:

| opponent | DefCon rate | mean actions | clean-sheet rate |
|---|---|---|---|
| weakest | 0.339 | 8.18 | 0.344 |
| Q2 | 0.328 | 8.05 | 0.308 |
| Q3 | **0.416** | 8.86 | 0.240 |
| strongest | 0.329 | 8.22 | **0.127** |

DefCon is **flat** across opponent strength (the Q3 bump is the odd one out), while clean
sheets collapse from 0.344 to 0.127.

### So the exchange rate is decisive — and one-sided

| opponent | DefCon pts | CS pts | total |
|---|---|---|---|
| weakest | 0.678 | 1.374 | **2.052** |
| Q2 | 0.656 | 1.231 | 1.888 |
| Q3 | 0.831 | 0.961 | 1.793 |
| strongest | 0.659 | 0.509 | **1.168** |

**There is no compensation.** Because DefCon volume does not rise against stronger
opponents, a hard fixture loses clean-sheet value and gains nothing back. Total defender EV
falls monotonically, 2.05 → 1.17. The "play a cheap defender into a tough fixture for the
DefCon floor" idea is not supported — pick defenders on fixture *ease*, exactly as for
clean sheets.

Own-team strength points the same way: DefCon roughly flat (0.371 → 0.310 from weakest to
strongest team) while clean sheets climb 0.200 → 0.361.

**This validates a current design choice:** `project()` scales DefCon by minutes but not by
opponent, and since the rate is flat in opponent strength, that is correct.

### Centre-backs vs full-backs — the largest effect found

Defenders split on their own action profile (headed clearances and aerial duels mark a CB,
crosses mark a FB) rather than routing through the unverified Understat crosswalk. The
split validates: CBs average 3.09 headed clearances/90 against 1.39 for FBs, and 0.07
crosses against 0.57.

| role | n | DefCon rate | mean actions | clean-sheet rate |
|---|---|---|---|---|
| **CB** | 1618 | **0.480** | 9.63 | 0.260 |
| **FB** | 1316 | **0.207** | 6.81 | 0.264 |

**Centre-backs hit the DefCon threshold 2.3× as often as full-backs** — +0.273, 95% CI
(+0.239, +0.308), clustered by player — while their clean-sheet value is *identical*
(0.260 vs 0.264). The gap holds at every level of opponent strength.

The mechanism is the threshold, not effort: 9.63 actions against 6.81 sits right on the
10-action requirement, so the same margin converts far more often for a CB.

**Implication for the model.** `to_priors` uses a single `PRIOR_DC` of 7.6 for all
defenders. For players with history this washes out — their own `defcon_alpha/beta`
dominates — but for **cold-start defenders the prior is the whole estimate**, and 7.6 is
too low for a centre-back and too high for a full-back. That is precisely the population
the cheap-defender board surfaces (promoted-club defenders at £4.0–4.5m).

**Not fixed**, because it is circular: classifying a cold-start defender as CB or FB
requires the action history they do not have. It needs a positional source — the FPL squad
list does not distinguish them.

---

## 3. `[MEASURED, NOT APPLIED]` Early-season dispersion

`studies/early_dispersion.py`. Does the model under-disperse strength in the opening weeks?

The archetype work found mismatch driving goals ~3× harder in GW1–6 (+0.185 vs +0.062 per
rank-gap). That used four coarse buckets, so it could be an artefact of where the edges
fall. Repeating it on the **continuous** scale the model actually uses:

```
log-xG slope on strength difference   GW1-6  0.189    GW7+  0.096
difference +0.093, 95% CI (-0.091, +0.262)   <- overlaps zero
```

Underpowered — r² ≈ 0.005, because a single match's log-xG is mostly noise. So the
better-specified version asks the question the model's own form asks: build the
multiplicative expectation `A × D_opp / league` and test whether its **residual** still
depends on the strength difference. If the form is right, the slope is zero.

| window | n | residual slope | 95% CI |
|---|---|---|---|
| GW1–6 | 1440 | **+0.101** | (+0.002, +0.206) |
| GW7+ | 7680 | −0.023 | (−0.056, +0.009) |
| **difference** | | **+0.127** | **(+0.018, +0.235)** |

**GW7+ is cleanly zero**, which is a strong validation that the multiplicative form is
right in general — and makes the early departure harder to dismiss as specification error.

### Why it was not applied

Three reasons, and they run against my own inclination to ship it:

1. **The GW1–6 interval barely clears zero** (+0.002 lower bound).
2. **This is the third test of the same question** — rank-gap, log-slope, residual. The
   first two did not clear; multiplicity is real.
3. **The effect is small.** A slope of 0.10 on a strength difference of ~0.6 (top-6 against
   promoted, about the widest fixture there is) is ~0.06 xG, roughly 4% of a team-match.

This session has already retired several early-season effects that looked real at this
significance level and dissolved on further testing. The consistent standard says wait.
It is the **top candidate for the next validation round** — the way to settle it is a
proper out-of-sample backtest of board accuracy with and without a GW1–6 amplifier, not a
fourth in-sample slope.

---

## What changed in the model

| change | status | escape hatch |
|---|---|---|
| Projected set-piece duty (20/20 clubs) | **applied** | `FPL_SETPIECE=fill\|off` |
| DefCon opponent adjustment | **not needed** — rate is flat in opponent strength | — |
| CB/FB DefCon prior split | **blocked** — needs a positional source | — |
| GW1–6 dispersion amplifier | **measured, not applied** | — |
