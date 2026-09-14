# Travel distance, holding team quality constant — 10 Sep 2026

*`studies/travel_distance.py`. Pre-registered in the module docstring. An outcome-blind
design pass (identification and power, reading no score) ran before the outcome pass.
Model hook: `src/travel.py`. Shipped off by default on 10 Sep. **On by default since
11 Sep 2026, by explicit owner override of the market gate** (§6); `FPL_TRAVEL=off` disables
it.*

**Headline: the trip matters for the traveller's defence, not its attack, and the market
already prices most of it.** A longer trip raises the goals the away side concedes by
**+0.032 per log-km** (z = +4.6, clustered on the club pair). That is **+8.8% home goals**
from the 10th- to the 90th-percentile trip (23 km → 316 km). Seen from the other end,
a derby cuts home advantage roughly in half. The same trip does **nothing measurable to
the goals the traveller scores** (null). The effect failed the market-orthogonality gate,
so under the pre-registered rule it shipped **off by default**. §6 put the guard's scope to
the owner, who **switched it on** on 11 Sep. That is a scoped exception to the gate,
recorded in CLAUDE.md, not a relaxation of it.

---

## 1. What was tested, and why it is not a revived null `[JUDGMENT]`

Only the away side travels, so distance is not a team attribute. It is a **modifier of
home advantage**, fixture by fixture. `bayes_model` carries one constant home
parameter, which averages derbies with 300 km trips and misprices both.

Congestion is dead three ways (`FIXTURE_CONGESTION_2026-09-01.md`), but it was tested as
**recovery time**. Distance is a different instrument. The mechanism is left open:
travel fatigue, away-support share and derby familiarity all load on it and cannot be
separated. For forecasting that does not matter.

Regressor: `z = log(max(km, 5))`, great-circle distance between the two grounds **on the
match date**. `src/travel.py` records the dated moves: Spurs at Wembley 2017-19, Everton
to Hill Dickinson in 2025, Fulham at Loftus Road in 2002/03, West Ham, Man City,
Southampton, Sunderland, Derby, Bolton, Leicester and Coventry. It also records the
Palace/Wimbledon groundshare (0 km). Log, because the contrast the league actually
contains is derbies against everything else.

## 2. Two ways to hold team quality constant

| | holds constant | sample |
|---|---|---|
| **E1 structural** | attack AND defence fixed effects per **team-season**, home advantage per season | 1993/94–2025/26, 31 seasons, **11,944 matches**, 925 club pairs |
| **E2 market** | pre-match consensus 1X2 + O/U 2.5 → (λ_H, λ_A) via `betting_odds_ingest.implied_lambdas` | 2005/06–2025/26, 21 seasons, **7,980 matches** |

E1 asks whether the effect is real. E2 asks whether it is already in the price, which is
the CLAUDE.md market gate. It is computed by `style_matchup.market_score_test`, split out
of `beats_the_market` today so a non-style covariate runs through the one implementation
of the gate. `beats_the_market` behaviour is unchanged and its selftest passes.

**Standard errors are clustered on the unordered club pair.** The regressor is fixed at
pair level. Match-level SEs would count every repeat of Arsenal–Spurs as new
information about distance.

## 3. Design pass (outcome-blind) `[VERIFIED]`

- **R1 identified.** Through the repo's own gate, `check_identification`, run season by
  season (the FE block is block-diagonal by season, which is what makes the projector
  affordable at n = 23,888), 7.4% of each column's variance survives the team-season FE,
  against a threshold of 5%. That figure understates identification. Each column is zero
  on the other side's rows, so most of its variance is the home/away level the home dummy
  absorbs by construction. **91.6% of the within-side variance of z survives.** Club
  fixed effects absorb only a club's average trip; the rest is who-plays-whom-where.
- **Power.** 95% half-width on the P10→P90 scale: **±0.040** (home goals) and **±0.045**
  (away goals). MDE at 80%/|z| > 2.5: 0.068 and 0.076. The materiality floor (0.03) sits
  below the MDE, so the rule could only pass an effect roughly twice that size.
- **Estimator validated by simulation** (40 synthetic leagues, true b_H = +0.05,
  b_A = −0.07). Mean estimates +0.052 and −0.066. Clustered SEs are slightly
  conservative (0.027 vs empirical 0.024). Size under the null is 5% / 2.5% at 1.96.
  Poisson FE has no incidental-parameter bias.

## 4. Results `[VERIFIED]`

### E1 — structural

| | b per log-km | SE (pair-clustered) | z | P10→P90 |
|---|---|---|---|---|
| **trip → home goals (traveller GA)**, 1993–2026 | **+0.0323** | 0.0070 | **+4.62** | **+0.084** |
| trip → away goals (traveller GF), 1993–2026 | −0.0096 | 0.0084 | −1.14 | −0.025 |
| traveller GA, 2016–2026 only | +0.0315 | 0.0111 | +2.84 | +0.082 |
| traveller GF, 2016–2026 only | +0.0128 | 0.0144 | +0.89 | +0.033 |

The goals-against effect is **the same size in the last decade as over 31 seasons**, even
though home advantage itself has roughly halved over that span. Robustness (Bonferroni
over 4, no decision rule):

| cut | traveller GA | traveller GF |
|---|---|---|
| linear, per 100 km | +0.0313, z = +4.65 | −0.0121, z = −1.46 |
| holiday fixtures dropped (24 Dec–3 Jan, regionalised on purpose) | +0.0316, z = +4.57 | −0.0050, z = −0.56 |

### E2 — over the market

| | score z (repo gate) | GLM b (free intercept + slope on log λ) | GLM z |
|---|---|---|---|
| home goals (traveller GA) | **+1.73** | +0.0152 ± 0.0087 | +1.75 |
| away goals (traveller GF) | −0.37 | −0.0037 ± 0.0099 | −0.37 |
| closing odds, 2019/20+ (n = 2,660), GA | +1.01 | | +1.21 |

The point estimate over the market is **about half** the structural one. The interval for
the unpriced part, (−0.002, +0.032), runs from "fully priced" to "not priced at all". The
market is not shown to miss it, and not shown to price all of it.

### Descriptive — behind closed doors

472 matches with no crowd (2019/20 from 17 Jun 2020, and 2020/21). Interaction on the
goals-against slope: −0.021, 95% (−0.106, +0.063). The interval is ±0.08 against an
effect of 0.032, so this **cannot separate a crowd mechanism from a travel one**.
Reported as a CI, as pre-registered, with no reading taken from it.

## 5. Decision

| | R1 identified | R2 real & material | R3 beats market | R4 still true | verdict |
|---|---|---|---|---|---|
| **traveller GA** | ✓ | ✓ z +4.62, +0.084 | ✗ +1.73 / +1.75 | ✓ recent +0.0315, z_diff +0.09 | **OFF BY DEFAULT** (pre-registered) → **ON** by owner override, 11 Sep |
| traveller GF | ✓ | ✗ | ✗ | ✗ sign flips | **NULL** |

**Traveller GF is a null.** No code path models it; `travel.fixture_shift` moves the
home side's λ only.

**Traveller GA ships as `src/travel.py`** (off by default on 10 Sep; on since 11 Sep, §6), hooked into
`bayes_model._home_effect`:

- `+ b·(z − z̄)` on the home side's log-λ, b = 0.0323.
- **Centred on the 25/26 league's mean trip** (131 km). The fitted `home` is that
  season's average, so average home advantage stays where the fit put it. In that league
  every ordered pair plays once, so the centre depends only on the twenty clubs. The
  26/27 league's trips average slightly longer: +0.3% on home goals league-wide.
- **The coefficient's uncertainty is carried, not dropped.** There is one draw of b per
  posterior draw, from N(0.0323, 0.0070), shared across every fixture in that draw.

What it does, in the calibrated home-advantage units (prior mean 0.184 log) `[DERIVED]`:

| 26/27 fixture type | home-goals multiplier | effective home advantage |
|---|---|---|
| derby at the 5 km floor (Chelsea–Fulham, Everton–Liverpool, Brentford–Fulham) | ×0.900 | 0.184 → **0.079** |
| North London (6 km) | ×0.906 | → 0.086 |
| league-centre trip (131 km) | ×1.000 | 0.184 |
| longest (Bournemouth/Brighton ↔ Newcastle/Sunderland, ~465 km) | ×1.042 | → 0.225 |

The log form makes the effect asymmetric. The derby end moves home advantage about 2.5×
as far as the long-trip end does.

## 6. The guard's scope — decided by the owner, 11 Sep 2026

> **Decision (11 Sep):** the owner switched `FPL_TRAVEL` **on by default**, an explicit
> override of the market gate scoped to this signal. It is recorded in the CLAUDE.md guard
> row. The gate stands for every other team signal. The override should be re-examined if
> per-fixture market λ ever enters the fit by default (`SOLIO_MARKET=on`), because the
> no-double-counting argument below would then no longer hold cleanly. The GW4 deadline
> lock (`gw4_board_locked_2026-09-11_deadline.csv`) was made with it off, so **GW5 is the
> first locked board that carries it.**

The argument put to the owner `[JUDGMENT]`:

The market gate exists so a team signal does not **double-count what the odds already
price**. This pipeline's per-fixture λ is `exp(mu + home + att − def)`, fitted on
`E0_recon` (xG-derived pseudo-odds plus results). **It ingests no match odds.** A
distance effect the market prices is therefore in no per-fixture number the model
produces, and there is nothing for it to double-count. If the market prices it fully,
adding it moves the model *toward* the market-calibrated benchmark the CS engine is
validated against, not away from it.

That argues the gate is the wrong test for this particular signal. The rule was fixed
before the result, though, and CLAUDE.md says a guard is not relaxed without saying so.
This document said so, and the owner acted on it (above). To run the board without it:

```bash
FPL_TRAVEL=off python scripts/gw_board.py
```

The scored validation is still owed; the switch-on did not replace it. Once enough 26/27
derbies have been played, compare the board's per-fixture goals-against calibration with
the term off and on. 26/27 has 34 fixtures under 20 km (14 under 10 km), so that is a season-long
accumulation. The Solio team compare cannot settle it: it covers a single gameweek, and
GW4 has exactly one fixture (Man United–Man City, 6 km, ×0.907) that moves by more than
3.5%.

## 7. A/B — what switching it on does to the board `[VERIFIED]`

Same seed, `LIVE_FPL=off`, `gw_board.py` + `export_team_projections.py`, off vs on.

**Team layer** (deterministic apart from the b draw; 760 team-fixtures):

- League mean λ_for goes 1.4577 → 1.4607 (+0.2%, the 26/27 league's slightly longer
  average trip). Away rows' λ_for is unchanged, exactly as intended.
- Per-fixture clean-sheet probability: mean |Δ| 0.4pp. The largest gain is **+3.6pp**,
  for away sides in derbies (Chelsea at Fulham, Liverpool at Everton, Brentford at Fulham;
  Man City at Old Trafford in GW4, 23.7% → 26.9%). The largest loss is **−1.4pp**, for
  the longest trips (Brighton/Bournemouth at Sunderland/Newcastle).
- Season expected clean sheets, GW1-38, move by at most ±0.12 per club: north-east and
  south-coast clubs down, London clubs up.

**Player board.** At the single-gameweek level the only large movers are in derbies. GW4
Man City defence/GK: Donnarumma +0.26, Guéhi +0.21. GW4 Man United attack: Mbeumo −0.21,
Fernandes −0.18. The GW4 top 30 is the same set in both arms.

**Read player deltas with care.** A same-seed A/B does not give common random numbers here.
numpy's Poisson sampler for small λ consumes a variable number of uniforms, so any change
to λ desynchronises the rest of that player's stream, and the arms become roughly
independent draws (SE of a GW difference ~0.15 pts at S = 3000 for a high-variance
attacker). Palmer's `att_ev` rose 5.4% for a 2.1% change in λ_for; Isak's rose 5.1% for
2.6%. Both are that noise around the right scaling, not a second effect. Players at clubs
whose shift is ~0 (Leeds, Newcastle, Bournemouth, Brentford in GW4) moved at most 0.015.
Season-sum player deltas (max 0.76 over GW4-38) are at the Monte Carlo noise floor. The team
table above is the clean read of what the term does.

## What was built

| file | what |
|---|---|
| `src/travel.py` | grounds with dated moves, great-circle trip, `fixture_shift`, 25/26 centre, `--selftest`, and a 26/27 table (`python src/travel.py`) |
| `src/bayes_model.py` | `_home_effect(..., trip=0.0)`: home side only, added after the early-season floor |
| `src/style_matchup.py` | `market_score_test`, the gate for any covariate; `beats_the_market` now calls it |
| `scripts/gw_board.py` | `FPL_TRAVEL` in the banner |
| `scripts/export_team_projections.py` | passes the trip, so the team table matches the board |
| `studies/travel_distance.py` / `.csv` | the pre-registered study, `--design` (outcome-blind) and `--selftest` |

**Wired 12 Sep 2026.** `captaincy.point_draws`, `cs_fixtures.py` and the `xga27` line in
`gw_board.py` built their own home term inline rather than through `_home_effect`, so they
ignored the GW1-3 discount and, once `FPL_TRAVEL` went on, the travel term with it. All
three now call `_home_effect` with the gameweek and one `fixture_shift` per fixture, as
`project()` does; so do the same `xga27` loops in `run_final_board.py` and
`tests/test_defcon_env.py`. The refactor is exact: with `FPL_TRAVEL=off`, all 700 GW4-38
team-fixtures give bitwise-identical λ. With it **on**, which is the default, the derbies
this document is about now reach the CS table — GW4 Man City at Old Trafford P(CS)
25.4% → 28.7%, GW5 Chelsea at Brentford 20.8% → 23.8%, and the longest trips fall (Brighton
at Sunderland −1.4pp). Measured effect in INTEGRATION_LOG, 12 Sep.
`ab_market_vs_recon.py` keeps its inline term: it cannot run on this machine (hard-coded
`/tmp` and `/home/claude` paths), its two fixtures are GW4 and GW8, and it reports a
difference between two fits that share one home convention.
