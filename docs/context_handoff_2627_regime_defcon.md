# Context Handoff — Regime Change & DefCon
### FPL-Core-Insights, 2026/27 · prepared 7 Aug 2026 · ~14 days to GW1

Self-contained. Paste into the model-building conversation without prior context.
Companion doc: `regime_weighting_note.md` (full variance derivation and test spec).

Claim flags used throughout:
`[VERIFIED]` read from code or a primary source · `[DERIVED]` analytic result ·
`[JUDGMENT]` not fitted, needs a sweep · `[CHECK]` action required before trusting

---

## 0. What this session changed

Four things, in descending order of urgency:

1. A **ninth regime-change club** (Newcastle) that the seed dict is missing.
2. A **variance bug**: the regime discount cannot widen any posterior. It is
   mean-only by construction.
3. **DefCon is not auditable** anywhere in the pipeline, and its one external
   benchmark is populated for 4 of 30 rows.
4. A **two-channel mis-specification**: DefCon has two team-level drivers that
   move in partially opposite directions, currently collapsed into one.

Nothing here requires a refit. Items 1 and 3 are hours of work.

---

## 1. Data corrections

### 1.1 Newcastle — ninth regime-change club `[VERIFIED]`

`NEW_MANAGER_CLUBS_2627` was frozen 2026-07-28. Newcastle appointed Matthias Jaissle
on 5 Aug 2026 (Eddie Howe stepped down; ~£9.5m compensation to Al-Ahli, four-year
deal). Not in the dict. Nine full regime changes, not eight.

Newcastle is the worst information state in the league: appointed ~18 days before the
opener, no pre-season with the group, and a squad stripped of Isak, Trippier, Gordon
and Tonali, with Guimarães reported to Arsenal. Proposed `δ = 0.35`, below the flat
0.50 used elsewhere. `[JUDGMENT]`

The other eleven managers named in the PL's 26/27 line-up are incumbents and correctly
excluded: Andrews (Jun 2025), Lampard (Nov 2024), Jakirovic (summer 2025), Le Bris,
Farke, Moyes, Hurzeler, Emery, Arteta, plus the two partial-regime cases below.

### 1.2 Rule stability confirmed `[VERIFIED]`

The Premier League confirmed on 20 July 2026 that defensive contributions are
**unchanged** for 26/27: 10 CBIT for defenders, 12 CBIRT (adds ball recoveries) for
midfielders and forwards, capped at +2 per match. This validates the prior session's
finding. BPS changes and the two-set chip structure are also unchanged.

### 1.3 Position reclassifications `[VERIFIED] [CHECK]`

Eleven players reclassified for 26/27. The DefCon-material one is **Mats Wieffer**
(Brighton), midfielder → defender. He played all 23 of his starts at right-back and
banked DefCon in 11 of 17 matches lasting 75+ minutes; as a defender — losing
recoveries but needing 10 instead of 12 — he would still have hit in 8 of those 11.
Also confirmed: Lewis-Skelly DEF→MID, Kroupi FWD→MID, Sessegnon MID→DEF, Dorgu and
Marmoush changed.

**Action:** confirm the position field in the build frame reflects the 26/27
classification, not 25/26. A stale `pos` breaks the DefCon threshold assignment
(10 vs 12) *and* the goal-points multiplier for every reclassified player.

### 1.4 Source blacklist

`onsidearena.com` claims the Assistant Manager chip is active for 26/27 and that
DefCon thresholds are "under review". Both contradicted by the PL. Do not ingest.

---

## 2. The regime-discount variance bug

### 2.1 What the code does `[VERIFIED]`

`starter_prior.apply_minutes_shrinkage_with_regime()`:

```python
N      = r.start_a + r.start_b
p_hist = r.start_a / N
p_own  = coldstart_start_prob(...)
w      = eff_min / (eff_min + k_min)      # eff_min = minutes * regime_discount(club)
p_new  = w * p_hist + (1 - w) * p_own
p.at[i, "start_a"] = p_new       * N       # <-- N PRESERVED
p.at[i, "start_b"] = (1 - p_new) * N       # <-- N PRESERVED
```

`Var[p] = p(1-p)/(N+1)`. Concentration is carried through unchanged, so the discount
**can only move the mean**. It slides toward the ownership-implied prior and never
widens anything. The module docstring says the opposite. (The same
concentration-preserving convention in `apply_transfer_fee_floor()` is correct there —
a floor genuinely is a mean statement.)

### 2.2 The fix — two decoupled knobs

| Symbol | Meaning | Range | No-op |
|---|---|---|---|
| `δ_club` | minutes discount → drives `w` → pulls the mean toward `p_own` | (0, 1] | `1.0` |
| `κ_club` | disagreement inflation → widens the posterior | [0, 1] | `0.0` |

`δ=1.0, κ=0.0` must reproduce the current baseline bitwise.

Moment-match a genuine two-component mixture instead of a mean blend:

```
p̄   = w·p_h + (1-w)·p_o
v_h = p_h(1-p_h)/(N_h+1);   v_o = p_o(1-p_o)/(N_o+1)     # N_o = kstart = 6.0
v   = w·v_h + (1-w)·v_o  +  κ · w(1-w)·(p_h - p_o)²
N_eff = p̄(1-p̄)/v - 1;   a = p̄·N_eff;   b = (1-p̄)·N_eff
```

The between-component term is the point. It peaks at `w = 0.5` — exactly where the
discount pushes affected players — and scales with the **squared disagreement** between
history and the market. A Newcastle regular whose ownership has collapsed gets a wide
interval; a Man City player whose history and ownership agree barely moves even at
`δ=0.5`. That asymmetry falls out of the algebra rather than being asserted.

Guard: clamp `N_eff ≥ 1.0` and log clamp events. A high clamp rate means `κ` is too
aggressive.

Keeping `δ` and `κ` separate matters because they encode different claims. `δ` asserts
the market's ownership signal beats history at this club. `κ` asserts only ignorance.
`δ=1.0, κ>0` — don't move the mean, just widen — is currently unreachable and may be
the honest setting.

### 2.3 Horizon result `[DERIVED]`

With `S_h | p ~ Bern(p)` i.i.d. given `p`, and points-given-start mean `μ`, variance `σ²`:

```
Var[X_H] = H·[ p̄σ² + p̄(1-p̄)μ² ]  +  H(H-1)·μ²·v_p
```

Three consequences:

- **At H=1 the second term vanishes.** Uncertainty in `p` is fully absorbed by the
  Bernoulli marginal. Anyone testing this on a GW1-only run sees nothing and concludes
  it's inert.
- **At H=6 the shared-p term carries a factor of 30 against 6.** This is where regime
  uncertainty belongs — the GW1–6 board.
- **It scales with `μ²`.** Premium assets at regime clubs take the widening
  quadratically harder than fodder. If Salah and Palmer don't end up with visibly
  fatter intervals than a £4.5m defender at the same clubs, it isn't wired up.

### 2.4 The line that makes or breaks it

**`p` must be drawn once per player per simulation path and held fixed across the
horizon.** Draw order: `p ~ Beta(a,b)` outer, `S_h ~ Bern(p)` inner. If the simulator
redraws `p` per gameweek or substitutes `p̄`, the `H(H-1)` term is destroyed and the
entire mixture widening is silently discarded. Highest-risk item in the change.

### 2.5 Proposed settings `[JUDGMENT]`

| Club | Manager | δ now | δ prop. | κ prop. |
|---|---|---|---|---|
| Newcastle | Jaissle | *absent* | 0.35 | 0.8 |
| Liverpool | Iraola | 0.50 | 0.50 | 0.8 |
| Chelsea | Alonso | 0.50 | 0.50 | 0.7 |
| Nott'm Forest | Glasner | 0.50 | 0.50 | 0.7 |
| Crystal Palace | Sage | 0.50 | 0.50 | 0.7 |
| Bournemouth | Rose | 0.50 | 0.50 | 0.7 |
| Fulham | Arbeloa | 0.50 | 0.50 | 0.7 |
| Ipswich | O'Neil | 0.50 | 0.55 | 0.6 |
| Man City | Maresca | 0.50 | 0.55 | 0.6 |
| Tottenham | De Zerbi | 0.55 | 0.45 | 0.7 |
| Man United | Carrick | 0.70 | 0.80 | 0.4 |

Starting points for a sweep, not fitted values. Move to a config surface (YAML/env/CLI)
and log resolved values in the run manifest alongside `market_share` and `older_weight`.

**Preferred alternative for Spurs and Man Utd:** split the per-match panel by
appointment date rather than applying a flat multiplier. The multiplier discards the
information that ~17 of Man United's matches *were* under Carrick. If the date split is
cheap, do that instead of tuning `δ`.

### 2.6 Acceptance tests

1. `δ=1.0, κ=0.0` reproduces `decision_gw1_6_full.csv` bitwise.
2. Non-regime clubs bitwise unchanged under any `(δ, κ)`.
3. `∂SD/∂κ > 0` and `∂SD/∂δ < 0` over GW1–6. Assert, don't eyeball.
4. **Horizon signature:** SD vs `H` for `H ∈ {1..6}` at `κ=0.8`. The `H=1` SD must
   equal baseline. If it differs, the simulator is redrawing `p` per GW.
5. **Decoupling:** a player with `p_h ≈ p_o` at `δ=0.5, κ=0.8` shows ≈0 mean shift but
   non-zero SD widening.
6. **`μ²` scaling:** SD inflation should correlate with projected EP, not be flat
   across price tiers.
7. Rank correlation against the Solio GW1 benchmark must not degrade.

---

## 3. New-manager first-ten evidence

### 3.1 Base-rate warning `[DERIVED]`

The "new manager bounce" does not apply here. It is a selection artefact: dismissal
follows a bad run, so post-change improvement is regression to the mean (Audas/Dobson/
Goddard 1997–2002; Bruinshoofd & ter Weel 2003; ter Weel 2011 matched-sample; van Ours
& van Tuijl 2016). **Seven of nine 26/27 appointments are close-season** — no
truncation on a bad run, so not even the artefact transfers. Do not import a positive
first-10 prior.

### 3.2 Evidence by club `[VERIFIED]`

| Club / manager | First-10 evidence | Read |
|---|---|---|
| Liverpool — Iraola | Winless in his first nine at Bournemouth, 0.33 ppg, GD −14; 22 pts in the next nine | Only same-league, same-role evidence in the set, and it's **negative** |
| Forest — Glasner | 24 pts from 13 at Palace but six of the last seven won → first ~6 ≈ 5 pts. Following season opened 2 pts from 4 | Consistent **slow starter** |
| Newcastle — Jaissle | Salzburg/Al-Ahli titles only; no top-5 league experience | Widest posterior in the league |
| Chelsea — Alonso | 13 of first 14 at Real Madrid, 10 of first 11 in LaLiga; then two wins in eight from November | **Fast starter, poor persistence** |
| Man City — Maresca | 10 pts from first 5 at Chelsea; 25 from 13; first new PL boss since Guardiola 2016 to win his first three away | Cleanest positive signal |
| Man Utd — Carrick | First 10 PL: W7 D2 L1, 2.30 ppg; 39 from 17; 1.14 GA/game, 1.50 big chances allowed (2nd best) | Excellent, but textbook in-season selection case — **discount** |
| Palace — Sage | 2.08 ppm at Lens, 2nd in Ligue 1 + Coupe de France; at Lyon took a bottom side to 46 pts from 22 | Strong, but zero English data |
| Bournemouth — Rose | Career ppg: Salzburg 2.35, Dortmund 1.85, Leipzig 1.85, Gladbach 1.61; 37 pts in his first 17 at Leipzig | Historically fast starts, all Bundesliga/Austria |
| Ipswich — O'Neil | Survival at Bournemouth, 14th at Wolves, Conference SF at Strasbourg | Competent floor, low ceiling |
| Fulham — Arbeloa | Promoted from Castilla mid-season; 2nd in LaLiga + CL QF with Madrid's squad | Near-uninformative |
| Spurs — De Zerbi | ~9 GWs, relegation survival on the final day | Small **and** unrepresentative sample |

---

## 4. DefCon

### 4.1 Three gaps found by inspection `[VERIFIED]`

**(a) Not auditable.** `decision_gw1_6_*.csv` carries only `mean, sd, p5…p95`. The
`defcon_alpha/defcon_beta` Beta-Binomial is consumed inside `project()` and never
surfaced. You cannot A/B a DefCon change, decompose a projection, or validate it.
Contrast the CS engine, validated to GA r=0.89 / CS r=0.93 precisely because
`cs_fixtures_gw1_10.csv` exposes `xGA` and `cs_prob` per fixture.
→ **Add a `defcon_ev` (or `defcon_hit_rate`) column to the board.**

**(b) Benchmark is 4 rows.** `solio_ensemble_demo-1.csv` has `defcon_pct` populated for
Tarkowski (55), Anderson (56), Murillo (48), Garner (45) — 4 of 30. The parser reads
the DefCon tab but the merge drops almost everything. Fixing this yields a proper GW1
DefCon benchmark for near-zero effort. Cheapest validation available before the
deadline.

**(c) Live natural experiment.** Elliot Anderson — top DefCon scorer last season, 52
points at Forest — is now at Man City. Projected xGA: Forest 1.41, Man City 1.04
(second-lowest in the league). If the model carries his Forest hit rate across as a
pure player trait it over-projects him. **Checkable today.**

This qualifies the existing project framing ("DefCon is a repeatable player trait, not
a fixture lottery"). The trait is repeatable *conditional on team environment*. Player
trait ≠ team-invariant.

### 4.2 Two-channel mis-specification `[DERIVED]`

The thresholds have different and partially **opposing** team-level drivers:

- **CBIT (DEF, 10)** — clearances and blocks dominate. Driven by *being defended
  against*: opponent territory and possession. Monotone increasing in xGA.
- **CBIRT (MID/FWD, 12)** — adds recoveries, generated by press intensity and
  loose-ball volume, which a possession-dominant high-press side produces plenty of in
  the opposition half.

A single team DefCon multiplier is therefore mis-specified. Need two:

```
defcon_env_CBIT[team]  ∝ f(xGA, block_height)
defcon_env_CBIRT[team] ∝ g(xGA, press_intensity)
```

### 4.3 Projected defensive-action environment (mean xGA, GW1–10) `[VERIFIED]`

```
Hull      2.30 | Leeds       1.52 | Brentford  1.37 | Fulham     1.31
Ipswich   1.90 | Bournemouth 1.52 | Sunderland 1.37 | Man United 1.30
Coventry  1.82 | Tottenham   1.50 | Newcastle  1.36 | Man City   1.04
               | Liverpool   1.43 | Palace     1.36 | Arsenal    0.82
               | Forest      1.41 | Brighton   1.34 |
               | Chelsea     1.40 | Everton    1.34 |
               | Aston Villa 1.39 |
```

**Newcastle at 1.36 is probably too low** given the squad stripping — a team-model
input issue, not a DefCon one, but it propagates into the DefCon channel. `[CHECK]`

### 4.4 The test worth running `[JUDGMENT]`

Do **not** add manager fixed effects. Fit on the 25/26 per-match panel:

```
logit P(hit | player, match) = α_player + β·log(xGA_team,match)
                             + γ·press_index + manager_FE
```

and test whether the manager FE are jointly zero given team strength.

**This is the one place in the model with the power to detect a style effect.** CBIT
counts run 40–70 per team-match against ~1.4 goals. The project's own finding is that
shots carry 3–4× the power of goals for detecting tactical effects; defensive-action
counts carry roughly an order of magnitude more. The prior null style results came back
null partly because goals are a terrible outcome variable — that objection does not
apply here.

**This is not the `style_matchup.py` situation.** That guard exists because style→goals
is already in the odds, making any real effect worthless alpha. There is **no market on
CBIT** — bookmakers price goals and totals. So a real effect here is real alpha, with
no double-counting risk against the market-calibrated CS engine. But only the residual
after conditioning on projected xGA is new information, since that part *is* indirectly
priced. Hence the specification above.

### 4.5 Interaction with κ `[DERIVED]`

DefCon is a threshold crossing on a roughly-Poisson count, so P(hit) as a function of
team defensive volume is S-shaped and steepest near the threshold. Team-environment
uncertainty therefore bites hardest on players near a 50% hit rate — **exactly the
£4.5–5.5 band the DefCon strategy targets.** Regime-change widening hurts most where
the strategy is most exposed. Bernoulli variance `p(1-p)` peaks at 0.5, so this falls
out naturally if you widen the Beta rather than shifting it.

---

## 5. Manager style → CB vs CM channel `[JUDGMENT]`

Determinants, in order. Press height is only third:

1. **Team xGA / possession share** — drives the CB channel almost entirely. Already in
   the team model.
2. **Single vs double pivot** — a lone #6 concentrates defensive-midfield volume and
   clears 12 CBIRT; a double pivot splits it and often neither hits. Mechanical, and
   the biggest CM lever.
3. **Back three vs back four** — a four concentrates CBIT in two players; a three
   dilutes across three.
4. **Press height** — shifts the *mix* (tackles/interceptions/recoveries up, clearances
   down) more than the total.

| Club / manager | Favours | Note |
|---|---|---|
| Ipswich — O'Neil | **CB, strongly** | xGA 1.90; environment overwhelms style |
| Forest — Glasner | **CB, strongly** | Archetype club; back three dilution is the only drag |
| Palace — Sage | **CB** | Back three defending behind advanced wing-backs |
| Bournemouth — Rose | CB on environment, CM on style | Least-disrupted DefCon environment of the nine — Iraola already pressed (PPDA 10.6, 4th lowest). But Senesi has gone to Spurs |
| Newcastle — Jaissle | Conflicted → **CB** | Press says CM; squad stripping says CB. The squad effect is larger and more certain |
| Liverpool — Iraola | CM, weakly | Dominant side → poor CB channel; recoveries split across a double pivot |
| Man Utd — Carrick | **CM** | 1.14 GA/game, 1.50 big chances allowed — a controlled low-event defence starves CBs |
| Spurs — De Zerbi | **CM** | High line gives some CB volume back via transitions |
| Chelsea — Alonso | **CM** | Possession-dominant; Caicedo-type profile only |
| Fulham — Arbeloa | **CM** | Lowest confidence in the table |
| Man City — Maresca | **CM only** | xGA 1.04; City CBs effectively unplayable for DefCon |

Promoted sides settle on environment before style: **Hull (Jakirovic, 2.30)** and
**Coventry (Lampard, 1.82)** are the richest CB environments in the league.

**Two structural notes.** First, the wing-back in a back three is the best available
archetype and both back-three managers produce it: FPL-classified as a defender, so 10
CBIT not 12 CBIRT, while doing midfield-volume covering work. Second, **no new manager
is a low-block CB-farmer** — all nine lean proactive. Combined with Anderson leaving
Forest for City and Senesi leaving Bournemouth for Spurs, cheap reliable CB DefCon is
concentrating into Forest, Ipswich, Hull and Coventry. The ownership-aware depth prior
won't see this, because it's the environment changing, not the player's role.

**Caveat:** for the eleven incumbents this table is decoration — fitted
`defcon_alpha/beta` on real PL matches beats any stylistic prior. It has decision value
only for the nine regime clubs and the two partial ones, which is also where it is
least reliable.

### 5.1 Out-of-league transfer bias `[DERIVED]`

Jaissle, Rose, Alonso, Sage and Arbeloa all come from clubs that were **dominant in
weaker leagues** (Salzburg 2.35 ppg in Austria; Al-Ahli; Leverkusen/Madrid; Lens 2.08
ppm; Castilla). A side with 65% possession against weak opposition records few
clearances. Transferring raw per-90 defensive rates to a PL club **systematically
under-predicts CBIT**, and the bias is largest where data is thinnest — Jaissle worst
on both axes at once.

The transferable quantity is defensive actions **per unit of opponent possession**
(essentially what PPDA constructs for the pressing component), which is roughly
league-invariant for a given coach. Raw per-90 is not.

Secondary: "ball recovery" and "clearance" are Opta-defined events. Consistency for the
Austrian Bundesliga and Saudi Pro League against the PL feed cannot be assumed — treat
any cross-league CBIRT figure as carrying unquantified *measurement* error on top of
sampling error.

---

## 6. Sub-7% DefCon watchlist

Drawn from `decision_gw1_6_depthprior.csv`, cross-referenced against projected xGA,
filtered on `cold_start = False` so the DefCon Beta is fitted rather than prior.

| Player | Pos | Club | £ | Own | Case |
|---|---|---|---|---|---|
| Wieffer | DEF | Brighton | 5.0 | 0.5% | Reclassified MID→DEF. Hit DefCon in 11/17 games of 75+ min; 8 of those 11 would have hit at CBIT-only. Threshold drops, volume stays |
| Milenković | DEF | Forest | 5.5 | 2.7% | Highest proven ceiling under 7% (+26 in the 24/25 counterfactual). Archetype club, back three |
| Thiaw | DEF | Newcastle | 5.0 | 2.7% | Top of the board at 25.70. xGA of 1.36 likely understated |
| Ampadu | MID | Leeds | 5.5 | 1.3% | Best CBIRT candidate. Single pivot at the highest-xGA established club; sd 6.62, lowest on the board |
| Rodon | DEF | Leeds | 4.5 | 2.6% | Same environment at £4.5; Farke incumbent so priors fitted |
| Ballard | DEF | Sunderland | 5.0 | 2.3% | Aggressive CB, xGA 1.37, zero regime risk |
| Collins | DEF | Brentford | 5.5 | 2.5% | 32 DefCon in the 24/25 counterfactual; Andrews settled |
| Keane | DEF | Everton | 5.0 | 1.6% | Moyes deep block at a fifth of Tarkowski's ownership. Branthwaite competition is real |
| Garner | MID | Everton | 6.0 | 3.9% | Only shortlist name with an external reading — Solio 45% |
| Xhaka | MID | Sunderland | 5.5 | 5.6% | High deep-lying volume, sd 6.16. Will likely breach 7% within a fortnight |

### 6.1 Traps

- **Promoted clubs.** Hull (2.30), Ipswich (1.90) and Coventry (1.82) own the top three
  environments, and the board is stuffed with their £4.0 defenders — Egan, Ajayi,
  Coyle, Kitching, Amenda. Nearly all `cold_start=True` at sub-1% ownership: the
  phantom-starter signature the depth prior exists to catch. Maximum environment, zero
  evidence on who starts. Revisit after team news.
- **Anderson** (£6.5, 12.6%) — see §4.1(c).
- **Zubimendi** (£5.5, 1.2%) — perfect role, worst environment (Arsenal 0.82).
- **Murillo `[CHECK]`** — 38-point DefCon contributor in the counterfactual, but the
  board has him at 0.1% ownership and mean 13.89 against Milenković's 19.58 at the same
  club and price. Suggests the availability layer has flagged an injury or exit. If a
  data artefact, he's the best name on the list; if real, ignore.

### 6.2 Known bug still open

`decision_gw1_6_depthprior.csv` contains duplicate rows for Hughes, Dasilva, Johnson
and Kamara (differing only on `cold_start`). Deduplication is on `player+team` rather
than player ID. Previously documented for Palmer; confirmed still present and now
affecting promoted-club players, where it interacts badly with the cold-start path.

---

## 7. Priority queue

1. **Add Newcastle to the regime dict** — one line, blocks nothing else. `[§1.1]`
2. **Verify 26/27 position classifications in the build frame** — breaks DefCon
   thresholds and goal multipliers if stale. `[§1.3]`
3. **Fix the Solio DefCon parser** — turns a 4-row benchmark into a real one.
   Cheapest validation available before the deadline. `[§4.1b]`
4. **Surface `defcon_ev` on the decision board** — precondition for validating or
   A/B-ing anything in §4. `[§4.1a]`
5. **Implement `δ`/`κ` reparameterisation** with the acceptance tests. `[§2]`
6. **Audit the simulator's draw order for `p`.** `[§2.4]`
7. Live odds feed cron (API key to env var) — still the largest single predictive gain.
8. Live lineups/injuries feed.
9. `market_share × κ` joint sweep — see below.
10. Fix ID-based deduplication. `[§6.2]`
11. 2023/24 data for walk-forward `older_weight` calibration.
12. Transfer/chip optimisation solver.

### 7.1 On the joint sweep

Nine regime changes means bookmaker disagreement on team ratings will be unusually wide
in GW1–3. This cuts both ways:

- Wider market disagreement argues for *lower* `market_share` (the market is noisy now)
- Regime change argues for *higher* `market_share` (our historical priors are stale)

These pull in opposite directions and the current 0.35 is asserted, not calibrated.
**Sweep `market_share × κ` on a grid, not coordinate-wise** — the optimum for one is
almost certainly conditional on the other, and a coordinate-wise sweep will find a local
answer and present it as the answer.

---

## 8. What not to do

The temptation with nine new managers is directional style priors — "Iraola presses
high, bullish Liverpool attackers"; "Glasner is defensively elite, buy Forest CBs".
Same class of heuristic as the rotation multipliers (null, p=0.23) and mean-reversion
(rejected, p=0.69). The Glasner slow-start pattern is the only one with even a
suggestion of a repeatable signature, and n=2 seasons is not a test.

`κ` is the correct home for everything we believe about regime change, because
everything we believe about regime change is a statement about ignorance.

The DefCon channel in §4.4 is the **exception that should be tested rather than
assumed** — not because the prior is stronger, but because the outcome variable finally
has the power to resolve it and the result isn't already priced.
