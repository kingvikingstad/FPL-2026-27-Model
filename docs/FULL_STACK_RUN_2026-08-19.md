# Full-stack run — 19 Aug 2026 (pre-GW1)

*Every validated layer on, GW1-10, true per-gameweek projections. Run two days before the
GW1 deadline (2026-08-21 17:30). This document records what was run, what the model says,
what was found wrong on the way, and which claims in `PROJECT_KNOWLEDGE_2627.md` this run
revises.*

Companion outputs:

| file | grain | what it carries |
|---|---|---|
| `outputs/gw_board_long.csv` | player × gameweek | mean, sd, p5/median/p95, `defcon_ev`, `cs_ev`, Solio blend |
| `outputs/gw_board_wide.csv` | player | gw1..gw10 + total |
| `outputs/projection_detail_gw1_10.csv` | player × gameweek, **79 cols** | the above **plus** the fixture, the full betting-odds provenance chain, the match-outcome distribution, the press covariate, and every installed prior |
| `outputs/team_projections_season.csv` | club | odds → de-vig → market z → market Elo → blend → fitted posterior attack/defence |
| `outputs/team_projections_gw1_10.csv` | club × gameweek | λ for/against with CIs, P(CS), P(win/draw/loss), P(BTTS), expected league points |

---

## 1. Configuration actually run

```
GW_HI=10  DRAWS=3000
MARKET_ODDS=on  MARKET_WEIGHT=0.6     betting-odds team strength
DEFCON_ENV=on                          DefCon conditioned on xGA (DEF) + press (MID/FWD)
REGIME_PANEL=on                        appointment-weighted minutes, Man Utd / Spurs
REGIME=kappa                           regime uncertainty as VARIANCE ONLY  [new mode]
FPL_SETPIECE=override                  FFS 26/27 projected set-piece duty
SOLIO=on  SOLIO_W_OURS=0.5             GW1 blend against the cached Solio feed
```

Engine defaults carrying previously-validated corrections: calibrated team hyperparameters
(31 seasons), `MINUTES_IF_START` positional minutes, and the GW1-3 symmetric home discount.

### 1.1 The one flag that is deliberately NOT on

The request was to include *every* aspect of the model. One switch was held back and is
reported as a sensitivity instead of a headline, because turning it on would contradict a
settled decision rather than add information.

`REGIME=proposed` applies **both** knobs in `REGIME_2627_PROPOSED`: a κ variance widening
*and* a δ mean-pull that shrinks effective minutes at the eleven regime clubs. §7 of
`PROJECT_KNOWLEDGE_2627.md` records the δ values as unfitted `[JUDGMENT]`, "wired but off.
Turning them on shifts point estimates on assertion. Sweep against live data first." §3
states the general position: regime change is an ignorance statement and belongs in
variance, not in directional priors.

A third mode was therefore added — `REGIME=kappa` (`starter_prior.resolve_regime`) — which
keeps each club's κ and forces δ to its no-op 1.0. That is the headline board: regime clubs
carry wider posteriors, and no point estimate moves on an unfitted assertion. The δ-on
variant is quantified in §5.3.

---

## 2. Health

`scripts/test_all.py --quick`, run against the board produced here, not a stale one:

* 36/36 module imports
* 5/5 module selftests
* 4/4 acceptance tests (`test_regime`, `test_defcon_env`, `test_regime_panel`, `validate_shrinkage`)
* 10/10 board invariants — no NaN, no negatives, one row per player-gameweek, wide totals
  reconcile to long sums, all four positions, ten gameweeks present

---

## 3. Defects found and fixed during the run

### 3.1 The penalty override silently kept demoted takers — [VERIFIED, FIXED]

`set_piece_takers.apply_to_signals(mode="override")` promoted the projected taker but did
not always demote the incumbent. Its club lookup was built from the taker table itself:

```python
club_of = dict(zip(R["skey"], R["club"]))     # R = the FFS projection only
```

A player absent from the projection therefore resolved to club `None`, never satisfied
`t in covered`, and fell through to the stale FPL flag — precisely the demotion the mode
exists to perform. The module's own log claimed the demotion had happened, because that
tally compared name sets rather than reading back what was written.

Effect on the board: **three clubs carried two order-1 penalty takers** — Liverpool
(Szoboszlai *and* Isak), Hull, Ipswich — each simulating roughly double the penalty xG it
should, at a documented ~16 points a season per settled taker.

Fixed by building the club map from the full squad frame. A post-write assertion now fails
the run if any club ends with more than one order-1 taker in any set-piece role.

### 3.2 Penalty duty was keyed on `web_name` — [VERIFIED, FIXED]

Both runners derived penalty duty from the name-keyed signals frame:

```python
pen1 = set(sig.loc[sig.pen_order == 1, "name"].str.lower().str.strip())
```

**15 web_names in the 26/27 squad belong to players at two different clubs.** One of them is
`Palmer` — first-choice taker at Chelsea, while Ipswich's projected taker is Clarke. A
name-keyed lookup cannot separate the two, so Ipswich retained a second taker even after
3.1 was fixed. This violated the project's own standing rule to join on `player_code`.

Added `set_piece_takers.pen1_codes(players, mode=...)`, keyed on `player_code` and covering
all three modes. Verified: exactly one order-1 taker per club, 20 codes across 20 clubs, in
`override`, `fill` and `off` alike.

### 3.3 Clean-sheet probability was the plug-in, not the posterior predictive — [VERIFIED, FIXED]

`export_projection_detail.fixtures()` computed `clean_sheet_prob = exp(-mean(λ_against))`.
Since `exp(-x)` is convex, Jensen's inequality makes this biased **downward** — it
understates every clean sheet. The engine itself is correct (it draws
`poisson(lam_against) == 0` per posterior sample, `bayes_model.py:413`), so the export also
disagreed with the board it exists to explain.

Replaced with `mean(exp(-λ_against))`, integrated over draws. Measured gap: **mean +0.0180,
max +0.0261** in probability. Both forms are emitted in the team export
(`p_clean_sheet`, `p_clean_sheet_plugin`) so the size stays visible.

### 3.4 The detail export did not mirror the board's flags — [FIXED]

`export_projection_detail.py` hardcoded `regime={}` and skipped `apply_regime_panel_split`,
so the priors it reported were the ones the simulation drew from only while both layers
happened to be off by default. With `REGIME_PANEL=on` it would have published prior columns
that no simulation ever used. It now reads the same environment flags as `gw_board.py`, and
its draw count defaults to `DRAWS` (3000) rather than a hardcoded 1500 — the team sampler is
seeded, but `S` changes the draw stream itself, so a different `S` reports λ the board never
saw.

It also no longer fits its own `TeamModel`. Both exports now come from
`export_team_projections.build()`, so the team file and the player file cannot drift apart.

---

## 4. What the model says

### 4.1 Top of the board, GW1-10 (per-gameweek means; GW1 Solio-blended where matched)

| player | pos | team | £ | gw1 | gw2 | gw3 | gw4 | gw5 | gw6 | gw7 | gw8 | gw9 | gw10 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Haaland | FWD | Man City | 15.5 | 6.78 | 6.70 | 8.47 | 6.31 | 7.27 | 6.47 | 9.26 | 6.40 | 6.76 | 6.42 | **70.84** |
| B.Fernandes | MID | Man United | 12.0 | 6.86 | 6.26 | 5.18 | 4.79 | 4.91 | 5.90 | 5.25 | 5.56 | 4.88 | 5.20 | 54.78 |
| Saka | MID | Arsenal | 9.5 | 5.80 | 4.59 | 5.00 | 4.49 | 4.28 | 5.49 | 4.59 | 5.09 | 4.50 | 6.68 | 50.51 |
| Mbeumo | MID | Man United | 8.0 | 6.06 | 5.85 | 4.52 | 4.12 | 4.42 | 5.48 | 4.65 | 5.28 | 4.52 | 4.92 | 49.81 |
| Semenyo | MID | Man City | 8.5 | 5.06 | 4.63 | 5.63 | 4.36 | 4.83 | 4.45 | 6.17 | 4.29 | 4.78 | 4.42 | 48.60 |
| Gabriel | DEF | Arsenal | 8.0 | 5.80 | 4.51 | 4.48 | 4.75 | 4.30 | 5.04 | 4.48 | 5.07 | 4.04 | 6.12 | 48.60 |
| Palmer | MID | Chelsea | 9.5 | 4.95 | 4.55 | 3.51 | 6.50 | 4.28 | 5.12 | 4.29 | 5.22 | 4.63 | 4.30 | 47.35 |
| O'Reilly | DEF | Man City | 6.5 | 4.64 | 4.68 | 5.55 | 3.83 | 4.91 | 3.97 | 5.90 | 4.15 | 4.79 | 4.36 | 46.79 |
| Guéhi | DEF | Man City | 6.0 | 4.32 | 4.65 | 5.27 | 3.74 | 5.01 | 3.78 | 5.63 | 3.95 | 4.74 | 4.30 | 45.39 |
| Thiago | FWD | Brentford | 8.0 | 4.58 | 4.32 | 4.24 | 4.13 | 4.33 | 4.13 | 4.55 | 5.21 | 4.46 | 3.85 | 43.81 |

Best value by position (points per £m over the horizon): **Thiaw** DEF Newcastle £5.0 (7.88),
**Gvardiol** DEF Man City £5.5 (7.66), **Guéhi** DEF Man City £6.0 (7.56), **Raya** GK
Arsenal £6.0 (6.38), **Mbeumo** MID Man United £8.0 (6.15).

### 4.2 Captain by gameweek

Haaland in eight of ten. The two exceptions are fixture-driven, not form-driven: **Palmer**
in GW4 (Chelsea host Hull, λ_for 2.74 — the single best attacking fixture of the window) and
**Saka** in GW10 (Arsenal host Hull, λ_for 3.15, P(win) 0.85). Haaland's own peak is GW7
(9.26, City host Ipswich, λ_for 2.93).

Captaincy is a tail problem, so the mean is the wrong statistic on its own — GW7 Haaland
carries p95 = 23.4 against 18.1 in a typical week, and GW10 Saka's p95 of 21.1 exceeds
Haaland's 18.1 that week despite a lower mean.

### 4.3 Team layer, GW1-10

Expected league points over the ten weeks, from the posterior match-outcome distribution:

| | club | xG/match | xGA/match | Σ P(CS) | exp. pts |
|---|---|---|---|---|---|
| 1 | Arsenal | 1.96 | 0.77 | 4.81 | 20.75 |
| 2 | Man City | 2.07 | 1.01 | 3.88 | 19.88 |
| 3 | Man United | 1.94 | 1.29 | 3.01 | 17.56 |
| 4 | Chelsea | 1.62 | 1.40 | 2.75 | 15.03 |
| 5 | Newcastle | 1.58 | 1.37 | 2.79 | 14.96 |
| … | | | | | |
| 18 | Coventry | 1.09 | 1.78 | 2.07 | 10.05 |
| 19 | Ipswich | 1.07 | 1.88 | 1.90 | 9.36 |
| 20 | Hull | 0.90 | 2.27 | 1.39 | 6.86 |

Arsenal are the clean-sheet asset of the window (Σ P(CS) 4.81 over ten games, xGA 0.77) and
own the best single CS fixture in five of the ten weeks. Hull are the opponent to target:
the best attacking fixture in the window belongs to whoever is playing them in GW1, 4, 5, 8
and 10.

### 4.4 Differentials (< 8% owned, > 28 projected)

Enzo (£7.0, 5.5%, 42.9), Foden (£7.0, 6.3%, 39.8), Thiaw (£5.0, 2.0%, 39.4), Tzolis (£6.5,
3.9%, 38.9), Truffert (£5.5, 4.9%, 37.9), Hincapié (£5.5, 3.9%), Gakpo (£7.0, 3.3%),
Schade (£6.0, 1.9%).

### 4.5 Agreement with Solio (independent public model, GW1, 30 matched)

Pearson **0.777**, Spearman **0.700**, MAE **0.867**, mean signed **−0.745**.

The negative mean is *not* straightforwardly evidence that this model under-projects. The
matched set is Solio's own published top-30, i.e. selected on Solio's ranking, so regression
to the mean guarantees any second model scores that set lower on average. It is a selected
sample, not a random one. The rank agreement (0.70) is the informative statistic.

Largest disagreements: Richarlison (−2.69), Mosquera (−1.67), Haaland (**+1.45**, the only
large positive), Dewsbury-Hall (−1.43), Gibbs-White (−1.42).

---

## 5. What each layer actually contributes

### 5.1 Betting odds — the headline finding: as wired, they are inert — [VERIFIED]

This is the layer the request specifically asked to see, and the honest answer is that it
currently does almost nothing.

**Player level** (A/B, market on vs off, everything else identical, 584 players):

* mean |Δ| over GW1-10 = **0.157 pts**, max **0.800 pts**
* rank correlation between the two boards = **0.9995**
* largest per-club mean effect = 0.073 pts over ten gameweeks

**Team level** — where the layer actually acts:

* mean |Δ attack strength| = **0.0030** log units (max 0.0062); |Δ defence| = 0.0032
* slope of fitted attack strength on market z: **+0.1789 with the market off, +0.1787 with
  it on** — a change of −0.0002

The posterior is no more market-aligned with the odds blended in than without.

**Why.** Two mechanisms, and the second dominates:

1. *Redundancy.* ClubElo and market Elo correlate **0.870 Pearson / 0.861 Spearman**. The
   odds mostly restate what ClubElo already says. Genuine disagreements exist — Tottenham
   16th on ClubElo vs 8th on the market, Chelsea 10th → 5th, Bournemouth 6th → 12th — but
   they are few.
2. *The prior is swamped by the likelihood.* The Elo path enters only as a **prior mean** on
   attack/defence with `prior_sd = 0.35`, against a likelihood of **380 matches (760 Poisson
   rows, 1043 goals)**. Sweeping `clubelo_weight` across its whole range — 0.00 to 0.95,
   i.e. from no Elo in the prior at all to a prior that is essentially pure Elo — moves
   fitted attack by a mean of only **0.013** log units, and the correlation with market z
   barely moves (0.885 → 0.892).

The decisive observation: fitted strength already correlates **0.885 with market z when
`clubelo_weight = 0.00`** — with no Elo, and therefore no odds, in the prior at all. The
market alignment is coming from the 25/26 match results themselves. The odds feed is
re-stating information the likelihood already contains.

This **revises** the claim in `PROJECT_KNOWLEDGE_2627.md` §5 that the blend "shifts team
strength toward the market". The market Elo it produces is sensible (Arsenal 1839 top, Hull
1390 bottom, as recorded); the *shift* is not measurable at the shipped weights.

To make betting odds bind, they have to enter the **likelihood**, not the prior — as
per-fixture supremacy/total-goals odds carrying an effective sample size, which is precisely
the attack/defence split already flagged as open item §6.1. Raising `market_weight` will not
do it; the sweep above shows the whole prior path is too weak to matter.

*(Level note: market Elo is centred at 1600 and ClubElo near 1830, so every blended value
falls. This is harmless — `bayes_model.py:119` normalises by the mean before use, so only
relative spread reaches the model.)*

### 5.2 The GW1-3 home discount — working exactly as specified — [VERIFIED]

| segment | away λ | home λ | gap |
|---|---|---|---|
| GW1-3 | 1.3967 | 1.4868 | **0.090** |
| GW4-10 | 1.2919 | 1.5853 | **0.293** |

Mean match total goals: **1.4417** (GW1-3) vs **1.4386** (GW4-10) — flat to 0.2%, confirming
the symmetric split preserves the measured total-goals null while reproducing the reduced
early home advantage.

### 5.3 Regime δ sensitivity — why it stays off — [JUDGMENT, quantified]

Turning on the unfitted δ mean-pulls (`REGIME=proposed`) against the κ-only headline:

* mean |Δ| = **0.537 pts**, max **5.518 pts**, rank correlation 0.9927
* the movement is almost entirely one club: **Joelinton −5.52, Schär −4.66, Botman −4.49,
  J.Murphy −3.58, Woltemade −3.25, Livramento −3.22, Thiaw −2.92** — Newcastle carries
  δ = 0.35, the harshest value in the table
* Chelsea (N.Jackson −5.10, Disasi −3.70) and Tottenham (Danso −3.48, Bentancur −3.30) follow

A 5-point swing on a £5.0m defender is decision-relevant, and it rests entirely on an
unfitted judgment constant. Holding δ at its no-op is the defensible default; the κ widening
still expresses the regime uncertainty, in variance where it belongs.

### 5.4 Point-source decomposition — [DERIVED]

Mean GW1-10 totals by channel:

| pos | total | clean sheets | DefCon | attack + appearance |
|---|---|---|---|---|
| DEF | 20.82 | 6.22 | 2.53 | 12.07 |
| GK | 14.18 | 6.23 | 0.00 | 7.95 |
| MID | 19.11 | 1.22 | 0.64 | 17.26 |
| FWD | 19.67 | 0.00 | 0.03 | 19.63 |

DefCon is a defender-and-holding-midfielder channel and nothing else. Its top earners are
concentrated at weak clubs — Targett (Hull) 11.41, Senesi (Spurs) 11.16, Egan (Hull) 10.04,
van Ewijk (Coventry) 9.92 — which is the xGA conditioning working as designed: bad defences
face more defensive actions. Note this pulls *against* clean-sheet value at the same clubs
(Targett 4.50 CS vs Senesi 9.62), so the two channels must be read together, and the
`defcon_matchups` finding stands: total defender EV still falls monotonically with fixture
difficulty.

### 5.5 Cold start, minutes and set pieces — [VERIFIED]

* **209 of 584** players are cold-start (no usable history), mean start probability 0.405 vs
  0.561 for established players; mean projection 14.39 vs 21.85.
* **375** players carry `exp_minutes` from their own history (mean 84.4 min), **209** fall
  back to the positional constant (85.3) — the shipped `minutes_persistence` result.
* Exactly **20** penalty takers across 20 clubs after the fix in §3.1/§3.2 (was 23).

---

## 6. Not included, and why

* **Event data (Understat / WhoScored).** `sd_ingest`, `setpiece`, `xg_calibrate` and
  `crosswalk` all import and self-test clean, but no scrape has been run, so nothing feeds
  the board. Unchanged from `SOCCERDATA_FINDINGS.md`.
* **Directional style priors.** Deliberately not built (§7). Style is collinear with strength
  in exactly the mismatched fixtures that carry leverage.
* **Rotation multipliers, mean reversion, deep history, age, congestion, tournament summers,
  churn, mid-table fade, European qualifying fade, late-form carryover.** All tested null.
  Not reintroduced.
* **`early_dispersion`.** Measured (+0.127 CI 0.018–0.235) but *not applied*, per its own
  pre-registered rule — it needs an out-of-sample board backtest, not a fourth in-sample slope.
* **Live lineups/injuries.** `lineups.py` exists; no API key or team-news file supplied, so
  availability comes from FPL status flags only.

---

## 7. Revisions this run makes to PROJECT_KNOWLEDGE_2627.md

1. **§5 "Market odds blend … shifts team strength toward the market"** — not measurable at
   the shipped weights. The blend changes fitted attack by 0.003 log units and the board by
   0.157 pts mean. See §5.1 above; the mechanism is diagnosed, not merely observed.
2. **§5 "Projected set-piece duty … Board: Szoboszlai +4.91, Kluivert +3.06 / Robinson −4.51,
   Hirst −2.04; mean ≈ 0"** — those figures were measured with the demotion bug present, so
   the losing side of each reassignment was understated. Isak alone moves **−3.70** over
   GW1-10 once he correctly loses Liverpool's penalty duty.
3. **§6.1** gains a sharper form: the remaining odds upgrade is not "a live cron" but
   *moving the odds out of the prior and into the likelihood*. A live feed on the current
   wiring would still move nothing.

