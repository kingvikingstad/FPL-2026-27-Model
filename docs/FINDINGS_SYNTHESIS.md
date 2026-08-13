# Findings Synthesis — 2026-08-09 → 2026-08-11

One file covering everything from the soccerdata integration through the deep-history and
mechanism work. The four detailed documents remain authoritative for method and caveats:
[SOCCERDATA_FINDINGS](SOCCERDATA_FINDINGS.md) ·
[DEEP_HISTORY_FINDINGS](DEEP_HISTORY_FINDINGS.md) ·
[TEAM_FIXTURE_FINDINGS](TEAM_FIXTURE_FINDINGS.md) ·
[PLAYER_LAYER_FINDINGS](PLAYER_LAYER_FINDINGS.md)

**Score:** 5 changes applied to the model, 9 tested nulls, 5 bugs fixed. Every applied
change was A/B'd against the board; every null is recorded with the evidence that killed it.

---

## 1. What changed in the model

| # | Change | Was | Now | Board effect |
|---|---|---|---|---|
| 1 | `revert` (team strength carryover) | 0.85 | **0.963** | strength spread ×1.11 |
| 2 | `home_prior` | 0.26 | **0.184** | folded into above |
| 3 | promoted priors (sd) | 0.30 | **~0.20** | folded into above |
| 4 | GW1–3 home discount | none | **−0.152**, split home/away | confined to GW1–3, ±0.09 |
| 5 | minutes exposure | 90 for every starter | **per-player `exp_minutes`** | MID −3.4%, FWD −4.2%, DefCon −13% |

**Combined:** board Spearman vs the pre-session model stays ≈0.99 — this is redistribution,
not inflation — while every applied change was validated on its own A/B.

### 1.1 Team hyperparameters, calibrated on 31 seasons

`history.py` already implemented these estimators and was **never wired in**;
`bayes_model` called the promoted priors "educated guesses" in its own comment. Validated
first on simulated data with known parameters (IV recovers `revert` to within 0.022 where
naive OLS is off by 0.158), then run on 1993–2025.

The big one is `revert` 0.85 → **0.963**: team strength carries over far more completely
than assumed, so the model was reverting teams toward the mean roughly four times harder
than the data supports. Home advantage was overstated ~40% and has declined
−0.0059/season. Promoted sides are *more* predictable than assumed, not less.

**Independent corroboration:** a completely separate analysis — Understat xG over 12
seasons, regressing next-season early form on full-season strength — gives a carryover
coefficient of **+0.926** against the IV estimate of 0.963 from football-data *goals* over
31 seasons. Different source, different metric, different method, 0.04 apart, and both far
from the 0.85 they replaced.

### 1.2 Home advantage is suppressed in GW1–3

Within-season log home advantage over matchdays 1–3 runs ~0.15 below the rest of the
season — 10/12 seasons down, robust to the baseline chosen (−0.152 vs md4+, −0.151 vs
md7+, −0.163 vs md20+).

Deliberately **one step, not a schedule**: md4–6 shows no significant discount (−0.071, CI
−0.155 to +0.023), the per-season trend spans zero, and the raw segment profile is
non-monotone (peaking md7–12, dipping md13–19). A multi-step schedule would encode noise.

**Split symmetrically**, which matters: the measured pattern is home goals *falling*
(−0.064 in logs) and away goals *rising* (+0.076) with the match total unchanged — itself a
firmly measured null. Shaving `h` alone would have dropped GW1–3 totals ~7% and
contradicted that null. Half goes to the home side, half as an away bonus; the expected
match total then moves −0.41%.

### 1.3 Minutes exposure — the largest correctness gain

The model drew minutes from a **two-point distribution**: every starter exactly 90, every
sub exactly 20. `m90` scales attacking involvement, penalty xG and DefCon counts, so any
gap between 90 and reality biases every projection in one direction.

Measured over 23,059 appearances — mean minutes *given* 60+ is **85.3, not 90**:

| pos | E[min given 60+] | inflation | finishes the match |
|---|---|---|---|
| GK | 89.9 | 0.1% | 99.5% |
| DEF | 87.5 | 2.9% | 81.8% |
| **MID** | **83.1** | **8.4%** | 52.2% |
| **FWD** | **81.5** | **10.5%** | 42.9% |

Then refined further from positional constants to **per-player**: conditional minutes is a
genuine player trait (reliability 0.819, persistence r=0.652, 0.796 disattenuated). Shrunk
player history beats the positional constant out of sample on MAE **1.968 vs 2.649** — a
25.7% gain, clustered CI (+0.579, +0.784) — and removes its +1.2 minute bias.

**DefCon fell 13.1%**, far more than the exposure change, through threshold convexity:
DefCon needs 10–12 actions and `P(Poisson(λ) ≥ 10)` drops steeply as λ falls. The channel
behind the cheap-defender strategy was materially over-valued. Clean sheets are untouched
(−0.06%, Monte Carlo noise) because `played60` is unchanged by construction.

Against Solio, **all four metrics improved** on the per-player step (Pearson 0.7936→0.7951,
Spearman 0.6055→0.6166, MAE 0.9465→0.9139, bias −0.882→−0.828).

---

## 2. Tested nulls — nine things that don't work

Recorded as prominently as the wins, per project convention. Each is a hypothesis that
looked plausible and died on evidence.

| Hypothesis | Result | Key number |
|---|---|---|
| **Deep player history** (8 extra seasons of minutes) | null | 0.0017 MAE vs existing baseline |
| **Late-season form** → next season's start | null | **+0.0000 r²** over full-season strength |
| **New-manager surge** persists | not supported | interaction collapses −0.83 (n=6) → −0.01 (n=46) |
| **Offseason hires** start better/worse | no early effect | +0.081/+0.075/+0.124 over first 3/6/12, only 12 borderline |
| **Major-tournament summers** hurt GW1–6 | null, underpowered | all p 0.25–0.99; MDE 0.34 goals/match |
| **Transfer churn** → worse results by month | null | +0.98 pts per SD of churn (t=1.08) |
| **Fixture congestion** (rest days) | null | **+0.00000 xG** per day of rest advantage |
| **Age** → minutes decline | null | incremental r² **+0.0027** |
| **Mid-table GW1–6 fade** | probable false positive | see §3 |

Two deserve emphasis:

**Deep history was the headline proposal and it failed.** vaastav's 8 extra seasons of
player-gameweek minutes buy 0.0017 MAE against the model's existing 24/25 baseline. Two
structural reasons: start rate is a property of a *role at a club*, not a player (Kelleher
0.94→0.50 on a move), and the denominator differs from the repo's per-match panel — 27.4%
of player-seasons have zero appearances yet contribute 38 "did not start" events. The
obvious fix makes it worse. The reusable win was the loader and the identity result below.

**Congestion and age are both nulls with tight intervals**, which is stronger than a null
with a wide one. Rest: ±0.004 xG per day, so a four-day swing sits inside 1% of a
team-match. Age: a 34- and 26-year-old with identical minutes records differ by **0.5
minutes** next season.

---

## 3. The mid-table fade — four mechanisms, all eliminated

The one significant early-season archetype effect: clubs finishing 7th–12th create ~7% less
xG in GW1–6 than their own season average. It absorbed four rounds of investigation.

| mechanism | verdict | evidence |
|---|---|---|
| harder opening fixtures | ruled out | opponent quality differs <1%; residual gap *larger* (−0.069) |
| squad churn | ruled out | no early-loaded effect on any churn measure |
| selling the best attacker | ruled out | mid-table are *mid-pack* on departures; orderings contradict |
| July–Aug European qualifying | ruled out | treated −0.065 vs untreated −0.069 *within* the bucket |

The European test was decisive: comparing clubs *inside* the prior-7-12 bucket that did and
did not actually play qualifying ties gives a difference of **+0.005**, CI (−0.202, +0.198).
And top-6 clubs who played August CL play-offs started **better** (+0.153 vs +0.055).

Combined with the effect failing a Bonferroni correction (p = 0.039 against 0.0125
required), the conclusion is that it is **probably a false positive**. No model change.

> **Methodological correction worth carrying forward:** leave-one-season-out stability was
> repeatedly cited as reassurance the effect was real. It is not evidence against noise — a
> fluke of the pooled sample is also LOO-stable, since LOO only shows no single season
> drives it, which is equally true of noise spread thinly across all of them.

---

## 4. Usable fixture signals (descriptive, not yet model inputs)

From 12 Understat seasons, classified only by **pre-season-knowable** information
(promoted status, prior-season finish) to avoid lookahead.

**xG created, row team vs column opponent** (1.00 = league mean):

| ↓team \ opp→ | top-6 | 7–12 | 13–20 | promoted |
|---|---|---|---|---|
| prior top-6 | 1.026 | 1.302 | 1.343 | **1.510** |
| prior 7–12 | 0.813 | 0.936 | 1.041 | 1.141 |
| prior 13–20 | 0.713 | 0.910 | 0.913 | 1.009 |
| promoted | **0.640** | 0.771 | 0.846 | 0.937 |

- **Games against top-six are the highest-scoring** (total xG index 1.044), not games
  against promoted sides (0.984). Top-six concede little but create so much the total rises.
- **Lowest-scoring environment is against prior 13–20** (0.971) — established bottom-half
  sides neither create nor collapse, and are routinely mispriced as "easy".
- **Clean sheets:** top-6 vs promoted 0.424; promoted vs top-6 0.091. High total goals does
  *not* rule out a clean sheet — the goals are one-sided. Read both matrices together.
- **Mismatch bites 3× harder early**: +0.185 goals per archetype rank-gap in GW1–6 vs
  +0.062 later.
- **Top-six gain ~50% more from home advantage** than anyone else (+0.269 vs ~+0.17) — but
  see §1.2, that edge is suppressed in GW1–3.
- **Attack persists more than defence** across seasons (r 0.765 vs 0.656), and it survives
  disattenuation (0.859 vs 0.794). Weight last season's attack over its defence; clean-sheet
  projection is intrinsically the harder half.
- **Promoted sides do not adapt** — attack creeps 0.735→0.784 while defence *worsens* and
  points/game fall. No payoff to waiting.

---

## 5. Bugs found and fixed

| Bug | Impact | Where |
|---|---|---|
| `solio_cache.md` destroyed on every run | cp1252 write truncated then raised, `except: pass` swallowed it | `solio_ensemble.py` |
| Season codes sorted lexicographically | 1990s sorted after 2020s; compared 93/94 to 25/26 as consecutive | `history.py` |
| Hardcoded sandbox paths | `build_all` blocked outright | `starter_prior`, `bayes_model`, `core_insights` |
| `exp_minutes` silently dropped | first A/B showed *exactly zero* difference | 6 runners + 4 tests |
| soccerdata drops Understat penalties | 92/season unlabelled; recovered under a guard | `sd_ingest.py` |

The cache bug is the instructive one: `open(cache,"w")` truncates *before* writing, and on
Windows encodes cp1252 — the feed contains U+2212 MINUS SIGN, so the write raised **after**
the file was already empty, and a bare `except: pass` hid it. Invisible on Linux. Now
UTF-8, atomic, and payload-validated.

The `exp_minutes` one is the transferable lesson: **a change that appears to do nothing is
more likely unplumbed than ineffective.**

---

## 6. Data infrastructure added

| Asset | What it gives |
|---|---|
| `src/sd_ingest.py` | Understat readers — shots, team-match, player-match, **player-season** |
| `src/fpl_history.py` | vaastav player-gameweek, 2016/17→, `player_code`-keyed |
| `data/team_hyperparams.json` | 31-season calibrated constants |
| `data/manager_changes.csv` | managerial changes, 2014→2026 |
| `data/transfer_counts.csv` | per club-season transactions, 12 seasons |
| `data/european_qualifying.csv` | July/Aug European ties by club-season |

**FPL's `code` IS `player_code`** — 841/841 exact on 25/26. The deep history joins to the
model with no fuzzy matching and no crosswalk; G5 is satisfied by construction. This does
*not* extend to Understat, which still needs the (unverified) crosswalk.

**Scraper reliability is poor and must be gated.** Transfermarkt returned a 2025/26 page
headed "2024/25"; Wikipedia falsely reported no English clubs in 2023/24 Conference League
qualifying when Aston Villa played the play-off, and contradicted itself on 2017/18. Every
scraped file now has a validation gate that refuses to run on a club-set mismatch. Only
club + date are used from the manager data — the transcribed *names* contain known errors.

**One tooling note:** prefer `understat_player_season` over `understat_player_match` when
season totals suffice — one request per season versus ~380. And FBref's soccerdata reader
installs a full browser-automation stack (`selenium`, `seleniumbase`, `PyAutoGUI`) for one
season per ~25 minutes; it was abandoned.

---

## 7. Current board (regenerated 2026-08-11)

**GW1–10 totals:**

| player | team | pos | cost | total |
|---|---|---|---|---|
| Haaland | Man City | FWD | 15.5 | 71.2 |
| B.Fernandes | Man United | MID | 12.0 | 55.3 |
| Mbeumo | Man United | MID | 8.0 | 53.4 |
| Saka | Arsenal | MID | 9.5 | 50.3 |
| Gabriel | Arsenal | DEF | 8.0 | 48.8 |
| Semenyo | Man City | MID | 8.5 | 48.8 |
| Palmer | Chelsea | MID | 9.5 | 47.1 |
| O'Reilly | Man City | DEF | 6.5 | 47.1 |

**DefCon value, <8% owned and ≤£5.5m** — note these are ~13% lower than before the minutes
fix, which is the correction landing where it should:

| player | team | cost | own | defcon_ev |
|---|---|---|---|---|
| Targett | Hull | 4.0 | 2.7 | 6.82 |
| Truffert | Bournemouth | 5.5 | 4.8 | 5.94 |
| Egan | Hull | 4.0 | 1.2 | 5.76 |
| Ballard | Sunderland | 5.0 | 3.8 | 5.41 |

Arsenal lead the clean-sheet ranking (9 strong fixtures in 10).

---

## 8. What's outstanding

1. **Hand-verify `data/crosswalk_review.csv`** — the Understat→`player_code` crosswalk gates
   any Understat-derived quantity reaching the model. Fuzzy rows are written
   `verified=False` and `load_crosswalk(strict=True)` hides them until a human signs off.
2. **Penalty coverage.** The declared `penalties_order` is validated as the better signal
   (85.7% precision vs 45.0% for measured history) — but it covers only **39%** of penalties
   actually taken, because ~6 clubs have no declared first-choice taker. At ~16 points a
   season per settled taker, the missing 60% is the largest single unexploited edge found.
   It needs a better source, not better modelling.
3. **§2.3 set-piece prior: do not build.** Component reliability is measured (set-piece
   ≈0.63, open-play ≈0.89) but persistence is *inconclusive* — the estimator saturates
   because the simulation lacks the real panel's persistent structural zeros.
4. **PPDA substitution.** Measured 25/26 PPDA correlates 0.80 with the hardcoded estimates
   but misplaces individual clubs by up to 3 (Crystal Palace 12.0 est vs 14.98 measured).
   Evidence is ready; the substitution touches the live DefCon channel.
5. **Archetype matrices as model input.** Currently descriptive. Natural use is as a sanity
   check on `TeamModel`'s fixture λ — do the model's own λ reproduce the 1.510 / 0.640
   corners?

---

## Appendix — reproducing

```bash
python scripts/build_all.py
python scripts/gw_board.py
python scripts/run_final_board.py
```

Escape hatches for every applied change: `FPL_TEAM_HYPER=guess` (team constants + GW1–3
home discount), `FPL_MINUTES_MODEL=flat` (original 90/20) or `=positional` (constants, no
player history).

Environment: `FPL_DATA` → FPL-Core-Insights `data`; `FPL_HISTORY` → vaastav `data`
(optional, deep history only).
