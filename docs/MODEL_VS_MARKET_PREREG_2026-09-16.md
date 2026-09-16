# Model λ vs market λ — does the team layer forecast realised goals better, worse, or no differently? Pre-registration, 2026-09-16

**Status: PRE-REGISTRATION. No outcome has been joined to either forecast in writing this.**
Nothing below is a result. The power figures come from a simulation on synthetic data
(§7). The only real inputs to it are the size of the model-vs-market gaps, which anyone
can see before kickoff.

Written by `research-preregistrar`. The owner writes this file into `docs/` word for word.
Anything changed after the first outcome-joined statistic is computed must go in §12
(Deviations), with its date. It must never be edited in place.

---

## 0. Premises audited before designing

1. **The two forecasts are not independent.** [VERIFIED] The model's team layer blends
   the 7 Aug outright market into Elo at `MARKET_WEIGHT=0.6` (`export_team_projections.py`
   l.104, `gw_board.py` l.414). That does not break the test, and §3 explains why. It does
   change what a result means.
2. **The model's pre-deadline per-fixture λ was never kept for GW2–GW4.** [VERIFIED] The
   locked boards are player boards. The current `team_projections_gw1_38.csv` rows for
   played weeks are today's posterior, and with `INSEASON=on` they have already absorbed
   those matches. §4 decides that no past week is admissible.
3. **The export and the board can compute different λ.** [VERIFIED by reading the source]
   `export_team_projections.py` hardcodes `clubelo_weight=0.45`, but `gw_board.py` reads
   `ELO_WEIGHT`. The export also ignores `INJURY_IMPACT`, `TEAM_OVERRIDES` and
   `SOLIO_MARKET`, all of which change the board's team samples. Those four are at their
   defaults today (0.45 and off), so today the two agree up to Monte Carlo error if `S`
   differs. The lock (§10) must record the flags and refuse if any of them is off-default.
   If it didn't, the locked λ could quietly differ from the λ the board used.
4. **`test_all.py` runs every study on every harness run.** [VERIFIED, l.278] A study
   script that joins outcomes would therefore compute and print the result every time
   anyone runs the harness. That is a way to peek, and §8 closes it.
5. **The routing of this item is stale.** [VERIFIED] `predictions/README.md` and
   `lock_board.py` cite "PROJECT_KNOWLEDGE §6.6" for scoring against real gameweeks.
   In the current PROJECT_KNOWLEDGE, §6.6 is the `player_code` dedup item. The live
   reference is CLAUDE.md open item 6. This study advances that item.

---

## 1. Question and estimand

**Population.** Premier League 2026/27 matches in gameweeks GW5 up to the stopping
gameweek (§8), restricted to matches that meet §4's admissibility rules. Each match gives
two team-fixtures, s ∈ {home, away}.

**Notation.** For team-fixture (m, s):
- y_ms = goals scored by that side, official full-time score, own goals included.
- λ^M_ms = the model's posterior-mean λ from the lock.
- λ^K_ms = the market's λ from the lock (§5).
- r_ms = log(λ^M_ms / λ^K_ms) is the model's log disagreement with the market.

**Primary estimand: the encompassing weight β.** β is the pseudo-true parameter of the
Poisson quasi-likelihood model

    E[y_ms | λ^K, λ^M] = λ^K_ms · exp(β · r_ms)          (no intercept)

- β = 0: the market encompasses the model. The model's disagreement carries no
  information about goals.
- β = 1: the model encompasses the market.
- β = ½: the two score equally. See the next line.

**How β relates to the literal question.** [DERIVED] Under the Poisson log score, log y!
cancels in a difference. The per-team-fixture score difference is therefore
D_ms = y·r − (λ^M − λ^K). It is linear in y, so its expectation depends only on the mean
of y. To second order in r, E[D_ms] ≈ λ·r²·(β − ½). "Model scores better / worse / the
same" is the same statement as β > ½ / β < ½ / β = ½, estimated from the same data with
the same noise. β also answers the two encompassing questions, and those are the ones
this season can power (§7). So β is primary, and the mean score difference D̄ is reported
beside it.

**Reported estimand, no decision attached.**

    D̄ = (1/N_m) Σ_m (D_m,home + D_m,away)

This is the mean paired Poisson log-score difference per match, model minus market, in
nats. Positive means the model scored better.

**Why this scoring rule.** [JUDGMENT]
- **Proper.** The Poisson log score is strictly proper for a Poisson forecast.
- **Like for like.** Both sides are scored as a point λ through the same Poisson
  likelihood. The market provides only a point λ. The posterior mean is the log-score-
  optimal point summary of the model's λ posterior, because a Poisson mixture has mean
  E[λ].
- **Match level handles within-match dependence.** Under the independent-Poisson
  assumption both sides make, the joint scoreline log score is the sum of the two
  team-fixture scores.
- **Mean only, and robust to dispersion.** E[D] depends on the mean of y alone, and the
  quasi-likelihood β with a sandwich SE stays consistent under under-dispersion. That
  matters because the league's upper tail is thinner than Poisson (`team_explosiveness`).
  This study makes no second-moment claim.

**Scores considered and demoted to secondary (§9).**
- **Posterior-predictive log score** (the model scored as a Poisson mixture over its
  posterior draws). The market has no posterior to set against it. The comparison mixes
  up location with the model's parameter uncertainty. And given the thinner-than-Poisson
  tail, the extra variance has a sign we can predict in advance. It is reported with no
  action attached.
- **Ranked probability score on 1X2.** It coarsens the scoreline to three categories, so
  it has strictly less information. The market's 1X2 would have to be rebuilt from λ
  anyway, because Solio publishes only λ. It is reported with no action attached.
- **Scoring against xG.** Not done. xG is not what pays FPL points, and the model is
  trained on xG, so the comparison would be tilted towards the model.

## 2. Instrument, and what it is confounded with

| Quantity | Source | Confounds, named in advance |
|---|---|---|
| λ^M | `export_team_projections.py` posterior mean `lam_for`/`lam_against`, written to the team lock (§10) at lock time T_lock ≤ deadline | Contains 0.6×(7 Aug outrights) in Elo space. Contains in-season xG through GW t−1 (`W_MATCH=1`). Does NOT contain lineups (`INJURY_IMPACT` off). Changes to the team-layer spec during the study move the estimand (§11). |
| λ^K | `fixture_market.attach` at T_lock: the last price at or before T_lock. Solio `prGoalsFor/Against` first; otherwise football-data.co.uk, Shin de-vig plus Poisson inversion. | By construction it contains team news up to T_lock, which the model's team layer does not. Solio's λ is "the market as Solio reports it" (§2 of the Solio doc). The books inversion assumes Poisson, and its margin handling can bias the level. Precedence was fixed on 2026-09-11 and is not revisited here. |
| y | Feed `By Gameweek/GW*/matches.csv` via `inseason.played(require_xg=False)`: `home_score`, `away_score`, finished guard per match | League matches only [CHECK that `played` filters to `-prem-` ids, as `reconstruct_e0` does]. A match played outside its scheduled gameweek is excluded (§4). |

**Consequence for interpretation.** β > 0 can come from the model's own content: xG
reconstruction, ClubElo, in-season xG, travel, the home discount. β < ½ can come from the
market's team-news information, which the team layer structurally lacks. A result is
therefore about **the team layer as deployed against the price at the same moment**. It is
not about "strength estimation" in the abstract. §9 runs a sensitivity using the market
price at the deadline.

## 3. Identification

- **Shared market content cancels in r.** The 7 Aug outright component enters both
  forecasts, so it drops out of r, which contains only the content where the two differ.
  Encompassing tests are designed for correlated forecasts, and correlation does not bias
  β. [DERIVED] What sharing does do is shrink the spread of r (σ_r), and that costs power
  (§7).
- **What β measures, and what it does not.** β is the weight on the model's
  *differential* content, with `MARKET_WEIGHT` held at 0.6. It does **not** identify the
  optimal `MARKET_WEIGHT`, because that knob blends a stale outright, not the current
  price. Reading β as "set `MARKET_WEIGHT` to 1−β" is a category error and is forbidden
  in §6.
- **No team fixed effects, no style block.** `check_identification()` does not apply:
  there is no team-style regression. It has one regressor (r) and an offset.
- **`beats_the_market` / `market_score_test`: does it apply?** Only as a partial
  cross-check. It is not the decision instrument.
  - The hypothesis β = 0 is algebraically the Poisson score test that `market_score_test`
    runs, with z = r.
  - That function, however, standardises (centres) z. Centring throws away the level
    component of r, i.e. the model's overall bias against the market, which is part of the
    literal question. It also uses |z| > 2.5, which is not this study's error control.
  - It tests one boundary of a three-way question and says nothing about β = ½ or β = 1.
  - It is a mean test, and this is a mean claim, so using it is not a category error. It
    just answers a narrower question.
  - The CLAUDE.md gate governs a signal **entering `TeamModel`**. This study puts nothing
    into `TeamModel` under any outcome, so the gate is never triggered. The guard that
    matters is the double-count guard (§6).

## 4. Admissibility: which weeks and fixtures count

**GW2, GW3 and GW4 are inadmissible, and they will not be reconstructed.** Reasons:
1. **No team-λ lock exists.** A rebuild has to pin the code commit, the feed commit and
   the flag vector. The tree was routinely dirty at lock time (see the 2026-09-08 commit
   "Commit the uncommitted tree"). `predictions/README.md` already classes current-code
   rebuilds (`_upto` boards) as "not forecasts".
2. **The spec to reconstruct is ambiguous for GW4.** Travel was ON by default from
   2026-09-11, yet PROJECT_KNOWLEDGE records that "the GW4 deadline lock predates the
   change". Team-layer code changed again on 2026-09-14 (`e105a30`, home-term routing),
   and `src/inseason.py` / `export_team_projections.py` carry uncommitted diffs. [VERIFIED
   via git]
3. **Verifying a rebuild against the locked board is not decisive.** `cs_ev` also depends
   on live availability and predicted-XI inputs that cannot be reproduced.
4. **Contamination.** GW4's model-vs-market gaps were examined before the deadline, and its
   scorelines are public. Deciding now whether to include a week whose outcomes are known
   is exactly the choice this register exists to take away.
5. **The gain is immaterial.** 30 matches would add about 13% to a final sample of about
   230 (§7). No power band in §7 changes.

**GW5 is the first admissible week, if and only if** its team lock (§10) is written
before 2026-09-18T17:30Z.

**Fixture-level exclusions.** These are mechanical, fixed now, and applied before any
outcome is joined:
- no team lock for that gameweek, or a lock written after the deadline;
- a lock whose recorded flags have `SOLIO_MARKET=on` (the model has then ingested the
  per-fixture market), `INJURY_IMPACT=on`, `TEAM_OVERRIDES=on`, or `ELO_WEIGHT` ≠ 0.45
  (the export would not match the board, §0.3);
- `mkt_lam_for` or `mkt_lam_against` missing for the fixture;
- the fixture was not finished inside its scheduled gameweek (postponed, abandoned or
  rescheduled);
- a double-gameweek fixture where either λ is missing or ambiguous. `fixture_market`
  already drops ambiguous Solio records, and books fill if they can.

Every exclusion is logged with its reason in the study CSV.

## 5. Which market price

- **Primary:** the `mkt_*` columns frozen *inside the team lock* at T_lock. The model and
  the market are then cut at the same moment.
- **Sensitivity (§9):** the last price at or before the deadline, re-derived later from
  the snapshot stores.

## 6. Decision rule, fixed now

**Estimator and error control.**
- β̂: Poisson quasi-likelihood with offset log λ^K, no intercept, solved by Newton's method.
- Sandwich SE clustered by match:

      V̂ = (Σ r²λ̂)⁻² · Σ_m (Σ_s r_ms(y_ms − λ̂_ms))²

- D̄ is reported with SE = sd(D_m)/√N_m.
- Inference uses **repeated confidence intervals** (Jennison–Turnbull) with O'Brien–Fleming
  critical values for two looks at information fractions ½ and 1: **z₁ = 2.797,
  z₂ = 1.977**. Simultaneous two-sided coverage is 95% across both looks and all claims
  read off the interval. [DERIVED: standard OBF constants for K=2, α=0.05]

**Robustness requirement.** A classification counts only if it is the same under the
match-clustered SE and under an SE clustered by gameweek. The gameweek clustering covers
league-wide weekly goal shocks, which load on the level of r. If the two disagree, the
weaker of the two classifications is recorded. There are no other forking paths: the
secondary specifications in §9 never change the classification.

**Classification of the final repeated CI for β, with its action:**

| Region | Reading | Action in the repo |
|---|---|---|
| **A.** CI entirely < ½ | Market scores better | Nothing enters `TeamModel`. Record in PROJECT_KNOWLEDGE §5. Promote PROJECT_KNOWLEDGE §6.1 (attack/defence split from current match odds, replacing the stale outright blend) to the next team-layer item. Open a **new** pre-registration for a post-fit per-fixture geometric combination λ = (λ^K)^(1−w)·(λ^M)^w, off by default, evaluated on weeks *after* this study's window. Open the `W_FIXTURE × MARKET_WEIGHT` joint sweep (Solio doc §9.3) as its own pre-registration. |
| **B.** CI entirely > ½ | Model scores better | No change to the model; it is already the deployed forecast. Extraordinary against a near-deadline price, so a `stats-referee` pass on lock timing and leakage is **mandatory** before the result is written into PROJECT_KNOWLEDGE. Lowering `MARKET_WEIGHT` on this evidence is forbidden (§3). |
| **C.** CI contains ½, excludes 0 and 1 | Each carries information the other lacks | Same combination pre-registration as A. Neither side is declared better. |
| **D.** CI contains ½ and 0, excludes 1 | **Informative null:** the model does not encompass the market, and its differential content is not shown to carry information | Recorded as a null in PROJECT_KNOWLEDGE §5. Promote PROJECT_KNOWLEDGE §6.1 as in A. No combination pre-registration. |
| **E.** CI contains ½ and 1, excludes 0 | The model adds information; the market does not encompass it | Record. No action. |
| **F.** CI contains 0 and 1 | **Uninformative** | Recorded as an uninformative null. It must **not** be written up as "no different". No re-run under a new instrument this season. |

**Rules that hold whatever the outcome.**
- **Nothing enters `TeamModel` under any outcome.** Per-fixture market λ in the fit double
  counts against `MARKET_WEIGHT`, and `solio_market.stack_e0` refuses it.
- `beats_the_market` is not reinterpreted as passed or failed by this study.
- "No different" means **equivalence**, i.e. a CI inside [0.25, 0.75]. It cannot be reached
  this season (§7), and the study does not claim it under any outcome.

**Interim look.**
- **When:** at information fraction ½ (§8).
- **Only stopping rule:** if the RCI at z₁ = 2.797 already lies entirely on one side of ½
  (region A or B), the study stops and that classification is final.
- **What the interim does not do:** no other interim reading triggers any action or any
  write-up. In particular, an interim CI that excludes 1 does **not** open region A's or
  D's follow-ups, because doing so would change the deployed team layer mid-study and
  shift the estimand.
- **No futility stop.** A null needs the full information to be informative.

## 7. Power

**Setup.** [DERIVED; simulation on synthetic data]
- 10 matches per gameweek.
- Market λ lognormal around 1.50 (home) and 1.20 (away), sd(log) 0.30.
- r ~ N(−0.03, σ_r) independently per team-fixture.
- Truth: log λ = log λ^K + β·r.
- Independent Poisson goals. 1,000 replications per cell. Fixed-sample 5% two-sided tests.

**Calibrating σ_r.** The disclosed gaps (λ MAE 0.107 on GW4 and 0.178 on GW5) correspond
to roughly σ_r ≈ 0.10–0.17. Using them is legitimate because gap size is known before
kickoff and says nothing about outcomes.

| matches | σ_r | SE(β̂) | P(exclude 1 \| β=0) | P(exclude 0 \| β=1) | P(exclude ½ \| β=0) | P(D̄ sig. > 0 \| β=1) |
|---|---|---|---|---|---|---|
| 100 | 0.10 / 0.13 / 0.17 | 0.56 / 0.44 / 0.34 | 0.43 / 0.59 / 0.82 | 0.39 / 0.60 / 0.83 | 0.15 / 0.22 / 0.32 | 0.13 / 0.19 / 0.28 |
| 200 | 0.10 / 0.13 / 0.17 | 0.40 / 0.31 / 0.24 | 0.72 / 0.85 / 0.98 | 0.70 / 0.88 / 0.98 | 0.28 / 0.36 / 0.52 | 0.25 / 0.38 / 0.50 |
| 340 (GW5–38) | 0.10 / 0.13 / 0.17 | 0.31 / 0.24 / 0.19 | 0.90 / 0.99 / 1.00 | 0.88 / 0.98 / 1.00 | 0.39 / 0.52 / 0.75 | 0.35 / 0.53 / 0.75 |

Size under β = ½ held at 0.02–0.03 for D̄ and 0.05–0.06 for β.

**What the study can and cannot deliver.**
1. **Can** produce an informative encompassing verdict (exclude 1, or exclude 0) with
   about 90% power once SE(β̂) ≤ 0.30. At z₂ = 1.977 and β = 0: P(Z > 1.977 − 1/0.30) ≈ 0.91.
2. **Cannot** reliably settle "better vs worse" this season. Even at the extremes, where
   one side *is* the truth, power to exclude ½ is 0.39–0.75 over the whole remaining season.
   For intermediate β it is lower.
3. **Cannot** establish equivalence. That needs SE ≈ 0.13, about 1,300 matches at
   σ_r = 0.13, which is more than three seasons.

**Prior expectation.** [JUDGMENT] Near-deadline prices are hard to beat, so the likeliest
informative outcome is region A or D. That is a result worth having: it decides whether
the model's differential content earns any per-fixture weight, and it sets the priority of
§6.1.

**Gameweek clustering costs power.** The robustness requirement in §6 means the
gameweek-clustered SE must also agree, and the table above does not account for that
[CHECK]. The information target is set with margin for it.

## 8. Stopping, and a ban on peeking

**Information is measured without outcomes.**

    I = Σ r²_ms · λ^K_ms

summed over admissible team-fixtures, using lock files only. 1/√I approximates SE(β̂)
under the Poisson model. Anyone may compute I at any time.

**Targets.**
- **Final look:** the first completed gameweek at which **I ≥ 11.1** (SE ≈ 0.30).
- **Interim look:** the first completed gameweek at which **I ≥ 5.55**, and only if at
  least 80 admissible matches have accrued by then.
- **Hard stop:** if I < 11.1 after GW38, the final look runs at GW38 with z₂. The CI will
  show the loss of power, and region F is the likely outcome.
- **Expected final gameweek:** [DERIVED] σ_r ≈ 0.17 → ~140 matches, ≈ GW19;
  σ_r ≈ 0.13 → ~230 matches, ≈ GW28; σ_r ≈ 0.10 → ~380 matches, so the GW38 cap applies.

**Ban on peeking.**
- No statistic that joins outcomes to both forecasts may be computed before a look. That
  includes D̄, β̂, calibration plots of the two λ, and "how did the market do on Arsenal".
- `studies/model_vs_market.py` must refuse to join outcomes unless I is at or past the
  next look's threshold and that look has not yet been taken. Otherwise it prints only I,
  the admissible match count and the next threshold, and exits 0. This is what makes it
  safe for `test_all.py` to run the study every time (§0.4).
- Each look writes a one-way marker (`studies/model_vs_market_looks.json`) so no look can
  be repeated.
- `postgw_review.py`, `forecast-scorer` and the explorer must not show any model-vs-market
  comparison on realised goals. Model-vs-market **gaps** on unplayed weeks stay allowed:
  they involve no outcomes.
- Any breach is logged in §12 with its date and what was seen. If a breach happens before
  the final look, the study is downgraded from confirmatory to exploratory.

## 9. Secondary analyses

All are pre-specified, all are reported, none can change the classification in §6, and
none has an action attached.

1. **D̄**, with SE. It is also broken down by gameweek, as a table and not as a test.
2. **β with an intercept:** log λ = log λ^K + α + β·r. α soaks up any market level bias
   (for example from the books inversion), so β comes from variation in r only.
3. **Market price at the deadline** instead of at the lock (§5).
4. **Solio-sourced fixtures only.**
5. **Gameweek fixed effects** in the encompassing model, which removes the weekly level of
   r. This is a different estimand and is labelled as one.
6. **Posterior-predictive log score:** the model as a Poisson mixture over its locked
   draws, against the market's plain Poisson. Descriptive only (§1).
7. **RPS on 1X2**, both sides built from independent Poisson. Descriptive only.
8. **Spec epochs.** β estimated separately for each epoch whose flags or team-layer
   commit differ (§11).
9. **H2 — DATA-SUGGESTED, flagged as such.** Does the market or the model have the
   strength dispersion right?
   - Split r into r_∥ (its within-gameweek projection on centred log λ^K) and r_⊥, and
     estimate β_∥ and β_⊥.
   - **Provenance:** this hypothesis was selected *after* the owner saw the GW5 gaps: the
     model below the market for big-club attack, above it for Newcastle v Hull. It has one
     anchor that predates that observation: `early_dispersion` (the model under-disperses
     strength in GW1–6, slope +0.101, CI +0.002..+0.206). That anchor is weak — it "barely
     clears zero", it is the third test of the same question, and it covers a different
     window.
   - **Consequence:** H2 is not primary, spends none of the primary α, and triggers
     nothing. Even a CI far from ½ only becomes a hypothesis for a fresh registration
     next season, or an input to the out-of-sample backtest that `early_dispersion`
     already calls for.

## 10. What must be locked each gameweek, and who does it

**Files.** Written together, before the deadline, from the same tree and feed as the board
lock.

1. `predictions/gw{N}_team_locked_{YYYY-MM-DD}_deadline.csv`
   - **Rows:** the 20 team-fixtures of GW N (more in a double gameweek).
   - **Columns:** `gw, team, opponent, is_home, lam_for, lam_against, lam_for_p5,
     lam_for_p95, lam_against_p5, lam_against_p95, p_clean_sheet, p_clean_sheet_plugin,
     p_win, p_draw, p_loss, mkt_lam_for, mkt_lam_against, mkt_p_clean_sheet, mkt_source,
     mkt_as_of`.
   - **Provenance columns:** `locked_at_utc, deadline_utc, repo_head_sha, repo_dirty,
     repo_diff_sha256, feed_commit_sha, draws_S, seed`, and the flag vector
     `MARKET_ODDS, MARKET_WEIGHT, ELO_WEIGHT, INSEASON, INSEASON_UPTO, INSEASON_W_MATCH,
     INSEASON_W_PROMOTED, FPL_TRAVEL, INJURY_IMPACT, TEAM_OVERRIDES, SOLIO_MARKET`.
   - **Name:** it deliberately does **not** match `gw*_board_locked_*`, so `score_gw._locked()`,
     `doctor` and `postgw_review` can never pick it up as a board.
2. `predictions/team_draws/gw{N}_team_draws_{YYYY-MM-DD}.csv.gz`
   - **Rows:** one per (home, away, draw), carrying `lam_home, lam_away`: 10 × S rows.
   - **Purpose:** the paired draws needed for secondary 6.
   - **Location:** a subdirectory, for the same glob reason `crosswalks/` exists.

**Who.**
- **GW5:** the owner, before **2026-09-18T17:30Z**. Run `export_team_projections.py` with
  the board's environment, copy the `gw==5` rows and the provenance into file 1, and
  regenerate the draws with the same seed into file 2. Before writing, check that
  `ELO_WEIGHT` is unset or 0.45 and that `INJURY_IMPACT`, `TEAM_OVERRIDES` and
  `SOLIO_MARKET` are off.
  - If the flag check fails, the lock is still written, but marked `inadmissible_flags`.
  - If the GW5 lock misses the deadline, GW5 is excluded and the study starts at GW6.
    No reconstruction.
- **GW6 onward:** `scripts/lock_board.py` calls a new `scripts/lock_team.py` straight after
  its board rebuild, in the same run. It inherits the refusals `lock_board` already has:
  never after the deadline, never overwrite, rename anything late to `_LATE_`. The GW2/GW3
  precedent shows a manual step gets missed.

**Also required, but not by the lock.**
- Put `python src/fixture_market.py --fetch` on the scheduled task (Solio doc §9.4). This
  is an owner decision because it edits a standing job. Without it, a fixture Solio did not
  price is lost for good, which is outcome-independent missingness but lost n.
- Consider moving the Solio poll to 2h (Solio doc §9.1).
- `studies/model_vs_market.py` ships `--selftest` built on the §7 simulation. It checks
  that β is recovered when β = ½, and that repeated-CI coverage is ≥ 0.93 over 500 synthetic
  seasons.
- Register its CSV and the looks marker as nodes in `src/manifest.py`.

## 11. Spec drift during the study

The object under evaluation is **the team layer as deployed at each lock**.
- Changes to the team layer that are unrelated to this study are allowed. They are
  recorded through the lock's provenance and become an epoch in secondary 8.
- Changes *motivated by* model-vs-market gaps are **frozen until the final look**: no
  `MARKET_WEIGHT` sweep, no `ELO_WEIGHT` change, no per-fixture market term. Otherwise the
  study would be evaluating a moving target that is steering itself toward the market.
- GK saves (CLAUDE.md open item 0) do not touch team λ and are unaffected.

## 12. Deviations log

(Empty at registration. Append only, dated, and state whether any outcome-joined
statistic had been computed at the time.)

**2026-09-16 — registration-time clarifications from implementing §10. No outcome-joined
statistic had been computed; no GW4 or GW5 outcome had been joined to either forecast.**

1. **Which team lock is used when a gameweek has more than one.** The text names only the
   `_deadline` file. Fixed now, before any lock is scored, mirroring
   `score_gw._locked()`: a `gw{N}_team_locked_*_deadline.csv` lock ALWAYS wins when one
   exists; an `_early` insurance lock is used only if no `_deadline` lock exists for
   that gameweek; a `_LATE_not_a_prediction` file is never used. The choice is made by
   filename, never by comparing the locks' contents.
2. **Draws filename carries the label:** `team_draws/gw{N}_team_draws_{date}_{label}.csv.gz`,
   not `gw{N}_team_draws_{date}.csv.gz`. An early and a deadline lock taken on the same
   date would otherwise collide, and the lock step refuses to overwrite.
3. **The lock step exists and is scheduled.** `scripts/lock_team.py` writes both files
   from one `export_team_projections.build(keep_draws_gw=N)` call, inheriting
   `lock_board`'s refusals. `lock_board.py --auto` (Claude scheduled task
   `fpl-lock-board`) calls it straight after its board lock, and retries it on a later
   in-window run if it failed. Flags off-default are written with `admissible =
   inadmissible_flags: ...` rather than refused, per §10.
4. **GW5 insurance lock written 2026-09-16T11:40:35Z**, 53.8h before the deadline:
   `predictions/gw5_team_locked_2026-09-16_early.csv`, 20 team-fixtures, market lambda
   on all 20 (Solio, as of 2026-09-16T11:04Z), S=3000, seed 7, `admissible=ok`,
   `repo_dirty=1` (concurrent sessions' uncommitted work; the diff hash is recorded).
   Verified outcome-free: the draws' means reproduce the locked `lam_for`/`lam_against`
   exactly on all 10 fixtures. The primary `_deadline` lock is expected from `--auto`
   inside 26h of 2026-09-18T17:30Z.
5. **§10 "owner decision" on the books fetch is taken:** scheduled 2026-09-16 (`3ffb7bc`).
6. **Not yet built:** `studies/model_vs_market.py` (§8's outcome-join refusal, looks
   marker, §7-based selftest) and its manifest nodes. No study script exists, so nothing
   can join outcomes yet; it must exist, with the refusal, before the interim threshold
   I ≥ 5.55 can be reached.

## 13. Contamination disclosed at registration, and its consequences

- **What was seen:** model-vs-market λ gaps on unplayed weeks — GW4 MAE 0.121 then 0.107;
  GW5 MAE 0.178, bias −0.046, with its directional pattern — and the Solio-vs-books MAE
  (0.032 / 0.055).
- **What that affects:** these involve no outcomes, so they cannot bias β̂. They could bias
  *hypothesis selection*. That is handled three ways: the primary hypothesis is two-sided
  and symmetric; the directional hypothesis is demoted to H2 and flagged DATA-SUGGESTED;
  and source precedence was fixed on 2026-09-11 and is not revisited.
- **What was used:** the size of the gaps sets σ_r in the power calculation. That is
  legitimate, because σ_r can be observed before any outcome.
- GW4's public scorelines are one of the reasons GW4 is inadmissible (§4).

## 14. If it fails

- **This study adds no model component, so no model code is deleted.**
  `studies/model_vs_market.py`, the team lock step and `predictions/team_*` stay whatever
  the outcome. They are also what CLAUDE.md open item 6 needs.
- **Region D or F:** the null goes into PROJECT_KNOWLEDGE §5 (validations) under "Model vs
  market, scored". A results appendix is added to this document, below §12, and the text
  above it is never edited. The GW4/GW5 "Model vs market" entries in §5 get a pointer
  saying the gaps have now been scored.
- **Region F specifically:** the finding recorded is "uninformative at SE X; would need
  about N matches", **not** "no different".
- **Any region:** the explorer tooltip's "inside/outside the model's 90% band" marker
  stays a display. It must never be relabelled as evidence about which side is right
  unless the result is region B.
