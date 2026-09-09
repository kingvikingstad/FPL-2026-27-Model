# docs/ — what to read, and what not to

28 files, ~63,000 words. Reading them in order is not a strategy; most of what is
here is the *record* of how a claim was established, not the claim itself. This
index routes by question, and marks what is stale so nobody spends context on it.

**If you read exactly two things:** `PROJECT_KNOWLEDGE_2627.md` (authoritative
context — it supersedes every other handoff) and `../CLAUDE.md` (the hard guards).
Everything below is the evidence behind those two.

---

## Route by question

| The question you have | Read |
|---|---|
| What is settled, what is open, what confidence? | `PROJECT_KNOWLEDGE_2627.md` |
| What was built when, and what was left out on purpose? | `INTEGRATION_LOG.md` |
| How do FPL points actually work? | `FPL_RULES.md` |
| Why is the team layer shaped this way? Archetypes, hyperparameters | `TEAM_FIXTURE_FINDINGS.md` (48 KB — the largest; read the section you need) |
| Minutes, congestion, penalties — the player layer | `PLAYER_LAYER_FINDINGS.md` |
| Does a start predict the next start? | `START_PERSISTENCE_2026-09-07.md` |
| Set-piece duty, DefCon matchups, early dispersion | `SETPIECE_DEFCON_FINDINGS.md` |
| DefCon by team/matchup, and the explosiveness null | `DEFCON_TEAM_EXPLOSIVENESS_2026-09-02.md` |
| Understat / shot-level integration | `SOCCERDATA_FINDINGS.md` |
| Deep history (2016/17→) — what it does and does not support | `DEEP_HISTORY_FINDINGS.md` |
| Regime change: why it widens κ and asserts no direction | `regime_weighting_note.md` |
| What the reduced-form baseline is, and the trap in testing against it | `REDUCED_FORM_NOTE.md` |
| Which external source contributes what, and what duplicates | `DATA_SOURCE_AUDIT_2026-08-19.md` |
| The Solio market feed — what it actually publishes | `SOLIO_MARKET_FEED_2026-08-27.md` |
| Live odds integration | `INTEGRATION.md` |
| Has the model ever been scored against a real gameweek? | `GW1_SCORING_2026-08-26.md`, then `GW2_REVIEW_2026-09-06.md`, then `GW3_REVIEW_2026-09-08.md`. The scored rows themselves are `predictions/scoring_ledger.csv` |
| What happened in a specific gameweek? | `GW1_REVIEW_2026-08-26.md`, `GW2_REVIEW_2026-09-06.md`, `GW3_REVIEW_2026-09-08.md` |
| Known correctness bugs and the tree cleanup | `CODE_AUDIT_2026-08-21.md`, `SIMULATION_FIXES_2026-08-24.md` |
| A synthesis across the first build phase | `FINDINGS_SYNTHESIS.md` |
| What a full pre-GW1 run looked like | `FULL_STACK_RUN_2026-08-19.md` |

## Measured non-issues

Things that look alarming, were measured, and are not urgent. Recorded so the next
reader does not re-raise them — and so the bound is re-checked if the setup changes.

| Concern | Measured | Evidence |
|---|---|---|
| `teams.csv` Elo is NULL upstream (verified on `origin/main`, will not self-heal), so the team layer runs on a pin dated 2026-08-14 containing no 26/27 result | Removing the Elo channel **entirely** — the upper bound on any error a stale pin can cause — clears the pre-registered rule (**<0.5 posterior SD**, Spearman **rho > 0.98**) with room to spare; the current numbers are in the CSV, and they move a little each time the feed does. It is a blend into the prior *means* at `bayes_model.py:166`, not a likelihood term, with 380 match rows on top. Re-check if `ELO_WEIGHT` rises above 0.45. | `studies/elo_pin_sensitivity.py` / `.csv` |

## The nulls, in one place

Tested, rejected, and **dead** — `CLAUDE.md` forbids reintroducing them under a new
name. The evidence, if you need to see it before you believe it:

| Null | Result | Evidence |
|---|---|---|
| Rotation multipliers | p=0.23 | `PLAYER_LAYER_FINDINGS.md` |
| Directional mean-reversion | p=0.69 | `PROJECT_KNOWLEDGE_2627.md` |
| Fixture congestion (GW1-26), three instruments | P(start) +0.001, CI ±0.02 | `FIXTURE_CONGESTION_2026-09-01.md` |
| Team explosiveness | r=+0.006 inside a simulated true-Poisson null | `DEFCON_TEAM_EXPLOSIVENESS_2026-09-02.md` |
| Four early-season scoring hypotheses | four nulls | `EARLY_SCORING_TRENDS_2026-08-20.md` |

GW27+ congestion is **untested, not null** — it needs knockout kickoff times and the
FA Cup, neither of which the data source carries.

## Do not read these

Superseded or historical. They are kept as a record of what was known on a date,
and quoting them will state something the project has since corrected.

| File | Status |
|---|---|
| `GW1_SCORING_2026-08-22.md` | superseded by `GW1_SCORING_2026-08-26.md` (interim, 6 of 10 fixtures, provisional bonus) |
| `context_handoff_2627_regime_defcon.md` | superseded by `PROJECT_KNOWLEDGE_2627.md` |
| `MISSING_MODULES_CHECKLIST.md` | resolved 2026-08-05; every module listed is present |
| `PATCHES.md` | historical; all three edits are pre-applied in the current tree |

---

## Where the non-prose answers live

Several questions have a *runnable* answer that is always current, and reading a doc
to answer them is how the answer goes stale:

- **"Is the tree current, and what do I run?"** → `.\fpl.ps1 doctor`
- **"What produces this file, and what does it need?"** → `.\fpl.ps1 graph`
- **"What must I rebuild after changing X?"** → `python src/manifest.py --downstream X`
- **"Does anything still work?"** → `.\fpl.ps1 test --quick`
- **"Where does this path resolve?"** → `.\fpl.ps1 paths`
- **"Where is the time going?"** → `.\fpl.ps1 run scripts/profile_pipeline.py`
  (`--full` also times each pipeline stage). Measured 2026-09-08: the 56-module import
  section costs ~216s, of which **163s is the unavoidable per-process pandas baseline**
  and only 53s is this project's code. The expensive item is process spawning, not any
  module — `captaincy`, the slowest, is 7.9s net and optimising it away would save under
  4% of that section.
