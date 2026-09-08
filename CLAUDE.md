# CLAUDE.md — working agreement for this repo

Hierarchical Bayesian FPL projection system for 2026/27. It estimates the underlying
events (minutes, team strength, attacking share, clean sheets, defensive contributions)
and composes them through the FPL scoring rules into posterior-predictive point
distributions, then into decisions.

**Read these before proposing anything:**
- `docs/PROJECT_KNOWLEDGE_2627.md` — authoritative context: settled decisions,
  validations with confidence, open questions, corrections. Supersedes all other handoffs.
- `README.md` — the map and the how-to-run.
- `docs/INDEX.md` — routes the other 26 docs by question, and marks the four that are
  superseded. Read it instead of opening files to find out what they are; `docs/` is
  ~62k words and most of it is the record of how a claim was established, not the claim.
- `docs/SOCCERDATA_FINDINGS.md`, `docs/TEAM_FIXTURE_FINDINGS.md`,
  `docs/PLAYER_LAYER_FINDINGS.md` — the evidence behind the guards below.

**Run `.\fpl.ps1 doctor` first.** It answers, in one call, what the paths resolve to,
whether the external feed has moved since the last deadline, which artifacts are
missing / schema-broken / empty / stale, and the ordered list of scripts that makes
the tree current. Answering that by inspection means four commands and a guess about
dependency order.

---

## Register

This is a research instrument, not a feature backlog. Hypotheses are pre-registered,
decision rules are fixed before results are seen, and a result that fails its rule is
recorded as a null with the code path deleted. **Nulls are deliverables.**

- Lead with what is *wrong or stale* in a proposal before implementing it. Auditing the
  premises is part of the job, not a detour.
- When a proposed method is statistically biased, replace it and document why. Fidelity
  to a spec does not outrank correctness.
- Explain in terms of estimators, bias direction and power — not code structure.
- Tag claims `[VERIFIED]` / `[DERIVED]` / `[JUDGMENT]` / `[CHECK]`.
- Prefer simulation-based validation over assertion.

---

## Hard guards — do not violate without saying so explicitly

| Guard | Why |
|---|---|
| **Tested-null heuristics stay dead.** Rotation multipliers (p=0.23) and directional mean-reversion (p=0.69) were tested and rejected. **Fixture congestion** was tested three ways — rest differential, July/Aug European participation, and actual midweek fixtures by recovery day (`fixture_congestion`, GW1-26: P(start) +0.001, CI ±0.02) — and is dead for GW1-26. | Do not reintroduce them under a new name. GW27+ congestion is untested, not null: it needs knockout kickoff times and the FA Cup, neither of which the data source has. **Team explosiveness** (per-club goal dispersion, return concentration) is dead too — `team_explosiveness`, reliability r=+0.006 inside a simulated true-Poisson null. |
| **Style main effects are unidentifiable under team FE.** `style_matchup.check_identification()` gates any team-style regression at a 5% residual-variance threshold. | Do not relax it. |
| **Market orthogonality.** The clean-sheet engine is validated at GA r=0.89 / CS r=0.93. | Any new team-level signal must clear `style_matchup.beats_the_market()` before entering `TeamModel`, or it double-counts what the odds already price. **Scope:** that gate is a Poisson score test on the MEAN, so it is structurally blind to second-moment claims (tail/dispersion) — a distribution can match the market's mean exactly and still have the wrong tail. Validate those against out-of-sample scorelines instead (`team_explosiveness`); invoking `beats_the_market` for them is a category error. |
| **A high-cardinality grouping explains any variance.** club x opponent is 380 cells on ~3,000 DefCon rows. | Report R2 against a PERMUTATION baseline, and permute the OUTCOME, not the labels — shuffling labels breaks the balanced fixture design and yields a null ABOVE the observed value. See `defcon_team_matchups.perm_r2`. |
| **Join on `player_code`.** | FPL reassigns `player_id` between seasons, and 15 `web_name`s in the 26/27 squad belong to players at two clubs. Never join on name. |
| **Regime change is a variance statement.** It widens uncertainty (κ). | It does not assert which way returns move. δ mean-pulls are unfitted and off. |
| **`defensive_contributions` is 100% null in 24/25.** | Never `fillna(0)` it — that dilutes every pooled DefCon rate. Exposure for DefCon is 25/26 minutes only (`mins_dc`). |

---

## Conventions, to follow mechanically

- `src/` stays **flat and importable** — engine modules import each other by name and must
  share one path. Do not package it into subdirectories.
- Runners live in `scripts/`, acceptance tests in `tests/`, studies + their evidence CSVs
  in `studies/`, one-off scenario runs in `outputs/plans/`.
- **Every path goes through `src/config.py`.** No literal path built anywhere else.
- **Every durable artifact is a node in `src/manifest.py`.** `config.py` says where a
  file lives; the manifest says which script writes it, what it reads, which columns
  must be present and non-null, and what the row floor is. A new artifact that is not
  registered makes `doctor` report a stale tree as clean — worse than having no check.
  Staleness is derived from mtimes and is transitive; there are no age thresholds.
- **The external feed is not sorted.** `gameweek_summaries.csv` in 26/27 has GW15 at
  row 0 and GW3 at row 29. Any `.iloc[0]` / `.head(1)` / `first()` on feed data without
  an explicit sort is a bug that returns a plausible wrong answer.
- Every module ships `--selftest`, offline, on synthetic fixtures.
  `scripts/test_all.py` discovers them automatically — do not add a list to maintain.
- Bracket indexing `d['col']`, never attribute access `d.col`, on frames that cross a
  module boundary.
- New components ship off-by-default or as validated corrections.

---

## Environment

- Windows 10 / PowerShell. **Python 3.13 is not on PATH** as `python` or `py`:
  ```
  $env:LOCALAPPDATA\Programs\Python\Python313\python.exe
  ```
- numpy / pandas / scipy / sklearn / openpyxl present. HiGHS comes in with scipy
  (`scipy.optimize.milp`) — the solver needs no new dependency.
- `FPL_DATA` points at the cloned `olbauday/FPL-Core-Insights` `data` dir; everything
  else auto-detects. `python src/config.py` prints the resolved paths.
- **Piped stdout on Windows is cp1252, the console is UTF-8.** A script printing `Δ`, `→`
  or `≈` passes by hand and dies with `UnicodeEncodeError` under `subprocess.PIPE`.
  `test_all.py` forces `PYTHONIOENCODING=utf-8` on every child for this reason; do the
  same in any new harness.

---

## Running it

Use the wrapper. It sets `FPL_DATA`, `FPL_HISTORY`, `PYTHONIOENCODING=utf-8` and the
Python 3.13 path, which otherwise have to be retyped into every invocation:

```bash
.\fpl.ps1 doctor              # environment + artifact graph + what to rebuild, in order
.\fpl.ps1 build               # regenerate inputs + priors      (scripts/build_all.py)
.\fpl.ps1 board               # CANONICAL per-gameweek board    (scripts/gw_board.py)
.\fpl.ps1 test --quick        # regression harness, no pipeline or studies
.\fpl.ps1 graph               # the artifact DAG                (src/manifest.py)
.\fpl.ps1 run studies/x.py    # anything else, environment already set
```

`test_all.py` is the gate: input-data schema checks, imports, every module selftest,
the four acceptance tests, the pipeline end to end, every study, board invariants, and
a determinism re-run. Run at least `--quick` before claiming anything works.

**Run it alone.** It spawns one interpreter per module and enforces a 900s cap per
selftest. Under concurrent load those caps fire on tests that take seconds in
isolation — `defcon_roles` (15s standalone) was reported as a 2000s TIMEOUT on
2026-09-08 purely because a profiler run was competing for the machine. A red harness
with something else running is not evidence of anything; re-run it serially before
believing it.

---

## Reviewing work

Seven subagents in `.claude/agents/`, each defined by **one error class no other
catches**. They are not general reviewers, they do not run on everything, and each
returns a fixed artifact contract rather than prose — an orchestrator that receives a
200-token typed artifact per worker stays usable; one that receives transcripts does not.

| Agent | Runs when | The error class it alone catches |
|---|---|---|
| `pipeline-hygiene` | any diff to `src/` `scripts/` `studies/` `tests/` | code that works here and breaks silently elsewhere, or that the harness never runs |
| `stats-referee` | a model-layer change, or a study reporting a result | runs clean, plausible numbers, **biased** |
| `deadline-devil` | the 48h before a deadline | the board is right given its inputs and wrong about this weekend |
| `forecast-scorer` | after a gameweek is scored | tracks the ranking well, **lying about its own uncertainty** |
| `structure-warden` | monthly, or when a knob changes | the repo says X and does Y; a component live with nothing behind it |
| `ux-reviewer` | after touching an output surface | a correct number that reliably produces a wrong decision |
| `research-preregistrar` | opening a new line of enquiry | a decision rule chosen after the result was seen |

`stats-referee`, `forecast-scorer` and `research-preregistrar` run on opus; the other
four on sonnet. Cost scales with tokens, not with the number of roles, and the three
opus roles are the three that reason about identification, bias and power. The
preregistrar is on that list because its job — deciding whether a proposal is a tested
null wearing a new instrument, and whether the n available can produce an informative
null — is the referee's reasoning moved before the fact rather than after it, and
because it is invoked only when a new line of enquiry opens. A missed revival costs a
whole study cycle; the opus premium on a rare call does not.

**Roles deliberately not created.** A separate QA-methodology agent duplicates
`stats-referee` exactly — same rubric, same evidence, findings twice at twice the
cost. A data-drift agent duplicates `doctor`, which detects drift deterministically.
An orchestrator agent duplicates the main session, and `src/manifest.py` is already
the shared state it would have owned. An open-ended efficiency agent has no baseline
to work against; `scripts/profile_pipeline.py` measures first, and any such agent
must work from a measured hot path and prove the board is unchanged.

**`research-preregistrar` never reports a result** — only a pre-registration. An
agent that reads data, notices a pattern and writes it up has chosen its decision rule
after seeing the outcome, which is the one thing this project's register forbids.

---

## Open items, ranked (see PROJECT_KNOWLEDGE §6)

1. Attack/defence **split** from match/supremacy odds (`oddsapi_feed`) — the outright
   markets give overall strength only.
2. Live lineups/injuries feed (`lineups.py` needs an API-Football key or team news).
3. 2023-24 data for a walk-forward joint fit of `older_weight` × shrinkage K.
4. The formal `manager_FE jointly zero` test for the press-conditioned CBIRT channel.
5. `market_share × κ` joint sweep once GW1 results land.
6. Score the model against **real** gameweeks — `src/squad_tracker.py` is the ledger, and
   nothing in this repo has yet been validated against a scored week.
