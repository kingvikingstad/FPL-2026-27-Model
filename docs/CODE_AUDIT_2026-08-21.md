# Code audit — 2026-08-21

A whole-repo scan for correctness bugs, plus a tree cleanup. Method: pyflakes across
`src/ scripts/ tests/ studies/`, an AST pass for format-brace strings that are missing
their `f` prefix and for string literals outside cp1252, every module's `--selftest`,
and a read of the modules added since the last commit (`solver`, `player_value`,
`squad_tracker`, `injury_impact`, `defcon_roles`, `team_overrides`, `predicted_xi`,
`external_projections`, `repo_events`, `player_names`) plus the uncommitted diffs.

---

## 1. Bugs found and fixed

### 1.1 `devig_shin` was a no-op — `src/betting_odds_ingest.py`  [VERIFIED]

**The most consequential finding.** The Shin (1992) de-vig returned the proportional
de-vig *bitwise*, so the whole `oddsapi_feed` path was silently running
normalise-by-overround while reporting that it had corrected the favourite-longshot bias.

The old z-update was `z_new = sum((sqrt(PI)*p - 1) * p)`. On a normal three-way book that
evaluates to roughly **−0.64**, which `np.clip(z_new, 0, 0.2)` pinned to 0; at z=0 the
formula returns `b/sqrt(PI)`, and the in-loop renormalisation turned that straight back
into `b`. The loop therefore converged on iteration one to the input it started from.

Replaced with the standard fixed point: `z` is the root of

    sum_i sqrt(z^2 + 4(1-z) * pi_i^2 / PI) = 2 + z(n - 2)

and `p_i = [sqrt(z^2 + 4(1-z) pi_i^2/PI) - z] / (2(1-z))`, which sums to 1 by
construction and must not be renormalised. Two-outcome books (over/under) degenerate at
n=2 and fall back to proportional, which is what Shin reduces to there anyway.

Measured on 1.55 / 4.20 / 6.50: favourite **+0.0080**, draw −0.0029, longshot **−0.0051**,
and the shift scales with the longshot skew (1.12 / 9.0 / 26.0 gives +0.0165 on the
favourite). Through `fixtures_to_e0` this moves implied lambdas by up to **0.044 goals**
per fixture.

**Board impact: none, today.** `market_odds.py` — the path the board actually uses — has
its own proportional `_devig` for the 20-team outright markets and never calls this. The
bug lives entirely in `oddsapi_feed`, the match/supremacy path that PROJECT_KNOWLEDGE §6.1
lists as the next upgrade and which has no feed wired. It would have become a live bug the
moment that feed landed, because `build_market_e0` defaults to `method="shin"`.

`src/oddsapi_feed.py --selftest` asserts the two methods differ and **had been failing on
exactly this** — see §1.6 for why nobody saw it.

### 1.2 Four `NameError`s at the end of long runs  [VERIFIED]

Every one is in an `if __name__ == "__main__":` block, so they fire only when the module is
run directly — and only *after* all the expensive computation, at the line that writes the
results out. All four share a root cause: the module imports `os as _os` (to keep the
namespace clean) and these blocks were written against a plain `os`, almost certainly when
paths were centralised into `config.py`.

| File | Was | Now |
|---|---|---|
| `src/bayes_model.py:449` | `os.path.join(...)` | `_os.path.join(...)` |
| `src/bayes_model.py:457` | `fos.path.join(...)` | `_os.path.join(...)` |
| `src/roster.py:231` | `os.path.join(...)` | `_os.path.join(...)` |
| `studies/multihorizon.py:190, :215` | `os.path.join(...)` | `_os.path.join(...)` |

`studies/multihorizon.py` is run by `scripts/test_all.py`, so the harness had a guaranteed
failure in section 5.

### 1.3 A missing `f` prefix — `src/bayes_model.py:457`  [VERIFIED]

Same line as the `fos` typo:

```python
res.round(2).to_csv(fos.path.join(config.OUTPUTS, "projection_2627_{tag}.csv"), ...)
```

`{tag}` was never interpolated. Had the `NameError` not fired first, both horizons
(`GW1-6` and `full-season`) would have been written to one literal file named
`projection_2627_{tag}.csv`, the second silently overwriting the first.

### 1.4 `TEAM_OVERRIDES` was nested inside `INJURY_IMPACT` — `scripts/gw_board.py`  [VERIFIED]

The manager-judgment override block sat inside the injury-impact branch, so
`INJURY_IMPACT=off` also dropped the Newcastle override — silently, with no log line. For
an A/B flag that is the wrong failure: a run intended to isolate the injury layer was also,
unknowably, a run without overrides, and the measured difference was the sum of two
changes. De-nested; both now log their own state. Behaviour with both flags at their
defaults (`on`) is unchanged.

### 1.5 UTF-8 output dies under `subprocess.PIPE` — `scripts/test_all.py`  [VERIFIED]

On Windows the *console* is UTF-8 but a *pipe* is cp1252. `test_all.py` runs every child
with `capture_output=True`, so any script printing `Δ`, `→` or `≈` raises
`UnicodeEncodeError` **inside the harness while passing when run by hand** — the worst
possible failure mode for a regression harness, because the failure is an artefact of the
thing doing the testing.

Confirmed against two victims: `tests/test_regime.py` (prints `ΔSD`) and
`src/ab_market_vs_recon.py --selftest` (prints `Δatt`). Seven files carry literals outside
cp1252.

Fixed at the class level rather than per-print: `test_all.py` now forces
`PYTHONIOENCODING=utf-8` on every child and decodes with `encoding="utf-8",
errors="replace"`. Both victims pass.

### 1.6 The harness ran 5 of 18 selftests — `scripts/test_all.py`  [VERIFIED]

`SELFTEST` was a hardcoded list — `sd_ingest, xg_calibrate, crosswalk, setpiece,
fpl_history` — written before most of the current modules existed. Thirteen modules
shipping a working `--selftest` were never exercised by the harness: `solver`,
`player_value`, `squad_tracker`, `injury_impact`, `predicted_xi`, `defcon_roles`,
`team_overrides`, `external_projections`, `player_names`, `repo_events`, `style_matchup`,
`ab_market_vs_recon`, `oddsapi_feed`.

This is why §1.1 went unnoticed: the selftest that catches it existed and was correct, and
nothing ran it.

Replaced the list with discovery over `src/*.py`, requiring both `--selftest` and
`def selftest(` in the source (`betting_odds_ingest` mentions the flag in a docstring
without implementing it). 18 modules found, 18 pass.

### 1.7 UTF-8 BOM — `scripts/plan_constrained.py`

The only BOM in the repo. Python tolerates it, but it defeats plain-text tooling and is
inconsistent with the LF normalisation in `.gitattributes`. Stripped.

---

## 2. Not fixed — flagged for a decision

- **`injury_impact.apply_to_samples` and `team_overrides.apply_to_samples` mutate `ts` in
  place** while also returning it. No live bug: both are called once, and no runner reuses
  a sampled `ts` across variants. But the return-value convention reads like a copy, and a
  future A/B that samples once and projects several variants would compound the shift
  silently. Either copy, or rename to make the mutation explicit.

- **`defcon_roles.role_map` breaks majority-role ties non-deterministically.**
  `sort_values("n", ascending=False).drop_duplicates("player_code")` uses an unstable sort,
  so a defender split exactly 50/50 between CB and FB is classified arbitrarily and can
  flip between pandas versions. `kind="mergesort"` plus a secondary key fixes it. Affects
  only exact ties.

- **`run_final_board.py` and `export_projection_detail.py` apply neither `injury_impact`
  nor `team_overrides`, but `gw_board.py` does.** The horizon board and the detail export
  therefore sit on a different team-strength posterior than the canonical board. That may
  be deliberate; it is not documented either way, and it is a modelling decision rather
  than a bug to fix unilaterally.

- **`scripts/gw_board.py`** has a no-op `.rename(columns={"mean": "mean"})` in the Solio
  blend. Harmless leftover.

---

## 3. Tree changes

| Was | Now | Why |
|---|---|---|
| `FFS 1-6 Projection.xlsx` (repo root) | `data/ffs_1_6_projection.xlsx` + `config.FFS_PROJECTION` | A data input read from `ROOT` by a path built inside `external_projections.py` — the one thing `config.py` exists to prevent. |
| `MISSING_MODULES_CHECKLIST.md` (root) | `docs/` | Unreferenced by any code; it is a doc. |
| `outputs/plan_*.csv`, `outputs/solver_*.csv` | `outputs/plans/` + `config.PLANS` | One-off scenario answers were sitting beside `gw_board_long.csv` with nothing to distinguish "the model's projection" from "what if I keep Haaland" six weeks later. |
| — | `CLAUDE.md` | The repo had no file that restores context automatically. Carries the guards, the conventions, the Python-path quirk and the cp1252 trap. |

`README.md`'s layout map was rewritten: it predated `solver`, `player_value`,
`squad_tracker`, `predicted_xi`, `injury_impact`, `defcon_roles`, `team_overrides`,
`external_projections`, `repo_events` and `player_names`, none of which appeared in it.

**`src/` was left flat**, per the standing decision — the engine modules import each other
by name and must share one path.

`outputs/solver_gw1_squad.csv` and `outputs/plan_hold_squad*.csv` are orphans: no script
writes them any more. Moved with the rest rather than deleted.

---

## 4. Recommendation not acted on

**`outputs/` is version-controlled and regenerated every run.** The working tree at the
start of this audit carried 12,948 changed lines, of which about 12,500 were regenerated
CSVs — `projection_detail_gw1_10.csv` alone is 5.3 MB and rewrites in full on every
export. That volume of derived churn makes the code diff unreadable, which is the diff
that matters.

The study CSVs in `studies/` are a different case and should stay: they are the null-record
the project is organised around, and they change only when a study is re-run.

Options, in order of preference:
1. Gitignore `outputs/` except a small committed snapshot of the board you want to argue
   from later.
2. Keep committing them, but stop committing `projection_detail_gw1_10.csv` — it is a
   debugging export, reproducible from the board and the priors.
3. Leave as is, and accept that `git diff` needs `-- src/ scripts/`.

This is a workflow decision, so nothing was changed.
