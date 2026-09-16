# Solio market feed — audit and install, 2026-08-27

## 1. The premise, corrected

The ask was to read Solio's **betting odds** and **movement tracking** and install them.
Neither exists. Verified against the feed's own agent documentation
(`https://fpl.solioanalytics.com/llms.txt`) and the complete JSON schema of
`/api/data/latest.json`:

- **No odds fields.** No bookmaker prices, no 1X2, no over/under, nothing de-vigged.
- **No movement or history.** The endpoint publishes `latest` only. There is no archive
  and no delta series; `generatedAt` stamps a single snapshot.
- "Clean sheet **odds**" in their published tables is a projected probability (`csProb`),
  not a price.

Endpoints are `latest.md` / `latest` / `latest.json`, refreshed every 4 hours. The full
schema is: `topProjected`, `topCaptains`, `topDifferentials`, `topGoals`, `topAssists`,
`bestCleanSheets`, `topBonus`, `topDefCon`, `bestAttackingFixtures`, `topTransfersIn`,
`topTransfersOut`. All are top-N truncations.

## 2. What is there instead, and why it is better than odds

`bestCleanSheets` and `bestAttackingFixtures` carry, per team-fixture:

    prGoalsFor, prGoalsAgainst, csProb

Solio states its model is "built on efficient sports markets" and that projections
"update whenever underlying market data moves". So `(prGoalsFor, prGoalsAgainst)` is a
**market-derived lambda pair** — the same quantity `betting_odds_ingest.fixtures_to_e0`
reconstructs by de-vigging 1X2 + O/U 2.5 and inverting a Poisson, except already
inverted.

**[JUDGMENT]** That is a strictly better input than the raw odds on three counts: no
de-vig method choice, no Poisson inversion, no bookmaker margin model. It is worse on
one: it is Solio's *estimate* of the market rather than the market, so their modelling
choices are baked in and unobservable. For a re-anchor of an existing market channel
that trade is favourable; for anything claiming orthogonality it is not.

## 3. Coverage is incidental and is checked every run

Both source lists are top-10 truncations, not fixture lists. Coverage is nonetheless
complete because **each record carries both sides' lambda** — a row for "Arsenal, away
at AVL" recovers Aston Villa's lambda without Villa appearing anywhere. On the
2026-08-27T12:51Z GW2 payload the union spans **all 10 fixtures and all 20 clubs, with
no conflicting duplicates**.

**[CHECK] That is luck, not a guarantee.** A double gameweek, a blank, or heavier
overlap between the two top-10s leaves fixtures uncovered — and the missing ones are not
random. The lists are ranked by clean-sheet probability and attacking output, so what
drops out is systematically the weak and the badly-fixtured. Feeding that subset into
`TeamModel` is selection on the dependent variable, biasing `att[]`/`dfn[]` toward
exactly the compression `GW1_SCORING_2026-08-26.md` §4 flagged.

`fixture_lambdas()` returns a coverage report; `stack_e0()` **raises** on incomplete
coverage rather than warning. The markdown backfill (§5) is a live demonstration: it
recovers only 8 of 10 fixtures and is therefore refused by the E0 path, correctly.

## 4. GW2 market lambda, as at 2026-08-27T12:51Z

| home | away | λ home | λ away | total | supremacy |
|---|---|---|---|---|---|
| Man United | Ipswich | 2.215 | 0.874 | 3.089 | **+1.341** |
| Liverpool | Nott'm Forest | 2.072 | 0.981 | 3.053 | +1.090 |
| Coventry | Hull | 1.675 | 1.029 | 2.703 | +0.646 |
| Chelsea | Brighton | 1.830 | 1.212 | 3.043 | +0.618 |
| Bournemouth | Everton | 1.578 | 1.134 | 2.712 | +0.444 |
| Tottenham | Newcastle | 1.640 | 1.434 | 3.074 | +0.206 |
| Sunderland | Fulham | 1.356 | 1.180 | 2.536 | +0.176 |
| Leeds | Brentford | 1.361 | 1.329 | 2.690 | +0.032 |
| Crystal Palace | Man City | 1.053 | 1.933 | 2.986 | −0.881 |
| Aston Villa | Arsenal | 0.852 | 1.902 | 2.754 | −1.050 |

Three of these are worth reading against `GW1_REVIEW_2026-08-26.md` §5:

- **Man United v Ipswich is the market's biggest supremacy of the round (+1.34)**, which
  independently supports the call made there on process grounds.
- **Chelsea v Brighton prices at 1.830–1.212.** The review downgraded that fixture to a
  "fatigue edge only" for Brighton; the market has Chelsea a clear favourite, so the
  fatigue argument is the *whole* case, not a supplement to a quality one.
- **Sunderland v Fulham is the lowest total on the card (2.536)**, matching the review's
  "low ceiling both ways".

## 5. Movement — first observation

The feed keeps no history, so the series can only ever be as long as the snapshots taken.
`data/solio_cache.md` happened to hold a 00:40Z capture, which was backfilled at its
published 2dp precision and marked `_precision: 0.01`; `movement()` carries a noise floor
of half that and flags anything inside it as not-signal.

12.19 hours, GW2, 8 clubs covered by the backfill:

| team | Δλ for | Δλ against | Δ CS prob | signal |
|---|---|---|---|---|
| **Chelsea** | **+0.070** | +0.042 | −0.013 | yes |
| **Liverpool** | **−0.058** | +0.041 | **−0.015** | yes |
| Leeds | −0.029 | −0.011 | +0.005 | yes |
| Bournemouth | +0.018 | +0.004 | +0.002 | yes |
| Arsenal | −0.018 | +0.012 | −0.003 | yes |
| Man United | −0.005 | +0.004 | −0.003 | no |
| Coventry | +0.005 | +0.009 | −0.003 | no |
| Man City | +0.004 | +0.033 | −0.011 | no |

**[VERIFIED]** as arithmetic on two stored snapshots. **[CHECK]** as meaning — n=1
interval, and nothing yet establishes that Solio's 4-hourly repricing tracks the market
rather than their own model's refresh noise. Do not act on this table. It exists so that
by GW5–6 there is a series to test, which is when `studies/inseason_weight.py` says the
weight stops being 0.01.

## 6. Why the E0 path is off by default

1. **Double counting.** `market_odds.py` already blends a 7 Aug outright snapshot into
   the ClubElo path at `MARKET_WEIGHT=0.6`, on by default in `gw_board.py`. Solio lambda
   is the same market through a different window. `stack_e0` raises unless
   `MARKET_ODDS=off`, rather than quietly compounding.
2. **No calibration transfers.** `studies/inseason_weight.py` fitted `W_MATCH` on
   *realised* early matches — a club's own played evidence against its prior. Solio
   lambda is a forward-looking prior for an *unplayed* fixture. Appending it as an E0 row
   needs no new machinery, but the weight is **uncalibrated**. `W_FIXTURE` defaults to 1
   and is explicitly not justified by that study. Sweep it; do not assert it.
3. **Orthogonality.** The market-orthogonality guard exists so a new team-level signal
   must beat the market before entering `TeamModel`. This signal *is* the market, so it
   cannot clear a test defined as beating one. It is a re-anchor of an existing channel,
   not a new channel.

## 7. What was installed

- `src/solio_market.py` — snapshot store, lambda parser, movement tracker, E0 re-anchor.
  Ships `--selftest` (offline, synthetic fixtures), auto-discovered by `test_all.py`.
- `src/config.py` — `SOLIO_SNAPSHOTS`. Committed, not scratch: a snapshot not taken is a
  movement observation that cannot be recovered later, which is the same reason
  `PREDICTIONS` is committed.
- `scripts/gw_board.py` — `SOLIO_MARKET=on` (default off), `SOLIO_MARKET_W`,
  `SOLIO_MARKET_GW`.
- `scripts/fetch_solio_snapshot.cmd` — launcher for the scheduler. Derives every path
  from `%~dp0` so the repo can move; sets `PYTHONIOENCODING=utf-8`; appends to
  `.cache/solio_snapshot.log`. Runs from `src/` because the engine modules are flat.
- **Windows Task Scheduler task "FPL Solio Snapshot"** — every 4 hours, indefinitely,
  `StartWhenAvailable` so a run missed to sleep fires on wake, runs on battery, 5-minute
  execution cap, `IgnoreNew` on overlap. Verified end to end: `LastTaskResult 0`.

  Registered with `schtasks /sc HOURLY /mo 4` and then patched via `Set-ScheduledTask`,
  because PowerShell 5.1 rejects `New-ScheduledTaskTrigger -RepetitionDuration
  ([TimeSpan]::MaxValue)` with "value incorrectly formatted or out of range" — it emits
  `P99999999DT23H59M59S`, which the task XML schema will not accept. `schtasks` expresses
  indefinite repetition natively and has no such bug.

  **Logon type is Interactive**, so it runs only while the user is logged on. Running
  whether-logged-on-or-not requires storing account credentials, which is not done here.

The selftest asserts the parts that are easy to get quietly wrong: a duplicate across the
two lists is absorbed rather than double-counted; one listed side recovers the whole
fixture; an internally inconsistent payload raises instead of picking a side; censored
coverage and double-counting both refuse; lambda round-trips through the E0 odds columns
back to within 5e-3; snapshots dedupe on `generatedAt`; and deltas are not computed across
a gameweek boundary.

```bash
python src/solio_market.py --fetch      # store a snapshot — no model change, do this often
python src/solio_market.py --movement
MARKET_ODDS=off SOLIO_MARKET=on python scripts/gw_board.py
```

## 8. Per-fixture market column — added 2026-09-11 (`src/fixture_market.py`)

The λ pair above was, until now, parsed and thrown away unless the off-by-default E0
re-anchor was switched on. It is now carried as an **annotation on the per-fixture
table**: `team_projections_gw1_38.csv` gains `mkt_lam_for`, `mkt_lam_against`,
`mkt_p_clean_sheet`, `mkt_source`, `mkt_as_of`, and the explorer's fixture tooltip shows
them beside the model's with the signed gap.

This is **display only and changes no projection**. §6.1 still holds: the market is
already in the team layer through `MARKET_WEIGHT=0.6`, so feeding per-fixture λ into
`TeamModel` would count it twice, and `stack_e0` still refuses. Reading the two side by
side is the part that was never available.

- **Which observation.** The last price at or before that gameweek's deadline
  (`deadlineIso`, falling back to `gameweek_summaries.deadline_time` looked up by id —
  that file is not sorted). For a played week that is the same information cut as
  `predictions/`, so the column stays usable when the week is scored.
- **Second source.** `football-data.co.uk/fixtures.csv` — free, no key — de-vigged and
  Poisson-inverted through `betting_odds_ingest`, stored under `data/odds_snapshots`
  (`python src/fixture_market.py --fetch`). It fills fixtures Solio did not price, PER
  FIXTURE rather than per gameweek, and where both priced one it is the only check that
  Solio still tracks the market: **λ MAE 0.032 on GW4's ten fixtures** (§PROJECT_KNOWLEDGE
  §5). Precedence is Solio first, for the three reasons in §2.
- **Double gameweeks are refused, not averaged.** A Solio record listing two fixtures
  carries one λ pair for both, and whether that is a sum or a mean is undocumented, so
  `fixture_lambdas` drops it and reports `ambiguous`. The opposing single-fixture record
  still recovers the match.
- **Clean sheets.** `mkt_p_clean_sheet` is exp(−λ_against) for every source — one
  definition, and within 0.003 of Solio's own `csProb` on every stored snapshot. Compare
  it with `p_clean_sheet_plugin`, never with `p_clean_sheet`. The explorer enforces that
  rather than trusting the reader to remember it: the tooltip pairs the market's figure
  with the model's PLUG-IN one and says so, because the two definitions round to the same
  2dp in 8 of 60 priced cells and that coincidence reads as agreement (`ux-reviewer`,
  2026-09-12).
- **Is a gap large?** The tooltip marks the market's λ inside or outside the model's own
  90% band for that fixture (`lam_for_p5`/`p95`, already exported). A signed difference
  with no scale invites treating any gap as signal.
- **Staleness.** `data/solio_snapshots` and `data/odds_snapshots` are manifest nodes and
  inputs to the per-fixture table, so a newly stored price makes `doctor` report the table
  (and the explorer behind it) STALE. That is the intended signal, and it fires only when
  a genuinely new price arrives — both fetchers dedupe.

## 9. Open

1. **[CHECK] Polling at the feed's own period is marginal.** We poll every 4h; Solio
   publishes every 4h. Each published version survives ~4h, so an on-cadence poll does
   capture every version — but with zero margin. One missed run, one drift past a
   publication boundary, and that version is gone for good. Halving the interval to 2h
   costs nothing: `fetch()` dedupes on `generatedAt`, so an extra poll stores nothing and
   logs "already have". Recommended if the series is ever to carry a test.

       schtasks /change /tn "FPL Solio Snapshot" /ri 120

2. **The pre-registered test is not written.** It should be fixed now, before the series
   is long enough to fit to: *does Solio lambda movement between snapshot t and t+1
   predict residual model error at GW t+1, conditional on appearance?*
3. **`W_FIXTURE` needs a sweep**, against held-out gameweeks, jointly with
   `MARKET_WEIGHT` — the two are substitutes, not complements.
4. **The books snapshot is not scheduled.** `data/odds_snapshots` fills only when
   `python src/fixture_market.py --fetch` is run by hand, and football-data's
   `fixtures.csv` rolls a few days ahead and then drops the fixture — an unfetched day is
   a price gone for good, exactly as for Solio. Adding the call to
   `scripts/fetch_solio_snapshot.cmd` puts it on the existing 4-hourly task; that edits a
   standing scheduled job, so it is left for a deliberate decision rather than done here.
5. **The cross-source check is n=1 snapshot.** λ MAE 0.032 on GW4 says Solio and the book
   consensus agree today. Whether that holds, and whether the two diverge in a direction
   that predicts anything, needs the series item 4 would accumulate.
