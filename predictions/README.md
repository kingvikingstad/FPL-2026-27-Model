# predictions/ — what the model said before it knew

One file per gameweek, copied from `outputs/gw_board_long.csv` **before the deadline** and
never regenerated. This is the only directory in the repo whose contents cannot be rebuilt:
once a deadline passes, the availability, ownership and team-news inputs that produced the
projection are gone.

Everything in this project so far has been projection validated against other projections
(Solio, FFS) or against measured base rates. `docs/PROJECT_KNOWLEDGE_2627.md` §6.6 is
explicit that the decisive validation is post-hoc scored data, and it is the last unticked
item. That test is only possible if the prediction is preserved at lock time, which is what
this directory is for.

| file | locked | window | notes |
|---|---|---|---|
| `gw1_board_locked_2026-08-21.csv` | 2026-08-21 11:31 local, deadline 17:30 UTC | GW1 | live FPL availability, GW1 predicted XIs, injury impact, Newcastle override, GW1 Solio blend at w_ours=0.5 |
| `gw4_board_locked_2026-09-07_early.csv` | 2026-09-07 01:20 UTC, deadline 2026-09-12 12:30 UTC | GW4 | **SECONDARY.** Taken 5.5 days out as insurance. Carries GW3 results, the npxG in-season channel newly open at 3 matches/club, xA still gated. Includes `player_code`. |
| `gw4_board_locked_<date>_deadline.csv` | *to be taken on 2026-09-12* | GW4 | **PRIMARY — pre-declared 2026-09-07, before any result was known.** |
| `gw2_board_reconstructed_from-2026-08-26-data.csv` | not locked; built 2026-08-31 from a 2026-08-26 data pull | GW2 | **NOT A LOCK.** Deadline was 2026-08-28 17:30 UTC, so its inputs predate kickoff by two days and it scores as a forecast under the rule below — but nothing in the file proves the pull date, which a lock proves by construction. Scored only via `--prediction`; `_locked()` will never select it. |
| `gw3_board_prekickoff_from-2026-09-01-data.csv` | not locked; the GW3 slice of the board written 2026-09-01 19:44 local | GW3 | **NOT A LOCK.** Deadline was 2026-09-04 17:30 UTC, so its inputs predate kickoff by 2d17h and it scores as a forecast under the rule below. Carries `player_code` on all 626 rows. Scored only via `--prediction`; the name deliberately does not match the `_locked` glob, so `_locked()` can never select it. |

## TWO LOCKS FOR GW4, AND WHICH ONE COUNTS

GW4 deliberately carries two. The early file exists because this step was missed for GW2
and GW3 and an unlockable gameweek cannot be scored at all; the deadline file exists
because a lock taken five days out is missing a week of team news, injuries, price moves
and availability, and scoring it would understate the model.

**The `_deadline` lock is the primary one. That was declared on 2026-09-07, before a ball
was kicked in GW4, and `score_gw._locked()` enforces it in code** — a file matching
`*_deadline*.csv` always wins, whatever the dates say. Choosing between two locks after
the results are in is choosing the flattering one, which is the single thing this
directory exists to prevent, so the choice is not available at scoring time.

The early lock is not wasted. Scored alongside the deadline lock it measures something
this project has never measured: **what a week of team news is actually worth to the
projection.** Run `score_gw.py --gw 4 --prediction <the early file>` to get it explicitly.

To score one: join on `player_code` — the stable key, and the one the rest of the repo
mandates — and compare `mean` (model only) against `blended` (model + Solio where
matched). The GW1 lock predates that column and needs `player` + `team`, which is why it
is the only file here that does. `src/squad_tracker.py` holds the squad-level ledger;
this is the player-level one.


## WHY GW2 IS SCORED AT ALL, AND WHAT `basis` RECORDS

GW2 and GW3 were both missed at the deadline. Neither is gone.

GW2 is scoreable because a board was rebuilt on 2026-08-31 from a data pull dated
2026-08-26, two days before the 2026-08-28 17:30 UTC deadline.

**GW3 was recorded here as gone — "nothing was preserved and the window has shut" —
and that was wrong.** Four boards written before the 2026-09-04 17:30 UTC deadline were
sitting in `outputs/` the whole time as hand-kept `.bak` and dated snapshots, each
carrying a full 626-row GW3 slice with `player_code`. They have since been moved to
`.cache/boardbak/`. The claim was never checked against the directory; the lesson is
that "no lock exists" and "no pre-deadline board exists" are different questions, and
only the first one was asked.

### The GW3 selection, what was claimed for it, and what actually holds

The first version of this section claimed FOUR candidates and justified the choice with
a pre-registered rule. Both parts needed correcting, and `forecast-scorer` caught them.

**There were seven, not four.** The original enumeration globbed one filename pattern
(`gw_board_long*`) in one directory, rather than testing the property that matters:
*a board written before the deadline that carries a GW3 slice keyed on `player_code`.*
Three files failed the pattern and satisfy the property. A pre-registration is only as
good as its choice set, and this one was enumerated after the outcome was known:

| candidate (all in `.cache/boardbak/`) | written | GW3 rows |
|---|---|---|
| `gw_board_long.preGW2.bak.csv` **(scored)** | 2026-09-01 19:44 | 626 |
| `gw_board_long.gw10.2026-09-01.bak.csv` | 2026-09-01 11:12 | 626 |
| `gw_board_long.gw18.csv` | 2026-09-01 10:41 | 626 |
| `gw_board_long.gw10.bak.csv` | 2026-08-31 12:43 | 626 |
| `board_outright_baseline.csv` | 2026-08-27 15:33 | 612 |
| `gw_board_long_INSEASON_ON.csv` | 2026-08-26 20:53 | 612 |
| `gw_board_long_noinseason.csv` | 2026-08-26 20:49 | 612 |

**"The latest information set wins" is more than the evidence supports.** A `.bak` is a
COPY, so its mtime is the copy time, not the build time — mtime does not order
information sets. The selected file is even named `preGW2`, a label implying contents
older than its own timestamp. The rule fails SAFE on the question that matters
(every candidate precedes the 2026-09-04 17:30 UTC deadline, so all seven are
admissible), but it does not establish that the scored board is the best-informed one.
Claim the weaker thing.

**The selection is empirically moot, which is stronger than any pre-registration.**
Scoring all seven identically against GW3:

| board | n | r | rho | MAE | bias | top-15 bias |
|---|---|---|---|---|---|---|
| `preGW2.bak` **(scored)** | 626 | 0.548 | 0.672 | 1.26 | -0.037 | +0.17 |
| `gw10.2026-09-01.bak` | 626 | 0.550 | 0.673 | 1.25 | -0.038 | +0.17 |
| `gw18` | 626 | 0.542 | 0.666 | 1.27 | -0.026 | +0.17 |
| `gw10.bak` | 626 | 0.545 | 0.667 | 1.27 | -0.026 | +0.16 |
| `board_outright_baseline` | 612 | 0.521 | 0.627 | 1.33 | -0.009 | +0.28 |
| `INSEASON_ON` | 612 | 0.518 | 0.622 | 1.33 | -0.011 | +0.07 |
| `noinseason` | 612 | 0.518 | 0.622 | 1.33 | -0.011 | +0.07 |

The whole r spread is **0.032**, against a **±0.055** sampling CI on any single estimate
at n=626. No candidate dominates: the scored board is best on r by 0.03 and second
*worst* on top-15 bias. **There was no flattering choice available to make**, so the
selection question is answered by the specification curve rather than by asking a reader
to trust a rule. Report the curve, not the pre-registration.

The reported per-player difference of up to 3.24 points is a maximum over a heavy-tailed
difference distribution; in aggregate these are the same board.

**Caveat on what these files are.** `gw10`, `gw18` and `preGW2` are horizon runs and
`INSEASON_ON`/`noinseason` an A/B pair — `outputs/` scenario artifacts, not deadline
boards. Scoring them answers "was the model's GW3 slice any good", which is defensible
but not identical to "was the forecast a manager acted on any good". The basis stays
`supplied-prekickoff`, weighted below GW1's lock, and the honest summary is that an
mtime is weaker evidence than a lock, which proves its information set by existing.

`score_gw.main()` states the rule this rests on: **out-of-sample is a property of what
the board KNEW, not of when the file was written.** A board built from a pull that
predates kickoff is a genuine forecast even if it was written to disk afterwards. By
that rule the GW2 row is admissible. It is still weaker evidence than GW1: a lock
proves its information set by existing before the deadline, whereas the reconstruction
asks you to trust a date in a filename.

So the ledger now carries `basis` and `source` on every row, and the two are not
interchangeable:

| basis | meaning |
|---|---|
| `deadline-lock` | the primary lock, taken on the day. The strongest row available. |
| `lock` | a locked board without the `_deadline` token — GW1's, taken ~6h out. |
| `supplied-prekickoff` | passed with `--prediction`. The caller asserts the inputs predate kickoff; the file does not prove it. GW2's. |

`forecast-scorer` is required to report BASIS on every scoring, and could not do it
from a table that did not carry the distinction. Weight the rows accordingly; do not
average them as if they were the same kind of evidence.

## `misses` IS NOT COMPARABLE ACROSS GAMEWEEKS

The ledger's `misses` / `missed_points` count players projected `>= 3.0` who played
zero minutes. That threshold is fixed while the projection SCALE is not, so the column
moves with the denominator rather than with model quality:

| | GW1 | GW2 | GW3 |
|---|---|---|---|
| n(proj >= 3.0) | 171 | 148 | **48** |
| misses | 13 | 6 | 2 |
| miss RATE | 7.6% [4.5, 12.6] | 4.1% [1.9, 8.6] | 4.2% [1.2, 14.0] |
| on a fixed top-50 set | 3/50 | 0/50 | 2/50 |

Read down the `misses` row and the model improved 6.5x in two weeks. Read the RATE and
all three Wilson intervals overlap heavily; read a fixed top-50 decision set and there
is no trend at all. The mean projection fell 32% over the same three weeks (2.04 ->
1.74 -> 1.38) while the regression slope of actual on projection rose 0.96 -> 1.14 ->
1.25, i.e. the board is COMPRESSING, and a compressing board mechanically trips a fixed
threshold less often.

So `13 -> 6 -> 2` is a denominator collapse, not a result. Compute the rate on a fixed
decision set or do not compare the column across rows. The same caution applies to GW3's
near-zero field bias (-0.04): with the projection level falling past the actual level,
that is a crossing, not convergence — the next reading should be expected to go
negative. [CHECK, `forecast-scorer` 2026-09-08]

## CROSSWALKS — why `predictions/crosswalks/` exists

The GW1 lock has no `player_code`, so `score_gw` fell back to joining it on name and
club. CLAUDE.md forbids that join, and this is why: FPL rewrites `web_name` mid-season.
When a second Sangaré arrived, Ibrahim Sangaré became `I.Sangaré`, and the frozen board
stopped matching him. **The same board, same gameweek, scored a fortnight apart gave
n=577 then n=561** — the model's only forecast was quietly shedding players, and the
ledger was recording the decay as though it were a result.

`scripts/crosswalk_gw1_lock.py` resolves that board's rows to `player_code` against
`outputs/fpl_2627_players.csv`, a *contemporaneous* 2026-08-28 roster — resolving
against today's roster is the exact thing that fails. 577 rows match exactly, one is a
recorded rename, six are unresolvable and are emitted null with a reason rather than
guessed. GW1 now scores 578 stable rows and reproduces run to run.

The sidecar lives one directory down, NOT beside its lock, because
`predictions/gw*_board_locked_*.csv` is what `_locked()` globs and a sidecar named for
its lock matches that pattern — on 2026-09-08 it did, and the scorer tried to score the
crosswalk. `glob` does not cross a directory separator, so the subdirectory makes the
collision structurally impossible instead of relying on a filter someone must maintain.

**The locked boards themselves are never edited.** A crosswalk is an annotation in a
separate file; it cannot change what a board predicted. Any board written from now on
carries `player_code` and needs none of this.


## THE LOOK-AHEAD THAT WASN'T — measured 2026-09-09

`press_index.press_factor` and `set_piece_takers.observed_takers` both defaulted to
"every completed gameweek at call time" until 2026-09-08. Because `gw2_*` was built on
08-31 and `gw3_*` on 09-08 — both AFTER their own gameweek finished — those channels
were suspected of conditioning the boards on the results they were meant to forecast,
and the two ledger rows were flagged as probably contaminated.

**They are not.** Measured, not inferred:

The external feed is itself a git repo, so the information set can be pinned exactly
rather than approximated. Each gameweek was rebuilt at the feed commit matching its
original pull date, with current code, in three arms:

| arm | feed commit | press source | `INSEASON_UPTO` |
|---|---|---|---|
| A control | as originally built | pitchapi (live) | unset |
| B bounded | same | proxy | 1 (GW2) / 2 (GW3) |
| C isolator | same | proxy | unset |

`B vs C` is the pure WINDOW effect; `A vs C` is the pure FEED effect. On GW3, 629
players:

    B vs C   window          max|delta| = 0.000000   rows moved =   0
    A vs C   feed source     max|delta| = 0.061      rows moved = 201
    A vs B   both            max|delta| = 0.061      rows moved = 201

**The window contributes exactly nothing.** GW2 is the same story and stronger — every
one of 614 players is identical across every component (`mean`, `app_ev`, `att_ev`,
`cs_ev`, `defcon_ev`, `sd`). That is a true null, not rounding: GW1 rows in the same
files move by up to 0.065 and the board is written at three decimals.

The reason is that pinning the feed already bounds every repo-derived channel — the
proxy press table and `observed_takers` both read `By Gameweek/GW*/`, which the pin
truncates. The only channel that was genuinely unbounded is PitchAPI, and PitchAPI is a
live API rather than a window into the future: at the GW3 pin it carried 1-2 matches per
club against the pinned repo's 2-2. It LAGS the repo. There was never future information
in it to leak.

So the ledger's GW2 and GW3 rows stand as they are. Bounding the channels was still the
right fix — an unbounded read is a defect whether or not it happens to bite — but it
corrects nothing already recorded, and the contamination warning attached to those rows
is withdrawn.

### The two `_upto` boards

`gw2_board_prekickoff_from-2026-08-26-data_upto1.csv` (feed `0d089b2`, 2026-08-26 16:36
UTC) and `gw3_board_prekickoff_from-2026-09-01-data_upto2.csv` (feed `a796c0c`,
2026-09-01 20:44 UTC) are those arm-B rebuilds, kept because they are the only boards
here whose information set is REPRODUCIBLE: name the feed commit and the env, and you
get the same file back. Every other pre-kickoff board asks you to trust a date in a
filename.

**They are not forecasts and are not in the ledger.** They were built with 2026-09-08
code — after the XI-constraint reordering and the RNG window fix — so they are not what
the model said at the time. Scoring them measures those model changes, which is a
different and separate question from scoring a forecast.
