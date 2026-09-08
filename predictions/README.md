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

GW2 and GW3 were both missed. GW3 is gone — nothing was preserved and the window has
shut. GW2 is not, because a board was rebuilt on 2026-08-31 from a data pull dated
2026-08-26, two days before the 2026-08-28 17:30 UTC deadline.

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
