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
