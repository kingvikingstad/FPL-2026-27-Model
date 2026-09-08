# Integration Log — 2026/27 build session

Chronological record of what was built, validated, and deliberately left out, with the
empirical result for each. Companion to `PROJECT_KNOWLEDGE_2627.md`.

## Foundation de-risking
- Verified prices/ownership/fixtures/transfers/injuries against the live 26/27 launch data.
- Corrected two earlier claims: Salah is absent from the data (not carried); Haaland's 70.3%
  is live pre-GW1 ownership, not a stale estimate.
- Reconstructed the two missing inputs faithfully: `E0_recon.csv` from the repo's 25/26 Opta
  match xG (fair-odds Poisson inversion → existing team model runs unchanged; ratings sound —
  Man City 2.02 attack, Arsenal 0.75 defence); `coldstart_hist.csv` from the per-match panel.
- Full pipeline reproduces prior numbers within MC noise (faithfulness confirmed).

## Signal layers built
1. **Cold-start depth prior** (`starter_prior.apply_coldstart_depth`) — ownership-aware start
   prior for no-history players (ownership predicts start-rate at Spearman ~0.68). Collapses
   cold-start × strong-team inflation (Alleyne 29.6→11.9); established players untouched;
   discriminating by ownership.
2. **Evidence-weighted minutes shrinkage** (`apply_minutes_shrinkage`) — trust history by
   sample size (`w=minutes/(minutes+900)`), shrink thin/role-changed cases to the market. The
   largest validated gain: Spearman vs Solio 0.482→0.609, MAE 0.75→0.70, Mosquera +1.32,
   Gabriel +0.02.
3. **Lineup/injury layer** (`lineups.py`) — source-agnostic XI ingestion feeding
   `apply_availability`; named starter→0.99, benched→sub-rate, excluded→0.05. API path wired.
4. **Solio ensemble** (`solio_ensemble.py`) — fetch/parse/align/blend/benchmark; reproduces
   Solio's leverage ranking; team-aware matching (fixed a Palmer GK/MID collision). Live GW1:
   Pearson 0.72, MAE 0.72. Added `benchmark_within_team` (isolates the share term; flagged
   Arsenal internal ranking rho 0.0 that pooled 0.61 hid).
5. **Regime variance decoupling** (`apply_regime_uncertainty`) — two knobs: δ (mean-pull, no-op
   1.0) and κ (variance-inflation via moment-matched mixture, no-op 0.0). Baseline identity
   exact; horizon signature passes (SD widens +0.32 at H=6, ~0 at H=1) — confirms the
   H(H−1) shared-p term propagates with no change to project(). Off by default
   (`REGIME_2627_PROPOSED`, includes Newcastle as the ninth regime club).
6. **Appointment panel split** (`regime_panel.py`) — for partial-regime clubs (Man Utd/Spurs),
   weight post-appointment 25/26 matches at full strength instead of a flat multiplier. Keyed on
   player_code (a player_id-keyed first cut matched only 2 of 41 incumbents). Market agreement:
   boosted incumbents avg 6.9% own vs cut 1.7%. Isolation passes.
7. **DefCon environment conditioning** (`defcon_env.py`) — scale the DefCon rate by the team
   xGA environment. DEF/CBIT full sensitivity (monotone in xGA); MID-FWD/CBIRT untouched
   (press-driven; Solio's Anderson 56% at City confirms it holds up). Anderson natural
   experiment correctly untouched; promoted-club defenders boosted (Hull/Coventry), City
   transfers cut (Guéhi/Khusanov).

## Bug fixes
- **Cold-start id collision (§6.2)** — `roster._coldstart_row` gave every cold-start player
  `id=-1`, collapsing ~198 into one under `drop_duplicates("id")`. Now unique (-1,-2,-3,...).
- **defcon_ev / cs_ev surfaced** — `bayes_model.project()` now returns per-player DefCon and
  clean-sheet expected points, making DefCon auditable/A-B-able (§4.1a of the DefCon handoff).
- **Solio name collision** — alignment made team-aware (name+team).

## Season-to-date facts labelled, and points-per-£m retired, 7 Sep 2026
Two changes to the explorer's Players tab, both about the same failure: a table that puts
an exact fact and a modelled quantity side by side, unlabelled, invites reading one as the
other.

**The realised 26/27 block is now the source of price and ownership, not just a column
beside them.** `season_to_date` joins `playerstats.csv` at its latest gameweek on
`player_code` — 653 of 653 board players, no misses — and `now_cost`,
`cost_change_event`, `cost_change_start`, `selected_by_percent`, `transfers_in/out_event`,
`total_points` and `minutes` come off it. `now_cost` and `selected_by_percent` now feed the
`Cost` and `Own %` cells directly rather than the board's copy taken at build time. The two
AGREE today — 653 of 653, max |difference| exactly 0.0 in both — so this is a no-op on
current inputs and a guard against reading a stale board's prices as live ones; a
disagreement is counted and printed at export rather than silently reconciled, because a
board built against different prices is a fact worth seeing.

Every one of these columns carries its own tooltip now, on the header and in the column
picker, saying which side of the line it sits on. The distinction matters most exactly
where it is least visible: at three gameweeks `total_points` is mostly variance and the
projection beside it already conditions on everything the number contains, while
`cost_change_start` and `selected_by_percent` are the two facts a transfer decision turns
on that the model does not model at all. They were previously separated only by a comment
in the source. Display precision was wrong too — counts and prices were printed to two
decimals, which reads as model output; minutes and points are integers, transfers carry a
thousands separator, and the price moves carry their sign, since direction is the whole of
what they say. The CSV export writes raw values and is untouched.

**`perM` is gone.** The points-per-£m column divided the window total by the WHOLE price,
so it charged every player for the first ~4.0m that buys nothing a free slot would not have
given you anyway — the near-floor artefact `player_value.py` documents at length. With
`par` on the metric list since 4 Sep, the page carried two columns claiming to answer the
same question, and they disagreed COMPLETELY: over GW3-8 par's top six were Haaland,
B.Fernandes, Thiago, Mbeumo, Semenyo, Raya; points-per-million's were van Ewijk, Guehi,
Thiaw, Thomas, Bassey, Mitchell. Removed as a column, as a filter field and as the
profile's `Value` axis, which is now `par` summed over the window — on the same profile
Haaland moves from near the bottom of the value axis to #1 of 27 forwards. `m:par` takes
its slot in the default column set, so the value question still gets an answer in the
default view, from the construct that answers it. The selftest asserts the string `perM`
appears nowhere in the rendered page, so it cannot come back by accident.

Profile axes are also filtered by availability now: an axis whose column the board does not
carry is not offered, rather than drawing an empty ring for every player, which reads as
"measured, and zero". The squad tab's rotation planner is untouched where it ranks on GAIN PER
£m PARKED — a marginal quantity over the capital actually left idle, which is the same
question asked correctly and stays the default. Its `Rotation total per £m` sort option
went with the column (8 Sep): it divided the pair's TOTAL by the pair's WHOLE price, so it
charged both players for the first ~4.0m each and ranked cheap pairs that rotate to nothing
above pairs that score. Three ranking options remain, none of them an average-per-price.

## The board is bit-identical, so `d_blended` has no noise floor, 7 Sep 2026
`d_blended` went live when the date rolled and immediately showed six players moving
-0.01 to -0.02, all in GW4, on identical data and code. Either that was Monte Carlo
jitter — in which case the delta column shows phantom movement daily and small values
cannot be read — or it was real.

Chasing it turned up that `test_all`'s DETERMINISM section is `if not quick`, so every
harness run this session skipped it, and its tolerance is `max |delta| < 0.75` — the board
is only asserted deterministic WITHIN Monte Carlo noise, not bit-identical.

Measured directly instead: a second board into a temp `FPL_OUTPUTS`, diffed on
(player_code, gw). **24,814 of 24,814 player-gameweeks exactly equal, max delta
0.000000.** The board is strictly deterministic — `bayes_model._player_rng` did its job
and the 0.75 tolerance is legacy conservatism from before that fix.

So `d_blended` has NO NOISE FLOOR: every non-zero value is a real change. The six movers
were the live FPL availability feed — the board's only non-deterministic input — revising
chance-of-playing overnight between the 22:29 and 06:26 runs, and GW4-only because that
cap applies to the imminent gameweek. The delta column caught a genuine team-news change
on its first day, which is exactly what it is for. Recorded on the metric so a small
value is not dismissed as jitter.

## Captaincy tail metrics surfaced, 7 Sep 2026
`src/captaincy.py` has computed P(haul), ceiling, floor, regret against the template
captain and a differential edge all season. Its only importer was `scripts/legacy/`, so
the board, the explorer and every export ranked captaincy on the MEAN — for a decision
its own docstring calls a tail problem, not a mean problem.

NOT by calling `captain_picks`. That re-derives the draws through its own `point_draws`
with its own RNG, so its `ceiling` would disagree with the `p95` on the same row by pure
Monte Carlo noise — the divergence trap `export_projection_detail` documents. The new
`gw_explorer.captaincy_tail(gw)` reads `.cache/draws_gw<N>.npz`, written by `gw_board`
itself, so every number comes from the same simulation as the rest of the board.
`HAUL_PTS` / `BLANK_PTS` moved to module constants in `captaincy` and are imported rather
than restated, so "haul" cannot come to mean two things.

`DUMP_DRAWS` now DEFAULTS to the projected gameweek instead of nothing. Percentiles do not
add and a joint event — "mine blanks while the template hauls" — cannot be recovered from
summary columns at all, so without the raw draws these metrics are not merely inconvenient
to compute, they are unavailable. ~1MB per gameweek in SCRATCH.

THE KNOWN BIAS, MEASURED RATHER THAN WAVED AT. The engine draws team goals as Poisson and
the league's upper tail is thinner: 4+ goal team-games happen on 4.08% of team-matches
where the model implies 6.54%. That inflates `p_haul` — and NOT uniformly. On GW4,
corr(p_haul, fixture lambda) = +0.34, and the top 25 by `p_haul` sit at mean lambda 1.95
against 1.46 board-wide. The captaincy shortlist is drawn from fixtures **34% above
average lambda**, exactly where the Poisson tail is too fat, so the error concentrates at
the TOP of the ranking and distorts the ordering, not just the level.

Surfaced anyway, uncorrected, with that written on the metric. I had said surfacing it
uncorrected would be worse than not surfacing it; that was wrong. The status quo ranks
captaincy on the mean and ignores the tail completely, and a biased tail metric with its
bias quantified beats no tail metric. `team_explosiveness.apply_tail_calibration` stays
OFF: applying it moves the clean-sheet engine validated at GA r=0.89 / CS r=0.93.

## Week-over-week view, 7 Sep 2026
The weekly decision is MARGINAL — what changed since you last picked — and the explorer
only ever showed levels, so the diff had to be held in your head. Two additions:

**1. `d_blended`, change since the last board.** `gw_board` now writes a compact dated
snapshot to `SCRATCH/board_history/board_YYYY-MM-DD.csv`. The baseline is the newest
snapshot from a PREVIOUS DAY, never the previous run: the board gets rebuilt several times
in an afternoon and diffing against the last run would make the column mean "since I last
pressed go", which is noise dressed as information. Same-day reruns overwrite today's file
and leave yesterday's intact. Additive, so a window sum is the total change to a player's
run. Absent — not zero — when there is no earlier snapshot, because a column of zeros
cannot be told apart from "nothing moved".

Snapshots live in SCRATCH, not OUTPUTS: neither regenerable (a past board cannot be
rebuilt once the data moves) nor precious (losing them costs a convenience view, never a
validation — that is what predictions/ is for).

Verified against a genuine earlier state — the board from before the npxG channel opened.
10,554 of 24,814 player-gameweeks moved, and the largest movers reproduced the npxG
measurement exactly (Yalcouye +0.51, Palacios +0.40, Grimes -0.38 on GW4).

**2. Realised 26/27 facts, five static columns.** Price move this gameweek, price move on
the season, net transfers this gameweek, points, minutes. Straight from `playerstats.csv`,
100% coverage of all 653 board players, exact rather than modelled — and labelled that
way, because at three gameweeks `total_points` is mostly variance while `now_cost` decides
what you can literally afford.

**A UNITS TRAP, caught in the browser.** `now_cost` arrives in millions (12.0) but
`cost_change_event` / `cost_change_start` are in TENTHS. Shown raw beside a price in
millions, Mbeumo's -0.1m read as "-1.00" — an order of magnitude out and plausible enough
to be believed. Converted once at the source.

## The predicted-XI layer had been dark since August, 7 Sep 2026
`PRED_XI_GW` defaulted to a hardcoded `1`. Correct in August; silently wrong from GW2 on.
The board's own log on 6 Sep:

    [pred-xi] GW1 Fantasy Football Scout   2026-08-20
    [pred-xi] applied to GW1 only; GW2+ use the untouched priors

So it was folding in 20 August's team news and running every gameweek anyone was actually
picking on untouched priors. That is the input the project's own scoring calls its largest
error source — correlation roughly halves conditioned on appearing (GW1 r=0.53 -> 0.32),
and GW1 lost 52.6 projected points to 13 availability misses. It failed silently because
the log line looked healthy: it said what it WAS doing, never what it was missing.

`src/fplpage.py` was written specifically to fetch this feed and was **imported by
nothing**.

TWO FIXES:

1. `PRED_XI_GW` is DERIVED, not hardcoded — the first gameweek with nothing played, the
   same marker the explorer and wildcard solver use. A hardcoded default cannot age well
   when the right answer changes weekly. Env var still overrides; falls back to 1 if the
   in-season feed is unreachable, as before.
2. `lock_board --auto` now fetches team news BEFORE rebuilding, since `gw_board` folds the
   predicted XI into the projection it is about to build. Best effort, never fatal: a lock
   WITHOUT team news still beats no lock, but which one happened is printed loudly because
   it changes what the lock is worth.

VERIFIED END TO END. The fplpage path had never run in anger. On GW1 it fetched, OCR'd the
pitch graphic and scraped **220 players**, which `predicted_xi.sources()` then discovered
as a fourth feed (test artefact removed afterwards; GW1 is settled). On GW4 it returns 404
— fpl.page publishes the day before a deadline — handled as the normal early answer and
reported as "the lock will carry NO team news".

## Schedule rank split by end of the pitch, 7 Sep 2026
Team trends showed ONE "Schedule rank". `sosRaw(t, mode)` already knew the difference —
an attacker meets the opponent's DEFENCE, a defender meets its ATTACK, so the two are
different opponents' qualities — and the SoS chart already had the toggle. Only the KPI
and the profile axis hardcoded `"all"`, averaging the two into a number that can be wrong
at both ends at once.

Now two KPIs (`Schedule · attack`, `Schedule · defence`, each with the mean opponent
strength behind it) and two profile axes (`Schedule (att)`, `Schedule (def)`). Home count
moved to its own KPI rather than riding along as a subtitle on a rank it did not describe.

It is not a cosmetic split. Over GW4-9 the two ranks differ by a mean of 3.5 places, up to
7, and only 1 of 20 clubs ranks the same at both ends:

    Chelsea    #3 attack   #10 defence   (old single rank: 5)
    Brentford  #4          #11           (8)
    Arsenal    #15         #8            (12)

The old rank sat between them and misled in both directions — it overstated Chelsea's
defensive run and understated its attacking one, in the same cell.

The profile caption needed no edit: it counts season-level versus window axes from
`team_components` rather than stating a number, so it moved from "the last 4" to "the last
5" on its own. That is the second time that change has paid for itself.

## test_all discovered only half of what it claimed, 7 Sep 2026
`_discover_selftests()` globbed `src/*.py` only, so **three runners shipped selftests the
gate never ran**: `scripts/lock_board.py` (which guards the integrity of the locked
prediction record), `scripts/postgw_review.py`, and `test_all` itself.

CLAUDE.md's convention is that selftests are DISCOVERED and never listed, exactly so that
nothing can be silently left out. The discovery then looked in one directory — the same
class of omission, one level up, and invisible because the count kept rising as `src/`
grew. Found by checking whether the harness that had just passed actually covered the
script it was passing on.

Now globs `src/` and `scripts/`, labelling the latter `scripts/<name>` so the report says
where each came from. `test_all` excludes itself: running the harness inside the harness
is a recursion. 29 selftests -> 31.

## Scheduled lock, 7 Sep 2026 — `lock_board.py --auto`, daily
`--auto` added and wired to a daily scheduled task (`fpl-lock-board`, 05:00 local).

WHY DAILY AND NOT "ON DEADLINE DAY". Deadlines do not fall on a weekday cron can express:
26/27 spreads them across 10:00, 11:00, 12:30, 13:30, 17:30 and 18:30 UTC on several
different days. Anything pinned to a time either fires days early — writing a PRIMARY lock
with none of the team news it exists to capture — or misses the week. So `--auto` is a
STATE test, the same trigger `postgw_review --auto` uses: act only when the next
gameweek's deadline is inside the window and no lock exists. Every other day it prints one
line and exits 0, so the scheduler sees success rather than a daily failure.

`--auto` implies a board rebuild, because the likeliest way to get a useless lock is not
locking late but locking a board built before the last data pull. The rebuild is expensive
and so happens only on the day the lock is actually taken.

TIMING, CHECKED AGAINST ALL 38 GAMEWEEKS. The machine is UTC-4, so the first attempt at
10:00 local (14:08 UTC) would have fired AFTER the 12:30 UTC deadlines and locked most
weeks ~22h early off the day-before run. Moved to 05:00 local (09:08 UTC), which catches
every one of the 38 SAME DAY, lead times 0.9h to 9.4h.

WINDOW WIDENED 24h -> 26h, for one specific hole that the coverage check exposed. The
earliest slot (10:00 UTC, GW6 and GW7) has only 0.9h of same-day lead; if that run is
missed — machine off — the fallback is the previous day's run at 24.9h, which a 24h window
would have REFUSED as too early, and the next run is past the deadline. Those two
gameweeks would have gone silently unlocked, the exact failure this path exists to
prevent. At 26h the fallback covers both tight slots; every other slot has 3.4h+ of
same-day margin and needs none. Cost is at most two extra hours of team news, in the rare
case the fallback is used at all.

## scripts/lock_board.py, 7 Sep 2026 — the missed step, made one command
GW2 has only a reconstruction and GW3 nothing, because locking was a manual copy nobody
performed in time. `python scripts/lock_board.py` now does it, defaulting to the gameweek
FPL flags `is_next` and reading deadlines offline from `gameweek_summaries.csv`.

FOUR REFUSALS, each earning its place:

* **After the deadline.** A board saved at kickoff+1 looks identical to a real lock and is
  worthless — it may carry team news published after the deadline, so scoring it measures
  hindsight. `--force-late` overrides but renames the file `_LATE_not_a_prediction`.
* **A PRIMARY lock taken days early.** This was the sharp one, found by dry-running the
  first version: with `--label deadline` as the default, running it today would have
  written the primary lock 131 hours out — becoming the file `score_gw` scores while
  containing none of the team news a deadline lock exists to capture, quietly undoing the
  pre-declaration made hours earlier. Refused outside `DEADLINE_WINDOW_H = 24`, pointing
  the user at `--label early` instead. `--anyway` overrides.
* **Overwriting an existing lock.** "Never regenerated" is the README's contract.
* **A gameweek the board does not carry** — refused, not written empty.

It also compares the board's mtime against the newest match file and says so: the likeliest
way to get a useless lock is not locking late but locking a board built before the last
data pull, preserving a projection the model had already superseded.

The selftest covers all four plus the dry run, and it caught its own fixture: `before` was
originally set 60 hours from the deadline, which the new primary-lock guard then correctly
refused.

## GW4 locked, 7 Sep 2026 — and the choice removed from scoring time
`predictions/gw4_board_locked_2026-09-07_early.csv`, 653 players, GW4 only. Carries GW3
results and the newly-opened npxG channel. Unlike the GW1 lock it includes `player_code`,
so it joins on the key the rest of the repo mandates rather than on `player` + `team`.

The deadline is 2026-09-12 12:30 UTC — **5.5 days out**, so this lock is missing a week of
team news, injuries, price moves and availability. A second `_deadline` lock will be taken
on the day. GW4 therefore carries TWO, which raises the obvious hazard: choosing between
them once the results are in is choosing the flattering one.

So the choice was removed rather than documented. **The `_deadline` lock is declared
primary as of 2026-09-07, before a ball was kicked, and `score_gw._locked()` enforces it**
— any file matching `*_deadline*.csv` wins regardless of date, verified against a fake
deadline lock deliberately dated eight months EARLIER. A README convention would have
relied on the filenames happening to sort the right way; this does not.

The early lock is not redundant. Scored alongside the deadline one it measures something
never measured here: what a week of team news is worth to the projection.

Why an early lock at all: this step was missed for GW2 (only a reconstruction exists) and
GW3 (nothing), and an unlocked gameweek cannot be scored at all. PROJECT_KNOWLEDGE §6.6
has been the last unticked item all season; a file that exists beats a better file that
does not.

## In-season rate channel: one gate per quantity, 6 Sep 2026
"Turn the current season on and weight it appropriately." The WEIGHT was already
appropriate — `W_RATE = 1.0`, fitted in `studies/inseason_rate_weight.py`, a current
season match worth about one prior-season match, the same answer the team channel reached
independently. What was wrong was the GATE.

`MIN_RATE_MATCHES = 5` applied a single threshold to two quantities that were calibrated
separately and do not clear at the same k:

    npxG/90  k=3  MAE 0.06270 -> 0.05999  CI (-0.00512, -0.00029)   clears
    xA/90    k=3       0.04877 -> 0.04840  CI (-0.00108, +0.00034)   does not

The constant's own comment said it: "xA does not clear the rule at k=3, so the gate sits
at 5". The gate was therefore set by the WEAKER channel, withholding npxG for two
gameweeks in which npxG had been measured to help, because a different quantity was
noisy. Split into `MIN_RATE_MATCHES_NPXG = 3` and `MIN_RATE_MATCHES_XA = 5`, applied per
channel inside `update_rates`. **Neither number is new** — both come straight out of the
table already in the module. `MIN_RATE_MATCHES` stays as an alias for the stricter one,
and an explicit `min_matches=` still forces the old single-gate behaviour.

Live at 3 matches per club: **npxG updated 243 players, xA updated 0**, mean |change| in
npxG/90 = 0.0228. Board effect: mean |change| 0.0255 pts per player-gameweek, 542 of 653
players moved, league total 32453.9 -> 32190.6 (-0.8%). The movement is concentrated in
fringe players whose priors were weakest — biggest risers Palacios +18.6, Yalcouye +14.4
over the season; biggest fallers Jacquet -13.2, Grimes -12.9. A settled XI barely moves:
the live squad's GW4 XI went 48.11 -> 48.12.

`gw_explorer.evidence()` now reports the two gates separately (`npxg_channel`,
`xa_channel`); `rate_channel` is true only when BOTH are open, so no existing reader is
told the channel is on while half of it is still gated. xA opens at GW5.

## GW3 refresh, 6 Sep 2026 — three counting bugs in the scoring/selection path
Pulled the data repo (stale since 1 Sep), rebuilt priors, board and exports. GW3 is
**9/10** finished — Arsenal 2-1 Chelsea was still in play — so it is NOT reviewable yet
and `postgw_review --gw 3` correctly refused. GW2 was reviewed and the ledger refreshed
to final (r=0.622, MAE=1.320, n=612). Three defects surfaced on the way, all in the code
that decides *what counts as a played gameweek*:

1. **`score_gw.played_clubs` counted cup ties as league fixtures.** It read the raw GW
   folder `matches.csv`, which carries cup rows, so GW2 read 20/20 done when the league
   round was 10 fixtures. It also mapped absent EFL team codes to NaN and passed them
   through `norm_team`, which stringifies — putting a phantom club literally named
   `"nan"` in the returned set and a `clubs` count of **21** in the scoring ledger. Now
   filtered on the `-prem-` token like `inseason`/`repo_events`, with unresolved clubs
   dropped as a second guard. The `final` flag was computed on the cup-inflated
   denominator, so a week could have been called final with league games outstanding.

2. **`postgw_review.league_fixtures` had the trap its own docstring warns about.** It
   identified cup ties as "team codes absent from teams.csv", which is true of a tie
   against an EFL club and **false of a cup tie between two Premier League clubs**. GW2
   contained exactly one — `26-27-efl-cup-nottingham-forest-vs-leeds-united` — so the
   round reported **11/11** when the league had 10. Both were finished so the equality
   held by luck; one unfinished all-PL cup tie would have made `--auto` withhold a review
   of a COMPLETE league round indefinitely. Now on the `-prem-` token, and the selftest
   gained that exact case as a regression: an unfinished PL-vs-PL cup tie in the fixture.

3. **`export_wildcard_xi` opened its window on a gameweek nobody could pick for.** Its
   comment claimed "the same marker the explorer uses"; it in fact used the explorer's
   `first_live` (first not-FULLY-played week), not `first_clean` (first week with nothing
   played). On 6 Sep that solved a "GW3 optimal 15" two days after GW3's deadline, for a
   week settled for 18 of 20 clubs. Now skips partly-played weeks too, and solves GW4-15.
   A partly-played week is neither past nor future, but for squad selection it is past.

## Gameweek Explorer, 4 Sep 2026
- **Points above replacement (`par`) wired in** — from `player_value.per_gameweek`, NOT
  reimplemented. Floors recompute inside every gameweek, which is what makes `par`
  additive across a window: each week is measured against that week's own replacement, so
  the sum is total points above replacement over the run. Added to `ADDITIVE`. Eligibility
  for the floor uses `app_ev / 2` as a start-probability proxy so a 4.0m player who will
  never play cannot set it; that proxy is used for the filter only and enters no
  projection.
  The two constructs disagree completely, which is the point — over GW3-8 the top six by
  `par` are Haaland, B.Fernandes, Thiago, Mbeumo, Semenyo, Raya; by points-per-million
  they are van Ewijk, Guehi, Thiaw, Thomas, Bassey, Mitchell, i.e. entirely the near-floor
  artefact `player_value.py` documents. The old `perM` column is left in place rather than
  removed; that is a separate decision. **Taken 7 Sep 2026: removed — see the entry above.**
- **Notebook on the My squad tab** — free-text team strategy plus per-player notes, which
  is the one input the model structurally cannot have. Same storage discipline as the
  existing locks/bans: `localStorage`, keyed on `player_code`, every read and write
  guarded, so a weekly rebuild of the page keeps them. Verified surviving a real
  `export_gw_explorer.py` rebuild.
  Three deliberate choices: a **write probe** at load, because storage that reads but
  cannot write would otherwise fail silently and the status line says so; note text set as
  a DOM property and never interpolated into markup, so a note containing quotes or tags
  cannot break its own row; and import **merges** rather than replaces, so importing on a
  machine that already has notes cannot wipe them. Notes on sold players are kept — the
  reasoning for selling is worth as much as the reasoning for holding.

## Verified, no change needed
- 26/27 position reclassifications are correct in the frame (Wieffer DEF, Sessegnon DEF,
  Lewis-Skelly MID); `DEFCON_THRESHOLD` keys off `pos` — no stale-position scoring bug.
- Team clean-sheet engine already market-calibrated (GA r=0.89, CS% r=0.93) — not recalibrated.

## Deliberately NOT built
- Directional style priors (matchup_design C1/C2/C3) — same class as the null heuristics;
  style unmeasurable for ~55/60 high-leverage GW1-6 fixtures.
- CBIRT xGA downscaling — press-driven, Solio-contradicted; pending a press-index test.
- δ regime mean-pull values — unfitted; wired but off.
- Recalibrating the team model on a single-GW competitor cross-section.

## Provenance
Numbers attributed to Solio are cross-sectional single-GW benchmarks, not ground truth. The
decisive validation is post-GW1 scored data; `older_weight`/Isak needs 2023-24 walk-forward.
