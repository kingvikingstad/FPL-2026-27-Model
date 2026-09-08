# Fixture congestion with the competition calendar — 1 Sep 2026

*`studies/fixture_congestion.py`. Re-opens the `rest_congestion` null with the instrument
the 19 Aug data audit identified: the actual midweek fixture, not days between league
matches. Pre-registered in the module docstring before any coefficient was read.*

**Headline: congestion is real, large, and spent somewhere the league table never sees it.
It does not move the following Premier League match.** Established starters lose
**0.1 percentage points** of start probability after a three-day turnaround, CI
(−1.4, +1.9). Team attacking output does not fall either. What congestion does change is
who plays the *cup* match, and that varies sharply by competition.

---

## 1. Why a third look at a settled null was justified `[VERIFIED]`

`rest_congestion` returned +0.00000 xG per day of rest advantage and flagged its own
weakness honestly: it could only see Premier League matches, so a club playing Thursday in
the Europa League read as fully rested by Sunday. The audit found the fix already in the
repo — `By Gameweek/*/matches.csv` carries every competition, tagged in `match_id`.

That error is now measured rather than asserted. Of 740 club-PL-matches in 25/26 with a
measurable gap:

| PL-only measure | → actually ≤2d | 3d | 4d | 5+d |
|---|---|---|---|---|
| 3d | 0 | 84 | 0 | 0 |
| 4d | 0 | 0 | 53 | 0 |
| **5+d** | **1** | **53** | **52** | 497 |

**106 of 740 (14.3%) sat in the wrong bucket.** 105 club-matches that looked like a full
week's rest were three- or four-day turnarounds. That is the population the earlier null
averaged over as rested, and it is exactly the attenuation it warned about.

So the re-opening was warranted. The result is that the null survives a much sharper test.

## 2. The competition *is* the turnaround `[VERIFIED]`

UEFA fixes the weekday, so competition entry and recovery time are the same fact:

| preceded by | ≤2d | 3d | 4d | 5+d |
|---|---|---|---|---|
| Champions League (Tue/Wed) | 0 | 15 | 26 | 7 |
| Europa League (Thu) | 0 | **14** | **0** | 0 |
| Conference League (Thu) | **1** | 5 | 0 | 0 |
| EFL Cup (Tue/Wed) | 0 | 19 | 26 | 7 |
| Premier League | 0 | 66 | 41 | 275 |

Thursday competitions produce three-day turnarounds and *never* four-day ones. Champions
League and EFL Cup mostly produce four. This is the mechanism by which "which competition"
matters, and it is why recovery days — not a participation flag — is the right regressor.

**The 2-day bucket you asked for essentially does not exist**: one club-match in the whole
season (Crystal Palace, after a Conference League tie). The Premier League does not
schedule ≤48-hour turnarounds. The usable buckets are 3d / 4d / 5+d.

## 3. `[NULL]` Rotation — congestion does not cost established starters minutes

Risk set is the matchday squad from `lineups.csv`, **not** appearances. This matters:
`playermatchstats` has a minimum minutes value of 1 and contains no zero rows, so a panel
built from it is conditioned on getting on the pitch and a rested player vanishes instead
of recording a zero. Rotation would have been undetectable by construction. Within-player
demeaning; 95% CIs cluster-bootstrapped on club.

269 established starters (started ≥50% of sheets, ≥10 sheets), 11,202 player-matches,
GW1–26:

| outcome | 3d vs 5+d | 95% CI | 4d vs 5+d | 95% CI |
|---|---|---|---|---|
| P(start) | **+0.001** | (−0.014, +0.019) | +0.016 | (−0.007, +0.038) |
| P(60+ min) | +0.000 | (−0.016, +0.018) | +0.022 | (−0.003, +0.047) |
| minutes | +0.05 | (−0.86, +1.11) | +1.56 | (+0.03, +3.09) |

Adoption thresholds were ≥5 minutes and ≥5pp. Nothing comes close, and every point
estimate is the *wrong sign* for fatigue. The 4-day minutes result clears zero but is worth
1.5 minutes and says congested teams play their starters slightly **more**.

Two pre-registered robustness cuts change nothing:

- **Clean reference week.** The 5+d bucket contains international breaks, where players fly
  the world and come back tired. Restricting the reference to a normal 6–8 day week:
  P(start) 3d **+0.003**, CI (−0.015, +0.022).
- **By what the midweek fixture was**, against the same players' own rested weeks:

| preceded by | n | clubs | ΔP(start) | Δminutes |
|---|---|---|---|---|
| Europe | 1,402 | 9 | +0.009 | +0.49 |
| EFL Cup | 1,058 | 18 | +0.003 | +0.81 |
| League | 2,278 | 20 | +0.012 | +1.13 |

A three-day turnaround after a Thursday in Europe looks like a three-day turnaround after a
Saturday league game. Travel and competition add nothing detectable.

**Fringe players** (everyone else) move slightly the *other* way — +2.3 minutes at 3d, CI
(+0.4, +4.3). Small, and again the opposite of a fatigue story: congested weeks pull a few
extra squad minutes into the league, they do not push starters out.

## 4. `[NULL]` Team performance — no drop in attacking output

Opponent- and venue-adjusted residual xG, club fixed effects, GW1–26:

| bucket | n | resid xG vs 5+d | 95% CI |
|---|---|---|---|
| 3d | 119 | **+0.122** | (−0.046, +0.287) |
| 4d | 93 | +0.063 | (−0.068, +0.186) |

Fails the pre-registered ≥0.10 xG threshold on significance, and the sign is again
backwards. Consistent with `rest_congestion` — and now established on the measure that one
could not see.

## 4b. `[NULL]` Results — points, goals, wins

xG is what a side *created*; points are what it *got*. Goals are noisier, so this is the
weaker test and the power is stated rather than left implicit: points sd 1.30, n(3d)=119
vs n(5+d)=289, so **the smallest detectable 3d-vs-5+d gap is ~0.40 points per match**. A
whole season's home advantage is worth ~0.35 pts/match, so this sample can only see a
congestion effect *larger than home advantage*. Read the nulls below with that in mind.

Within club, GW1–26 (12 tests in this block; nothing Bonferroni-corrected):

| outcome | 3d vs 5+d | 95% CI |
|---|---|---|
| **points** | **+0.03** | (−0.238, +0.280) |
| win rate | −0.003 | (−0.110, +0.099) |
| goals for | +0.31 | (+0.042, +0.585) |
| goals against | +0.16 | (−0.037, +0.379) |

The goals-for interval clears zero, but it is one of twelve, its lower bound is +0.04, and
it says congested sides score **more**. Not a finding.

### The rest differential, and a trap worth recording `[VERIFIED]`

The raw two-sided comparison looks like a large, significant effect in the *wrong*
direction — the tired side earning more:

| rest differential | n | points | gf | ga | win% |
|---|---|---|---|---|---|
| opponent fresher by 2+ | 90 | **1.667** | 1.60 | 1.16 | 48.9% |
| level (within 1d) | 322 | 1.348 | 1.40 | 1.40 | 34.8% |
| we are fresher by 2+ | 90 | **1.133** | 1.16 | 1.60 | 31.1% |

`rest_congestion` argued the differential is "symmetric by construction… whatever one team
gains the other loses". **Symmetry of the regressor is not exchangeability of the units.**
The side on short rest is the side that played midweek, which is the side in Europe, which
is the better side. Season points-per-game by bucket:

| rest differential | own ppg | opp ppg | **quality gap** |
|---|---|---|---|
| opponent fresher by 2+ | 1.516 | 1.262 | **+0.255** |
| level | 1.353 | 1.353 | 0.000 |
| we are fresher by 2+ | 1.262 | 1.516 | **−0.255** |

A perfectly symmetric ±0.255 ppg quality gap, which accounts for essentially the entire raw
points gradient. `rest_congestion` was right to pair the differential with an
*opponent-adjusted* outcome; pairing it with a raw one manufactures a backwards effect.

Removing own and opponent season strength:

| | vs level | 95% CI |
|---|---|---|
| opponent fresher by 2+ | +0.064 | (−0.229, +0.352) |
| we are fresher by 2+ | +0.040 | (−0.237, +0.308) |

Both null, and both *positive* — no gradient at all. The sharp cut agrees: on ≤3 days
against an opponent on 6+, 1.613 points from 31 matches versus 1.366 when both sides are
rested, difference +0.247, CI (−0.277, +0.776), not significant and again favouring the
tired side because that side is Arsenal.

**Rest time does not measurably move results.** What looks like a rest effect in the raw
table is the quality of whoever plays midweek.

## 5. `[VERIFIED]` Where the rotation actually goes — the answer to "by competition"

The nulls above only make sense next to this. Rotation is not absent from a congested week;
it is spent on the cup team. Share of players with 60+ minutes who are established PL
regulars, GW1–26, same definition applied to every competition:

| competition | 60+ min appearances | **% established PL regulars** | fixtures |
|---|---|---|---|
| **Premier League** | 5,366 | **84.3%** | 261 |
| Champions League | 448 | **74.3%** | 44 |
| Conference League | 45 | 66.7% | 5 |
| Europa League | 134 | **58.2%** | 13 |
| EFL Cup | 522 | **46.9%** | 37 |

That gradient is the finding. Clubs field near-full strength in the Champions League
(74% vs a league baseline of 84%), roughly a half-and-half side in the Europa League, and
something close to a reserve team in the EFL Cup. **The congestion is absorbed in the
competition the club cares least about, which is why the league match afterwards is
unaffected.**

*Measurement note:* this table cannot come from `lineups.csv`. Upstream populates
`player_id` on Premier League teamsheets (400/400 in a sample gameweek) but leaves it null
on most cup rows — 15 of 129 Arsenal Champions League rows — so a lineup-based cup XI would
be a 5% subset selected by whichever names upstream resolved. `playermatchstats` joins
completely in every competition, so involvement is measured there.

## 6. `[JUDGMENT]` What cannot be answered from this data

Stated plainly because the question asked for competition-level effects:

- **Competition main effects are club identity.** In 25/26 the Conference League *is*
  Crystal Palace; the Europa League is Villa and Forest; the Champions League is six clubs.
  No estimator separates a competition effect from those clubs' seasons. It was
  pre-committed that no competition-specific parameter would be adopted from this sample,
  and none is. Only the §5 descriptive gradient and the §2 turnaround mechanism survive.
- **One season.** Cup fixtures exist in this source for 25/26 only — 24/25 ships 380 PL
  matches and no cup rows at all. There is no walk-forward validation available.
- **The FA Cup is absent entirely.** Benign for GW1–26, where FA Cup rounds sit on blank
  PL weekends (R3 on ~10 Jan 2026 falls between GW21 on 6 Jan and GW22 on 17 Jan) and still
  leave 5+ days to the next league game, so no club changes bucket. Not benign from GW27,
  where quarter- and semi-finals go midweek.
- **GW27+ is not evidence.** Cup kickoff times are present for 114/114 club-matches in
  GW1–26 and 6/50 afterwards, so late-season short turnarounds are systematically coded as
  rested. Reported as an appendix with no decision rule applied. This is the half of the
  season where congestion peaks and squads thin out, and it remains untested.

## 7. Decision

**No congestion term enters the model.** Both hypotheses failed their pre-registered rules
on the sharper instrument, with intervals tight enough to exclude anything worth a
parameter: ±2pp on start probability, ±0.29 xG on team output.

`rest_congestion`'s null stands, and its stated caveat is now discharged rather than
outstanding — the attenuation was real (14.3% misclassification) and correcting it did not
change the answer.

**Guard added:** congestion has now been tested three ways — rest differential
(`rest_congestion`), July/August European participation (`euro_qualifying_fade`), and
actual midweek fixtures by recovery day (this study). It joins rotation multipliers and
directional mean-reversion as a **tested-null heuristic that stays dead**. Do not
reintroduce it under a new name. The one live question is GW27+, and answering it needs
knockout kickoff times and the FA Cup, neither of which this source has.

**Kept for use:** `repo_events.competitive_calendar()` also reads the 26/27 folders, which
already carry *forward-scheduled* European fixtures (CL/EL in GW4, 5, 7, 8, 10, 12, 15, 16,
22, 23). That is a fixture-difficulty and blank/double-gameweek input, not a fatigue one.

---

## What was built

| file | what |
|---|---|
| `src/repo_events.py` | `competitive_calendar`, `recovery_panel`, `all_comp_player_matches`, `squad_sheets` — all `player_code`-keyed, selftested |
| `studies/fixture_congestion.py` | the pre-registered study, `--selftest` on synthetic fixtures |
| `studies/fixture_congestion.csv` | club-PL-match panel with both recovery measures |
| `studies/fixture_congestion_players.csv` | player-match panel over the squad risk set |

### Two data traps found, both now regression-tested

1. **`matches.csv` `home_team`/`away_team` hold `code`, not `id`.** Both are small
   integers with overlapping ranges, so mapping through `id` resolves about half the rows
   to the *wrong club* and returns a full, plausible frame — Burnley reads as a Champions
   League regular. Existing repo code (`reconstruct_e0`, `score_gw`, `inseason`) already
   joins on `code` correctly; the selftest now pins it.
2. **`players.csv` inside a gameweek folder is a whole-league snapshot**, so its
   `team_code` is a player's *current* club, not his club at that match. Attributing cup
   appearances through it hands a January transfer's autumn European matches to his new
   club. Club attribution comes from the match row instead.

Both produce wrong-but-plausible output rather than an error, which is the failure mode
this project is organised against.
