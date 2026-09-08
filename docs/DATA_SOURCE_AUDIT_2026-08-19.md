# Data source audit — 19 Aug 2026

*What each available source contributes, what duplicates something already in the model,
and what was genuinely missing. Triggered by "check for redundancy, integrate into the
existing model".*

---

## 1. Verdict table

| Source | Status | Verdict |
|---|---|---|
| `FPL-Core-Insights/2026-2027/{players,playerstats,teams,gameweek_summaries}.csv` | consumed | core roster/prices/Elo — keep |
| `.../2025-2026/By Gameweek/*/playermatchstats.csv` | consumed by `build_pms` | two-season priors — keep |
| `.../2024-2025/{players,playermatchstats}` | consumed by `multiseason` | older-season priors — keep |
| `.../By Gameweek/*/matches.csv` | consumed by `reconstruct_e0` | team model input — keep |
| **`.../By Gameweek/*/lineups.csv`** | **NOT consumed** | **real XIs — what `lineups.py` was waiting for** |
| **`.../By Gameweek/*/shots.csv`, `xg_by_minute`, `momentum`, `incidents`, `average_positions`, `*_enrichment`** | **NOT consumed** | shot/possession detail, unexploited |
| **`.../2026-2027/By Tournament/Friendlies/GW0`** | **NOT consumed** | **pre-season minutes for the 26/27 squad** |
| `.../By Tournament/{Champions,Europa,Conference,EFL}` | NOT consumed | midweek load; see §4 |
| `.../2025-2026/supplemental/incidents_quarantined.csv` | NOT consumed | 49 rows the upstream deliberately quarantined — correctly ignored |
| `.../team_history.csv` | consumed by `inseason.team_at_gw` [2026-09-07] | per-gameweek club; before it, a transferred player's old-club starts were credited to his new one |
| **Scraped Understat (`.cache/soccerdata`)** | consumed by `setpiece` only | **redundant for xG — see §2** |
| vaastav/Fantasy-Premier-League | `config.HISTORY` | already tested **null** (`deep_history_study`, 0.0017 MAE) — no action |
| Live FPL bootstrap-static | NOT consumed by the board | **was the largest live gap — see §3** |
| FFS 1-6 projection workbook | new | third-party comparator, §5 |

---

## 2. `[VERIFIED]` The Understat scrape is redundant for player xG

`crosswalk.py` exists to map Understat ids onto `player_code` by name. But FPL-Core-Insights
already ships the same quantities keyed on the FPL `player_id`, with a `players.csv` in every
folder carrying `player_code`. Measured on the 584-player 26/27 squad:

| | coverage | join |
|---|---|---|
| local repo player-match xG | **423 / 584 (72.4%)** | exact `player_code` |
| scraped Understat crosswalk | 373 / 584 (63.9%) | fuzzy name matching, 5 unverified rows |

**In local but not Understat: 50. In Understat but not local: 0.** Understat is a strict
subset, on a weaker join, with a hand-verification burden. Goalkeepers are the clearest case:
65.6% locally against 37.5% via Understat, because a shots table cannot contain a player who
never shoots.

Understat keeps exactly one advantage: **shot-level history back to 2014/15**, which
`setpiece.py` needs for the component-reliability study and the repo (24/25 onward) cannot
supply. Keep the scrape for that; do not use it for player-season xG.

New module `src/repo_events.py` reads the repo-native equivalents.

### 2.1 The crosswalk was also broken, and is now fixed

While establishing the above, `crosswalk.py --build` was found to match **20 of 461** Understat
players (4%). Three causes, all fixed:

1. **The CLI dropped the name columns the matcher needs.** It passed
   `d[["player_code","web_name","team","pos"]]`, but `_aliases` builds its full-name alias from
   `first_name`/`second_name`. Understat publishes full names ("Fabian Schär") while FPL's
   `web_name` is the short form ("Schär"), so stage 1 collapsed to the handful of players whose
   short name *is* their full name. → 20 to 267.
2. **Compound surnames never matched.** FPL carries the full legal name, Understat the common
   one: "Matheus Santos Carneiro da Cunha" vs "Matheus Cunha", "Bruno Guimarães Rodriguez
   Moura" vs "Bruno Guimarães". Added first-name + first/last surname-token aliases.
3. **Transfers cannot match on `(name, club)`.** Mateus Fernandes moved West Ham → Tottenham,
   Morgan Rogers Aston Villa → Chelsea. Added a league-wide exact-name stage, gated on the name
   being unique on *both* sides, so it stays an exact match and never auto-resolves a collision.
   Also switched the build from the shots table to player-season stats so non-shooters exist.

Final: **373 matched (368 exact)**, 63.9% of the squad. The residual is structural — Hull 0%,
Coventry and Ipswich 6.5%, because promoted clubs have no PL 25/26 data at all. Excluding
those three clubs, coverage is **369/490 = 75.3%**, and unmatched players average 0.77%
ownership against 3.58% for matched.

This work was still worth doing: it is what established that the scrape is redundant.

---

## 3. `[VERIFIED]` The board was running on stale availability — now fixed

The board built its availability from the data-repo snapshot (commit 2026-08-14). Compared
against the live FPL endpoint on 2026-08-19, **36 players the snapshot treats as available were
projected a combined 356 points over GW1-6 while the live endpoint has them out.** FFS assigns
those same 36 players 55 points.

Many are not injuries at all — they are **completed transfers still sitting in the board**:

| player | own | model GW1-6 | live status |
|---|---|---|---|
| Van den Berg (Brentford) | 0.7% | 17.75 | injured, no return date |
| Romero (Tottenham) | 1.1% | 14.30 | *joined Atlético Madrid permanently* |
| Minteh (Brighton) | 0.7% | 14.09 | leg injury, back 28 Nov |
| Nedeljkovic (Aston Villa) | 0.7% | 13.67 | *joined Rangers on loan* |
| **Spence (Tottenham)** | **7.0%** | 12.03 | *joined Internazionale permanently* |
| Enes Ünal (Bournemouth) | 0.2% | 11.71 | *joined Getafe permanently* |
| Vicario (Tottenham) | 1.4% | 8.99 | *joined Juventus on loan* |

A departed player cannot be priced by a rotation prior; he has to be removed.

**Fixed** by `signals.fetch_live_signals()` plus a `LIVE_FPL` flag (default on, fails soft to
the snapshot with a loud message so offline runs still work). The live feed covers 100% of the
squad and rules out 87 players. `apply_availability` now also joins on `player_code` when both
frames carry it — the previous `web_name` join is ambiguous for the 15 names shared by players
at two clubs.

The reverse check is clean: **zero** players the board writes off are fit per live FPL.

---

## 4. Pre-season minutes — the one genuinely new signal `[JUDGMENT]`

`2026-2027/By Tournament/Friendlies/GW0` holds 103 matches and 1,087 player-match rows for the
26/27 squad, with `player_code` in the folder's own `players.csv`. It reaches **133 of the 209
cold-start players** — the population whose start prior currently rests on ownership alone.

This matters because the model over-projects exactly that group: against FFS, cold-start
players are **+4.46 points** higher over GW1-6 (8.62 vs 4.16) while established players are
only +1.45 higher.

**It is exposed as data and deliberately not wired into any prior**, because:

- opposition quality is uncontrolled and often non-league (the fixture list includes York City);
- managers rotate whole XIs at half time, compressing minutes toward 45-60 and flattening the
  top of the distribution;
- there is **no 25/26 pre-season in this repo**, so the natural validation — do pre-season
  minutes predict GW1-6 starts, controlling for ownership? — cannot be run here.

Wiring an unvalidated signal into the dominant lever of the model would be exactly the move
this project is organised against. `repo_events.preseason_minutes()` surfaces it; the decision
rule needs a season of pre-season data that does not yet exist.

**European/cup load** (`By Tournament/{Champions,Europa,Conference,EFL}`) is a better instrument
than anything the two congestion nulls used — `rest_congestion` measured days of rest, and
`euro_qualifying_fade` used participation flags — whereas this is actual midweek minutes per
player per FPL gameweek. That is a legitimate reason to *re-open* a null with a sharper
instrument, but it needs pre-registration before it is measured, not after.

**[DONE 1 Sep 2026]** Pre-registered and run as `studies/fixture_congestion.py`; consumed
through four new `repo_events` readers. The instrument was as sharp as expected — 14.3% of
club-PL-matches were in the wrong recovery bucket without it — and **the null held**. The
same data is read from `By Gameweek/*/`, which carries every competition tagged in
`match_id`, rather than from `By Tournament/`; the two are the same matches, and the
By-Gameweek path also supplies the league outcome and the `lineups.csv` risk set.
See `docs/FIXTURE_CONGESTION_2026-09-01.md`.

---

## 5. FFS six-gameweek projections — the model is compressed `[VERIFIED]`

The supplied FFS member workbook (595 players, GW1-6) matched **581/584 board players (99.5%)**
via `src/external_projections.py`. Agreement is stable across all six weeks:

| | Pearson | Spearman | MAE | bias (model − FFS) |
|---|---|---|---|---|
| GW1 | 0.718 | 0.721 | 0.923 | +0.346 |
| GW3 | 0.699 | 0.705 | 0.910 | +0.315 |
| GW6 | 0.667 | 0.659 | 0.899 | +0.321 |
| GW1-6 total | 0.707 | 0.704 | 5.304 | +1.951 |

### 5.1 `[VERIFIED]` The live-availability fix validates out of sample

Those figures are *after* the §3 fix. Before it, on the identical player set and with nothing
else changed:

| | Pearson | Spearman | MAE | bias |
|---|---|---|---|---|
| GW1, stale availability | 0.685 | 0.661 | 1.011 | +0.435 |
| GW1, live availability | **0.718** | **0.721** | **0.923** | **+0.346** |
| GW1-6 total, stale | 0.689 | 0.665 | 5.729 | +2.519 |
| GW1-6 total, live | **0.707** | **0.704** | **5.304** | **+1.951** |

Agreement with an **independent** model improved on every metric, in every gameweek, and
nothing was tuned against FFS to achieve it — the only change was reading availability from the
live endpoint instead of a five-day-old snapshot. Spearman gaining 0.06 at GW1 is the
meaningful part: that is ranking, which is what team selection actually consumes.

The bias is not a level offset, it is **compression**. Regressing model on FFS gives a slope of
**0.473**; the sd ratio is 0.686. By FFS decile the model is +7.0 points at the bottom and −3.0
at the top:

| FFS decile | 1 | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|
| FFS mean | 0.03 | 1.80 | 8.55 | 17.16 | 24.23 |
| model mean | 7.07 | 8.93 | 12.24 | 12.14 | 21.21 |
| model − FFS | **+7.03** | +7.13 | +3.68 | −5.02 | −3.02 |

Most of the bottom-decile gap is the availability problem in §3 — FFS projects ~0 for players
who will not play, and before the fix the model did not. It also explains the apparent conflict
with the Solio comparison: against Solio's *published top 30* the model reads low, against the
*whole pool* it reads high. That is the signature of a compressed distribution, not of a level
bias, and the Solio gap was largely selection.

**FFS is loaded as its own columns and is not blended into the board.** Averaging two
projections is a modelling decision that needs a decision rule and a validation; the
`Sources` and `Biggest Disagreements` sheets exist to support that decision, not pre-empt it.

---

## 6. Set-piece takers — one live disagreement, not auto-resolved

A web check of published 26/27 set-piece guides agrees with the committed
`data/set_piece_takers.csv` on Saka, Palmer, Haaland, B.Fernandes, Thiago, Mateta, Wood,
Solanke and Groß. It disagrees on **Liverpool**: several outlets list Isak on penalties, while
the committed FFS projection gives them to Szoboszlai.

Left as-is. The committed file is a member projection with known provenance; a search summary
is not, and the module's own warning is that every scrape into this project has contained at
least one error. Isak's penalty duty is worth ~3.7 points over GW1-10 on this board, so it is
material enough to resolve deliberately rather than by overwrite.

---

## 7. `[VERIFIED]` Player naming — `web_name` is not an identifier

In the 26/27 squad **15 surnames belong to 32 different players**:

| surname | clubs |
|---|---|
| Wilson | Brentford / Coventry / Leeds |
| Phillips | Hull / Man City / Tottenham |
| Palmer | Chelsea / Ipswich |
| Johnson | Everton / Ipswich |
| Dasilva | Brentford / Coventry |
| Davies, Gomez, Henderson, Hughes, James, Kamara, King, Martinez, Patterson, Sangaré | two clubs each |

This is not cosmetic. It has caused **three separate defects in this project**: the penalty
override cleared the wrong player's duty (§ the run note), the availability join could rule out
the wrong player, and a name-keyed merge in the projection export fanned out. `player_code` was
always the right key; the missing piece was a name a human can read that is also unambiguous.

`src/player_names.py` defines one convention, used by every export:

| field | rule | example |
|---|---|---|
| `player_code` | the identifier — stable across seasons, never reused. **All joins use this.** | `244851` |
| `display_name` | minimal disambiguation: plain surname if unique; else initial + club; else full first name + club | `Haaland`, `C. Palmer (CHE)`, `Brennan Johnson (EVE)` |
| `unique_label` | always carries the club code — consistent shape, safe as a spreadsheet lookup key | `Haaland (MCI)`, `C. Palmer (CHE)` |
| `player` | FPL `web_name`, retained for continuity only — **not unique, do not key on it** | `Palmer` |

Minimal disambiguation means 552 of 584 players keep their plain surname and only the 32 that
need it pay the cost. The initial is enough for 30 of those; Brennan/Ben **Johnson** and
Josh/Jay **Dasilva** collide on the initial too and fall through to the full first name.

Verified on the real squad: 584 rows, **584 distinct `display_name`, 584 distinct
`unique_label`**, against 567 distinct `web_name`. The two Palmers now read Cole (29.04 over
GW1-6) and Alex (7.76) rather than sharing a row.

Stability is stated honestly in the module: `display_name` is computed against the current
squad, so a new arrival can force an existing player's name to grow a qualifier;
`unique_label` changes only on transfer; only `player_code` is stable full stop — which is
precisely why it, and not either name, is the join key.

---

## 8. What changed in the model

| File | Change |
|---|---|
| `src/player_names.py` | **new** — the one naming convention: `display_name`, `unique_label`, `full_name` |
| `src/signals.py` | `fetch_live_signals()`; `apply_availability` joins on `player_code` when available |
| `src/repo_events.py` | **new** — PL-only season aggregates, pre-season minutes, real XIs, all `player_code`-keyed |
| `src/external_projections.py` | **new** — FFS loader, within-club matching, agreement metrics |
| `src/crosswalk.py` | full-name aliases passed in, compound-surname aliases, transfer stage, player-season build |
| `scripts/gw_board.py` | `LIVE_FPL` flag, fails soft to the snapshot |
| `scripts/export_projection_detail.py` | same availability source as the board |
| `scripts/export_workbook.py` | **new** — the seven-sheet analysis workbook |

`repo_events` and `external_projections` both ship `--selftest`; the `repo_events` selftest
specifically asserts that cup and European rows are excluded from a league season, which is the
trap that made Haaland read 48 appearances instead of 38.
