# Deep Historical Data — Findings

**Opened:** 2026-08-09 · **Status:** infrastructure landed; the headline proposal is a **tested null**
**Sources evaluated:** `vaastav/Fantasy-Premier-League` (player-gameweek, 2016/17→) and Understat
shot-level via `sd_ingest` (2014/15→)

Per project convention this records the null as prominently as the win.

---

## Bottom line

| Claim in the proposal | Verdict |
|---|---|
| vaastav is "the big one" for the minutes/availability layer | **NULL** — 0.0017 MAE vs the existing baseline, held out |
| vaastav "already does the Understat-to-FPL-ID merge" | **False** for the current repo layout — no id map ships |
| Understat has shot-level depth to 2014/15 | **True** — 12 EPL seasons resolve |
| FPL `code` is a stable cross-season player key | **True, and better than expected** — it *is* `player_code`, exactly |

Nothing was wired into the board. `to_priors` is bitwise unchanged unless `deep_starts` is
passed explicitly, and no runner passes it.

---

## 1. `[VERIFIED]` Identity is free — `code` **is** `player_code`

The one unambiguous win. `merged_gw.csv` keys on `element`, which FPL **reassigns every
season**, so joining on it across seasons would silently merge different players.
`players_raw.csv` carries `code`, FPL's permanent id, and it matches the repo's
`player_code` exactly: **841 of 841** 25/26 players intersect, spot-checked by name
(Mbeumo 446008, Tavernier 201658, van Hecke 469142).

So the FPL-side history joins to the model with **no fuzzy matching and no crosswalk** —
G5 is satisfied by construction. `fpl_history` enforces this: it rebuilds the
element→code map per season, reports the match rate, and **raises** below 95% rather
than falling back to name matching. All 10 seasons match at 100.0%.

This does *not* extend to Understat, whose ids are its own — that path still needs the
crosswalk (§5).

---

## 2. `[NULL]` Deep minutes history does not improve the minutes prior

`multiseason.py` established that prior-season evidence roughly doubles early-season
predictive power. That is a case for *a* prior, not for an arbitrarily deep one, and the
difference is the whole question.

Held out 25/26; predicted each player's start rate from 24/25 alone (the model's current
baseline) versus 24/25 plus eight decayed earlier seasons. Same players, same metric:

| group | shallow (24/25) | deep (8 more seasons) | deep, active-only |
|---|---|---|---|
| all (n=533) | 0.2005 | 0.1988 | 0.2935 |
| regulars, ≥900 min prev (n=266) | 0.2531 | 0.2519 | 0.2487 |
| fringe, <900 min prev (n=267) | 0.1480 | 0.1458 | **0.3381** |

**0.0017 MAE.** Nothing. And the steelman is worse, not better.

### Why — two structural reasons, neither fixable by tuning

**Start rate is a property of a role at a club, not of a player.** Eight seasons encode a
role a transfer has already invalidated. The largest movers are confident predictions in
the wrong direction: Kelleher (Liverpool's backup keeper for years, now Brentford's first
choice) 0.94 → 0.50; Ampadu 0.93 → 0.41; Rodon 0.86 → 0.38. These players have *more*
recent minutes than most, and deep history drags them down toward a role they left.

**The denominator does not mean what the repo's does.** `merged_gw.csv` carries a row per
gameweek for every player in the FPL game, whether or not they were in a matchday squad —
76.6% of player-seasons show exactly 38 "games", and **27.4% have zero appearances** while
still contributing 38 "did not start" events to the Beta denominator. The repo's own panel
is per-*match*, so its `games` means squad appearances. These are different quantities
wearing the same name.

The obvious fix makes it worse: excluding zero-appearance seasons (`deep, active-only`)
removes exactly the evidence that a fringe player *doesn't* play, and MAE for that group
more than doubles, 0.148 → 0.338.

### Coverage compounds it

The extra evidence lands on the players who least need it:

```
median weighted games gained
  thin in 25/26  (<900 min, n=502):   0.44
  thick in 25/26 (>=900 min, n=339): 23.50     <- 53x
```

`multiseason.py` showed the prior stops mattering after 6–10 games. Deep history
concentrates its evidence on players who are already past that threshold.

### What is true, and does not rescue it

Within deep-history variants, decay genuinely helps: on a common n=609, half-life 1.5
gives MAE 0.1975 against 0.2043 for flat pooling, and aggressive decay (0.25) is worst at
0.2174. So older seasons *do* carry information — just not information the model lacks.
`DEEP_HALF_LIFE = 1.5` is set from this, and matters only if the extension is ever used.

**Methodological note.** A first pass ranked half-life 0.25 best. That was an artefact:
each half-life was scored on whoever cleared a weighted-games floor *under that half-life*,
so aggressive decay dropped the thinnest players and won on an easier sample. Fixing the
evaluation set to a common 609 players reversed the ordering — 0.25 went from best to
worst. The sweep is now run on a fixed player set.

---

## 3. `[VERIFIED]` What cannot be backfilled at all

`to_priors` builds four priors. They do not backfill equally, and treating them alike
would fabricate evidence:

| prior | seasons available | status |
|---|---|---|
| `start_a`/`start_b` (Beta, minutes) | all 10 | available — but a null, §2 |
| `npxgi` / `xa` (Gamma) | xG only 2022/23+ | **blocked**, see below |
| `defcon` (Gamma) | **2025/26 only** | impossible |

**DefCon cannot be backfilled**, because the statistic itself is new — FPL introduced
DefCon scoring in 2025/26. Nine seasons of history buy exactly nothing. A proxy from the
tackles/CBI/recoveries columns fails on a second problem: those exist for 2016/17–2018/19
and 2025/26 but are **absent for the six seasons between**, so any proxy would cross a
definitional change with a six-season hole in the middle.

**xG is blocked on the penalty split.** `to_priors` is explicit that its Gamma uses
*non-penalty* xG (`build_2425_panel` does `npxg = xg − 0.79·pen_attempts`).
`merged_gw.csv` carries `penalties_missed` but **not `penalties_scored`**, so attempts
cannot be recovered, and its `expected_goals` is penalty-inclusive. Feeding it in raw
would inflate the npxGI prior for every penalty taker — precisely the players the model is
most sensitive to. `fpl_history` loads these columns and flags them (`has_xg`);
deliberately nothing consumes them.

---

## 4. `[CHECK]` Schema drift — the same filename is three different files

`merged_gw.csv` is not one schema. Handled in `fpl_history` by deriving what is derivable
and **flagging** what is derived, never by silently filling:

| era | `position`/`team` | `starts` | xG | defensive cols | encoding |
|---|---|---|---|---|---|
| 2016/17–2018/19 | absent | absent | absent | present | **latin-1** |
| 2019/20 | absent | absent | absent | absent | utf-8 |
| 2020/21–2021/22 | present | absent | absent | absent | utf-8 |
| 2022/23–2024/25 | present | present | present | absent | utf-8 |
| 2025/26 | present | present | present | present | utf-8 |

- `position` falls back to `players_raw.element_type` (stable in every season).
- `starts` falls back to `minutes >= 60`, which is what `multiseason_priors` and
  `build_pms` already use as their own definition of a start — the fallback matches the
  project's existing convention rather than introducing a new one.
- **2016/17 and 2017/18 are latin-1.** Nothing marks this. `_read_csv` tries strict UTF-8
  first so genuine mojibake surfaces instead of being silently mis-decoded. (Same class of
  bug as the cp1252 write that was destroying `solio_cache.md`.)

`xP`, `ep_this` and `ep_next` are dropped on read — FPL's own forward-looking projections,
not available pre-deadline in the form recorded. The known all-zero GW35 `xP` bug is moot
once the column is gone.

---

## 5. `[CHECK]` The Understat claim, corrected

Understat depth is real: **12 EPL seasons resolve, 2014/15 → 2025/26**, shot-level with
coordinates and per-shot xG. `sd_ingest` already reads it and currently caches 24/25+25/26.

But the proposal's claim that vaastav "already does the Understat-to-FPL-ID merge you'd
otherwise have to build yourself" **does not hold for the current repo layout**. There is
no `id_dict.csv` and no id map anywhere in the clone. What ships is raw Understat data in
`{season}/understat/`, and its coverage is patchy and inconsistent:

```
2016-17..2018-19    0 files
2019-20             22   (team-level)
2020-21             24   (team-level)
2021-22            736   (player-level)
2022-23            575
2023-24            798
2024-25            789
2025-26              0
```

Neither the earliest seasons nor the current one are covered, and the unit changes
mid-way. `sd_ingest`'s direct pull is the better spine — uniform across all 12 seasons —
and the Understat→`player_code` join remains the crosswalk problem it always was (G5),
still requiring hand-verification of `data/crosswalk_review.csv`.

---

## 6. What landed

| Module | Purpose | Selftest |
|---|---|---|
| `src/fpl_history.py` | vaastav player-GW loader, `player_code`-keyed, era-aware | ✅ offline, synthetic 3-era fixture |
| `src/multiseason_priors.deep_start_evidence` | decayed deep evidence — **null, not wired in** | via study |
| `studies/deep_history_study.py` | coverage, magnitude, half-life, deep-vs-shallow | ✅ runs |
| `config.HISTORY` / `FPL_HISTORY` | optional path, fails loudly when needed and absent | — |

253,900 player-GW rows across 10 seasons, 2,643 distinct players, 100% code match in
every season.

```bash
python src/fpl_history.py --selftest
```

---

## 7. Outstanding

1. **Do not enable `deep_starts`.** §2 is a null. If revisited, the estimand has to change
   — model start rate as role-at-club (with a transfer/promotion indicator), not as a
   player-intrinsic rate. That is a different model, not a deeper prior.
2. The reusable win is the loader and the `code == player_code` identity. Candidate uses
   that do **not** inherit §2's role problem: price/ownership dynamics, and the
   `starter_prior` ownership→start calibration, which currently fits on 25/26 alone and
   could fit on ten.
3. Understat pre-2022 xG depth stays blocked behind the crosswalk hand-verification and
   the penalty split (§3), in that order.
