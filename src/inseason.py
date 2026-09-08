from __future__ import annotations
import config
"""
inseason.py — let 26/27 results update the model. OFF by default.
==================================================================
Until now nothing in this repo read a 26/27 result: `TeamModel` fits reconstructed
25/26, player priors pool 24/25+25/26, and the only module touching the current season
reads friendlies. After a gameweek the model still projected from last season
(docs/GW1_SCORING_2026-08-22.md §4). This is the update path.

WHAT THE CALIBRATION SAYS  (studies/inseason_weight.py, 24-31 seasons)
-----------------------------------------------------------------------
Two findings drive every choice here.

1. PROCESS BEATS RESULTS, and it is not close. Predicting rest-of-season goal difference
   from prior-season strength plus a club's first three matches, out-of-sample R2:

       prior alone                     0.574
       prior + early GOAL difference   0.581
       prior + early SHOT difference   0.618

   Put both in one model and the goal term dies: early_goals +0.027 (p=0.334) against
   early_shots +0.041 (p=0.000). **Once you know how a club has been playing, its
   scorelines add nothing.** At k=1 adding goal difference is actively WORSE than
   ignoring the season entirely (0.577 vs 0.578).

   Consequence: the team channel updates on **xG**, not on goals. `TeamModel` takes its
   Poisson response from FTHG/FTAG, so this module writes xG into those columns — which
   is exactly what `betting_odds_ingest.fixtures_to_e0` already does with market lambda,
   so no new convention and no change to the fit.

2. THE WEIGHT IS SMALL AND IT GROWS SLOWLY. Optimal blend weight on a club's own early
   matches, (1-w)*prior + w*early, chosen out-of-sample:

       k       1     2     3     5     8    10
       w    0.01  0.03  0.07  0.13  0.21  0.25

   At k=3 that is w=0.07 against a 38-match prior, i.e. roughly ONE prior-season match
   of evidence per new match. Hence W_MATCH = 1.0. xG is a lower-variance measurement of
   the same quantity than goals are, so if anything it earns slightly more; 1.0 is the
   conservative read of a goals-fitted number and is exposed rather than buried.

   PROMOTED CLUBS move faster — w = 0.12 at k=3, 0.30 at k=5 — because they have no
   prior PL season at all, only a base rate. `W_MATCH_PROMOTED` is separate for that
   reason. Their own results are the only club-specific evidence that exists.

3. MINUTES get full weight. A realised start is not a proxy for the thing being
   estimated, it IS a draw from it, so the Beta prior takes a straight conjugate update.
   GW1 scoring found this is where the model's skill actually lives (r=0.51 overall
   falling to 0.26 among players who appeared), and where its worst errors were: eight
   players projected 3.0+ played zero minutes.

4. RATES ARE NOW ON, at W_RATE = 1.0  [fitted 2026-08-27, studies/inseason_rate_weight.py]
   This channel shipped disabled because no calibration existed. It now does. Holding out
   one season and fitting the weight on the other, both directions, a conjugate update of
   the attacking Gamma with the current season's events beats W_RATE = 0 at every horizon
   tested for npxG and at three of four for xA:

       npxG/90   k= 3  MAE 0.06270 -> 0.05999   CI (-0.00512, -0.00029)
                 k= 5       0.06464 -> 0.06166  CI (-0.00550, -0.00045)
                 k= 8       0.06386 -> 0.06109  CI (-0.00506, -0.00049)
                 k=12       0.06624 -> 0.06294  CI (-0.00645, -0.00015)
       xA/90     k= 3       0.04877 -> 0.04840  CI (-0.00108, +0.00034)  <- does not clear
                 k= 5       0.04955 -> 0.04769  CI (-0.00337, -0.00035)
                 k= 8       0.05031 -> 0.04808  CI (-0.00433, -0.00013)
                 k=12       0.05318 -> 0.04992  CI (-0.00536, -0.00117)

   The selected weight wanders between 0.88 and 1.75 across horizons, which is noise
   around one: a current-season match is worth about one prior-season match. That is the
   same answer the TEAM channel reached independently, and 1.0 is the round number both
   sets of CIs comfortably contain.

   EACH CHANNEL IS GATED WHERE IT WAS SHOWN TO WORK  [split 2026-09-06]
   npxG clears the rule at k=3; xA does not until k=5. Those are different numbers and
   they are now applied as different gates — `MIN_RATE_MATCHES_NPXG = 3`,
   `MIN_RATE_MATCHES_XA = 5`. A single gate at 5 was the conservative reading, but it was
   set by the WEAKER channel: it withheld npxG for two gameweeks in which npxG had been
   measured to help, because a different quantity was noisy. Neither threshold is new —
   both come straight out of the table above. `MIN_RATE_MATCHES` remains as an alias for
   the stricter one so existing callers are unchanged.

   Power is thin and stated: Understat match-level data covers two seasons, so the
   hold-out is one season each way.

MEASURED EFFECT, AND A CAVEAT ABOUT READING IT  [2026-08-23]
--------------------------------------------------------------
Turned on after GW1 (6 finished matches, 12 clubs), minutes channel only:

    clubs that had played      n=364   mean |change| 0.114 pts   max 0.60
    clubs that had NOT played  n=240   mean |change| 0.049 pts   max 0.25

Direction is right. Coventry's XI rises (Onyeka +0.45, Amenda +0.41, Dasilva +0.36);
players who sat out fall (Kitching -0.60, Targett -0.48). Promoted-club squads move most,
which is what the calibration predicts: their priors were the weakest.

The second row should be ZERO and is not. It is NOT leakage — `update_minutes` skips any
club with `club_matches == 0`, asserted in the selftest and verified directly on the live
frame. It is **Monte Carlo re-randomisation, and it is pre-existing**:

    perturbing ONE Arsenal player's start_a and re-projecting moves 48 players,
    19 of them at other clubs, mean |change| 0.080

`project()` draws sequentially from the module-level `bayes_model.rng`, and the number of
draws consumed per player depends on that player's parameters. Change one player and the
stream shifts for everyone drawn after him. The board is still deterministic given
identical input — `test_all`'s determinism check passes — but it is not stable to a LOCAL
perturbation.

**FIXED 2026-08-24.** `project()` now seeds a generator per player from `player_code`
(`bayes_model._player_rng`), so player j's draws cannot depend on player i's parameters.
Re-measured: perturbing one player now moves exactly one player, and the board is
additionally invariant to shuffling or dropping rows — neither of which held before. The
distribution is unchanged (league total 1795.2 -> 1796.5 at S=3000, per-player mean
|delta| 0.061, pure MC noise). `tests/test_rng_isolation.py` locks the properties down.

So the second row above is a MEASUREMENT TAKEN UNDER THE OLD BEHAVIOUR and would now read
zero. It is kept because it is the evidence that motivated the fix.

WHAT THIS DELIBERATELY DOES NOT DO
-----------------------------------
No form, streaks, momentum or confidence terms. Rotation multipliers (p=0.23) and
directional mean-reversion (p=0.69) are tested nulls in this repo and stay dead; nothing
here reintroduces them under the name "in-season update". This module changes only how
much EVIDENCE the existing estimators see, never their functional form.

ONE AMENDMENT TO THAT  [2026-09-07, studies/start_persistence.py + start_recency.py]
------------------------------------------------------------------------------------
`recency_starts` weights realised matches by recency, which changes not how much evidence
the minutes estimator sees but how that evidence is DISTRIBUTED IN TIME. Total mass is
renormalised to the raw match count, so it is a pure reallocation and lam = 1 is the old
behaviour exactly — but the claim "never their functional form" no longer holds without
qualification, and the distinction from the dead heuristics has to be stated rather than
assumed:

  A ROTATION MULTIPLIER predicted WHICH gameweeks a given player would be rested, from
  fixture congestion. That is dead three ways over (rest differential, European
  participation, midweek fixtures by recovery day; P(start) +0.001, CI +/-0.02) and
  nothing here touches it — this module still has no fixture-conditional term.
  A MOMENTUM TERM would assert that a run of starts predicts MORE than the player's own
  rate. That is also dead: measured against a frailty-preserving permutation null, the
  streak excess is zero from k=10 on, which is exactly why there is no `streak_k`
  covariate here and why adding one would double-count the Beta prior.

What survives measurement is narrower than either: the ORDER of a player's own realised
matches is informative over a window of roughly six, and a Beta update on a count is
exchangeable and therefore blind to it. The fix is a weight on the likelihood's own
observations, not a new predictor. It is OFF by default regardless.

Run:  python src/inseason.py --selftest
      python src/inseason.py --report          (what 26/27 currently supplies)
"""
import io
import os
import glob
import numpy as np
import pandas as pd
import betting_odds_ingest as boi

# --- pre-registered weights, from studies/inseason_weight.py -----------------
W_MATCH = 1.0            # prior-season matches per 26/27 xG match, established clubs
W_MATCH_PROMOTED = 2.0   # promoted clubs have no prior season; w rises ~2x as fast
W_MINUTES = 1.0          # a start is a direct Bernoulli observation, not a proxy
W_RATE = 1.0             # fitted; see studies/inseason_rate_weight.py
# ONE GATE PER CHANNEL, each at the k where that channel actually cleared its rule.
# A single gate at 5 was set by the WEAKER of the two: npxG/90 beats w=0 at k=3 with
# CI (-0.00512, -0.00029), xA/90 does not, CI (-0.00108, +0.00034). Holding both at 5
# suppressed npxG for two gameweeks in which it had been shown to help, on the strength
# of a different channel's noise. Splitting them is not a relaxation — each number is
# the threshold `studies/inseason_rate_weight.py` already fitted for that quantity.
MIN_RATE_MATCHES_NPXG = 3   # npxG/90 clears the decision rule from three matches
MIN_RATE_MATCHES_XA = 5     # xA/90 does not clear until five
MIN_RATE_MATCHES = MIN_RATE_MATCHES_XA   # back-compat: the stricter of the two

MIN_XG_MATCHES = 4       # below this the team channel is not worth the moving parts
PROMOTED_2627 = ("Coventry", "Hull", "Ipswich")


# ---------------------------------------------------------------- reading 26/27
def played(season="2026-2027", upto_gw=None, base=None, require_xg=True, verbose=True):
    """Finished 26/27 matches with team names and, optionally, xG.

    Only `finished` rows are read. A `finished_provisional` match can still have its
    stats revised, and a match in progress has partial numbers that would enter the fit
    as if they were a full game.
    """
    import core_insights as ci
    base = base or config.repo(season)
    teams = pd.read_csv(os.path.join(base, "teams.csv"))
    name = dict(zip(teams["code"].astype(float), teams["name"]))
    frames = []
    for d in sorted(glob.glob(os.path.join(base, "By Gameweek", "GW*"))):
        f = os.path.join(d, "matches.csv")
        if not os.path.exists(f):
            continue
        m = pd.read_csv(f)
        if not len(m):
            continue
        frames.append(m)
    if not frames:
        return pd.DataFrame()
    M = pd.concat(frames, ignore_index=True)
    M = M[M.get("finished") == True]                                  # noqa: E712
    # PREMIER LEAGUE ONLY. `By Gameweek/GW*/matches.csv` also carries cup ties — the EFL
    # Cup second round lands inside the GW2 window — and they must never reach the team
    # model. A League Cup tie against lower-division opposition is not evidence about
    # Premier League strength, and the opponent does not even exist in the 20-club frame,
    # so its name resolves to NaN and poisons the fit. PROJECT_KNOWLEDGE §8 states the
    # rule: keep only match_id containing "-prem-".
    # Caught the hard way: without this, GW2 added five cup fixtures and Nott'm Forest v
    # Leeds appeared twice, once as a league match and once as a cup tie.
    if "match_id" in M.columns:
        n_all = len(M)
        M = M[M["match_id"].astype(str).str.contains("-prem-", na=False)]
        if verbose and len(M) < n_all:
            print(f"[inseason] dropped {n_all - len(M)} non-league matches "
                  f"(cup ties in the gameweek folders)")
    if upto_gw is not None and "gameweek" in M.columns:
        M = M[M["gameweek"] <= int(upto_gw)]
    if not len(M):
        return pd.DataFrame()
    M["home"] = M["home_team"].astype(float).map(name).map(ci.norm_team)
    M["away"] = M["away_team"].astype(float).map(name).map(ci.norm_team)
    M["xg_home"] = pd.to_numeric(M.get("home_expected_goals_xg"), errors="coerce")
    M["xg_away"] = pd.to_numeric(M.get("away_expected_goals_xg"), errors="coerce")
    M["date"] = pd.to_datetime(M.get("kickoff_time"), errors="coerce", utc=True)
    _unres = int(M["home"].isna().sum() + M["away"].isna().sum())
    if _unres and verbose:
        print(f"[inseason] WARNING {_unres} team names did not resolve to the 26/27 "
              f"frame and are dropped — check teams.csv coverage")
    out = M[["gameweek", "date", "home", "away", "home_score", "away_score",
             "xg_home", "xg_away"]].dropna(subset=["home", "away"])
    n_all = len(out)
    if require_xg:
        out = out.dropna(subset=["xg_home", "xg_away"])
    if verbose:
        print(f"[inseason] {n_all} finished matches, {len(out)} with xG"
              + ("" if require_xg else " (xG not required)"))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- team channel
def _fair_odds(lam_h, lam_a):
    """Decimal no-vig odds that encode (lam_h, lam_a).

    `TeamModel.fit` reads its prior means through `betting_features.build`, which recovers
    lambda from the 1X2 and over/under columns. So an appended row has to carry odds that
    invert back to the xG it is meant to represent, or the likelihood and the prior would
    describe different matches. Built with the same Poisson machinery the forward
    direction uses, and the round trip is asserted in the selftest.
    """
    h, d, a = boi._wdl(float(lam_h), float(lam_a))
    p_le2 = boi._p_total_le2(float(lam_h) + float(lam_a))
    over = max(min(1.0 - p_le2, 1 - 1e-6), 1e-6)
    f = lambda p: 1.0 / max(float(p), 1e-6)
    return f(h), f(d), f(a), f(over), f(1.0 - over)


def e0_rows(matches):
    """26/27 matches as E0-format rows carrying xG, ready to stack."""
    rows = []
    for _, r in matches.iterrows():
        oh, od, oa, ov, un = _fair_odds(r["xg_home"], r["xg_away"])
        rows.append({
            "Date": (r["date"].strftime("%d/%m/%Y") if pd.notna(r["date"]) else ""),
            "HomeTeam": r["home"], "AwayTeam": r["away"],
            # xG, NOT the scoreline — see finding 1 at the top of this module
            "FTHG": round(float(r["xg_home"]), 4),
            "FTAG": round(float(r["xg_away"]), 4),
            "AvgH": round(oh, 4), "AvgD": round(od, 4), "AvgA": round(oa, 4),
            "Avg>2.5": round(ov, 4), "Avg<2.5": round(un, 4),
        })
    return pd.DataFrame(rows)


def stack_e0(matches, e0_path=None, out=None, weight=W_MATCH,
             weight_promoted=W_MATCH_PROMOTED, promoted=PROMOTED_2627, verbose=True):
    """Prior-season E0 with 26/27 xG matches appended, returning the new path.

    Weighting is by row replication, which is exact for integer weights and needs no
    change to the Poisson fit — the same device `oddsapi_feed.stack_e0` uses to hit a
    target market share. A match involving a promoted club is replicated
    `weight_promoted` times instead: those clubs have no prior season, so the same
    absolute evidence has to move a far weaker prior.

    Fractional weights are NOT supported. They would need sample weights inside
    `TeamModel.fit`, and silently rounding one would be a quiet change to the very
    number this whole module is about.
    """
    e0_path = e0_path or config.E0_RECON
    base = pd.read_csv(e0_path)
    if not len(matches):
        if verbose:
            print("[inseason] no 26/27 matches with xG — team layer unchanged")
        return e0_path, 0
    for w, nm in ((weight, "weight"), (weight_promoted, "weight_promoted")):
        if abs(w - round(w)) > 1e-9 or w < 0:
            raise ValueError(f"{nm}={w} must be a non-negative whole number "
                             f"(replication cannot express a fraction)")
    new = e0_rows(matches)
    pro = set(promoted)
    reps = np.where(new["HomeTeam"].isin(pro) | new["AwayTeam"].isin(pro),
                    int(round(weight_promoted)), int(round(weight)))
    stacked = new.loc[np.repeat(new.index.values, reps)]
    both = pd.concat([base, stacked[base.columns]], ignore_index=True)
    out = out or os.path.join(config.SCRATCH, "E0_recon_inseason.csv")
    both.to_csv(out, index=False)
    if verbose:
        print(f"[inseason] team layer: {len(base)} prior rows + {len(stacked)} "
              f"26/27 rows from {len(new)} matches "
              f"(x{int(weight)}, x{int(weight_promoted)} for promoted) -> {out}")
    return out, len(stacked)


# ------------------------------------------------------------- minutes channel
def team_at_gw(season="2026-2027", base=None, gws=None):
    """player_id x gw -> the club the player was AT in that gameweek.

    Reads `team_history.csv`, which `core_insights` documents and
    `docs/DATA_SOURCE_AUDIT_2026-08-19.md` recorded as present but never consumed. It is
    the only per-gameweek team column in the repo, and without it every per-gameweek
    quantity is keyed on the CURRENT `players.csv` snapshot: a player who moved has his
    old club's appearances credited to his new one, and both clubs' per-gameweek start
    totals go wrong by one in opposite directions.

    `gws` fills the frame out to a fixed gameweek list, carrying each player's club
    forward and then BACKWARD. Backward is not cosmetic: a player who first appears in
    `team_history` at GW2 was still at a club in GW1, and `appearances` charges him a
    non-start for that gameweek, so it needs a club to charge it against. Carrying his
    earliest known club back reproduces the pre-existing denominator exactly rather than
    silently shortening it, which would be a different correction.

    Returns an empty frame when the file is absent; callers fall back to the snapshot.
    """
    import core_insights as ci
    base = base or config.repo(season)
    f = os.path.join(base, "team_history.csv")
    cols = ["player_id", "gw", "team"]
    if not os.path.exists(f):
        return pd.DataFrame(columns=cols)
    teams = pd.read_csv(os.path.join(base, "teams.csv"))
    tname = dict(zip(teams["code"], teams["name"].map(ci.norm_team)))
    th = pd.read_csv(f)
    th["team"] = th["team_code"].map(tname)
    th = th.dropna(subset=["team"])
    if not len(th):
        return pd.DataFrame(columns=cols)
    if gws is not None:
        keep = sorted({int(g) for g in gws})
        if not keep:
            return pd.DataFrame(columns=cols)
        P = (th.pivot_table(index="player_id", columns="gw", values="team", aggfunc="last")
               .reindex(columns=keep).ffill(axis=1).bfill(axis=1))
        P.columns.name = "gw"
        th = P.stack().rename("team").reset_index()
    th["gw"] = pd.to_numeric(th["gw"], errors="coerce")
    return th.dropna(subset=["gw"])[cols].astype({"gw": int})


def _finished_clubs(m, name):
    """Clubs in one gameweek's matches.csv with a finished LEAGUE match and no unfinished
    one, as normalised names.

    Both halves are load-bearing and neither is about cup ties (see `gw_panel`). The
    first admits a club only on evidence that exists; the second holds out a club whose
    gameweek is not over, because `player_gameweek_stats` sums a double gameweek into one
    row and a finished match cannot be separated from a live one at that grain.
    """
    if not len(m) or "finished" not in m.columns:
        return []
    lg = m
    if "match_id" in lg.columns:
        lg = lg[lg["match_id"].astype(str).str.contains("-prem-", na=False)]
    if not len(lg):
        return []
    fin = lg["finished"] == True                                          # noqa: E712

    def clubs(x):
        return set(pd.concat([x["home_team"], x["away_team"]])
                     .astype(float).map(name).dropna())

    return sorted(clubs(lg[fin]) - clubs(lg[~fin]))


def gw_panel(season="2026-2027", upto_gw=None, base=None):
    """Per player-gameweek: player_code, team, team_now, gw, minutes, starts — for
    FINISHED gameweeks only.

    Extracted from `appearances` so that anything else needing the per-gameweek grain
    reads it through one loader rather than re-globbing the directory tree.

    THE FINISHED GUARD IS PER MATCH, NOT PER GAMEWEEK  [2026-09-08]
    It used to admit a whole gameweek as soon as ANY match in it had finished, which
    defeated the purpose it was written for. `player_gameweek_stats.csv` is a snapshot of
    live FPL element data, so a match still in progress is ALREADY IN IT with partial
    numbers. 26/27 GW3 caught it: the file was pulled while Arsenal-Chelsea was at half
    time, and all 22 of those starters carry exactly 45 minutes with no substitute used,
    against 985-989 minutes and 14-16 players used for each of the eighteen clubs whose
    match had finished. Nine finished matches were enough to let the tenth in, so the
    panel counted 220 starts against 9 finished matches — breaching `starts == 11 x
    finished league matches` for Arsenal and Chelsea, and pushing 16 players past
    `starts <= club_matches`, which `update_minutes` had been silently clamping. Half-time
    numbers are worse than the row of zeros the old guard worried about, because they
    read as real evidence rather than as an obvious gap.

    A club is admitted for a gameweek only when it has a finished LEAGUE match in it and
    no unfinished one (`_finished_clubs`).

    THIS IS NOT ABOUT CUP TIES, and an earlier note here and in `recency_starts` saying
    it was is wrong. `player_gameweek_stats` is league-only: 26/27 GW2 carried ten EFL
    Cup ties alongside its ten league matches and the file still holds exactly 22 x 10
    starts. A cup tie has never been able to reach this panel.

    `team` IS THE CLUB HE PLAYED FOR THAT WEEK (`team_at_gw`), not the club he is at
    now. Keying the panel on the current `players.csv` snapshot credited a transferred
    player's old-club appearances to his new club, which showed up directly as a breach
    of the per-(club, gameweek) invariant `starts == 11 x finished league matches`
    (measured 2026-09-07 on 26/27: Man City 12 and Everton 10 in GW1 and GW2, Nott'm
    Forest 12 and Crystal Palace 10 in GW1 — Ndiaye and Munoz moving in each direction).
    The breaches net to zero across the league because it is a misattribution, not
    missing data, and re-keying on the per-gameweek club clears all six. It matters twice
    downstream: `recency_starts` places a start by the LEAGUE fixture its (team, gw) cell
    points at, and `appearances` bounds a player's starts by his club's match count. Both
    denominators were the wrong club's.

    `team_now` is the snapshot club, kept because it is the right label for a player and
    the right grouping key for anything that must stay one row per player.
    """
    import core_insights as ci
    base = base or config.repo(season)
    roster = pd.read_csv(os.path.join(base, "players.csv"))
    teams = pd.read_csv(os.path.join(base, "teams.csv"))
    roster = roster.merge(teams[["code", "name"]], left_on="team_code",
                          right_on="code", how="left")
    roster["team"] = roster["name"].map(ci.norm_team)
    name = dict(zip(teams["code"].astype(float), teams["name"].map(ci.norm_team)))
    frames, admissible = [], []
    for d in sorted(glob.glob(os.path.join(base, "By Gameweek", "GW*"))):
        f = os.path.join(d, "player_gameweek_stats.csv")
        mf = os.path.join(d, "matches.csv")
        if not (os.path.exists(f) and os.path.exists(mf)):
            continue
        tag = os.path.basename(d)[2:]
        if not tag.isdigit():        # no gameweek number, so no fixture to check against
            continue
        gw = int(tag)
        if upto_gw is not None and gw > int(upto_gw):
            continue
        mm = pd.read_csv(mf)
        done = _finished_clubs(mm, name)
        if not done:
            continue
        admissible.append(pd.DataFrame({"gw": gw, "team": done}))
        g = pd.read_csv(f)
        g["gw"] = gw
        frames.append(g[["id", "gw", "minutes", "starts"]])
    if not frames:
        return pd.DataFrame(columns=["player_code", "team", "team_now", "gw",
                                     "minutes", "starts"])
    G = pd.concat(frames, ignore_index=True)
    G = G.merge(roster[["player_id", "player_code", "team"]].rename(
                    columns={"team": "team_now"}),
                left_on="id", right_on="player_id", how="left").dropna(subset=["player_code"])
    TH = team_at_gw(season, base)
    if len(TH):
        G = G.merge(TH, on=["player_id", "gw"], how="left")
        # no history row (a player registered after the gameweek) -> the snapshot club is
        # the only answer available, and is right for everyone who never moved
        G["team"] = G["team"].fillna(G["team_now"])
    else:
        G["team"] = G["team_now"]
    # drop anyone whose club's gameweek is not over — his row is a live snapshot, not a
    # result. Keyed on the per-gameweek club, so a player who moved is judged on the
    # fixture he actually played in.
    OK = pd.concat(admissible, ignore_index=True).assign(_done=True)
    G = G.merge(OK, on=["team", "gw"], how="left")
    return G[G["_done"] == True].drop(columns=["_done"])                  # noqa: E712


def last_appearance(season="2026-2027", upto_gw=None, base=None):
    """Per player_code: his club's most recent FINISHED gameweek, and what he did in it.

    Reported for every player on a club that has played, including the ones who did not
    get on — `minutes = 0` with `played = False` is the informative case, not a gap. A
    missing row means the club itself has no finished gameweek.
    """
    G = gw_panel(season, upto_gw, base)
    cols = ["player_code", "team", "gw", "minutes", "starts"]
    if not len(G):
        return pd.DataFrame(columns=cols)
    # A player's own club's last finished gameweek, not the league's. Keyed on the club
    # he is at NOW, because that is the club whose next fixture he is being reported for;
    # the row that survives is his latest, where the two agree in any case.
    last_gw = G.groupby("team_now")["gw"].transform("max")
    L = G[G["gw"] == last_gw].copy()
    L = L.sort_values(["player_code", "gw"]).drop_duplicates("player_code", keep="last")
    L["team"] = L["team_now"]
    return L[cols]


# --- recency weighting of realised starts (studies/start_recency.py) ------------
# A conjugate Beta update reads a COUNT, so it is EXCHANGEABLE: start-start-bench and
# bench-start-start produce an identical posterior. `studies/start_persistence.py`
# measured that this is the wrong likelihood. Against a frailty-preserving permutation
# null (each player's own start count held fixed, only the ORDER destroyed), the excess
# in P(start at t+1 | k consecutive starts) is +0.190 at k=1, +0.153 at k=3, +0.080 at
# k=6, +0.030 at k=8 and indistinguishable from zero from k=10 on. Ordering carries real
# information and it decays out over roughly six matches.
#
# The encoding is a geometric recency weight, NOT a streak covariate: the excess is
# already zero past k~8, so a streak term would mostly re-encode the player's base rate,
# which the Beta prior already holds. It also picks up the asymmetry the persistence
# study found for free — P(start next | 3 benchings) = 0.114 against P(start next | 3
# starts) = 0.800 — because recent zeros dominate a weighted count automatically.
#
# LAM IS A FUNCTION OF FORECAST HORIZON, which is why it is exposed and not pinned.
# `start_recency.py`, leave-one-season-out over 22/23-25/26 at kappa=4:
#
#     horizon (matches)     1     2     3     5    10   rest
#     lam*               0.40  0.50  0.55  0.65  0.75  0.82
#     LOSO Brier gain   +7.9% +5.6% +4.0% +2.6% +1.2% +0.3%
#     folds positive     4/4   4/4   4/4   4/4   4/4    3/4
#
# Short horizon, short memory. The board projects ~10 gameweeks, so 0.75 is its value and
# `RECENCY_LAM_H10` names it; the one-match number is much lower and must NOT be used for
# a horizon board, which is the whole reason the constant is horizon-tagged. Rest-of-
# season fails the 1% gate, so nothing licenses this for a season-long projection.
#
# OFF by default (lam = 1.0 reproduces the flat update EXACTLY, weight for weight), for
# two reasons: it was fitted at kappa=4, which `INSEASON_KAPPA` itself leaves off, and at
# the uncapped production prior lam* pins to the grid boundary rather than fitting.
# `INSEASON_LAM=0.75` turns it on and should be set together with `INSEASON_KAPPA=4`.
RECENCY_LAM = 1.0
RECENCY_LAM_H10 = 0.75


def recency_starts(G, matches, lam=RECENCY_LAM):
    """Recency-weighted start count per (player_code, team).

    Weight u_d = lam**d on the club's d-th most recent gameweek (d = 0 is the latest),
    renormalised so the total weighted MATCH mass equals the club's actual match count.
    Total evidence is therefore unchanged and only its ORDER is re-weighted — which is
    the only thing `start_recency.py` identified, and it keeps this from confounding with
    W_MINUTES / START_KAPPA, whose ridge `start_prior_strength.py` already fitted.

    Consequences of that renormalisation, both wanted: the result is bounded by the
    club's match count exactly as a raw count is, so `update_minutes` keeps its invariant;
    and lam = 1 returns the raw LEAGUE start count to floating-point equality.

    "League" is load-bearing. A start is placed by the league fixture it belongs to, so a
    start with no finished league fixture behind it has no recency position and cannot
    enter. That used to be a live discrepancy: `gw_panel` admitted a whole gameweek as
    soon as any match in it had finished, so 26/27 GW3 carried 22 half-time starts from
    an unfinished Arsenal-Chelsea that `played` correctly excluded, and the raw `starts`
    column disagreed with its own denominator for 16 players — masked all along by
    `update_minutes` clamping to `club_matches`. `gw_panel` now applies the finished
    guard PER MATCH, so the two agree at source and lam = 1 is a true identity on the raw
    count. The note that used to stand here blamed EFL Cup ties; that was wrong, and
    `gw_panel` records why.

    A double gameweek is one gameweek carrying two matches, so both of its matches share
    the one weight. Ordering WITHIN a double gameweek is not resolvable at this grain and
    is not worth resolving: it is two adjacent matches sharing a recency position.
    """
    cols = ["player_code", "team", "w_starts"]
    if not len(G) or not len(matches):
        return pd.DataFrame(columns=cols)
    lam = float(lam)
    mg = (pd.concat([matches[["gameweek", "home"]].rename(columns={"home": "team"}),
                     matches[["gameweek", "away"]].rename(columns={"away": "team"})])
            .groupby(["team", "gameweek"]).size().rename("m").reset_index())
    scaled = []
    for _team, t in mg.groupby("team"):
        t = t.sort_values("gameweek")
        d = np.arange(len(t) - 1, -1, -1, dtype=float)        # 0 = most recent
        u = lam ** d
        mass = float((u * t["m"].to_numpy()).sum())
        if mass <= 0:
            continue
        t = t.assign(u=u * (float(t["m"].sum()) / mass))
        scaled.append(t)
    if not scaled:
        return pd.DataFrame(columns=cols)
    U = pd.concat(scaled, ignore_index=True)
    if "team_now" not in G.columns:
        G = G.assign(team_now=G["team"])
    # `team` here is the club he played for THAT week, so a start lands on the recency
    # position of the fixture it actually belongs to; the result is keyed on `team_now`
    # so a player who moved stays ONE row and keeps his whole record.
    g = G.merge(U[["team", "gameweek", "u"]], left_on=["team", "gw"],
                right_on=["team", "gameweek"], how="left")
    g["u"] = g["u"].fillna(0.0)
    g["ws"] = g["u"] * pd.to_numeric(g["starts"], errors="coerce").fillna(0.0)
    return (g.groupby(["player_code", "team_now"], as_index=False)["ws"].sum()
             .rename(columns={"ws": "w_starts", "team_now": "team"}))


def appearances(season="2026-2027", upto_gw=None, base=None, verbose=True,
                lam=RECENCY_LAM):
    """Per player_code: starts, appearances, and how many club matches were available.

    `starts` is read from the repo rather than inferred from a minutes threshold, so a
    60-minute substitute is not miscounted as a starter.

    `w_starts` is the same quantity with realised matches discounted by recency at `lam`
    (see `recency_starts`), and is attached ONLY when `lam != 1.0` so that the default
    frame is unchanged. It is a SEPARATE column rather than a replacement because
    `update_exp_minutes` divides `start_minutes` by `starts` to get minutes-per-start,
    and a re-weighted denominator there would be a different estimand, not a better one.
    """
    G = gw_panel(season, upto_gw, base)
    if not len(G):
        return pd.DataFrame(columns=["player_code", "team", "starts", "apps", "club_matches"])
    # A club's completed matches bound how many starts were even possible — but for a
    # player who moved it is not ONE club's count. Summing per gameweek over the club he
    # was at that week gives him the matches HE could have started, which is the honest
    # denominator for the Beta update and the bound `update_minutes` clamps to. Every
    # club has played exactly once per gameweek so far, so this reproduces the old
    # per-club count for every player in 26/27 to date; it diverges the moment a blank or
    # double gameweek straddles a transfer, which is exactly when the old count is wrong.
    pl = played(season, upto_gw, base=base, require_xg=False, verbose=False)
    cm = pd.concat([pl["home"], pl["away"]]).value_counts().rename("club_matches")
    mg = (pd.concat([pl[["gameweek", "home"]].rename(columns={"home": "team"}),
                     pl[["gameweek", "away"]].rename(columns={"away": "team"})])
            .groupby(["team", "gameweek"]).size().rename("m").reset_index())
    TH = team_at_gw(season, base, gws=mg["gameweek"].unique()) if len(mg) else mg.head(0)
    cmp_ = None
    if len(TH):
        code = G.drop_duplicates("player_id").set_index("player_id")["player_code"]
        cmp_ = (TH.merge(mg, left_on=["team", "gw"], right_on=["team", "gameweek"],
                         how="left")
                  .assign(player_code=lambda d: d["player_id"].map(code))
                  .dropna(subset=["player_code"])
                  .groupby("player_code", as_index=False)["m"].sum()
                  .rename(columns={"m": "club_matches"}))
        cmp_ = cmp_ if len(cmp_) else None
    # Minutes accrued IN MATCHES HE STARTED, kept apart from total minutes. Dividing
    # total minutes by starts is wrong for anyone who both started some matches and came
    # off the bench in others: the substitute minutes land in the numerator with no start
    # in the denominator, and the ratio can exceed 90. Nothing in this module consumes it
    # yet — `update_minutes` is a Beta update on STARTS alone and does not touch
    # `exp_minutes` — but it is the quantity any future minutes-per-start channel needs,
    # and it is the honest denominator for reporting one.
    G["start_minutes"] = G["minutes"].where(G["starts"] > 0, 0)
    A = (G.groupby(["player_code", "team_now"], as_index=False)
           .agg(starts=("starts", "sum"), apps=("minutes", lambda s: int((s > 0).sum())),
                minutes=("minutes", "sum"), start_minutes=("start_minutes", "sum"))
           .rename(columns={"team_now": "team"}))
    if cmp_ is not None:
        A = A.merge(cmp_, on="player_code", how="left")
    else:
        A["club_matches"] = np.nan
    # no team_history row at all -> fall back to the club he is at now, as before
    A["club_matches"] = A["club_matches"].fillna(A["team"].map(cm))
    A["club_matches"] = A["club_matches"].fillna(0).astype(int)
    # At lam = 1 the column is not attached AT ALL, so `update_minutes` falls through to
    # the raw count and the flag-off board is byte-identical to before this existed. That
    # gate used to carry a second job: `recency_starts` drops a start with no finished
    # league fixture behind it, and `gw_panel` was feeding it 22 such starts from the
    # unfinished 26/27 GW3 Arsenal-Chelsea, so turning recency on silently applied a
    # SEPARATE correction too. `gw_panel` now applies its finished guard per match, so
    # there is nothing left to drop and lam = 1.0 really is the identity here. The gate
    # stays because the raw path is cheaper and byte-identity is worth keeping, not
    # because the two answers still differ.
    if float(lam) != 1.0:
        W = recency_starts(G, pl, lam)
        A = A.merge(W, on=["player_code", "team"], how="left")
        A["w_starts"] = A["w_starts"].fillna(0.0).clip(lower=0.0)
        A["w_starts"] = np.minimum(A["w_starts"], A["club_matches"].astype(float))
    if verbose:
        print(f"[inseason] minutes: {len(A)} players, {int(A.starts.sum())} starts "
              f"over {len(pl)} club-matches"
              + ("" if float(lam) == 1.0 else
                 f"; recency lam={float(lam):g} (half-life "
                 f"{np.log(0.5) / np.log(float(lam)):.1f} matches), weighted starts "
                 f"{A.w_starts.sum():.1f} (same total, reweighted by recency)"))
    return A



def update_minutes(players, apps, weight=W_MINUTES, verbose=True):
    """Conjugate Beta update of the start prior from realised starts.

    start_a += w * starts,  start_b += w * (club matches - starts)

    `starts` is the recency-weighted count (`w_starts`) when `appearances` produced one,
    otherwise the raw count. The two are identical at lam = 1.

    Keyed on `player_code`. A player is only charged non-starts for matches his club
    actually played, so a mid-window signing is not penalised for games that happened
    before he arrived.

    This runs BEFORE `signals.apply_availability`, which overrides the prior outright for
    anyone currently flagged. Order matters: an injury today should beat an appearance
    record, not be averaged with it.

    THE CLAMP ON `starts` IS A DEFENCE, NOT A CORRECTION  [2026-09-08]
    `gw_panel` now applies the finished guard per match and keys the panel on the club a
    player actually played for, so `starts <= club_matches` holds at source (verified on
    26/27: 0 breaches over 653 players, where the old panel had 16). The clamp stays for
    the cases that guard cannot see — a player with no `team_history` row falls back to
    his snapshot club — but it must SAY when it fires. Silently repairing the frame is
    how 22 half-time starts survived unnoticed for a day. It warns rather than raises
    because a partial data pull is a transient the nightly board should survive.
    """
    p = players.copy()
    if not len(apps) or "player_code" not in p.columns:
        if verbose:
            print("[inseason] minutes: nothing to apply")
        return p, pd.DataFrame()
    a = apps.dropna(subset=["player_code"]).drop_duplicates("player_code")
    scol = "w_starts" if "w_starts" in a.columns else "starts"
    m = dict(zip(a["player_code"], zip(a[scol], a["club_matches"])))
    before = p["start_a"] / (p["start_a"] + p["start_b"])
    rows, breaches = [], []
    for i, code in p["player_code"].items():
        if code not in m:
            continue
        starts, cm = m[code]
        starts = float(starts); cm = float(cm)
        if cm <= 0:
            continue
        if starts > cm + 1e-9:                      # invariant gw_panel should guarantee
            breaches.append((code, starts, cm))
        starts = min(starts, cm)                    # cannot start more than were played
        p.at[i, "start_a"] = float(p.at[i, "start_a"]) + weight * starts
        p.at[i, "start_b"] = float(p.at[i, "start_b"]) + weight * (cm - starts)
        rows.append({"player_code": code, "starts": starts, "club_matches": cm})
    if breaches:
        print(f"[inseason] WARNING {len(breaches)} players had starts > club_matches, "
              f"which `gw_panel` is supposed to make impossible — clamped, but the panel "
              f"is wrong, not just this update. Worst: "
              + ", ".join(f"{c} {s:.1f}/{m:.0f}" for c, s, m in
                          sorted(breaches, key=lambda b: b[2] - b[1])[:3]))
    after = p["start_a"] / (p["start_a"] + p["start_b"])
    rep = pd.DataFrame(rows)
    if len(rep):
        rep = rep.merge(pd.DataFrame({"player_code": p["player_code"],
                                      "web_name": p.get("web_name"),
                                      "p_before": before, "p_after": after}),
                        on="player_code", how="left")
        rep["delta"] = rep["p_after"] - rep["p_before"]
    if verbose and len(rep):
        big = rep.reindex(rep["delta"].abs().sort_values(ascending=False).index).head(8)
        print(f"[inseason] minutes: updated {len(rep)} players; largest moves:")
        for _, r in big.iterrows():
            print(f"    {str(r.get('web_name','?')):20s} {r['starts']:.0f}/"
                  f"{r['club_matches']:.0f} starts   "
                  f"p_start {r['p_before']:.2f} -> {r['p_after']:.2f} "
                  f"({r['delta']:+.2f})")
    return p, rep



# --- minutes-per-start channel (studies/minutes_per_start.py) -------------------
# The study fitted a weight PER CUTOFF, not a functional form:
#     k      1     2     3     5     8    10
#     w   0.18  0.24  0.32  0.40  0.54  0.63
# `w = k / (k + K)` with K = 6 reproduces those closely (0.14, 0.25, 0.33, 0.45, 0.57,
# 0.63) and extends sensibly past k = 10, which the study did not measure. The smooth
# form is an INTERPOLATION of fitted points, not itself fitted; K is exposed so it can be
# re-fit against the production prior (PROJECT_KNOWLEDGE §6.9) rather than pinned here.
EXP_MINUTES_K = 6.0
MPS_FLOOR, MPS_CEIL = 30.0, 90.0


def update_exp_minutes(players, apps, k_half=EXP_MINUTES_K, verbose=True):
    """Blend `exp_minutes` toward realised minutes-per-start. OFF unless called.

    `update_minutes` answers WHETHER a player starts. This answers HOW LONG he lasts once
    he does, which nothing in this module previously touched — a player withdrawn at half
    time every week was indistinguishable from one who played every minute. It matters
    twice over, because `bayes_model._minutes_if_start` feeds `m90`, which scales
    attacking involvement, penalty xG and DefCon as well as the appearance points.

    Denominator is `start_minutes`, NOT `minutes`: a player who also came off the bench
    would otherwise have substitute minutes in the numerator with no start beneath them,
    and the ratio can exceed 90.

    Players with no `exp_minutes` (cold start) blend against the POSITIONAL constant that
    `_minutes_if_start` would otherwise have used, so the channel refines the fallback
    rather than silently replacing it with two matches of evidence.
    """
    p = players.copy()
    if not len(apps) or "player_code" not in p.columns or "exp_minutes" not in p.columns:
        if verbose:
            print("[inseason] exp_minutes: nothing to apply")
        return p, pd.DataFrame()
    from bayes_model import MINUTES_IF_START
    a = apps.dropna(subset=["player_code"]).drop_duplicates("player_code")
    if "start_minutes" not in a.columns:
        if verbose:
            print("[inseason] exp_minutes: appearances carries no start_minutes column")
        return p, pd.DataFrame()
    m = {r.player_code: (float(r.starts or 0), float(r.start_minutes or 0))
         for r in a.itertuples()}
    rows = []
    for i, code in p["player_code"].items():
        if code not in m:
            continue
        st, sm = m[code]
        if st < 1:
            continue
        realised = min(MPS_CEIL, max(MPS_FLOOR, sm / st))
        prior = p.at[i, "exp_minutes"]
        if prior is None or not np.isfinite(prior) or prior <= 0:
            prior = MINUTES_IF_START.get(p.at[i, "pos"] if "pos" in p.columns else None, 85.3)
        w = st / (st + k_half)
        new = (1 - w) * float(prior) + w * realised
        p.at[i, "exp_minutes"] = new
        rows.append({"player_code": code, "web_name": p.at[i, "web_name"] if "web_name" in p.columns else None,
                     "starts": st, "realised": realised, "w": w,
                     "before": float(prior), "after": new, "delta": new - float(prior)})
    rep = pd.DataFrame(rows)
    if verbose and len(rep):
        big = rep.reindex(rep["delta"].abs().sort_values(ascending=False).index).head(6)
        print(f"[inseason] exp_minutes: updated {len(rep)} players (K={k_half:g}); "
              f"mean |move| {rep['delta'].abs().mean():.1f} min; largest:")
        for _, r in big.iterrows():
            print(f"    {str(r.get('web_name','?')):20s} {r['starts']:.0f} starts, "
                  f"{r['realised']:.0f} min/start   exp_minutes "
                  f"{r['before']:.1f} -> {r['after']:.1f} ({r['delta']:+.1f})")
    return p, rep


# --- start-prior strength (studies/start_prior_strength.py) ---------------------
# The study showed the current corner (w=1, kappa=inf) is dominated by ~11% on Brier, but
# that only the RATIO w/kappa is identified — the surface is a ridge. kappa=5 at w=1 is
# ONE endpoint of that ridge; w~8 uncapped is the other and scores the same. This exposes
# the cap because it is the more conservative reading: it bounds how much prior any one
# player may carry without changing how a realised match is counted.
# 4, the centre of the interior optimum once the grid was extended below 5. The first
# run of the study put it at 5 — the LOWEST value then tested — which was a boundary
# artefact, not a fit. At w=1 the single-lever optimum is kappa = 3/4/4/5 across k =
# 2/3/5/8. Still one endpoint of an unidentified ridge (only w/kappa is pinned), so this
# is a default for a flag that is OFF, not a settled constant.
START_KAPPA = 4.0


def cap_start_prior(players, kappa=START_KAPPA, verbose=True):
    """Cap the Beta start prior at `kappa` pseudo-matches, holding its mean. OFF unless
    called, and it MUST run before `update_minutes` so realised matches land against the
    capped prior rather than the raw one.

    Without it a player with two full seasons behind him arrives carrying ~34
    pseudo-matches and two weeks of not being picked move him almost nowhere: Dubravka
    after GW2 sat at 0.831 having started none of two, above Kinsky at 0.696 who started
    both.
    """
    p = players.copy()
    if not {"start_a", "start_b"} <= set(p.columns) or not np.isfinite(kappa):
        return p, pd.DataFrame()
    a = p["start_a"].astype(float).to_numpy()
    b = p["start_b"].astype(float).to_numpy()
    tot = a + b
    scale = np.where(tot > kappa, kappa / np.maximum(tot, 1e-9), 1.0)
    n_capped = int((scale < 1.0).sum())
    p["start_a"] = a * scale
    p["start_b"] = b * scale
    if verbose:
        print(f"[inseason] start prior capped at {kappa:g} pseudo-matches: "
              f"{n_capped} of {len(p)} players rescaled "
              f"(max prior strength {tot.max():.0f} -> {min(tot.max(), kappa):.0f})")
    return p, pd.DataFrame({"n_capped": [n_capped], "kappa": [kappa]})


def rates(season="2026-2027", upto_gw=None, base=None, verbose=True):
    """Current-season attacking evidence per player_code: exposure and events.

    Non-penalty xG is derived the way `build_pms` derives it — `xg` less 0.79 per penalty
    attempted — so the in-season numerator is on the same scale as the prior's. Using raw
    `xg` here would quietly credit penalty takers twice, once in the rate and again
    through `pen_xg90`.
    """
    import core_insights as ci
    base = base or config.repo(season)
    roster = pd.read_csv(os.path.join(base, "players.csv"))
    teams = pd.read_csv(os.path.join(base, "teams.csv"))
    roster = roster.merge(teams[["code", "name"]], left_on="team_code",
                          right_on="code", how="left")
    roster["team"] = roster["name"].map(ci.norm_team)
    look = roster[["player_id", "player_code", "team"]]

    frames = []
    for d in sorted(glob.glob(os.path.join(base, "By Gameweek", "GW*"))):
        gw = os.path.basename(d)[2:]
        if upto_gw is not None and gw.isdigit() and int(gw) > int(upto_gw):
            continue
        f = os.path.join(d, "playermatchstats.csv")
        if not os.path.exists(f):
            continue
        pm = pd.read_csv(f)
        if "match_id" in pm.columns:            # league only, as everywhere else
            pm = pm[pm["match_id"].astype(str).str.contains("-prem-", na=False)]
        if not len(pm):
            continue
        frames.append(pm)
    if not frames:
        if verbose:
            print("[inseason] rates: no league player-match rows yet")
        return pd.DataFrame(columns=["player_code", "n90", "npxg", "xa", "kp", "matches"])
    P = pd.concat(frames, ignore_index=True)
    for c in ("xg", "xa", "chances_created", "minutes_played",
              "penalties_scored", "penalties_missed"):
        if c not in P.columns:
            P[c] = 0.0
        P[c] = pd.to_numeric(P[c], errors="coerce").fillna(0.0)
    pen_att = P["penalties_scored"] + P["penalties_missed"]
    P["npxg"] = (P["xg"] - 0.79 * pen_att).clip(lower=0)
    P = P.merge(look, on="player_id", how="left").dropna(subset=["player_code"])
    A = (P.groupby("player_code", as_index=False)
           .agg(mins=("minutes_played", "sum"), npxg=("npxg", "sum"),
                xa=("xa", "sum"), kp=("chances_created", "sum"),
                matches=("minutes_played", "size")))
    A["n90"] = A["mins"] / 90.0
    if verbose:
        print(f"[inseason] rates: {len(A)} players, {A.n90.sum():.0f} 90s, "
              f"npxG {A.npxg.sum():.1f}, xA {A.xa.sum():.1f}")
    return A[["player_code", "n90", "npxg", "xa", "kp", "matches"]]


def update_rates(players, ev, weight=W_RATE, min_matches=None, verbose=True,
                 min_npxg=MIN_RATE_MATCHES_NPXG, min_xa=MIN_RATE_MATCHES_XA):
    """Conjugate Gamma update of the attacking priors from current-season events.

        npxgi_alpha += w * npxg_new      npxgi_beta += w * n90_new
        xa_alpha    += w * xa_new        xa_beta    += w * n90_new

    This is the SAME estimator the prior already is, shown more evidence. It is not a
    form term: no recency, no trend, no streak. `weight=0` is exactly the identity, which
    is how this channel shipped before it was calibrated.

    TWO GATES, NOT ONE. npxG and xA are separate quantities with separate calibrations
    and they do not open at the same k — npxG clears its decision rule at three matches,
    xA at five. They are applied per channel so a player with three matches has his npxG
    updated and his xA left alone, rather than neither. `min_matches`, if passed,
    overrides both and restores the old single-gate behaviour for callers that want it.
    """
    p = players.copy()
    if min_matches is not None:
        min_npxg = min_xa = int(min_matches)
    if weight <= 0 or not len(ev) or "player_code" not in p.columns:
        if verbose:
            print(f"[inseason] rates: off (weight={weight})")
        return p, pd.DataFrame()
    e = ev.dropna(subset=["player_code"]).drop_duplicates("player_code")
    lo = min(min_npxg, min_xa)
    e = e[e["matches"] >= lo]
    if not len(e):
        if verbose:
            print(f"[inseason] rates: no player has {lo}+ league matches yet "
                  f"— channel idle")
        return p, pd.DataFrame()
    m = e.set_index("player_code")
    before = p["npxgi_alpha"] / p["npxgi_beta"]
    before_xa = p["xa_alpha"] / p["xa_beta"]
    rows = []
    n_npxg = n_xa = 0
    for i, code in p["player_code"].items():
        if code not in m.index:
            continue
        r = m.loc[code]
        n90 = float(r["n90"])
        if n90 <= 0:
            continue
        k = float(r["matches"])
        did_npxg = k >= min_npxg
        did_xa = k >= min_xa
        if did_npxg:
            p.at[i, "npxgi_alpha"] = float(p.at[i, "npxgi_alpha"]) + weight * float(r["npxg"])
            p.at[i, "npxgi_beta"] = float(p.at[i, "npxgi_beta"]) + weight * n90
            n_npxg += 1
        if did_xa:
            p.at[i, "xa_alpha"] = float(p.at[i, "xa_alpha"]) + weight * float(r["xa"])
            p.at[i, "xa_beta"] = float(p.at[i, "xa_beta"]) + weight * n90
            n_xa += 1
        if did_npxg or did_xa:
            rows.append({"player_code": code, "n90": n90, "npxg": float(r["npxg"]),
                         "matches": k, "npxg_updated": did_npxg, "xa_updated": did_xa})
    after = p["npxgi_alpha"] / p["npxgi_beta"]
    after_xa = p["xa_alpha"] / p["xa_beta"]
    rep = pd.DataFrame(rows)
    if verbose:
        d = (after - before).abs()
        dx = (after_xa - before_xa).abs()
        print(f"[inseason] rates: w={weight}; npxG updated {n_npxg} players "
              f"(k>={min_npxg}), mean |change| {d[d > 0].mean() if (d > 0).any() else 0:.4f}"
              f"; xA updated {n_xa} (k>={min_xa}), mean |change| "
              f"{dx[dx > 0].mean() if (dx > 0).any() else 0:.4f}")
    return p, rep

# ---------------------------------------------------------------- report / CLI
def report(season="2026-2027", upto_gw=None):
    print("=== 26/27 evidence available to the model ===")
    m_all = played(season, upto_gw, require_xg=False, verbose=False)
    m_xg = played(season, upto_gw, require_xg=True, verbose=False)
    print(f"  finished matches            {len(m_all)}")
    print(f"  ... with xG (team channel)  {len(m_xg)}"
          + ("" if len(m_xg) >= MIN_XG_MATCHES
             else f"   BELOW MIN_XG_MATCHES={MIN_XG_MATCHES}, team channel stays off"))
    a = appearances(season, upto_gw, verbose=False)
    print(f"  players with an appearance  {int((a['apps'] > 0).sum()) if len(a) else 0}")
    print(f"  total starts recorded       {int(a['starts'].sum()) if len(a) else 0}")
    if len(m_xg):
        print("\n  matches carrying xG:")
        for _, r in m_xg.iterrows():
            print(f"    GW{int(r['gameweek'])}  {r['home']:14s} "
                  f"{r['home_score']:.0f}-{r['away_score']:.0f} {r['away']:14s}   "
                  f"xG {r['xg_home']:.2f}-{r['xg_away']:.2f}")


def selftest():
    # --- odds round-trip: an appended row must invert back to the xG it encodes ---
    import betting_features as bf
    import tempfile
    for lh, la in [(1.81, 1.26), (0.6, 2.4), (2.0, 2.0), (3.1, 0.4)]:
        oh, od, oa, ov, un = _fair_odds(lh, la)
        assert abs((1/oh + 1/od + 1/oa) - 1.0) < 1e-6, "1X2 must be vig-free"
        assert abs((1/ov + 1/un) - 1.0) < 1e-6, "over/under must be vig-free"
    M = pd.DataFrame({"gameweek": [1, 1], "date": pd.to_datetime(["2026-08-22"] * 2),
                      "home": ["Hull", "Arsenal"], "away": ["Man United", "Coventry"],
                      "home_score": [2, 3], "away_score": [0, 0],
                      "xg_home": [1.26, 2.4], "xg_away": [1.81, 0.6]})
    rows = e0_rows(M)
    tmp = os.path.join(tempfile.mkdtemp(), "rt.csv")
    rows.to_csv(tmp, index=False)
    got = bf.build(tmp)
    assert np.allclose(got["lambda_home"].values, M["xg_home"].values, atol=0.02), \
        f"lambda_home round trip failed: {got['lambda_home'].values} vs {M['xg_home'].values}"
    assert np.allclose(got["lambda_away"].values, M["xg_away"].values, atol=0.02), \
        f"lambda_away round trip failed: {got['lambda_away'].values}"
    # and the response column must be xG, not the scoreline
    assert abs(rows["FTHG"].iloc[0] - 1.26) < 1e-9, "FTHG must carry xG, not goals"
    assert rows["FTHG"].iloc[0] != M["home_score"].iloc[0]

    # --- stacking replicates by weight, and promoted clubs get their own ---
    base = pd.DataFrame({"Date": ["01/11/2025"], "HomeTeam": ["Brighton"],
                         "AwayTeam": ["Leeds"], "FTHG": [3], "FTAG": [0],
                         "AvgH": [1.5], "AvgD": [4.0], "AvgA": [6.0],
                         "Avg>2.5": [2.0], "Avg<2.5": [2.0]})
    bp = os.path.join(tempfile.mkdtemp(), "base.csv"); base.to_csv(bp, index=False)
    out, n = stack_e0(M, e0_path=bp, out=os.path.join(tempfile.mkdtemp(), "s.csv"),
                      weight=1, weight_promoted=3, promoted=("Hull", "Coventry"),
                      verbose=False)
    S = pd.read_csv(out)
    # both fixtures involve a promoted club here, so both replicate 3x
    assert len(S) == 1 + 6, f"expected 1 base + 6 stacked rows, got {len(S)}"
    out2, _ = stack_e0(M, e0_path=bp, out=os.path.join(tempfile.mkdtemp(), "s2.csv"),
                       weight=1, weight_promoted=1, promoted=(), verbose=False)
    assert len(pd.read_csv(out2)) == 1 + 2
    # weight 0 must drop the new evidence entirely, not error
    out3, n3 = stack_e0(M, e0_path=bp, out=os.path.join(tempfile.mkdtemp(), "s3.csv"),
                        weight=0, weight_promoted=0, promoted=(), verbose=False)
    assert n3 == 0 and len(pd.read_csv(out3)) == 1
    # a fractional weight must be refused, not silently rounded
    try:
        stack_e0(M, e0_path=bp, out=os.path.join(tempfile.mkdtemp(), "s4.csv"),
                 weight=0.5, verbose=False)
        raise AssertionError("fractional weight should be rejected")
    except ValueError:
        pass
    # cup ties must never enter: a non "-prem-" match_id is dropped before anything else
    import tempfile as _tf
    _b = _tf.mkdtemp()
    os.makedirs(os.path.join(_b, "By Gameweek", "GW2"))
    pd.DataFrame({"code": [3.0, 9.0], "name": ["Arsenal", "Coventry City"]}).to_csv(
        os.path.join(_b, "teams.csv"), index=False)
    pd.DataFrame({
        "gameweek": [2, 2],
        "match_id": ["26-27-prem-arsenal-vs-coventry-city",
                     "26-27-efl-cup-arsenal-vs-port-vale"],
        "kickoff_time": ["2026-08-29T14:00:00Z"] * 2,
        "home_team": [3.0, 3.0], "away_team": [9.0, 999.0],
        "home_score": [2, 4], "away_score": [0, 1], "finished": [True, True],
        "home_expected_goals_xg": [1.7, 3.1], "away_expected_goals_xg": [0.4, 0.3],
    }).to_csv(os.path.join(_b, "By Gameweek", "GW2", "matches.csv"), index=False)
    got = played(base=_b, require_xg=True, verbose=False)
    assert len(got) == 1, f"cup tie must be dropped, got {len(got)} matches"
    assert got.iloc[0]["away"] == "Coventry", got.iloc[0].to_dict()
    assert not got["home"].isna().any() and not got["away"].isna().any()

    # empty evidence leaves the prior path untouched
    p0, n0 = stack_e0(pd.DataFrame(), e0_path=bp, verbose=False)
    assert p0 == bp and n0 == 0

    # --- rates: conjugate Gamma update, gated on match count ---
    pr = pd.DataFrame({"player_code": [1, 2, 3], "web_name": ["A", "B", "C"],
                       "npxgi_alpha": [1.0, 1.0, 1.0], "npxgi_beta": [10.0, 10.0, 10.0],
                       "xa_alpha": [0.5, 0.5, 0.5], "xa_beta": [10.0, 10.0, 10.0]})
    evd = pd.DataFrame({"player_code": [1, 2, 3], "n90": [8.0, 8.0, 2.0],
                        "npxg": [4.0, 0.0, 4.0], "xa": [2.0, 0.0, 2.0],
                        "kp": [10, 0, 5], "matches": [8, 8, 2]})
    up, rep = update_rates(pr, evd, weight=1.0, verbose=False)
    rate = lambda d, i: d.npxgi_alpha.iloc[i] / d.npxgi_beta.iloc[i]
    assert rate(up, 0) > rate(pr, 0), "a hot start must raise the rate"
    assert rate(up, 1) < rate(pr, 1), "a cold start must lower it"
    # exact conjugate arithmetic: (1 + 1*4) / (10 + 1*8)
    assert abs(rate(up, 0) - 5.0 / 18.0) < 1e-12, rate(up, 0)
    # player 3 has only 2 matches -> below MIN_RATE_MATCHES, untouched
    assert rate(up, 2) == rate(pr, 2), "a player under the match gate must not move"
    # weight 0 is exactly the identity, which is how this shipped before calibration
    up0, _ = update_rates(pr, evd, weight=0.0, verbose=False)
    assert np.allclose(up0[["npxgi_alpha", "npxgi_beta", "xa_alpha", "xa_beta"]].values,
                       pr[["npxgi_alpha", "npxgi_beta", "xa_alpha", "xa_beta"]].values),         "weight 0 must be the identity"
    # the assist channel moves with its own evidence, not the shooting channel's
    xr = lambda d, i: d.xa_alpha.iloc[i] / d.xa_beta.iloc[i]
    assert xr(up, 0) > xr(pr, 0) and xr(up, 1) < xr(pr, 1)

    # --- the two gates are independent, and open at different k ---
    # A player on FOUR matches sits above the npxG gate (3) and below the xA gate (5):
    # his shooting prior must update and his assist prior must not. A single gate would
    # have moved both or neither, which is what this split exists to stop.
    pr4 = pd.DataFrame({"player_code": [7], "web_name": ["D"],
                        "npxgi_alpha": [1.0], "npxgi_beta": [10.0],
                        "xa_alpha": [0.5], "xa_beta": [10.0]})
    ev4 = pd.DataFrame({"player_code": [7], "n90": [4.0], "npxg": [3.0], "xa": [1.5],
                        "kp": [6], "matches": [4]})
    up4, rep4 = update_rates(pr4, ev4, weight=1.0, verbose=False)
    assert rate(up4, 0) != rate(pr4, 0), "npxG must update at k=4 (gate is 3)"
    assert abs(rate(up4, 0) - 4.0 / 14.0) < 1e-12, rate(up4, 0)
    assert xr(up4, 0) == xr(pr4, 0), "xA must NOT update at k=4 (gate is 5)"
    assert bool(rep4["npxg_updated"].iloc[0]) and not bool(rep4["xa_updated"].iloc[0])
    # at k=5 both open
    ev5 = ev4.assign(matches=[5], n90=[5.0])
    up5, _ = update_rates(pr4, ev5, weight=1.0, verbose=False)
    assert rate(up5, 0) != rate(pr4, 0) and xr(up5, 0) != xr(pr4, 0), "both open at k=5"
    # an explicit min_matches still forces the old single-gate behaviour
    upS, _ = update_rates(pr4, ev4, weight=1.0, min_matches=5, verbose=False)
    assert rate(upS, 0) == rate(pr4, 0) and xr(upS, 0) == xr(pr4, 0), (
        "min_matches= must override both gates")
    # non-penalty derivation: a penalty taker must not be credited twice
    import tempfile as _tf2
    _b2 = _tf2.mkdtemp()
    os.makedirs(os.path.join(_b2, "By Gameweek", "GW1"))
    pd.DataFrame({"code": [3.0], "name": ["Arsenal"]}).to_csv(
        os.path.join(_b2, "teams.csv"), index=False)
    pd.DataFrame({"player_id": [1], "player_code": [101], "team_code": [3.0]}).to_csv(
        os.path.join(_b2, "players.csv"), index=False)
    pd.DataFrame({"player_id": [1], "match_id": ["26-27-prem-a-vs-b"],
                  "minutes_played": [90], "xg": [1.0], "xa": [0.2],
                  "chances_created": [2], "penalties_scored": [1],
                  "penalties_missed": [0]}).to_csv(
        os.path.join(_b2, "By Gameweek", "GW1", "playermatchstats.csv"), index=False)
    rr = rates(base=_b2, verbose=False)
    assert abs(float(rr.npxg.iloc[0]) - 0.21) < 1e-9,         f"npxg must subtract 0.79 per penalty attempt, got {rr.npxg.iloc[0]}"

    # --- minutes: conjugate Beta update ---
    pl = pd.DataFrame({"player_code": [1, 2, 3], "web_name": ["A", "B", "C"],
                       "start_a": [2.0, 2.0, 20.0], "start_b": [2.0, 2.0, 2.0]})
    ap = pd.DataFrame({"player_code": [1, 2, 3], "team": ["T"] * 3,
                       "starts": [3, 0, 0], "apps": [3, 0, 0],
                       "club_matches": [3, 3, 3]})
    up, rep = update_minutes(pl, ap, weight=1.0, verbose=False)
    p_of = lambda d, i: d.start_a.iloc[i] / (d.start_a.iloc[i] + d.start_b.iloc[i])
    assert p_of(up, 0) > p_of(pl, 0), "3 starts from 3 must raise the prior"
    assert p_of(up, 1) < p_of(pl, 1), "0 starts from 3 must lower it"
    assert abs(p_of(up, 0) - 5.0 / 7.0) < 1e-9, "conjugate arithmetic"
    # a strong prior must move LESS than a weak one on identical evidence
    assert (p_of(pl, 1) - p_of(up, 1)) > (p_of(pl, 2) - p_of(up, 2)), \
        "a 22-observation prior must be harder to move than a 4-observation one"
    # weight 0 is exactly the identity
    up0, _ = update_minutes(pl, ap, weight=0.0, verbose=False)
    assert np.allclose(up0[["start_a", "start_b"]].values,
                       pl[["start_a", "start_b"]].values), "weight 0 must be identity"
    # starts can never exceed the club's completed matches, and the clamp must SAY so:
    # `gw_panel` guarantees the invariant at source, so a breach here means the panel is
    # wrong, and swallowing it silently is how 22 half-time starts once went unnoticed.
    # Captured rather than allowed to print, so a passing selftest stays quiet.
    ap_bad = ap.copy(); ap_bad.loc[0, "starts"] = 99
    _stdout, sys.stdout = sys.stdout, io.StringIO()
    try:
        upb, _ = update_minutes(pl, ap_bad, weight=1.0, verbose=False)
        _warned = sys.stdout.getvalue()
    finally:
        sys.stdout = _stdout
    assert upb.start_b.iloc[0] >= pl.start_b.iloc[0], "start_b must not go backwards"
    assert "starts > club_matches" in _warned,         "a clamped breach must warn even when verbose=False"
    # and a frame that respects the invariant must stay silent
    _stdout, sys.stdout = sys.stdout, io.StringIO()
    try:
        update_minutes(pl, ap, weight=1.0, verbose=False)
        _quiet = sys.stdout.getvalue()
    finally:
        sys.stdout = _stdout
    assert _quiet == "", f"clean frame must not warn, got: {_quiet!r}"
    # a player absent from the appearance frame is untouched
    up2, _ = update_minutes(pl, ap[ap.player_code != 3], weight=1.0, verbose=False)
    assert up2.start_a.iloc[2] == 20.0 and up2.start_b.iloc[2] == 2.0

    # --- minutes: recency weighting (studies/start_recency.py) ---
    # Two players, four single-match gameweeks each. Same COUNT, opposite ORDER: the
    # exchangeable update cannot tell them apart and the recency-weighted one must.
    Gr = pd.DataFrame({"player_code": [1, 1, 1, 1, 2, 2, 2, 2],
                       "team": ["T"] * 8, "gw": [1, 2, 3, 4] * 2,
                       "starts": [1, 1, 0, 0, 0, 0, 1, 1],
                       "minutes": [90, 90, 0, 0, 0, 0, 90, 90]})
    Mr = pd.DataFrame({"gameweek": [1, 2, 3, 4], "home": ["T"] * 4,
                       "away": ["U"] * 4})
    flat = recency_starts(Gr, Mr, lam=1.0)
    assert np.allclose(sorted(flat["w_starts"]), [2.0, 2.0]), \
        "lam=1 must return the raw counts exactly"
    rec = recency_starts(Gr, Mr, lam=0.5).set_index("player_code")["w_starts"]
    assert rec[2] > rec[1], "the player who started RECENTLY must weigh more"
    assert abs((rec[1] + rec[2]) - 4.0) < 1e-9, \
        "renormalisation must conserve total evidence mass across the pair"
    for lam in (0.25, 0.5, 0.75, 1.0):
        w = recency_starts(Gr, Mr, lam=lam)["w_starts"].to_numpy()
        assert (w >= -1e-12).all() and (w <= 4.0 + 1e-9).all(), \
            f"weighted starts must stay inside [0, club matches] at lam={lam}"
    # a double gameweek shares one recency position and still conserves mass
    Gd = pd.DataFrame({"player_code": [1, 1], "team": ["T", "T"], "gw": [1, 2],
                       "starts": [2, 1], "minutes": [180, 90]})
    Md = pd.DataFrame({"gameweek": [1, 1, 2], "home": ["T"] * 3, "away": ["U"] * 3})
    wd = float(recency_starts(Gd, Md, lam=0.5)["w_starts"].iloc[0])
    assert 0.0 <= wd <= 3.0 + 1e-9, f"DGW mass must respect the 3-match bound, got {wd}"
    # and the whole path through update_minutes stays conjugate
    apr = pd.DataFrame({"player_code": [1, 2], "team": ["T", "T"], "starts": [2, 2],
                        "apps": [2, 2], "club_matches": [4, 4],
                        "w_starts": [rec[1], rec[2]]})
    plr = pd.DataFrame({"player_code": [1, 2], "web_name": ["A", "B"],
                        "start_a": [2.0, 2.0], "start_b": [2.0, 2.0]})
    upr, _ = update_minutes(plr, apr, weight=1.0, verbose=False)
    assert p_of(upr, 1) > p_of(upr, 0), \
        "update_minutes must consume w_starts when it is present"

    # --- minutes: a transfer is placed by the club he played FOR, not the club he is at
    # A blanks in gw2, B blanks in gw1, and the player started once for each. Attributing
    # both rows to his current club B loses the gw1 start outright, because B has no gw1
    # fixture for it to sit on; that is the shape of the 26/27 GW1/GW2 breach (Man City
    # 12, Everton 10) with a blank added so the error shows up in the WEIGHT and not only
    # in the per-club total. `team` is the per-gameweek club, `team_now` the snapshot one.
    Gt = pd.DataFrame({"player_code": [7, 7], "team": ["A", "B"], "team_now": ["B", "B"],
                       "gw": [1, 2], "starts": [1, 1], "minutes": [90, 90]})
    Mt = pd.DataFrame({"gameweek": [1, 2], "home": ["A", "B"], "away": ["X", "X"]})
    wt = recency_starts(Gt, Mt, lam=1.0)
    assert len(wt) == 1 and wt["team"].iloc[0] == "B", \
        "a player who moved must stay ONE row, labelled with his current club"
    assert abs(float(wt["w_starts"].iloc[0]) - 2.0) < 1e-9, \
        f"both starts must survive the move, got {float(wt['w_starts'].iloc[0])}"
    wrong = recency_starts(Gt.assign(team=Gt["team_now"]), Mt, lam=1.0)
    assert abs(float(wrong["w_starts"].iloc[0]) - 1.0) < 1e-9, \
        "the current-roster attribution should lose the old club's start (guard is live)"

    # --- minutes: the finished guard is per MATCH, not per gameweek
    # One finished match must not carry an unfinished one into the panel with it. This is
    # the 26/27 GW3 shape: nine matches done, Arsenal-Chelsea live at half time, and
    # `player_gameweek_stats` already holding all 22 of its starters at 45 minutes.
    nm_ = {1.0: "A", 2.0: "B", 3.0: "C", 4.0: "D"}
    mx = pd.DataFrame({"match_id": ["26-27-prem-a-vs-b", "26-27-prem-c-vs-d",
                                    "26-27-efl-a-vs-x"],
                       "home_team": [1.0, 3.0, 1.0], "away_team": [2.0, 4.0, 9.0],
                       "finished": [True, False, True]})
    assert set(_finished_clubs(mx, nm_)) == {"A", "B"}, \
        "an unfinished league match must hold its own clubs out, and a cup tie admit none"
    # a club with one match played and one still live in the same gameweek is held out,
    # because `player_gameweek_stats` sums a double gameweek into a single row
    my = pd.DataFrame({"match_id": ["26-27-prem-a-vs-b", "26-27-prem-a-vs-c"],
                       "home_team": [1.0, 1.0], "away_team": [2.0, 3.0],
                       "finished": [True, False]})
    assert set(_finished_clubs(my, nm_)) == {"B"}, \
        "a DGW club with a match still in progress must not enter"

    print("SELFTEST OK: xG round-trips through the odds columns into lambda, FTHG "
          "carries xG not goals, stacking replicates by whole weights and refuses "
          "fractions, promoted clubs weighted separately, Beta update is conjugate and "
          "respects prior strength, weight 0 is the identity on both channels, recency "
          "weighting conserves evidence mass and is the identity at lam=1, a "
          "transferred player's starts are placed by the club he played for, and an "
          "unfinished match cannot ride into the panel on a finished one.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    if "--report" in sys.argv:
        report(); sys.exit(0)
    print(__doc__)
