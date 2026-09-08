from __future__ import annotations
import config
"""
gw_explorer.py — turn the board into an interactive, filterable, re-sortable view
=================================================================================
`gw_board_long.csv` is the canonical per-player per-gameweek projection. It is a flat
file: to ask "who is best over the NEXT THREE" and then "over the next ten" you have to
re-pivot it, and the `total` column in `gw_board_wide.csv` answers exactly one horizon —
whichever GW_HI the board happened to run at. This module leaves the projections
untouched and builds a view over them in which the HORIZON IS A CONTROL, not a baked-in
column.

IT DOES NOT PROJECT ANYTHING. It reads the board and annotates it. Every number in the
output is the number the simulation produced, joined to the fixture it was produced for.
Re-running `project()` here would hand each player a different draw from the seeded RNG
stream and diverge from the board by Monte Carlo noise — the trap documented at length in
scripts/export_projection_detail.py.

WHAT IT ADDS TO THE BOARD
-------------------------
  * THE FIXTURE, per cell — opponent and venue from `schedule_2627`, and, where the team
    export exists at a matching horizon, the posterior lambdas the projection was built
    from (`lam_for`, `lam_against`, `p_clean_sheet`, `p_win`) plus the opponent's fitted
    attack and defence on the model's own log scale. Fixture strength is not recomputed
    as a difficulty rating; it IS the projection, so the underlying quantities are
    carried across rather than a proxy for them.
  * WHICH FIXTURES HAVE ALREADY BEEN PLAYED — per CLUB, not per gameweek. The board
    projects GW1..GW_HI unconditionally, including weeks gone by, and a "next N" window
    that silently starts at GW1 is summing history. But a gameweek is not a unit that is
    played or unplayed: GW2 of 26/27 is settled for eighteen clubs and still live for
    Arsenal and Villa. So the flag sits on the CELL, from `inseason.played()`, and two
    horizon markers come out of it — `first_live` (anything still to come) and
    `first_clean` (nothing played at all). The window opens at `first_clean` and "next N"
    counts from there. Played weeks are still carried, dimmed rather than dropped, so a
    projection can be read back against what actually happened.
  * BLANKS ARE BLANK, NOT ZERO. A (player, gameweek) with no fixture is null. It
    contributes nothing to a window sum and is excluded from the per-fixture mean, so a
    blank does not read as a bad week — and the fixture count is shown beside every
    total so an eight-fixture window and a six-fixture window are never compared as if
    they were the same bet.

WHAT IT REFUSES TO ADD
----------------------
A window interval. `p5`/`p95` are per-gameweek posterior percentiles and PERCENTILES DO
NOT ADD: their sum across a window is not the interval of the sum. Quadrature on `sd`
would not rescue it either — every gameweek in the window is drawn against one shared
sample of team strength, so the weeks are positively correlated and an independence
assumption understates the spread. The window total is therefore a POINT ESTIMATE,
deliberately without a band. The object that answers "how likely is this player to
out-score that one over six weeks" is the joint draw matrix — `DUMP_DRAWS=all python
scripts/gw_board.py` writes it — and that is a different tool.

Run:  python scripts/export_gw_explorer.py
Out:  outputs/gw_explorer.html   self-contained; no network, no CDN, opens from disk
      outputs/gw_explorer.csv    the same table, tidy long, for your own pivoting
"""
import json
import os
import numpy as np
import pandas as pd

# Per-gameweek quantities offered to the grid, in the order they appear in the picker.
# `blended` leads because it is what the board's wide view totals.
METRICS = [
    ("blended",       "Projected points",       "The board's headline number: the model, or the Solio blend where one was applied."),
    ("mean",          "Projected (model only)", "Posterior mean before any external blend."),
    ("sd",            "SD",                     "Posterior SD of the single-gameweek score. Does not add across weeks."),
    ("p5",            "5th percentile",         "Per-gameweek posterior percentile. Does not add across weeks."),
    ("median",        "Median",                 "Per-gameweek posterior percentile. Does not add across weeks."),
    ("p95",           "95th percentile",        "Per-gameweek posterior percentile. Does not add across weeks."),
    ("app_ev",        "Appearance pts",         "EV of the appearance points alone; near 2 means a near-certain 60+ minutes, near 0 means the model does not expect him on the pitch."),
    ("att_ev",        "Attacking pts",          "EV from goals and assists."),
    ("cs_ev",         "Clean sheet pts",        "EV from the clean-sheet channel."),
    ("defcon_ev",     "DefCon pts",             "EV from defensive contributions."),
    ("conc_ev",       "Goals-conceded pts",     "EV of the goals-conceded deduction; negative."),
    ("lam_for",       "Team xG for",            "Posterior lambda for the player's club in this fixture — the fixture strength, on the model's own scale."),
    ("lam_against",   "Team xG against",        "Posterior lambda against the player's club in this fixture."),
    ("p_clean_sheet", "P(clean sheet)",         "Posterior-predictive: the mean over draws of exp(-lambda_against), not exp(-mean lambda)."),
    ("p_win",         "P(win)",                 "From independent Poisson draws per posterior sample."),
    ("opp_att",       "Opponent attack",        "The opponent's fitted attack strength, model log scale. Higher is a harder fixture for a defender."),
    ("opp_def",       "Opponent defence",       "The opponent's fitted defence strength, model log scale. Higher is a harder fixture for an attacker."),
    ("d_blended",     "Change since last board", "How much this gameweek's projection has MOVED since the last dated board snapshot — positive means the model likes him more than it did. The weekly decision is marginal (who to transfer), and a table of levels makes you hold last week's numbers in your head to see it; this is the difference, computed for you. Baseline is the most recent snapshot from a PREVIOUS day, so rebuilding the board three times this afternoon does not reset it to zero. Blank when there is no earlier snapshot — a first run, or a cleared cache. It ADDS across the window: the total change to a player's projected run. THERE IS NO NOISE FLOOR: the board is bit-identical run to run on identical inputs (measured 2026-09-07, 24,814 of 24,814 player-gameweeks exactly equal, max delta 0.000000), so any non-zero value here is a REAL change and not Monte Carlo jitter. The first live example was six players moving -0.01 to -0.02 on GW4 overnight, from a chance-of-playing revision on the FPL feed — the only non-deterministic input the board has."),
    ("par",           "Pts above replacement",  "Points above the REPLACEMENT player at this position — what the money above the positional floor actually bought. `player_value` computes it: floor_price is the cheapest price at which the position can be filled by someone who might play, replacement_ep is the BEST expected points obtainable at that floor, and par = ep - replacement_ep. Negative means a free slot would have done as well. Floors are recomputed EVERY GAMEWEEK, because a cheap defender who is flagged out or has a blank stops being a viable replacement that week and lifts the whole position's floor. It ADDS across the window: total points above replacement over the run. It REPLACED the raw points-per-million column, retired 2026-09-07: that column charged a premium for the first ~4.0m of every price that buys nothing a free slot would not have given you anyway, and two competing value columns side by side invite reading the wrong one. Over GW3-8 the two disagreed completely — par's top six were Haaland, B.Fernandes, Thiago, Mbeumo, Semenyo, Raya; points-per-million's were van Ewijk, Guehi, Thiaw, Thomas, Bassey, Mitchell, i.e. entirely the near-floor artefact."),
    ("opp_defcon",    "Opponent DefCon (DEF)",  "How much this opponent lifts a DEFENDER's chance of hitting the 10-action DefCon threshold, in probability points, against an average opponent. Measured within player on 25/26 and empirical-Bayes shrunk (`defcon_team`). DEFENDER ROWS ONLY: the same rating for midfielders failed its reliability gate (split-half r=+0.24) and is not shipped, so showing it on a MID row would assert something the study declined to. Null for GK, who cannot score DefCon, and for the promoted clubs, who have no 25/26 rating. A TIEBREAKER, not a driver — the full spread is 0.129 in probability, about 0.26 points a match, against roughly 0.87 for the clean-sheet swing across fixture difficulty. It also runs the SAME way as clean sheets (permissive opponents correlate +0.26 with attacking output), so it does not buy back a hard fixture."),
]
# Metrics whose SUM over a window is a quantity that exists. Points and their component
# EVs add, and so do the fixture lambdas (expected goals over the window) and the
# per-fixture probabilities (the expected COUNT of clean sheets, of wins). Nothing else
# here does: `sd` and the percentiles are per-gameweek dispersion, and opponent attack
# and defence are fitted levels on a log scale, so their sum over six weeks is a number
# with no referent. The view refuses to compute it and silently uses the mean instead
# rather than offering a control that produces nonsense.
ADDITIVE = {"blended", "mean", "app_ev", "att_ev", "cs_ev", "defcon_ev", "conc_ev",
            "lam_for", "lam_against", "p_clean_sheet", "p_win", "par", "d_blended"}
# Metrics where a LOWER number is the better outcome for the player holding the row, so
# the heat scale has to run the other way. Only these three: `conc_ev` is already signed
# (less negative is better, which the default scale gets right), and `sd` is neither —
# wide is bad for a captain and good for a differential, so it is left unshaded either way.
INVERT = {"lam_against", "opp_att", "opp_def"}
# ---------------------------------------------------------------------------
# PERCENTILE PROFILES
# ---------------------------------------------------------------------------
# The axes of the player and team profiles. Both are rendered as percentile bars and,
# optionally, as a radar — from ONE definition, so the two views can never disagree
# about what an axis means or what order the axes come in.
#
# A percentile needs a REFERENCE POPULATION, and the choice changes every number on the
# page. Ranking a player against all 626 puts most of the squad-fillers at zero and
# squashes everyone who actually plays into the top decile, which flatters the profile
# and says nothing.
#
# The obvious fix — a minimum expected-minutes threshold — is a WORSE estimator than it
# looks, because appearance EV is not on the same scale across positions. At a cutoff of
# 1.0 it admits 22 goalkeepers, 82 defenders, 67 midfielders and 13 forwards: one fixed
# number, four different selectivities, and a forward ranked against 13 peers has a
# percentile that can only move in steps of 7.7. That is not a percentile.
#
# So the pool is a fixed SIZE per position — the top N by expected minutes over the
# window, with N set by how many of that position are on the pitch in a normal week
# (one keeper, four defenders, four midfielders, two forwards per club across twenty
# clubs). Granularity is then comparable across positions and the definition is one a
# reader can check. The page exposes the switch and always names the population and its
# size, because a percentile quoted without one is not a number.
POOL_N = {"GK": 20, "DEF": 80, "MID": 80, "FWD": 40}
#
# Percentiles are computed IN THE PAGE, over the selected window, not baked in here: the
# whole point of the tool is that the horizon moves, and a profile fixed to GW1-10 would
# silently answer a different question from the table beside it.
PLAYER_COMPONENTS = [
    ("blended",   "Total points",     "sum",  False, "Projected points over the window."),
    ("att_ev",    "Attack",           "sum",  False, "Goals and assists."),
    ("cs_ev",     "Clean sheets",     "sum",  False, "The clean-sheet channel."),
    ("defcon_ev", "DefCon",           "sum",  False, "Defensive contributions."),
    ("app_ev",    "Minutes security", "mean", False, "Appearance points per fixture; the model's rotation read."),
    ("p95",       "Ceiling",          "mean", False, "Typical 95th-percentile week — the haul potential."),
    ("par",       "Value",            "sum",  False, "Points above the REPLACEMENT player at this position, summed over the window. This axis used to be points-per-£m; that construct divides by the WHOLE price, so it charges every player for the first ~4.0m that buys nothing a free slot would not have given you anyway, and it ranks near-floor squad-fillers above every premium. `par` divides nothing — it subtracts what the floor would have returned, gameweek by gameweek, so the window sum is the points the money above the floor actually bought."),
    ("defcon_stab", "DefCon stability", "calc", False,
     "His WORST DefCon week in the window as a share of his own average. 100 means he "
     "returns the same every week; a low score means the DefCon points are lumpy and "
     "arrive only in particular fixtures. This measures DELIVERY across the run, not the "
     "evidence behind his underlying rate — defensive_contributions is 100% null in 24/25, "
     "so every DefCon prior rests on a single season of exposure and the rate itself is "
     "held fixed across the window. Read it beside the DefCon level, never instead of it: "
     "a player who returns nothing every week is perfectly stable. THE SPREAD IS NARROW - "
     "over six gameweeks the 5th to 95th percentile of defenders runs about 0.92 to 0.99, "
     "so the percentile magnifies differences of around one per cent. The raw value sits "
     "beside the percentile for exactly that reason; a low rank here is a mild signal, not "
     "a damning one."),
]
TEAM_COMPONENTS = [
    ("att_strength", "Attack",          "team", False, "Fitted attack strength, model log scale."),
    ("def_strength", "Defence",         "team", False, "Fitted defence strength, model log scale."),
    ("exp_pts",      "Expected points", "team", False, "Projected league points across the season."),
    ("exp_cs",       "Clean sheets",    "team", False, "Projected clean sheets across the season."),
    ("press_factor", "Press",           "team", False, "Pressing intensity from the PPDA channel."),
    ("lam_for",      "Fixture attack",  "mean", False, "Mean expected goals FOR across the window — how kind the run is going forward."),
    ("lam_against",  "Fixture safety",  "mean", True,  "Mean expected goals AGAINST across the window, inverted — how kind the run is at the back."),
    ("sos_att",      "Schedule (att)",  "sos",  False, "Strength of schedule FOR THE ATTACKERS: the mean fitted DEFENCE strength of the opponents faced across the window, inverted so higher is an easier run. Unlike the two fixture rows above it, this holds the club itself out — it is a statement about who they play, not about how good they are. Split from the defenders' version because the two are different opponents' qualities and a club can have a kind run for one and a hard run for the other: you face a side's defence when you attack it and its attack when you defend."),
    ("sos_def",      "Schedule (def)",  "sos",  False, "Strength of schedule FOR THE DEFENCE AND GOALKEEPER: the mean fitted ATTACK strength of the opponents faced across the window, inverted so higher is an easier run. The mirror of the attackers' row above — read them apart, never as one number."),
    ("opp_defcon",   "DefCon schedule", "mean", False, "Mean DefCon permissiveness of the OPPONENTS faced across the window, in probability points against an average opponent — how easily THIS club's defenders should reach the 10-action threshold. Do not confuse it with the `DefCon conceded` KPI above, which is the mirror image: how much this club hands to the OPPOSITION's defenders. High here is good for owning their defenders; high there is good for owning the other side's. Defender-derived (the midfielder rating failed its reliability gate) and blank against the three promoted clubs, which have no 25/26 rating. A tiebreaker: the league spans 0.129, about 0.26 points a match, against roughly 0.87 for the clean-sheet swing."),
]

# STRENGTH OF SCHEDULE, and why it is a separate row from "Fixture attack".
# `lam_for` is the club's own expected goals in a fixture, so it moves with the club's
# attack as much as with the opponent's defence — Manchester City lead a lambda-based
# fixture ranking in every window, which is a fact about City. Strength of schedule is
# the same question with the club divided out: the mean fitted strength of the opponents
# actually faced. Both belong on the profile because they answer different questions —
# "how much will this side produce over the run" and "how kind is the run" — and only
# the second is comparable across clubs.

# Per-player quantities that do not vary by gameweek.
STATIC = [
    ("player", "Player"), ("pos", "Pos"), ("team", "Team"),
    ("cost", "Cost"), ("own", "Own %"), ("player_code", "Code"),
]
# Three letters per club. Derived rather than sliced from the name because "Man City"
# and "Man United" share their first three characters and a fixture column that reads
# MAN for both is worse than no fixture column.
ABBR = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Chelsea": "CHE", "Coventry": "COV", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Hull": "HUL", "Ipswich": "IPS", "Leeds": "LEE",
    "Liverpool": "LIV", "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Sunderland": "SUN", "Tottenham": "TOT",
}


# ---------------------------------------------------------------- inputs

def load_board(path=None):
    """The canonical board, long form. Asserts one row per (player_code, gameweek): the
    grid indexes on that pair, and a duplicate would silently keep whichever row was
    pivoted last rather than failing."""
    path = path or os.path.join(config.OUTPUTS, "gw_board_long.csv")
    df = pd.read_csv(path)
    missing = [c for c in ("player", "team", "pos", "gw", "blended") if c not in df.columns]
    assert not missing, f"{path} is not a gw_board_long.csv — missing {missing}"
    key = ["player_code", "gw"] if "player_code" in df.columns else ["player", "team", "gw"]
    dup = int(df.duplicated(key).sum())
    assert not dup, f"{dup} duplicate {tuple(key)} rows in {path}"
    return df


def fixtures(gw_hi):
    """One row per (team, gameweek) with opponent and venue, GW1..gw_hi."""
    from schedule_2627 import schedule
    _, long = schedule()
    f = long[long["gameweek"] <= gw_hi].rename(columns={"gameweek": "gw", "opp": "opponent"})
    return f[["team", "gw", "opponent", "is_home"]].copy()


def team_layer(gw_hi, outputs=None):
    """Per-fixture posterior lambdas from `export_team_projections`, and per-club fitted
    strength.

    Both are OPTIONAL. They are separate exports carrying their own GW_HI and may lag
    the board, so the explorer degrades to opponent-and-venue rather than refusing to
    build — an annotation that is merely absent is recoverable, one that silently
    describes a different run is not. Only an export covering the board's full horizon
    is accepted, and the narrowest such export wins.

    Returns (per_gw, per_club); either may be None."""
    outputs = outputs or config.OUTPUTS
    per_gw = None
    best = None
    for fn in (sorted(os.listdir(outputs)) if os.path.isdir(outputs) else []):
        if fn.startswith("team_projections_gw1_") and fn.endswith(".csv"):
            try:
                n = int(fn[len("team_projections_gw1_"):-len(".csv")])
            except ValueError:
                continue
            if n >= gw_hi and (best is None or n < best[0]):
                best = (n, fn)
    if best:
        t = pd.read_csv(os.path.join(outputs, best[1]))
        keep = [c for c in ("team", "gw", "lam_for", "lam_against", "p_clean_sheet",
                            "p_win") if c in t.columns]
        per_gw = t[keep].copy()
        per_gw.attrs["source"] = best[1]

    per_club = None
    p = os.path.join(outputs, "team_projections_season.csv")
    dc = None
    if os.path.exists(p):
        t = pd.read_csv(p)
        for a, b in (("own_att_strength", "own_def_strength"), ("attack", "defence")):
            if {"team", a, b} <= set(t.columns):
                per_club = t[["team", a, b]].rename(columns={a: "att", b: "def"})
                break
        # DefCon permissiveness, carried on the same per-club frame. Optional: the
        # export omits the columns entirely when the rating fails its reliability
        # gate, and an absent annotation must degrade rather than raise.
        # Only the number is carried. The tercile category in the export is a
        # coarsening of the same quantity, and the Fixtures tab's basis control is
        # hardcoded rather than metric-driven, so a label would be dead weight in the
        # frame with nothing to render it.
        if {"team", "defcon_conceded_hit"} <= set(t.columns):
            dc = t[["team", "defcon_conceded_hit"]].drop_duplicates("team")
    # There is deliberately NO fallback source here. Until 2026-09-08 this read
    # `team_strength_2627.csv` when the export above was absent — a file whose only
    # writer, `src/run_2627.py`, is not in the live tree, so it froze at 2026-08-06
    # while the export it stood in for was regenerated daily. The failure was silent
    # and one-directional: club strength a month stale, rendered identically to
    # current, with nothing on the surface saying which one you were reading. An
    # absent per-club frame is reported in `notes` and drops the columns instead,
    # which is the honest degradation. Restore a fallback only with a live producer.
    if per_club is not None:
        per_club = per_club.drop_duplicates("team")
        if dc is not None:
            per_club = per_club.merge(dc, on="team", how="left")
    elif dc is not None:
        per_club = dc
    return per_gw, per_club


def evidence():
    """How much of the 26/27 season the model has actually absorbed, and which update
    channels that switches on.

    A board rebuilt after a gameweek looks identical to one that was not, and the
    difference is invisible in the numbers themselves. This reports it: matches played,
    matches carrying xG, and the state of each in-season channel against its own
    documented gate. The gates are real and they are not all open at once — the team xG
    channel opens at `inseason.MIN_XG_MATCHES` matches, and the player attacking-rate
    channel opens in TWO STAGES because its two quantities were calibrated separately —
    npxG at `MIN_RATE_MATCHES_NPXG` (3) league matches, xA at `MIN_RATE_MATCHES_XA` (5).
    So after three gameweeks the minutes, team and npxG channels are live and xA is not.
    `rate_channel` is reported as open only when BOTH are, so a reader is never told the
    channel is on while half of it is still gated. Saying
    "the model is updated" without saying which channels moved is the kind of claim this
    project exists to avoid."""
    out = {"matches": 0, "matches_xg": 0, "max_per_club": 0,
           "team_channel": False, "rate_channel": False,
           "min_xg": None, "min_rate": None, "gw_complete": []}
    try:
        import inseason as ins
        out["min_xg"] = int(ins.MIN_XG_MATCHES)
        # Two rate gates, not one: npxG opens at 3 matches, xA at 5, each where its own
        # calibration cleared. `min_rate` stays the stricter for back-compat.
        out["min_rate_npxg"] = int(getattr(ins, "MIN_RATE_MATCHES_NPXG",
                                           ins.MIN_RATE_MATCHES))
        out["min_rate_xa"] = int(getattr(ins, "MIN_RATE_MATCHES_XA",
                                         ins.MIN_RATE_MATCHES))
        out["min_rate"] = out["min_rate_xa"]
        m = ins.played(upto_gw=38, require_xg=False, verbose=False)
        mx = ins.played(upto_gw=38, require_xg=True, verbose=False)
        if m is not None and len(m):
            out["matches"] = int(len(m))
            per = {}
            for r in m.itertuples():
                for t in (str(r.home), str(r.away)):
                    per[t] = per.get(t, 0) + 1
            out["max_per_club"] = int(max(per.values())) if per else 0
            done = m.groupby("gameweek").size()
            out["gw_complete"] = sorted(int(g) for g, n in done.items() if int(n) >= 10)
        if mx is not None:
            out["matches_xg"] = int(len(mx))
        out["team_channel"] = out["matches_xg"] >= (out["min_xg"] or 0)
        # the rate gates are per club; report each as open once every club could clear it
        out["npxg_channel"] = out["max_per_club"] >= (out["min_rate_npxg"] or 0)
        out["xa_channel"] = out["max_per_club"] >= (out["min_rate_xa"] or 0)
        # `rate_channel` means FULLY open (both quantities), so an existing reader is
        # never told the channel is on while half of it is still gated
        out["rate_channel"] = out["npxg_channel"] and out["xa_channel"]
    except Exception:
        pass
    return out


def minutes_per_start():
    """Realised 26/27 starts, minutes and MINUTES PER START, keyed on player_code.

    WHY THIS IS ON THE PAGE. `inseason.update_minutes` performs a Beta update on the
    START prior from realised starts, and that is all it does — `exp_minutes`, the
    minutes a player is expected to last GIVEN a start, is never touched by any
    in-season path (grep `exp_minutes` in src/inseason.py: no hits). The appearance feed
    carries the minutes column and nothing consumes it.

    The consequence is that a player who starts every week and is withdrawn at half time
    is, to the model, indistinguishable from one who plays every minute. Both register as
    "started"; both keep whatever `exp_minutes` their prior gave them. It biases the
    projection twice over, because `exp_minutes` scales attacking exposure as well as
    appearance points.

    It is worst for cold-start players, who have no measured `exp_minutes` at all and
    whose start probability is inferred from price and ownership. Tzolis is the case that
    exposed this: cold start, p_start 0.97 from the ownership calibration, two starts out
    of two, and 120 minutes across them — 60 a start, which the projection cannot see.

    This function does NOT change any projection. It surfaces the realised number beside
    the projection so the discrepancy is visible per player rather than having to be
    discovered. Fixing the estimator is a separate, calibrated piece of work.

    NEVER_STARTED. The same record exposes a second and larger failure, at the START
    probability rather than the minutes. `update_minutes` adds ONE pseudo-observation per
    realised match to a Beta prior that already carries one per prior-season match, so a
    player with two full seasons behind him arrives with ~34 pseudo-matches and two weeks
    of not being picked moves him almost nowhere. Dubravka after GW2: prior 0.880 on 34.1
    pseudo-matches, zero starts from two, posterior 0.831 — while Kinsky, who started both
    on a weaker prior, sits at 0.696. The model ranks the keeper who has not played above
    the one who has. Goalkeeper selection is close to deterministic and highly persistent,
    which is precisely the process a Beta over exchangeable coin flips misrepresents.
    `apply_availability` cannot catch it either: these players are FIT, just not picked.

    The view holds such players out of the "likely starters" pool on the realised record
    rather than the prior. That is a presentation guard, not a fix; the estimator change
    belongs in `src/inseason.py` behind its own calibration.

    HALF OF THAT ESTIMATOR CHANGE NOW EXISTS  [2026-09-07]
    `inseason.RECENCY_LAM` / `INSEASON_LAM` addresses the sentence above about exchangeable
    coin flips directly: realised matches are discounted by recency, so a run of recent
    non-starts outweighs older starts instead of averaging with them. It does NOT address
    the other half — the 34-pseudo-match prior a realised match is landing against — which
    is `INSEASON_KAPPA`. Dubravka needs both, and both are still OFF by default and
    validated on the start probability rather than on points (PROJECT_KNOWLEDGE §6.9).
    This function is unaffected either way: it reports the RAW realised record, which is
    what a presentation guard should key on, and passes no `lam`.
    """
    out = {}
    try:
        import inseason as ins
        ap = ins.appearances(upto_gw=38, verbose=False)
        if ap is None or not len(ap):
            return out
        # what he did in his club's most recent finished gameweek
        last = {}
        try:
            L = ins.last_appearance(upto_gw=38)
            for r in L.dropna(subset=["player_code"]).itertuples():
                last[int(r.player_code)] = {
                    "gw": int(r.gw), "mins": int(float(r.minutes or 0)),
                    "start": bool(float(r.starts or 0) > 0),
                }
        except Exception:
            pass
        for r in ap.dropna(subset=["player_code"]).itertuples():
            st = float(getattr(r, "starts", 0) or 0)
            mn = float(getattr(r, "minutes", 0) or 0)
            # minutes in STARTED matches over starts. Total minutes over starts is wrong
            # for a player who also appeared off the bench — his substitute minutes have
            # no start in the denominator and the ratio can come out above 90.
            sm = getattr(r, "start_minutes", None)
            sm = float(sm) if sm is not None and sm == sm else None
            out[int(r.player_code)] = {
                "starts": int(st), "mins": int(mn),
                "cm": int(float(getattr(r, "club_matches", 0) or 0)),
                "mps": (round(sm/st, 1) if (sm is not None and st >= 1) else None),
                "last": last.get(int(r.player_code)),
            }
    except Exception:
        pass
    return out


def live_squad():
    """The manager's own squad, if `src/fpl_entry.py` has been run.

    Optional in every sense: absent, the squad tab explains how to populate it and every
    other tab is unaffected. Keyed on `player_code`, so it joins the board directly
    rather than through the `web_name` match that `squad_tracker`'s hand-edited file
    needs."""
    try:
        import fpl_entry as fe
        sq, meta = fe.load()
        if sq is None or not len(sq):
            return None
        picks = []
        for r in sq.itertuples():
            picks.append({
                "c": int(r.player_code), "n": str(r.web_name),
                "p": str(r.pos), "t": str(r.team),
                "cost": float(r.now_cost), "slot": int(r.slot),
                "xi": bool(r.in_xi), "cap": bool(r.is_captain),
                "vice": bool(r.is_vice),
                "bench": (None if r.bench_order != r.bench_order else int(r.bench_order)),
            })
        return {"picks": picks, "meta": meta or {}}
    except Exception:
        return None


def wildcard_xi():
    """Per-gameweek optimal fifteens from `scripts/export_wildcard_xi.py`.

    Read, never solved here. Building a legal fifteen under budget, club and position
    constraints is an integer program and `src/solver.py` owns it; the page displays what
    the solver decided. Absent, the section explains how to generate it."""
    p = os.path.join(config.OUTPUTS, "wildcard_xi.csv")
    if not os.path.exists(p):
        return None
    try:
        w = pd.read_csv(p)
    except Exception:
        return None
    if not len(w):
        return None
    by = {}
    for gw, g in w.groupby("gw"):
        xi = g[g["in_xi"]]
        by[str(int(gw))] = {
            "xi_ep": round(float(xi["ep"].sum()), 2),
            "cap_ep": round(float(g.loc[g["is_captain"], "ep"].sum()), 2),
            "cost": round(float(g["price"].sum()), 1),
            "picks": [{"c": int(r.player_code), "n": str(r.player), "p": str(r.pos),
                       "t": str(r.team), "pr": float(r.price), "ep": round(float(r.ep), 2),
                       "xi": bool(r.in_xi), "cap": bool(r.is_captain),
                       "own": bool(r.owned)} for r in g.itertuples()],
        }
    freq = (w[w["in_xi"]].groupby(["player_code", "player", "pos", "team"])
             .agg(n=("gw", "size"), owned=("owned", "max")).reset_index()
             .sort_values("n", ascending=False).head(20))
    return {
        "gws": sorted(int(g) for g in w["gw"].unique()),
        "by": by,
        "freq": [{"c": int(r.player_code), "n": str(r.player), "p": str(r.pos),
                  "t": str(r.team), "w": int(r.n), "own": bool(r.owned)}
                 for r in freq.itertuples()],
    }


def played_fixtures():
    """The set of (team, gameweek) pairs already played, from 26/27 results.

    PER TEAM, not per gameweek. A gameweek is not a unit that is played or unplayed: at
    the time of writing GW2 is settled for eighteen clubs and live for two, so a single
    flag on the gameweek either dims a week that is still a decision for Arsenal and
    Villa, or leaves eighteen clubs' finished matches presented as forecasts. Both are
    wrong in the same table. The flag therefore belongs on the cell.

    Returns an empty set if the results feed is unreachable, which keeps this module
    offline-safe — the cost is a board with nothing dimmed, not a wrong one."""
    try:
        import inseason as ins
        m = ins.played(upto_gw=38, require_xg=False, verbose=False)
        if m is None or not len(m) or "gameweek" not in m.columns:
            return set()
        out = set()
        for r in m.itertuples():
            gw = int(r.gameweek)
            out.add((str(r.home), gw))
            out.add((str(r.away), gw))
        return out
    except Exception:
        return set()


def team_season(outputs=None):
    """The per-club season table from `export_team_projections`. Optional: the team
    dashboard degrades to fixture-derived quantities without it rather than failing."""
    outputs = outputs or config.OUTPUTS
    p = os.path.join(outputs, "team_projections_season.csv")
    if not os.path.exists(p):
        return None
    t = pd.read_csv(p)
    keep = [c for c in ("team", "att_strength", "def_strength", "att_rank", "def_rank",
                        "exp_pts", "exp_cs", "press_factor", "ppda_2627", "xga_env",
                        "xg_env", "promoted", "p_title_devig", "p_relegation",
                        "blended_elo", "market_strength_z",
                        # how much DefCon this club concedes to the OPPOSITION's
                        # defenders. The mirror of the `opp_defcon` schedule axis, which
                        # is what its own defenders face — keep the two labelled apart.
                        "defcon_conceded_hit", "defcon_opponent_category")
            if c in t.columns]
    return t[keep].drop_duplicates("team")


def results():
    """Realised 26/27 team-gameweeks: goals and xG, for and against.

    This is the only thing on the page that is not a projection, and it is kept
    strictly separate from one. It exists so the team trend can be read against what
    has actually happened rather than only against itself — the project has never
    scored the model on a real gameweek (PROJECT_KNOWLEDGE §6.6), and a chart that
    cannot show the outturn cannot start. Empty frame if the feed is unreachable."""
    cols = ["team", "gw", "gf", "ga", "xgf", "xga"]
    try:
        import inseason as ins
        m = ins.played(upto_gw=38, require_xg=False, verbose=False)
        if m is None or not len(m):
            return pd.DataFrame(columns=cols)
        rows = []
        for r in m.itertuples():
            gw = int(r.gameweek)
            rows.append({"team": str(r.home), "gw": gw, "gf": r.home_score,
                         "ga": r.away_score, "xgf": r.xg_home, "xga": r.xg_away})
            rows.append({"team": str(r.away), "gw": gw, "gf": r.away_score,
                         "ga": r.home_score, "xgf": r.xg_away, "xga": r.xg_home})
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)


def board_delta(board, history_dir=None, today=None):
    """Per player-gameweek change in `blended` since the last dated board snapshot.

    BASELINE IS A PREVIOUS DAY, never the previous run. The board is rebuilt several
    times in an afternoon; diffing against the last run would make the column read
    "since I last pressed go", which is noise dressed as information. `gw_board` writes
    one snapshot per day and overwrites within a day, so the newest file strictly older
    than today is a genuinely earlier state of the model.

    Returns [player_code, gw, d_blended], empty when there is no earlier snapshot — a
    first run, or a cleared cache. Empty is a normal outcome and the column simply does
    not appear, which is better than a column of zeros that cannot be told apart from
    "nothing changed".
    """
    import glob as _g
    hd = history_dir or getattr(config, "BOARD_HISTORY", None)
    cols = ["player_code", "gw", "d_blended"]
    if not hd or not os.path.isdir(hd):
        return pd.DataFrame(columns=cols)
    today = today or pd.Timestamp.now().strftime("%Y-%m-%d")
    snaps = []
    for f in sorted(_g.glob(os.path.join(hd, "board_*.csv"))):
        stamp = os.path.basename(f)[len("board_"):-len(".csv")]
        if stamp < today:
            snaps.append((stamp, f))
    if not snaps:
        return pd.DataFrame(columns=cols)
    stamp, path = snaps[-1]
    try:
        prev = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=cols)
    if not {"player_code", "gw", "blended"} <= set(prev.columns):
        return pd.DataFrame(columns=cols)
    prev = prev[["player_code", "gw", "blended"]].rename(
        columns={"blended": "_prev"}).drop_duplicates(["player_code", "gw"])
    cur = board[["player_code", "gw", "blended"]].drop_duplicates(["player_code", "gw"])
    j = cur.merge(prev, on=["player_code", "gw"], how="inner")
    j["d_blended"] = j["blended"] - j["_prev"]
    out = j[cols]
    out.attrs["baseline"] = stamp
    return out


def _first_clean(df, gws):
    """The gameweek the tail metrics are FOR: the first with nothing played."""
    try:
        fx = df.drop_duplicates(["team", "gw"])
        by = fx.groupby("gw")["played"]
        touched = {int(g) for g, s in by if s.any()}
        rest = [g for g in gws if int(g) not in touched]
        return int(rest[0]) if rest else int(gws[-1])
    except Exception:
        return int(gws[0])


def captaincy_tail(gw, draws_dir=None):
    """Tail metrics for one gameweek, from the BOARD'S OWN posterior draws.

    Captaincy is a tail problem, not a mean problem — you are doubling a score, so the
    spread matters as much as the centre — and `src/captaincy.py` has computed these all
    season. Nothing current consumed it: the only importer was `scripts/legacy/`, so the
    board, the explorer and every export ranked captaincy on the mean alone.

    WHY NOT CALL `captaincy.captain_picks` DIRECTLY. It re-derives the draws through its
    own `point_draws` with its own RNG, so its `ceiling` would disagree with the `p95`
    sitting next to it on the same row by pure Monte Carlo noise — the divergence trap
    `export_projection_detail` documents at length. These read `.cache/draws_gw<N>.npz`,
    written by `gw_board` itself under `DUMP_DRAWS`, so every number here comes from the
    same simulation as the rest of the board. The thresholds are imported from
    `captaincy` rather than restated, so "haul" cannot come to mean two things.

    KNOWN BIAS, STATED RATHER THAN SILENTLY CORRECTED  [studies/team_explosiveness_study]
    The engine draws team goals as Poisson, and the league's upper tail is measurably
    THINNER than that: 4+ goal team-games happen on 4.08% of team-matches where the model
    implies 6.54%, outside a simulated true-Poisson null at the 0.2nd percentile. So
    `p_haul` is inflated, and NOT uniformly — measured on GW4, corr(p_haul, fixture
    lambda) = +0.34 and the top 25 by `p_haul` sit at mean lambda 1.95 against 1.46 for
    the board as a whole. The captaincy shortlist is drawn from fixtures 34% above
    average lambda, which is precisely where the Poisson tail is too fat. The error is
    therefore concentrated at the TOP of the ranking, which distorts the ordering and not
    merely the level.

    It is surfaced anyway, uncorrected, because the alternative is what the tool does
    today: rank captaincy on the mean, which ignores the tail completely. A biased tail
    metric with its bias written down beats no tail metric. The correction exists
    (`team_explosiveness.apply_tail_calibration`) and stays OFF, because applying it moves
    the clean-sheet engine validated at GA r=0.89 / CS r=0.93 and that revalidation is a
    separate exercise against scored gameweeks.

    Returns {player_code: {...}}, empty if the dump for `gw` is absent.
    """
    import numpy as _np
    d = draws_dir or config.SCRATCH
    path = os.path.join(d, f"draws_gw{int(gw)}.npz")
    if not os.path.exists(path):
        return {}
    try:
        z = _np.load(path, allow_pickle=True)
        draws = z["draws"]
        codes = z["player_code"]
    except Exception:
        return {}
    if draws.ndim != 2 or len(codes) != draws.shape[0]:
        return {}
    try:
        from captaincy import HAUL_PTS, BLANK_PTS
    except Exception:
        HAUL_PTS, BLANK_PTS = 10, 2

    p_haul = (draws >= HAUL_PTS).mean(1)
    p_blank = (draws <= BLANK_PTS).mean(1)
    p_plays = (draws > 0).mean(1)
    ceiling = _np.percentile(draws, 95, axis=1)
    mean = draws.mean(1)

    # The template captain: the most-owned of the genuinely strong options. `regret` is
    # measured against him because the cost of a differential is not "fewer points", it
    # is "fewer points in the week everyone else scored", which is a JOINT event and
    # needs the two players' draws from the same posterior columns.
    own = None
    try:
        b = pd.read_csv(os.path.join(config.OUTPUTS, "gw_board_long.csv"))
        b = b[b["gw"] == int(gw)].drop_duplicates("player_code")
        own = b.set_index("player_code")["own"].to_dict()
    except Exception:
        pass
    strong = _np.flatnonzero(mean >= _np.quantile(mean, 0.90))
    tmpl_i = None
    if own and len(strong):
        best, bo = None, -1.0
        for i in strong:
            o = own.get(float(codes[i]), own.get(int(codes[i]), None))
            if o is not None and float(o) > bo:
                bo, best = float(o), int(i)
        tmpl_i = best
    if tmpl_i is None:
        tmpl_i = int(mean.argmax())
    td = draws[tmpl_i]
    t_haul, t_blank = td >= HAUL_PTS, td <= BLANK_PTS

    out = {}
    for i, c in enumerate(codes):
        if c != c:
            continue
        di = draws[i]
        regret = float(((di <= BLANK_PTS) & t_haul).mean())
        gain = float(((di >= HAUL_PTS) & t_blank).mean())
        out[int(c)] = {
            "p_haul": round(float(p_haul[i]), 4),
            "p_blank": round(float(p_blank[i]), 4),
            "p_plays": round(float(p_plays[i]), 4),
            "ceiling": round(float(ceiling[i]), 2),
            "regret": round(regret, 4),
            "gain": round(gain, 4),
            "tmpl": bool(i == tmpl_i),
        }
    return out


def season_to_date(season="2026-2027", base=None):
    """Realised 26/27 facts per `player_code`: price, price movement, ownership, net
    transfers, points, minutes.

    These come straight from `playerstats.csv` and are EXACT — no model, no snapshot, no
    inference. That matters because at three gameweeks the noisy ones and the exact ones
    look alike in a table: `total_points` over three matches is mostly variance and
    should not drive a pick, while `now_cost` and `cost_change_start` are facts that
    decide what you can literally afford. They are labelled accordingly.
    """
    root = base or config.repo(season)
    ps = os.path.join(root, "playerstats.csv")
    pl = os.path.join(root, "players.csv")
    if not (os.path.exists(ps) and os.path.exists(pl)):
        return {}
    try:
        d = pd.read_csv(ps)
        p = pd.read_csv(pl)
    except Exception:
        return {}
    if "id" not in d.columns or "gw" not in d.columns:
        return {}
    d = d[d["gw"] == d["gw"].max()]
    j = d.merge(p[["player_id", "player_code"]], left_on="id", right_on="player_id",
                how="left").dropna(subset=["player_code"])
    num = lambda r, c: (None if c not in j.columns or pd.isna(r.get(c))
                        else float(r[c]))
    out = {}
    for r in j.itertuples(index=False):
        row = r._asdict()
        tin, tout = row.get("transfers_in_event"), row.get("transfers_out_event")
        try:
            net = (float(tin) - float(tout)) if tin == tin and tout == tout else None
        except Exception:
            net = None
        # UNITS TRAP: `now_cost` arrives already in millions (12.0), but the
        # `cost_change_*` columns are in TENTHS (3 means +0.3m). Shown raw beside a
        # price in millions, a 0.1 rise reads as "1.00" — an order of magnitude out, and
        # plausible enough to be believed. Converted here, once, at the source.
        out[int(row["player_code"])] = {
            "cost": num(row, "now_cost"),
            "dcost_gw": (lambda v: None if v is None else v / 10.0)(
                num(row, "cost_change_event")),
            "dcost_season": (lambda v: None if v is None else v / 10.0)(
                num(row, "cost_change_start")),
            "own": num(row, "selected_by_percent"),
            "net_tr": net,
            "pts": num(row, "total_points"),
            "mins": num(row, "minutes"),
        }
    return out


def value_above_replacement(board, ep_col="blended"):
    """Per player-gameweek points above replacement, from `player_value`.

    NOT reimplemented here. `player_value.per_gameweek` owns the construct — floor
    price, best-EP-at-the-floor, marginal cost — and recomputes the floors inside every
    gameweek, which is the behaviour that makes `par` additive across a window: each
    week's par is measured against that week's own replacement, so the sum is "total
    points above replacement over the run".

    ELIGIBILITY. `position_floors` wants a start probability so that a 4.0m player who
    will never play cannot set the floor. The board carries no such column, but
    `app_ev` is the EV of the appearance points and saturates at 2 for a certain 60+
    minutes, so `app_ev / 2` is a serviceable proxy and is used only for that filter —
    it never enters a projection.

    Returns [player_code, gw, par, marginal_cost]; empty frame if the inputs are absent.
    """
    need = {"player_code", "gw", "pos", "cost", ep_col}
    if not need <= set(board.columns):
        return pd.DataFrame(columns=["player_code", "gw", "par", "marginal_cost"])
    try:
        import player_value as pv
    except Exception:
        return pd.DataFrame(columns=["player_code", "gw", "par", "marginal_cost"])
    b = board[list(need)].copy()
    b["_p_start"] = (pd.to_numeric(board.get("app_ev"), errors="coerce")
                     / 2.0).clip(0, 1) if "app_ev" in board.columns else 1.0
    try:
        v = pv.per_gameweek(b, ep_col=ep_col, price_col="cost", pos_col="pos",
                            gw_col="gw", start_col="_p_start")
    except Exception as e:                                        # noqa: BLE001
        print(f"[explorer] points-above-replacement unavailable ({e}); column omitted")
        return pd.DataFrame(columns=["player_code", "gw", "par", "marginal_cost"])
    keep = [c for c in ("player_code", "gw", "par", "marginal_cost") if c in v.columns]
    return v[keep].drop_duplicates(["player_code", "gw"])


# ---------------------------------------------------------------- assembly

def assemble(board, per_gw=None, per_club=None, played=None):
    """Board + fixture context, one row per player-gameweek.

    Every join is a LEFT join onto the board and every one asserts its row count. The
    board is authoritative: a missing annotation must not drop a projection, and a
    fanned-out join must not duplicate one."""
    gw_hi = int(board["gw"].max())
    df = board.copy()
    n0 = len(df)
    df = df.merge(fixtures(gw_hi), on=["team", "gw"], how="left")
    assert len(df) == n0, f"fixture join fanned out {n0} -> {len(df)}"
    if per_gw is not None:
        df = df.merge(per_gw, on=["team", "gw"], how="left")
        assert len(df) == n0, f"team-gameweek join fanned out {n0} -> {len(df)}"
    if per_club is not None:
        # Joined on OPPONENT, not on team. The DefCon rating is "actions this club
        # concedes to the opposition", so the number that belongs on a player's row is
        # his OPPONENT's rating. Joining it on `team` would invert the meaning and
        # still produce a full, plausible column.
        df = df.merge(per_club.rename(columns={
            "team": "opponent", "att": "opp_att", "def": "opp_def",
            "defcon_conceded_hit": "opp_defcon"}),
            on="opponent", how="left")
        assert len(df) == n0, f"opponent-strength join fanned out {n0} -> {len(df)}"
        # DEFENDER ROWS ONLY, for the PLAYER metric. The midfielder rating failed its
        # reliability gate and is not shipped; goalkeepers cannot score DefCon at all.
        # Blanking here rather than in the view means the CSV export, which is
        # METRICS-driven and player-shaped, carries the same restriction.
        #
        # `opp_defcon_fx` keeps the UNBLANKED value for the fixtures tab, which is
        # per CLUB and has no position at all — the rating is a property of the
        # opponent, not of whoever happens to be on the row. It is deliberately a
        # second column rather than a re-read: `_team_gw` takes one arbitrary row per
        # (team, gw) via drop_duplicates, so if it read the blanked column it would
        # return a value or a null depending on whether that club's first row that
        # week happened to be a defender. That is an order-dependent bug that would
        # show up as a half-empty fixtures grid.
        if "opp_defcon" in df.columns:
            df["opp_defcon_fx"] = df["opp_defcon"]
            if "pos" in df.columns:
                not_def = df["pos"].astype(str).str.upper() != "DEF"
                df.loc[not_def, "opp_defcon"] = np.nan
    dlt = board_delta(board)
    if not dlt.empty:
        df = df.merge(dlt, on=["player_code", "gw"], how="left")
        assert len(df) == n0, f"delta join fanned out {n0} -> {len(df)}"
    par = value_above_replacement(board)
    if not par.empty:
        df = df.merge(par, on=["player_code", "gw"], how="left")
        assert len(df) == n0, f"par join fanned out {n0} -> {len(df)}"
    done = played_fixtures() if played is None else played
    df["played"] = [(t, int(g)) in done for t, g in zip(df["team"], df["gw"])]
    return df


def _series(g, col, gws):
    """One metric as a dense list over `gws`, null where the player has no row."""
    if col not in g.columns:
        return None
    s = g.set_index("gw")[col].reindex(gws)
    return [None if pd.isna(v) else round(float(v), 3) for v in s]


def _teams_meta(df, teams):
    """Per-club season quantities, keyed by club."""
    if teams is None:
        return {}
    out = {}
    for _, r in teams.iterrows():
        t = str(r["team"])
        row = {}
        for k in teams.columns:
            if k == "team":
                continue
            v = r[k]
            if pd.isna(v):
                row[k] = None
            elif k == "promoted":
                row[k] = int(v)
            elif isinstance(v, str):
                # Categorical columns (the DefCon tercile) pass through as text. The
                # previous unconditional float() would have raised ValueError on the
                # first one, taking the whole team dashboard down.
                row[k] = v
            else:
                row[k] = float(v)
        out[t] = row
    return out


def _team_gw(df, gws):
    """Per club, the fixture run over `gws`: opponent, venue and the posterior
    quantities. Taken from the SAME frame the player rows were annotated from, so the
    fixtures tab and the players tab can never show a different opponent for a week.

    `opp_defcon_fx` is emitted as `opp_defcon` — the view has no notion of the
    blanked/unblanked split, which exists only to keep the player CSV honest."""
    cols = [c for c in ("lam_for", "lam_against", "p_clean_sheet", "p_win",
                        "opp_defcon_fx") if c in df.columns]
    fx = df.drop_duplicates(["team", "gw"]).set_index(["team", "gw"])
    out = {}
    for t in sorted(df["team"].dropna().unique()):
        row = {"opp": [], "h": [], "played": []}
        for c in cols:
            row[c] = []
        for g in gws:
            if (t, g) in fx.index:
                r = fx.loc[(t, g)]
                row["opp"].append(None if pd.isna(r.get("opponent")) else str(r["opponent"]))
                row["h"].append(None if pd.isna(r.get("is_home")) else int(r["is_home"]))
                row["played"].append(int(bool(r.get("played"))))
                for c in cols:
                    v = r.get(c)
                    row[c].append(None if pd.isna(v) else round(float(v), 4))
            else:
                row["opp"].append(None); row["h"].append(None); row["played"].append(0)
                for c in cols:
                    row[c].append(None)
        if "opp_defcon_fx" in row:
            row["opp_defcon"] = row.pop("opp_defcon_fx")
        out[str(t)] = row
    return out


def _results_block(res, gws):
    """Realised goals and xG per club-gameweek, aligned to `gws`. Nulls where a match
    has not been played — never zeros, which would read as a goalless draw."""
    if res is None or not len(res):
        return {}
    out = {}
    ri = res.drop_duplicates(["team", "gw"]).set_index(["team", "gw"])
    for t in sorted(res["team"].unique()):
        row = {k: [] for k in ("gf", "ga", "xgf", "xga")}
        for g in gws:
            for k in row:
                v = ri.loc[(t, g), k] if (t, g) in ri.index else None
                row[k].append(None if v is None or pd.isna(v) else round(float(v), 3))
        out[str(t)] = row
    return out


def _exact(row, board_col, std, std_key, stale):
    """The realised FPL value where there is one, the board's copy otherwise.

    Counts disagreements into `stale` so a board built against last week's prices
    announces itself instead of quietly pricing a transfer wrong."""
    b = None if pd.isna(row.get(board_col)) else float(row[board_col])
    rec = std.get(int(row["player_code"])) if pd.notna(row.get("player_code")) else None
    v = None if rec is None else rec.get(std_key)
    if v is None:
        return b
    if b is not None and abs(b - v) > 1e-9:
        stale[std_key] += 1
    return float(v)


def payload(df, board_path=None, notes=None, teams=None, res=None):
    """Everything the page needs, as one JSON-able dict."""
    gws = sorted(int(g) for g in df["gw"].unique())
    metrics = [(k, lab, doc) for k, lab, doc in METRICS if k in df.columns]
    key = "player_code" if "player_code" in df.columns and df["player_code"].notna().all() \
        else "player"
    mps = minutes_per_start()
    std = season_to_date()
    # PRICE AND OWNERSHIP ARE THE EXACT FACT, NOT THE BOARD'S COPY OF IT.
    # `now_cost` and `selected_by_percent` decide what you can literally afford and how
    # exposed you are, and the board carries its own copy taken when the board was built.
    # The two agree today — checked 2026-09-07, 653 of 653 players, max |difference| 0.0
    # in both — so this is a no-op on current inputs and a guard against reading a stale
    # board's prices as live ones. Divergence is counted and reported rather than
    # silently reconciled: a board built against different prices is a fact worth seeing.
    stale = {"cost": 0, "own": 0}
    # Tail metrics for the gameweek actually being picked for — the first with nothing
    # played. A captaincy question about a week already under way is not a question.
    tail = captaincy_tail(_first_clean(df, gws))
    players = []
    for _, g in df.groupby(key, sort=False):
        r = g.iloc[0]
        gi = g.set_index("gw")
        opp = gi["opponent"].reindex(gws) if "opponent" in g.columns else pd.Series(index=gws, dtype=object)
        hom = gi["is_home"].reindex(gws) if "is_home" in g.columns else pd.Series(index=gws, dtype=float)
        players.append({
            "c": None if pd.isna(r.get("player_code")) else int(r["player_code"]),
            "n": str(r["player"]),
            "p": str(r["pos"]),
            "t": str(r["team"]),
            "cost": _exact(r, "cost", std, "cost", stale),
            "own": _exact(r, "own", std, "own", stale),
            "opp": [None if pd.isna(v) else str(v) for v in opp],
            "h": [None if pd.isna(v) else int(v) for v in hom],
            "pd": [int(bool(v)) for v in gi["played"].reindex(gws).fillna(False)],
            "ap": (mps.get(int(r["player_code"]))
                   if pd.notna(r.get("player_code")) else None),
            "std": (std.get(int(r["player_code"]))
                    if pd.notna(r.get("player_code")) else None),
            "tail": (tail.get(int(r["player_code"]))
                     if pd.notna(r.get("player_code")) else None),
            "m": {k: _series(g, k, gws) for k, _, _ in metrics},
        })
    for k, n in stale.items():
        if n:
            print(f"[explorer] {n} players' {k} differed from playerstats.csv; the page "
                  f"shows the FPL value. The board is priced on its own copy, so `par` "
                  f"and gw_explorer.csv still carry the board's — rebuild the board.")
    # NOTE ON SHADING. There used to be a table of per-position quantiles baked in here,
    # computed over every player on the board. It was calibrated on the wrong population
    # and the page showed it: roughly half of each position barely plays, which drags the
    # 95th percentile down to where the players you actually pick begin. For midfielders
    # it landed at 3.38 while likely starters run 1.22-5.72, so the whole top half of the
    # useful range clamped to one colour — 17.7% of the cells a reader looks at were
    # painted the identical maximum, which is the "non-gradient block" you would see at
    # the top of the table.
    #
    # The scale is now computed IN THE PAGE, over the same reference pool the percentile
    # profiles use and over the selected window, so it re-derives when either changes.
    # That is deliberately not "rescale to the visible rows" — filtering to six players
    # must not repaint them across the full ramp — the pool is a fixed, named population,
    # not whatever survived the current filter.
    # Two different horizon markers, because a partly-played gameweek is neither past
    # nor future. `first_live` is the earliest week with any fixture still to come — the
    # earliest a window may legitimately start. `first_clean` is the earliest week where
    # NOTHING has been played, and it is what "next N" counts from and where the default
    # window opens: a window that starts on a week already settled for eighteen of twenty
    # clubs is summing history for almost everyone in it.
    fx = df.drop_duplicates(["team", "gw"])
    by_gw = fx.groupby("gw")["played"]
    fully = sorted(int(g) for g, s in by_gw if s.all())
    partial = sorted(int(g) for g, s in by_gw if s.any() and not s.all())
    live = [g for g in gws if g not in fully]
    clean = [g for g in gws if g not in fully and g not in partial]
    return {
        "meta": {
            "board": os.path.basename(board_path) if board_path else "gw_board_long.csv",
            "built": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "n_players": len(players),
            "gw_lo": gws[0], "gw_hi": gws[-1],
            "first_live": live[0] if live else gws[-1] + 1,
            "first_clean": clean[0] if clean else (live[0] if live else gws[-1] + 1),
            "notes": notes or [],
            "evidence": evidence(),
        },
        "gws": gws,
        "played_gws": fully,
        "partial_gws": partial,
        "teams": sorted(df["team"].dropna().unique().tolist()),
        "positions": [p for p in ("GK", "DEF", "MID", "FWD") if p in set(df["pos"])],
        "metrics": [{"k": k, "label": lab, "doc": doc} for k, lab, doc in metrics],
        # A profile axis whose column is absent would render as an empty ring for every
        # player rather than as nothing, which reads as "measured, and zero". `calc` axes
        # are computed in the page from other columns and are always available.
        "player_components": [{"k": k, "label": lab, "agg": a, "inv": inv, "doc": d}
                              for k, lab, a, inv, d in PLAYER_COMPONENTS
                              if a == "calc" or k in df.columns],
        "pool_n": POOL_N,
        "team_components": [{"k": k, "label": lab, "agg": a, "inv": inv, "doc": d}
                            for k, lab, a, inv, d in TEAM_COMPONENTS],
        "squad": live_squad(),
        "wildcard": wildcard_xi(),
        "teams_meta": _teams_meta(df, teams),
        "team_gw": _team_gw(df, gws),
        "results": _results_block(res, gws),
        "invert": sorted(INVERT),
        "additive": sorted(ADDITIVE),
        "abbr": {t: ABBR.get(t, str(t)[:3].upper()) for t in df["team"].dropna().unique()},
        "players": players,
    }


def long_csv(df):
    """The same table, tidy, for pivoting elsewhere: identity, window keys, fixture,
    then every metric."""
    head = [c for c in ("player_code", "player", "pos", "team", "cost", "own", "gw",
                        "played", "opponent", "is_home") if c in df.columns]
    rest = [k for k, _, _ in METRICS if k in df.columns and k not in head]
    tail = [c for c in ("solio", "src") if c in df.columns]
    return df[head + rest + tail].sort_values(["player", "gw"])


# ---------------------------------------------------------------- render

def render_html(pay):
    html = _template().replace("__PAYLOAD__", json.dumps(pay, separators=(",", ":")))
    assert "__PAYLOAD__" not in html
    return html


def build(board_path=None, out_html=None, out_csv=None, verbose=True):
    board = load_board(board_path)
    gw_hi = int(board["gw"].max())
    per_gw, per_club = team_layer(gw_hi)
    notes = []
    if per_gw is None:
        notes.append(
            f"No team_projections_gw1_N.csv covering GW{gw_hi} was found, so the per-fixture "
            f"lambdas, clean-sheet and win probabilities are absent from this build. "
            f"GW_HI={gw_hi} python scripts/export_team_projections.py adds them.")
    else:
        notes.append(f"Per-fixture lambdas joined from {per_gw.attrs.get('source')}.")
    if per_club is None:
        notes.append("No fitted club-strength export found; the opponent attack/defence "
                     "columns are absent from this build.")
    df = assemble(board, per_gw, per_club)
    tm = team_season()
    if tm is None:
        notes.append("No team_projections_season.csv found; the team dashboard falls "
                     "back to fixture-derived quantities only.")
    rs = results()
    df_res = rs if len(rs) else None
    pay = payload(df, board_path=board_path, notes=notes, teams=tm, res=df_res)
    out_html = out_html or os.path.join(config.OUTPUTS, "gw_explorer.html")
    out_csv = out_csv or os.path.join(config.OUTPUTS, "gw_explorer.csv")
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(render_html(pay))
    long_csv(df).round(4).to_csv(out_csv, index=False)
    if verbose:
        print(f"[explorer] {pay['meta']['n_players']} players x GW{pay['meta']['gw_lo']}-"
              f"{pay['meta']['gw_hi']}, {len(pay['metrics'])} per-gameweek metrics")
        print(f"[explorer] first fully-unplayed gameweek: GW{pay['meta']['first_clean']}"
              f"  (complete: {pay['played_gws'] or 'none'};"
              f" part-played: {pay['partial_gws'] or 'none'})")
        for n in notes:
            print(f"[explorer] {n}")
        _e = pay["meta"]["evidence"]
        print(f"[explorer] in-season evidence: {_e['matches']} matches "
              f"({_e['matches_xg']} with xG), complete gameweeks {_e['gw_complete'] or 'none'}; "
              f"team channel {'ON' if _e['team_channel'] else 'off'}, "
              f"npxG channel {'ON' if _e.get('npxg_channel') else 'off'} "
              f"(opens at {_e.get('min_rate_npxg')}/club), "
              f"xA channel {'ON' if _e.get('xa_channel') else 'off'} "
              f"(opens at {_e['min_rate']} matches per club, currently {_e['max_per_club']})")
        _mp = [p["ap"]["mps"] for p in pay["players"]
               if p.get("ap") and p["ap"].get("mps") is not None and p["ap"]["starts"] >= 2]
        if _mp:
            _low = sum(1 for v in _mp if v < 70)
        _bench = [p for p in pay["players"]
                  if p.get("ap") and p["ap"]["starts"] == 0 and p["ap"]["cm"] >= 2]
        if _bench:
            print(f"[explorer] {len(_bench)} players have started NOTHING in 2+ club matches "
                  f"while the model still expects them to; they are held out of the "
                  f"likely-starter pool (see gw_explorer NEVER_STARTED)")
        _lg = [p["ap"]["last"] for p in pay["players"] if p.get("ap") and p["ap"].get("last")]
        if _lg:
            _gw = max(x["gw"] for x in _lg)
            print(f"[explorer] last-match minutes attached for {len(_lg)} players "
                  f"(latest finished gameweek GW{_gw}; "
                  f"{sum(1 for x in _lg if x['mins'] == 0)} did not feature)")
            print(f"[explorer] realised minutes-per-start attached for {len(_mp)} players "
                  f"with 2+ starts; {_low} are averaging under 70 minutes a start "
                  f"(the projection cannot see this — see gw_explorer.minutes_per_start)")
        if pay.get("squad"):
            _m = pay["squad"]["meta"]
            print(f"[explorer] live squad: {_m.get('entry_name','?')} GW{_m.get('gw','?')}, "
                  f"{len(pay['squad']['picks'])} picks, budget "
                  f"{(_m.get('squad_value') or 0) + (_m.get('bank') or 0):.1f}")
        else:
            print("[explorer] no live squad on file — run "
                  "`python src/fpl_entry.py --team-id <id>` to add the squad tab")
        if pay.get("wildcard"):
            _w = pay["wildcard"]
            print(f"[explorer] wildcard XI: GW{_w['gws'][0]}-{_w['gws'][-1]} solved "
                  f"({len(_w['gws'])} gameweeks)")
        else:
            print("[explorer] no wildcard XI on file — run "
                  "`python scripts/export_wildcard_xi.py`")
        print(f"[explorer] team dashboard: {len(pay['teams_meta'])} clubs with season "
              f"metrics, {len(pay['results'])} with realised results")
        print(f"[explorer] wrote {out_html}  ({os.path.getsize(out_html)/1e6:.1f} MB)")
        print(f"[explorer] wrote {out_csv}")
    return out_html, out_csv


# ---------------------------------------------------------------- selftest

def selftest():
    """Offline, on a synthetic board. Exercises the joins, the blank-gameweek path, the
    played-week flag and the duplicate-key guard without touching outputs or the
    network."""
    import tempfile
    from schedule_2627 import schedule
    _, sched = schedule()
    teams = sorted(sched["team"].unique())[:4]
    rng = np.random.default_rng(0)
    rows = []
    for i, t in enumerate(teams):
        for j in range(3):
            for gw in range(1, 6):
                # player 0 of team 0 has no GW4 row — the blank-gameweek path
                if i == 0 and j == 0 and gw == 4:
                    continue
                rows.append({"player_code": 1000 + i * 10 + j, "player": f"P{i}{j}",
                             "pos": ["GK", "DEF", "MID"][j], "team": t, "cost": 4.5 + j,
                             "own": 5.0, "gw": gw, "mean": float(rng.normal(4, 1)),
                             "blended": float(rng.normal(4, 1)), "sd": 2.0,
                             "app_ev": 1.8, "att_ev": 1.0, "cs_ev": 0.5,
                             "defcon_ev": 0.2, "conc_ev": -0.1, "p5": 0.0,
                             "median": 3.0, "p95": 9.0})
    board = pd.DataFrame(rows)
    # GW1 complete for all four clubs, GW2 complete for one — the part-played week
    done = {(t, 1) for t in teams} | {(teams[0], 2)}
    df = assemble(board, played=done)
    assert len(df) == len(board), f"assemble changed the row count {len(board)} -> {len(df)}"
    assert df["opponent"].notna().all(), "a fixture failed to join"
    assert set(df.loc[df["played"], "gw"]) == {1, 2}, "played flag is wrong"
    assert df.loc[df["played"], "team"].nunique() == 4
    assert set(df.loc[df["played"] & (df["gw"] == 2), "team"]) == {teams[0]}

    pay = payload(df)
    assert pay["meta"]["n_players"] == 12, pay["meta"]["n_players"]
    assert pay["meta"]["first_clean"] == 3, pay["meta"]
    assert pay["played_gws"] == [1] and pay["partial_gws"] == [2], pay["played_gws"]
    p0 = [p for p in pay["players"] if p["n"] == "P00"][0]
    assert len(p0["m"]["blended"]) == 5
    assert p0["m"]["blended"][3] is None, "blank gameweek did not survive as null"
    assert p0["m"]["blended"][0] is not None
    assert p0["opp"][3] is None and p0["opp"][0] is not None
    assert p0["pd"] == [1, 1, 0, 0, 0], p0["pd"]

    # THE POINTS-PER-£m COLUMN IS RETIRED (2026-09-07) and must not come back under any
    # name: it is the near-floor artefact `player_value` documents, and beside `par` it is
    # a second column claiming to answer the same question with the wrong answer.
    assert not [c for c in pay["player_components"] if c["k"] == "perM"],         "the retired points-per-£m profile axis is back"
    # `par` took its place as the value axis, and it is the SUM over the window — the
    # aggregate that construct supports — not a mean.
    _val = [c for c in pay["player_components"] if c["k"] == "par"]
    assert _val and _val[0]["agg"] == "sum", pay["player_components"]
    # A profile axis is offered only where the board can fill it: a board `player_value`
    # could not price must drop the value axis rather than draw an empty ring for every
    # player. The `calc` axes, computed in the page, always survive.
    _bare = payload(df.drop(columns=["par"]))
    assert not [c for c in _bare["player_components"] if c["k"] == "par"],         "a par axis was offered on a board that carries no par"
    assert [c for c in _bare["player_components"] if c["k"] == "defcon_stab"],         "the availability filter dropped a calc axis"

    # PRICE AND OWNERSHIP are the realised FPL fact, and a board that disagrees with it is
    # counted, not silently preferred either way.
    st = {"cost": 0, "own": 0}
    row = {"player_code": 1000, "cost": 4.5, "own": 5.0}
    fpl = {1000: {"cost": 4.6, "own": 5.0}}
    assert _exact(row, "cost", fpl, "cost", st) == 4.6 and st["cost"] == 1, st
    assert _exact(row, "own", fpl, "own", st) == 5.0 and st["own"] == 0, st
    assert _exact(row, "cost", {}, "cost", st) == 4.5, "no FPL row must fall back to the board"

    html = render_html(pay)
    assert html.lstrip().startswith("<!DOCTYPE"), "template did not render"
    assert "perM" not in html, "the retired points-per-£m column is back in the page"
    assert json.dumps(pay["gws"]) in html or '"gws"' in html
    assert len(long_csv(df)) == len(board)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gw_board_long.csv")
        pd.concat([board, board.iloc[[0]]], ignore_index=True).to_csv(p, index=False)
        caught = False
        try:
            load_board(p)
        except AssertionError:
            caught = True
        assert caught, "a duplicate (player_code, gw) was not caught"
    print("gw_explorer selftest OK")


def _template():
    """The page shell, read from `config.EXPLORER_TEMPLATE` at render time."""
    with open(config.EXPLORER_TEMPLATE, encoding="utf-8") as fh:
        return fh.read()




if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        build()
