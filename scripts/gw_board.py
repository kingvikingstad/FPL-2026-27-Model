"""
gw_board.py — week-by-week per-player predictions, GW1..GW_HI.
==============================================================
Produces TRUE per-gameweek projections (project(gw,gw) per week, not a horizon
aggregate split proportionally), on the full validated stack: two-season priors,
depth prior, minutes shrinkage, availability, betting-odds team strength
(market_odds, embedded snapshot), DefCon environment (xGA + press index).

Solio weighting (offline): the cached Solio feed (data/solio_cache.md) carries
their GW1 projections. For matched players, GW1 is blended
    blend = w_ours * ours + (1 - w_ours) * solio        (default w_ours=0.5;
set SOLIO_W_OURS, or from inverse-MAE once results exist). Later GWs are ours
alone — Solio publishes single-GW only. Blended cells are flagged.

Outputs (in config.OUTPUTS):
  gw_board_long.csv  player x gw rows: mean, sd, p5..p95, defcon_ev, cs_ev, solio, blend
  gw_board_wide.csv  one row per player, gw1..gwN blended means + total
Env: GW_HI (default 38 = full season), SOLIO_W_OURS (default 0.5), SOLIO=off, MARKET_ODDS=off,
     MARKET_WEIGHT, DEFCON_ENV=off, REGIME_PANEL=on, REGIME=kappa|proposed,
     INSEASON=off (default ON), INSEASON_UPTO, INSEASON_W_MIN, INSEASON_W_MATCH,
     INSEASON_W_RATE, INSEASON_EXP_MINUTES=on (default off), INSEASON_EXP_K,
     INSEASON_KAPPA=<n> (default off/inf),
     SOLIO_MARKET=on (default off; requires MARKET_ODDS=off), SOLIO_MARKET_W,
     SOLIO_MARKET_GW, XI_CONSTRAINT=off,
     FPL_SETPIECE=fpl|override|fill|observed (default fpl; `observed` is the
       unshrunk n=1 channel withdrawn on 2026-09-08 — see the note at its use),
     INJURY_IMPACT=on (default off), TEAM_OVERRIDES=on (default off).

INSEASON: fold 26/27 results back into the priors — realised starts into the Beta
minutes prior, and finished matches' xG (not scorelines) into the team model's E0
stack. ON by default since 2026-09-01: once the season is under way, projecting from
last season alone is the WORSE estimator, not the neutral one, and a default that
quietly discards every realised start is a bias you have to remember to turn off. Set
INSEASON=off to reproduce a pre-season board. See src/inseason.py for the 24-31 season
calibration; the weights are small and grow with k, so early on this changes little.

REGIME: `kappa` widens the posterior on regime clubs without moving any point
estimate (the project's stated position — regime change is an ignorance statement).
`proposed` additionally applies the UNFITTED delta mean-pulls, which shift point
estimates on assertion; that is a sensitivity run, not the headline board. See
starter_prior.resolve_regime.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np, pandas as pd
import core_insights as ci, bayes_model, signals as sg, starter_prior as sp, defcon_env as de
import solio_ensemble as se
from bayes_model import TeamModel, project
from roster import calibrate_cold_start, _coldstart_row
from schedule_2627 import schedule

def _flag(name, default, on_values=None, off_values=("off", "0")):
    """Resolve one flag and record it for the banner."""
    raw = os.environ.get(name, default)
    v = str(raw).lower() if raw is not None else ""
    on = (v in on_values) if on_values else (v not in off_values)
    _FLAGS.append((name, "ON " if on else "off", str(raw)))
    return on


_FLAGS = []


def _banner():
    """Print exactly what this run has switched on, before it does anything.

    Every layer here is env-gated and several default OFF, so two runs of the same
    command can produce different boards with nothing in the output to say why. The
    README drifted out of date on precisely this within a week. A banner cannot drift:
    it reports the flags the run actually resolved.
    """
    print("=" * 78)
    print("BOARD CONFIGURATION")
    print("=" * 78)
    for name, state, raw in _FLAGS:
        note = "" if raw in ("", "None") else f"   ({raw})"
        print(f"  {state}  {name}{note}")
    # Every env gate in this file must be represented. If a new one is added and not
    # routed through `_flag`, this fails loudly rather than quietly under-reporting.
    import re as _re
    _src = open(__file__, encoding="utf-8").read()
    _gated = set(_re.findall(r'_flag\(\s*"([A-Z_0-9]+)"', _src))
    _gated |= {"FPL_SETPIECE", "PRED_XI_GW"}
    _seen = {n for n, _, _ in _FLAGS}
    _missing = _gated - _seen
    assert not _missing, f"flags gated but absent from the banner: {sorted(_missing)}"
    off = [n for n, st, _ in _FLAGS if st.strip() == "off"]
    if off:
        print(f"  -> OFF this run: {', '.join(off)}")
    print("=" * 78)


# Full season by default. A 10-gameweek board cannot answer a question about the next
# ten once any week has been played, and the failure is silent: the window just stops
# early and the totals quietly compare different numbers of fixtures. Projecting a
# distant gameweek is not a claim that the projection is as good as a near one — the
# priors are the same estimator either way, and the fixture is known — but it is the
# horizon the decisions actually run over. GW_HI=10 restores the old behaviour.
GW_HI = int(os.environ.get("GW_HI", "38"))
S = int(os.environ.get("DRAWS", "3000"))
W_OURS = float(os.environ.get("SOLIO_W_OURS", "0.5"))

# ---------- frame (identical to run_final_board) ----------
d26, t26, _ = ci.load(base=config.repo("2026-2027")); sig = ci.to_signals(d26)
cal_cs = calibrate_cold_start(hist_csv=config.COLDSTART_HIST)
cal_own = sp.calibrate_ownership_start()
pri = pd.read_pickle(config.MS_PRIORS).dropna(subset=["player_code"]).drop_duplicates("player_code")
pmap = pri.set_index("player_code")
rows = []
for _, r in d26.iterrows():
    c = r.get("player_code")
    if pd.notna(c) and c in pmap.index:
        s = pmap.loc[c]
        row = {k: s[k] for k in ["npxgi_alpha","npxgi_beta","xa_alpha","xa_beta",
                                 "defcon_alpha","defcon_beta","start_a","start_b","sub_app_rate","exp_minutes"]}
        row.update({"id": r.player_id, "player_code": c, "web_name": r.web_name, "pos": r.pos,
                    "team": r.team, "own": r.selected_by_percent, "cost": r.now_cost,
                    "minutes": s["minutes"], "cold_start": False,
                    "pen_xg90": float(s.get("pen_xg90_measured", 0) or 0)}); rows.append(row)
    else:
        cs = _coldstart_row(r.web_name, r.team, r.pos, r.now_cost, r.selected_by_percent, cal_cs)
        cs["minutes"] = 0.0; cs["player_code"] = c; rows.append(cs)
pl = pd.DataFrame(rows)
pl = sp.apply_coldstart_depth(pl, cal_own)
if _flag("REGIME_PANEL", None, on_values=("on",)):
    import regime_panel as rpn
    pl = rpn.apply_regime_panel_split(pl, pd.read_pickle(config.PMS_PANEL), config.REPO, w_pre=0.30)
_regime, _regime_label = sp.resolve_regime()
print(f"[regime] {_regime_label}" + (f" — {len(_regime)} clubs" if _regime else ""))
pl = sp.apply_regime_uncertainty(pl, regime=_regime, cal=cal_own, k_min=900)
# Availability. The repo snapshot trails team news and, at pre-season, trails completed
# transfers — so prefer the live endpoint and fall back loudly rather than silently.
# LIVE_FPL=off pins to the snapshot for a reproducible offline run.
sig["player_code"] = d26["player_code"].values
if _flag("LIVE_FPL", "on"):
    try:
        _live = sg.fetch_live_signals()
        _cover = d26["player_code"].isin(_live["player_code"]).mean()
        if _cover < 0.9:
            raise RuntimeError(f"live feed covers only {_cover:.0%} of the squad")
        _out = _live[_live.status.isin(list("isun")) | (_live.chance_play <= 0)]
        print(f"[live-fpl] availability from the live endpoint: {len(_live)} players, "
              f"{len(_out)} ruled out, {_cover:.0%} squad coverage")
        # carry the snapshot's set-piece orders across; the live frame is an
        # availability feed and the set-piece override still needs them downstream
        _sp = sig[["player_code", "pen_order", "fk_order", "corner_order"]]
        sig = _live.merge(_sp, on="player_code", how="left")
    except Exception as e:
        print(f"[live-fpl] UNAVAILABLE ({type(e).__name__}: {e}) — using the repo "
              f"snapshot, which may trail team news")
else:
    print("[live-fpl] off — using the repo snapshot")
# Baseline start prior, captured BEFORE availability collapses it. injury_impact needs
# to know what a player was expected to contribute, which is unrecoverable once his
# prior has already been zeroed by the injury we are trying to price.
# The XI constraint USED to run here, before availability. It now runs after it — see
# the block following `apply_availability` below. Moving it is a correction, not a
# preference: normalising to eleven and then zeroing players out of that eleven destroyed
# the mass instead of reallocating it, and left the league at 184.9 expected starters
# against a structural 220.
pl["p_start_base"] = pl["start_a"] / (pl["start_a"] + pl["start_b"])

# In-season evidence. ON by default (INSEASON=off to disable). The minutes channel goes in HERE,
# before availability, because the order encodes a precedence: an appearance record is
# evidence about a player's role, a current injury flag is knowledge about this weekend,
# and the second must win. Running it after would let a start in GW1 partially undo a
# GW3 injury. See src/inseason.py for the calibration behind the weights.
_INSEASON = _flag("INSEASON", "on", on_values=("on", "1", "true"))
# Resolved UNCONDITIONALLY, outside the INSEASON branch. `_banner` asserts that every
# name it finds behind a `_flag(` call was actually registered, so a flag resolved inside
# a branch crashes the board whenever that branch is skipped — which is exactly what
# happened when these two were first added and INSEASON=off. The banner caught it, which
# is what it is for; the fix is to resolve here and act below.
_EXP_MINUTES = _flag("INSEASON_EXP_MINUTES", "off", on_values=("on", "1", "true"))
_kap = os.environ.get("INSEASON_KAPPA", "").strip()
_FLAGS.append(("INSEASON_KAPPA", "ON " if _kap else "off", _kap or "inf"))
_lam = os.environ.get("INSEASON_LAM", "").strip()
_FLAGS.append(("INSEASON_LAM", "ON " if _lam else "off", _lam or "1.0 (flat)"))
_e0_path = config.E0_RECON
if _INSEASON:
    import inseason as ins
    # INSEASON_LAM discounts realised matches by recency inside the Beta update, which is
    # otherwise exchangeable and so cannot tell start-start-bench from bench-start-start.
    # OFF by default; lam=1.0 is the flat update exactly. The fitted value is a function
    # of FORECAST HORIZON (inseason.RECENCY_LAM_H10 = 0.75 at this board's ~10 gameweeks)
    # and was fitted at kappa=4, so set INSEASON_LAM=0.75 together with INSEASON_KAPPA=4.
    _apps = ins.appearances(upto_gw=int(os.environ.get("INSEASON_UPTO", "38")),
                            lam=float(_lam) if _lam else ins.RECENCY_LAM)
    # Cap the Beta start prior BEFORE the realised matches land on it, or they land on
    # the raw 30+ pseudo-match prior and barely move it. OFF by default: the calibration
    # passed (studies/start_prior_strength.py) but only the ratio w/kappa is identified
    # and it was fitted against a previous-season prior, not this one — see
    # PROJECT_KNOWLEDGE 6.9. INSEASON_KAPPA=5 turns it on.
    if _kap:
        pl, _ = ins.cap_start_prior(pl, kappa=float(_kap))
    pl, _mrep = ins.update_minutes(
        pl, _apps, weight=float(os.environ.get("INSEASON_W_MIN", ins.W_MINUTES)))
    # How long he lasts GIVEN a start, which no in-season path touched before. OFF by
    # default for the same reason: validated on the parameter (studies/minutes_per_start.py),
    # not yet on points. INSEASON_EXP_MINUTES=on turns it on.
    if _EXP_MINUTES:
        pl, _xrep = ins.update_exp_minutes(
            pl, _apps, k_half=float(os.environ.get("INSEASON_EXP_K", ins.EXP_MINUTES_K)))
    # Player attacking rates. Calibrated 2026-08-27 and now ON at w=1.0; it shipped at
    # 0.0 only because no calibration existed. TWO gates, each where that quantity
    # actually cleared its decision rule: npxG from 3 league matches, xA from 5. A single
    # gate at 5 held npxG back on the strength of xA's noise (see inseason §4).
    pl, _rrep = ins.update_rates(
        pl, ins.rates(upto_gw=int(os.environ.get("INSEASON_UPTO", "38")), verbose=True),
        weight=float(os.environ.get("INSEASON_W_RATE", ins.W_RATE)))
    _mx = ins.played(upto_gw=int(os.environ.get("INSEASON_UPTO", "38")),
                     require_xg=True, verbose=False)
    if len(_mx) >= ins.MIN_XG_MATCHES:
        _e0_path, _n = ins.stack_e0(
            _mx, weight=int(os.environ.get("INSEASON_W_MATCH", int(ins.W_MATCH))),
            weight_promoted=int(os.environ.get("INSEASON_W_PROMOTED",
                                               int(ins.W_MATCH_PROMOTED))))
    else:
        print(f"[inseason] team layer OFF: {len(_mx)} matches carry xG, "
              f"need {ins.MIN_XG_MATCHES} — minutes channel still applied")
else:
    print("[inseason] off — projecting from last season only")

# Solio's market-implied lambda for the UPCOMING gameweek, appended to E0 as
# forward-looking rows. OFF by default (SOLIO_MARKET=on), and it refuses to run unless
# MARKET_ODDS=off, because market_odds.py already blends the outright market into the
# ClubElo path below at weight 0.6 — see src/solio_market.py for why stacking both
# double-counts, and why SOLIO_W is uncalibrated rather than merely untuned.
if _flag("SOLIO_MARKET", "off", on_values=("on", "1", "true")):
    import solio_market as sm
    _sgw = os.environ.get("SOLIO_MARKET_GW")
    _snaps = sm.load_snapshots(gw=int(_sgw) if _sgw else None)
    if not _snaps:
        print("[solio-market] no stored snapshot — run `python src/solio_market.py --fetch`")
    else:
        _e0_path, _sn, _srep = sm.stack_e0(
            _snaps[-1][1], e0_path=_e0_path,
            weight=int(os.environ.get("SOLIO_MARKET_W", sm.W_FIXTURE)))

pl = sg.apply_availability(pl, sig, lineups=None)

# ELEVEN PLAYERS START. Applied AFTER availability [CORRECTED 2026-09-08]. Validated
# against GW1 in its original position: expected starters 260.6 -> 195.6 against an
# actual 183, Brier 0.109 -> 0.077, log loss 0.364 -> 0.276. That validation measured the
# constraint against an UNCONSTRAINED board and remains the reason the layer exists; it
# did not measure the ordering, and the ordering was wrong. Running before availability
# normalised each club to eleven and then let injuries delete players out of that eleven,
# so the surplus vanished rather than moving to whoever actually starts — league 220.0 ->
# 184.9, nineteen clubs under 10.5, Aston Villa 8.06. Every EV component downstream
# (appearance, attacking, DefCon, clean sheet) was scaled down with it, worst at the clubs
# with the most team news.
#
# `hold` preserves what the old ordering bought by accident: availability's ruled-out
# players keep 0.01 exactly and are excluded from the tilt, so an injury flag still beats
# the constraint. `p_start_base` is captured ABOVE, pre-availability, because
# injury_impact needs the uncollapsed baseline.
if _flag("XI_CONSTRAINT", "on"):
    pl, _xirep = sp.apply_xi_constraint(pl, hold=pl.get("avail_ruled_out"))
    # BUILD-TIME INVARIANT. Checked here rather than on gw_board_long.csv because the
    # board does not carry p_start, and because a board that violates this should not be
    # written at all. The old ordering failed it at 184.9/220 and nothing noticed for
    # weeks: every downstream EV was scaled down together, so no ratio looked wrong and
    # the marginals stayed self-consistent. Clubs the solver SKIPS (fewer listed players
    # than the residual target) are reported, not asserted — they are unsolvable, not
    # wrong.
    if len(_xirep):
        _solved = _xirep[_xirep["shift"].abs() > 0]
        _bad = _solved[(_solved["after"] - sp.XI_SIZE).abs() > 0.25]
        _skipped = _xirep[_xirep["shift"] == 0]
        if len(_bad):
            raise SystemExit(
                "[xi-constraint] INVARIANT VIOLATED — these clubs do not sum to eleven "
                "after the tilt:" + chr(10) + _bad.to_string(index=False))
        _lg = float(_xirep["after"].sum())
        print(f"[xi-constraint] league {_lg:.1f} expected starters across "
              f"{len(_xirep)} clubs ({len(_skipped)} skipped)")
else:
    print("[xi-constraint] off")

# Predicted XIs (team news). Applied AFTER availability so it refines a prior that
# already knows who is injured, and applied SOFTLY — a prediction the day before a
# deadline is not a teamsheet.
#
# IT APPLIES TO ONE GAMEWEEK ONLY. A GW1 teamsheet says nothing about GW5, so it is
# held in a SEPARATE prior set used for that gameweek alone; every later gameweek
# projects from the untouched priors. Folding it into `pl` would pin ten gameweeks to
# one week's rotation, which is a bigger error than not using the team news at all.
def _default_pred_xi_gw():
    """The gameweek a teamsheet would be FOR: the first with nothing played yet.

    This defaulted to a hardcoded 1 and stayed there. Correct in August, silently wrong
    from GW2 onward: on 2026-09-06 the board was still applying 20 August's GW1 team news
    and running GW2-38 on untouched priors, so the single input the project's own scoring
    identifies as its largest error source — availability, 13 misses worth 52.6 projected
    points in GW1 — was dark for every gameweek anyone was actually picking.

    A hardcoded default cannot age well here because the right answer changes weekly. It
    is derived instead, from the same marker the explorer and the wildcard solver use:
    the first gameweek in which NOTHING has been played. A partly-played week is past for
    selection purposes — you cannot pick into it — so a teamsheet for it is worthless.

    Falls back to 1 if the in-season feed is unreachable, which is what it always did.
    """
    try:
        import inseason as _ins
        m = _ins.played(upto_gw=38, require_xg=False, verbose=False)
        if m is None or not len(m):
            return 1
        done = set()
        for r in m.itertuples():
            gw = int(r.gameweek)
            done.add((str(r.home), gw)); done.add((str(r.away), gw))
        played_gws = {g for (_, g) in done}
        nxt = [g for g in range(1, 39) if g not in played_gws]
        return min(nxt) if nxt else 38
    except Exception:
        return 1


PRED_XI_GW = int(os.environ.get("PRED_XI_GW") or _default_pred_xi_gw())
pl_by_gw = {}
if _flag("PRED_XI", "on"):
    try:
        import predicted_xi as pxi
        _sq = d26[["web_name", "team", "player_code", "first_name", "second_name"]]
        # Which previews exist for THIS gameweek. An empty list is the normal state for
        # a week whose deadline is days away — previews land the day before — and it must
        # skip the layer rather than reach for another week's teamsheet.
        _srcs = pxi.sources(PRED_XI_GW, verbose=True)
        if not _srcs:
            raise FileNotFoundError(
                f"no predicted-XI files for GW{PRED_XI_GW} in {config.DATA} "
                f"(expected predicted_xi_gw{PRED_XI_GW}[_source].csv)")
        # Multi-source consensus: confidence scales with how many independent previews
        # name a player, so a contested pick (Saka, Guéhi) is treated as contested
        # rather than deleted on one outlet's opinion. Falls back to the single-source
        # path if only one source file is present.
        _cons = pxi.consensus(_sq, gw=PRED_XI_GW)
        if len(_cons):
            _plp, _rep = pxi.apply_consensus(pl, _cons, _sq)
        else:
            _res = pxi.resolve(pxi.load(path=_srcs[0]["path"]), _sq)
            _plp, _rep = pxi.apply_soft(
                pl, _res,
                start_confidence=float(os.environ.get("PRED_XI_START_CONF", "0.75")),
                omit_confidence=float(os.environ.get("PRED_XI_OMIT_CONF", "0.35")))
        # Availability OUTRANKS a predicted XI. Applied last so a lineup source cannot
        # promote a player above what the injury feed permits (Doku: named by two of
        # three XIs, 25% chance of playing, ruled out in the manager's pre-match
        # remarks). `sig` here is the live FPL feed where it was reachable.
        _plp, _ = pxi.apply_injury_ceiling(_plp, sig)
        # Minutes management is a SEPARATE lever — a capped starter still starts.
        _plp, _ = pxi.apply_minutes_caps(_plp, gw=PRED_XI_GW)
        pl_by_gw[PRED_XI_GW] = _plp
        print(f"[pred-xi] applied to GW{PRED_XI_GW} only; "
              f"GW{PRED_XI_GW + 1}+ use the untouched priors")
        if len(_rep):
            _rep.to_csv(os.path.join(config.OUTPUTS, "predicted_xi_moves.csv"),
                        index=False)
    except Exception as e:
        print(f"[pred-xi] skipped ({type(e).__name__}: {e})")
else:
    print("[pred-xi] off")
# Fill set-piece duty where FPL declares none. FPL's flag is the more precise signal and
# always wins; this only covers the clubs it leaves blank (studies/penalty_assignment.py).
#
# Default reverted from "observed" to "fpl" on 2026-09-08. `studies/penalty_assignment.py`
# is the registered result on exactly this question and points the other way: the declared
# `penalties_order` BEATS measured history at 85.7% precision. The observed channel let a
# SINGLE realised penalty overturn a declaration — no minimum n, no shrinkage toward the
# prior, ties broken by `.iloc[0]`. On live data all six clubs that had taken a penalty
# were at n=1 and two declarations were already overturned (Coventry Wright->Torp,
# Sunderland Diarra->Le Fee), moving roughly 15 projected points per club to the wrong
# player in an arbitrary direction. Against an 85.7% prior one observation does not move
# the posterior across 50%; that estimator moved it all the way.
#
# `observed` is still reachable explicitly. To make it the default it needs to come back
# as a Beta-Binomial update on the declared taker, prior mass set from the measured 85.7%,
# with the n at which it may flip registered BEFORE the takers are looked at.
_spt = os.environ.get("FPL_SETPIECE", "fpl").lower()
_FLAGS.append(("FPL_SETPIECE", "ON " if _spt not in ("off", "0") else "off", _spt))
_FLAGS.append(("PRED_XI_GW", "ON ", str(PRED_XI_GW)))
import set_piece_takers as spt
if _spt not in ("off", "0", "fpl", "observed"):
    sig = spt.apply_to_signals(sig, d26[["web_name", "team", "player_code"]],
                               mode="fill" if _spt == "fill" else "override")
# Keyed on player_code, NOT web_name: 15 names are shared across two clubs and a
# name-keyed lookup gave Ipswich a second penalty taker (see set_piece_takers.pen1_codes).
# The SAME window the inseason channel uses. Until 2026-09-08 only `inseason` honoured
# it: `press_index.press_factor` and `set_piece_takers.observed_takers` both defaulted to
# "every completed gameweek at call time", so a board rebuilt for a past week was
# conditioned on that week's own results through two club-level, persistent channels.
# That is exactly how predictions/gw2_* and gw3_* were produced — a pre-deadline PLAYER
# pull, rebuilt days after the gameweek finished — so it inflates precisely the
# reconstruction rows now in the scoring ledger. 38 (the default) means "all", which is
# right for a forward board and leaves live behaviour unchanged.
_UPTO = int(os.environ.get("INSEASON_UPTO", "38"))
# `None` means "every completed gameweek" to both channels, and 38 means the same thing —
# but NOT to press_measured, which treats ANY non-None gw as an explicit cut and falls
# back from PitchAPI to the proxy feed because PitchAPI cannot honour one. Passing the
# default 38 therefore switched the press feed on an ordinary forward board, which this
# change must not do: bounding the window is only meant to bite when a window was
# actually asked for. Collapse the no-cut case back to None.
_UPTO_CUT = None if _UPTO >= 38 else _UPTO

_pen_src = d26[["web_name", "team", "player_code"]].copy()
if "penalties_order" in d26.columns:
    _pen_src["penalties_order"] = d26["penalties_order"]
pen1 = spt.pen1_codes(_pen_src, mode=_spt, upto_gw=_UPTO_CUT)
pl["pen_xg90"] = np.where(pl.player_code.isin(pen1),
                          pl.pen_xg90.fillna(0).clip(lower=0.10), 0.0)
print(f"[set-piece] penalty xG floor applied to {int(pl.player_code.isin(pen1).sum())} "
      f"players across {len(pen1)} resolved codes")

# ---------- team model with betting-odds strength ----------
elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
if _flag("MARKET_ODDS", None):
    import market_odds as mo
    elo = mo.blended_elo(elo, market_weight=float(os.environ.get("MARKET_WEIGHT", "0.6")))
    print(f"[market-odds] betting-odds team strength blended (weight {os.environ.get('MARKET_WEIGHT','0.6')})")
_elo_w = float(os.environ.get("ELO_WEIGHT", "0.45"))
_FLAGS.append(("ELO_WEIGHT", "ON ", str(_elo_w)))
tm = TeamModel(promoted_per_club=pclub).fit(e0_path=_e0_path, clubelo=elo, clubelo_weight=_elo_w)
bayes_model.rng = np.random.default_rng(7); ts = tm.sample_2627(S=S)

# An unavailable starter weakens his TEAM, not just his own row. Applied to the sampled
# att/def so the whole side's clean-sheet and scoring expectations move — otherwise
# Saliba being out leaves every other Arsenal defender at full clean-sheet value.
# [JUDGMENT] betas, hard-capped. Default flipped to OFF on 2026-09-08, which is what
# src/injury_impact.py's own docstring already claimed ("the whole layer is OFF unless
# asked for") while the board switched it on.
#
# Its orthogonality argument answers the `market_odds` leg only: the odds snapshot is
# 7 Aug and the injuries post-date it. It does not answer INSEASON, which is ON by
# default and stacks realised 26/27 match xG into E0 below. Those matches were played
# WITH the absentees out, so the fitted team strength already carries the loss and this
# layer subtracts it a second time. The double count is not static — it grows with every
# gameweek stacked, so the error was smallest the week the argument was written and has
# been widening since. INJURY_IMPACT=on restores it.
if _flag("INJURY_IMPACT", "off"):
    import injury_impact as ii
    _avail = {}
    if "player_code" in sig.columns:
        for _r in sig.dropna(subset=["player_code"]).itertuples():
            _c = 0.0 if str(getattr(_r, "status", "a")) in ("i", "s", "u", "n") \
                else float(getattr(_r, "chance_play", 1.0) or 0.0)
            _avail[_r.player_code] = _c
    _losses = ii.team_losses(pl, availability=_avail)
    ts, _moved = ii.apply_to_samples(ts, _losses)
    _losses.to_csv(os.path.join(config.OUTPUTS, "team_injury_losses.csv"), index=False)
else:
    print("[injury-impact] off")

# Manager judgment on top of the fitted layers, applied openly and last so it is never
# confused with a measured effect. TEAM_OVERRIDES=off disables.
#
# This block used to sit INSIDE the injury-impact branch, so INJURY_IMPACT=off silently
# dropped the manager overrides as well and said nothing about it. That is exactly the
# wrong behaviour for an A/B flag: a run isolating the injury layer was also, unknowably,
# a run without overrides, and the measured difference was the sum of two changes.
# Default flipped to OFF on 2026-09-08. The committed shift is a pure LOCATION shift on
# a club's attack and defence with no widening, which asserts direction with full
# confidence — the opposite of the kappa encoding CLAUDE.md's guard requires, and the
# same delta mean-pull that `starter_prior.resolve_regime()` exists to keep switched off.
# It also never clears `style_matchup.beats_the_market()`, and it is applied AFTER the
# market-Elo blend, so it re-prices an appointment the odds already carry.
#
# To switch it back on legitimately: express the disagreement as a kappa widening on the
# regime club rather than a delta on the mean, or clear `beats_the_market()` against a
# CURRENT odds snapshot — not the 7 Aug one. TEAM_OVERRIDES=on restores it meanwhile.
if _flag("TEAM_OVERRIDES", "off"):
    import team_overrides as tov
    ts, _ov = tov.apply_to_samples(ts)
else:
    print("[team-overrides] off")

# DefCon environment (xGA over the full window + press index)
idx = ts["idx"]; mu = ts["mu"]; home = ts["home"]; A = ts["att"]; D = ts["dfn"]
sched, long = schedule(); win = long[long.gameweek <= GW_HI]
xga27 = {}
for t, g in win.groupby("team"):
    if t not in idx: continue
    xga27[t] = float(np.mean([np.exp(mu + (0.0 if r.is_home else home) + A[:, idx[r.opp]] - D[:, idx[t]]).mean()
                              for _, r in g.iterrows() if r.opp in idx]))
# The press leg of the CBIRT channel revises itself against 26/27 results as the season
# accrues (src/press_measured.py). Resolved here purely so the banner reports it: the
# switch is read inside press_index, and both read the same env var with the same default.
if _flag("PRESS_MEASURED", "on"):
    import press_index as _pix
    _tbl, _n = _pix._measured(_UPTO_CUT)
    if _n:
        _ws = [_pix.resolve_ppda(c, "2627", _UPTO_CUT)[1] for c in _pix.PPDA_2627]
        print(f"[press] measured PPDA for {len(_n)} clubs over "
              f"{max(_n.values())} match(es); blend weight "
              f"{min(_ws):.2f}-{max(_ws):.2f} on the judgment prior")
    else:
        print("[press] no 26/27 matches measured yet — judgment PPDA table stands")
else:
    print("[press] measured PPDA off — judgment table pinned")

if _flag("DEFCON_ENV", None):
    pl = de.apply_defcon_environment(pl, xga27, config.REPO, gw=_UPTO_CUT)

# resolved here rather than at its gate, which sits after the projection loop —
# a banner that omits a flag is worse than no banner.
_SOLIO = _flag("SOLIO", None)
_banner()

# ---------- true per-GW projections ----------
frames = []
# DUMP_DRAWS=<gw> persists the raw posterior draw matrix for one gameweek to
# .cache/draws_gw<N>.npz. Off by default: nothing on the normal board path needs it and
# the matrix is large. It exists because questions about a SUM of players — a squad
# total, a bench boost, a captaincy tail — cannot be answered from the summary columns,
# since percentiles do not add.
# Default to the gameweek being projected, not nothing. The tail metrics the explorer
# shows — P(haul), ceiling, regret against the template captain — can only be computed
# from the raw draws, because percentiles do not add and a JOINT event like "mine blanks
# while the template hauls" cannot be recovered from summary columns at all. Leaving this
# off by default meant `src/captaincy.py` sat unused all season and captaincy was ranked
# on the mean. One gameweek is ~1MB in SCRATCH, which is a cheap price for the only
# question captaincy actually asks.
_DUMP = (os.environ.get("DUMP_DRAWS") or str(PRED_XI_GW)).strip().lower()
if _DUMP in ("all", "*"):
    _dump_gws = set(range(1, GW_HI + 1))
else:
    _dump_gws = {int(g) for g in _DUMP.split(",") if g.strip().isdigit()}
for gw in range(1, GW_HI + 1):
    bayes_model.rng = np.random.default_rng(7)
    if gw in _dump_gws:
        g, _draws = project(pl_by_gw.get(gw, pl), tm, ts, gw, gw, S=S, return_draws=True)
        _keep = ~g["id"].duplicated()
        g = g[_keep]; _draws = _draws[_keep.to_numpy()]
        _p = os.path.join(config.SCRATCH, f"draws_gw{gw}.npz")
        # carry player_code so a consumer never has to join these back on a name
        _cmap = pl.dropna(subset=["id"]).drop_duplicates("id").set_index("id")["player_code"]
        np.savez_compressed(_p, draws=_draws, ids=g["id"].to_numpy(),
                            player_code=g["id"].map(_cmap).to_numpy(dtype=float),
                            player=g["player"].to_numpy(), team=g["team"].to_numpy())
        print(f"[draws] wrote {_draws.shape[0]} players x {_draws.shape[1]} draws -> {_p}")
    else:
        g = project(pl_by_gw.get(gw, pl), tm, ts, gw, gw, S=S).drop_duplicates("id")
    g["gw"] = gw; frames.append(g)
    print(f"[gw {gw}] projected {len(g)} players (top: {g.iloc[0].player} {g.iloc[0]['mean']:.2f})")
long_df = pd.concat(frames, ignore_index=True)
# id -> player_code. project() carries the per-player `id` from the frame it was given;
# player_code is the stable key the rest of the project joins on and the one a scoring
# pass needs. The map has to come from `pl`, NOT from d26: cold-start players are issued
# a synthetic id by roster._COLDSTART_ID which appears nowhere in d26.player_id, so
# mapping through d26 silently dropped the code for all 228 of them.
_codes = pl[["id", "player_code"]].dropna(subset=["id"]).drop_duplicates("id")
long_df = long_df.merge(_codes, on="id", how="left")
_miss = int(long_df["player_code"].isna().sum())
if _miss:
    print(f"[board] WARNING {_miss} rows have no player_code — scoring will need a name join")

# ---------- Solio blend (GW1, offline from the cached feed) ----------
long_df["solio"] = np.nan; long_df["blended"] = long_df["mean"]; long_df["src"] = "model"
if _SOLIO:
    try:
        sol = se.parse_solio(se.fetch_solio())
        Sp = sol["players"]; sgw = sol.get("gameweek")
        if sgw is not None and 1 <= sgw <= GW_HI:
            g1 = long_df[long_df.gw == sgw]
            al = se.align_single_gw(g1.rename(columns={"mean": "mean"}), Sp)
            key2solio = al.set_index(["player", "team"])["solio_proj"].to_dict()
            mask = long_df.gw == sgw
            for i in long_df[mask].index:
                k = (long_df.at[i, "player"], long_df.at[i, "team"])
                if k in key2solio and pd.notna(key2solio[k]):
                    long_df.at[i, "solio"] = key2solio[k]
                    long_df.at[i, "blended"] = W_OURS * long_df.at[i, "mean"] + (1 - W_OURS) * key2solio[k]
                    long_df.at[i, "src"] = f"blend(w_ours={W_OURS})"
            n = long_df.loc[mask, "solio"].notna().sum()
            print(f"[solio] GW{sgw}: blended {n} matched players from the cached feed "
                  f"(w_ours={W_OURS}); other GWs are model-only")
        else:
            print(f"[solio] cached feed is GW{sgw} — outside window, no blend")
    except Exception as e:
        print(f"[solio] unavailable ({e}); model-only")

# ---------- outputs ----------
# `player_code` is on the board so it can be SCORED without a name join. The first
# scoring pass (docs/GW1_SCORING_2026-08-22.md) had to match the locked prediction to
# actuals on (web_name, team), which is the join this project forbids everywhere else —
# tolerable only because the club disambiguates within one season, and not something to
# leave in the path that validates the model.
cols = ["player_code", "player", "pos", "team", "cost", "own", "gw", "mean", "solio",
        "blended", "src", "sd", "app_ev", "att_ev", "cs_ev", "defcon_ev", "conc_ev",
        "p5", "median", "p95"]
long_out = long_df[[c for c in cols if c in long_df.columns]].round(3)
long_out.to_csv(os.path.join(config.OUTPUTS, "gw_board_long.csv"), index=False)

# A DATED SNAPSHOT, so the explorer can show what changed since you last looked.
# One file per DAY, overwritten by same-day reruns on purpose: the board gets rebuilt
# several times in an afternoon and a per-run history would make "the delta" mean
# "since I last pressed go", which is noise. Overwriting today leaves yesterday's
# intact, so the baseline is always a genuinely earlier state of the model.
try:
    _snap = long_out[[c for c in ("player_code", "gw", "blended", "cost")
                      if c in long_out.columns]]
    _sp = os.path.join(config.BOARD_HISTORY,
                       f"board_{pd.Timestamp.now():%Y-%m-%d}.csv")
    _snap.to_csv(_sp, index=False)
    print(f"[snapshot] {len(_snap)} rows -> {os.path.basename(_sp)}")
except Exception as _e:                                            # noqa: BLE001
    print(f"[snapshot] skipped ({_e}); the delta column will be blank")

wide = long_df.pivot_table(index=["player", "pos", "team", "cost", "own"],
                           columns="gw", values="blended", aggfunc="first")
wide.columns = [f"gw{int(c)}" for c in wide.columns]
wide["total"] = wide.sum(axis=1)
wide = wide.reset_index().sort_values("total", ascending=False)
wide.round(2).to_csv(os.path.join(config.OUTPUTS, "gw_board_wide.csv"), index=False)

print("\n" + "=" * 100)
print(f"WEEK-BY-WEEK BOARD  GW1-{GW_HI}  (GW1 Solio-blended where matched; betting-odds team strength on)")
print("=" * 100)
show = ["player", "pos", "team", "cost"] + [f"gw{g}" for g in range(1, min(GW_HI, 8) + 1)] + ["total"]
print(wide.head(15)[show].round(2).to_string(index=False))
print(f"\nwrote gw_board_long.csv ({len(long_out)} rows) and gw_board_wide.csv ({len(wide)} players) to {config.OUTPUTS}")
