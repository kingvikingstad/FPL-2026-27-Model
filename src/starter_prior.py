from __future__ import annotations
import config
"""
starter_prior.py — depth-aware start probability for COLD-START players
=======================================================================
The projection's minutes gate is its dominant single-GW lever. For players
with real minutes history the Beta(start_a, start_b) prior is well-identified.
For cold-start players (new signings, promoted-club squads, youth) the roster
builder falls back to a price-only regression, which badly over-rates fit-but-
fringe players at strong clubs (a £4.5m Man City centre-back inherits City's
clean-sheet EV because nothing says he won't start).

Ownership is the market's forecast of playing time and is the strongest
predictor of starts after minutes themselves (Spearman ~0.68 on 25/26; ~0.62
even among <=£5.5m players). It is NOT otherwise an input to the points model,
so using it to set the cold-start start prior is new information, not double
counting. This is a STATIC depth prior, not a week-to-week rotation multiplier
(rotation tested null — deliberately not modelled).

Applied to cold_start rows only. Established players keep their minutes-derived
Beta; apply_availability still overrides everyone on injuries and confirmed XIs.
"""
import numpy as np, pandas as pd, pickle, os

POS = ("GK", "DEF", "MID", "FWD")
CAL_PATH = config.OWN_START_CAL


def calibrate_ownership_start(panel_path=config.PMS_PANEL,
                              stats_csv=None,
                              team_games=38, min_rows=1):
    """Fit logit P(start per GW) ~ 1 + log(ownership) + price, per position,
    from 25/26. Start = played >=60 min in a match. Returns {pos: (b0,b1,b2)}."""
    stats_csv = stats_csv or os.path.join(config.repo("2025-2026"), "playerstats.csv")
    panel = pd.read_pickle(panel_path)
    panel["is_start"] = (panel.mins >= 60).astype(float)
    gp = panel.groupby("player_id").agg(
        starts=("is_start", "sum"), rows=("is_start", "size"),
        pos=("pos", "last")).reset_index()
    last = pd.read_csv(stats_csv).sort_values("gw").groupby("id").last()
    gp["own"] = gp.player_id.map(last["selected_by_percent"])
    gp["price"] = gp.player_id.map(last["now_cost"])
    g = gp.dropna(subset=["own", "price"])
    g = g[g.rows >= min_rows]
    g["sr"] = (g.starts / team_games).clip(0.01, 0.99)
    g["logit"] = np.log(g.sr / (1 - g.sr))
    g["logown"] = np.log(g.own.clip(0.05, 95))
    cal = {}
    for p in POS:
        s = g[g.pos == p]
        if len(s) < 8:                          # fall back to pooled fit
            s = g
        X = np.column_stack([np.ones(len(s)), s.logown, s.price])
        b, *_ = np.linalg.lstsq(X, s.logit.values, rcond=None)
        cal[p] = b
    pickle.dump(cal, open(CAL_PATH, "wb"))
    return cal


def load_calibration():
    if os.path.exists(CAL_PATH):
        return pickle.load(open(CAL_PATH, "rb"))
    return calibrate_ownership_start()


def coldstart_start_prob(own, price, pos, cal):
    """Ownership+price -> P(start per GW), clipped to a sane [0.03, 0.97]."""
    b = cal.get(pos, cal.get("MID"))
    lo = b[0] + b[1] * np.log(max(float(own), 0.05)) + b[2] * float(price)
    return float(np.clip(1.0 / (1.0 + np.exp(-lo)), 0.03, 0.97))


def apply_coldstart_depth(players: pd.DataFrame, cal=None, kstart=6.0,
                          blend_price=0.35):
    """Reset start_a/start_b for cold_start rows from the ownership-aware prob.

    blend_price: weight on the ORIGINAL price-derived start prob (retains some
    of the roster builder's signal); the rest goes to the ownership model.
    Established rows (cold_start False/absent) are untouched.
    """
    if cal is None:
        cal = load_calibration()
    p = players.copy()
    if "cold_start" not in p.columns:
        return p
    for i, r in p.iterrows():
        if not bool(r.get("cold_start", False)):
            continue
        prior = r.start_a / (r.start_a + r.start_b) if (r.start_a + r.start_b) > 0 else 0.5
        own_p = coldstart_start_prob(r.get("own", 0.0), r.get("cost", 4.5), r.pos, cal)
        new_p = blend_price * prior + (1 - blend_price) * own_p
        p.at[i, "start_a"] = float(np.clip(new_p, 1e-3, 1) * kstart)
        p.at[i, "start_b"] = float(np.clip(1 - new_p, 1e-3, 1) * kstart)
    return p


def apply_minutes_shrinkage(players: pd.DataFrame, cal=None, k_min=900.0):
    """Evidence-weighted start prior for ALL players with a minutes column.

    Blends the historical start probability with the ownership-implied prior,
    weighting history by its precision: w_hist = minutes / (minutes + k_min).
    Rich-history players (Gabriel, 3900 min) are essentially untouched; thin-
    history players whose role has changed (Mosquera, 989 min, promoted into the
    XI by an injury ahead of him) are pulled toward the market's current view.
    This is the Bayesian-correct generalisation of the cold-start depth prior:
    minutes=0 -> w_hist=0 -> fully ownership-implied (identical to that prior).

    Preserves each Beta's concentration (total pseudo-count), shifting only the
    mean. Injuries/confirmed XIs still override afterward via apply_availability.
    """
    if cal is None:
        cal = load_calibration()
    p = players.copy()
    if "minutes" not in p.columns:
        return p
    for i, r in p.iterrows():
        if bool(r.get("cold_start", False)):
            continue                                  # handled by apply_coldstart_depth
        N = r.start_a + r.start_b
        if N <= 0:
            continue
        p_hist = r.start_a / N
        p_own = coldstart_start_prob(r.get("own", 0.0), r.get("cost", 4.5), r.pos, cal)
        w = float(r.get("minutes", 0.0)) / (float(r.get("minutes", 0.0)) + k_min)
        p_new = w * p_hist + (1 - w) * p_own
        p.at[i, "start_a"] = float(np.clip(p_new, 1e-3, 1) * N)
        p.at[i, "start_b"] = float(np.clip(1 - p_new, 1e-3, 1) * N)
    return p


def apply_regime_uncertainty(players: pd.DataFrame, regime=None, cal=None,
                             k_min=900.0, kstart=6.0):
    """Superset of apply_minutes_shrinkage with a regime *variance* knob.

    Two independent, per-club knobs (both no-op by default):
      delta (in (0,1], no-op 1.0): scales effective minutes -> pulls the MEAN start
        probability toward the ownership-implied prior (directional claim).
      kappa (in [0,1], no-op 0.0): weight on the between-component variance term ->
        WIDENS the posterior (ignorance claim, no direction).

    regime: {club: kappa} or {club: (delta, kappa)}. Absent/empty -> reproduces
    apply_minutes_shrinkage exactly for every player (bitwise baseline identity).

    Variance mechanics (moment-matched two-component mixture):
        v = w*v_h + (1-w)*v_o + kappa * w(1-w) * (p_h - p_o)^2
    Maximised at w=0.5; scales with squared history-vs-market disagreement. N_eff is
    moment-matched, widen-only (never sharpens below baseline), Uniform-floored;
    clamp events counted.

    Horizon: project() draws p ~ Beta once per path and holds it across the horizon,
    so this widening vanishes at H=1 and grows ~H(H-1) thereafter.
    """
    if cal is None:
        cal = load_calibration()
    reg = {}
    for club, v in (regime or {}).items():
        reg[club] = (float(v[0]), float(v[1])) if isinstance(v, (tuple, list)) else (1.0, float(v))
    p = players.copy()
    if "minutes" not in p.columns:
        return p
    clamps = 0
    for i, r in p.iterrows():
        if bool(r.get("cold_start", False)):
            continue
        N_h = r.start_a + r.start_b
        if N_h <= 0:
            continue
        delta, kappa = reg.get(r.team, (1.0, 0.0))
        p_h = r.start_a / N_h
        p_o = coldstart_start_prob(r.get("own", 0.0), r.get("cost", 4.5), r.pos, cal)
        eff_min = float(r.get("minutes", 0.0)) * delta
        w = eff_min / (eff_min + k_min)
        p_bar = w * p_h + (1 - w) * p_o
        if kappa <= 0.0:
            N_eff = N_h                                    # no-op: baseline identity
        else:
            v_h = p_h * (1 - p_h) / (N_h + 1)
            v_o = p_o * (1 - p_o) / (kstart + 1)
            v = w * v_h + (1 - w) * v_o + kappa * w * (1 - w) * (p_h - p_o) ** 2
            vmax = p_bar * (1 - p_bar)
            if v >= vmax:
                N_eff = 1.0; clamps += 1
            else:
                N_eff = max(min(N_h, vmax / v - 1.0), 1.0)  # widen-only
        p.at[i, "start_a"] = float(np.clip(p_bar, 1e-3, 1) * N_eff)
        p.at[i, "start_b"] = float(np.clip(1 - p_bar, 1e-3, 1) * N_eff)
    if clamps:
        print(f"[regime] Uniform clamp on {clamps} rows (kappa may be too high)")
    return p


# Proposed regime settings (Aug 2026) — UNFITTED [JUDGMENT], OFF BY DEFAULT.
# Starting points for a sweep, NOT active values. Pass explicitly to opt in.
REGIME_2627_PROPOSED = {
    "Newcastle": (0.35, 0.8), "Liverpool": (0.50, 0.8), "Chelsea": (0.50, 0.7),
    "Nott'm Forest": (0.50, 0.7), "Crystal Palace": (0.50, 0.7),
    "Bournemouth": (0.50, 0.7), "Fulham": (0.50, 0.7), "Ipswich": (0.55, 0.6),
    "Man City": (0.55, 0.6), "Tottenham": (0.45, 0.7), "Man United": (0.80, 0.4),
}


def resolve_regime(mode=None):
    """Turn the REGIME env flag into a regime dict, so every runner reads it identically.

    Three modes, and the distinction between the last two is the whole point:

      off       (default) -> {}  : bitwise baseline identity.
      kappa                      : VARIANCE ONLY. Keeps each club's kappa, forces
                                   delta to its no-op 1.0. This is the project's
                                   stated position — "regime change is an ignorance
                                   statement, it belongs in variance, not directional
                                   style priors" (PROJECT_KNOWLEDGE §3). Point
                                   estimates are untouched; only the posterior widens.
      proposed                   : delta AND kappa. The deltas are UNFITTED [JUDGMENT]
                                   (§7: "wired but off. Turning them on shifts point
                                   estimates on assertion. Sweep against live data
                                   first."). Use for sensitivity analysis, not as the
                                   headline board.

    Returns (regime_dict, label).
    """
    mode = (mode if mode is not None else os.environ.get("REGIME", "off")).lower()
    if mode == "proposed":
        return dict(REGIME_2627_PROPOSED), "proposed (delta+kappa; deltas UNFITTED [JUDGMENT])"
    if mode == "kappa":
        # tuple -> kappa only; a bare scalar is already a kappa (see apply_regime_uncertainty)
        reg = {c: (float(v[1]) if isinstance(v, (tuple, list)) else float(v))
               for c, v in REGIME_2627_PROPOSED.items()}
        return reg, "kappa-only (variance widening; point estimates unchanged)"
    return {}, "off (baseline identity)"

# ---------------------------------------------------------------------------
XI_SIZE = 11.0


def _logit(x, eps=1e-6):
    x = np.clip(np.asarray(x, float), eps, 1 - eps)
    return np.log(x / (1 - x))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def apply_xi_constraint(players, target=XI_SIZE, group="team", verbose=True,
                        max_iter=80, tol=1e-9, hold=None):
    """Make each club's start probabilities sum to ELEVEN.

    THE DEFECT THIS FIXES  [VERIFIED 2026-08-24 against GW1]
    ---------------------------------------------------------
    Start priors are built per player and never see each other, so nothing enforces the
    hardest constraint in the sport: a club starts exactly eleven players. Measured on the
    GW1 locked board, EVERY club exceeded it and the league expected 295.1 starters
    against a structural 220 — Chelsea 17.68, Tottenham 16.61, Man United 15.94.

    It shows up as an unconditional over-projection. On GW1, model bias was +0.46 points
    per player overall but only -0.05 among players who actually appeared: essentially
    ALL of the error sat on players the model half-expected to play who never got on the
    pitch. Reliability was fine at the extremes (predicted 0.94 -> 0.92 realised; 0.03 ->
    0.00) and badly over-confident in the middle (0.33 -> 0.04, 0.66 -> 0.37) — the exact
    signature of unnormalised probabilities, where the surplus lands on the fringe.

    THE CORRECTION
    --------------
    A per-club shift in LOG-ODDS, solved so the club's probabilities sum to `target`:

        p_i  ->  sigmoid(logit(p_i) + c_club)

    Chosen over multiplicative rescaling for two reasons: it cannot push a probability
    outside (0, 1), and it preserves the ordering AND the relative odds of every pair of
    players at the club. It is the minimal exponential tilt that satisfies the constraint,
    so it removes the surplus without inventing new information about who loses out.

    Because the shift is negative (every club is over), nailed starters barely move — a
    player at 0.97 has a large log-odds and absorbs a small shift almost invisibly —
    while fringe players at 0.3 take most of the correction. That is the right incidence:
    the extremes were already calibrated and the middle was not.

    ORDER  [CORRECTED 2026-09-08]
    -----------------------------
    Applied AFTER availability, not before it. Running it first normalised each club to
    eleven and then let availability zero players out of that eleven, so the mass assigned
    to a player who was subsequently ruled out simply EVAPORATED instead of moving to the
    team-mate who will actually start. Measured on the live board: league Sigma p_start
    324.7 -> 220.0 under the constraint -> 184.9 after availability, with nineteen of
    twenty clubs below 10.5 expected starters and Aston Villa at 8.06. That is a ~16%
    structural under-count of appearance, attacking, DefCon and clean-sheet exposure,
    biting hardest at the clubs with the most team news — precisely where the model is
    supposed to be sharpest.

    `hold` keeps the injury precedence that the old ordering bought accidentally. Pass a
    boolean mask (typically `players["avail_ruled_out"]`) of players availability has
    ruled out: they keep their probability exactly, their mass is subtracted from the
    club's target, and the tilt is solved over the remaining players only. Without it the
    upward tilt would lift a ruled-out player off 0.01 — sigmoid(logit(0.01) + 1.0) is
    0.027 — which is the injury flag losing to the constraint, the wrong way round.

    Clubs are solved independently by bisection.
    """
    p = players.copy()
    if "start_a" not in p.columns or group not in p.columns:
        return p, pd.DataFrame()
    a = p["start_a"].astype(float)
    b = p["start_b"].astype(float)
    strength = a + b
    p0 = a / strength
    if hold is None:
        held_mask = pd.Series(False, index=p.index)
    else:
        held_mask = pd.Series(hold, index=p.index).fillna(False).astype(bool)
    rows = []
    for club, idx_all in p.groupby(group).groups.items():
        idx_all = list(idx_all)
        tot = float(np.sum(p0.loc[idx_all].values))
        # Held players keep their probability and their mass comes off the target; the
        # tilt is solved over the free players only.
        held = [i for i in idx_all if held_mask.loc[i]]
        idx = [i for i in idx_all if not held_mask.loc[i]]
        held_mass = float(np.sum(p0.loc[held].values)) if held else 0.0
        target_free = target - held_mass
        q = p0.loc[idx].values if idx else np.array([])
        # TWO-SIDED, because Sigma p = 11 is an equality, not a ceiling. A club whose
        # probabilities sum to 10 is also wrong: someone has to be the eleventh, and the
        # tilt spreads that mass in proportion to the odds already assigned, which is the
        # maximum-entropy answer given no further information. NB every club in the live
        # 26/27 board is OVER, so only the downward direction is exercised on real data;
        # the upward one is correct by the same argument but is not yet validated.
        # A club with fewer than eleven listed players cannot be solved and is skipped.
        # A club with too few FREE players to reach the residual target cannot be solved
        # (the tilt is bounded by the number of players it can move), and one already at
        # target needs no shift.
        if len(idx) < int(np.ceil(target_free)) or target_free <= 0                 or abs(tot - target) < 1e-9:
            rows.append({"team": club, "before": tot, "after": tot, "shift": 0.0,
                         "held": len(held)})
            continue
        z = _logit(q)
        lo, hi = -30.0, 30.0
        c = 0.0
        for _ in range(max_iter):
            c = 0.5 * (lo + hi)
            s_ = float(np.sum(_sigmoid(z + c)))
            if abs(s_ - target_free) < tol:
                break
            if s_ > target_free:
                hi = c
            else:
                lo = c
        newp = _sigmoid(z + c)
        # hold the PRIOR STRENGTH fixed: this changes the mean, not the confidence.
        # Re-deriving a+b from scratch would silently discard how much evidence each
        # player's prior was built on, which is the thing the shrinkage layer exists to
        # get right.
        p.loc[idx, "start_a"] = newp * strength.loc[idx].values
        p.loc[idx, "start_b"] = (1.0 - newp) * strength.loc[idx].values
        rows.append({"team": club, "before": tot,
                     "after": float(np.sum(newp)) + held_mass,
                     "shift": float(c), "held": len(held)})
    rep = pd.DataFrame(rows)
    if verbose and len(rep):
        over = rep[rep.before > target + 1e-9]
        under = rep[rep.before < target - 1e-9]
        print(f"[xi-constraint] {len(over)} clubs over {target:.0f} expected starters, "
              f"{len(under)} under; league {rep.before.sum():.1f} -> {rep.after.sum():.1f}")
        for _, r in over.nlargest(4, "before").iterrows():
            print(f"    {r['team']:16s} {r['before']:5.2f} -> {r['after']:5.2f}  "
                  f"(log-odds shift {r['shift']:+.2f})")
    return p, rep


# ---------------------------------------------------------------------------
def selftest():
    """Offline, synthetic. Guards `apply_xi_constraint` — added 2026-09-08 when the
    constraint moved to AFTER availability and grew a `hold` mask. The module shipped
    without a selftest until then, so `test_all` discovered nothing here and the
    estimator that sets every club's expected starter count was uncovered."""
    def frame(ps, teams, strength=50.0, ruled=None):
        d = pd.DataFrame({"team": teams,
                          "start_a": [p * strength for p in ps],
                          "start_b": [(1 - p) * strength for p in ps]})
        if ruled is not None:
            d["avail_ruled_out"] = ruled
        return d

    # 1. an OVER club is pulled down to exactly eleven
    over = frame([0.9] * 15, ["A"] * 15)
    out, rep = apply_xi_constraint(over, verbose=False)
    got = (out["start_a"] / (out["start_a"] + out["start_b"])).sum()
    assert abs(got - 11.0) < 1e-6, f"over club not normalised: {got}"

    # 2. an UNDER club is pushed UP — Sigma p = 11 is an equality, not a ceiling
    under = frame([0.4] * 15, ["A"] * 15)
    out, _ = apply_xi_constraint(under, verbose=False)
    got = (out["start_a"] / (out["start_a"] + out["start_b"])).sum()
    assert abs(got - 11.0) < 1e-6, f"under club not normalised: {got}"

    # 3. relative odds are preserved among the players that move (the tilt is a shift
    #    in log-odds, so every pairwise odds RATIO is invariant)
    mixed = frame([0.9, 0.6, 0.3] * 5, ["A"] * 15)
    out, _ = apply_xi_constraint(mixed, verbose=False)
    q0 = mixed["start_a"] / (mixed["start_a"] + mixed["start_b"])
    q1 = out["start_a"] / (out["start_a"] + out["start_b"])
    r0 = (q0 / (1 - q0)).values
    r1 = (q1 / (1 - q1)).values
    ratio = (r1 / r0)
    assert np.allclose(ratio, ratio[0]), "log-odds shift is not uniform within the club"

    # 4. THE POINT OF THE hold MASK. A ruled-out player sits at 0.01, not 0. Without the
    #    mask an upward tilt lifts him off it — that is the injury flag losing to the
    #    constraint. With it he is untouched and the club still sums to eleven.
    ps = [0.01] * 4 + [0.5] * 11
    ruled = [True] * 4 + [False] * 11
    held = frame(ps, ["A"] * 15, ruled=ruled)
    out, rep = apply_xi_constraint(held, hold=held["avail_ruled_out"], verbose=False)
    q = out["start_a"] / (out["start_a"] + out["start_b"])
    assert np.allclose(q.values[:4], 0.01, atol=1e-9), \
        f"ruled-out players were tilted: {q.values[:4]}"
    assert abs(q.sum() - 11.0) < 1e-6, f"club with held players does not sum to 11: {q.sum()}"
    assert int(rep["held"].iloc[0]) == 4, rep

    # 5. and WITHOUT the mask the same board does resurrect them — the regression this
    #    test exists to catch, asserted in the failing direction so it cannot pass
    #    vacuously if `hold` is ever silently ignored.
    out_bad, _ = apply_xi_constraint(held, verbose=False)
    q_bad = out_bad["start_a"] / (out_bad["start_a"] + out_bad["start_b"])
    assert q_bad.values[0] > 0.01 + 1e-6, \
        "unheld ruled-out player did not move; the mask test proves nothing"

    # 6. clubs are independent — solving one must not touch another
    two = frame([0.9] * 15 + [0.2] * 15, ["A"] * 15 + ["B"] * 15)
    out, _ = apply_xi_constraint(two, verbose=False)
    q = out["start_a"] / (out["start_a"] + out["start_b"])
    assert abs(q[:15].sum() - 11.0) < 1e-6 and abs(q[15:].sum() - 11.0) < 1e-6

    # 7. a club with fewer listed players than the target is skipped, not forced
    tiny = frame([0.5] * 6, ["A"] * 6)
    out, _ = apply_xi_constraint(tiny, verbose=False)
    assert np.allclose(out["start_a"].values, tiny["start_a"].values), "tiny club forced"

    print("starter_prior selftest ok — XI constraint normalises both directions, "
          "preserves relative odds, holds ruled-out players fixed, and skips short squads")
    return 0


if __name__ == "__main__":
    import sys as _sys2
    _sys2.exit(selftest() if "--selftest" in _sys2.argv else 0)
