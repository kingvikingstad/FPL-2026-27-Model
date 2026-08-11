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
