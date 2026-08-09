from __future__ import annotations
import config
"""
setpiece.py — is set-piece xG a stable player trait, or noise? (spec §2.1-2.2)
==============================================================================
An AR(1) slope is attenuated by the reliability of its REGRESSOR:

    plim rho_hat = rho_true * reliability(x_t)

A season set-piece xG rate is a low-count average, so it is measured with error, so
its measured persistence understates its true persistence. Simulation (studies/
sd_design.py, [VERIFIED]) puts set-piece reliability near 0.49 against 0.93 for
open-play — meaning a measured set-piece rho is roughly HALF the truth while the
open-play one is nearly right. Shrinking both components at a common rate therefore
over-shrinks set-piece specialists.

Two functions, in order:
  split_half_reliability()  — how well is each component measured? (well-powered)
  component_ar1()           — persistence, measured and disattenuated, with a CI

CORRECTION TO THE SPEC [VERIFIED by inspection]
-----------------------------------------------
The integration spec motivates this as fixing "a global ~80% set-piece mean-reversion
haircut derived from a measured AR(1) of rho = 0.204". **No such haircut and no such
rho exist in this repo.** There is no set-piece/open-play decomposition at all: the
npxGI prior is a single pooled Gamma-Poisson (`multiseason_priors.to_priors`), and the
tested-null mean-reversion result (p=0.69, studies/edge_study.py) is a DIRECTIONAL
over/under-performance test — a different quantity entirely.

So the pre-committed rule cannot be "does rho_SP differ from 0.20". There is nothing
in the codebase that implies 0.20. What the code does imply is a single shrinkage
weight applied uniformly to every component:

    posterior_mean = w * own_rate + (1 - w) * position_prior
    w(n90) = revert * n90 / (k0 + revert * n90)          # revert=0.70, k0=3.0

For an established player (~30 nineties) that is w ~= 0.875. **That uniform 0.875 is
the real status quo**, and it is what a component-specific persistence has to be
tested against. `implied_pooled_persistence()` reads the live constants out of
`multiseason_priors`, so the rule tracks the code rather than a copied number.

DECISION RULE (pre-committed, re-derived — see docs/SOCCERDATA_FINDINGS.md)
--------------------------------------------------------------------------
Let w0 = implied_pooled_persistence(median established n90).
  * If the 95% CI on rho_SP_disattenuated EXCLUDES w0 -> the uniform treatment is
    wrong for set pieces. Build the two-component prior (§2.3) with component-specific
    shrinkage, then A/B it (§2.5).
  * If the CI COVERS w0 -> keep the pooled prior. Record the reliability estimate in
    docs/ as a [VERIFIED] bound and stop.
  * Either way, record it. Do NOT run the rho_OP vs rho_SP DIFFERENCE test yet: §6.C
    puts the MDE at 0.29-0.36 on one season transition, which is underpowered.
    Revisit when 23/24 provides a second transition.

Under G7 a rule that fails is written up as a null and the code path deleted, not
softened.
"""
import argparse, sys
import numpy as np, pandas as pd

# Understat `situation` -> component. Kept explicit: 'SetPiece' in Understat means
# dead-ball generally, 'DirectFreekick' is the direct shot, and the two are pooled
# here because separately each is too thin to fit.
COMPONENTS = {
    "open_play": ("OpenPlay",),
    "from_corner": ("FromCorner",),
    "set_piece": ("SetPiece", "DirectFreekick"),
    "penalty": ("Penalty",),
}
# what §2.3 actually splits on: open play vs all non-penalty dead-ball
TWO_COMPONENT = {"open_play": ("OpenPlay",),
                 "set_piece_all": ("FromCorner", "SetPiece", "DirectFreekick")}

MIN_MINUTES_SEASON = 900          # spec §2.2
MIN_MINUTES_HALF = 450            # both halves must carry real exposure


def implied_pooled_persistence(n90: float = 30.0) -> float:
    """The shrinkage weight the CURRENT pooled prior puts on a player's own rate.
    This is the status quo the component-specific estimate is tested against."""
    try:
        import multiseason_priors as mp
        import inspect
        d = inspect.signature(mp.to_priors).parameters
        revert = float(d["revert"].default)
        k0 = float(d["k0"].default)
    except Exception:                                        # noqa: BLE001
        revert, k0 = 0.70, 3.0
    return revert * n90 / (k0 + revert * n90)


# --------------------------------------------------------------- panel build
def component_panel(shots: pd.DataFrame, minutes: pd.DataFrame,
                    components: dict = None) -> pd.DataFrame:
    """[player_code, season, component, npxg, minutes, npxg_p90], one row per
    player-season-component.

    `shots`:   calibrated Understat shots keyed on player_code (see crosswalk.G5),
               with columns season, match_id, situation, xg
    `minutes`: [player_code, season, match_id, minutes]

    Penalties are excluded from every non-penalty component by construction, so the
    components sum to total xG without double-counting.

    The panel spans every (player-season) x component present in `minutes`, with
    npxg=0 where the player took no shots of that type. Building it from the shot
    table alone would silently drop those rows — which is selection on the outcome,
    and it biases persistence upward: a player is retained only in the seasons they
    happened to register a set-piece shot.
    """
    comps = components or COMPONENTS
    s = shots.copy()
    s["component"] = None
    for name, sits in comps.items():
        s.loc[s["situation"].isin(sits), "component"] = name
    s = s[s["component"].notna()]

    xg = (s.groupby(["player_code", "season", "component"], as_index=False)
          .agg(npxg=("xg", "sum"), n_shots=("xg", "size")))
    mins = (minutes.groupby(["player_code", "season"], as_index=False)["minutes"].sum())

    grid = mins.merge(pd.DataFrame({"component": list(comps)}), how="cross")
    out = grid.merge(xg, on=["player_code", "season", "component"], how="left")
    out[["npxg", "n_shots"]] = out[["npxg", "n_shots"]].fillna(0.0)
    out["npxg_p90"] = out["npxg"] / (out["minutes"] / 90.0).clip(lower=1e-9)
    return out


# ------------------------------------------------------------ 2.1 reliability
def split_half_reliability(shots: pd.DataFrame, minutes: pd.DataFrame,
                           n_splits: int = 200, seed: int = 0,
                           components: dict = None,
                           min_minutes_half: float = MIN_MINUTES_HALF) -> pd.DataFrame:
    """Per player-season, randomly split MATCHES (not shots) into halves, compute
    npxg_p90 within each component, correlate across players, Spearman-Brown up to
    full length. Returns reliability per component per season.

    Splitting on matches matters: shots within a match are correlated (a team that
    wins six corners takes six set-piece shots), so a shot-level split would
    understate the error and overstate reliability.
    """
    comps = components or COMPONENTS
    rng = np.random.default_rng(seed)
    s = shots.copy()
    s["component"] = None
    for name, sits in comps.items():
        s.loc[s["situation"].isin(sits), "component"] = name
    s = s[s["component"].notna()]

    mins = minutes.groupby(["player_code", "season", "match_id"],
                           as_index=False)["minutes"].sum()
    rows = []
    for (season, comp), sub in s.groupby(["season", "component"]):
        pm = (sub.groupby(["player_code", "match_id"], as_index=False)["xg"].sum()
              .merge(mins[mins["season"] == season], on=["player_code", "match_id"],
                     how="right").fillna({"xg": 0.0}))
        pm = pm[pm["minutes"] > 0]
        if pm["player_code"].nunique() < 20:
            continue
        rs = []
        for _ in range(n_splits):
            half = rng.integers(0, 2, len(pm))
            agg = (pm.assign(half=half)
                   .groupby(["player_code", "half"], as_index=False)
                   .agg(xg=("xg", "sum"), minutes=("minutes", "sum")))
            w = agg.pivot(index="player_code", columns="half",
                          values=["xg", "minutes"])
            if w.shape[1] < 4:
                continue
            m0, m1 = w[("minutes", 0)], w[("minutes", 1)]
            keep = (m0 >= min_minutes_half) & (m1 >= min_minutes_half)
            if keep.sum() < 20:
                continue
            a = w[("xg", 0)][keep] / (m0[keep] / 90.0)
            b = w[("xg", 1)][keep] / (m1[keep] / 90.0)
            r = _corr_log(a.values, b.values)
            if np.isfinite(r):
                rs.append(2 * r / (1 + r))               # Spearman-Brown
        if rs:
            rows.append({"season": season, "component": comp,
                         "reliability": float(np.mean(rs)),
                         "reliability_sd": float(np.std(rs, ddof=1)) if len(rs) > 1 else np.nan,
                         "n_players": int(pm["player_code"].nunique()),
                         "n_splits_used": len(rs)})
    return pd.DataFrame(rows)


def _floor_log(x):
    """Log of a non-negative rate with a principled floor. A zero-xG season is real
    information (the player took no set-piece shots), so it must not be dropped —
    dropping zeros selects on the outcome and biases persistence upward.

    The floor is half the 5th percentile of POSITIVE rates, not half the minimum.
    [VERIFIED 2026-08-09] the minimum is set by a single outlier — one player with heavy
    minutes and one grazing low-xG shot — and using it puts the zero lump ~8 log units
    below the typical rate, inflating var(log) to 2.06 where the real spread is ~0.6.
    That corrupts every variance-ratio built on it and overflows the Poisson draw in
    `_simulate_measured_rho`. A quantile floor is stable against that single point.
    """
    x = np.asarray(x, float)
    pos = x[x > 0]
    if len(pos) >= 20:
        floor = float(np.quantile(pos, 0.05)) / 2.0
    elif len(pos):
        floor = float(np.min(pos)) / 2.0
    else:
        floor = 1e-6
    return np.log(np.maximum(x, max(floor, 1e-12)))


def _corr_log(a, b):
    la, lb = _floor_log(a), _floor_log(b)
    if np.std(la) < 1e-12 or np.std(lb) < 1e-12:
        return np.nan
    return float(np.corrcoef(la, lb)[0, 1])


# ------------------------------------------------------------------ 2.2 AR(1)
def _ols_cluster(x, y, groups):
    """Slope of y on x with an intercept, cluster-robust SE. Players recur across
    season transitions, so the residuals are not independent within player."""
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    meat = np.zeros((2, 2))
    for g in np.unique(groups):
        m = groups == g
        Xg, ug = X[m], u[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    n = len(y)
    scale = (G / max(G - 1, 1)) * ((n - 1) / max(n - 2, 1))
    V = scale * (XtX_inv @ meat @ XtX_inv)
    return float(beta[1]), float(np.sqrt(max(V[1, 1], 0.0)))


def component_ar1(panel: pd.DataFrame, reliability: dict,
                  min_minutes: float = MIN_MINUTES_SEASON,
                  reliability_se: dict = None) -> pd.DataFrame:
    """AR(1) of season t+1 on season t per component, on log rates, players with
    >= `min_minutes` in BOTH seasons. Cluster SE by player_code.

    Reports rho_measured, reliability, rho_disattenuated = rho / reliability, and a
    delta-method SE. If `reliability_se` is supplied, its uncertainty propagates:

        var(rho/rel) ~= var(rho)/rel^2 + (rho/rel^2)^2 * var(rel)

    Ignoring the second term understates the CI, which is exactly the direction that
    would make a marginal result look decisive — so pass the SEs from
    `split_half_reliability` when you have them.
    """
    rows = []
    p = panel.copy()
    p["season_ord"] = p["season"].astype(str)
    seasons = sorted(p["season_ord"].unique())
    for comp, sub in p.groupby("component"):
        pairs = []
        for s0, s1 in zip(seasons, seasons[1:]):
            a = sub[(sub["season_ord"] == s0) & (sub["minutes"] >= min_minutes)]
            b = sub[(sub["season_ord"] == s1) & (sub["minutes"] >= min_minutes)]
            j = a.merge(b, on="player_code", suffixes=("_t", "_t1"))
            if len(j):
                pairs.append(j)
        if not pairs:
            continue
        j = pd.concat(pairs, ignore_index=True)
        if len(j) < 30:
            print(f"[setpiece] {comp}: only {len(j)} season pairs — skipping")
            continue
        x, y = _floor_log(j["npxg_p90_t"]), _floor_log(j["npxg_p90_t1"])
        rho, se = _ols_cluster(x, y, j["player_code"].values)

        rel = float(reliability.get(comp, np.nan))
        rel_se = float((reliability_se or {}).get(comp, 0.0))
        rho_d = rho / rel if rel and np.isfinite(rel) else np.nan
        var_d = (se ** 2) / rel ** 2 + (rho / rel ** 2) ** 2 * rel_se ** 2 \
            if rel and np.isfinite(rel) else np.nan
        se_d = float(np.sqrt(var_d)) if np.isfinite(var_d) else np.nan
        rows.append({"component": comp, "n_pairs": len(j),
                     "rho_measured": rho, "se_measured": se,
                     "reliability": rel,
                     "rho_disattenuated": rho_d, "se_disattenuated": se_d,
                     "ci_lo": rho_d - 1.96 * se_d if np.isfinite(se_d) else np.nan,
                     "ci_hi": rho_d + 1.96 * se_d if np.isfinite(se_d) else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------- 2.2b attenuation by simulation
# `rho_disattenuated = rho / reliability` is the spec's correction. It cannot be trusted
# on the thin components, but the reason is narrower than an earlier version of this
# comment claimed, and the correction of that claim is itself worth recording.
#
# [VERIFIED] 12-seed study at rho_true=0.55, reliability ~0.68, n=400, under the CURRENT
# `_floor_log`:
#
#     estimator            mean     bias     95% CI coverage
#     ratio  rho/rel       0.567   +0.017         100%
#     simulation inverse   0.566   +0.016         100%
#
# i.e. in a WELL-SPECIFIED regime the two are indistinguishable. An earlier run of the
# same study reported +0.037 for the ratio against +0.003 for the inversion, and that
# gap was largely an artefact of the old floor (half the MINIMUM positive rate), which
# let a single outlier stretch var(log) and break the variance-ratio identity. The
# quantile floor fixed most of it for BOTH estimators. Do not cite the old numbers.
#
# What survives, and it is the part that matters: on REAL data the ratio returns
# rho > 1 for `from_corner` (1.229) and `set_piece_all` (1.080) — impossible for a
# stationary process. The simulated regime does not reproduce that failure because it
# lacks the real panel's persistent structural zeros. So the case against the ratio
# rests on the real-data evidence, not the simulation.
#
# The inversion is kept for a different reason than "less biased": it FAILS VISIBLY.
# When the observed slope exceeds anything the count process can produce, it pins at the
# grid edge and sets `saturated`, and `decide()` refuses to rule. The ratio has no such
# signal — it just returns 1.229 and lets you build a prior on it.
def _simulate_measured_rho(rho_true, stats, n_sims, rng):
    """E[measured AR(1) slope] under a known rho_true, at the observed panel's
    count intensity. Same estimator as the real fit, including the log floor."""
    out = []
    n, m = stats["n_players"], stats["matches"]
    sd, rate, xps = stats["sd_log_latent"], stats["mean_rate"], stats["xg_per_shot"]
    lo, hi = stats.get("rate_lo", 1e-6), stats.get("rate_hi", 1.0)
    for _ in range(n_sims):
        mu_t = np.clip(np.exp(rng.normal(np.log(rate), sd, n)), lo, hi)
        mu_t1 = np.clip(
            np.exp(np.log(rate) + rho_true * (np.log(mu_t) - np.log(rate))
                   + rng.normal(0, sd * np.sqrt(max(1 - rho_true ** 2, 0.0)), n)), lo, hi)

        def observe(mu):
            k = rng.poisson(mu / xps * m)
            return rng.gamma(2.0 * np.maximum(k, 1e-9), xps / 2.0) / m

        x, y = _floor_log(observe(mu_t)), _floor_log(observe(mu_t1))
        if np.std(x) < 1e-9:
            continue
        out.append(np.polyfit(x, y, 1)[0])
    return float(np.mean(out)) if out else np.nan


def panel_stats(panel: pd.DataFrame, component: str, reliability: float,
                min_minutes: float = MIN_MINUTES_SEASON) -> dict:
    """Everything the simulation needs to mimic THIS panel's count intensity.

    Restricted to the SAME subpopulation the AR(1) is fitted on. Computing it over the
    whole panel lets 20-minute cameos in — a player with one minute and one shot has a
    per-90 rate in the tens, which drags the fitted lognormal wide enough to overflow
    the Poisson draw. The simulation has to mimic the estimation sample, not the roster.
    """
    p = panel[(panel["component"] == component) & (panel["minutes"] >= min_minutes)]
    if p.empty:
        raise ValueError(f"no {component} rows with >= {min_minutes} minutes")
    rates = p["npxg_p90"].values
    lx = _floor_log(rates)
    var_latent = max(reliability * np.var(lx, ddof=1), 1e-6)
    shots = p["n_shots"].sum() if "n_shots" in p.columns else 0
    pos = rates[rates > 0]
    st = {"n_players": int(p["player_code"].nunique()),
          "matches": float(np.median(p["minutes"] / 90.0)),
          "mean_rate": float(np.exp(np.mean(lx))),
          "sd_log_latent": float(np.sqrt(var_latent)),
          "xg_per_shot": float(p["npxg"].sum() / shots) if shots else 0.09,
          # Observed support. The latent rate distribution is genuinely wide — open-play
          # xG/90 runs from a centre-back's 0.0003 to Haaland's 0.70, so sd(log) ~ 1.6 is
          # real heterogeneity, not an artefact. But an unbounded lognormal draw at that
          # spread produces players taking millions of shots, which overflows the Poisson.
          # Bounding to the observed support keeps the simulation inside football.
          "rate_lo": float(np.quantile(pos, 0.01)) if len(pos) else 1e-6,
          "rate_hi": float(np.quantile(pos, 0.99)) if len(pos) else 1.0}
    # Fail here rather than 19 grid points deep inside the simulation, where the only
    # symptom is an opaque "lam value too large" from the Poisson draw.
    if not all(np.isfinite(v) for v in st.values()):
        raise ValueError(f"{component}: non-finite panel stats {st}")
    if st["mean_rate"] <= 1e-5 or shots == 0:
        raise ValueError(
            f"{component}: degenerate panel (mean_rate={st['mean_rate']:.2e}, "
            f"{shots} shots). Usually means the component matched no shots — check "
            f"that the `components` mapping is a PARTITION and not two overlapping ones.")
    if st["sd_log_latent"] > 3.0:
        raise ValueError(
            f"{component}: sd_log_latent={st['sd_log_latent']:.2f} is beyond anything a "
            f"per-90 rate should show even across roles — check the floor in _floor_log.")
    return st


def invert_attenuation(rho_measured: float, se_measured: float, stats: dict,
                       grid=None, n_sims: int = 60, seed: int = 0) -> dict:
    """Find the rho_true whose simulated measured slope equals the observed one.

    The CI is obtained by inverting `rho_measured ± 1.96 se` through the same curve,
    so it inherits the curve's own steepness — a flat curve (heavily attenuated
    component) correctly produces a very wide interval, which is the honest answer
    when the data cannot pin the quantity down.
    """
    grid = np.arange(0.05, 0.99, 0.02) if grid is None else np.asarray(grid, float)
    rng = np.random.default_rng(seed)
    curve = np.array([_simulate_measured_rho(r, stats, n_sims, rng) for r in grid])
    ok = np.isfinite(curve)
    grid, curve = grid[ok], curve[ok]
    order = np.argsort(curve)                       # np.interp needs increasing x

    def inv(v):
        return float(np.interp(v, curve[order], grid[order]))

    return {"rho_corrected": inv(rho_measured),
            "ci_lo": inv(rho_measured - 1.96 * se_measured),
            "ci_hi": inv(rho_measured + 1.96 * se_measured),
            "curve_rho_true": grid.tolist(),
            "curve_rho_measured": curve.tolist(),
            "saturated": bool(rho_measured > curve.max() or rho_measured < curve.min())}


def corrected_ar1(ar1: pd.DataFrame, panel: pd.DataFrame, n_sims: int = 60,
                  seed: int = 0) -> pd.DataFrame:
    """Attach the simulation-inverted persistence to an `component_ar1` frame:
    rho_corrected, ci_lo_sim, ci_hi_sim. This is the estimate `decide()` uses."""
    out = ar1.copy()
    cols = {"rho_corrected": [], "ci_lo_sim": [], "ci_hi_sim": [], "sim_saturated": []}
    for _, r in out.iterrows():
        st = panel_stats(panel, r["component"], r["reliability"])
        inv = invert_attenuation(r["rho_measured"], r["se_measured"], st,
                                 n_sims=n_sims, seed=seed)
        cols["rho_corrected"].append(inv["rho_corrected"])
        cols["ci_lo_sim"].append(inv["ci_lo"])
        cols["ci_hi_sim"].append(inv["ci_hi"])
        cols["sim_saturated"].append(inv["saturated"])
    for k, v in cols.items():
        out[k] = v
    return out


def decide(ar1: pd.DataFrame, n90_median: float = 30.0,
           component: str = "set_piece_all") -> dict:
    """Apply the pre-committed rule. Returns the verdict and the numbers behind it.
    Does not modify anything — the caller records it, per G7."""
    w0 = implied_pooled_persistence(n90_median)
    r = ar1[ar1["component"] == component]
    if r.empty:
        return {"verdict": "no_estimate", "component": component,
                "w0_pooled": w0,
                "note": f"no AR(1) row for {component}; nothing to decide"}
    r = r.iloc[0]

    # A saturated inversion means the observed slope is larger than the simulated count
    # process can produce at ANY persistence — the estimate is pinned at the grid edge
    # and its CI is meaningless. That is a model-misspecification signal, not a result,
    # and it must not be laundered into a verdict.
    if bool(r.get("sim_saturated", False)):
        return {"verdict": "inconclusive_saturated", "component": component,
                "basis": "simulation", "w0_pooled": w0,
                "rho": float(r["rho_corrected"]), "n_pairs": int(r["n_pairs"]),
                "reliability": float(r["reliability"]),
                "note": ("the measured slope exceeds anything the simulated count process "
                         "produces at any rho_true, so the inversion is pinned at the grid "
                         "edge. The simulation is missing structure the real panel has — "
                         "most likely persistent structural zeros (a centre-back takes no "
                         "set-piece shots in either season, which the simulation redraws "
                         "at random). Do NOT act on this; fix the specification first.")}

    # prefer the simulation-inverted estimate; the ratio correction is biased upward
    # for exactly this component (see the note above invert_attenuation)
    if "rho_corrected" in r.index and np.isfinite(r["rho_corrected"]):
        rho, lo, hi, basis = (float(r["rho_corrected"]), float(r["ci_lo_sim"]),
                              float(r["ci_hi_sim"]), "simulation")
    else:
        rho, lo, hi, basis = (float(r["rho_disattenuated"]), float(r["ci_lo"]),
                              float(r["ci_hi"]), "ratio_disattenuation")
        print("[setpiece] WARNING: deciding on the ratio correction, which is biased "
              "upward for low-count components. Run corrected_ar1() first.")
    excludes = not (lo <= w0 <= hi)
    return {
        "verdict": "build_two_component" if excludes else "keep_pooled_prior",
        "component": component,
        "basis": basis,
        "w0_pooled": w0,
        "rho": rho,
        "ci": (lo, hi),
        "reliability": float(r["reliability"]),
        "n_pairs": int(r["n_pairs"]),
        "note": ("95% CI excludes the pooled prior's implied persistence — "
                 "component-specific shrinkage is warranted (§2.3, then A/B §2.5)"
                 if excludes else
                 "95% CI covers the pooled prior's implied persistence — keep the "
                 "pooled prior, record the reliability bound in docs/, stop"),
    }


# ------------------------------------------------------------------- selftest
def _simulate(n_players=400, matches=34, rho_true=0.55, rate=0.035,
              minutes_per_match=90.0, seed=11):
    """Compound-Poisson shots with a KNOWN latent AR(1). The point of the selftest
    is that disattenuation recovers `rho_true` while the raw slope does not."""
    rng = np.random.default_rng(seed)
    codes = np.arange(1000, 1000 + n_players)
    mu_t = np.exp(rng.normal(np.log(rate), 0.55, n_players))
    mu_t1 = np.exp(np.log(rate) + rho_true * (np.log(mu_t) - np.log(rate))
                   + rng.normal(0, 0.55 * np.sqrt(1 - rho_true ** 2), n_players))

    shots, mins = [], []
    for season, mu in (("2425", mu_t), ("2526", mu_t1)):
        for i, code in enumerate(codes):
            k = rng.poisson(mu[i] / 0.09 * matches)          # ~0.09 xG per shot
            m = rng.integers(0, matches, k)
            shots.append(pd.DataFrame({
                "player_code": code, "season": season, "match_id": m,
                "situation": "FromCorner",
                "xg": rng.gamma(2.0, 0.045, k)}))
            mins.append(pd.DataFrame({
                "player_code": code, "season": season,
                "match_id": np.arange(matches), "minutes": minutes_per_match}))
    return pd.concat(shots, ignore_index=True), pd.concat(mins, ignore_index=True)


def selftest(n_seeds=3):
    """Validates the estimator in a WELL-SPECIFIED regime, across seeds rather than on
    one draw — the sampling noise on a 400-player AR(1) is larger than the bias being
    measured, so a single seed proves nothing either way.

    Deliberately does NOT assert that the inversion beats the ratio. Under the current
    floor the two are indistinguishable here (see the note above `invert_attenuation`);
    the case against the ratio is real-data evidence, not simulated. Asserting an
    ordering that holds by ~0.005 would be a flaky test dressed up as a finding.
    """
    rho_true = 0.55
    comps = {"set_piece_all": ("FromCorner",)}
    ratio, sim, covered, first = [], [], [], None

    for seed in range(n_seeds):
        shots, mins = _simulate(rho_true=rho_true, seed=100 + seed)
        rel = split_half_reliability(shots, mins, n_splits=25, components=comps,
                                     seed=seed)
        assert len(rel) == 2, "expected one reliability row per season"
        r = float(rel[rel["season"] == "2425"]["reliability"].iloc[0])
        assert 0.2 < r < 0.95, f"implausible reliability {r:.3f}"

        panel = component_panel(shots, mins, components=comps)
        # every player-season gets a row per component, shots or not
        assert len(panel) == panel[["player_code", "season"]].drop_duplicates().shape[0] \
            * len(comps), "panel must not be selected on having taken a shot"
        assert panel["npxg_p90"].notna().all()

        ar1 = component_ar1(panel, {"set_piece_all": r},
                            reliability_se={"set_piece_all": float(
                                rel[rel["season"] == "2425"]["reliability_sd"].iloc[0])})
        row = ar1.iloc[0]
        assert row["rho_measured"] < rho_true, \
            f"measured rho {row['rho_measured']:.3f} should be attenuated"
        assert row["se_disattenuated"] > row["se_measured"], \
            "correcting for measurement error must widen the interval, not narrow it"

        corr = corrected_ar1(ar1, panel, n_sims=40, seed=seed)
        crow = corr.iloc[0]
        # NOT asserted per seed: the ratio sits above the inversion in 11 of 12 seeds,
        # so a single fit can invert the ordering. Only the aggregate is an invariant.
        assert crow["ci_lo_sim"] <= crow["rho_corrected"] <= crow["ci_hi_sim"]
        ratio.append(crow["rho_disattenuated"]); sim.append(crow["rho_corrected"])
        covered.append(bool(crow["ci_lo_sim"] <= rho_true <= crow["ci_hi_sim"]))
        first = first or (r, crow, corr)

    bias_ratio = float(np.mean(ratio) - rho_true)
    bias_sim = float(np.mean(sim) - rho_true)
    # both estimators should be near-unbiased when the model is well specified; the
    # inversion's interval must cover. That is what this regime can actually establish.
    assert abs(bias_sim) < 0.08, f"inversion biased by {bias_sim:+.3f}"
    assert abs(bias_ratio) < 0.10, f"ratio biased by {bias_ratio:+.3f} in a clean regime"
    assert all(covered), "the inversion CI must cover the truth"

    r, crow, corr = first
    w0 = implied_pooled_persistence(30.0)
    assert 0.8 < w0 < 0.95, f"implied pooled persistence {w0:.3f} off expected scale"
    d = decide(corr, component="set_piece_all")
    assert d["basis"] == "simulation", "decide() must prefer the inverted estimate"
    assert d["verdict"] in ("build_two_component", "keep_pooled_prior")

    print(f"SELFTEST OK ({n_seeds} seeds, true rho {rho_true}): reliability ~{r:.3f}; "
          f"ratio mean {np.mean(ratio):.3f} (bias {bias_ratio:+.3f}) vs simulation "
          f"mean {np.mean(sim):.3f} (bias {bias_sim:+.3f}), CI coverage "
          f"{np.mean(covered):.0%}; pooled-prior w0={w0:.3f}; "
          f"verdict '{d['verdict']}' on {d['basis']}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seasons", nargs="+", default=["2425", "2526"])
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    print("Needs a calibrated, player_code-keyed shot panel. On a networked box:\n"
          "  python sd_ingest.py --what shots --seasons " + " ".join(a.seasons) + "\n"
          "  python crosswalk.py --build     # then hand-verify the review file\n"
          "  python xg_calibrate.py          # confirm the gate passes\n"
          "then drive component_panel / split_half_reliability / component_ar1 from a\n"
          "runner in studies/ and write the CSVs listed in the spec §9.")
