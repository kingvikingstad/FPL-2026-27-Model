import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
start_prior_production.py — re-fit the start-prior strength against the PRODUCTION prior
========================================================================================
PROJECT_KNOWLEDGE §6.9(a). `start_prior_strength.py` showed the in-season minutes update
(`w = 1`, uncapped prior) is dominated by ~11% on Brier, but it fitted a stand-in prior —
last season's native starts, one pseudo-count per match — and on that stand-in only the
ratio w/kappa is identified. Its own docstring says to re-fit against the real prior
before pinning any constant. This is that re-fit. It changes the PRIOR and nothing else:
same grids, same Brier endpoint, same leave-one-season-out folds, same held-out seasons.

THE PRIOR AS INSTALLED, AND HOW IT IS REPLICATED  (read from the code, 2026-09-17)
-----------------------------------------------------------------------------------
For an established player (pooled weighted minutes >= 270), `gw_board.py` builds:

  1. `multiseason_priors.two_season_evidence(older_weight=0.5)`: season s-1 at full
     weight plus season s-2 at 0.5 — minutes, starts, games. A START HERE IS mins >= 60,
     not the native `starts` column. And `games` is the row count of `pms_panel` /
     `build_2425_panel`, which are MATCH-SHEET rows: 24/25 carries no zero-minute row at
     all and 25/26 only 9.9%. So `games` is roughly APPEARANCES, `start_b` counts
     substitute appearances rather than matches missed, and the installed prior
     estimates P(start | appeared) — while step 4 updates it with P(start | club match).
     [VERIFIED 2026-09-17, found by the fidelity gate below — see FIRST RUN.]
  2. `to_priors(revert=0.70)`:  start_a = 2 + 0.7*starts,  start_b = 2 + 0.7*(games-starts).
     A two-season regular therefore carries ~4 + 0.7*57 = 44 pseudo-matches, not 38.
  3. `starter_prior.apply_regime_uncertainty` (REGIME=off, the default): the MEAN is pulled
     toward an ownership+price logit with weight w = minutes/(minutes + 900), strength N
     held. Ownership is the LIVE `selected_by_percent` at the time the board runs, and the
     logit is fitted on the most recent completed season (`calibrate_ownership_start`:
     end-of-season ownership and price against mins>=60 starts / 38).
  4. `inseason.cap_start_prior(kappa)` when INSEASON_KAPPA is set, then
     `inseason.update_minutes(w)`:  start_a += w*starts, start_b += w*(matches - starts),
     with NATIVE starts from the current season.

Replicated here from vaastav history for each held-out season s: steps 1-2 from s-1 and
s-2 minutes; the step-3 logit refitted on s-1 exactly as production fits it on 25/26;
ownership read at the deadline AFTER the k-th match, which is when a board using k
matches of evidence would run (percent = selected / managers, managers = total selected
over distinct players / 15). Cold starts (<270 pooled minutes) are out of scope, as they
were in the original: their prior comes from `apply_coldstart_depth`, not from history.
Not replicated: `apply_availability` and the predicted-XI overrides, which act after the
update and replace it outright for flagged players.

FIRST RUN (2026-09-17) FAILED THE FIDELITY GATE and scored nothing: the replica then
counted every vaastav row as a game, giving median strength 43.9 against 26.6 installed
(r of means 0.863). The replica now counts appearances (mins > 0) as games, matching the
match-sheet panels. The gate thresholds were NOT changed. S5 below was added at the same
time — after that failure, before any Brier score had been computed.

FIDELITY GATE, before any held-out season is scored: build the same replica for 26/27
(25/26 full + 24/25 at 0.5) and compare it with the installed `ms_priors.pkl`. The
replica is accepted only if Pearson r of the prior MEAN >= 0.95 AND the median relative
error of prior STRENGTH (a+b) <= 10%. If either fails, the study stops and claims nothing.

PRE-REGISTERED — WRITTEN 2026-09-17, BEFORE THIS FILE WAS FIRST RUN
--------------------------------------------------------------------
Held-out seasons 2023-24, 2024-25, 2025-26 (native `starts` is observed throughout; the
22/23 zero block does not enter, because priors are built from minutes). Cutoffs
k = 2, 3, 5, 8. Target: each REMAINING match's native start indicator (primary, as in the
original). Endpoint: pooled Brier over remaining matches. Baseline: w = 1, kappa = inf.

Q1  REPLICATION. At each k: adopt "the current corner is dominated" only if the LOSO-fitted
    (w, kappa) improves pooled Brier by >= 2% AND the gain is positive in 3/3 held-out
    seasons. Otherwise NULL at that k — the original result was an artefact of its
    stand-in prior.

Q2  WHICH LEVER. The two single-lever corners, each fitted on the training seasons only:
      CAP     w = 1,  kappa* = argmin over KAPPA_GRID
      WEIGHT  kappa = inf,  w* = argmin over W_GRID
    At each k they are DISTINGUISHABLE if the pooled held-out Brier differs by >= 0.5% of
    the baseline Brier AND the sign agrees in 3/3 folds. The lever is DECIDED if one corner
    wins that way at >= 3 of the 4 cutoffs, in the same direction. Otherwise UNIDENTIFIED,
    and the cap stays the implemented form by default — it bounds how much prior any
    player can carry without rescaling what a realised match counts for.

Q3  THE CONSTANT (only if Q1 passes at >= 3 cutoffs and Q2 does not decide for WEIGHT).
    At w = 1, kappa_rec = the KAPPA_GRID value minimising Brier pooled over all four
    cutoffs and all three seasons; ties go to the LARGER kappa (less aggressive). It is
    pinned as the INSEASON_KAPPA default only if it is INTERIOR to the grid (not 1, not
    inf) AND every cutoff's median LOSO kappa* lies within one grid step of it.
    Otherwise no constant is pinned, and the report says which condition failed.

Not decision-bearing, declared in advance so none can be promoted after the fact:
  S1  target mins >= 60 instead of native starts (the prior's own definition)
  S2  the same rows with the step-3 ownership shrink removed (isolates its effect)
  S3  the ORIGINAL stand-in prior on the same rows (is any difference the prior or the rows?)
  S4  horizon = the next 10 matches rather than the rest of the season (the board's window)
  S5  the prior with games = ALL listed matches, i.e. P(start | club match), the
      denominator the in-season update and the target both use (shrink kept, so the
      denominator is the only difference from the primary)
  GK-only line at kappa_rec: descriptive, because H-POS is a recorded null.

Run:  python studies/start_prior_production.py
Out:  studies/start_prior_production.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import fpl_history as fh
import start_prior_strength as sps

# --- pre-registered constants -------------------------------------------------
TEST_SEASONS = ("2023-24", "2024-25", "2025-26")
CUTOFFS = sps.CUTOFFS
W_GRID, KAPPA_GRID, BASE = sps.W_GRID, sps.KAPPA_GRID, sps.BASE
MIN_GAIN, CONSISTENT = sps.MIN_GAIN, 3
LEVER_MARGIN = 0.005
LEVER_CUTOFFS = 3
MIN_REMAIN_MATCHES = sps.MIN_REMAIN_MATCHES
HORIZON_S4 = 10
# the production prior, as installed (multiseason_priors / starter_prior)
OLDER_WEIGHT, REVERT, A0, B0 = 0.5, 0.70, 2.0, 2.0
MIN_POOLED_MINUTES = 270
K_MIN_SHRINK = 900.0
TEAM_GAMES = 38
FID_R, FID_STRENGTH = 0.95, 0.10
OUT = _os.path.join(config.STUDIES, "start_prior_production.csv")


# ---------------------------------------------------------------- the replica
def season_evidence(P, season, weight=1.0):
    """`games` = appearances, as the installed match-sheet panels count it;
    `rows` = every listed match, the per-match denominator (S5)."""
    g = P[P["season"] == season]
    ev = g.groupby("player_code").agg(mins=("mins", "sum"),
                                      st60=("mins", lambda s: int((s >= 60).sum())),
                                      games=("mins", lambda s: int((s > 0).sum())),
                                      rows=("mins", "size"))
    return ev * weight


def installed_prior(P, s1, s2):
    """Steps 1-2: pooled evidence -> Beta(start_a, start_b), established players only."""
    ev = season_evidence(P, s1).add(season_evidence(P, s2, OLDER_WEIGHT), fill_value=0.0)
    ev = ev[ev["mins"] >= MIN_POOLED_MINUTES].copy()
    ev["raw_a"] = A0 + REVERT * ev["st60"]
    ev["raw_b"] = B0 + REVERT * (ev["games"] - ev["st60"]).clip(lower=0)
    ev["rows_b"] = B0 + REVERT * (ev["rows"] - ev["st60"]).clip(lower=0)
    return ev


def managers_by_gw(g):
    u = g.drop_duplicates(["player_code", "gw"])
    return u.groupby("gw")["selected"].sum() / 15.0


def fit_ownership_logit(P, season):
    """`calibrate_ownership_start`, applied to one completed season."""
    g = P[P["season"] == season].sort_values(["player_code", "gw"], kind="stable")
    mgr = managers_by_gw(g)
    last = g.groupby("player_code").tail(1).set_index("player_code")
    gp = g.groupby("player_code").agg(starts=("mins", lambda s: int((s >= 60).sum())),
                                      pos=("pos", "last"))
    gp["own"] = 100.0 * last["selected"] / last["gw"].map(mgr)
    gp["price"] = last["price"]
    gp = gp.dropna(subset=["own", "price", "pos"])
    sr = (gp["starts"] / TEAM_GAMES).clip(0.01, 0.99)
    gp["logit"] = np.log(sr / (1 - sr))
    gp["logown"] = np.log(gp["own"].clip(0.05, 95))
    cal = {}
    for pos in ("GK", "DEF", "MID", "FWD"):
        s = gp[gp["pos"] == pos]
        if len(s) < 8:
            s = gp
        X = np.column_stack([np.ones(len(s)), s["logown"], s["price"]])
        cal[pos], *_ = np.linalg.lstsq(X, s["logit"].to_numpy(), rcond=None)
    return cal


def own_prob(own, price, pos, cal):
    b = cal.get(pos, cal["MID"])
    lo = b[0] + b[1] * np.log(np.maximum(own, 0.05)) + b[2] * price
    return np.clip(1.0 / (1.0 + np.exp(-lo)), 0.03, 0.97)


def build(P):
    """One row per (player_code, held-out season, cutoff), with the production prior."""
    order = list(fh.SEASONS)
    rows = []
    for s in TEST_SEASONS:
        i = order.index(s)
        s1, s2 = order[i - 1], order[i - 2]
        prior = installed_prior(P, s1, s2)
        cal = fit_ownership_logit(P, s1)
        g = P[P["season"] == s].sort_values(["player_code", "gw"], kind="stable")
        mgr = managers_by_gw(g)
        prev = P[P["season"] == s1].groupby("player_code").agg(
            pa=("is_start", "sum"), pn=("is_start", "size"))
        for code, h in g.groupby("player_code"):
            if code not in prior.index:
                continue
            pr = prior.loc[code]
            y = h["is_start"].to_numpy().astype(float)
            y60 = (h["mins"].to_numpy() >= 60).astype(float)
            own = (100.0 * h["selected"] / h["gw"].map(mgr)).to_numpy()
            price = h["price"].to_numpy().astype(float)
            pos = h["pos"].dropna().iloc[-1] if h["pos"].notna().any() else "MID"
            N = pr["raw_a"] + pr["raw_b"]
            p_h = pr["raw_a"] / N
            N5 = pr["raw_a"] + pr["rows_b"]
            p_h5 = pr["raw_a"] / N5
            w_sh = pr["mins"] / (pr["mins"] + K_MIN_SHRINK)
            pa, pn = (prev.loc[code, "pa"], prev.loc[code, "pn"]) if code in prev.index \
                else (np.nan, np.nan)
            for k in CUTOFFS:
                if len(y) < k + MIN_REMAIN_MATCHES or not np.isfinite(own[k]):
                    continue
                p_o = float(own_prob(own[k], price[k], pos, cal))
                p_bar = w_sh * p_h + (1 - w_sh) * p_o
                p_bar5 = w_sh * p_h5 + (1 - w_sh) * p_o
                rest = slice(k, None)
                h10 = slice(k, k + HORIZON_S4)
                rows.append({
                    "player_code": code, "season": s, "pos": pos, "k": k,
                    "prior_a": float(np.clip(p_bar, 1e-3, 1) * N),
                    "prior_b": float(np.clip(1 - p_bar, 1e-3, 1) * N),
                    "raw_a": pr["raw_a"], "raw_b": pr["raw_b"],
                    "rows_a": float(np.clip(p_bar5, 1e-3, 1) * N5),
                    "rows_b": float(np.clip(1 - p_bar5, 1e-3, 1) * N5),
                    "p_prior": float(p_bar), "p_prior_rows": float(p_bar5),
                    "prev_a": float(pa) if pn >= sps.MIN_PRIOR_MATCHES else np.nan,
                    "prev_b": float(pn - pa) if pn >= sps.MIN_PRIOR_MATCHES else np.nan,
                    "early_starts": float(y[:k].sum()),
                    "rest_starts": float(y[rest].sum()), "rest_n": int(len(y) - k),
                    "rest_60": float(y60[rest].sum()),
                    "rest_h10": float(y[h10].sum()), "rest_n_h10": int(len(y[h10])),
                })
    return pd.DataFrame(rows)


def fidelity(P):
    """Replica for 26/27 against the installed ms_priors.pkl."""
    rep = installed_prior(P, "2025-26", "2024-25")
    ins = pd.read_pickle(config.MS_PRIORS).dropna(subset=["player_code"])
    ins = ins.drop_duplicates("player_code").set_index("player_code")
    j = rep.join(ins[["start_a", "start_b"]], how="inner")
    m_rep = j["raw_a"] / (j["raw_a"] + j["raw_b"])
    m_ins = j["start_a"] / (j["start_a"] + j["start_b"])
    n_rep, n_ins = j["raw_a"] + j["raw_b"], j["start_a"] + j["start_b"]
    return {"n_joined": len(j), "n_replica": len(rep), "n_installed": len(ins),
            "r_mean": float(np.corrcoef(m_rep, m_ins)[0, 1]),
            "strength_rel_err": float(((n_rep - n_ins).abs() / n_ins).median()),
            "strength_median_installed": float(n_ins.median()),
            "strength_median_replica": float(n_rep.median())}


# ---------------------------------------------------------------- evaluation
def view(D, target="rest_starts", n="rest_n", a="prior_a", b="prior_b"):
    V = D[["k", "early_starts", "pos", "season"]].copy()
    V["prior_a"], V["prior_b"] = D[a], D[b]
    V["rest_starts"], V["rest_n"] = D[target], D[n]
    return V.dropna(subset=["prior_a", "prior_b"])


def argmin_grid(df, pairs):
    return min(pairs, key=lambda wk: sps.brier(df, *wk))


def loso(V):
    """Q1 and Q2 per cutoff: every setting fitted on two seasons, scored on the third."""
    out = []
    cap_pairs = [(1.0, kp) for kp in KAPPA_GRID]
    wt_pairs = [(w, np.inf) for w in W_GRID]
    full = [(w, kp) for w in W_GRID for kp in KAPPA_GRID]
    for k in CUTOFFS:
        Vk = V[V["k"] == k]
        folds = []
        for s in TEST_SEASONS:
            tr, te = Vk[Vk["season"] != s], Vk[Vk["season"] == s]
            if len(tr) < 30 or len(te) < 20:
                continue
            best = argmin_grid(tr, full)
            cap = argmin_grid(tr, cap_pairs)
            wt = argmin_grid(tr, wt_pairs)
            folds.append({"season": s, "n": int(te["rest_n"].sum()),
                          "base": sps.brier(te, *BASE), "best": sps.brier(te, *best),
                          "cap": sps.brier(te, *cap), "wt": sps.brier(te, *wt),
                          "w_best": best[0], "kappa_best": best[1],
                          "kappa_cap": cap[1], "w_wt": wt[0]})
        F = pd.DataFrame(folds)
        pool = lambda c: float((F[c] * F["n"]).sum() / F["n"].sum())
        base = pool("base")
        # Q2 difference per fold, as a share of that fold's baseline: + means CAP better
        F["lever"] = (F["wt"] - F["cap"]) / F["base"]
        lever = (pool("wt") - pool("cap")) / base
        out.append({"k": k, "n_rows": len(Vk), "n_matches": int(Vk["rest_n"].sum()),
                    "folds": len(F), "brier_base": base,
                    "gain_best": (base - pool("best")) / base,
                    "folds_pos_best": int((F["best"] < F["base"]).sum()),
                    "gain_cap": (base - pool("cap")) / base,
                    "gain_wt": (base - pool("wt")) / base,
                    "lever_diff": lever,
                    "lever_sign_agree": int((np.sign(F["lever"]) == np.sign(lever)).sum()),
                    "kappa_cap_median": float(np.median(F["kappa_cap"])),
                    "w_wt_median": float(np.median(F["w_wt"])),
                    "_F": F})
    return out


def q1_pass(r):
    return r["gain_best"] >= MIN_GAIN and r["folds_pos_best"] >= CONSISTENT


def q2_lever(r):
    if abs(r["lever_diff"]) >= LEVER_MARGIN and r["lever_sign_agree"] >= CONSISTENT:
        return "CAP" if r["lever_diff"] > 0 else "WEIGHT"
    return "tie"


def kappa_rec(V):
    scores = {kp: sps.brier(V, 1.0, kp) for kp in KAPPA_GRID}
    best = min(scores.values())
    return max(kp for kp, v in scores.items() if v <= best + 1e-12), scores


def grid_index(x):
    g = list(KAPPA_GRID)
    if not np.isfinite(x):
        return len(g) - 1
    return min((i for i, v in enumerate(g) if np.isfinite(v)), key=lambda i: abs(g[i] - x))


def grid_step_ok(kp, med):
    return abs(grid_index(kp) - grid_index(med)) <= 1


def fmt_k(kp):
    return "inf" if not np.isfinite(kp) else f"{kp:g}"


def print_table(res, title):
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)
    print(f"{'k':>3} {'matches':>9} {'Brier now':>10} {'best':>7} {'folds+':>6} "
          f"{'CAP':>7} {'kap*':>5} {'WEIGHT':>7} {'w*':>4} {'CAP-WT':>7} {'agree':>5}")
    for r in res:
        print(f"{r['k']:>3} {r['n_matches']:>9,} {r['brier_base']:>10.4f} "
              f"{r['gain_best']:>6.1%} {r['folds_pos_best']:>3}/{r['folds']:<2} "
              f"{r['gain_cap']:>6.1%} {fmt_k(r['kappa_cap_median']):>5} "
              f"{r['gain_wt']:>6.1%} {r['w_wt_median']:>4g} "
              f"{r['lever_diff']:>+6.2%} {r['lever_sign_agree']:>3}/{r['folds']}")


def main():
    print("=" * 78)
    print("START PRIOR STRENGTH, RE-FIT ON THE PRODUCTION PRIOR (PROJECT_KNOWLEDGE §6.9a)")
    print("=" * 78)
    P = fh.load_history(verbose=False)

    fid = fidelity(P)
    ok = fid["r_mean"] >= FID_R and fid["strength_rel_err"] <= FID_STRENGTH
    print(f"\nFIDELITY GATE — replica vs installed ms_priors.pkl for 26/27")
    print(f"  joined {fid['n_joined']} (replica {fid['n_replica']}, installed "
          f"{fid['n_installed']})")
    print(f"  prior mean  r = {fid['r_mean']:.3f}   (gate >= {FID_R})")
    print(f"  strength    median rel err = {fid['strength_rel_err']:.1%}   (gate <= "
          f"{FID_STRENGTH:.0%}); median a+b installed {fid['strength_median_installed']:.1f}"
          f", replica {fid['strength_median_replica']:.1f}")
    rows_csv = [{"section": "fidelity", **fid, "verdict": "PASS" if ok else "FAIL"}]
    if not ok:
        pd.DataFrame(rows_csv).to_csv(OUT, index=False)
        print("\nRESULT: FIDELITY GATE FAILED — the replica is not the installed prior, "
              "so nothing below would be about production. No claim made.")
        return 0

    D = build(P)
    print(f"\nrows {len(D)} ({D['player_code'].nunique()} players, "
          f"{int(D.drop_duplicates(['player_code', 'season'])['rest_n'].sum()):,} "
          f"distinct remaining matches at k={CUTOFFS[0]})")
    s_prior = D.drop_duplicates(["player_code", "season"])
    print(f"prior strength a+b: median {(s_prior.prior_a + s_prior.prior_b).median():.1f}, "
          f"p90 {(s_prior.prior_a + s_prior.prior_b).quantile(.9):.1f}")

    s1 = D[D["k"] == CUTOFFS[0]]
    real = (s1["early_starts"] + s1["rest_starts"]) / (s1["k"] + s1["rest_n"])
    print(f"calibration-in-the-large, prior mean vs realised native start rate: installed "
          f"{s1['p_prior'].mean():.3f}, per-match denominator {s1['p_prior_rows'].mean():.3f}, "
          f"realised {real.mean():.3f}")

    V = view(D)
    res = loso(V)
    print_table(res, "PRIMARY — production prior, native-start target, rest of season "
                     "(gains vs w=1, kappa=inf)")

    q1 = [r for r in res if q1_pass(r)]
    levers = [q2_lever(r) for r in res]
    decided = None
    for L in ("CAP", "WEIGHT"):
        if levers.count(L) >= LEVER_CUTOFFS:
            decided = L
    for r, L in zip(res, levers):
        rows_csv.append({"section": "primary",
                         **{k: v for k, v in r.items() if k != "_F"},
                         "q1": "ADOPT" if q1_pass(r) else "null", "q2": L})

    print(f"\nQ1  dominated at k = {[r['k'] for r in q1] or 'none'} "
          f"(rule: >= {MIN_GAIN:.0%} and {CONSISTENT}/{CONSISTENT} folds)")
    print(f"Q2  per cutoff: {dict(zip([r['k'] for r in res], levers))} -> "
          f"{'DECIDED: ' + decided if decided else 'UNIDENTIFIED'} "
          f"(rule: |diff| >= {LEVER_MARGIN:.1%}, {CONSISTENT}/{CONSISTENT} folds, "
          f">= {LEVER_CUTOFFS} cutoffs)")

    kp_rec, scores = kappa_rec(V)
    base_all = sps.brier(V, *BASE)
    print("\nQ3  w=1, Brier pooled over all cutoffs and seasons, by kappa:")
    print("    " + "  ".join(f"{fmt_k(kp)}:{(base_all - v) / base_all:+.1%}"
                             for kp, v in scores.items()))
    q3_run = len(q1) >= 3 and decided != "WEIGHT"
    interior = np.isfinite(kp_rec) and kp_rec != KAPPA_GRID[0]
    near = all(grid_step_ok(kp_rec, r["kappa_cap_median"]) for r in res)
    pinned = q3_run and interior and near
    why = ("Q1 did not pass at >= 3 cutoffs" if len(q1) < 3 else
           "Q2 decided for WEIGHT" if decided == "WEIGHT" else
           "kappa_rec is on the grid boundary" if not interior else
           "a cutoff's LOSO kappa* is more than one grid step away" if not near else "")
    print(f"    kappa_rec = {fmt_k(kp_rec)}; interior {interior}; LOSO medians "
          f"{[fmt_k(r['kappa_cap_median']) for r in res]} within one step: {near}")
    print(f"    -> {'PIN INSEASON_KAPPA = ' + fmt_k(kp_rec) if pinned else 'NO CONSTANT PINNED: ' + why}")
    k4 = scores.get(4.0)
    if k4 is not None:
        print(f"    kappa=4 (the current flag default) vs kappa_rec: "
              f"{(k4 - scores[kp_rec]) / base_all:+.2%} of baseline Brier")
    rows_csv.append({"section": "q3", "kappa_rec": kp_rec, "pinned": pinned, "why": why,
                     **{f"gain_kappa_{fmt_k(kp)}": (base_all - v) / base_all
                        for kp, v in scores.items()}})

    # --- ridge on the production prior (reported, as the original does) ---
    print("\nIDENTIFIABILITY on the production prior (full sample, reported only)")
    for k in CUTOFFS:
        R = sps.ridge(V[V["k"] == k])
        rl = (f"[{R['ratio_lo']:.2f}, {R['ratio_hi']:.2f}]" if R["ratio_lo"] is not None
              else "n/a")
        print(f"  k={k:<2} best {R['best_gain']:.1%}; {R['n_near']}/{R['n_grid']} grid points "
              f"within 0.5pp; w/kappa in {rl}")

    # --- declared sensitivities, never decision-bearing ---
    sens = {
        "S1 target mins>=60": view(D, target="rest_60"),
        "S2 no ownership shrink": view(D, a="raw_a", b="raw_b"),
        "S3 original stand-in prior, same rows": view(D, a="prev_a", b="prev_b"),
        "S4 next 10 matches": view(D, target="rest_h10", n="rest_n_h10"),
        "S5 per-match denominator": view(D, a="rows_a", b="rows_b"),
    }
    for label, Vs in sens.items():
        rs = loso(Vs)
        print_table(rs, f"SENSITIVITY {label} (not decision-bearing)")
        for r, L in zip(rs, [q2_lever(r) for r in rs]):
            rows_csv.append({"section": label,
                             **{k: v for k, v in r.items() if k != "_F"}, "q2": L})

    G = V[V["pos"] == "GK"]
    if len(G):
        bg = sps.brier(G, *BASE)
        print(f"\nGK only, descriptive: kappa_rec={fmt_k(kp_rec)} at w=1 gains "
              f"{(bg - sps.brier(G, 1.0, kp_rec)) / bg:+.1%} over {int(G['rest_n'].sum()):,} "
              f"match-cutoffs")

    pd.DataFrame(rows_csv).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
