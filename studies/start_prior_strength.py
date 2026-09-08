import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
start_prior_strength.py — how much is a realised non-start worth against the prior?
===================================================================================
`inseason.update_minutes` is a conjugate Beta update: `start_a += w*starts`,
`start_b += w*(matches - starts)`, at `W_MINUTES = 1.0`. The prior it updates carries
roughly ONE pseudo-match per prior-season appearance. So a player with two full seasons
behind him arrives carrying ~34 pseudo-matches, and two weeks of not being picked move
him almost nowhere.

Measured on the 26/27 board after GW2:

    Dubravka   prior 0.880 on 34.1 pseudo-matches, 0 starts from 2  ->  0.831
    Kinsky     prior 0.652 on 13.8 pseudo-matches, 2 starts from 2  ->  0.696

The keeper who had started both ranked BELOW the two who had not played a minute, and
Dubravka was being offered as a rotation option. `apply_availability` cannot catch this:
these players are fit, they are simply not selected. Across the board 372 players had
started none of their club's two completed matches and 139 of those still carried
`app_ev > 0.5`.

That is a symptom. Whether `W_MINUTES = 1.0` against an uncapped prior is actually the
wrong trade-off is an empirical question, and this study answers it before any constant
is touched.

PRE-REGISTERED — WRITTEN BEFORE THE RESULTS WERE SEEN
------------------------------------------------------
Unit: player-season. For each player with a usable prior season and each cutoff k:

    prior     (a, b) = (starts, matches - starts) in season s-1, one count per match
    early     starts over his FIRST k matches of season s
    target    each of his REMAINING matches of season s, as a 0/1 start indicator

    p_hat = (a' + w * early_starts) / (a' + b' + w * k)

where (a', b') is (a, b) rescaled to hold at most KAPPA pseudo-matches, mean preserved.
The model as it stands is `w = 1.0, KAPPA = inf`. Two levers are swept because they are
not the same thing: `w` up-weights realised evidence uniformly, `KAPPA` caps how much
prior any one player may carry, and they differ exactly where it matters — on the
long-career players whose priors are heaviest.

PRIMARY ENDPOINT   Brier score against each remaining match's start indicator. A proper
                   scoring rule on the quantity the model actually uses, rather than RMSE
                   on a rate, which would let a confident wrong answer off lightly.
DECISION RULE      Adopt a setting only if BOTH hold, leave-one-season-out:
                     (1) pooled Brier improves by at least MIN_GAIN (2%) against the
                         current `w = 1.0, KAPPA = inf`; and
                     (2) the improvement has the same sign in at least CONSISTENT (3)
                         held-out seasons.
                   Otherwise: NULL, the constant is not touched, and this file records why.

PRE-REGISTERED HYPOTHESIS (one, alpha = 0.05)
  H-POS  THE RIGHT WEIGHT IS POSITION-SPECIFIC, and goalkeepers need realised evidence
         weighted more heavily than forwards. A forward left out of two matches is
         ordinary rotation; a keeper left out of two has been dropped, and goalkeeper
         selection is close to deterministic and highly persistent. Tested as: is the
         LOSO-fitted w for GK strictly greater than for FWD in every held-out season?
         A single pre-registered comparison, so no multiplicity correction.

w AND KAPPA ARE NOT SEPARATELY IDENTIFIED — READ THE RATIO
-----------------------------------------------------------
Found after the fact and reported here rather than buried. For a player whose prior
exceeds the cap, the posterior is

    p_hat = (a' + w*s) / (a' + b' + w*k)   with   a' + b' = kappa

so scaling `kappa` and `w` by the same factor leaves the balance of prior mass against
evidence mass unchanged. Most players in this panel are capped, so the Brier surface is
a RIDGE, not a peak: at k=5, ten of the forty-eight grid points sit within 0.5 percentage
points of the optimum and they trace a diagonal — (w=1, kappa=5), (1.5, 8), (2, 8),
(3, 12), (4, 20), (6, 20), (8, inf) — all worth about 11%.

The consequence for what may be claimed: the DIRECTION is robust and the current corner
(w=1, kappa=inf) is dominated, but any single fitted pair is one arbitrary point on a
flat ridge and must not be pinned as "the" answer. `ridge()` reports the near-optimal set
and the identified quantity, the ratio w/kappa. Both endpoints are usable and equivalent:
cap the prior near four pseudo-matches at the current weight, or leave it uncapped and
weight a realised match about eight times more. The study cannot tell them apart.

WHAT THIS STUDY APPROXIMATES
-----------------------------
The production prior is not "last season's starts". It is a two-season pooled Beta
(`multiseason_priors`, older_weight=0.5) after cold-start fill, depth adjustment and
regime shrinkage. This study uses the previous season alone, which reproduces the
STRENGTH property under test — one pseudo-count per observed match, uncapped — without
reproducing the whole stack. So it measures the weighting PRINCIPLE and not the exact
production numbers, and a positive result licenses a calibrated change to `W_MINUTES` /
a prior cap, not a drop-in constant. Cold-start players are outside its scope entirely:
they have no prior season, so they never enter, and their start prior comes from the
ownership calibration instead.

Run:  python studies/start_prior_strength.py
Out:  studies/start_prior_strength.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import fpl_history as fh

# --- pre-registered constants -------------------------------------------------
CUTOFFS = (2, 3, 5, 8)
W_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
# 1..4 were added after the first run: the optimum came back at kappa=5, the LOWEST value
# then tested, so the grid did not bracket it and every fitted kappa was pinned to the
# boundary. Extended, the optimum is INTERIOR at 3-4 — which is what makes the fitted
# value a fit rather than an edge effect.
KAPPA_GRID = (1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0, 40.0, np.inf)
MIN_PRIOR_MATCHES = 5
MIN_REMAIN_MATCHES = 5
MIN_GAIN = 0.02
CONSISTENT = 3
BASE = (1.0, np.inf)          # the model as it stands
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "start_prior_strength.csv")


def usable_seasons(panel):
    return sorted(s for s, g in panel.groupby("season")
                  if not bool(g["starts_derived"].any()))


def build(panel, seasons):
    """One row per (player_code, season, cutoff)."""
    panel = panel.sort_values(["player_code", "season", "gw"])
    order = {s: i for i, s in enumerate(sorted(panel["season"].unique()))}

    per = (panel.groupby(["player_code", "season"])
                .agg(st=("is_start", "sum"), n=("is_start", "size")).reset_index())
    per["si"] = per["season"].map(order)
    prev = per[per["n"] >= MIN_PRIOR_MATCHES][["player_code", "si", "st", "n"]].copy()
    prev["si"] += 1
    prev = prev.rename(columns={"st": "prior_a", "n": "prior_n"})
    pmap = {(r.player_code, r.si): (float(r.prior_a), float(r.prior_n))
            for r in prev.itertuples()}

    rows = []
    for (code, season), g in panel.groupby(["player_code", "season"]):
        if season not in seasons:
            continue
        key = (code, order[season])
        if key not in pmap:
            continue
        a, n = pmap[key]
        b = n - a
        y = g["is_start"].to_numpy().astype(float)
        pos = g["pos"].iloc[0]
        for k in CUTOFFS:
            if len(y) < k + MIN_REMAIN_MATCHES:
                continue
            rows.append({
                "player_code": code, "season": season, "pos": pos, "k": k,
                "prior_a": a, "prior_b": b,
                "early_starts": float(y[:k].sum()),
                "rest_starts": float(y[k:].sum()),
                "rest_n": int(len(y) - k),
            })
    return pd.DataFrame(rows)


def phat(df, w, kappa):
    """Posterior start probability under (w, kappa). Kappa rescales the prior to at most
    kappa pseudo-matches, holding its mean."""
    a, b = df["prior_a"].to_numpy(), df["prior_b"].to_numpy()
    tot = a + b
    scale = np.ones_like(tot)
    if np.isfinite(kappa):
        scale = np.where(tot > kappa, kappa / np.maximum(tot, 1e-9), 1.0)
    a2, b2 = a * scale, b * scale
    return (a2 + w * df["early_starts"].to_numpy()) / (a2 + b2 + w * df["k"].to_numpy())


def brier(df, w, kappa):
    """Pooled Brier over every remaining MATCH, not every player: a player with 30 matches
    left carries more of the score than one with 6, which is the right weighting for a
    per-match prediction."""
    p = phat(df, w, kappa)
    s, n = df["rest_starts"].to_numpy(), df["rest_n"].to_numpy()
    # sum over remaining matches of (p - y)^2  ==  s*(1-p)^2 + (n-s)*p^2
    tot = (s * (1 - p) ** 2 + (n - s) * p ** 2).sum()
    return float(tot / n.sum())


def best_setting(df):
    best, bs = None, np.inf
    for w in W_GRID:
        for kp in KAPPA_GRID:
            e = brier(df, w, kp)
            if e < bs:
                bs, best = e, (w, kp)
    return best, bs


def ridge(df, tol=0.005):
    """Every (w, kappa) within `tol` of the best, and the ratio they share.

    Exists because the grid search returns a point and the surface is a ridge: without
    this the study would report a precise-looking constant that the data does not
    support. The identified quantity is w/kappa — how much one realised match is worth
    against the capped prior mass — not either lever alone."""
    base = brier(df, *BASE)
    best = min(brier(df, w, kp) for w in W_GRID for kp in KAPPA_GRID)
    near = [(w, kp, (base - brier(df, w, kp)) / base)
            for kp in KAPPA_GRID for w in W_GRID
            if (base - brier(df, w, kp)) / base >= (base - best) / base - tol]
    ratios = [w / kp for w, kp, _ in near if np.isfinite(kp)]
    return {"base": base, "best_gain": (base - best) / base, "n_near": len(near),
            "n_grid": len(W_GRID) * len(KAPPA_GRID), "near": near,
            "ratio_lo": (min(ratios) if ratios else None),
            "ratio_hi": (max(ratios) if ratios else None),
            # the two usable single-lever corners
            "w_at_no_cap": max(((base - brier(df, w, np.inf)) / base, w)
                               for w in W_GRID),
            "kappa_at_w1": max(((base - brier(df, 1.0, kp)) / base, kp)
                               for kp in KAPPA_GRID)}


def evaluate(D, label=""):
    seasons = sorted(D["season"].unique())
    out = []
    for k in CUTOFFS:
        Dk = D[D["k"] == k]
        if len(Dk) < 50:
            continue
        per = []
        for s in seasons:
            tr, te = Dk[Dk["season"] != s], Dk[Dk["season"] == s]
            if len(tr) < 30 or len(te) < 20:
                continue
            (w, kp), _ = best_setting(tr)
            e_new = brier(te, w, kp)
            e_base = brier(te, *BASE)
            per.append({"season": s, "w": w, "kappa": kp, "new": e_new, "base": e_base,
                        "gain": (e_base - e_new) / e_base if e_base else 0.0,
                        "n": int(te["rest_n"].sum())})
        if not per:
            continue
        ps = pd.DataFrame(per)
        pooled_new = float((ps["new"] * ps["n"]).sum() / ps["n"].sum())
        pooled_base = float((ps["base"] * ps["n"]).sum() / ps["n"].sum())
        gain = (pooled_base - pooled_new) / pooled_base if pooled_base else 0.0
        (w_all, kp_all), _ = best_setting(Dk)
        out.append({"scope": label or "ALL", "k": k, "n_players": int(len(Dk)),
                    "n_matches": int(Dk["rest_n"].sum()), "folds": len(ps),
                    "w_fitted": w_all, "kappa_fitted": kp_all,
                    "w_loso_median": float(ps["w"].median()),
                    "kappa_loso_median": float(ps["kappa"].median()),
                    "brier_base": pooled_base, "brier_new": pooled_new, "gain": gain,
                    "n_positive": int((ps["gain"] > 0).sum()),
                    "_per": ps})
    return out


def main():
    print("=" * 78)
    print("START PRIOR STRENGTH — is W_MINUTES=1.0 against an uncapped prior right?")
    print("=" * 78)
    print(f"pre-registered: adopt only if Brier improves >= {MIN_GAIN:.0%} AND the sign "
          f"holds in >= {CONSISTENT} held-out seasons")
    print(f"baseline = the model as it stands: w={BASE[0]}, kappa={BASE[1]}")

    panel = fh.load_history(verbose=False)
    seasons = usable_seasons(panel)
    order = {s: i for i, s in enumerate(sorted(panel["season"].unique()))}
    evaluable = [s for s in seasons if any(order.get(t, -9) == order[s] - 1 for t in seasons)]
    print(f"\nseasons with a READ starts column: {seasons}")
    print(f"evaluable (previous season usable too): {evaluable}")
    if len(evaluable) < CONSISTENT:
        print("NOT ENOUGH EVALUABLE SEASONS — no claim made.")
        return 0

    D = build(panel, set(evaluable))
    print(f"player-season-cutoff rows: {len(D)}  ({D['player_code'].nunique()} players, "
          f"{int(D['rest_n'].sum()):,} scored matches)")

    res = evaluate(D)
    print("\n" + "-" * 78)
    print("PRIMARY — pooled Brier on remaining matches (lower is better)")
    print("-" * 78)
    print(f"{'k':>3} {'players':>8} {'matches':>9} {'w':>5} {'kappa':>6} "
          f"{'Brier now':>10} {'Brier new':>10} {'gain':>7} {'folds+':>7}  verdict")
    adopt, rows_csv = [], []
    for r in res:
        ok = (r["gain"] >= MIN_GAIN) and (r["n_positive"] >= CONSISTENT)
        v = "ADOPT" if ok else "null"
        if ok:
            adopt.append(r)
        kp = "inf" if not np.isfinite(r["kappa_loso_median"]) else f"{r['kappa_loso_median']:.0f}"
        print(f"{r['k']:>3} {r['n_players']:>8} {r['n_matches']:>9,} "
              f"{r['w_loso_median']:>5.1f} {kp:>6} {r['brier_base']:>10.4f} "
              f"{r['brier_new']:>10.4f} {r['gain']:>6.1%} "
              f"{r['n_positive']:>3}/{r['folds']:<3}  {v}")
        rows_csv.append({k: v2 for k, v2 in r.items() if k != "_per"} | {"verdict": v})

    # --- identifiability: is any single (w, kappa) actually pinned down? ---
    print("\n" + "-" * 78)
    print("IDENTIFIABILITY — is the fitted pair a peak or a ridge?")
    print("-" * 78)
    for k in CUTOFFS:
        Dk = D[D["k"] == k]
        if len(Dk) < 50:
            continue
        R = ridge(Dk)
        g_w, w_star = R["w_at_no_cap"]
        g_k, kp_star = R["kappa_at_w1"]
        print(f"  k={k:<2} best {R['best_gain']:.1%}; {R['n_near']} of {R['n_grid']} grid "
              f"points within 0.5pp of it -> a RIDGE, w/kappa in "
              f"[{R['ratio_lo']:.2f}, {R['ratio_hi']:.2f}]")
        print(f"        equivalent single-lever corners: keep kappa=inf and set w="
              f"{w_star:g} ({g_w:.1%}), or keep w=1 and cap kappa={kp_star:g} ({g_k:.1%})")

    # --- H-POS: is the fitted weight position-specific? ---
    print("\n" + "-" * 78)
    print("H-POS — does the right weight differ by position? (pre-registered, alpha=0.05)")
    print("-" * 78)
    bypos = {}
    for pos in ("GK", "DEF", "MID", "FWD"):
        Dp = D[D["pos"] == pos]
        if len(Dp) < 60:
            print(f"  {pos}: too few rows ({len(Dp)}) — not evaluated")
            continue
        rp = evaluate(Dp, label=pos)
        bypos[pos] = rp
        for r in rp:
            kp = "inf" if not np.isfinite(r["kappa_loso_median"]) else f"{r['kappa_loso_median']:.0f}"
            print(f"  {pos} k={r['k']:<2} w={r['w_loso_median']:>4.1f} kappa={kp:>4} "
                  f"Brier {r['brier_base']:.4f} -> {r['brier_new']:.4f} "
                  f"({r['gain']:+.1%}), folds+ {r['n_positive']}/{r['folds']}")
            rows_csv.append({k: v2 for k, v2 in r.items() if k != "_per"} | {"verdict": "hpos"})
    verdict_pos = "not evaluated"
    if "GK" in bypos and "FWD" in bypos:
        wins = 0, 0
        gk = {r["k"]: r["_per"] for r in bypos["GK"]}
        fw = {r["k"]: r["_per"] for r in bypos["FWD"]}
        common = sorted(set(gk) & set(fw))
        strictly, total = 0, 0
        for k in common:
            m = gk[k].merge(fw[k], on="season", suffixes=("_gk", "_fw"))
            strictly += int((m["w_gk"] > m["w_fw"]).sum())
            total += len(m)
        verdict_pos = (f"GK weight strictly above FWD in {strictly}/{total} "
                       f"season-cutoff comparisons")
        print(f"\n  -> {verdict_pos}")
        print(f"  -> H-POS {'SUPPORTED' if total and strictly == total else 'NOT SUPPORTED'} "
              f"(pre-registered as: strictly greater in EVERY held-out season)")
        if not (total and strictly == total):
            print("  -> NULL RECORDED. The weight `w` alone does not separate goalkeepers "
                  "from forwards.")
            print("     Post-hoc only, and NOT a rescue of the failed test: in the ratio "
                  "parameterisation")
            print("     w/kappa the positions do look ordered (GK ~0.30-0.40 against DEF "
                  "~0.15-0.20).")
            print("     That comparison was not pre-registered, the ratio is the identified "
                  "quantity and w")
            print("     is not, so it is a hypothesis for a future study to register — not "
                  "a finding here.")

    pd.DataFrame(rows_csv).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")

    print("\n" + "=" * 78)
    if adopt:
        print("RESULT: the decision rule is MET at k = " +
              ", ".join(str(r["k"]) for r in adopt) + ".")
        for r in adopt:
            kp = "inf" if not np.isfinite(r["kappa_loso_median"]) else f"{r['kappa_loso_median']:.0f}"
            print(f"          k={r['k']:<3} w={r['w_loso_median']:.1f}  kappa={kp}  "
                  f"({r['gain']:.1%} Brier reduction)")
        print()
        print("        DO NOT PIN THE PAIR ABOVE. w and kappa are not separately identified")
        print("        (see IDENTIFIABILITY): the fitted pair is one arbitrary point on a")
        print("        flat ridge. What the data supports is the RATIO and the direction —")
        print("        the current corner (w=1, kappa=inf) is dominated by ~11%. Implement")
        print("        it as EITHER a prior cap near 4 pseudo-matches at the current weight")
        print("        (the interior optimum is kappa 3-5 across cutoffs),")
        print("        OR an uncapped prior with a realised match weighted ~8x. They score")
        print("        the same and this study cannot choose between them.")
        print("        Ship behind a flag, off by default, then A/B the board on scored")
        print("        gameweeks. NOTE the approximation: this fits the previous-season")
        print("        prior, not the two-season pooled production prior — re-fit against")
        print("        the real prior before pinning any constant.")
    else:
        print("RESULT: NULL. No (w, kappa) beats the current setting by the pre-registered")
        print("        margin. `W_MINUTES = 1.0` against an uncapped prior stands, and the")
        print("        Dubravka case is then a PRESENTATION problem, correctly handled by")
        print("        holding never-started players out of the pool, not a constant to")
        print("        retune. Recorded so it is not proposed again under a new name.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
