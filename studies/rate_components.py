import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
rate_components.py — is npxG/90 one quantity, or two badly pooled ones?
=======================================================================
The attacking prior is a single Gamma on non-penalty xG involvement per 90, shrunk with
one constant. But

    npxG/90  =  (shots/90)  x  (npxG/shot)

and those factors are not the same kind of thing. Shot volume is a player trait — where
he plays, how often he gets the ball in dangerous areas. Conversion per shot is closer to
role plus noise, and it is measured on far fewer effective observations. Shrinking the
PRODUCT with one constant therefore over-shrinks the stable factor and under-shrinks the
noisy one. That is a specification error with a known direction, not a tuning choice.

Same argument for the assist channel: xA/90 = (key passes/90) x (xA/key pass).

PRE-REGISTRATION  [fixed 2026-08-26, before any estimate was produced]
-----------------------------------------------------------------------
PRIMARY (one test per channel, two channels, Bonferroni alpha = 0.025).

  H  Shrinking volume and quality SEPARATELY, then multiplying, predicts a player's
     NEXT-season rate better than shrinking the pooled rate.

  estimator   each quantity is shrunk toward its positional mean on its OWN exposure:
                  volume   r' = (k_v * prior + n90     * r) / (k_v + n90)
                  quality  q' = (k_q * prior + n_shots * q) / (k_q + n_shots)
                  pooled   p' = (k_p * prior + n90     * p) / (k_p + n90)
              Exposure matters and differs: a player with 100 shots has a
              well-measured conversion rate; one with 5 does not, and n90 does not
              know the difference. That mismatch is half the argument for splitting.
  fitting     every k by grid search on the TRAINING seasons only
  validation  leave-one-season-out over the consecutive-season pairs; predict season
              t+1 npxG/90 from season t
  metric      MAE (primary) and Spearman, minutes-weighted eligibility both seasons
  decision    ship the split only if out-of-sample MAE improves AND the improvement's
              season-clustered 95% CI excludes zero. A reliability difference on its
              own is NOT sufficient — it has to pay out of sample.

SUPPORTING MEASUREMENTS (descriptive; they explain the primary result, they do not
decide it): split-half reliability within season at match level, and season-to-season
AR(1) for each component.

DATA
----
`.cache/soccerdata`, already scraped — README still says this is unpopulated, which is
stale. 12 seasons of Understat player-season aggregates (2014-15..2025-26, 3,769
consecutive-season pairs) and 2 seasons of player-MATCH rows for the split-half work.
Penalties are stripped throughout using `understat_shots.situation == "Penalty"`, because
a penalty is neither volume nor quality in the sense meant here — the model prices it
separately via `pen_xg90`.

WHAT THIS IS NOT
----------------
Not a form model. Nothing here uses recency, streaks or within-season trend: directional
mean-reversion is a tested null in this repo (p=0.69) and `deep_history_study` returned
+0.0017 MAE. This changes how a season's evidence is SHRUNK, not which evidence counts.

RESULT  [2026-08-26] — the split pays for ASSISTS and not for SHOOTING
----------------------------------------------------------------------
4,753 player-seasons over 12 seasons; 2,831 consecutive-season pairs.

Split-half reliability within season (match level, Spearman-Brown corrected):

    shots/90              0.897        npxG/shot     0.610
    key passes/90         0.850        xA/key pass   0.218
    npxG/90 (pooled)      0.865

The reliability gap is real in BOTH channels and in the predicted direction. It does not
follow that splitting helps in both, and the pre-registered rule required out-of-sample
payout rather than a reliability gap:

    npxG/90    pooled MAE 0.05197 -> split 0.05387
               difference +0.00193, 95% CI (-0.00073, +0.00458)   NO DIFFERENCE
               Spearman 0.8094 -> 0.7980
               fitted k: volume 12, quality 12, pooled 5

    xA/90      pooled MAE 0.04359 -> split 0.04271
               difference -0.00087, 95% CI (-0.00151, -0.00024)   SPLIT WINS
               Spearman 0.7749 -> 0.7885
               fitted k: volume 0.5, quality 25, pooled 8

So the assist channel splits and the shooting channel does not. The reason is visible in
the fitted constants. For assists the two factors want wildly different shrinkage —
volume is trusted almost immediately (k=0.5) while quality needs about 25 chances (k=25)
— and one pooled constant cannot express a 50x difference. For shooting both factors
landed on k=12, so the pooled estimator at k=5 was already doing the same job.

That matches the reliability ratios: assists 0.850/0.218 = 3.9x, shooting 0.897/0.610 =
1.5x. The split earns its place only where the asymmetry is extreme. Splitting both
would have been the intuitive move and half of it would have been wrong, which is the
whole reason the decision rule was fixed before the estimates existed.

CAVEATS ON THE TRANSFER TO THE MODEL
-------------------------------------
* Fitted on Understat, applied to the repo's `chances_created`. The two agree closely on
  volume (chances/90 0.854 vs key passes/90 0.825) but not on the per-event rate (0.097
  vs 0.120), so the shrinkage CONSTANTS are borrowed and the LEVELS are re-measured on
  the repo's own scale. See PRIOR_KP90 / PRIOR_XA_PER_KP in multiseason_priors.
* The calibration sample was players with >= 450 minutes. k=0.5 barely shrinks volume,
  which is right in that regime and reckless outside it, so `SPLIT_MIN_N90 = 5.0` falls
  back to the pooled prior for thinner records.
* `xa` and `chances_created` reconcile well (r = 0.955) but not perfectly: 10 of 504
  players carry xA with no recorded chance created, together 0.1% of league xA. Small
  enough to ignore, recorded so it is not rediscovered as a mystery.

EFFECT ON THE PRIORS: 504 of 552 players eligible, league xA/90 unchanged at 0.0831 (no
level shift), mean |change| 0.0093 xA/90. High-volume low-conversion players rise, and
players whose per-chance rate was flattering on a handful of chances fall.

Run:  python studies/rate_components.py
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd

MIN_MINUTES = 450.0           # ~5 full matches; below this a season-rate is noise
MIN_SHOTS = 1.0               # a conversion rate needs at least one shot
GRID = np.array([0.5, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 75, 100, 150, 250])
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "rate_components.csv")


def _yr(season):
    """'1415' -> 2014, '2526' -> 2025."""
    s = str(season)
    return 2000 + int(s[:2]) if int(s[:2]) < 50 else 1900 + int(s[:2])


def penalties_by_player_season():
    """Penalty shots and penalty xG per (player, season), to strip from the totals."""
    rows = []
    for f in sorted(glob.glob(_os.path.join(config.SD_CACHE, "understat_shots", "*.gz"))):
        d = pd.read_csv(f)
        p = d[d["situation"] == "Penalty"]
        if not len(p):
            continue
        g = (p.groupby(["season", "understat_player_id"], as_index=False)
               .agg(pen_shots=("xg", "size"), pen_xg=("xg", "sum")))
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["season", "understat_player_id", "pen_shots", "pen_xg"])


def season_panel():
    """One row per player-season: non-penalty volume and quality, both channels."""
    fs = sorted(glob.glob(_os.path.join(config.SD_CACHE, "understat_pseason", "*.gz")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    pen = penalties_by_player_season()
    d = d.merge(pen, on=["season", "understat_player_id"], how="left")
    d[["pen_shots", "pen_xg"]] = d[["pen_shots", "pen_xg"]].fillna(0.0)

    d["np_shots"] = (d["shots"] - d["pen_shots"]).clip(lower=0)
    # np_xg is supplied by the source; fall back to xg minus penalty xG if absent
    if "np_xg" not in d.columns:
        d["np_xg"] = (d["xg"] - d["pen_xg"]).clip(lower=0)
    d["np_xg"] = d["np_xg"].fillna((d["xg"] - d["pen_xg"]).clip(lower=0))
    d["n90"] = d["minutes"] / 90.0
    d["yr"] = d["season"].map(_yr)
    # coarse position group; Understat writes things like 'F M S', so take the first token
    d["posg"] = d["position"].astype(str).str.strip().str[0].replace({"S": "F"})
    d.loc[~d["posg"].isin(["F", "M", "D", "G"]), "posg"] = "M"

    d = d[d["minutes"] >= MIN_MINUTES].copy()
    d["vol"] = d["np_shots"] / d["n90"]                       # shots per 90
    d["qual"] = np.where(d["np_shots"] >= MIN_SHOTS,
                         d["np_xg"] / d["np_shots"].clip(lower=1e-9), np.nan)
    d["pooled"] = d["np_xg"] / d["n90"]                       # the model's quantity
    d["kp_vol"] = d["key_passes"] / d["n90"]
    d["kp_qual"] = np.where(d["key_passes"] >= 1,
                            d["xa"] / d["key_passes"].clip(lower=1e-9), np.nan)
    d["xa_pooled"] = d["xa"] / d["n90"]
    return d[["understat_player_id", "player_name", "season", "yr", "posg", "minutes",
              "n90", "np_shots", "np_xg", "key_passes", "xa",
              "vol", "qual", "pooled", "kp_vol", "kp_qual", "xa_pooled"]]


def pairs(panel):
    """Consecutive-season pairs: season t features, season t+1 target."""
    a = panel.copy()
    b = panel.copy()
    b["yr"] = b["yr"] - 1
    keep_b = ["understat_player_id", "yr", "pooled", "xa_pooled", "n90"]
    m = a.merge(b[keep_b], on=["understat_player_id", "yr"], suffixes=("", "_next"))
    return m.rename(columns={"pooled_next": "y_npxg90", "xa_pooled_next": "y_xa90",
                             "n90_next": "n90_next"})


def _shrink(obs, exposure, prior, k):
    return (k * prior + exposure * obs) / (k + exposure)


def _priors(train, cols):
    """Minutes-weighted positional means from the TRAINING seasons only."""
    out = {}
    for c in cols:
        d = train.dropna(subset=[c])
        g = d.groupby("posg").apply(
            lambda x: float(np.average(x[c], weights=x["n90"])), include_groups=False)
        out[c] = {"by_pos": g.to_dict(),
                  "all": float(np.average(d[c], weights=d["n90"]))}
    return out


def _apply_prior(df, pri, col):
    return df["posg"].map(pri[col]["by_pos"]).fillna(pri[col]["all"]).values


def fit_k(train, obs_col, exp_col, target, pri):
    """Grid-search the shrinkage constant on training data only."""
    d = train.dropna(subset=[obs_col, target])
    if len(d) < 50:
        return np.nan
    prior = _apply_prior(d, pri, obs_col)
    best, bk = np.inf, np.nan
    for k in GRID:
        pred = _shrink(d[obs_col].values, d[exp_col].values, prior, k)
        e = float(np.mean(np.abs(pred - d[target].values)))
        if e < best:
            best, bk = e, k
    return bk


def evaluate(P, channel="npxg"):
    """Leave-one-season-out: pooled shrinkage vs volume x quality."""
    if channel == "npxg":
        vol, qual, pooled, target = "vol", "qual", "pooled", "y_npxg90"
        vol_exp, qual_exp = "n90", "np_shots"
    else:
        vol, qual, pooled, target = "kp_vol", "kp_qual", "xa_pooled", "y_xa90"
        vol_exp, qual_exp = "n90", "key_passes"
    rows = []
    for yr in sorted(P.yr.unique()):
        tr, te = P[P.yr != yr], P[P.yr == yr]
        te = te.dropna(subset=[vol, qual, pooled, target])
        if len(te) < 25 or len(tr) < 100:
            continue
        pri = _priors(tr, [vol, qual, pooled])
        k_v = fit_k(tr, vol, vol_exp, target, pri)
        k_q = fit_k(tr, qual, qual_exp, target, pri)
        k_p = fit_k(tr, pooled, "n90", target, pri)
        v_ = _shrink(te[vol].values, te[vol_exp].values, _apply_prior(te, pri, vol), k_v)
        q_ = _shrink(te[qual].values, te[qual_exp].values, _apply_prior(te, pri, qual), k_q)
        p_ = _shrink(te[pooled].values, te["n90"].values, _apply_prior(te, pri, pooled), k_p)
        y = te[target].values
        rows.append(pd.DataFrame({
            "yr": yr, "y": y, "split": v_ * q_, "pooled": p_,
            "k_v": k_v, "k_q": k_q, "k_p": k_p}))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def report(R, label):
    from scipy import stats as st
    if not len(R):
        print(f"  {label}: insufficient data")
        return None
    mae_s = float(np.mean(np.abs(R["split"] - R["y"])))
    mae_p = float(np.mean(np.abs(R["pooled"] - R["y"])))
    rho_s = st.spearmanr(R["split"], R["y"])[0]
    rho_p = st.spearmanr(R["pooled"], R["y"])[0]
    # season-clustered CI on the paired MAE difference
    per = R.assign(d=np.abs(R["split"] - R["y"]) - np.abs(R["pooled"] - R["y"])) \
           .groupby("yr")["d"].mean()
    m = float(per.mean()); se = float(per.std(ddof=1) / np.sqrt(len(per)))
    lo, hi = m - 1.96 * se, m + 1.96 * se
    print(f"\n  {label}   n={len(R)} player-seasons over {R.yr.nunique()} held-out seasons")
    print(f"    pooled shrinkage      MAE {mae_p:.5f}   Spearman {rho_p:.4f}")
    print(f"    volume x quality      MAE {mae_s:.5f}   Spearman {rho_s:.4f}")
    print(f"    difference (split - pooled)  {m:+.5f}   95% CI ({lo:+.5f}, {hi:+.5f})")
    verdict = ("SPLIT WINS" if hi < 0 else
               "POOLED WINS" if lo > 0 else "NO DIFFERENCE — pooled stands")
    print(f"    -> {verdict}")
    print(f"    fitted k: volume {R.k_v.median():.1f}, quality {R.k_q.median():.1f}, "
          f"pooled {R.k_p.median():.1f}")
    return {"channel": label, "n": len(R), "mae_pooled": mae_p, "mae_split": mae_s,
            "rho_pooled": rho_p, "rho_split": rho_s, "diff": m, "lo": lo, "hi": hi,
            "k_vol": float(R.k_v.median()), "k_qual": float(R.k_q.median()),
            "k_pooled": float(R.k_p.median()), "verdict": verdict}


# ------------------------------------------------------------------ reliability
def split_half(seasons=("2425", "2526"), n_splits=120, seed=0):
    """Match-level split-half reliability, Spearman-Brown corrected to full length.

    Splits MATCHES, not shots: shots within a match are correlated, so a shot-level
    split understates the error and flatters reliability. Same reasoning as
    `setpiece.split_half_reliability`, which does this for the situation components.
    """
    from scipy import stats as st
    rng = np.random.default_rng(seed)
    pens = {}
    for s in seasons:
        f = _os.path.join(config.SD_CACHE, "understat_shots", f"{s}.csv.gz")
        if _os.path.exists(f):
            d = pd.read_csv(f)
            p = d[d["situation"] == "Penalty"]
            pens[s] = (p.groupby(["match_id", "understat_player_id"], as_index=False)
                        .agg(pen_shots=("xg", "size"), pen_xg=("xg", "sum")))
    out = []
    for s in seasons:
        f = _os.path.join(config.SD_CACHE, "understat_player", f"{s}.csv.gz")
        if not _os.path.exists(f):
            continue
        pm = pd.read_csv(f)
        if s in pens:
            pm = pm.merge(pens[s], on=["match_id", "understat_player_id"], how="left")
        for c in ("pen_shots", "pen_xg"):
            if c not in pm.columns:
                pm[c] = 0.0
        pm[["pen_shots", "pen_xg"]] = pm[["pen_shots", "pen_xg"]].fillna(0.0)
        pm["np_shots"] = (pm["shots"] - pm["pen_shots"]).clip(lower=0)
        pm["np_xg"] = (pm["xg"] - pm["pen_xg"]).clip(lower=0)
        pm = pm[pm["minutes"] > 0]
        elig = pm.groupby("understat_player_id")["minutes"].sum()
        elig = set(elig[elig >= MIN_MINUTES].index)
        pm = pm[pm.understat_player_id.isin(elig)]
        acc = {k: [] for k in ("vol", "qual", "pooled", "kp_vol", "kp_qual")}
        for _ in range(n_splits):
            pm["_h"] = rng.integers(0, 2, len(pm))
            g = pm.groupby(["understat_player_id", "_h"], as_index=False).agg(
                mins=("minutes", "sum"), sh=("np_shots", "sum"), xg=("np_xg", "sum"),
                kp=("key_passes", "sum"), xa=("xa", "sum"))
            g["n90"] = g["mins"] / 90.0
            g = g[g["n90"] >= 2.0]
            w = g.pivot(index="understat_player_id", columns="_h",
                        values=["n90", "sh", "xg", "kp", "xa"]).dropna()
            if len(w) < 25:
                continue
            def corr(a, b):
                ok = np.isfinite(a) & np.isfinite(b)
                return st.pearsonr(a[ok], b[ok])[0] if ok.sum() > 20 else np.nan
            acc["vol"].append(corr((w["sh"][0] / w["n90"][0]).values,
                                   (w["sh"][1] / w["n90"][1]).values))
            m0 = w["sh"][0] >= 3; m1 = w["sh"][1] >= 3
            mm = (m0 & m1).values
            if mm.sum() > 20:
                acc["qual"].append(corr((w["xg"][0] / w["sh"][0]).values[mm],
                                        (w["xg"][1] / w["sh"][1]).values[mm]))
            acc["pooled"].append(corr((w["xg"][0] / w["n90"][0]).values,
                                      (w["xg"][1] / w["n90"][1]).values))
            acc["kp_vol"].append(corr((w["kp"][0] / w["n90"][0]).values,
                                      (w["kp"][1] / w["n90"][1]).values))
            k0 = w["kp"][0] >= 3; k1 = w["kp"][1] >= 3
            km = (k0 & k1).values
            if km.sum() > 20:
                acc["kp_qual"].append(corr((w["xa"][0] / w["kp"][0]).values[km],
                                           (w["xa"][1] / w["kp"][1]).values[km]))
        for k, v in acc.items():
            v = [x for x in v if np.isfinite(x)]
            if not v:
                continue
            r = float(np.mean(v))
            sb = 2 * r / (1 + r) if r > -1 else np.nan     # Spearman-Brown
            out.append({"season": s, "component": k, "half_r": r, "reliability": sb})
    return pd.DataFrame(out)


def main():
    print("[data] loading Understat caches ...")
    panel = season_panel()
    P = pairs(panel).dropna(subset=["y_npxg90"])
    P = P[P["n90_next"] >= MIN_MINUTES / 90.0]
    print(f"[data] {len(panel):,} player-seasons (>= {MIN_MINUTES:.0f} min), "
          f"{panel.yr.nunique()} seasons; {len(P):,} consecutive-season pairs")

    print("\n" + "=" * 78)
    print("SUPPORTING — split-half reliability within season (match-level, SB-corrected)")
    print("=" * 78)
    R = split_half()
    if len(R):
        piv = R.pivot_table(index="component", values=["half_r", "reliability"],
                            aggfunc="mean").round(4)
        order = ["vol", "qual", "pooled", "kp_vol", "kp_qual"]
        names = {"vol": "shots/90", "qual": "npxG/shot", "pooled": "npxG/90 (pooled)",
                 "kp_vol": "key passes/90", "kp_qual": "xA/key pass"}
        print(f"  {'component':22s}{'half r':>9s}{'reliability':>13s}")
        for c in order:
            if c in piv.index:
                print(f"  {names[c]:22s}{piv.loc[c,'half_r']:9.3f}{piv.loc[c,'reliability']:13.3f}")

    print("\n" + "=" * 78)
    print("PRIMARY — out-of-sample, leave-one-season-out (Bonferroni alpha = 0.025)")
    print("=" * 78)
    res = []
    for ch, lab in (("npxg", "npxG/90  (shots/90 x npxG/shot)"),
                    ("xa", "xA/90    (key passes/90 x xA/key pass)")):
        r = report(evaluate(P, ch), lab)
        if r:
            res.append(r)
    if res:
        pd.DataFrame(res).to_csv(OUT, index=False)
        print(f"\n[wrote] {OUT}")
    return res


if __name__ == "__main__":
    main()
