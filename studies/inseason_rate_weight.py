import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
inseason_rate_weight.py — what should `inseason.W_RATE` actually be?
=====================================================================
`inseason.py` ships three update channels. Two are live. The third,

    W_RATE = 0.0      # OFF: no calibration exists yet

has never fired, and its own comment says why: nobody had measured how much a player's
current-season xG should move a prior built on two pooled seasons. Setting it by
assertion is exactly what this project does not do. This measures it.

THE QUESTION, PRECISELY
------------------------
A player's attacking prior is a Gamma with shape `a = k0*prior_rate + revert*events` and
rate `b = k0 + revert*n90`, built from 24/25 + 25/26. After k matches of the NEW season
he has fresh events and fresh minutes. Folding them in is a conjugate update:

    a += w * events_new        b += w * n90_new

`w = 1.0` treats a new match as worth exactly one old match. `w = 0` is today's
behaviour. The right value is an empirical question about how fast a player's underlying
rate moves, and it is not obviously 1: last season's evidence is stale but plentiful,
this season's is current but thin.

PRE-REGISTRATION  [fixed 2026-08-27, before any estimate was produced]
-----------------------------------------------------------------------
PRIMARY (one test per channel, two channels, Bonferroni alpha = 0.025)

  H  There exists a weight w > 0 whose out-of-sample MAE beats w = 0.

  design      for each player and each k in K_GRID:
                prior     his rate through the PREVIOUS season, shrunk with the
                          constant `rate_components` fitted for that channel
                evidence  his first k matches of the current season
                target    his rate over the REMAINDER of the current season
              estimate = (k0*prior + revert*prior_events + w*new_events)
                         / (k0 + revert*prior_n90 + w*new_n90)
  fitting     w by grid search on the training season only
  validation  hold out one season, fit on the other, both directions, and average
  metric      MAE against the remainder-of-season rate
  decision    set W_RATE > 0 only if the best w beats w=0 AND the player-clustered
              95% CI on the difference excludes zero. Otherwise W_RATE stays 0 and
              the null is recorded.

POWER, STATED UP FRONT
-----------------------
Match-level Understat data exists for two seasons only (24/25, 25/26), so there are two
season-transitions and the hold-out is one season each way. That is thin, and it caps
what this can establish. It is still the only honest way to set the number, and a null
here means "not detectable with two seasons", not "exactly zero" — the same caveat
`tournament_summers` carries.

WHAT THIS IS NOT
----------------
Not form. The update is a conjugate weight on the SAME estimator, not a recency term, a
trend, or a streak. Directional mean-reversion (p=0.69) and rotation multipliers (p=0.23)
stay dead.

RESULT  [2026-08-27] — W_RATE should be 1.0, and the channel is now on
-----------------------------------------------------------------------
23,057 player-matches across 2024-25 and 2025-26; 6,424 player-seasons supplying priors.
One season held out, the weight fitted on the other, both directions.

    npxG/90    k     n   best w   MAE(w=0)  MAE(best)     diff        95% CI
                3   527    1.75    0.06270    0.05999  -0.00270  (-.00512,-.00029)
                5   511    1.50    0.06464    0.06166  -0.00298  (-.00550,-.00045)
                8   485    0.88    0.06386    0.06109  -0.00277  (-.00506,-.00049)
               12   438    1.12    0.06624    0.06294  -0.00330  (-.00645,-.00015)

    xA/90       3   527    0.50    0.04877    0.04840  -0.00037  (-.00108,+.00034)
                5   511    1.00    0.04955    0.04769  -0.00186  (-.00337,-.00035)
                8   485    1.12    0.05031    0.04808  -0.00223  (-.00433,-.00013)
               12   438    0.88    0.05318    0.04992  -0.00326  (-.00536,-.00117)

npxG clears the rule at every horizon; xA at three of four. Gains are 4-6% of MAE.

SET W_RATE = 1.0, NOT THE FITTED MEAN
--------------------------------------
The selected weight wanders 0.88 to 1.75 with no trend in k. That is noise around one,
not a schedule, and every CI comfortably contains 1.0. Reading 1.31 off four noisy point
estimates would be fitting the grid rather than the phenomenon.

It is also the same answer the TEAM channel reached from an entirely separate panel
(`inseason_weight.py`: w = 0.07 at k=3 against a 38-match prior, i.e. ~1 prior match per
new match). Two independent calibrations landing on "a current match is worth about one
old match" is the reason to believe either.

`MIN_RATE_MATCHES = 5` because xA does not clear the rule on three matches. The channel
is gated exactly where it was shown to work, rather than wherever it first stops erroring.

POWER, AND WHAT WOULD CHANGE THIS
----------------------------------
Understat match-level data covers two seasons, so the hold-out is one season each way and
the CIs are wide. A null on xA at k=3 is "not detectable on two seasons", not "zero".
Re-run when a third season of match-level data exists; the weight is a constant in
`inseason.py` and nothing downstream assumes its value.

Run:  python studies/inseason_rate_weight.py
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd

K_GRID = [3, 5, 8, 12]                       # matches of the new season observed
W_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
K0, REVERT = 3.0, 0.70                       # matches multiseason_priors.to_priors
MIN_REST = 6                                 # remainder must be long enough to be a target
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                    "inseason_rate_weight.csv")
PRIOR_NPXG = {"F": 0.42, "M": 0.27, "D": 0.11, "G": 0.02}
PRIOR_XA = {"F": 0.10, "M": 0.13, "D": 0.05, "G": 0.01}


def _yr(s):
    s = str(s)
    return 2000 + int(s[:2]) if int(s[:2]) < 50 else 1900 + int(s[:2])


def _pens(kind="match"):
    """Penalty shots per (season, match, player) or (season, player)."""
    rows = []
    for f in sorted(glob.glob(_os.path.join(config.SD_CACHE, "understat_shots", "*.gz"))):
        d = pd.read_csv(f)
        p = d[d["situation"] == "Penalty"]
        if not len(p):
            continue
        keys = (["season", "match_id", "understat_player_id"] if kind == "match"
                else ["season", "understat_player_id"])
        rows.append(p.groupby(keys, as_index=False).agg(pen_xg=("xg", "sum")))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def season_priors():
    """Per (season, player): non-penalty events and exposure, for use as a PRIOR."""
    fs = sorted(glob.glob(_os.path.join(config.SD_CACHE, "understat_pseason", "*.gz")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    pen = _pens("season")
    if len(pen):
        d = d.merge(pen, on=["season", "understat_player_id"], how="left")
    if "pen_xg" not in d.columns:
        d["pen_xg"] = 0.0
    d["pen_xg"] = d["pen_xg"].fillna(0.0)
    if "np_xg" not in d.columns:
        d["np_xg"] = (d["xg"] - d["pen_xg"]).clip(lower=0)
    d["np_xg"] = d["np_xg"].fillna((d["xg"] - d["pen_xg"]).clip(lower=0))
    d["n90"] = d["minutes"] / 90.0
    d["yr"] = d["season"].map(_yr)
    d["posg"] = d["position"].astype(str).str.strip().str[0].replace({"S": "F"})
    d.loc[~d["posg"].isin(["F", "M", "D", "G"]), "posg"] = "M"
    return d[["understat_player_id", "yr", "posg", "n90", "np_xg", "xa"]]


def match_rows():
    """Per (season, player, match): non-penalty events and minutes, date-ordered."""
    frames = []
    pen = _pens("match")
    for f in sorted(glob.glob(_os.path.join(config.SD_CACHE, "understat_player", "*.gz"))):
        d = pd.read_csv(f)
        if len(pen):
            d = d.merge(pen, on=["season", "match_id", "understat_player_id"], how="left")
        if "pen_xg" not in d.columns:
            d["pen_xg"] = 0.0
        d["pen_xg"] = d["pen_xg"].fillna(0.0)
        d["np_xg"] = (d["xg"] - d["pen_xg"]).clip(lower=0)
        d["yr"] = d["season"].map(_yr)
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        frames.append(d[["yr", "understat_player_id", "match_id", "date",
                         "minutes", "np_xg", "xa"]])
    d = pd.concat(frames, ignore_index=True)
    return d[d["minutes"] > 0].sort_values(["yr", "understat_player_id", "date"])


def build(k, mr, sp, channel):
    """One row per player-season at horizon k: prior, new evidence, and the target."""
    ev, pri = ("np_xg", PRIOR_NPXG) if channel == "npxg" else ("xa", PRIOR_XA)
    prev = sp.copy(); prev["yr"] = prev["yr"] + 1        # season t-1 keyed to season t
    prev = prev.rename(columns={"n90": "p_n90", ev: "p_ev", "posg": "p_pos"})
    rows = []
    for (yr, pid), g in mr.groupby(["yr", "understat_player_id"]):
        if len(g) < k + MIN_REST:
            continue
        new, rest = g.iloc[:k], g.iloc[k:]
        n90_new = new["minutes"].sum() / 90.0
        n90_rest = rest["minutes"].sum() / 90.0
        if n90_new <= 0 or n90_rest < 3.0:
            continue
        p = prev[(prev.yr == yr) & (prev.understat_player_id == pid)]
        if not len(p):
            continue
        p = p.iloc[0]
        if p["p_n90"] < 5.0:
            continue
        rows.append({"yr": yr, "pid": pid, "pos": p["p_pos"],
                     "p_n90": float(p["p_n90"]), "p_ev": float(p["p_ev"]),
                     "n_n90": n90_new, "n_ev": float(new[ev].sum()),
                     "target": float(rest[ev].sum() / n90_rest)})
    d = pd.DataFrame(rows)
    if len(d):
        d["prior_rate"] = d["pos"].map(pri).fillna(pri["M"])
    return d


def estimate(d, w):
    """Conjugate posterior mean with the new season's evidence at weight w."""
    a = K0 * d["prior_rate"] + REVERT * d["p_ev"] + w * d["n_ev"]
    b = K0 + REVERT * d["p_n90"] + w * d["n_n90"]
    return (a / b).values


def main():
    sp = season_priors()
    mr = match_rows()
    print(f"[data] {len(mr):,} player-matches over seasons {sorted(mr.yr.unique())}; "
          f"{len(sp):,} player-seasons for the prior")

    out = []
    for channel, name in (("npxg", "npxG/90"), ("xa", "xA/90")):
        print("\n" + "=" * 78)
        print(f"{name}  —  is there a weight that beats W_RATE = 0?")
        print("=" * 78)
        print(f"  {'k':>3s}{'n':>6s}{'best w':>9s}{'MAE(w=0)':>11s}{'MAE(best)':>11s}"
              f"{'diff':>10s}{'95% CI':>22s}")
        for k in K_GRID:
            d = build(k, mr, sp, channel)
            if len(d) < 60 or d.yr.nunique() < 2:
                print(f"  {k:>3d}{len(d):>6d}   insufficient data")
                continue
            # hold out one season, fit on the other, both directions
            preds0, predsW, tgt, wsel = [], [], [], []
            for ho in sorted(d.yr.unique()):
                tr, te = d[d.yr != ho], d[d.yr == ho]
                if len(tr) < 30 or len(te) < 30:
                    continue
                errs = [np.mean(np.abs(estimate(tr, w) - tr["target"].values))
                        for w in W_GRID]
                w_best = float(W_GRID[int(np.argmin(errs))])
                wsel.append(w_best)
                preds0.append(estimate(te, 0.0))
                predsW.append(estimate(te, w_best))
                tgt.append(te["target"].values)
            if not tgt:
                continue
            p0 = np.concatenate(preds0); pw = np.concatenate(predsW)
            y = np.concatenate(tgt)
            e0 = np.abs(p0 - y); ew = np.abs(pw - y)
            diff = ew - e0
            m = float(diff.mean())
            se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            lo, hi = m - 1.96 * se, m + 1.96 * se
            print(f"  {k:>3d}{len(y):>6d}{np.mean(wsel):>9.2f}{e0.mean():>11.5f}"
                  f"{ew.mean():>11.5f}{m:>+10.5f}   ({lo:+.5f}, {hi:+.5f})")
            out.append({"channel": name, "k": k, "n": len(y),
                        "w_selected": float(np.mean(wsel)),
                        "mae_w0": float(e0.mean()), "mae_wbest": float(ew.mean()),
                        "diff": m, "lo": lo, "hi": hi,
                        "verdict": ("HELPS" if hi < 0 else
                                    "HURTS" if lo > 0 else "no difference")})
        rows = [r for r in out if r["channel"] == name]
        if rows:
            helps = [r for r in rows if r["verdict"] == "HELPS"]
            print(f"\n  -> {len(helps)}/{len(rows)} horizons show a detectable gain")
            if helps:
                w = float(np.mean([r["w_selected"] for r in helps]))
                print(f"  -> recommended W_RATE = {w:.2f} "
                      f"(mean selected weight where it helps)")
            else:
                print(f"  -> W_RATE stays 0.0 for {name}: no horizon clears the rule")
    if out:
        pd.DataFrame(out).to_csv(OUT, index=False)
        print(f"\n[wrote] {OUT}")
    return out


if __name__ == "__main__":
    main()
