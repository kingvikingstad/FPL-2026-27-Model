"""
fm_priors.py — Football Manager attributes as Bayesian priors
=============================================================
Solves the model's blind spot: players with NO Premier League history — promoted
squads (Coventry, Hull, Ipswich) and foreign signings. FM's database rates every
professional worldwide on the same absolute 1-20 attribute scale, so it covers
exactly that population.

THE DESIGN PRINCIPLE (why this isn't just hand-weighting attributes)
--------------------------------------------------------------------
We never assert "Finishing is worth X". Instead:

  1. CALIBRATE on the OVERLAP — players who appear in BOTH the FM export and the
     25/26 FPL data (~300-400 established PL players). Regress their OBSERVED
     per-90 rates on their FM attributes:
         log(npxGI/90)  ~ Finishing, OffTheBall, Anticipation, Composure, ...
         log(xA/90)     ~ Passing, Vision, Crossing, Technique, ...
         log(DefCon/90) ~ Tackling, Marking, Positioning, WorkRate, Aggression
         start_rate     ~ CurrentAbility, NaturalFitness, Stamina
     This produces an EMPIRICALLY FITTED attribute -> rate mapping.

  2. APPLY that fitted mapping to cold-start players to get a prior mean.

  3. WEIGHT it correctly. FM informs the PRIOR only; observed minutes are the
     likelihood. The FM prior's influence decays as ~ K/(K + minutes), so it
     dominates for a promoted-team player with 0 PL minutes and is irrelevant
     for a 3,000-minute incumbent. That is the whole point of doing this in a
     Bayesian frame rather than blending point estimates.

  4. DISCOUNT for league step-up. Championship/second-tier players' output does
     not transfer 1:1 to the Premier League; `league_factor` applies a haircut
     (default 0.80 attacking, calibrate if you have promoted-player history).

GETTING THE DATA
----------------
Export from your own copy of Football Manager (recommended, clean and legitimate):
  Scouting -> Players -> set filters -> select attribute columns -> Print/Export
  to Web Page (.html) or Text. Then `load_fm_export(path)`.
A CSV with the same column names works too. Community sites (e.g. sortitoutsi)
browse the same database but are HTML-per-player and are SI/Sega IP — prefer
your own in-game export.
"""
from __future__ import annotations
import numpy as np, pandas as pd

# ---------------------------------------------------------------------------
# Attribute groups (FM names; loader tolerates common abbreviations)
# ---------------------------------------------------------------------------
ATTR_GOAL   = ["Finishing", "OffTheBall", "Anticipation", "Composure", "Technique", "Acceleration"]
ATTR_ASSIST = ["Passing", "Vision", "Crossing", "Technique", "Flair", "Dribbling"]
ATTR_DEF    = ["Tackling", "Marking", "Positioning", "WorkRate", "Aggression", "Anticipation"]
ATTR_MIN    = ["CurrentAbility", "NaturalFitness", "Stamina", "Determination"]

ALIASES = {
    "fin": "Finishing", "finishing": "Finishing", "otb": "OffTheBall",
    "off the ball": "OffTheBall", "ant": "Anticipation", "anticipation": "Anticipation",
    "cmp": "Composure", "composure": "Composure", "tec": "Technique", "technique": "Technique",
    "acc": "Acceleration", "acceleration": "Acceleration", "pas": "Passing", "passing": "Passing",
    "vis": "Vision", "vision": "Vision", "cro": "Crossing", "crossing": "Crossing",
    "fla": "Flair", "flair": "Flair", "dri": "Dribbling", "dribbling": "Dribbling",
    "tck": "Tackling", "tackling": "Tackling", "mar": "Marking", "marking": "Marking",
    "positioning": "Positioning", "wor": "WorkRate",
    "work rate": "WorkRate", "workrate": "WorkRate", "agg": "Aggression",
    "aggression": "Aggression", "ca": "CurrentAbility", "current ability": "CurrentAbility",
    "pa": "PotentialAbility", "potential ability": "PotentialAbility",
    "nat": "NaturalFitness", "natural fitness": "NaturalFitness",
    "sta": "Stamina", "stamina": "Stamina", "det": "Determination", "determination": "Determination",
    "name": "name", "player": "name", "club": "team", "team": "team",
    "position": "pos", "pos": "pos", "pos.": "pos", "age": "age",
}
# NOTE: FM abbreviates Positioning as "Pos", which collides with the position
# column. We resolve "pos" -> position (far more common in exports); if your
# export uses "Pos" for Positioning, rename that column before loading.
ALL_ATTRS = sorted(set(ATTR_GOAL + ATTR_ASSIST + ATTR_DEF + ATTR_MIN))


def load_fm_export(path) -> pd.DataFrame:
    """Load an FM export: .html (FM's 'Print to Web Page'), .csv or .tsv.
    Normalises column names to the canonical attribute names above."""
    p = str(path).lower()
    if p.endswith((".html", ".htm")):
        tables = pd.read_html(path)                      # FM exports one big table
        df = max(tables, key=len)
    elif p.endswith(".tsv") or p.endswith(".txt"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    ren = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in ALIASES:
            ren[c] = ALIASES[key]
    df = df.rename(columns=ren)
    # a rename can collide (e.g. two source columns mapping to one name); keep first
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    # FM sometimes exports attributes as "12-15" ranges (unscouted); take midpoint
    for c in [c for c in df.columns if c in ALL_ATTRS or c == "PotentialAbility"]:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.extract(r"(\d+)\s*-?\s*(\d+)?")\
                     .astype(float).mean(axis=1)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "pos" in df:
        df["pos"] = df["pos"].map(_norm_pos)
    return df


def _norm_pos(p):
    s = str(p).upper()
    if "GK" in s: return "GK"
    if any(k in s for k in ["D (", "DC", "DL", "DR", "WB"]): return "DEF"
    if any(k in s for k in ["ST", "FW", "AM (C)" if False else "STC"]): return "FWD"
    if any(k in s for k in ["M (", "AM", "DM", "MC"]): return "MID"
    return "MID"


def _z(df, cols):
    """Standardised attribute matrix; missing attributes filled at the mean."""
    X = pd.DataFrame(index=df.index)
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
        mu = v.mean() if v.notna().any() else 10.0
        sd = v.std() if v.notna().any() and v.std() > 0 else 3.0
        X[c] = ((v.fillna(mu) - mu) / sd).clip(-3, 3)
    return X


def _ridge(X, y, lam=1.0):
    """Ridge regression with intercept (attributes are collinear by construction)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    X1 = np.column_stack([np.ones(len(X)), X])
    P = np.eye(X1.shape[1]) * lam; P[0, 0] = 0.0
    beta = np.linalg.solve(X1.T @ X1 + P, X1.T @ y)
    pred = X1 @ beta
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - pred) ** 2).sum() / ss if ss > 0 else np.nan
    return beta, float(r2)


# ---------------------------------------------------------------------------
# 1. CALIBRATE the attribute -> rate mapping on the overlap population
# ---------------------------------------------------------------------------
def calibrate(fm: pd.DataFrame,
              hist_csv="/mnt/user-data/uploads/fpl-data-stats.csv",
              min_minutes=600, lam=1.0, verbose=True):
    """Fit FM attributes -> observed 25/26 per-90 rates on players present in both.
    Returns a mapping dict consumed by `apply_fm_priors`."""
    d = pd.read_csv(hist_csv)
    d["pos"] = d.element_type.map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    ag = d.groupby(["web_name", "pos"]).agg(
        minutes=("minutes", "sum"),
        npxgi=("non_penalty_expected_goal_involvements", "sum"),
        xa=("expected_assists", "sum"),
        defcon=("defensive_contribution", "sum"),
        starts=("minutes", lambda s: (s >= 60).sum()),
        games=("minutes", "size")).reset_index()
    ag = ag[ag.minutes >= min_minutes].copy()
    nnf = ag.minutes / 90.0
    ag["inv90"] = ag.npxgi / nnf; ag["xa90"] = ag.xa / nnf
    ag["dc90"] = ag.defcon / nnf; ag["startrate"] = ag.starts / ag.games

    # join on surname-insensitive key (FM gives full names, FPL gives web_name)
    ag["k"] = ag.web_name.str.lower().str.strip().str.split().str[-1]
    fm = fm.copy()
    fm["k"] = fm["name"].astype(str).str.lower().str.strip().str.split().str[-1]
    ov = ag.merge(fm.drop_duplicates("k"), on="k", suffixes=("", "_fm"))
    if verbose:
        print(f"[fm calibrate] overlap: {len(ov)} players "
              f"(FM {len(fm)}, PL>={min_minutes}min {len(ag)})")
    if len(ov) < 30:
        raise ValueError(f"overlap too small ({len(ov)}) to calibrate; check name matching")

    mapping = {"n_overlap": len(ov), "targets": {}}
    specs = [("inv90", ATTR_GOAL, True), ("xa90", ATTR_ASSIST, True),
             ("dc90", ATTR_DEF, True), ("startrate", ATTR_MIN, False)]
    for target, attrs, use_log in specs:
        y = ov[target].values
        y = np.log(y + 0.02) if use_log else y
        X = _z(ov, attrs)
        beta, r2 = _ridge(X.values, y, lam)
        # store the standardisation so new players are scaled identically
        norm = {c: (float(pd.to_numeric(ov[c], errors="coerce").mean()) if c in ov else 10.0,
                    float(pd.to_numeric(ov[c], errors="coerce").std() or 3.0) if c in ov else 3.0)
                for c in attrs}
        mapping["targets"][target] = {"attrs": attrs, "beta": beta.tolist(),
                                      "log": use_log, "r2": r2, "norm": norm,
                                      # clamp to the observed range: +3SD on every
                                      # attribute at once otherwise extrapolates the
                                      # log-link far beyond anything real
                                      "lo": float(np.percentile(ov[target], 1)),
                                      "hi": float(np.percentile(ov[target], 99))}
        if verbose:
            top = sorted(zip(attrs, beta[1:]), key=lambda t: -abs(t[1]))[:3]
            print(f"  {target:10s} R^2={r2:5.3f}  top: " +
                  ", ".join(f"{a}{b:+.2f}" for a, b in top))
    return mapping


def _predict(mapping, target, row_df):
    spec = mapping["targets"][target]
    X = pd.DataFrame(index=row_df.index)
    for c in spec["attrs"]:
        mu, sd = spec["norm"][c]
        v = pd.to_numeric(row_df[c], errors="coerce") if c in row_df else pd.Series(np.nan, index=row_df.index)
        X[c] = ((v.fillna(mu) - mu) / (sd if sd else 3.0)).clip(-3, 3)
    beta = np.array(spec["beta"])
    eta = beta[0] + X.values @ beta[1:]
    out = np.exp(eta) - 0.02 if spec["log"] else eta
    lo, hi = spec.get("lo"), spec.get("hi")
    if lo is not None and hi is not None:
        out = np.clip(out, lo, hi)      # no extrapolation past observed reality
    return out


# ---------------------------------------------------------------------------
# 2. APPLY as priors — strongest where there is least data
# ---------------------------------------------------------------------------
def apply_fm_priors(players: pd.DataFrame, fm: pd.DataFrame, mapping,
                    league_factor=0.80, K=900.0, verbose=True):
    """Set/blend Gamma & Beta priors from FM attributes.

    Weight on FM = K / (K + observed_minutes):
      * cold-start (0 PL minutes)  -> weight 1.0, prior is entirely FM-driven
      * 900 PL minutes             -> weight 0.5
      * 3000 PL minutes            -> weight 0.23, observed data dominates
    `league_factor` haircuts attacking output for players stepping up from a
    lower division (applied to cold-start players only).
    """
    p = players.copy()
    fm = fm.copy()
    fm["k"] = fm["name"].astype(str).str.lower().str.strip().str.split().str[-1]
    fm = fm.drop_duplicates("k").set_index("k")

    pred_inv = _predict(mapping, "inv90", fm)
    pred_xa  = _predict(mapping, "xa90", fm)
    pred_dc  = _predict(mapping, "dc90", fm)
    pred_st  = np.clip(_predict(mapping, "startrate", fm), 0.05, 0.97)
    fm_pred = pd.DataFrame({"inv90": pred_inv, "xa90": pred_xa, "dc90": pred_dc,
                            "startrate": pred_st}, index=fm.index)

    n_applied = n_cold = 0
    for i, r in p.iterrows():
        k = str(r.web_name).lower().strip().split()[-1]
        if k not in fm_pred.index:
            continue
        f = fm_pred.loc[k]
        obs_min = float(r.get("minutes", 0) or 0)
        is_cold = bool(r.get("cold_start", False)) or obs_min <= 0
        w = K / (K + max(obs_min, 0.0))                  # FM weight
        lf = league_factor if is_cold else 1.0

        for col, alpha_c, beta_c in [("inv90", "npxgi_alpha", "npxgi_beta"),
                                     ("xa90", "xa_alpha", "xa_beta"),
                                     ("dc90", "defcon_alpha", "defcon_beta")]:
            if alpha_c not in p.columns:
                continue
            k0 = float(r[beta_c]) if pd.notna(r[beta_c]) and r[beta_c] else 1.5
            cur = pd.to_numeric(r[alpha_c], errors="coerce")
            fm_mean = max(float(f[col]) * (lf if col != "dc90" else 1.0), 1e-3)
            if pd.isna(cur):
                blended = fm_mean            # no usable prior -> FM entirely
            else:
                cur_mean = float(cur) / k0
                blended = (1 - w) * cur_mean + w * fm_mean
            if not np.isfinite(blended):
                blended = fm_mean
            p.at[i, alpha_c] = max(blended, 1e-3) * k0

        # availability prior: only where we have little/no evidence
        if is_cold and "start_a" in p.columns:
            s = float(f["startrate"]); strength = 4.0
            p.at[i, "start_a"] = s * strength
            p.at[i, "start_b"] = (1 - s) * strength
            n_cold += 1
        n_applied += 1
    if verbose:
        print(f"[fm priors] applied to {n_applied} players "
              f"({n_cold} cold-start got FM-driven availability)")
    p["fm_matched"] = [str(x).lower().strip().split()[-1] in fm_pred.index for x in p.web_name]
    return p


if __name__ == "__main__":
    print(__doc__.split("GETTING THE DATA")[0])
    print("Usage:\n"
          "  import fm_priors as fmp\n"
          "  fm = fmp.load_fm_export('fm26_export.html')\n"
          "  m  = fmp.calibrate(fm)                 # fit on FM x PL overlap\n"
          "  players = fmp.apply_fm_priors(players, fm, m)\n")
