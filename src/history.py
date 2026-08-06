"""
history.py — multi-season historical calibration of the model's hyperparameters
===============================================================================
WHAT HISTORY IS ACTUALLY FOR
----------------------------
Historical Premier League data is MATCH RESULTS ONLY (no player stats), so it
cannot inform player priors. What it *can* do — and this matters a lot — is
replace the hyperparameters currently HARD-CODED in `bayes_model.TeamModel`,
which are presently my educated guesses:

    revert       = 0.85   how far team strength reverts between seasons
    season_sd    = 0.15   between-season innovation (new manager, transfers)
    promoted     = (-0.20 attack, -0.22 defence, sd 0.30)   <- pure guess
    home_prior   = 0.26   home advantage (has DECLINED markedly over 30 years)
    (dixon-coles low-score correlation, currently not modelled)

These are exactly the parameters that dominate when the likelihood is thin —
i.e. preseason and the opening gameweeks of 26/27, which is when we need the
model most. Thirty seasons of results identify all of them cleanly.

The promoted-team prior is the standout: Coventry, Hull and Ipswich have no
Premier League data at all, so their entire 26/27 posterior IS that prior. With
30 seasons we can estimate how promoted sides *actually* perform in their first
top-flight season, instead of guessing.

DATA SOURCE
-----------
football-data.co.uk (FREE, no authentication, no API key) has every season from
1993/94 in the same CSV schema the pipeline already reads:

    https://www.football-data.co.uk/mmz4281/{SEASON}/E0.csv    e.g. 9394 ... 2526

This is preferable to the Kaggle mirror (evangower/premier-league-matches),
which contains the same match results but requires a Kaggle account/API token
and lacks the odds columns.
"""
from __future__ import annotations
import io, urllib.request
import numpy as np, pandas as pd

FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
NAME_NORM = {"Man Utd": "Man United", "Spurs": "Tottenham",
             "Nott'm Forest": "Nott'm Forest", "Middlesboro": "Middlesbrough"}


def season_codes(start=1993, end=2025):
    """['9394', '9495', ... ] for seasons beginning `start` .. `end`."""
    out = []
    for y in range(start, end + 1):
        out.append(f"{y % 100:02d}{(y + 1) % 100:02d}")
    return out


def download_seasons(start=1993, end=2025, verbose=True, cache_dir="/tmp/fd_hist"):
    """Fetch every available season from football-data.co.uk. FREE, no key.
    Cached to disk; missing/failed seasons are skipped with a note."""
    import os
    os.makedirs(cache_dir, exist_ok=True)
    frames = []
    for code in season_codes(start, end):
        path = f"{cache_dir}/E0_{code}.csv"
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, encoding="latin-1")
            else:
                with urllib.request.urlopen(FD_URL.format(code=code), timeout=25) as r:
                    raw = r.read()
                df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
                df.to_csv(path, index=False)
            need = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
            if not need.issubset(df.columns):
                if verbose: print(f"  skip {code}: missing columns")
                continue
            df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            df["season"] = code
            df["HomeTeam"] = df.HomeTeam.replace(NAME_NORM)
            df["AwayTeam"] = df.AwayTeam.replace(NAME_NORM)
            frames.append(df[["season", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]])
            if verbose: print(f"  {code}: {len(df)} matches")
        except Exception as e:
            if verbose: print(f"  skip {code}: {type(e).__name__}")
    if not frames:
        raise RuntimeError("no seasons downloaded (offline?)")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-season attack/defence ratings (results-based Poisson, ridge-regularised)
# ---------------------------------------------------------------------------
def fit_season(df, l2=1.0):
    """log E[goals] = mu + home*1{home} + attack_i - defence_j, centred."""
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}; n = len(teams)
    rows, y = [], []
    for _, r in df.iterrows():
        h, a = idx[r.HomeTeam], idx[r.AwayTeam]
        v = np.zeros(2 * n + 2); v[0] = 1; v[1] = 1; v[2 + h] = 1; v[2 + n + a] = -1
        rows.append(v); y.append(r.FTHG)
        v = np.zeros(2 * n + 2); v[0] = 1; v[2 + a] = 1; v[2 + n + h] = -1
        rows.append(v); y.append(r.FTAG)
    X = np.array(rows); y = np.array(y, float)
    theta = np.zeros(X.shape[1]); P = np.eye(X.shape[1]) * l2; P[0, 0] = P[1, 1] = 0
    for _ in range(60):
        mu = np.exp(np.clip(X @ theta, -8, 6))
        g = X.T @ (mu - y) + P @ theta
        H = X.T @ (X * mu[:, None]) + P
        step = np.linalg.solve(H, g); theta -= step
        if np.max(np.abs(step)) < 1e-9: break
    att = pd.Series(theta[2:2 + n], index=teams)
    dfn = pd.Series(theta[2 + n:], index=teams)
    return (att - att.mean()), (dfn - dfn.mean()), float(theta[1]), float(theta[0])


def season_ratings(hist, verbose=True):
    """Fit every season; return tidy ratings + per-season home advantage."""
    rows, ha = [], []
    for s, g in hist.groupby("season", sort=True):
        att, dfn, home, mu = fit_season(g)
        for t in att.index:
            rows.append({"season": s, "team": t, "attack": att[t], "defence": dfn[t]})
        ha.append({"season": s, "home_adv": home, "log_mu": mu,
                   "goals_per_game": (g.FTHG + g.FTAG).mean()})
        if verbose: print(f"  {s}: home_adv={home:+.3f}  gpg={(g.FTHG+g.FTAG).mean():.2f}")
    return pd.DataFrame(rows), pd.DataFrame(ha)


# ---------------------------------------------------------------------------
# THE ESTIMATES (these replace the hard-coded hyperparameters)
# ---------------------------------------------------------------------------
def estimate_reversion(rat, use_iv=True):
    """Regress season t+1 strength on season t strength for SURVIVING teams.
    slope = reversion coefficient (`revert`); residual sd = `season_sd`.

    IMPORTANT — errors-in-variables. The season-t rating is itself an ESTIMATE
    with sampling noise, so the naive OLS slope is attenuated toward zero (and
    the residual sd correspondingly inflated). We correct with instrumental
    variables: season t-1's rating instruments for season t's, since the
    measurement errors are independent across seasons while the true strengths
    are persistent. `revert_ols` is retained for comparison.
    """
    out = {}
    order = sorted(rat.season.unique())
    pairs = []
    for i in range(1, len(order) - 1):
        sm1, s0, s1 = order[i - 1], order[i], order[i + 1]
        a = rat[rat.season == s0].set_index("team")
        b = rat[rat.season == s1].set_index("team")
        z = rat[rat.season == sm1].set_index("team")
        common = a.index.intersection(b.index)
        for t in common:
            rec = {"att0": a.loc[t, "attack"], "att1": b.loc[t, "attack"],
                   "def0": a.loc[t, "defence"], "def1": b.loc[t, "defence"],
                   "attz": z.loc[t, "attack"] if t in z.index else np.nan,
                   "defz": z.loc[t, "defence"] if t in z.index else np.nan}
            pairs.append(rec)
    P = pd.DataFrame(pairs)
    for k in ["att", "def"]:
        x, y = P[f"{k}0"].values, P[f"{k}1"].values
        b_ols = np.polyfit(x, y, 1)
        out[f"revert_{k}_ols"] = float(b_ols[0])
        z = P[f"{k}z"].values
        m = np.isfinite(z)
        if use_iv and m.sum() > 30:
            # just-identified IV: b = cov(z,y)/cov(z,x)
            cxz = np.cov(z[m], x[m])[0, 1]
            cyz = np.cov(z[m], y[m])[0, 1]
            b_iv = cyz / cxz if abs(cxz) > 1e-9 else b_ols[0]
            b_iv = float(np.clip(b_iv, 0.0, 1.2))
            out[f"revert_{k}"] = b_iv
            resid = y[m] - (np.mean(y[m]) + b_iv * (x[m] - np.mean(x[m])))
            # The residual still contains the measurement error in y, so its sd
            # over-states the true between-season innovation. The OLS/IV slope
            # ratio IS the reliability ratio lambda = var(signal)/var(measured),
            # so var(meas err in y) ~ (1-lambda)*var(y). Subtract it.
            lam = float(np.clip(b_ols[0] / b_iv, 0.05, 1.0)) if abs(b_iv) > 1e-9 else 1.0
            var_true = resid.var() - (1 - lam) * np.var(y[m])
            out[f"season_sd_{k}"] = float(np.sqrt(max(var_true, 1e-6)))
            out[f"reliability_{k}"] = lam
        else:
            out[f"revert_{k}"] = float(b_ols[0])
            out[f"season_sd_{k}"] = float((y - np.polyval(b_ols, x)).std())
    out["revert"] = float(np.mean([out["revert_att"], out["revert_def"]]))
    out["revert_ols"] = float(np.mean([out["revert_att_ols"], out["revert_def_ols"]]))
    out["season_sd"] = float(np.mean([out["season_sd_att"], out["season_sd_def"]]))
    out["n_pairs"] = len(P)
    return out


def estimate_promoted_prior(rat):
    """Teams appearing in season t+1 but NOT t are promoted. Their FIRST-season
    attack/defence ratings give the promoted prior mean and sd directly."""
    order = sorted(rat.season.unique())
    rows = []
    for s0, s1 in zip(order[:-1], order[1:]):
        prev = set(rat[rat.season == s0].team)
        cur = rat[rat.season == s1]
        for _, r in cur.iterrows():
            if r.team not in prev:
                rows.append({"season": s1, "team": r.team,
                             "attack": r.attack, "defence": r.defence})
    P = pd.DataFrame(rows)
    return {
        "promoted_att_mean": float(P.attack.mean()), "promoted_att_sd": float(P.attack.std()),
        "promoted_def_mean": float(P.defence.mean()), "promoted_def_sd": float(P.defence.std()),
        "n_promoted": len(P), "detail": P,
    }


def estimate_home_advantage(ha, recent=5):
    """Home advantage and its trend. It has declined substantially since the
    1990s (and stepped down around the crowdless 2020/21 season), so the RECENT
    mean is the right prior, not the 30-year average."""
    ha = ha.sort_values("season")
    x = np.arange(len(ha))
    slope, intercept = np.polyfit(x, ha.home_adv.values, 1)
    return {"home_adv_all": float(ha.home_adv.mean()),
            "home_adv_recent": float(ha.home_adv.tail(recent).mean()),
            "home_adv_sd_recent": float(ha.home_adv.tail(recent).std()),
            "home_adv_trend_per_season": float(slope),
            "goals_per_game_recent": float(ha.goals_per_game.tail(recent).mean())}


def estimate_dixon_coles_rho(hist):
    """Low-score dependence: compare observed vs independent-Poisson frequency
    of 0-0/1-0/0-1/1-1. rho>0 => draws/low scores more common than independent."""
    lam_h = hist.FTHG.mean(); lam_a = hist.FTAG.mean()
    from scipy.stats import poisson
    out = {}
    for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        obs = float(((hist.FTHG == i) & (hist.FTAG == j)).mean())
        exp = float(poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a))
        out[f"obs_{i}{j}"] = obs; out[f"exp_{i}{j}"] = exp
        out[f"ratio_{i}{j}"] = obs / exp if exp > 0 else np.nan
    return out


def calibrate_all(start=1993, end=2025, verbose=True):
    """Full pipeline: download -> per-season ratings -> hyperparameters."""
    if verbose: print("Downloading seasons (football-data.co.uk, free)...")
    hist = download_seasons(start, end, verbose=verbose)
    if verbose: print(f"\nTotal: {len(hist)} matches, {hist.season.nunique()} seasons")
    if verbose: print("\nFitting per-season team ratings...")
    rat, ha = season_ratings(hist, verbose=verbose)
    res = {}
    res.update(estimate_reversion(rat))
    prom = estimate_promoted_prior(rat)
    res.update({k: v for k, v in prom.items() if k != "detail"})
    res.update(estimate_home_advantage(ha))
    res.update(estimate_dixon_coles_rho(hist))
    return res, rat, ha, prom.get("detail")


def to_model_kwargs(res):
    """Map estimates onto TeamModel/sample_2627 arguments."""
    return {
        "home_prior": (res["home_adv_recent"], max(res["home_adv_sd_recent"], 0.03)),
        "revert": res["revert"],
        "season_sd": res["season_sd"],
        "promoted_att": (res["promoted_att_mean"], res["promoted_att_sd"]),
        "promoted_def": (res["promoted_def_mean"], res["promoted_def_sd"]),
    }


if __name__ == "__main__":
    try:
        res, rat, ha, prom = calibrate_all()
        print("\n=== CALIBRATED HYPERPARAMETERS (replace the hard-coded guesses) ===")
        for k in ["revert", "season_sd", "promoted_att_mean", "promoted_att_sd",
                  "promoted_def_mean", "promoted_def_sd", "home_adv_all",
                  "home_adv_recent", "home_adv_trend_per_season", "n_pairs", "n_promoted"]:
            print(f"  {k:28s} {res[k]:+.4f}" if isinstance(res[k], float) else f"  {k:28s} {res[k]}")
        print("\nmodel kwargs:", to_model_kwargs(res))
    except Exception as e:
        print(f"[history] {type(e).__name__}: {e}")
        print("Needs network. Run in your environment: python history.py")
</content>
