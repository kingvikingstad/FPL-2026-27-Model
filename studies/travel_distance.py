"""
travel_distance.py — does the distance the away side travels move goals, holding team
quality constant?
==================================================================================
PRE-REGISTRATION, written 2026-09-10 before any outcome was read. The `--design` pass
below is outcome-blind (it reads fixtures and stadium geography, never a score) and ran
first; it fixes identification and power. The decision rules are stated here and are
not revisited after the outcome pass.

WHAT THE REGRESSOR IS, AND WHAT IT IS NOT
----------------------------------------
Only the away side travels, so distance is not a team attribute — it is a MODIFIER OF
HOME ADVANTAGE, fixture by fixture. The model carries one constant home parameter
(`bayes_model._home_effect`); if distance matters, that constant is an average over
short and long trips and misprices both ends.

This is not the congestion null under a new name. Congestion was tested as RECOVERY
TIME (rest differential, European participation, midweek fixture by recovery day) and is
dead for GW1-26. Distance is a different instrument and the mechanism is left open:
travel fatigue, away-support share and derby familiarity all load on it and cannot be
separated here. For forecasting that does not matter — the covariate is what it is.

    z = log(max(km, 5))      great-circle km between the two grounds, on the match date

Log because the contrast that exists in the Premier League is derbies (1-20 km) against
everything else; beyond ~150 km both plausible mechanisms flatten. Floored at 5 km so the
Palace/Wimbledon groundshare (0 km, 1994-99) and the Merseyside derby (0.8 km) do not
become leverage points. Linear-per-100km is a pre-registered SECONDARY form.

ESTIMANDS  (one row per team-match; two coefficients)
---------
  log E[goals_r] = home_s * 1[home row] + att_{team,season} - def_{opp,season}
                   + b_H * z * 1[home row] + b_A * z * 1[away row]

  b_A  effect of the trip on the TRAVELLING side's goals FOR       (travel: b_A < 0)
  b_H  effect of the trip on the travelling side's goals AGAINST   (travel: b_H > 0)

Team quality is held constant two ways, and they answer different questions:
  E1 STRUCTURAL: attack AND defence fixed effects per TEAM-SEASON, home advantage per
     season (it has roughly halved since 1993, and the geography of the league changed
     with it). 31 seasons, 11,944 matches. Poisson FE has no incidental-parameter bias.
     SEs cluster-robust on the UNORDERED CLUB PAIR — the regressor is fixed at pair level,
     so match-level SEs would treat every repeat of a fixture as new information.
  E2 MARKET: pre-match consensus 1X2 + O/U 2.5 inverted to (lam_H, lam_A) by
     `betting_odds_ingest.implied_lambdas`, 2005/06-2025/26. Does z predict goals over
     the market lambda? This is the CLAUDE.md market-orthogonality gate, computed by
     `style_matchup.market_score_test` — the same Poisson score test `beats_the_market`
     runs, refactored so it takes an arbitrary covariate rather than only style products.

DECISION RULES (fixed before results)
-------------------------------------
R1 IDENTIFIED. Share of z's variance left after projecting out the team-season FE, via
   `style_matchup.check_identification` (season by season; the FE block is block-
   diagonal by season, which is what makes the n x n projector affordable). > 5%.
R2 REAL AND MATERIAL. E1 on the full sample: |z| > 2.5 (cluster-robust) for b_H or b_A,
   AND the implied change in log goals between the P10 and P90 trip is >= 0.03
   (about 0.04 goals, ~1.4pp of clean-sheet probability at lam_against 1.2).
R3 BEATS THE MARKET. E2: |score z| > 2.5 for the same side, AND the robustness GLM
   (free intercept and free slope on log lam_mkt, pair-clustered) agrees in sign with
   |z| > 2.5. Conjunctive on purpose: the score test assumes the market lambda is
   calibrated in level, and a Poisson inversion of 1X2/O-U is not exactly.
R4 STILL TRUE. The 2016/17-2025/26 E1 estimate has the same sign as the full-sample
   one and the two do not differ (|z_diff| < 1.96). Sign agreement, not significance:
   a decade alone has ~sqrt(3) the full-sample SE and would fail a significance rule
   on any effect the full sample can see.

  R1 & R2 & R3 & R4  -> SHIP ON: a per-fixture term in `_home_effect`, full-sample b,
                        centred on the 25/26 mean trip so average home advantage is kept.
  R1 & R2 & R4, ~R3  -> real football the market already prices. The guard forbids it
                        entering TeamModel, so it ships OFF BY DEFAULT behind a flag and
                        the guard's scope is raised with the user rather than overridden:
                        this pipeline's per-fixture lambda does not ingest match odds.
  anything else      -> NULL. Recorded; no model code path is created.

A wrong-sign result that passes everything triggers a join/coordinate audit before any
code ships.

SECONDARY, Bonferroni over 4 (2 forms x 2 sides) and no decision rule: linear km;
holiday fixtures (24 Dec-3 Jan, when the league regionalises the schedule on purpose)
dropped; closing odds for E2 on 2019/20+. DESCRIPTIVE, no rule: behind-closed-doors
matches (2019/20 from 17 Jun 2020, all of 2020/21) interacted with z — a crowd mechanism
should vanish there, a travel one should not — at n~470 matches it is underpowered and
reported as a CI only.

DATA
----
football-data.co.uk E0, read from the cache `history.download_seasons` fills
(<SCRATCH>/fd_hist). 03/04 and 04/05 are absent from it. Offline: an empty cache is
reported, not fetched. Stadium coordinates, with dated moves, are in src/travel.py;
they are [CHECK] to ~0.5 km, which the 5 km floor makes immaterial.

Run:  python studies/travel_distance.py --design     outcome-blind pass only
      python studies/travel_distance.py              full study
      python studies/travel_distance.py --selftest
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import scipy.sparse as sp
import config
import history as H
import style_matchup as sm

OUT = _os.path.join(config.STUDIES, "travel_distance.csv")
FLOOR_KM = 5.0
RECENT_FROM = 2016
CLOSED_DOORS = (pd.Timestamp("2020-06-17"), pd.Timestamp("2021-05-23"))

# Grounds, dated moves and the distance itself live in src/travel.py — ONE table, which
# the board path reads when FPL_TRAVEL=on. A study-local copy would drift from it.
from travel import GROUNDS, ground, haversine_km, trip_km  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_ODDS = [("AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"),
         ("BbAvH", "BbAvD", "BbAvA", "BbAv>2.5", "BbAv<2.5")]
_CLOSE = ("AvgCH", "AvgCD", "AvgCA", "AvgC>2.5", "AvgC<2.5")


def load_matches(cache_dir=None) -> pd.DataFrame:
    """Every cached E0 season, one row per match, with trip km and consensus odds.
    Reads the cache only; never the network."""
    cache_dir = cache_dir or _os.path.join(config.SCRATCH, "fd_hist")
    rows = []
    for f in glob.glob(_os.path.join(cache_dir, "E0_*.csv")):
        code = _os.path.basename(f)[3:7]
        d = pd.read_csv(f, encoding="latin-1")
        d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"])
        out = pd.DataFrame({
            "season": code, "y0": H.season_start_year(code),
            "date": pd.to_datetime(d["Date"], dayfirst=True, format="mixed"),
            "home": d["HomeTeam"].replace(H.NAME_NORM).str.strip(),
            "away": d["AwayTeam"].replace(H.NAME_NORM).str.strip(),
            "hg": d["FTHG"].astype(int), "ag": d["FTAG"].astype(int)})
        for tag, cols in (("", None), ("c_", _CLOSE)):
            use = cols
            if use is None:
                use = next((c for c in _ODDS if set(c).issubset(d.columns)), None)
            for k, c in zip(("oH", "oD", "oA", "oO", "oU"), use or (None,) * 5):
                out[tag + k] = pd.to_numeric(d[c], errors="coerce") if (
                    c is not None and c in d.columns) else np.nan
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    m = pd.concat(rows, ignore_index=True).sort_values(["date", "home"]).reset_index(drop=True)
    m["km"] = [trip_km(h, a, t) for h, a, t in zip(m["home"], m["away"], m["date"])]
    m["z"] = np.log(np.maximum(m["km"], FLOOR_KM))
    m["pair"] = [" | ".join(sorted((h, a))) for h, a in zip(m["home"], m["away"])]
    md = m["date"].dt.month * 100 + m["date"].dt.day
    m["holiday"] = (md >= 1224) | (md <= 103)
    m["closed"] = (m["date"] >= CLOSED_DOORS[0]) & (m["date"] <= CLOSED_DOORS[1])
    return m


def to_long(m: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: the home side's goals and the away side's goals."""
    h = pd.DataFrame({"team": m["home"], "opp": m["away"], "g": m["hg"], "is_home": 1})
    a = pd.DataFrame({"team": m["away"], "opp": m["home"], "g": m["ag"], "is_home": 0})
    keep = ["season", "y0", "date", "z", "km", "pair", "holiday", "closed"]
    h = pd.concat([h, m[keep].reset_index(drop=True)], axis=1)
    a = pd.concat([a, m[keep].reset_index(drop=True)], axis=1)
    h["match"] = a["match"] = np.arange(len(m))
    return pd.concat([h, a], ignore_index=True)


# ---------------------------------------------------------------------------
# E1: Poisson with team-season FE, sparse Newton, pair-clustered sandwich
# ---------------------------------------------------------------------------
def fe_design(L: pd.DataFrame, extra: dict[str, np.ndarray]):
    """Sparse [home_s | att_ts | def_ts (one dropped per season) | extra]."""
    n = len(L)
    seasons = sorted(L["season"].unique())
    s_ix = L["season"].map({s: i for i, s in enumerate(seasons)}).values
    ts_att = (L["team"] + "@" + L["season"]).values
    ts_def = (L["opp"] + "@" + L["season"]).values
    levels = sorted(set(ts_att) | set(ts_def))
    drop = {sorted(x for x in levels if x.endswith("@" + s))[0] for s in seasons}
    att_map = {k: i for i, k in enumerate(levels)}
    def_levels = [k for k in levels if k not in drop]
    def_map = {k: i for i, k in enumerate(def_levels)}
    S, A, D = len(seasons), len(levels), len(def_levels)
    r, c, v = [], [], []
    ih = np.where(L["is_home"].values == 1)[0]
    r += list(ih); c += list(s_ix[ih]); v += [1.0] * len(ih)
    r += list(range(n)); c += [S + att_map[k] for k in ts_att]; v += [1.0] * n
    for i, k in enumerate(ts_def):
        if k in def_map:
            r.append(i); c.append(S + A + def_map[k]); v.append(-1.0)
    X = sp.csr_matrix((v, (r, c)), shape=(n, S + A + D))
    names = list(extra)
    Z = np.column_stack([extra[k] for k in names]) if names else np.zeros((n, 0))
    return sp.hstack([X, sp.csr_matrix(Z)]).tocsr(), names, S + A + D, S


def poisson_fe(X, y, n_fe, n_season, cluster, iters=100, ridge=1e-8):
    """Newton on the Poisson log-likelihood. Returns (beta, V_cluster, V_model) for the
    non-FE columns. `cluster` is an integer code per row."""
    p = X.shape[1]
    b = np.zeros(p)
    # Every row carries exactly one attack column (+1 entries; defence columns are -1),
    # so starting the attack block at the log mean starts every mu at the mean.
    att_cols = np.where(np.asarray(X[:, n_season:n_fe].max(0).todense()).ravel() > 0)[0]
    b[n_season + att_cols] = np.log(max(y.mean(), 1e-3))
    for _ in range(iters):
        eta = np.clip(X @ b, -10, 6)
        mu = np.exp(eta)
        g = X.T @ (y - mu)
        Hm = (X.T @ sp.diags(mu) @ X).toarray() + ridge * np.eye(p)
        step = np.linalg.solve(Hm, g)
        t = 1.0
        ll0 = float(y @ eta - mu.sum())
        while t > 1e-4:
            bn = b + t * step
            en = np.clip(X @ bn, -10, 6)
            if float(y @ en - np.exp(en).sum()) >= ll0 - 1e-9:
                break
            t /= 2
        b = bn
        if np.max(np.abs(t * step)) < 1e-9:
            break
    eta = np.clip(X @ b, -10, 6); mu = np.exp(eta)
    Hm = (X.T @ sp.diags(mu) @ X).toarray() + ridge * np.eye(p)
    Hinv = np.linalg.inv(Hm)
    k = p - n_fe
    A = Hinv[n_fe:, :]                                        # (k, p)
    Sc = X.multiply((y - mu)[:, None]).tocsr()                # (n, p) scores
    C = int(cluster.max()) + 1
    G = sp.csr_matrix((np.ones(len(y)), (cluster, np.arange(len(y)))), shape=(C, len(y)))
    U = (G @ Sc)                                              # (C, p), sparse
    AU = np.asarray((U @ A.T))                                # (C, k)
    V = AU.T @ AU * C / max(C - 1, 1)
    return b[n_fe:], V, Hinv[n_fe:, n_fe:], mu


def e1(L: pd.DataFrame, z_col="z", extra_fn=None):
    """Fit the structural model; return a frame of (term, b, se_cluster, se_model, z)."""
    zc = L[z_col].values - L[z_col].values.mean()
    ex = {"b_H (trip -> home goals = traveller GA)": zc * (L["is_home"].values == 1),
          "b_A (trip -> away goals = traveller GF)": zc * (L["is_home"].values == 0)}
    if extra_fn is not None:
        ex.update(extra_fn(L, zc))
    X, names, n_fe, n_s = fe_design(L, ex)
    cl = pd.factorize(L["pair"])[0]
    b, V, Vm, _ = poisson_fe(X, L["g"].values.astype(float), n_fe, n_s, cl)
    se, sem = np.sqrt(np.diag(V)), np.sqrt(np.diag(Vm))
    return pd.DataFrame({"term": names, "b": b, "se": se, "se_model": sem, "z": b / se,
                         "n_rows": len(L), "n_pairs": int(cl.max()) + 1})


# ---------------------------------------------------------------------------
# E2: the market gate
# ---------------------------------------------------------------------------
def market_lambdas(m: pd.DataFrame, prefix="") -> pd.DataFrame:
    import betting_odds_ingest as boi
    cols = [prefix + k for k in ("oH", "oD", "oA", "oO", "oU")]
    ok = m[cols].notna().all(1) & (m[cols] > 1.0).all(1)
    mm = m[ok].copy()
    lh, la = [], []
    for oH, oD, oA, oO, oU in mm[cols].itertuples(index=False):
        a, b, _ = boi.implied_lambdas(oH, oD, oA, oO, oU)
        lh.append(a); la.append(b)
    mm["lam_h"], mm["lam_a"] = lh, la
    return mm


def glm_offset(y, lam, z, cluster):
    """log E[y] = a + c*log(lam) + b*z, pair-clustered. Returns (b, se)."""
    X = np.column_stack([np.ones_like(z), np.log(lam), z])
    b = np.array([0.0, 1.0, 0.0])
    for _ in range(100):
        mu = np.exp(X @ b)
        Hm = (X * mu[:, None]).T @ X
        step = np.linalg.solve(Hm, X.T @ (y - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = np.exp(X @ b)
    Hinv = np.linalg.inv((X * mu[:, None]).T @ X)
    Sc = X * (y - mu)[:, None]
    U = pd.DataFrame(Sc).groupby(cluster).sum().values
    C = U.shape[0]
    V = Hinv @ (U.T @ U) @ Hinv * C / (C - 1)
    return b[2], float(np.sqrt(V[2, 2]))


def e2(mm: pd.DataFrame, z_col="z"):
    cl = pd.factorize(mm["pair"])[0]
    z = mm[z_col].values.astype(float)
    out = []
    for side, y, lam in (("home goals = traveller GA", mm["hg"], mm["lam_h"]),
                         ("away goals = traveller GF", mm["ag"], mm["lam_a"])):
        y = y.values.astype(float); lam = lam.values.astype(float)
        st = sm.market_score_test(z, y, lam)
        bg, sg = glm_offset(y, lam, z - z.mean(), cl)
        out.append({"side": side, "n": len(y), "score_z": st["score_z"],
                    "glm_b": bg, "glm_se": sg, "glm_z": bg / sg,
                    "obs_over_mkt": y.sum() / lam.sum()})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
def design(m: pd.DataFrame) -> dict:
    """Outcome-blind: identification share and minimum detectable effect."""
    print("=" * 78)
    print("DESIGN PASS — outcome-blind (no score is read)")
    print("=" * 78)
    print(f"[sample] {len(m)} matches, {m['season'].nunique()} seasons, "
          f"{m['home'].nunique()} clubs, {m['pair'].nunique()} club pairs")
    q = m["km"].quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    print("[trip km] " + "  ".join(f"p{int(k*100)}={v:.0f}" for k, v in q.items()))
    print(f"          share < 20 km (derby-range): {(m['km'] < 20).mean():.3f}; "
          f"at the {FLOOR_KM:.0f} km floor: {(m['km'] <= FLOOR_KM).mean():.3f}")
    # R1 — the repo's own gate, one season at a time (FE block is block-diagonal)
    ss_res = ss_tot = 0.0
    per = []
    for s, g in m.groupby("season"):
        teams = sorted(set(g["home"]) | set(g["away"]))
        ix = {t: i for i, t in enumerate(teams)}
        att = [ix[t] for t in g["home"]] + [ix[t] for t in g["away"]]
        dfn = [ix[t] for t in g["away"]] + [ix[t] for t in g["home"]]
        ish = np.r_[np.ones(len(g)), np.zeros(len(g))]
        zz = np.r_[g["z"].values, g["z"].values]
        Z = np.column_stack([zz * ish, zz * (1 - ish)])
        Xfe = sm._fe_design(att, dfn, ish, len(teams))
        share = sm.check_identification(Xfe, Z)["resid_var_share"]
        tot = ((Z - Z.mean(0)) ** 2).sum(0)
        ss_res = ss_res + share * tot; ss_tot = ss_tot + tot
        per.append(share.min())
    share = ss_res / ss_tot
    r1 = bool(np.all(share > 0.05))
    print(f"\n[R1] residual variance share after team-season FE: "
          f"b_H column {share[0]:.3f}, b_A column {share[1]:.3f} "
          f"(per-season min {min(per):.3f})  -> {'IDENTIFIED' if r1 else 'NOT IDENTIFIED'}")
    # The gate's denominator is the whole column, and each column is ZERO on the other
    # side's rows, so most of its variance is the home/away level the home dummy absorbs
    # by construction. The share of the variance that could carry signal — z within a
    # side, about its season mean — is the informative figure.
    within = sum(((g["z"] - g["z"].mean()) ** 2).sum() for _, g in m.groupby("season"))
    print(f"     of the within-side variance of z (the part not trivially absorbed): "
          f"{float(ss_res[0] / within):.3f} survives the club fixed effects")
    # MDE: information at fixed a-priori rates (1.50 home, 1.15 away), not fitted ones
    L = to_long(m)
    zc = L["z"].values - L["z"].values.mean()
    ex = {"bH": zc * (L["is_home"].values == 1), "bA": zc * (L["is_home"].values == 0)}
    X, _, n_fe, _ = fe_design(L, ex)
    mu = np.where(L["is_home"].values == 1, 1.50, 1.15)
    Hm = (X.T @ sp.diags(mu) @ X).toarray() + 1e-8 * np.eye(X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(Hm))[n_fe:])
    span = float(np.subtract(*m["z"].quantile([0.9, 0.1])))
    r = L["y0"].values >= RECENT_FROM
    print(f"\n[power] P10->P90 trip spans {span:.2f} log-km units "
          f"({np.exp(m['z'].quantile(0.1)):.0f} km -> {np.exp(m['z'].quantile(0.9)):.0f} km)")
    for nm, s in zip(("b_H", "b_A"), se):
        mde = (2.5 + 0.8416) * s * span
        print(f"  {nm}: model SE {s:.4f}/log-km; on the P10->P90 scale SE {s*span:.3f}, "
              f"95% half-width {1.96*s*span:.3f}, MDE(80%, |z|>2.5) {mde:.3f}")
    print(f"  recent decade ({RECENT_FROM}+, {int(r.sum()/2)} matches): SE x "
          f"{np.sqrt(len(L)/r.sum()):.2f}")
    print("  (cluster-robust SEs will be >= these if pairs carry repeat signal)")
    return {"r1": r1, "share_H": float(share[0]), "share_A": float(share[1]),
            "span": span, "se_design": se}


def main(design_only=False):
    m = load_matches()
    if m.empty:
        print("[study] no cached football-data seasons in <SCRATCH>/fd_hist — populate "
              "with `python scripts/calibrate_team_history.py` (needs network)")
        return
    D = design(m)
    if design_only:
        return
    span = D["span"]
    L = to_long(m)

    print("\n" + "=" * 78)
    print("E1 STRUCTURAL — team-season attack & defence FE, home per season")
    print("=" * 78)
    full = e1(L)
    rec = e1(L[L["y0"] >= RECENT_FROM].reset_index(drop=True))
    for lab, t in (("full 1993-2026", full), (f"recent {RECENT_FROM}-2026", rec)):
        print(f"  [{lab}] {int(t['n_rows'].iloc[0]/2)} matches, {t['n_pairs'].iloc[0]} pairs")
        for _, rr in t.iterrows():
            print(f"    {rr['term']:44s} b={rr['b']:+.4f}  se={rr['se']:.4f} "
                  f"(model {rr['se_model']:.4f})  z={rr['z']:+.2f}  "
                  f"P10->P90 {rr['b']*span:+.3f}")

    print("\n" + "=" * 78)
    print("E2 MARKET — does the trip predict goals over the market lambda?")
    print("=" * 78)
    mm = market_lambdas(m)
    print(f"  {len(mm)} matches with consensus 1X2 + O/U 2.5 "
          f"({mm['season'].nunique()} seasons)")
    mk = e2(mm)
    for _, rr in mk.iterrows():
        print(f"    {rr['side']:28s} score z={rr['score_z']:+.2f}   "
              f"GLM b={rr['glm_b']:+.4f} se={rr['glm_se']:.4f} z={rr['glm_z']:+.2f}   "
              f"obs/mkt {rr['obs_over_mkt']:.3f}")

    # ---- decision ----
    print("\n" + "=" * 78)
    print("DECISION (rules fixed in the docstring)")
    print("=" * 78)
    rows, verdicts = [], []
    for i, side in enumerate(("GA", "GF")):
        f, rr, k = full.iloc[i], rec.iloc[i], mk.iloc[i]
        r2 = bool(abs(f["z"]) > 2.5 and abs(f["b"]) * span >= 0.03)
        r3 = bool(abs(k["score_z"]) > 2.5 and abs(k["glm_z"]) > 2.5
                  and np.sign(k["score_z"]) == np.sign(k["glm_z"]) == np.sign(f["b"]))
        zd = (f["b"] - rr["b"]) / np.sqrt(max(rr["se"] ** 2 - f["se"] ** 2, 1e-12))
        r4 = bool(np.sign(rr["b"]) == np.sign(f["b"]) and abs(zd) < 1.96)
        v = ("SHIP ON" if D["r1"] and r2 and r3 and r4 else
             "OFF BY DEFAULT (priced by market)" if D["r1"] and r2 and r4 else "NULL")
        verdicts.append(v)
        print(f"  traveller {side}: R1 {D['r1']}  R2 {r2} (z {f['z']:+.2f}, "
              f"P10->P90 {f['b']*span:+.3f})  R3 {r3} (score {k['score_z']:+.2f}, "
              f"glm {k['glm_z']:+.2f})  R4 {r4} (recent {rr['b']:+.4f}, zdiff {zd:+.2f})"
              f"  -> {v}")
        rows.append({"side": f"traveller {side}", "b_full": f["b"], "se_full": f["se"],
                     "z_full": f["z"], "p10_p90": f["b"] * span, "b_recent": rr["b"],
                     "se_recent": rr["se"], "score_z_mkt": k["score_z"],
                     "glm_z_mkt": k["glm_z"], "R1": D["r1"], "R2": r2, "R3": r3,
                     "R4": r4, "verdict": v})

    # ---- secondary (Bonferroni over 4, no decision rule) ----
    print("\n" + "=" * 78)
    print("SECONDARY — Bonferroni over 4 (|z| > 2.50), no decision rule")
    print("=" * 78)
    L["z_lin"] = L["km"] / 100.0
    lin = e1(L, z_col="z_lin")
    for _, rr in lin.iterrows():
        print(f"  linear per 100 km   {rr['term']:44s} b={rr['b']:+.4f} z={rr['z']:+.2f}")
    nh = e1(L[~L["holiday"]].reset_index(drop=True))
    for _, rr in nh.iterrows():
        print(f"  holidays dropped    {rr['term']:44s} b={rr['b']:+.4f} z={rr['z']:+.2f}")
    mc = market_lambdas(m, prefix="c_")
    if len(mc):
        for _, rr in e2(mc).iterrows():
            print(f"  closing odds 19/20+ {rr['side']:28s} n={rr['n']} "
                  f"score z={rr['score_z']:+.2f} glm z={rr['glm_z']:+.2f}")

    print("\n" + "=" * 78)
    print("DESCRIPTIVE — behind closed doors (no crowd), no decision rule")
    print("=" * 78)
    cd = e1(L, extra_fn=lambda LL, zc: {
        "b_H x closed": zc * (LL["is_home"].values == 1) * LL["closed"].values,
        "b_A x closed": zc * (LL["is_home"].values == 0) * LL["closed"].values})
    print(f"  {int(L['closed'].sum()/2)} closed-door matches")
    for _, rr in cd.iterrows():
        lo, hi = rr["b"] - 1.96 * rr["se"], rr["b"] + 1.96 * rr["se"]
        print(f"    {rr['term']:44s} b={rr['b']:+.4f}  95% ({lo:+.4f}, {hi:+.4f})")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n-> {OUT}")
    return verdicts


# ---------------------------------------------------------------------------
def selftest():
    # 1. geography: two known distances
    km = trip_km("Man United", "Chelsea", "2020-01-01")
    assert 250 < km < 275, km                       # Old Trafford - Stamford Bridge ~262
    assert trip_km("Wimbledon", "Crystal Palace", "1995-01-01") == 0.0
    assert trip_km("Tottenham", "Arsenal", "2018-01-01") > 10       # Wembley season
    assert trip_km("Tottenham", "Arsenal", "2019-05-01") < 7        # new ground
    try:
        ground("Nowhere FC", "2020-01-01"); raise AssertionError("unknown club passed")
    except KeyError:
        pass
    # 2. recovery: synthetic league, true b_H=+0.05, b_A=-0.07 on log-km
    rng = np.random.default_rng(11)
    T, seasons = 16, 6
    teams = [f"T{i:02d}" for i in range(T)]
    xy = {t: rng.normal(0, 1.2, 2) for t in teams}           # a fake map, degrees
    rows = []
    for s in range(seasons):
        att = dict(zip(teams, rng.normal(0, 0.25, T)))
        dfn = dict(zip(teams, rng.normal(0, 0.25, T)))
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                km = float(np.hypot(*(xy[h] - xy[a])) * 111)
                z = np.log(max(km, FLOOR_KM))
                rows.append((f"{s:04d}", 2000 + s, h, a, km, z, att, dfn))
    zbar = np.mean([r[5] for r in rows])
    recs = []
    for s, y0, h, a, km, z, att, dfn in rows:
        lh = np.exp(0.30 + att[h] - dfn[a] + 0.05 * (z - zbar))
        la = np.exp(0.05 + att[a] - dfn[h] - 0.07 * (z - zbar))
        recs.append({"season": s, "y0": y0, "date": pd.Timestamp(f"{y0}-10-01"),
                     "home": h, "away": a, "hg": rng.poisson(lh), "ag": rng.poisson(la),
                     "km": km, "z": z, "pair": " | ".join(sorted((h, a))),
                     "holiday": False, "closed": False})
    m = pd.DataFrame(recs)
    t = e1(to_long(m))
    bH, bA = t["b"].values
    assert abs(bH - 0.05) < 3 * t["se"].iloc[0], (bH, t)
    assert abs(bA + 0.07) < 3 * t["se"].iloc[1], (bA, t)
    # 3. a regressor that is a pure team main effect must be flagged unidentified
    ish = np.r_[np.ones(len(m)), np.zeros(len(m))]
    ix = {tm: i for i, tm in enumerate(teams)}
    att_i = [ix[x] for x in m["home"]] + [ix[x] for x in m["away"]]
    dfn_i = [ix[x] for x in m["away"]] + [ix[x] for x in m["home"]]
    zteam = np.array([xy[teams[i]][0] for i in att_i])       # own latitude only
    Xfe = sm._fe_design(att_i, dfn_i, ish, T)
    assert not sm.check_identification(Xfe, zteam[:, None])["identified"]
    # 4. market gate: a covariate with no effect over a true lambda is not significant
    lam = rng.uniform(0.6, 2.5, 4000)
    y = rng.poisson(lam).astype(float)
    zz = rng.normal(size=4000)
    assert abs(sm.market_score_test(zz, y, lam)["score_z"]) < 3.5
    print("SELFTEST OK: distances (OT-Bridge ~262 km, groundshare 0, Spurs' Wembley year), "
          f"FE recovery b_H={bH:+.3f} (0.05) b_A={bA:+.3f} (-0.07), a team-level "
          "regressor is refused by the identification gate, null market test is null.")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        selftest(); _sys.exit(0)
    main(design_only="--design" in _sys.argv)
