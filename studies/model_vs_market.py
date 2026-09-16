"""
model_vs_market.py — the model's per-fixture lambda against the market's, on realised goals.
===========================================================================================
PRE-REGISTERED: docs/MODEL_VS_MARKET_PREREG_2026-09-16.md. This file implements that
document and decides nothing it does not. Written 2026-09-16, before any outcome of any
admissible gameweek existed (GW5, the first, kicks off 2026-09-19).

WHAT A RUN DOES
---------------
A bare run — which is what `scripts/test_all.py` does to every study — first runs the
offline selftest, then computes the INFORMATION

    I = sum over admissible team-fixtures of  r^2 * lambda_K,    r = log(lambda_M / lambda_K)

from the team locks alone (`predictions/gw*_team_locked_*.csv`, written by
`scripts/lock_team.py`). I needs no outcome: it is the Poisson information about beta, and
1/sqrt(I) approximates its standard error. The run then asks whether a LOOK is due:

  * interim  I >= 5.55 and >= 80 admissible matches, not yet taken
  * final    I >= 11.1, or GW38 complete (hard stop)

If no look is due it prints I, the match count and the next threshold, and EXITS WITHOUT
READING A SINGLE SCORE. That refusal is the point of the file (prereg §8, §0.4): the
harness runs every study on every run, so a study that joined outcomes on every run would
print the answer to whoever ran the tests. `load_outcomes` is called from exactly one
place, `take_look`, and only after the gate has passed; the selftest proves it by
handing the run a loader that raises if it is called early.

A look is ONE-WAY. It appends to `studies/model_vs_market_looks.json` and is never taken
twice. The interim may only stop the study, and only on region A or B (CI entirely on one
side of 1/2). An interim that does not stop records `continue` and NOTHING ELSE — no
beta, no CI, no score difference is printed or stored, because "no other interim reading
triggers any action or any write-up" (§6) is only enforceable if there is no reading.

ESTIMATOR (prereg §1, §6)
-------------------------
Primary: beta in E[y] = lambda_K * exp(beta * r), no intercept, Poisson quasi-likelihood
by Newton, sandwich variance clustered by MATCH, and again by GAMEWEEK. The repeated CI
is beta_hat +/- z * SE with O'Brien-Fleming z = 2.797 (interim) / 1.977 (final).
Classification regions A-F exactly as §6. "If the two SEs disagree, record the weaker":
both CIs share one centre, so they are nested and the WIDER one excludes a subset of what
the narrower excludes — its region is the weaker by construction. The recorded region is
the one from max(SE_match, SE_gameweek); both are reported.

Secondaries (§9) are computed at the deciding look only, all reported, none classified.

ADMISSIBILITY (prereg §4), applied per MATCH, before any outcome is joined
------------------------------------------------------------------------
Which lock: `_deadline` wins; `_early` only if no `_deadline` exists; `_LATE_` never; two
of the same label -> the later date (prereg §12, 2026-09-16). A match is excluded, with
its reason logged, when: the lock was written at or after the deadline; the lock's
`admissible` is not `ok`; either side lacks market or model lambda; or the match did not
finish inside its scheduled gameweek. "Finished" is read by `finished_fixtures`, which
drops the score columns before they leave the function. Only COMPLETED gameweeks
(gameweek_summaries `finished`) contribute to I; a gameweek still being played counts as
pending, not excluded.

Run:  python studies/model_vs_market.py              # selftest, then status (or a due look)
      python studies/model_vs_market.py --status     # status only: no selftest, and
                                                     # no look taken even if one is due
      python studies/model_vs_market.py --selftest
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import glob
import json
import re
import datetime as _dt
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import config

FIRST_GW, LAST_GW = 5, 38
I_INTERIM, I_FINAL = 5.55, 11.1
MIN_MATCHES_INTERIM = 80
Z = {"interim": 2.797, "final": 1.977}
PREREG = "docs/MODEL_VS_MARKET_PREREG_2026-09-16.md"
_LOCK_RE = re.compile(r"^gw(\d+)_team_locked_(\d{4}-\d{2}-\d{2})_(early|deadline)\.csv$")


# ------------------------------------------------------------------ locks
def select_locks(pred_dir=None):
    """gw -> lock path, by the rule fixed in prereg §12 (2026-09-16): `_deadline` wins,
    `_early` only without one, `_LATE_` never, ties on label -> the later date."""
    pred_dir = pred_dir or config.PREDICTIONS
    best = {}
    for p in glob.glob(_os.path.join(pred_dir, "gw*_team_locked_*.csv")):
        m = _LOCK_RE.match(_os.path.basename(p))
        if not m:                                  # _LATE_not_a_prediction, or anything odd
            continue
        gw, date, label = int(m.group(1)), m.group(2), m.group(3)
        key = (1 if label == "deadline" else 0, date)
        if gw not in best or key > best[gw][0]:
            best[gw] = (key, p)
    return {gw: p for gw, (_, p) in sorted(best.items())}


def draws_path(lock_path):
    d, name = _os.path.split(lock_path)
    return _os.path.join(d, "team_draws",
                         name.replace("_team_locked_", "_team_draws_").replace(".csv", ".csv.gz"))


def load_locks(pred_dir=None, first_gw=FIRST_GW):
    frames = []
    for gw, p in select_locks(pred_dir).items():
        if gw < first_gw:
            continue
        L = pd.read_csv(p)
        frames.append(L.assign(lock_file=_os.path.basename(p), lock_path=p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ------------------------------------------------------------------ outcome-free facts
def completed_gws(gw_frame):
    """Gameweek ids flagged finished. Keyed by id; the feed is not sorted."""
    f = gw_frame["finished"].astype(str).str.lower().isin(["true", "1", "yes"])
    return set(int(i) for i in gw_frame.loc[f, "id"])


def finished_fixtures(played=None):
    """{(gw, home, away)} of league matches finished inside their gameweek.

    The ONLY place the study reads a match file before a look. Score columns are dropped
    here and never leave the function."""
    if played is None:
        import inseason
        played = inseason.played(require_xg=False, verbose=False)
    if played is None or not len(played):
        return set()
    keep = played[["gameweek", "home", "away"]]
    return set((int(g), h, a) for g, h, a in zip(keep["gameweek"], keep["home"], keep["away"]))


def admissible(L, done_gws, finished):
    """Per match: `status` in {admissible, pending, excluded} and a `reason`.

    Every row of the selected locks is returned, so every exclusion is logged. A lock-level
    exclusion (flags, timing, missing lambda) is reported as soon as the lock exists, even
    while its gameweek is still being played; only the finished-in-gameweek test waits for
    the gameweek to complete."""
    if not len(L):
        return pd.DataFrame()
    A = L.copy()
    A["home"] = np.where(A["is_home"] == 1, A["team"], A["opponent"])
    A["away"] = np.where(A["is_home"] == 1, A["opponent"], A["team"])
    A["match"] = A["gw"].astype(int).astype(str) + ":" + A["home"] + "-" + A["away"]
    reasons = pd.Series("", index=A.index)

    def add(mask, why):
        reasons.loc[mask & (reasons == "")] = why

    locked = pd.to_datetime(A["locked_at_utc"], utc=True, errors="coerce")
    dl = pd.to_datetime(A["deadline_utc"], utc=True, errors="coerce")
    add(locked.isna() | dl.isna() | (locked >= dl), "lock not written before the deadline")
    add(A["admissible"].astype(str) != "ok", "lock flags inadmissible")
    for c in ("lam_for", "lam_against", "mkt_lam_for", "mkt_lam_against"):
        v = pd.to_numeric(A[c], errors="coerce")
        add(v.isna() | (v <= 0), f"missing {c}")
    # match-level: one bad side excludes the match, with the first reason found
    first = reasons.groupby(A["match"]).agg(lambda s: next((x for x in s if x), ""))
    reasons = A["match"].map(first)
    A["status"] = "admissible"
    A.loc[reasons != "", "status"] = "excluded"
    pending = ~A["gw"].astype(int).isin(done_gws)
    fin = [(int(g), h, a) in finished for g, h, a in zip(A["gw"], A["home"], A["away"])]
    not_fin = (~pending) & (~pd.Series(fin, index=A.index)) & (reasons == "")
    reasons.loc[not_fin] = "not finished inside its scheduled gameweek"
    A.loc[not_fin, "status"] = "excluded"
    A.loc[pending & (A["status"] == "admissible"), "status"] = "pending"
    A["reason"] = reasons
    return A


def information(A):
    """(I, admissible matches) from admissible rows only. Outcome-free."""
    ok = A[A["status"] == "admissible"] if len(A) else A
    if not len(ok):
        return 0.0, 0
    # one row per team-fixture: its own lambda FOR, model and market
    r = np.log(ok["lam_for"].to_numpy(float) / ok["mkt_lam_for"].to_numpy(float))
    I = float(np.sum(r * r * ok["mkt_lam_for"].to_numpy(float)))
    return I, int(ok["match"].nunique())


# ------------------------------------------------------------------ looks
def read_looks(path=None):
    path = path or config.MODEL_VS_MARKET_LOOKS
    if not _os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def append_look(entry, path=None):
    path = path or config.MODEL_VS_MARKET_LOOKS
    looks = read_looks(path) + [entry]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(looks, fh, indent=1, default=str)
    _os.replace(tmp, path)


def due_look(I, n_matches, looks, season_done):
    taken = {l["look"] for l in looks}
    if "final" in taken:
        return None
    if I >= I_FINAL or season_done:
        return "final"
    if "interim" not in taken and I >= I_INTERIM and n_matches >= MIN_MATCHES_INTERIM:
        return "interim"
    return None


# ------------------------------------------------------------------ estimation
def qpois(y, offset, X, clusters, iters=50):
    """Poisson quasi-likelihood log E[y] = offset + X b, by Newton, with cluster-robust
    sandwich variances. Returns (b, {name: V}). `clusters` maps name -> label array."""
    y = np.asarray(y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    b = np.zeros(X.shape[1])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError(f"design not identified (rank {np.linalg.matrix_rank(X)} "
                         f"< {X.shape[1]} columns)")
    for _ in range(iters):
        mu = np.exp(offset + X @ b)
        U = X.T @ (y - mu)
        H = (X * mu[:, None]).T @ X
        step = np.linalg.solve(H, U)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = np.exp(offset + X @ b)
    Hinv = np.linalg.inv((X * mu[:, None]).T @ X)
    s = X * (y - mu)[:, None]
    V = {}
    for name, lab in clusters.items():
        S = pd.DataFrame(s).groupby(np.asarray(lab)).sum().to_numpy()
        V[name] = Hinv @ (S.T @ S) @ Hinv
    return b, V


def classify(lo, hi):
    """Prereg §6 regions for an interval [lo, hi]."""
    if hi < 0.5:
        return "A"
    if lo > 0.5:
        return "B"
    ex0 = lo > 0 or hi < 0
    ex1 = lo > 1 or hi < 1
    return {(True, True): "C", (False, True): "D", (True, False): "E",
            (False, False): "F"}[(ex0, ex1)]


REGION_TEXT = {
    "A": "market scores better",
    "B": "model scores better — stats-referee pass MANDATORY before writing it up (§6)",
    "C": "each carries information the other lacks",
    "D": "informative null: the model does not encompass the market",
    "E": "the model adds information; the market does not encompass it",
    "F": "UNINFORMATIVE — record the SE and the n it would need, never 'no different'",
}


def primary(T, z):
    """T: one row per team-fixture with y, lamM, lamK, match, gw."""
    r = np.log(T["lamM"] / T["lamK"]).to_numpy()
    b, V = qpois(T["y"], np.log(T["lamK"]).to_numpy(), r[:, None],
                 {"match": T["match"].to_numpy(), "gameweek": T["gw"].to_numpy()})
    beta = float(b[0])
    se = {k: float(np.sqrt(v[0, 0])) for k, v in V.items()}
    out = {"beta": beta, "se_match": se["match"], "se_gameweek": se["gameweek"], "z": z}
    for k in ("match", "gameweek"):
        out[f"ci_{k}"] = [beta - z * se[k], beta + z * se[k]]
        out[f"region_{k}"] = classify(*out[f"ci_{k}"])
    wide = max(se.values())
    out["ci_recorded"] = [beta - z * wide, beta + z * wide]
    out["region"] = classify(*out["ci_recorded"])
    return out


def score_diff(T):
    """D per match: sum over sides of y*r - (lamM - lamK); model minus market, nats."""
    d = T["y"] * np.log(T["lamM"] / T["lamK"]) - (T["lamM"] - T["lamK"])
    per = d.groupby(T["match"]).sum()
    n = len(per)
    return {"D_bar": float(per.mean()), "se": float(per.std(ddof=1) / np.sqrt(n)) if n > 1
            else float("nan"), "n_matches": n}


def secondaries(T, draws_loader=None, deadline_price=None):
    """Prereg §9, 1-9. Reported, never classified.

    Each secondary is computed independently: one that is not identified on the data at
    hand (a single gameweek makes the intercept, fixed-effects and H2 designs collinear)
    records its error and does not cost the others. A primary result never waits on a
    secondary."""
    out = {}
    off = np.log(T["lamK"]).to_numpy()
    r = np.log(T["lamM"] / T["lamK"]).to_numpy()
    cl = {"match": T["match"].to_numpy()}

    def guard(key, fn):
        try:
            v = fn()
            if v is not None:
                out[key] = v
        except Exception as e:                                        # noqa: BLE001
            out[key] = {"error": f"{type(e).__name__}: {e}"}

    def one_beta(D, lamK_col="lamK"):
        b, V = qpois(D["y"], np.log(D[lamK_col]).to_numpy(),
                     np.log(D["lamM"] / D[lamK_col]).to_numpy()[:, None],
                     {"match": D["match"].to_numpy()})
        return {"beta": float(b[0]), "se_match": float(np.sqrt(V["match"][0, 0])),
                "n_team_fixtures": int(len(D))}

    # 1. D-bar, overall and by gameweek
    guard("1_D_bar", lambda: score_diff(T))
    guard("1_D_bar_by_gw", lambda: {int(g): score_diff(x) for g, x in T.groupby("gw")})

    # 2. beta with an intercept
    def s2():
        b, V = qpois(T["y"], off, np.column_stack([np.ones_like(r), r]), cl)
        return {"alpha": float(b[0]), "beta": float(b[1]),
                "se_beta_match": float(np.sqrt(V["match"][1, 1]))}
    guard("2_intercept", s2)

    # 3. market price at the deadline instead of at the lock
    def s3():
        if deadline_price is None:
            return {"error": "deadline prices unavailable"}
        D = T.merge(deadline_price, on=["gw", "team", "opponent", "is_home"], how="inner")
        return one_beta(D, "mkt_dl") if len(D) else {"error": "no fixture matched"}
    guard("3_deadline_price", s3)

    # 4. Solio-sourced fixtures only
    guard("4_solio_only", lambda: one_beta(T[T["mkt_source"].astype(str) == "solio"]))

    # 5. gameweek fixed effects (a different estimand, labelled as one)
    def s5():
        G = pd.get_dummies(T["gw"].astype(int)).to_numpy(float)
        b, V = qpois(T["y"], off, np.column_stack([G, r]), cl)
        return {"beta": float(b[-1]), "se_match": float(np.sqrt(V["match"][-1, -1])),
                "note": "removes the weekly level of r; not the primary estimand"}
    guard("5_gameweek_FE", s5)

    # 6. posterior-predictive log score: the model as a Poisson mixture over its locked draws
    def s6():
        if draws_loader is None:
            return {"error": "no draws loader"}
        from scipy.special import gammaln
        rows = []
        for path, x in T.groupby("lock_path"):
            dr = draws_loader(path)
            if dr is None or not len(dr):
                continue
            for _, t in x.iterrows():
                home, away = ((t["team"], t["opponent"]) if t["is_home"] == 1
                              else (t["opponent"], t["team"]))
                d = dr[(dr["home"] == home) & (dr["away"] == away)]
                if not len(d):
                    continue
                lam = (d["lam_home"] if t["is_home"] == 1 else d["lam_away"]).to_numpy(float)
                y = t["y"]
                lp = y * np.log(lam) - lam - gammaln(y + 1)
                lp_mix = float(np.log(np.mean(np.exp(lp - lp.max()))) + lp.max())
                lp_mkt = y * np.log(t["lamK"]) - t["lamK"] - gammaln(y + 1)
                rows.append((t["match"], lp_mix - lp_mkt))
        if not rows:
            return {"error": "no draws found for any fixture"}
        sm = pd.DataFrame(rows, columns=["match", "d"]).groupby("match")["d"].sum()
        return {"D_bar": float(sm.mean()),
                "se": float(sm.std(ddof=1) / np.sqrt(len(sm))) if len(sm) > 1 else float("nan"),
                "n_matches": int(len(sm)), "note": "descriptive only (§1)"}
    guard("6_posterior_predictive", s6)

    # 7. RPS on 1X2, both sides from independent Poisson
    def s7():
        from scipy.stats import poisson
        k = np.arange(16)

        def wdl(lh, la):
            M = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
            return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])

        def rps(p, o):
            return 0.5 * np.sum((np.cumsum(p)[:2] - np.cumsum(o)[:2]) ** 2)
        rr = []
        for _, x in T.groupby("match"):
            h, a = x[x["is_home"] == 1], x[x["is_home"] == 0]
            if len(h) != 1 or len(a) != 1:
                continue
            yh, ya = float(h["y"].iloc[0]), float(a["y"].iloc[0])
            res = np.array([yh > ya, yh == ya, yh < ya], float)
            rr.append(rps(wdl(h["lamK"].iloc[0], a["lamK"].iloc[0]), res)
                      - rps(wdl(h["lamM"].iloc[0], a["lamM"].iloc[0]), res))
        rr = np.array(rr)
        return {"market_minus_model": float(rr.mean()),
                "se": float(rr.std(ddof=1) / np.sqrt(len(rr))) if len(rr) > 1 else float("nan"),
                "n_matches": int(len(rr)), "note": "positive = model better; descriptive only"}
    guard("7_rps_1x2", s7)

    # 8. spec epochs: the flag vector plus the repo HEAD the lock was written from
    def s8():
        cols = [c for c in ("MARKET_ODDS", "MARKET_WEIGHT", "ELO_WEIGHT", "INSEASON",
                            "INSEASON_UPTO", "INSEASON_W_MATCH", "INSEASON_W_PROMOTED",
                            "FPL_TRAVEL", "INJURY_IMPACT", "TEAM_OVERRIDES", "SOLIO_MARKET",
                            "repo_head_sha") if c in T.columns]
        ep = (T[cols].astype(str).agg("|".join, axis=1) if cols
              else pd.Series("all", index=T.index))
        res = {}
        for key, x in T.groupby(ep):
            try:
                res[key] = dict(one_beta(x), n_matches=int(x["match"].nunique()))
            except Exception as e:                                    # noqa: BLE001
                res[key] = {"n_matches": int(x["match"].nunique()),
                            "error": f"{type(e).__name__}: {e}"}
        return res
    guard("8_spec_epochs", s8)

    # 9. H2 — DATA-SUGGESTED (dispersion): r split into its within-gameweek projection on
    #    centred log lamK and the orthogonal remainder.
    def s9():
        c = np.log(T["lamK"]) - np.log(T["lamK"]).groupby(T["gw"]).transform("mean")
        rs = pd.Series(r, index=T.index)
        bg = ((rs * c).groupby(T["gw"]).transform("sum")
              / (c * c).groupby(T["gw"]).transform("sum"))
        r_par = (bg * c).fillna(0.0).to_numpy()
        b, V = qpois(T["y"], off, np.column_stack([r_par, r - r_par]), cl)
        return {"beta_parallel": float(b[0]), "beta_perp": float(b[1]),
                "se_parallel": float(np.sqrt(V["match"][0, 0])),
                "se_perp": float(np.sqrt(V["match"][1, 1])),
                "note": "DATA-SUGGESTED, selected after seeing GW5 gaps; triggers nothing"}
    guard("9_H2_DATA_SUGGESTED", s9)
    return out


# ------------------------------------------------------------------ outcomes (gated)
def load_outcomes(A, played=None):
    """Goals per admissible team-fixture. Called ONLY from `take_look`."""
    if played is None:
        import inseason
        played = inseason.played(require_xg=False, verbose=False)
    P = played[["gameweek", "home", "away", "home_score", "away_score"]].copy()
    P["gw"] = P["gameweek"].astype(int)
    ok = A[A["status"] == "admissible"].merge(P[["gw", "home", "away", "home_score",
                                                   "away_score"]], on=["gw", "home", "away"],
                                              how="inner")
    ok["y"] = np.where(ok["is_home"] == 1, ok["home_score"], ok["away_score"]).astype(float)
    return ok.drop(columns=["home_score", "away_score"])


def take_look(look, A, I, n, looks_path=None, out_csv=None, outcomes=load_outcomes,
              draws_loader=None, deadline_price=None, now=None, verbose=True):
    now = now or _dt.datetime.now(_dt.timezone.utc)
    O = outcomes(A)
    T = O.rename(columns={"lam_for": "lamM", "mkt_lam_for": "lamK"})
    res = primary(T, Z[look])
    stop = look == "final" or res["region"] in ("A", "B")
    base = {"look": look, "at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "I": round(I, 4),
            "n_matches": n, "gameweeks": sorted(int(g) for g in T["gw"].unique())}
    if not stop:
        # §6: an interim that does not stop yields no reading at all.
        append_look(dict(base, decision="continue"), looks_path)
        if verbose:
            print(f"[model_vs_market] INTERIM look taken at I={I:.2f}, {n} matches: "
                  f"CONTINUE. Nothing else is reported (prereg §6).")
        return {"look": look, "decision": "continue"}
    result = dict(base, decision="stop", primary=res, region_text=REGION_TEXT[res["region"]],
                  secondaries=secondaries(T, draws_loader, deadline_price))
    append_look(result, looks_path)
    if look == "interim":
        append_look(dict(base, look="final", via="interim stop"), looks_path)
    if out_csv:
        T.to_csv(out_csv, index=False)
    if verbose:
        print(json.dumps(result, indent=1, default=str))
    return result


# ------------------------------------------------------------------ run
def run(pred_dir=None, looks_path=None, out_csv=None, gw_frame=None, played=None,
        outcomes=None, draws_loader=None, deadline_price_loader=None, verbose=True,
        take=True):
    """Status, or a look if one is due and `take`. Returns a dict.

    `take=False` (the CLI's --status) reports whether a look is due and reads no outcome
    even then: inspecting the gate must never be the thing that opens it."""
    looks_path = looks_path or config.MODEL_VS_MARKET_LOOKS
    out_csv = out_csv if out_csv is not None else config.MODEL_VS_MARKET_CSV
    if gw_frame is None:
        import core_insights as ci
        gw_frame = pd.read_csv(_os.path.join(ci.UPLOADS, "gameweek_summaries.csv"))
    done = completed_gws(gw_frame)
    L = load_locks(pred_dir)
    A = admissible(L, done, finished_fixtures(played)) if len(L) else pd.DataFrame()
    I, n = information(A) if len(A) else (0.0, 0)
    looks = read_looks(looks_path)
    season_done = LAST_GW in done
    look = due_look(I, n, looks, season_done)
    status = {"I": I, "n_matches": n, "locks": sorted(set(L["lock_file"])) if len(L) else [],
              "excluded": (A.loc[A["status"] == "excluded"].groupby("reason")["match"]
                           .nunique().to_dict() if len(A) else {}),
              "pending_matches": int(A.loc[A["status"] == "pending", "match"].nunique())
              if len(A) else 0,
              "looks_taken": [l["look"] for l in looks], "due": look}
    if verbose:
        print(f"[model_vs_market] pre-registered: {PREREG}")
        print(f"  team locks: {len(status['locks'])}  admissible matches: {n}  "
              f"pending: {status['pending_matches']}  I = {I:.3f}")
        for why, k in status["excluded"].items():
            print(f"  excluded: {k} match(es) — {why}")
        nxt = ("none — the final look has been taken" if "final" in status["looks_taken"]
               else f"interim at I >= {I_INTERIM} with >= {MIN_MATCHES_INTERIM} matches"
               if "interim" not in status["looks_taken"] else f"final at I >= {I_FINAL}")
        print(f"  looks taken: {status['looks_taken'] or 'none'}   next: {nxt}")
    if look is None:
        if verbose:
            print("  no look is due — no outcome was read.")
        return status
    if not take:
        if verbose:
            print(f"  a {look.upper()} look is DUE but was not taken (inspect only) — no "
                  f"outcome was read. A bare run takes it.")
        return status
    if outcomes is None:
        outcomes = lambda A_: load_outcomes(A_, played)                 # noqa: E731
    if draws_loader is None:
        def draws_loader(lock_path):
            p = draws_path(lock_path)
            return pd.read_csv(p) if _os.path.exists(p) else None
    dp = None
    try:
        dp = (deadline_price_loader or _deadline_price)()
    except Exception as e:                                            # noqa: BLE001
        if verbose:
            print(f"  secondary 3 unavailable ({type(e).__name__}: {e})")
    status["result"] = take_look(look, A, I, n, looks_path, out_csv, outcomes,
                                 draws_loader, dp, verbose=verbose)
    return status


def _deadline_price():
    import fixture_market as fm
    fx, _ = fm.fixtures(deadlines=fm.feed_deadlines())
    t = fm.per_team(fx)
    return t.rename(columns={"mkt_lam_for": "mkt_dl"})[["gw", "team", "opponent",
                                                         "is_home", "mkt_dl"]]


# ------------------------------------------------------------------ selftest
def _synthetic_season(rng, beta, n_gw=34, sigma_r=0.13, first_gw=5):
    rows = []
    for g in range(first_gw, first_gw + n_gw):
        for k in range(10):
            lk = np.exp(rng.normal(np.log([1.5, 1.2]), 0.30))
            r = rng.normal(-0.03, sigma_r, 2)
            lm = lk * np.exp(r)
            mu = lk * np.exp(beta * r)
            y = rng.poisson(mu)
            for s in (0, 1):
                rows.append(dict(gw=g, match=f"{g}:{k}", lamK=lk[s], lamM=lm[s], y=float(y[s])))
    return pd.DataFrame(rows)


def selftest(reps=500, verbose=True):
    import tempfile
    import shutil
    try:
        _sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rng = np.random.default_rng(20260916)

    # --- 1. regions, and "weaker" is the wider interval ---------------------------------
    assert classify(-0.2, 0.4) == "A" and classify(0.6, 1.4) == "B"
    assert classify(0.1, 0.9) == "C" and classify(-0.2, 0.8) == "D"
    assert classify(0.2, 1.3) == "E" and classify(-0.5, 1.5) == "F"

    # --- 2. beta recovered at 1/2, and repeated-CI coverage >= 0.93 over `reps` seasons --
    betas, cover = [], 0
    for _ in range(reps):
        S = _synthetic_season(rng, 0.5)
        half = S[S["gw"] < S["gw"].min() + 17]
        ok = True
        for T, look in ((half, "interim"), (S, "final")):
            p = primary(T, Z[look])
            lo, hi = p["ci_match"]
            ok = ok and lo <= 0.5 <= hi
        betas.append(p["beta"])
        cover += ok
    bias = float(np.mean(betas) - 0.5)
    coverage = cover / reps
    assert abs(bias) < 0.05, f"beta biased at 1/2: {bias:+.3f}"
    assert coverage >= 0.93, f"repeated-CI coverage {coverage:.3f} < 0.93"

    # --- 3. score-difference identity: E[D] ~ lambda r^2 (beta - 1/2) --------------------
    big = pd.concat([_synthetic_season(rng, 1.0, sigma_r=0.3).assign(
        match=lambda x, k=k: f"s{k}:" + x["match"]) for k in range(4)], ignore_index=True)
    assert score_diff(big)["D_bar"] > 0, "a model that IS the truth must score better"

    # --- 4. the gate: no outcome is read before a threshold, looks are one-way ----------
    d = tempfile.mkdtemp()
    try:
        pred = _os.path.join(d, "predictions")
        _os.makedirs(pred)
        looks = _os.path.join(d, "looks.json")
        out_csv = _os.path.join(d, "mvm.csv")
        called = []

        def write_lock(gw, label, date, rows, **meta):
            df = pd.DataFrame(rows)
            base = dict(locked_at_utc=f"{date}T10:00:00Z", deadline_utc="2026-12-31T00:00:00Z",
                        admissible="ok", mkt_source="solio")
            base.update(meta)
            df.assign(**base).to_csv(_os.path.join(pred, f"gw{gw}_team_locked_{date}_{label}.csv"),
                                     index=False)

        def fixtures_for(gw, lamK, lamM):
            rows = []
            for k in range(10):
                h, a = f"H{k}", f"A{k}"
                for team, opp, ih in ((h, a, 1), (a, h, 0)):
                    rows.append(dict(gw=gw, team=team, opponent=opp, is_home=ih,
                                     lam_for=lamM, lam_against=lamM, mkt_lam_for=lamK,
                                     mkt_lam_against=lamK))
            return rows

        # lock selection: deadline beats early, LATE never, later deadline beats earlier
        write_lock(5, "early", "2026-09-16", fixtures_for(5, 1.3, 1.3))
        write_lock(5, "deadline", "2026-09-18", fixtures_for(5, 1.3, 1.3 * np.exp(0.5)))
        open(_os.path.join(pred, "gw5_team_locked_2026-09-19_deadline_LATE_not_a_prediction.csv"),
             "w").close()
        assert _os.path.basename(select_locks(pred)[5]) == "gw5_team_locked_2026-09-18_deadline.csv"

        gw_frame = pd.DataFrame({"id": [6, 5], "finished": [False, True]})
        played = pd.DataFrame([dict(gameweek=5, home=f"H{k}", away=f"A{k}", home_score=1,
                                    away_score=1) for k in range(10)])

        def forbidden(A_):
            called.append("outcomes")
            raise AssertionError("OUTCOMES READ BEFORE A LOOK WAS DUE")

        K = dict(pred_dir=pred, looks_path=looks, out_csv=out_csv, played=played,
                 draws_loader=lambda p: None, deadline_price_loader=lambda: None,
                 verbose=False)
        # I for GW5: 20 team-fixtures * 0.25 * 1.3 = 6.5, but 10 matches < 80 -> no look
        st = run(gw_frame=gw_frame, outcomes=forbidden, **K)
        assert st["due"] is None and not called and abs(st["I"] - 6.5) < 1e-9, st
        assert not _os.path.exists(looks) and not _os.path.exists(out_csv)

        # a postponed match (not in `played`) is excluded, logged, not silently dropped
        st = run(gw_frame=gw_frame, outcomes=forbidden,
                 **dict(K, played=played[played["home"] != "H0"]))
        assert st["excluded"] == {"not finished inside its scheduled gameweek": 1}, st
        assert abs(st["I"] - 6.5 * 9 / 10) < 1e-9, st

        # a clean lock for a gameweek still being played is PENDING, adds nothing to I
        lock6 = _os.path.join(pred, "gw6_team_locked_2026-09-25_deadline.csv")
        write_lock(6, "deadline", "2026-09-25", fixtures_for(6, 1.3, 1.3 * np.exp(0.5)))
        st = run(gw_frame=gw_frame, outcomes=forbidden, **K)
        assert st["pending_matches"] == 10 and not st["excluded"], st
        assert abs(st["I"] - 6.5) < 1e-9, st
        # an inadmissible lock is excluded, logged per match, and adds nothing — whether or
        # not its gameweek has finished
        _os.remove(lock6)
        write_lock(6, "deadline", "2026-09-25", fixtures_for(6, 1.3, 1.3 * np.exp(0.5)),
                   admissible="inadmissible_flags: ELO_WEIGHT=0.5")
        for fin6 in (False, True):
            st = run(gw_frame=pd.DataFrame({"id": [6, 5], "finished": [fin6, True]}),
                     outcomes=forbidden, **K)
            assert st["excluded"] == {"lock flags inadmissible": 10}, st
            assert abs(st["I"] - 6.5) < 1e-9, "an excluded gameweek must add no information"
        # a lock whose own timestamps say it was written at or after the deadline is excluded
        _os.remove(lock6)
        write_lock(6, "deadline", "2026-09-25", fixtures_for(6, 1.3, 1.3 * np.exp(0.5)),
                   locked_at_utc="2026-09-26T18:00:00Z", deadline_utc="2026-09-26T17:30:00Z")
        st = run(gw_frame=pd.DataFrame({"id": [6, 5], "finished": [True, True]}),
                 outcomes=forbidden, **K)
        assert st["excluded"] == {"lock not written before the deadline": 10}, st
        _os.remove(lock6)
        write_lock(6, "deadline", "2026-09-25", fixtures_for(6, 1.3, 1.3 * np.exp(0.5)),
                   admissible="inadmissible_flags: ELO_WEIGHT=0.5")

        # GW2-GW4 are outside the study (§4) and are never loaded, even if a lock exists
        lock4 = _os.path.join(pred, "gw4_team_locked_2026-09-12_deadline.csv")
        write_lock(4, "deadline", "2026-09-12", fixtures_for(4, 1.3, 1.3 * np.exp(0.5)))
        st = run(gw_frame=pd.DataFrame({"id": [4, 5], "finished": [True, True]}),
                 outcomes=forbidden, **K)
        assert not any(f.startswith("gw4_") for f in st["locks"]), st["locks"]
        _os.remove(lock4)

        # enough matches for an interim: many completed weeks at I well past 5.55
        for g in range(7, 16):
            write_lock(g, "deadline", "2026-10-01", fixtures_for(g, 1.3, 1.3 * np.exp(0.2)))
        many = pd.DataFrame({"id": list(range(5, 17)), "finished": [True] * 11 + [False]})
        played_many = pd.DataFrame([dict(gameweek=g, home=f"H{k}", away=f"A{k}",
                                         home_score=int(rng.poisson(1.3)),
                                         away_score=int(rng.poisson(1.3)))
                                    for g in range(5, 16) for k in range(10)])
        interim_I = run(gw_frame=many, outcomes=forbidden, take=False,
                        **dict(K, pred_dir=pred, played=played_many))
        # GW6 is inadmissible, so 100 admissible matches; I = 6.5 + 9*20*0.04*1.3 = 15.86 -> FINAL
        assert interim_I["due"] == "final", interim_I

        # the interim path: remove weeks until I sits between the thresholds with >= 80 matches
        for g in range(12, 16):
            _os.remove(_os.path.join(pred, f"gw{g}_team_locked_2026-10-01_deadline.csv"))
        for g in range(7, 12):                      # weaken the disagreement
            p_ = _os.path.join(pred, f"gw{g}_team_locked_2026-10-01_deadline.csv")
            _os.remove(p_)
        for g in range(7, 16):
            write_lock(g, "deadline", "2026-10-02", fixtures_for(g, 1.3, 1.3 * np.exp(0.05)))
        st = run(gw_frame=many, outcomes=forbidden, take=False, **dict(K, played=played_many))
        # I = 6.5 + 9*20*0.0025*1.3 = 7.085, 100 matches -> interim due
        assert st["due"] == "interim" and not called, st

        # take it with real outcomes: a continue records NOTHING but the decision
        res = run(gw_frame=many, **dict(K, played=played_many))
        L1 = read_looks(looks)
        # r = 0.05 on nine of ten weeks carries almost no information: this interim cannot
        # reach region A or B, so it must CONTINUE, and the branch below must run
        assert res["result"]["decision"] == "continue", res
        assert len(L1) == 1 and set(L1[0]) == {"look", "at_utc", "I", "n_matches",
                                              "gameweeks", "decision"}, L1
        assert "beta" not in json.dumps(L1) and not _os.path.exists(out_csv)
        # one-way: the interim is never taken again
        st = run(gw_frame=many, outcomes=forbidden, **dict(K, played=played_many))
        assert st["due"] is None and not called, st
        # and after the final, nothing is ever due again
        append_look({"look": "final", "decision": "stop"}, looks)
        st = run(gw_frame=pd.DataFrame({"id": list(range(5, 39)), "finished": [True] * 34}),
                 outcomes=forbidden, **dict(K, played=played_many))
        assert st["due"] is None and not called, st

        # hard stop: GW38 complete forces the final even at low I
        looks2 = _os.path.join(d, "looks2.json")
        st = run(gw_frame=pd.DataFrame({"id": [38, 5], "finished": [True, True]}),
                 **dict(K, looks_path=looks2, pred_dir=pred, played=played))
        assert st["due"] == "final" and read_looks(looks2)[0]["look"] == "final", st
    finally:
        shutil.rmtree(d, ignore_errors=True)

    if verbose:
        print(f"SELFTEST OK: regions A-F; beta at 1/2 unbiased ({bias:+.3f}) with repeated-CI "
              f"coverage {coverage:.3f} over {reps} synthetic seasons; a true model scores "
              f"better; lock selection per prereg §12; no outcome read before a look is due; "
              f"exclusions logged per match, pending weeks not excluded; an interim continue "
              f"records only the decision; looks are one-way; GW38 forces the final.")
    return 0


if __name__ == "__main__":
    try:
        _sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if "--selftest" in _sys.argv:
        _sys.exit(selftest())
    if "--status" in _sys.argv:
        run(take=False)
    else:
        selftest()
        run()
