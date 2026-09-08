"""
betting_odds_ingest.py — resolves handoff §7 item #1 (live 26/27 team-strength odds).

Feed swap, not new architecture. Converts a forward-looking odds snapshot into an
E0-format match file whose "goals" columns hold MARKET-IMPLIED expected goals
(lambda_home, lambda_away) per fixture, so TeamModel.fit() factorises them into
att[]/dfn[] exactly as it does the reconstructed 25/26 Opta-xG E0 — except these
price in new managers (De Zerbi->TOT, Alonso->CHE, Iraola->LIV) and the promoted
sides that have no PL history.

Pipeline:  fixtures.csv  --de-vig 1X2 + O/U2.5-->  (P(H),P(D),P(A),P(over))
                          --Poisson inversion-->   (lambda_H, lambda_A)
                          --write E0 rows-->        /tmp/E0_market.csv  --> fit(e0_path=...)

Primary source : football-data.co.uk fixtures.csv  (free, no-auth, plain CSV).
Consensus cols : Avg* / Max* (these already EXCLUDE the stale-since-Jul-2025 Pinnacle
                 fields, per the sourcing note). We never read PSH/PSD/PSA or P>2.5.
Snapshot only  : good for weekly team-strength refits; won't catch post-team-news moves.

The de-vig is a two-1D-solve inversion (robust, no 2D optimiser):
  (1) O/U 2.5 pins the total  Lambda = lam_H + lam_A   (P(total<=2) is monotone in Lambda)
  (2) 1X2 supremacy splits it  via s in (0,1), lam_H = s*Lambda   (P(H)-P(A) monotone in s)
Independent-Poisson match model by default; an optional Dixon-Coles low-score rho is
exposed but off by default (the O/U anchor already pins totals; DC mainly nudges draws).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from math import exp, factorial

_FACT = np.array([factorial(k) for k in range(31)], float)

def _bisect(f, lo, hi, xtol=1e-9, maxit=200):
    """Dependency-free 1D root-find (replaces scipy.optimize.brentq so the module
    runs on Termux, where scipy is a hard build blocker). Both call sites are
    monotone, so bisection is exact to tolerance. Falls back to the smaller-|f|
    endpoint if the root isn't bracketed (keeps batch ingest crash-free)."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0: return lo
    if fhi == 0.0: return hi
    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(maxit):
        mid = 0.5 * (lo + hi); fm = f(mid)
        if fm == 0.0 or (hi - lo) < xtol:
            return mid
        if flo * fm < 0: hi = mid
        else: lo, flo = mid, fm
    return 0.5 * (lo + hi)

# ---------------------------------------------------------------- team naming
# Our canonical names (schedule_2627 / core_insights). football-data is already
# ~identical; The Odds API uses FULL club names ("Tottenham Hotspur", "Manchester
# City", "Nottingham Forest"...). One map covers both sources -> canonical.
_CANON = {"Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Chelsea",
          "Coventry","Crystal Palace","Everton","Fulham","Hull","Ipswich","Leeds",
          "Liverpool","Man City","Man United","Newcastle","Nott'm Forest",
          "Sunderland","Tottenham"}
_ALIAS = {
    # The Odds API full names
    "Tottenham Hotspur": "Tottenham", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton", "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham", "Leeds United": "Leeds", "Ipswich Town": "Ipswich",
    "Hull City": "Hull", "Coventry City": "Coventry", "Leicester City": "Leicester",
    "AFC Bournemouth": "Bournemouth", "Sunderland AFC": "Sunderland",
    # football-data / other short forms
    "Nott'ham Forest": "Nott'm Forest", "Spurs": "Tottenham", "Man Utd": "Man United",
    "Wolverhampton": "Wolves", "Sheffield Utd": "Sheffield United",
}
def norm_team(name: str) -> str:
    n = str(name).strip()
    if n in _CANON: return n
    if n in _ALIAS: return _ALIAS[n]
    # light fallback: drop trailing "FC"/"AFC" and retry
    stripped = n.replace(" FC", "").replace(" AFC", "").strip()
    if stripped in _CANON: return stripped
    return _ALIAS.get(stripped, stripped)

# ---------------------------------------------------------------- de-vig
def devig_multiplicative(odds: np.ndarray) -> np.ndarray:
    """Proportional (normalise-by-overround) de-vig. odds -> fair probs."""
    inv = 1.0 / np.asarray(odds, float)
    return inv / inv.sum()

def devig_shin(odds: np.ndarray, iters: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Shin (1992) de-vig — odds -> fair probs, correcting the favourite-longshot bias.

    Shin models the book as a fraction `z` of insider money. With published implied
    probabilities pi_i = 1/odds_i and booksum PI = sum(pi), the fair probability is

        p_i = [ sqrt(z^2 + 4(1-z) * pi_i^2 / PI) - z ] / (2(1-z))

    and `z` is the root of  sum_i sqrt(z^2 + 4(1-z) pi_i^2 / PI) = 2 + z(n-2),
    which is what the fixed-point iteration below solves. The resulting p sums to 1 by
    construction, so it must NOT be renormalised — renormalising is what makes the
    correction vanish.

    Reduces to proportional de-vig at z=0 (a book with no insider money), and shifts
    probability TOWARD the favourite and away from the longshot as z rises. On a typical
    1.55/4.20/6.50 market: +0.008 on the favourite, -0.005 on the longshot.

    NB (fixed 2026-08-21) the previous implementation was a no-op. Its z-update
    `sum((sqrt(PI)*p - 1)*p)` evaluates to about -0.64 on a normal three-way book, was
    clipped to 0, and the in-loop renormalisation then returned the proportional
    probabilities BITWISE. `oddsapi_feed.build_market_e0` defaults to method="shin", so
    the whole odds path was silently running proportional de-vig. `oddsapi_feed
    --selftest` asserts the two methods differ, and had been failing on exactly this.

    Two-outcome books (over/under) have no Shin solution — n=2 makes the root equation
    degenerate — so they fall back to proportional, which is what Shin reduces to there.
    """
    pi = 1.0 / np.asarray(odds, float)
    PI = pi.sum()
    n = len(pi)
    if n < 3 or PI <= 1.0:
        return pi / PI                      # no vig to remove, or a two-way book
    z = 0.0
    for _ in range(iters):
        s = np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / PI).sum()
        z_new = float(np.clip((s - 2.0) / (n - 2.0), 0.0, 0.5))
        if abs(z_new - z) < tol:
            z = z_new
            break
        z = z_new
    if z <= 0.0:
        return pi / PI
    p = (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi / PI) - z) / (2.0 * (1.0 - z))
    return p / p.sum()                      # guards float drift only; already ~1


# ---------------------------------------------------------------- free odds feed
FD_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def fetch_football_data_fixtures(url=FD_FIXTURES_URL, div="E0", timeout=30, cache=None,
                                 verbose=True):
    """Upcoming fixtures with de-vigable odds. FREE, no API key.

    WHY THIS EXISTS. `oddsapi_feed` needs `ODDS_API_KEY`, which is the only thing keeping
    the match-odds path — PROJECT_KNOWLEDGE's top open item, the attack/defence split —
    from running. It does not need to be. football-data.co.uk publishes a rolling
    `fixtures.csv` of upcoming matches carrying exactly the columns `fixtures_to_e0`
    already consumes: AvgH/AvgD/AvgA and Avg>2.5/Avg<2.5. That is not a coincidence —
    `fixtures_to_e0` was written against football-data's schema in the first place.

    So the key is optional, not required. The Odds API remains the better feed (more
    books, more frequent) and `oddsapi_feed` is unchanged; this is the zero-cost path.

    TWO GOTCHAS, both load-bearing:
      * The file is UTF-8 WITH BOM. Read as latin-1 — which the rest of this module does,
        because the historical E0 files need it — and the first column arrives named
        'i≫¿Div' rather than 'Div', so a `Div == "E0"` filter silently matches nothing
        and the caller sees an empty frame with no error. Read as utf-8-sig.
      * It is ROLLING and multi-league. It carries only the next few days across every
        division football-data covers, so outside a Premier League matchweek there may be
        no E0 rows at all. An empty result is normal, not a failure.
    """
    import io
    import urllib.request
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
        timeout=timeout).read()
    d = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    d.columns = [str(c).lstrip("﻿").strip() for c in d.columns]
    if "Div" not in d.columns:
        raise RuntimeError(f"fixtures.csv has no Div column; got {list(d.columns)[:6]}")
    if div:
        d = d[d["Div"] == div]
    need = ["Date", "HomeTeam", "AwayTeam", "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise RuntimeError(f"fixtures.csv missing {missing}")
    d = d.dropna(subset=need)
    if cache:
        d.to_csv(cache, index=False)
    if verbose:
        print(f"[football-data] {len(d)} {div or 'all'} fixture(s) with odds"
              + ("" if len(d) else " — none listed right now; the file rolls a few days "
                                  "ahead and covers every division"))
    return d.reset_index(drop=True)

# ---------------------------------------------------------------- Poisson match
def _p_total_le2(Lam: float) -> float:
    return exp(-Lam) * (1 + Lam + Lam * Lam / 2.0)

def lambda_from_over(p_over: float, line: float = 2.5, lo=1e-3, hi=12.0) -> float:
    """Invert P(total > line) = p_over for the total mean Lambda (line=2.5 => <=2)."""
    assert abs(line - 2.5) < 1e-9, "only the 2.5 line is wired (uses P(total<=2))"
    f = lambda L: (1.0 - _p_total_le2(L)) - p_over
    return _bisect(f, lo, hi, xtol=1e-8)

def _wdl(lamH: float, lamA: float, kmax: int = 15, rho: float = 0.0):
    i = np.arange(kmax + 1)
    ph = np.exp(-lamH) * lamH ** i / _FACT[:kmax + 1]
    pa = np.exp(-lamA) * lamA ** i / _FACT[:kmax + 1]
    M = np.outer(ph, pa)
    if rho:  # Dixon-Coles low-score correction
        tau = np.ones((kmax + 1, kmax + 1))
        tau[0, 0] = 1 - lamH * lamA * rho; tau[0, 1] = 1 + lamH * rho
        tau[1, 0] = 1 + lamA * rho;        tau[1, 1] = 1 - rho
        M = M * tau; M = M / M.sum()
    home = np.tril(M, -1).sum(); draw = np.trace(M); away = np.triu(M, 1).sum()
    return home, draw, away

def split_supremacy(Lam: float, pH: float, pA: float, rho: float = 0.0) -> tuple[float, float]:
    """Given total Lam, find s in (0,1) with lamH=s*Lam so model (P(H)-P(A)) matches (pH-pA)."""
    target = pH - pA
    def g(s):
        lamH, lamA = s * Lam, (1 - s) * Lam
        h, _, a = _wdl(lamH, lamA, rho=rho)
        return (h - a) - target
    # target monotone increasing in s; guard the brackets
    s = _bisect(g, 1e-4, 1 - 1e-4, xtol=1e-7)
    return s * Lam, (1 - s) * Lam

def implied_lambdas(oH, oD, oA, o_over, o_under, method="multiplicative",
                    rho: float = 0.0, line: float = 2.5):
    dv = devig_shin if method == "shin" else devig_multiplicative
    pH, pD, pA = dv(np.array([oH, oD, oA]))
    p_over = dv(np.array([o_over, o_under]))[0]
    Lam = lambda_from_over(p_over, line)
    lamH, lamA = split_supremacy(Lam, pH, pA, rho=rho)
    return lamH, lamA, dict(pH=pH, pD=pD, pA=pA, p_over=p_over, Lam=Lam)

# ---------------------------------------------------------------- fixtures.csv -> E0
# consensus column preference: Avg (market mean) is the robust default; Max (best price)
# is closer to no-vig but noisier. Both already exclude Pinnacle on football-data.
_1X2 = {"avg": ("AvgH", "AvgD", "AvgA"), "max": ("MaxH", "MaxD", "MaxA"),
        "b365": ("B365H", "B365D", "B365A")}
_OU  = {"avg": ("Avg>2.5", "Avg<2.5"), "max": ("Max>2.5", "Max<2.5"),
        "b365": ("B365>2.5", "B365<2.5")}

def fixtures_to_e0(fx: pd.DataFrame, consensus: str = "avg", method: str = "multiplicative",
                   rho: float = 0.0, goal_cols=("FTHG", "FTAG"),
                   xg_cols=("HxG", "AxG")) -> pd.DataFrame:
    """football-data fixtures.csv -> E0-format frame with market-implied expected goals.
    Writes lambda into BOTH the goals columns and explicit xG columns so the fit reads
    whichever E0_recon used. `goal_cols`/`xg_cols` let you match the exact schema."""
    h1, d1, a1 = _1X2[consensus]; ov, un = _OU[consensus]
    out = []
    for _, r in fx.iterrows():
        try:
            oH, oD, oA = float(r[h1]), float(r[d1]), float(r[a1])
            oOv, oUn = float(r[ov]), float(r[un])
        except (KeyError, ValueError):
            continue
        if not np.isfinite([oH, oD, oA, oOv, oUn]).all():
            continue
        lamH, lamA, meta = implied_lambdas(oH, oD, oA, oOv, oUn, method=method, rho=rho)
        out.append({
            "Div": r.get("Div", "E0"), "Date": r.get("Date", ""),
            "HomeTeam": norm_team(r["HomeTeam"]), "AwayTeam": norm_team(r["AwayTeam"]),
            goal_cols[0]: round(lamH, 4), goal_cols[1]: round(lamA, 4),
            xg_cols[0]: round(lamH, 4), xg_cols[1]: round(lamA, 4),
            "FTR": "H" if lamH > lamA else ("A" if lamA > lamH else "D"),
            "mkt_pH": round(meta["pH"], 4), "mkt_pA": round(meta["pA"], 4),
            "mkt_total": round(meta["Lam"], 4),
        })
    return pd.DataFrame(out)

def load_footballdata_fixtures(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# ---------------------------------------------------------------- The Odds API v4 -> fixtures
def oddsapi_to_fixtures(events, line: float = 2.5, min_books: int = 1) -> pd.DataFrame:
    """The Odds API v4 /odds JSON (list of events, markets=h2h,totals) -> a frame in
    football-data fixtures schema (AvgH/D/A, MaxH/D/A, Avg/Max >2.5/<2.5) so it flows
    straight through fixtures_to_e0. Consensus = mean(Avg)/best(Max) across bookmakers.

    Events with no O/U at `line` (depth thins for far GWs) are dropped and counted —
    the total anchor is required for the inversion.
    """
    import json
    if isinstance(events, str):
        events = json.loads(events)
    rows, skipped_ou = [], 0
    for ev in events:
        home = norm_team(ev.get("home_team", "")); away = norm_team(ev.get("away_team", ""))
        h, d, a, ov, un = [], [], [], [], []
        for bk in ev.get("bookmakers", []):
            mk = {m["key"]: m for m in bk.get("markets", [])}
            if "h2h" in mk:
                o = {norm_team(x["name"]) if x["name"] != "Draw" else "Draw": x["price"]
                     for x in mk["h2h"]["outcomes"]}
                if home in o and away in o and "Draw" in o:
                    h.append(o[home]); d.append(o["Draw"]); a.append(o[away])
            if "totals" in mk:
                pts = {(x["name"], x.get("point")): x["price"] for x in mk["totals"]["outcomes"]}
                if ("Over", line) in pts and ("Under", line) in pts:
                    ov.append(pts[("Over", line)]); un.append(pts[("Under", line)])
        if len(h) < min_books:            # no usable 1X2
            continue
        if not ov or not un:              # no O/U anchor at the line
            skipped_ou += 1; continue
        rows.append({
            "Div": "E0", "Date": ev.get("commence_time", "")[:10],
            "HomeTeam": home, "AwayTeam": away,
            "AvgH": np.mean(h), "AvgD": np.mean(d), "AvgA": np.mean(a),
            "MaxH": np.max(h), "MaxD": np.max(d), "MaxA": np.max(a),
            "Avg>2.5": np.mean(ov), "Avg<2.5": np.mean(un),
            "Max>2.5": np.max(ov),  "Max<2.5": np.max(un),
        })
    if skipped_ou:
        import warnings; warnings.warn(f"{skipped_ou} events dropped (no O/U {line} depth)")
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- unified entry point
def build_market_e0(source: str, data, consensus: str = "avg",
                    method: str = "multiplicative", rho: float = 0.0,
                    goal_cols=("FTHG", "FTAG"), xg_cols=("HxG", "AxG"),
                    out_path: str | None = None) -> pd.DataFrame:
    """ONE call for run_2627. source in {'footballdata','oddsapi'}.
      data: a path/URL (footballdata csv) OR a parsed/JSON-string events list (oddsapi).
    Returns the E0-format frame; writes it if out_path given.
    """
    if source == "footballdata":
        fx = load_footballdata_fixtures(data)
    elif source == "oddsapi":
        fx = oddsapi_to_fixtures(data)
    else:
        raise ValueError("source must be 'footballdata' or 'oddsapi'")
    e0 = fixtures_to_e0(fx, consensus=consensus, method=method, rho=rho,
                        goal_cols=goal_cols, xg_cols=xg_cols)
    if out_path:
        e0.to_csv(out_path, index=False)
    return e0

# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="footballdata", choices=["footballdata", "oddsapi"])
    ap.add_argument("--fixtures", required=True,
                    help="footballdata: fixtures.csv path/URL | oddsapi: saved /odds JSON path")
    ap.add_argument("--out", default="/tmp/E0_market.csv")
    ap.add_argument("--consensus", default="avg", choices=list(_1X2))
    ap.add_argument("--method", default="multiplicative", choices=["multiplicative", "shin"])
    ap.add_argument("--rho", type=float, default=0.0, help="Dixon-Coles low-score rho (0=off)")
    a = ap.parse_args()
    data = a.fixtures
    if a.source == "oddsapi":
        with open(a.fixtures) as fh:
            data = json.load(fh)
    e0 = build_market_e0(a.source, data, consensus=a.consensus, method=a.method,
                         rho=a.rho, out_path=a.out)
    print(f"{len(e0)} fixtures -> {a.out}")
    print(e0[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "mkt_total"]].to_string(index=False))
