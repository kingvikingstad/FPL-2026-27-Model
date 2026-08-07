"""
market_odds.py — live betting-market signal for the 2026/27 projection
======================================================================
Closes the project's single biggest documented accuracy gap: team strength has
been running on backward-looking 25/26 Opta xG + ClubElo, with no forward-looking
market input. The market is the sharpest available consensus on team strength —
it prices in transfers, managerial change, injuries and preseason form that
last-season data cannot see.

TWO INDEPENDENT SOURCES, BOTH FREE AND KEY-LESS
-----------------------------------------------
1. PER-FIXTURE ODDS (preferred — highest information).
   football-data.co.uk publishes `fixtures.csv`, a rolling file of UPCOMING
   fixtures carrying 1X2, over/under 2.5 and Asian-handicap prices from ~10
   bookmakers, in exactly the same column schema as the historical E0.csv the
   pipeline already parses. De-vig the 1X2 + O/U pair and invert the independent
   -Poisson to recover (lambda_home, lambda_away) per fixture — the same
   machinery in `betting_features.py`, pointed at forward fixtures instead of
   played ones.

   CAVEAT ON TIMING: bookmakers post EPL prices roughly 1-2 weeks before kickoff,
   so early in preseason this file contains only leagues already underway (e.g.
   the Scottish divisions in early August) and NO `E0` rows. That is expected,
   not an error — `load_fixture_odds` returns an empty frame and the caller
   falls back to source 2. Re-run closer to GW1 and the odds appear.

2. SEASON OUTRIGHT ODDS (the fallback proxy — always available).
   Title / top-4 / relegation prices exist year-round and encode the market's
   full-season view of every club, including promoted sides with no top-flight
   data at all. `outright_to_elo` converts them into a pseudo-Elo rating frame
   that drops straight into `TeamModel.fit(clubelo=...)`, reusing the existing
   blend machinery with no model changes. Lower resolution than per-fixture odds
   (no venue or opponent detail) but it covers all 20 clubs from day one.

   These are entered by hand in an editable CSV — see `write_outright_template`.
   Hand entry is deliberate: outright markets are not published in any free,
   stable, machine-readable feed, and copying 20 numbers off a price comparison
   page once a season is cheaper than maintaining a scraper against a site that
   does not want to be scraped.

WHY NOT A PAID ODDS API
-----------------------
The Odds API / SportsGameOdds / BallDontLie all cover EPL with richer markets and
live in-play prices, but every one is key-gated and the useful tiers are paid.
Hooks are left in `run_2627.py` (`--props`) for anyone who wants to wire one in;
the two sources above keep the default pipeline free and unauthenticated.
"""
from __future__ import annotations
import io, urllib.request
import numpy as np, pandas as pd

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# football-data.co.uk / bookmaker naming -> the model's short names
ODDS_NAME = {
    "Man United": "Man United", "Man Utd": "Man United", "Manchester United": "Man United",
    "Man City": "Man City", "Manchester City": "Man City",
    "Tottenham": "Tottenham", "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Nott'm Forest": "Nott'm Forest", "Nottingham Forest": "Nott'm Forest",
    "Newcastle": "Newcastle", "Newcastle United": "Newcastle",
    "Bournemouth": "Bournemouth", "AFC Bournemouth": "Bournemouth",
    "Brighton": "Brighton", "Brighton & Hove Albion": "Brighton",
    "Leeds": "Leeds", "Leeds United": "Leeds",
    "Coventry": "Coventry", "Coventry City": "Coventry",
    "Hull": "Hull", "Hull City": "Hull",
    "Ipswich": "Ipswich", "Ipswich Town": "Ipswich",
    "Crystal Palace": "Crystal Palace", "Aston Villa": "Aston Villa",
}
def norm(t):
    return ODDS_NAME.get(str(t).strip(), str(t).strip())


TEAMS_2627 = ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
              "Chelsea", "Coventry", "Crystal Palace", "Everton", "Fulham",
              "Hull", "Ipswich", "Leeds", "Liverpool", "Man City", "Man United",
              "Newcastle", "Nott'm Forest", "Sunderland", "Tottenham"]


# ---------------------------------------------------------------------------
# 1. PER-FIXTURE ODDS -> per-fixture expected goals
# ---------------------------------------------------------------------------
def _first_col(df, names):
    """Return the first present column from `names`, else None. Bookmaker
    coverage varies week to week, so never hard-code a single book."""
    for n in names:
        if n in df.columns and df[n].notna().any():
            return n
    return None


def load_fixture_odds(path_or_url=FIXTURES_URL, div="E0", timeout=30):
    """Upcoming fixtures with de-vigged market probabilities and implied
    (lambda_home, lambda_away). Returns an EMPTY frame if the division has no
    priced fixtures yet — that is the normal preseason state, not a failure."""
    if str(path_or_url).startswith("http"):
        req = urllib.request.Request(path_or_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    else:
        df = pd.read_csv(path_or_url, encoding="latin-1")
    if "Div" not in df.columns:
        return pd.DataFrame()
    df = df[df.Div == div].copy()
    if not len(df):
        return pd.DataFrame()

    # prefer the market average (sharper than any single book), fall back to B365
    cH = _first_col(df, ["AvgH", "B365H", "MaxH"])
    cD = _first_col(df, ["AvgD", "B365D", "MaxD"])
    cA = _first_col(df, ["AvgA", "B365A", "MaxA"])
    cO = _first_col(df, ["Avg>2.5", "B365>2.5", "Max>2.5"])
    cU = _first_col(df, ["Avg<2.5", "B365<2.5", "Max<2.5"])
    if not all([cH, cD, cA]):
        return pd.DataFrame()

    import betting_features as bf
    df = df.dropna(subset=[cH, cD, cA]).copy()
    if not len(df):
        return pd.DataFrame()
    p = bf.devig(df[[cH, cD, cA]].values.astype(float))
    df["pH"], df["pD"], df["pA"] = p[:, 0], p[:, 1], p[:, 2]
    if cO and cU:
        ou = df[[cO, cU]].astype(float)
        ok = ou.notna().all(axis=1)
        pov = np.full(len(df), np.nan)
        if ok.any():
            pov[ok.values] = bf.devig(ou[ok].values)[:, 0]
        df["p_over25"] = pov
    else:
        df["p_over25"] = np.nan
    # where O/U is missing, fall back to the league-average over rate so the
    # 2-parameter solve stays identified rather than dropping the fixture
    df["p_over25"] = df["p_over25"].fillna(0.53)

    lam_h, lam_a = [], []
    for _, r in df.iterrows():
        try:
            lh, la = bf.solve_lambdas(r.pH, r.pD, r.pA, r.p_over25)
        except Exception:
            lh, la = np.nan, np.nan
        lam_h.append(lh); lam_a.append(la)
    df["lambda_home"] = lam_h; df["lambda_away"] = lam_a
    df["HomeTeam"] = df.HomeTeam.map(norm); df["AwayTeam"] = df.AwayTeam.map(norm)
    keep = ["Div", "Date", "HomeTeam", "AwayTeam", "pH", "pD", "pA",
            "p_over25", "lambda_home", "lambda_away"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["lambda_home"])


def fixture_odds_to_market_map(fix, sched_long, date_col="Date"):
    """Map priced fixtures onto model gameweeks -> {(gameweek, team): (lam_for,
    lam_against)}. Matching is on the (home, away) pair, which is unambiguous
    within a season, so no date parsing is required."""
    if fix is None or not len(fix):
        return {}
    pair2gw = {}
    for _, r in sched_long[sched_long.is_home == 1].iterrows():
        pair2gw[(r.team, r.opp)] = int(r.gameweek)
    out = {}
    for _, r in fix.iterrows():
        gw = pair2gw.get((r.HomeTeam, r.AwayTeam))
        if gw is None:
            continue
        out[(gw, r.HomeTeam)] = (float(r.lambda_home), float(r.lambda_away))
        out[(gw, r.AwayTeam)] = (float(r.lambda_away), float(r.lambda_home))
    return out


# ---------------------------------------------------------------------------
# 2. SEASON OUTRIGHT ODDS -> pseudo-Elo team strength prior
# ---------------------------------------------------------------------------
OUTRIGHT_COLUMNS = ["team", "title_odds", "top4_odds", "relegation_odds",
                    "predicted_finish"]


def write_outright_template(path, teams=None):
    """Write an editable CSV of 26/27 clubs for hand-entered outright prices.

    Fill in ANY ONE of the columns and leave the rest blank:
      title_odds       decimal odds to win the league   (e.g. 2.50)
      top4_odds        decimal odds to finish top 4
      relegation_odds  decimal odds to be relegated
      predicted_finish your own 1-20 forecast, if you'd rather skip odds entirely
    More columns filled = a better-identified strength estimate, but one is enough.
    """
    teams = teams or TEAMS_2627
    df = pd.DataFrame({"team": teams})
    for c in OUTRIGHT_COLUMNS[1:]:
        df[c] = np.nan
    df.to_csv(path, index=False)
    return path


def _implied(col):
    """Decimal odds -> de-vigged probability across the field."""
    p = 1.0 / pd.to_numeric(col, errors="coerce")
    if p.notna().sum() < 2:
        return None
    return p / p.sum()


def outright_to_elo(path, base=1500.0, spread=180.0):
    """Convert hand-entered outright odds into a ClubElo-shaped frame
    (`team`, `Elo`) so it can be passed straight to `TeamModel.fit(clubelo=...)`.

    Method: each supplied market is de-vigged across the 20 clubs, converted to a
    standardised strength score (relegation odds inverted, since short relegation
    odds mean a WEAK team), the available scores are averaged, and the result is
    mapped onto the Elo scale. `spread` sets how many Elo points one standard
    deviation of market opinion is worth — 180 approximates the observed spread
    of real ClubElo ratings across a Premier League season.
    """
    d = pd.read_csv(path)
    d["team"] = d.team.map(norm)
    scores = []
    for col, invert in [("title_odds", False), ("top4_odds", False),
                        ("relegation_odds", True)]:
        if col not in d:
            continue
        p = _implied(d[col])
        if p is None:
            continue
        # log-probability is far better behaved than probability here: title
        # odds span three orders of magnitude and would otherwise be dominated
        # entirely by the single favourite.
        s = np.log(p.clip(lower=1e-6))
        s = (s - s.mean()) / (s.std() or 1.0)
        scores.append(-s if invert else s)
    if "predicted_finish" in d:
        f = pd.to_numeric(d.predicted_finish, errors="coerce")
        if f.notna().sum() >= 2:
            s = -(f - f.mean()) / (f.std() or 1.0)      # finish 1 = strongest
            scores.append(s)
    if not scores:
        raise ValueError(
            f"{path} has no usable columns — fill in at least one of "
            f"{OUTRIGHT_COLUMNS[1:]} for at least two clubs.")
    z = pd.concat(scores, axis=1).mean(axis=1)
    z = (z - z.mean()) / (z.std() or 1.0)
    return pd.DataFrame({"team": d.team, "Elo": base + spread * z}).dropna()


# ---------------------------------------------------------------------------
# Export helper — a fully editable view of the model's team inputs
# ---------------------------------------------------------------------------
def write_team_inputs(path, tsamp, market_map=None):
    """Dump the fitted team attack/defence posteriors (and any market lambdas)
    to CSV so they can be inspected or hand-overridden."""
    A, D = tsamp["att"], tsamp["dfn"]
    t = pd.DataFrame({"team": tsamp["teams"],
                      "attack": A.mean(0), "attack_sd": A.std(0),
                      "defence": D.mean(0), "defence_sd": D.std(0)})
    t["net_strength"] = t.attack + t.defence
    if market_map:
        m = {}
        for (gw, team), (lf, la) in market_map.items():
            m.setdefault(team, []).append((lf, la))
        t["market_xg_for"] = t.team.map(
            lambda x: np.mean([v[0] for v in m[x]]) if x in m else np.nan)
        t["market_xg_against"] = t.team.map(
            lambda x: np.mean([v[1] for v in m[x]]) if x in m else np.nan)
        t["market_fixtures_priced"] = t.team.map(lambda x: len(m.get(x, [])))
    return t.sort_values("net_strength", ascending=False).round(4).to_csv(path, index=False) or path


if __name__ == "__main__":
    print(__doc__.split("WHY NOT")[0])
    fx = load_fixture_odds()
    if len(fx):
        print(f"\nPriced EPL fixtures available now: {len(fx)}")
        print(fx.head(10).round(3).to_string(index=False))
    else:
        print("\nNo EPL fixtures priced yet in football-data fixtures.csv "
              "(normal >1-2 weeks before kickoff).")
        print("Use the outright-odds proxy instead:")
        print("  python -c \"import market_odds as m; m.write_outright_template('outright_odds.csv')\"")
