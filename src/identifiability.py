"""
identifiability.py — can a market-odds E0 identify the team model on its own?

TeamModel factorises each fixture's expected goals as
    log lam_home = mu + home + att[H] - dfn[A]
    log lam_away = mu        + att[A] - dfn[H]
so a set of fixtures gives 2 observations per fixture against 2 + 2*T free
parameters (T teams), less 2 for the standard centring constraints
(sum att = sum dfn = 0) and less 1 for the joint shift att+c, dfn+c which leaves
every att[i]-dfn[j] unchanged.

The Odds API only prices fixtures that are open for betting -- in practice the
next round or two. This module answers, empirically, how many gameweeks of
forward odds are needed before the design matrix has full column rank, i.e.
before a market-only E0 can be fitted WITHOUT the backward-looking E0_recon.

Run:  python identifiability.py --fixtures cs_fixtures_gw1_10.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def design_matrix(fixtures: pd.DataFrame, teams: list[str]) -> np.ndarray:
    """Rows = 2 per fixture (home goals, away goals). Columns = [mu, home, att*T, dfn*T]."""
    T = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    rows = []
    for _, f in fixtures.iterrows():
        h, a = idx[f.HomeTeam], idx[f.AwayTeam]
        r = np.zeros(2 + 2 * T); r[0] = 1; r[1] = 1; r[2 + h] = 1; r[2 + T + a] = -1
        rows.append(r)                                   # log lam_home
        r = np.zeros(2 + 2 * T); r[0] = 1; r[2 + a] = 1; r[2 + T + h] = -1
        rows.append(r)                                   # log lam_away
    return np.array(rows)


def free_params(T: int) -> int:
    """2 (mu, home) + 2T (att, dfn) - 2 centring - 1 joint shift."""
    return 2 + 2 * T - 3


def rank_report(fixtures: pd.DataFrame, teams: list[str]) -> dict:
    X = design_matrix(fixtures, teams)
    r = np.linalg.matrix_rank(X, tol=1e-9)
    k = free_params(len(teams))
    # which teams appear at all
    seen = set(fixtures.HomeTeam) | set(fixtures.AwayTeam)
    return {"fixtures": len(fixtures), "obs": X.shape[0], "rank": int(r),
            "needed": k, "identified": bool(r >= k),
            "teams_seen": len(seen), "teams_missing": sorted(set(teams) - seen)}


def main(path: str):
    d = pd.read_csv(path)
    fx = (d[d.venue == "H"][["gw", "team", "opp"]]
          .rename(columns={"team": "HomeTeam", "opp": "AwayTeam"})
          .drop_duplicates().sort_values("gw").reset_index(drop=True))
    teams = sorted(set(fx.HomeTeam) | set(fx.AwayTeam))
    T = len(teams)

    print("=" * 78)
    print(f"IDENTIFIABILITY OF A MARKET-ONLY E0   ({T} teams, "
          f"{free_params(T)} free parameters)")
    print("=" * 78)
    print(f"{'GWs of odds':>12} {'fixtures':>9} {'obs':>5} {'rank':>5} "
          f"{'needed':>7} {'teams':>6}  identified?")
    first_ok = None
    for g in sorted(fx.gw.unique()):
        sub = fx[fx.gw <= g]
        rep = rank_report(sub, teams)
        flag = "YES" if rep["identified"] else f"no  (short {rep['needed']-rep['rank']})"
        print(f"{'GW1-'+str(g):>12} {rep['fixtures']:>9} {rep['obs']:>5} "
              f"{rep['rank']:>5} {rep['needed']:>7} {rep['teams_seen']:>6}  {flag}")
        if rep["identified"] and first_ok is None:
            first_ok = g

    print()
    if first_ok:
        print(f"=> a market-only E0 becomes identified at GW1-{first_ok} "
              f"({len(fx[fx.gw <= first_ok])} fixtures).")
    else:
        print("=> never identified within the supplied horizon.")
    print("   The Odds API typically prices only the NEXT round (10 fixtures, 20 obs),")
    print(f"   which is {free_params(T)-rank_report(fx[fx.gw==1], teams)['rank']} "
          "columns short. Fitting on that alone leaves most of att/dfn")
    print("   determined by the prior, not the market -- the numbers would look")
    print("   plausible and mean nothing. Market rows must be STACKED onto E0_recon.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="cs_fixtures_gw1_10.csv")
    main(ap.parse_args().fixtures)
