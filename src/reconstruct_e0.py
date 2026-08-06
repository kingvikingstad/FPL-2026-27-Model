"""
reconstruct_e0.py — build a football-data-style E0.csv from the repo's real
25/26 Opta match xG, so the existing team model (betting_features.build ->
team_ratings -> TeamModel.fit) runs unchanged when the original odds-based
E0.csv is unavailable.

Method: for each PL match, take home/away expected_goals_xg as the true
per-fixture goal rates, then generate FAIR (overround-free) 1X2 + O/U 2.5 odds
from an independent bivariate Poisson. betting_features de-vigs these back to
the same lambdas, so the round-trip is faithful (validated: team att/def
ratings are football-sound — Man City ~2.02 attack, Arsenal ~0.75 defence).

Edit REPO / OUT for your environment.
"""
import glob
import numpy as np, pandas as pd
from scipy.stats import poisson

REPO = "/home/claude/repo/FPL-Core-Insights-main/data"
OUT = "E0_recon.csv"
SEASON = "2025-2026"


def fair_odds(lh, la, maxg=15):
    ph = poisson.pmf(np.arange(maxg + 1), lh)
    pa = poisson.pmf(np.arange(maxg + 1), la)
    P = np.outer(ph, pa)
    p_home = np.tril(P, -1).sum(); p_draw = np.trace(P); p_away = np.triu(P, 1).sum()
    tot = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / tot, p_draw / tot, p_away / tot
    idx = np.add.outer(np.arange(maxg + 1), np.arange(maxg + 1))
    p_over = 1 - P[idx <= 2].sum()
    return p_home, p_draw, p_away, p_over


def build():
    t = pd.read_csv(f"{REPO}/{SEASON}/teams.csv")
    name = dict(zip(t.code, t.name)); name.update({1: "Man United", 6: "Tottenham"})
    files = sorted(glob.glob(f"{REPO}/{SEASON}/By Gameweek/GW*/matches.csv"))
    M = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    M = M[M.match_id.astype(str).str.contains("-prem-", na=False) & (M.finished == True)].copy()
    M["kickoff_time"] = pd.to_datetime(M.kickoff_time, errors="coerce")
    rows = []
    for _, r in M.iterrows():
        h, a = name.get(r.home_team), name.get(r.away_team)
        if h is None or a is None:
            continue
        lh, la = r.home_expected_goals_xg, r.away_expected_goals_xg
        if not np.isfinite(lh) or not np.isfinite(la) or (lh == 0 and la == 0):
            lh = r.home_score if np.isfinite(r.home_score) else 1.4
            la = r.away_score if np.isfinite(r.away_score) else 1.4
        lh, la = max(float(lh), 0.05), max(float(la), 0.05)
        ph, pd_, pa, po = fair_odds(lh, la)
        dt = r.kickoff_time.strftime("%d/%m/%Y") if pd.notna(r.kickoff_time) else "01/01/2026"
        rows.append(dict(Date=dt, HomeTeam=h, AwayTeam=a,
                         FTHG=int(r.home_score), FTAG=int(r.away_score),
                         AvgH=1 / ph, AvgD=1 / pd_, AvgA=1 / pa,
                         **{"Avg>2.5": 1 / po, "Avg<2.5": 1 / (1 - po)}))
    E0 = pd.DataFrame(rows)
    E0.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(E0)} rows, {E0.HomeTeam.nunique()} teams")
    return E0


if __name__ == "__main__":
    build()
