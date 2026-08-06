"""
roster.py — 2026/27 squad ingestion for the Bayesian model
==========================================================
Fixes the two gaps in bayes_model when moving to a real new season:

 1. TRANSFERS. A returning player's rate POSTERIORS (his skill) travel with him,
    but he is re-pointed to his 2026/27 team's fixtures and clean-sheet context.
 2. COLD-START. Players with no 25/26 Premier League history (promoted-team
    squads, new signings, youth) get an informative prior built from POSITION +
    PRICE — the market signal shown to be strongly predictive. Priors are
    calibrated from 25/26 (log-involvement and start-rate both rise with price).

Production input: the official FPL `bootstrap-static` JSON (elements + teams +
element_types). A generic roster CSV is also accepted. Team names are normalised
to the model's short names.
"""
from __future__ import annotations
import json, numpy as np, pandas as pd
import sys; sys.path.insert(0, "/home/claude/fpl")
from bayes_model import player_posteriors

# FPL/api or common names -> model short names
TEAM_NORM = {
    "Man Utd": "Man United", "Manchester Utd": "Man United", "Manchester United": "Man United",
    "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Man City": "Man City", "Manchester City": "Man City",
    "Nott'm Forest": "Nott'm Forest", "Nottingham Forest": "Nott'm Forest",
    "Newcastle": "Newcastle", "Newcastle United": "Newcastle",
    "Bournemouth": "Bournemouth", "AFC Bournemouth": "Bournemouth",
    "Brighton": "Brighton", "Brighton & Hove Albion": "Brighton",
    "Leeds": "Leeds", "Leeds United": "Leeds",
    "Coventry": "Coventry", "Coventry City": "Coventry",
    "Hull": "Hull", "Hull City": "Hull", "Ipswich": "Ipswich", "Ipswich Town": "Ipswich",
}
def norm_team(t): return TEAM_NORM.get(str(t).strip(), str(t).strip())
POS_MAP = {"GKP": "GK", "GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD",
           1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_fpl_bootstrap(path):
    """Parse the official FPL bootstrap-static JSON into a roster frame."""
    with open(path) as f:
        b = json.load(f)
    teams = {t["id"]: t["name"] for t in b["teams"]}
    etype = {e["id"]: e["singular_name_short"] for e in b["element_types"]}
    rows = []
    for e in b["elements"]:
        rows.append({"name": e["web_name"], "team": norm_team(teams[e["team"]]),
                     "pos": POS_MAP[etype[e["element_type"]]],
                     "price": e["now_cost"] / 10.0,
                     "own": float(e.get("selected_by_percent", 0.0))})
    return pd.DataFrame(rows)


def load_roster_csv(path):
    """Accept a simple CSV; flexible column names."""
    df = pd.read_csv(path)
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("name", "web_name", "player"): ren[c] = "name"
        elif "team" in cl: ren[c] = "team"
        elif cl in ("pos", "position", "element_type"): ren[c] = "pos"
        elif "cost" in cl or "price" in cl: ren[c] = "price"
        elif "select" in cl or "own" in cl: ren[c] = "own"
    df = df.rename(columns=ren)
    df["team"] = df.team.map(norm_team)
    df["pos"] = df.pos.map(lambda x: POS_MAP.get(x, x))
    if df.price.max() > 30: df["price"] = df.price / 10.0     # tenths -> millions
    if "own" not in df: df["own"] = 0.0
    return df[["name", "team", "pos", "price", "own"]]


# ---------------------------------------------------------------------------
# Cold-start prior calibration (price + position -> rates, start prob)
# ---------------------------------------------------------------------------
def calibrate_cold_start(hist_csv="/mnt/user-data/uploads/fpl-data-stats.csv",
                         min_minutes=450):
    d = pd.read_csv(hist_csv)
    d["pos"] = d.element_type.map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    ag = d.groupby(["id", "pos"]).agg(
        minutes=("minutes", "sum"),
        npxgi=("non_penalty_expected_goal_involvements", "sum"),
        xa=("expected_assists", "sum"),
        defcon=("defensive_contribution", "sum"),
        starts=("minutes", lambda s: (s >= 60).sum()),
        apps=("minutes", lambda s: (s > 0).sum()),
        games=("minutes", "size"), price=("now_cost", "last")).reset_index()
    ag = ag[ag.minutes >= min_minutes]
    nnf = ag.minutes / 90.0
    ag["inv90"] = ag.npxgi / nnf; ag["xa90"] = ag.xa / nnf; ag["dc90"] = ag.defcon / nnf
    ag["startrate"] = ag.starts / ag.games
    ag["subrate"] = (ag.apps - ag.starts).clip(lower=0) / ag.games
    cal = {}
    for p in ["GK", "DEF", "MID", "FWD"]:
        s = ag[ag.pos == p]; x = s.price.values
        b_inv = np.polyfit(x, np.log(s.inv90 + 0.02), 1)
        b_st = np.polyfit(x, s.startrate, 1)
        cal[p] = dict(inv_b=b_inv[0], inv_a=b_inv[1],
                      st_b=b_st[0], st_a=b_st[1],
                      xa_share=float((s.xa.sum() / max(s.npxgi.sum(), 1e-6))),
                      dc90=float(s.dc90.mean()), subrate=float(s.subrate.mean()))
    return cal


def _coldstart_row(name, team, pos, price, own, cal, k0=1.5, kstart=4.0):
    c = cal[pos]
    inv90 = float(np.exp(c["inv_a"] + c["inv_b"] * price) - 0.02)
    inv90 = max(inv90, 1e-3)
    xa90 = inv90 * c["xa_share"]
    dc90 = c["dc90"]
    start = float(np.clip(c["st_a"] + c["st_b"] * price, 0.05, 0.97))
    return {"id": -1, "web_name": name, "pos": pos, "team": team,
            "own": own, "cost": price,
            # weak (wide) Gamma priors centred on the price-implied means
            "npxgi_alpha": inv90 * k0, "npxgi_beta": k0,
            "xa_alpha": xa90 * k0, "xa_beta": k0,
            "defcon_alpha": dc90 * k0, "defcon_beta": k0,
            "start_a": start * kstart, "start_b": (1 - start) * kstart,
            "sub_app_rate": c["subrate"], "cold_start": True}


# ---------------------------------------------------------------------------
# Build the unified 2026/27 player frame
# ---------------------------------------------------------------------------
def build_players_2627(roster, cal=None, revert=0.70):
    if cal is None:
        cal = calibrate_cold_start()
    hist = player_posteriors(revert=revert)                 # 25/26 posteriors
    hist["key"] = hist.web_name.str.lower().str.strip()
    roster = roster.copy(); roster["key"] = roster.name.str.lower().str.strip()

    matched, used = [], set()
    hmap = {(r.key, r.pos): r for _, r in hist.iterrows()}
    hmap_name = {}
    for _, r in hist.iterrows():
        hmap_name.setdefault(r.key, r)
    for _, r in roster.iterrows():
        h = hmap.get((r.key, r.pos))
        if h is None:
            h = hmap_name.get(r.key)
        if h is not None and h["id"] not in used:
            used.add(h["id"])
            row = h.to_dict()
            row.update({"team": r.team, "cost": r.price, "own": r.own,  # RE-POINT to 26/27 team (cost in millions)
                        "cold_start": False})
            matched.append(row)
        else:
            matched.append(_coldstart_row(r["name"], r.team, r.pos, r.price, r.own, cal))
    players = pd.DataFrame(matched)
    return players, cal


def apply_transfers(roster, moves: dict):
    """One-line transfer updates: {player_name: new_team}. Names normalised loosely."""
    roster = roster.copy()
    low = {k.lower(): v for k, v in moves.items()}
    m = roster.name.str.lower().map(low)
    roster.loc[m.notna(), "team"] = m[m.notna()].map(norm_team).values
    return roster


# ---------------------------------------------------------------------------
# Demo: construct a 26/27 roster (returning squads carried; promoted squads as
# illustrative cold-start entries) and project. Replace `demo_roster` with
# load_fpl_bootstrap('bootstrap-static.json') for the real squads.
# ---------------------------------------------------------------------------
def demo_roster():
    """Returning-team players from 25/26 + illustrative promoted-team squads.
    The promoted names are placeholders (Championship data not available here);
    swap in the real FPL roster to get named promoted-team players."""
    d = pd.read_csv("/mnt/user-data/uploads/fpl-data-stats.csv")
    d["pos"] = d.element_type.map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    d["team"] = d.team_name.replace({"Man Utd": "Man United", "Spurs": "Tottenham"})
    returning = sorted(set(pd.read_csv("/mnt/user-data/uploads/E0.csv")
                           .HomeTeam.replace({"Man Utd": "Man United", "Spurs": "Tottenham"}).unique()))
    keep_teams = [t for t in returning if t not in ("Burnley", "West Ham", "Wolves")]
    ag = d.groupby(["web_name", "pos", "team"]).agg(
        minutes=("minutes", "sum"), price=("now_cost", "last"),
        own=("selected_by_percent", "last")).reset_index()
    ag = ag[(ag.minutes >= 450) & (ag.team.isin(keep_teams))]
    ret = ag.rename(columns={"web_name": "name"})[["name", "team", "pos", "price", "own"]]
    # price already in millions in this file
    # illustrative promoted squads (1 GK, 5 DEF, 5 MID, 3 FWD) with plausible prices
    promo = []
    tiers = {"GK": [4.5], "DEF": [4.5, 4.5, 4.0, 5.0, 4.0],
             "MID": [5.5, 5.0, 6.0, 4.5, 5.0], "FWD": [6.0, 5.5, 5.0]}
    for team in ["Coventry", "Hull", "Ipswich"]:
        for pos, prices in tiers.items():
            for i, pr in enumerate(prices, 1):
                promo.append({"name": f"{team}_{pos}{i}", "team": team,
                              "pos": pos, "price": pr, "own": 1.0})
    return pd.concat([ret, pd.DataFrame(promo)], ignore_index=True)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from bayes_model import TeamModel, project

    print("Calibrating cold-start priors from 25/26 (price + position -> rates)...")
    cal = calibrate_cold_start()
    for p in ["DEF", "MID", "FWD"]:
        c = cal[p]
        print(f"  {p}: inv90 at £5m={np.exp(c['inv_a']+c['inv_b']*5)-0.02:.3f}, "
              f"£8m={np.exp(c['inv_a']+c['inv_b']*8)-0.02:.3f}; "
              f"start@£5m={np.clip(c['st_a']+c['st_b']*5,0,1):.2f}")

    roster = demo_roster()
    # example transfer re-pointing (mechanism demo):
    roster = apply_transfers(roster, {})   # e.g. {"Isak": "Liverpool"}
    print(f"\nRoster: {len(roster)} players across {roster.team.nunique()} teams "
          f"({(roster.team.isin(['Coventry','Hull','Ipswich'])).sum()} promoted-squad slots)")

    players, cal = build_players_2627(roster, cal)
    n_cold = int(players.cold_start.sum())
    print(f"Built priors: {len(players)} players, {n_cold} cold-start "
          f"(no 25/26 PL history), {len(players)-n_cold} carried forward.")

    tm = TeamModel().fit(); tsamp = tm.sample_2627(S=1200)
    res = project(players, tm, tsamp, 1, 38, S=1200)
    res = res.merge(players[["web_name", "pos", "team", "cold_start"]].drop_duplicates(["web_name", "pos", "team"]),
                    left_on=["player", "pos", "team"], right_on=["web_name", "pos", "team"], how="left")
    res.round(2).to_csv("/mnt/user-data/outputs/projection_2627_with_roster.csv", index=False)

    print("\n=== Promoted-team players now projected (top 8 cold-start) ===")
    cold = res[res.cold_start == True].head(8)
    print(cold[["player", "pos", "team", "mean", "p5", "p95"]].round(1).to_string(index=False))
    print("\n=== Overall top 12 (full season, with roster) ===")
    print(res.head(12)[["player", "pos", "team", "mean", "p5", "p95", "own"]].round(1).to_string(index=False))


def load_pull_csv(path):
    """Read the compact CSV produced by pull_fpl.py and return (roster, signals)
    — everything the model needs from one small upload."""
    df = pd.read_csv(path)
    df["team"] = df["team"].map(norm_team)
    df["pos"] = df["pos"].map(lambda x: POS_MAP.get(x, x))
    roster = pd.DataFrame({"name": df.web_name, "team": df.team, "pos": df.pos,
                           "price": pd.to_numeric(df.price, errors="coerce"),
                           "own": pd.to_numeric(df.get("own", 0), errors="coerce").fillna(0)})
    chance = pd.to_numeric(df.get("chance_next"), errors="coerce")
    signals = pd.DataFrame({
        "name": df.web_name,
        "status": df.get("status", "a").fillna("a"),
        # FPL leaves chance_of_playing null when there's no doubt -> treat as 100%
        "chance_play": (chance.fillna(100) / 100.0).clip(0, 1),
        "news": df.get("news", "").fillna(""),
        "ep_next": pd.to_numeric(df.get("ep_next"), errors="coerce").fillna(0),
        "pen_order": pd.to_numeric(df.get("pen_order"), errors="coerce"),
        "fk_order": pd.to_numeric(df.get("fk_order"), errors="coerce"),
        "corner_order": pd.to_numeric(df.get("corner_order"), errors="coerce"),
    })
    return roster, signals
</content>
