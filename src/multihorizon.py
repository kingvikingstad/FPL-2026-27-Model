"""
multihorizon.py
===============
Project CUMULATIVE FPL points over the next H gameweeks (H = 3, 6, 12) by
applying each player's forward FIXTURE SCHEDULE. Everything is leak-free and
handles double/blank gameweeks (window length in *fixtures* != H weeks).

Signal design for multi-week horizons:
  * form            EWMA of per-90 rates & points, short half-life (~2 GW)
  * market          lagged ownership + price   (strongest predictors, no leak)
  * availability     E[appearances] = recent start-rate * #scheduled fixtures
  * schedule         sum over the window's fixtures of odds-derived difficulty
                     - attackers: expected goals FOR (rolling team att vs opp def)
                     - def/GK:    sum of clean-sheet probs exp(-opp expected goals)
Team ratings are rolling (expanding mean up to t-1) so nothing from the future
leaks; difficulty is applied to the KNOWN schedule, which is what makes 3/6/12
week look-aheads work.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import sys; sys.path.insert(0, "/home/claude/fpl")
import betting_features as bf

NAME = {"Man Utd": "Man United", "Spurs": "Tottenham"}       # player -> betting
LEAGUE_AVG = 1.45

# ---------------------------------------------------------------------------
# 1. Load player panel + betting-derived per-fixture team expected goals
# ---------------------------------------------------------------------------
p = pd.read_csv("/mnt/user-data/uploads/fpl-data-stats.csv")
p["pos"] = p.element_type.map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
p["team_b"] = p.team_name.replace(NAME)
p["opp_b"]  = p.opponent_team_name.replace(NAME)
p = p.sort_values(["id", "gameweek"]).reset_index(drop=True)

bet = bf.build("/mnt/user-data/uploads/E0.csv")

# ---------------------------------------------------------------------------
# 2. Map betting matches -> gameweeks via the player schedule, build team-GW xG
# ---------------------------------------------------------------------------
# unique home-fixtures from player file give (team, opp, gameweek)
homefix = (p[p.was_home == 1][["team_b", "opp_b", "gameweek"]]
           .drop_duplicates().rename(columns={"team_b": "HomeTeam", "opp_b": "AwayTeam"}))
bm = bet.merge(homefix, on=["HomeTeam", "AwayTeam"], how="left")
# rows: each team's expected goals for/against that gameweek
tg = pd.concat([
    bm.rename(columns={"HomeTeam": "team", "lambda_home": "xg_for", "lambda_away": "xg_against"})[["team", "gameweek", "xg_for", "xg_against"]],
    bm.rename(columns={"AwayTeam": "team", "lambda_away": "xg_for", "lambda_home": "xg_against"})[["team", "gameweek", "xg_for", "xg_against"]],
], ignore_index=True).dropna(subset=["gameweek"])
tg["gameweek"] = tg.gameweek.astype(int)
tg = tg.sort_values(["team", "gameweek"])

# rolling (leak-free) team attack / defence ratings = mean of prior xg_for / xg_against
g = tg.groupby("team")
tg["att_rating"] = g["xg_for"].apply(lambda s: s.shift(1).expanding(1).mean()).values
tg["def_rating"] = g["xg_against"].apply(lambda s: s.shift(1).expanding(1).mean()).values
tg[["att_rating", "def_rating"]] = tg[["att_rating", "def_rating"]].fillna(LEAGUE_AVG)
rating = tg.set_index(["team", "gameweek"])[["att_rating", "def_rating"]]

# ---------------------------------------------------------------------------
# 3. Per team-GW fixture difficulty (attack difficulty + clean-sheet prob),
#    using rolling ratings; then prefix-sum over gameweeks for fast windows.
# ---------------------------------------------------------------------------
fix = p[["team_b", "gameweek", "opp_b", "was_home"]].drop_duplicates().rename(
    columns={"team_b": "team", "opp_b": "opp"})
fix = fix.merge(rating.rename(columns={"att_rating": "own_att", "def_rating": "own_def"}),
                left_on=["team", "gameweek"], right_index=True, how="left")
fix = fix.merge(rating.rename(columns={"att_rating": "opp_att", "def_rating": "opp_def"}),
                left_on=["opp", "gameweek"], right_index=True, how="left")
fix[["own_att", "own_def", "opp_att", "opp_def"]] = fix[["own_att", "own_def", "opp_att", "opp_def"]].fillna(LEAGUE_AVG)
home_adj = np.where(fix.was_home == 1, 0.9, 1.1)
fix["xg_for"]     = np.sqrt(fix.own_att * fix.opp_def) * np.where(fix.was_home == 1, 1.1, 0.9)
fix["xg_against"] = np.sqrt(fix.opp_att * fix.own_def) * home_adj
fix["cs_prob"]    = np.exp(-fix.xg_against)
fix["att_diff"]   = fix.xg_for / LEAGUE_AVG
fix["nfix"]       = 1

# full team x gameweek grid (missing = blank GW -> zeros), prefix sums
teams = sorted(fix.team.unique()); gws = list(range(1, 39))
grid = (fix.groupby(["team", "gameweek"])[["att_diff", "cs_prob", "nfix", "xg_for", "cs_prob"]]
        .sum().rename(columns={"cs_prob": "cs_sum"}))
grid = fix.groupby(["team", "gameweek"]).agg(
    att_diff=("att_diff", "sum"), cs_prob=("cs_prob", "sum"),
    nfix=("nfix", "sum"), xg_for=("xg_for", "sum")).reset_index()
full = (pd.MultiIndex.from_product([teams, gws], names=["team", "gameweek"]).to_frame(index=False))
grid = full.merge(grid, on=["team", "gameweek"], how="left").fillna(0.0).sort_values(["team", "gameweek"])
for c in ["att_diff", "cs_prob", "nfix", "xg_for"]:
    grid[f"cum_{c}"] = grid.groupby("team")[c].cumsum()
cum = grid.set_index(["team", "gameweek"])

def window_sum(team, t, H, col):
    """sum of `col` over gameweeks [t, t+H-1] via prefix sums (0 for blanks)."""
    hi = min(t + H - 1, 38)
    a = cum.loc[(team, hi), f"cum_{col}"] if (team, hi) in cum.index else np.nan
    b = cum.loc[(team, t - 1), f"cum_{col}"] if t > 1 and (team, t - 1) in cum.index else 0.0
    return a - b

# ---------------------------------------------------------------------------
# 4. Player form (EWMA, half-life 2) + market, all leak-free (shift 1)
# ---------------------------------------------------------------------------
HL = 2.0; alpha = 1 - 0.5 ** (1 / HL)
gp = p.groupby("id")
def ewm_prior(col):
    return gp[col].apply(lambda s: s.shift(1).ewm(alpha=alpha, min_periods=2).mean()).values
for c in ["total_points", "minutes", "non_penalty_expected_goal_involvements",
          "expected_assists", "expected_goals", "defensive_contribution",
          "expected_goals_conceded"]:
    p[f"f_{c}"] = ewm_prior(c)
p["f_start"] = gp["minutes"].apply(lambda s: (s.shift(1) >= 60).ewm(alpha=alpha, min_periods=2).mean()).values
p["l1_own"]  = gp["selected_by_percent"].shift(1)
p["l1_cost"] = gp["now_cost"].shift(1)
p["prior_apps"] = gp["minutes"].apply(lambda s: (s.shift(1) > 0).expanding().sum()).values

# player cumulative points prefix (for horizon targets)
pp_prefix = p.pivot_table(index="id", columns="gameweek", values="total_points", aggfunc="sum")
pp_cum = pp_prefix = pp_prefix if False else pp_prefix  # keep name
pp = p.pivot_table(index="id", columns="gameweek", values="total_points", aggfunc="sum").reindex(columns=gws)
pp_cum = pp.cumsum(axis=1)

def target_sum(pid, t, H):
    hi = min(t + H - 1, 38)
    a = pp_cum.loc[pid, hi]
    b = pp_cum.loc[pid, t - 1] if t > 1 else 0.0
    return a - b

# ---------------------------------------------------------------------------
# 5. Assemble modelling rows per horizon and evaluate (temporal split + gap)
# ---------------------------------------------------------------------------
FORM = ["f_total_points", "f_minutes", "f_non_penalty_expected_goal_involvements",
        "f_expected_assists", "f_expected_goals", "f_defensive_contribution",
        "f_expected_goals_conceded", "f_start"]
MARKET = ["l1_own", "l1_cost"]

def build_rows(H):
    base = p[(p.prior_apps >= 3) & (p.gameweek <= 38 - H + 1)].copy()
    # schedule features over the window
    base["sched_nfix"]  = [window_sum(tm, t, H, "nfix") for tm, t in zip(base.team_b, base.gameweek)]
    base["sched_att"]   = [window_sum(tm, t, H, "att_diff") for tm, t in zip(base.team_b, base.gameweek)]
    base["sched_cs"]    = [window_sum(tm, t, H, "cs_prob") for tm, t in zip(base.team_b, base.gameweek)]
    base["exp_apps"]    = base["f_start"] * base["sched_nfix"]
    base["y"]           = [target_sum(i, t, H) for i, t in zip(base.id, base.gameweek)]
    return base.dropna(subset=FORM + MARKET + ["y"])

VARIANTS = {
    "naive (form x #fixtures)": ["f_total_points", "sched_nfix"],
    "+ schedule difficulty":    ["f_total_points", "sched_nfix", "sched_att", "sched_cs"],
    "+ availability":           ["f_total_points", "sched_nfix", "sched_att", "sched_cs", "exp_apps", "f_minutes", "f_start"],
    "full (+ market + form)":   FORM + MARKET + ["sched_nfix", "sched_att", "sched_cs", "exp_apps"],
}

def evaluate(H):
    d = build_rows(H)
    origins = sorted(d.gameweek.unique())
    # origin-ordered split: train on earliest 60% of origins, test on the rest.
    # Features are computed AS OF the origin (leak-free); for long H the target
    # windows of the last train origins overlap early test windows -- unavoidable
    # in a single season, flagged in the writeup. Use a gap where data allows.
    boundary = origins[int(len(origins) * 0.6)]
    gap = H if origins[-1] >= boundary + H else 0     # clean gap if feasible
    train = d[d.gameweek <= boundary]
    test  = d[d.gameweek >= boundary + gap]
    if len(test) < 30:                                 # fall back: no gap
        test = d[d.gameweek > boundary]
    out = []
    for name, feats in VARIANTS.items():
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=4,
                                          learning_rate=0.05, l2_regularization=1.0)
        m.fit(train[feats].values, train["y"].values)
        pr = m.predict(test[feats].values)
        sp = stats.spearmanr(test["y"], pr).correlation
        mae = np.mean(np.abs(test["y"] - pr))
        out.append({"H": H, "variant": name, "Spearman": sp, "MAE": mae,
                    "n_test": len(test), "gap": gap})
    return pd.DataFrame(out), d, train, test

if __name__ == "__main__":
    print("Double GWs:", int((grid.nfix == 2).sum()), " Blank GWs:", int((grid.nfix == 0).sum()))
    all_res = []
    for H in [1, 3, 6, 12]:
        r, *_ = evaluate(H)
        all_res.append(r)
        print(f"\n===== HORIZON H = {H} gameweeks  (test n={r.n_test.iloc[0]}) =====")
        print(r[["variant", "Spearman", "MAE"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    res = pd.concat(all_res, ignore_index=True)
    res.to_csv("/mnt/user-data/outputs/multihorizon_results.csv", index=False)

    # feature importance shift across horizons (full model)
    print("\n" + "="*66); print("Feature importance shift: single-GW vs 12-GW (full model)"); print("="*66)
    for H in [3, 12]:
        _, d, train, test = evaluate(H)
        feats = VARIANTS["full (+ market + form)"]
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05,
                                          l2_regularization=1.0).fit(train[feats].values, train["y"].values)
        pi = permutation_importance(m, test[feats].values, test["y"].values, n_repeats=5, random_state=0)
        imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
        print(f"\nH={H}: top 6 features")
        print(imp.head(6).to_string(float_format=lambda x: f"{x:.4f}"))

    # ---- concrete deliverable: ranked 6-GW projection from a sample origin ----
    H, ORIGIN = 6, 20
    d = build_rows(H)
    tr = d[d.gameweek < ORIGIN]; te = d[d.gameweek == ORIGIN]
    feats = VARIANTS["full (+ market + form)"]
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05,
                                      l2_regularization=1.0).fit(tr[feats].values, tr["y"].values)
    te = te.assign(proj=m.predict(te[feats].values))
    rank = (te.sort_values("proj", ascending=False)
              [["web_name","team_name","pos","sched_nfix","sched_att","proj","y"]]
              .rename(columns={"y":"actual_next6"}).head(25).reset_index(drop=True))
    rank.to_csv("/mnt/user-data/outputs/sample_projection_gw20_next6.csv", index=False)
    print("\n" + "="*66); print(f"Sample: top-25 projected NEXT-{H}-GW totals from GW{ORIGIN}"); print("="*66)
    print(rank.round(2).to_string())
</content>
