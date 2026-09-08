import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
export_workbook.py — every player, every source, one spreadsheet.
==================================================================
Assembles the model board, the two external projection feeds, the live FPL
availability endpoint, and the competitive/pre-season history in the data repo into a
single .xlsx with filters and frozen headers, so the whole thing can be sorted and
pivoted without touching Python.

Sheets
  Players       one row per player: model GW1-10, FFS GW1-6, Solio GW1, live injury
                status, every installed prior, 25/26 competitive actuals, 26/27
                pre-season minutes, and the team context behind the projection
  By Gameweek   player x gameweek long form — model and FFS side by side with the
                fixture, so a disagreement can be traced to a specific match
  Teams         the team layer, odds provenance through to fitted attack/defence
  Team Fixtures team x gameweek: lambdas, clean sheet, win/draw/loss
  Sources       where the three projection sources agree and where they do not
  Notes         column provenance, and what each source may and may not be used for

SOURCES ARE KEPT SEPARATE, NOT AVERAGED
---------------------------------------
Model, FFS and Solio each get their own columns. No blended "consensus" column is
written beyond the GW1 Solio blend the board itself already applies, because averaging
projections is a modelling decision that needs a decision rule and a validation, and
neither exists for FFS yet. The comparison sheet is there to support that decision, not
to pre-empt it.

Run:  python scripts/export_workbook.py
Out:  outputs/fpl_2627_analysis.xlsx  (+ a CSV of the Players sheet)
"""
import warnings; warnings.filterwarnings("ignore")
import json
import urllib.request
import numpy as np
import pandas as pd

import core_insights as ci
import external_projections as ext
import player_names as pn
import player_value as pv
import repo_events as rev

GW_HI = int(_os.environ.get("GW_HI", "10"))
OUT_XLSX = _os.path.join(config.OUTPUTS, "fpl_2627_analysis.xlsx")
OUT_CSV = _os.path.join(config.OUTPUTS, "fpl_2627_players.csv")

FPL_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def live_availability():
    """Current status / chance of playing / news, straight from FPL.

    The board is built on the data-repo snapshot, which is refreshed on a cron and can
    be a day or two behind team news. This is the live read, joined on `player_code`
    (never `id` — FPL reassigns those between seasons). Falls back to the snapshot's
    own columns if the endpoint is unreachable, and says which it used.
    """
    try:
        req = urllib.request.Request(FPL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        rows = [{"player_code": e.get("code"),
                 "live_status": e.get("status"),
                 "live_chance_next": e.get("chance_of_playing_next_round"),
                 "live_chance_this": e.get("chance_of_playing_this_round"),
                 "live_news": (e.get("news") or "").strip(),
                 "live_price": (e.get("now_cost") or 0) / 10.0,
                 "live_own": pd.to_numeric(e.get("selected_by_percent"), errors="coerce"),
                 "live_ep_next": pd.to_numeric(e.get("ep_next"), errors="coerce")}
                for e in data["elements"]]
        d = pd.DataFrame(rows)
        print(f"[live] FPL bootstrap: {len(d)} players, "
              f"{(d.live_status != 'a').sum()} flagged not fully available")
        return d, "live FPL API"
    except Exception as e:
        print(f"[live] endpoint unavailable ({type(e).__name__}); using repo snapshot")
        return pd.DataFrame(columns=["player_code"]), "repo snapshot (live fetch failed)"


def build_players():
    board_w = pd.read_csv(_os.path.join(config.OUTPUTS, "gw_board_wide.csv"))
    board_l = pd.read_csv(_os.path.join(config.OUTPUTS, "gw_board_long.csv"))
    detail = pd.read_csv(_os.path.join(config.OUTPUTS,
                                       f"projection_detail_gw1_{GW_HI}.csv"))

    # ---- model, one row per player ----
    P = board_w.rename(columns={f"gw{i}": f"model_gw{i}" for i in range(1, GW_HI + 1)})
    P = P.rename(columns={"total": "model_total"})

    # per-player identity + priors + team context, from the detail export (already
    # one row per player-gameweek; the player-level columns are constant within player)
    idcols = ["player", "team", "pos", "player_code", "cold_start", "own", "cost"]
    prior_cols = ["p_start_prior", "exp_minutes", "exp_minutes_src", "minutes_prior",
                  "npxgi_rate90", "xa_rate90", "defcon_rate90", "pen_xg90",
                  "pen_order", "fk_order", "corner_order"]
    team_cols = ["own_att_strength", "own_def_strength", "own_att_rank", "own_def_rank",
                 "own_market_z", "own_blended_elo", "title_odds_dec", "releg_odds_dec",
                 "xga_env", "ppda_2627", "press_factor", "promoted"]
    keep = [c for c in idcols + prior_cols + team_cols if c in detail.columns]
    D1 = detail[keep].drop_duplicates(["player", "team"])
    P = P.merge(D1, on=["player", "team", "pos", "own", "cost"], how="left")

    # channel split over the horizon
    ch = board_l.groupby(["player", "team"]).agg(
        model_cs_pts=("cs_ev", "sum"), model_defcon_pts=("defcon_ev", "sum"),
        model_sd_mean=("sd", "mean"), model_p95_mean=("p95", "mean")).reset_index()
    P = P.merge(ch, on=["player", "team"], how="left")
    P["model_attack_pts"] = P["model_total"] - P["model_cs_pts"] - P["model_defcon_pts"]

    # Solio GW1 (already on the board where matched)
    sol = board_l[board_l.gw == 1][["player", "team", "solio"]].rename(
        columns={"solio": "solio_gw1"})
    P = P.merge(sol, on=["player", "team"], how="left")

    # ---- FFS GW1-6 ----
    P = ext.align_to_board(P, verbose=True)
    P["model_gw1_6"] = P[[f"model_gw{i}" for i in range(1, 7)]].sum(axis=1)
    P["ffs_minus_model_1_6"] = P["ffs_total"] - P["model_gw1_6"]

    # ---- live availability ----
    live, live_src = live_availability()
    if not live.empty:
        P = P.merge(live, on="player_code", how="left")

    # ---- history from the data repo ----
    P = P.merge(rev.pl_player_season("2025-2026"), on="player_code", how="left")
    P = P.merge(rev.start_rate_from_lineups("2025-2026"), on="player_code", how="left")
    P = P.merge(rev.preseason_minutes("2026-2027"), on="player_code", how="left")

    P["value_per_m"] = P["model_total"] / P["cost"].replace(0, np.nan)
    P["ffs_value_per_m"] = P["ffs_total"] / P["cost"].replace(0, np.nan)

    # ---- disambiguated names ----
    # 15 surnames in this squad belong to 32 different players (three Wilsons, three
    # Phillipses, two Palmers at different clubs). `player` is FPL's web_name and is
    # therefore NOT unique; sorting or looking up on it silently merges two people.
    d26, t26, _ = ci.load(base=config.repo("2026-2027"))
    nm = pn.canonical(d26, t26)[["player_code", "display_name", "unique_label",
                                 "full_name", "team_code", "name_ambiguous"]]
    P = P.merge(nm, on="player_code", how="left")
    audit = pn.check(P.dropna(subset=["display_name"]))
    assert audit["unique_label_is_unique"], f"labels are not unique: {audit}"
    print(f"[names] {audit['ambiguous_players']} players share a surname with someone "
          f"else; display_name and unique_label are unique across "
          f"{audit['players']} rows")

    order = (["display_name", "unique_label", "team", "pos", "cost", "own",
              "player_code", "full_name", "team_code", "name_ambiguous",
              "player", "cold_start"]
             + ["model_total", "model_gw1_6", "value_per_m"]
             + [f"model_gw{i}" for i in range(1, GW_HI + 1)]
             + ["ffs_total", "ffs_value_per_m"] + [f"ffs_gw{i}" for i in range(1, 7)]
             + ["ffs_minus_model_1_6", "solio_gw1", "ffs_matched"]
             + ["model_attack_pts", "model_cs_pts", "model_defcon_pts",
                "model_sd_mean", "model_p95_mean"]
             + ["live_status", "live_chance_next", "live_news", "live_price",
                "live_own", "live_ep_next"]
             + prior_cols
             + ["pl_apps", "pl_minutes", "pl_xg", "pl_xa", "pl_xg90", "pl_xa90",
                "pl_shots", "pl_goals", "pl_assists", "xi_starts", "xi_sheets", "xi_rate"]
             + ["pre_minutes", "pre_apps", "pre_mean_minutes", "pre_xg", "pre_xa"]
             + team_cols)
    cols = [c for c in order if c in P.columns] + \
           [c for c in P.columns if c not in order]
    return P[cols].sort_values("model_total", ascending=False), live_src


def _add_names(df, players):
    """Carry display_name / unique_label onto a (player, team)-keyed frame."""
    key = players[["player", "team", "player_code", "display_name", "unique_label",
                   "team_code"]]
    out = df.merge(key, on=["player", "team"], how="left")
    assert len(out) == len(df), "name join fanned out"
    front = ["player_code", "display_name", "unique_label", "team", "team_code", "pos"]
    return out[[c for c in front if c in out.columns] +
               [c for c in out.columns if c not in front]]


def build_by_gameweek(players):
    detail = pd.read_csv(_os.path.join(config.OUTPUTS,
                                       f"projection_detail_gw1_{GW_HI}.csv"))
    keep = ["player", "team", "pos", "cost", "own", "gw", "mean", "blended", "solio",
            "sd", "p5", "median", "p95", "cs_ev", "defcon_ev", "opponent", "is_home",
            "lam_for", "lam_against", "clean_sheet_prob", "p_win", "attack_ease",
            "opp_att_rank", "opp_def_rank", "home_discount_applied"]
    G = detail[[c for c in keep if c in detail.columns]].copy()
    G = G.rename(columns={"mean": "model_pts", "blended": "model_pts_blended"})

    ffs_long = players[["player", "team"] + [f"ffs_gw{i}" for i in range(1, 7)]].melt(
        id_vars=["player", "team"], var_name="gw", value_name="ffs_pts")
    ffs_long["gw"] = ffs_long["gw"].str.replace("ffs_gw", "").astype(int)
    G = G.merge(ffs_long, on=["player", "team", "gw"], how="left")
    G["ffs_minus_model"] = G["ffs_pts"] - G["model_pts"]
    G = _add_names(G, players)

    # Value above replacement, recomputed WITHIN each gameweek — the floor moves when a
    # cheap enabler is flagged out or has a blank, and that movement is exactly what a
    # per-gameweek solver exploits.
    start = players[["player", "team", "p_start_prior"]] \
        if "p_start_prior" in players.columns else None
    Gv = G.merge(start, on=["player", "team"], how="left") if start is not None else G
    Gv = pv.per_gameweek(Gv, ep_col="model_pts", price_col="cost", pos_col="pos",
                         gw_col="gw",
                         start_col="p_start_prior" if start is not None else None)
    keep = ["floor_price", "replacement_ep", "marginal_cost", "par", "value",
            "ep_cap", "cap_par", "cap_value", "is_replacement", "below_replacement",
            "captain_candidate", "value_rank_pos"]
    G = Gv[[c for c in G.columns] + [c for c in keep if c in Gv.columns]]
    return G.sort_values(["gw", "model_pts"], ascending=[True, False])


def build_budget(bygw):
    """Per gameweek: how much of the £100m is committed at floor prices, and how much is
    actually being allocated. The committed share is not a decision — it is the price of
    fielding a legal squad — so it is the discretionary figure a solver is optimising."""
    rows = []
    for gw, g in bygw.groupby("gw"):
        fl = g.drop_duplicates("pos")[["pos", "floor_price", "replacement_ep"]]
        meta = pv.committed_spend(fl)
        row = {"gw": gw, "committed_spend": meta["committed"],
               "discretionary_budget": meta["discretionary"]}
        for pos in ["GK", "DEF", "MID", "FWD"]:
            sub = fl[fl.pos == pos]
            row[f"floor_{pos}"] = float(sub["floor_price"].iloc[0]) if len(sub) else np.nan
            row[f"replacement_ep_{pos}"] = float(sub["replacement_ep"].iloc[0]) if len(sub) else np.nan
            row[f"committed_{pos}"] = meta["per_position"].get(pos, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gw")


def build_sources(players, bygw):
    rows = []
    for gw in range(1, 7):
        d = bygw[bygw.gw == gw][["model_pts", "ffs_pts"]].dropna()
        if len(d) < 3:
            continue
        rows.append({"comparison": f"model vs FFS, GW{gw}", "n": len(d),
                     "pearson": d["model_pts"].corr(d["ffs_pts"]),
                     "spearman": d["model_pts"].corr(d["ffs_pts"], method="spearman"),
                     "mae": (d["model_pts"] - d["ffs_pts"]).abs().mean(),
                     "bias_model_minus_other": (d["model_pts"] - d["ffs_pts"]).mean()})
    d = players[["model_gw1_6", "ffs_total"]].dropna()
    if len(d) >= 3:
        rows.append({"comparison": "model vs FFS, GW1-6 total", "n": len(d),
                     "pearson": d["model_gw1_6"].corr(d["ffs_total"]),
                     "spearman": d["model_gw1_6"].corr(d["ffs_total"], method="spearman"),
                     "mae": (d["model_gw1_6"] - d["ffs_total"]).abs().mean(),
                     "bias_model_minus_other": (d["model_gw1_6"] - d["ffs_total"]).mean()})
    s = players[["model_gw1", "solio_gw1"]].dropna()
    if len(s) >= 3:
        rows.append({"comparison": "model vs Solio, GW1", "n": len(s),
                     "pearson": s["model_gw1"].corr(s["solio_gw1"]),
                     "spearman": s["model_gw1"].corr(s["solio_gw1"], method="spearman"),
                     "mae": (s["model_gw1"] - s["solio_gw1"]).abs().mean(),
                     "bias_model_minus_other": (s["model_gw1"] - s["solio_gw1"]).mean()})
    summary = pd.DataFrame(rows)

    dis = players[players.ffs_matched & players.ffs_total.notna()].copy()
    dis = dis[["display_name", "unique_label", "team", "pos", "cost", "own",
               "model_gw1_6", "ffs_total", "ffs_minus_model_1_6", "cold_start",
               "p_start_prior", "pre_minutes"]]
    dis = dis.reindex(dis["ffs_minus_model_1_6"].abs().sort_values(ascending=False).index)
    return summary, dis.head(60)


NOTES = [
    ("Sheet", "Grain", "What it is"),
    ("Players", "one row per player", "model GW1-10, FFS GW1-6, Solio GW1, live availability, priors, history"),
    ("By Gameweek", "player x gameweek", "model and FFS per week with the fixture behind them"),
    ("Teams", "one row per club", "betting odds -> market Elo -> blended Elo -> fitted attack/defence"),
    ("Team Fixtures", "club x gameweek", "lambdas with credible intervals, clean sheet, win/draw/loss"),
    ("Budget Floors", "per gameweek", "floor price and replacement EP per position; how much of the £100m is committed vs discretionary"),
    ("Sources", "comparison", "agreement between the three projection sources, and the largest disagreements"),
    ("", "", ""),
    ("Column", "Source", "Notes / permitted use"),
    ("display_name", "player_names.canonical", "READ THIS ONE. Unique. Plain surname where that is unambiguous, else 'C. Palmer (CHE)', else 'Brennan Johnson (EVE)'"),
    ("unique_label", "player_names.canonical", "Always carries the club code. Use as the VLOOKUP / pivot key - guaranteed unique and consistent in shape"),
    ("player", "FPL web_name", "NOT UNIQUE - 15 surnames cover 32 players (three Wilsons, three Phillipses, two Palmers). Kept only for continuity; do not key on it"),
    ("player_code", "FPL", "The real identifier. Stable across seasons and never reused. Every join in the model uses this"),
    ("name_ambiguous", "player_names.canonical", "TRUE if this surname belongs to more than one player in the league this season"),
    ("model_gw1..gw10", "this model", "true per-gameweek posterior means, project(gw,gw) run ten times"),
    ("model_gw1", "this model + Solio", "GW1 only is blended 50/50 with Solio where a player matched"),
    ("ffs_gw1..gw6", "Fantasy Football Scout", "member projection supplied by the user; NOT blended into the model"),
    ("solio_gw1", "Solio Analytics", "public feed, cached snapshot; single gameweek only"),
    ("live_status / live_news", "FPL API, fetched at run time", "a=available i=injured d=doubtful s=suspended u=unavailable"),
    ("pl_*", "FPL-Core-Insights 25/26", "COMPETITIVE Premier League only; cup and European rows excluded"),
    ("xi_rate", "FPL-Core-Insights lineups.csv", "share of league lineup sheets started, from real XIs not inferred from minutes"),
    ("pre_*", "FPL-Core-Insights 26/27 friendlies", "[JUDGMENT] pre-season only; uncontrolled opposition incl. non-league. NOT a fitted signal"),
    ("p_start_prior", "this model", "Beta start probability the simulation drew from for GW2+"),
    ("p_start_prior_predxi", "FFS predicted XI + this model", "start probability for the TEAM-NEWS gameweek only (GW1). Applied softly: 0.75 weight if named in the XI, 0.35 if omitted - a prediction is not a teamsheet"),
    ("npxgi_rate90 etc", "this model", "implied per-90 rates from the installed Gamma priors"),
    ("own_market_z, *_odds_dec", "betting markets, 7 Aug 2026", "de-vigged outright markets; see the run note on how little they move the board"),
    ("cold_start", "this model", "no usable minutes history; start prior comes from ownership and depth"),
    ("", "", ""),
    ("VALUE CONSTRUCT", "player_value.py", "on 'By Gameweek'; recomputed independently within each gameweek"),
    ("floor_price", "player_value", "cheapest price at which the position can be filled by someone who might play"),
    ("replacement_ep", "player_value", "BEST expected points obtainable at the floor price - the opportunity cost of the slot"),
    ("marginal_cost", "player_value", "price - floor_price. The only part of the price that is a decision"),
    ("par", "player_value", "points above replacement = ep - replacement_ep. Negative means worse than a free slot"),
    ("value", "player_value", "par / marginal_cost = points above replacement per marginal £m. THE ranking column"),
    ("ep_cap", "player_value", "2 x ep. CONDITIONAL on wearing the armband - never sum this across a squad"),
    ("cap_par", "player_value", "2 x ep - replacement_ep. Which player most repays the armband (a level, not a ratio)"),
    ("cap_value", "player_value", "cap_par / marginal_cost, populated only for top-decile ep. 'If I spend up for a captain, who repays most'"),
    ("below_replacement", "player_value", "par < 0. A do-not-buy flag; the magnitude of a negative `value` is NOT a ranking"),
    ("is_replacement", "player_value", "this player IS the floor option, so marginal_cost is 0 and `value` is undefined by design"),
]


def main():
    players, live_src = build_players()
    bygw = build_by_gameweek(players)
    teams = pd.read_csv(_os.path.join(config.OUTPUTS, "team_projections_season.csv"))
    fixtures = pd.read_csv(_os.path.join(config.OUTPUTS,
                                         f"team_projections_gw1_{GW_HI}.csv"))
    summary, disagree = build_sources(players, bygw)
    budget = build_budget(bygw)

    players.to_csv(OUT_CSV, index=False)
    bygw.to_csv(_os.path.join(config.OUTPUTS, "fpl_2627_solver_inputs.csv"), index=False)

    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as xl:
        sheets = {"Players": players, "By Gameweek": bygw, "Teams": teams,
                  "Team Fixtures": fixtures, "Budget Floors": budget,
                  "Sources": summary, "Biggest Disagreements": disagree,
                  "Notes": pd.DataFrame(NOTES[1:], columns=list(NOTES[0]))}
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name, index=False)
            ws = xl.sheets[name]
            ws.freeze_panes(1, 0)
            if len(df.columns):
                ws.autofilter(0, 0, max(len(df), 1), len(df.columns) - 1)
            for i, c in enumerate(df.columns):
                width = max(len(str(c)) + 2, 11)
                if df[c].dtype == object and len(df):
                    longest = pd.to_numeric(
                        df[c].astype(str).str.len(), errors="coerce").max()
                    if pd.notna(longest):
                        width = min(max(width, int(longest) + 2), 42)
                ws.set_column(i, i, width)

    print(f"\n[workbook] {OUT_XLSX}")
    for name, df in [("Players", players), ("By Gameweek", bygw), ("Teams", teams),
                     ("Team Fixtures", fixtures), ("Budget Floors", budget),
                     ("Sources", summary), ("Biggest Disagreements", disagree)]:
        print(f"           {name:24s} {len(df):6d} rows x {len(df.columns)} cols")
    print(f"[workbook] availability source: {live_src}")
    print(f"[workbook] CSV of the Players sheet: {OUT_CSV}")

    print("\n=== source agreement ===")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
