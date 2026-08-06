#!/usr/bin/env python3
"""
run_2627.py — one entry point for 2026/27 FPL projections
=========================================================
Pulls the FREE data (FPL bootstrap-static, Understat shots, ClubElo), fits the
hierarchical Bayesian model, applies the single-GW signal layer, and emits ranked
posterior-predictive projections for ANY horizon.

    python run_2627.py --gw-from 1 --gw-to 6 --top 30
    python run_2627.py --gw-from 1 --gw-to 1 --demo-signals      # single GW
    python run_2627.py --gw-from 1 --gw-to 38 --understat 2026 --clubelo

Every external source is optional and resolved at run time: if it's reachable we
use it, otherwise we fall back to local files and say so. Nothing crashes if you
are offline — you just get a clearly-labelled, lower-information projection.

Paid/optional inputs (off by default): --lineups <json>, --props <csv>.
"""
from __future__ import annotations
import argparse, json, sys, urllib.request, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/fpl")

from bayes_model import TeamModel, project, player_posteriors
from roster import (load_fpl_bootstrap, build_players_2627, apply_transfers,
                    demo_roster, calibrate_cold_start)
import signals as sg

FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
LOCAL_E0 = "/mnt/user-data/uploads/E0.csv"
LOG = []
def log(msg, used=True):
    tag = "USED " if used else "SKIP "
    LOG.append(f"  [{tag}] {msg}"); print(f"  [{tag}] {msg}")


def current_season():
    """Return (football_data_code, understat_year) for the latest season whose
    results anchor the prior. Pre-August -> the season that just finished."""
    from datetime import date
    d = date.today()
    start = d.year if d.month >= 8 else d.year - 1
    yy = start % 100
    return f"{yy:02d}{yy+1:02d}", str(start)


def resolve_results(path=None, offline=False):
    """Team-model input: prefer a freshly downloaded near-complete season from
    football-data.co.uk, else the local E0.csv, else a downloaded prior season.
    Returns a local file path the team model can read."""
    if path:
        log(f"results from file: {path}"); return path
    fd_code, _ = current_season()
    if not offline:
        for code, label in [(fd_code, "current"),
                            (f"{(int(fd_code[:2])-1):02d}{int(fd_code[:2]):02d}", "prior")]:
            try:
                df = sg.load_football_data(code)
                if len(df.dropna(subset=["FTHG", "FTAG"])) >= 100:
                    tmp = f"/tmp/E0_{code}.csv"; df.to_csv(tmp, index=False)
                    log(f"results auto-downloaded: football-data {code} ({label}, {len(df)} matches)")
                    return tmp
            except Exception as e:
                log(f"football-data {code} unavailable ({type(e).__name__})", used=False)
    import os
    if os.path.exists(LOCAL_E0):
        log(f"results from local file: {LOCAL_E0}"); return LOCAL_E0
    raise FileNotFoundError("no results source available")


# ---------------------------------------------------------------------------
# Source resolvers (try live free source -> fall back to local, always logged)
# ---------------------------------------------------------------------------
def resolve_bootstrap(path=None, timeout=15, offline=False):
    """Return (roster_df, signals_df) from the FPL API or a saved JSON; if
    neither is reachable, fall back to the local demo roster (no live signals)."""
    raw = None
    if path:
        try:
            with open(path) as f: raw = json.load(f)
            log(f"FPL bootstrap from file: {path}")
        except Exception as e:
            log(f"bootstrap file unreadable ({e})", used=False)
    if raw is None and not offline:
        try:
            with urllib.request.urlopen(FPL_BOOTSTRAP, timeout=timeout) as r:
                raw = json.loads(r.read()); log("FPL bootstrap from live API (roster + availability + set-pieces)")
        except Exception as e:
            log(f"FPL API unreachable ({type(e).__name__}) -> demo roster, no live signals", used=False)
    elif raw is None and offline:
        log("FPL API skipped (offline) -> demo roster", used=False)
    if raw is None:
        return demo_roster(), None
    # build roster + signals from the same JSON
    import io
    tmp = "/tmp/_bootstrap.json"
    with open(tmp, "w") as f: json.dump(raw, f)
    roster = load_fpl_bootstrap(tmp)
    sig = sg.load_fpl_signals(tmp)
    return roster, sig


def resolve_understat(season, offline=False):
    if offline:
        log("Understat skipped (offline)", used=False); return None
    try:
        u = sg.load_understat_shots(str(season)); log(f"Understat shots season {season} ({len(u)} players)")
        return u
    except Exception as e:
        log(f"Understat unavailable ({type(e).__name__}) -> FPL npxGI prior only", used=False)
        return None


def resolve_clubelo(offline=False):
    if offline:
        log("ClubElo skipped (offline)", used=False); return None
    try:
        e = sg.load_clubelo("today"); log(f"ClubElo snapshot ({len(e)} clubs)")
        return e
    except Exception as ex:
        log(f"ClubElo unavailable ({type(ex).__name__}) -> betting-odds team prior only", used=False)
        return None


# ---------------------------------------------------------------------------
# Main projection routine
# ---------------------------------------------------------------------------
def run_projection(gw_from=1, gw_to=6, top=30, bootstrap_path=None,
                   understat_season=None, lineups_path=None, props_path=None,
                   transfers=None, S=1000, demo_signals=False, offline=False,
                   no_understat=False, no_clubelo=False, results_path=None,
                   apifootball_key=None, af_season=2026, af_budget=20,
                   af_lineup_fixtures=None, fm_export=None, history_seasons=None,
                   fpl_strength=False, players_csv=None,
                   out_path="/mnt/user-data/outputs/projection_2627_run.csv"):
    fd_code, us_year = current_season()
    understat_season = understat_season or us_year
    print("="*74); print(f"FPL 2026/27 projection  |  horizon GW{gw_from}-{gw_to}  |  S={S} draws"
                          f"{'  |  OFFLINE' if offline else '  |  auto-pulling free sources'}")
    print("="*74); print("Data sources:")

    # 1. results (auto-download latest) + ClubElo -> team model
    e0 = resolve_results(results_path, offline)
    elo = resolve_clubelo(offline or no_clubelo)
    if fpl_strength:
        try:
            import fpl_live_data as _L
            fs = _L.as_clubelo_frame()
            elo = fs if elo is None else elo
            log(f"FPL official team strength ratings applied ({len(fs)} teams, 2025/26)")
        except Exception as e:
            log(f"FPL strength unavailable ({type(e).__name__})", used=False)
    # 1b. optional: calibrate hyperparameters on 30 seasons of history instead of
    #     using the hard-coded guesses (reversion, season sd, promoted prior,
    #     home advantage). Cached to disk after the first run.
    hp = {}
    if history_seasons and not offline:
        try:
            import history as Hy, json as _json, os as _os
            cachep = "/tmp/history_hyperparams.json"
            if _os.path.exists(cachep):
                hp = _json.load(open(cachep))
                log(f"history hyperparameters from cache: {cachep}")
            else:
                res, *_ = Hy.calibrate_all(start=history_seasons, verbose=False)
                hp = Hy.to_model_kwargs(res)
                _json.dump(hp, open(cachep, "w"))
                log(f"history calibrated from {history_seasons}-: "
                    f"revert={hp['revert']:.3f}, season_sd={hp['season_sd']:.3f}, "
                    f"promoted_att={hp['promoted_att'][0]:+.3f}, home={hp['home_prior'][0]:+.3f}")
        except Exception as e:
            log(f"history calibration unavailable ({type(e).__name__}: {e})", used=False)
    tm = TeamModel(home_prior=tuple(hp["home_prior"]) if hp else (0.26, 0.08),
                   promoted_att=tuple(hp["promoted_att"]) if hp else (-0.20, 0.30),
                   promoted_def=tuple(hp["promoted_def"]) if hp else (-0.22, 0.30)
                   ).fit(e0, clubelo=elo)
    log(f"Team model fit ({len(tm.teams)} teams; ClubElo blended into {tm.clubelo_used}"
        f"{'; history-calibrated priors' if hp else '; default priors'})")
    tsamp = tm.sample_2627(S=S, revert=hp.get("revert", 0.85),
                           season_sd=hp.get("season_sd", 0.15))

    # 2. roster + availability/set-piece signals (FPL bootstrap)
    if players_csv:
        from roster import load_pull_csv
        roster, fsig = load_pull_csv(players_csv)
        log(f"roster+signals from pull_fpl CSV: {players_csv} ({len(roster)} players)")
    else:
        roster, fsig = resolve_bootstrap(None if offline else bootstrap_path,
                                         offline=offline)
    if transfers:
        roster = apply_transfers(roster, transfers); log(f"applied {len(transfers)} transfer overrides")
    players, _ = build_players_2627(roster)
    n_cold = int(players.cold_start.sum())
    log(f"players: {len(players)} ({len(players)-n_cold} carried fwd, {n_cold} cold-start)")

    # 3. offline illustrative signals only if asked
    if fsig is None and demo_signals:
        fsig = _demo_signal_frame(players); log("using ILLUSTRATIVE demo signals (offline)")

    # 3b. API-Football (FREE TIER, needs key): injuries -> availability, dated
    #     fixtures, and confirmed lineups. Quota-aware + disk-cached.
    af_lineups = None
    if apifootball_key and not offline:
        try:
            import apifootball as af
            c = af.Client(key=apifootball_key, budget=af_budget)
            inj = af.injuries(c, season=af_season)
            if fsig is not None and len(inj):
                fsig = af.merge_into_signals(fsig, inj)
            log(f"API-Football injuries: {len(inj)} players ({c.used} req used)")
            fx = af.fixtures(c, season=af_season)
            if len(fx):
                kd = af.kickoff_dates(fx)
                kd.to_csv("/mnt/user-data/outputs/kickoff_dates_2627.csv", index=False)
                log(f"API-Football fixtures: {len(fx)} with kick-off dates -> kickoff_dates_2627.csv")
            if af_lineup_fixtures:
                af_lineups = {}
                for fid in af_lineup_fixtures:
                    af_lineups.update(af.lineups(c, fid))
                log(f"API-Football confirmed lineups for {len(af_lineups)} teams")
        except Exception as e:
            log(f"API-Football unavailable ({type(e).__name__}: {e})", used=False)
    elif apifootball_key:
        log("API-Football skipped (offline)", used=False)

    # 4. signal layer
    lineups = json.load(open(lineups_path)) if lineups_path else None
    if af_lineups:
        lineups = {**(lineups or {}), **af_lineups}
    if lineups: log(f"confirmed lineups for {len(lineups)} teams")
    if fsig is not None:
        players = sg.apply_availability(players, fsig, lineups=lineups)
        players = sg.apply_setpieces(players, fsig)
        log("applied availability + set-piece priors")

    # 4b. Football Manager attribute priors — covers players with NO PL history
    #     (promoted squads, new signings). Calibrated on the FM x PL overlap.
    if fm_export:
        try:
            import fm_priors as fmp
            fm = fmp.load_fm_export(fm_export)
            mp = fmp.calibrate(fm, verbose=False)
            players["minutes"] = players.get("minutes", 0)
            players = fmp.apply_fm_priors(players, fm, mp, verbose=False)
            nm = int(players.fm_matched.sum()) if "fm_matched" in players else 0
            r2 = mp["targets"]["inv90"]["r2"]
            log(f"FM priors from {fm_export}: {len(fm)} players, "
                f"{mp['n_overlap']} overlap (inv90 R^2={r2:.2f}), {nm} matched")
        except Exception as e:
            log(f"FM export unusable ({type(e).__name__}: {e})", used=False)
    ushots = resolve_understat(understat_season, offline or no_understat)
    if ushots is not None:
        players = sg.apply_understat_prior(players, ushots); log("applied Understat shot-volume prior")
    if props_path:
        try:
            props = pd.read_csv(props_path); od = dict(zip(props.iloc[:,0], props.iloc[:,1]))
            players = sg.fuse_prop_odds(players, od); log(f"fused {len(od)} player-prop odds (PAID)")
        except Exception as e:
            log(f"props file unreadable ({e})", used=False)

    # 5. project + rank
    res = project(players, tm, tsamp, gw_from, gw_to, S=S).drop_duplicates("player")
    res = res.merge(players[["web_name","pos","team","cold_start"]].drop_duplicates(["web_name","pos","team"]),
                    left_on=["player","pos","team"], right_on=["web_name","pos","team"], how="left")
    res["cost_m"] = np.where(res["cost"] > 30, res["cost"] / 10.0, res["cost"])
    res["pts_per_m"] = res["mean"] / res["cost_m"].clip(lower=3.8)
    res = res.sort_values("mean", ascending=False).reset_index(drop=True)
    res.round(2).to_csv(out_path, index=False)

    # 6. report
    print("\n" + "="*74); print(f"TOP {top} by projected points (GW{gw_from}-{gw_to}, 90% CI)"); print("="*74)
    cols = ["player","pos","team","cost_m","nfix","mean","p5","p95","pts_per_m","own"]
    print(res.head(top)[cols].round(2).to_string(index=False))
    print("\nBest value (pts per £m, >=2 expected starts):")
    val = res[res.nfix >= max(1, (gw_to-gw_from+1)//2)].sort_values("pts_per_m", ascending=False)
    print(val.head(10)[["player","pos","team","cost_m","mean","pts_per_m"]].round(2).to_string(index=False))
    print("\nHighest ceiling (best captain, by p95):")
    cap = res.sort_values("p95", ascending=False)
    print(cap.head(6)[["player","pos","team","mean","p95"]].round(1).to_string(index=False))
    print(f"\nSaved: {out_path}")
    return res


def _demo_signal_frame(players):
    """Illustrative offline signals so the layer is exercised without the API."""
    names = players.web_name.tolist()
    sig = pd.DataFrame({"name": names, "status": "a", "chance_play": 1.0, "news": "",
                        "ep_next": 0.0, "pen_order": np.nan, "fk_order": np.nan,
                        "corner_order": np.nan})
    # REAL set-piece orders captured from the live FPL API (see fpl_live_data)
    real = {"Saka": (1, 2, 2), "Rice": (None, 1, 1), "Ødegaard": (3, None, 4),
            "Trossard": (4, None, None), "Madueke": (None, None, 3)}
    for nm, (po, fk, co) in real.items():
        m = sig.name == nm
        sig.loc[m, "pen_order"] = po
        sig.loc[m, "fk_order"] = fk
        sig.loc[m, "corner_order"] = co
    return sig


def main():
    ap = argparse.ArgumentParser(description="FPL 2026/27 projection — auto-pulls all free data by default")
    ap.add_argument("--gw-from", type=int, default=1)
    ap.add_argument("--gw-to", type=int, default=6)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--offline", action="store_true", help="skip all network; use local files only")
    ap.add_argument("--no-understat", action="store_true", help="disable Understat pull")
    ap.add_argument("--no-clubelo", action="store_true", help="disable ClubElo pull")
    ap.add_argument("--bootstrap", default=None, help="use a saved FPL bootstrap JSON instead of live")
    ap.add_argument("--results", default=None, help="use a specific football-data E0.csv instead of auto-download")
    ap.add_argument("--understat", default=None, help="override Understat season start year")
    ap.add_argument("--lineups", default=None, help="confirmed-lineups JSON (optional)")
    ap.add_argument("--props", default=None, help="anytime-goalscorer odds CSV (PAID, optional)")
    ap.add_argument("--apifootball-key", default=None,
                    help="API-Football key (free tier) -> injuries, lineups, kickoff dates")
    ap.add_argument("--af-season", type=int, default=2026)
    ap.add_argument("--af-budget", type=int, default=20, help="max API-Football requests this run")
    ap.add_argument("--af-lineup-fixtures", default=None,
                    help="comma-separated fixture ids to pull confirmed XIs for")
    ap.add_argument("--fm-export", default=None,
                    help="Football Manager export (.html/.csv) -> attribute priors for players with no PL history")
    ap.add_argument("--players", default=None, metavar="CSV",
                    help="fpl_players.csv from pull_fpl.py (roster + signals in one file)")
    ap.add_argument("--fpl-strength", action="store_true",
                    help="use FPL's own team strength ratings as a team prior")
    ap.add_argument("--history-seasons", type=int, default=None, metavar="YEAR",
                    help="calibrate hyperparameters on PL history from YEAR (e.g. 1993)")
    ap.add_argument("--demo-signals", action="store_true", help="inject illustrative signals when offline")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--out", default="/mnt/user-data/outputs/projection_2627_run.csv")
    a = ap.parse_args()
    af_fx = [int(x) for x in a.af_lineup_fixtures.split(',')] if a.af_lineup_fixtures else None
    run_projection(a.gw_from, a.gw_to, a.top, a.bootstrap, a.understat, a.lineups,
                   a.props, None, a.draws, a.demo_signals, a.offline,
                   a.no_understat, a.no_clubelo, a.results,
                   a.apifootball_key, a.af_season, a.af_budget, af_fx, a.fm_export,
                   a.history_seasons, a.fpl_strength, a.players,
                   a.out)


if __name__ == "__main__":
    main()
</content>
