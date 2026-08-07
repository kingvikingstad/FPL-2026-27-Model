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
import argparse, json, os, sys, tempfile, urllib.request, warnings
warnings.filterwarnings("ignore")

# Windows consoles default to a legacy codepage (cp1252), which raises
# UnicodeEncodeError on player names the FPL API returns as-is — Ødegaard,
# Martínez, Šeško, N'Golo and so on. Force UTF-8 on stdout/stderr so the ranked
# table prints intact rather than dying two hours into a projection run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                      # pre-3.7 streams / redirected pipes
        pass

import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/fpl")   # harmless if absent; sibling imports resolve via the script's own dir

from bayes_model import TeamModel, project, player_posteriors
from roster import (load_fpl_bootstrap, build_players_2627, apply_transfers,
                    demo_roster, calibrate_cold_start)
import signals as sg

FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
LOCAL_E0 = "/mnt/user-data/uploads/E0.csv"
# Legacy per-gameweek player panel. Optional: if absent, the prior layer is
# rebuilt from the live FPL bootstrap payload instead (see step 2b below).
LOCAL_HIST = "/mnt/user-data/uploads/fpl-data-stats.csv"
# scratch dir + default output dir: portable across Linux/Mac/Windows (the two
# original hardcoded paths above/below were sandbox-specific and fail on Windows)
_SCRATCH = tempfile.gettempdir()
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
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
                    tmp = os.path.join(_SCRATCH, f"E0_{code}.csv"); df.to_csv(tmp, index=False)
                    log(f"results auto-downloaded: football-data {code} ({label}, {len(df)} matches)")
                    return tmp
                log(f"football-data {code} published but only "
                    f"{len(df.dropna(subset=['FTHG','FTAG']))} played matches — too few to fit",
                    used=False)
            except Exception as e:
                # A 404 on the CURRENT season code simply means that season's file
                # doesn't exist yet (preseason) — expected, not a failure.
                if label == "current" and "HTTPError" in type(e).__name__:
                    log(f"football-data {code} not published yet (preseason, expected) "
                        f"-> falling back to prior season", used=False)
                else:
                    log(f"football-data {code} unavailable ({type(e).__name__})", used=False)
    if os.path.exists(LOCAL_E0):
        log(f"results from local file: {LOCAL_E0}"); return LOCAL_E0
    raise FileNotFoundError("no results source available")


# ---------------------------------------------------------------------------
# Source resolvers (try live free source -> fall back to local, always logged)
# ---------------------------------------------------------------------------
def resolve_bootstrap(path=None, timeout=15, offline=False):
    """Return (roster_df, signals_df, json_path) from the FPL API or a saved JSON;
    if neither is reachable, fall back to the local demo roster (no live signals).
    `json_path` is returned so the caller can rebuild season-total priors from the
    same payload — see roster.season_aggregate_from_bootstrap."""
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
        # Last resort. demo_roster() itself needs the legacy upload CSVs, so if
        # those are absent too there is genuinely no roster to project — say so
        # plainly instead of surfacing a bare pandas FileNotFoundError.
        try:
            return demo_roster(), None, None
        except FileNotFoundError as e:
            raise SystemExit(
                "Could not reach the FPL API and no local roster is available.\n"
                f"  (demo fallback needs {e.filename})\n"
                "Fix: re-run with network access, or generate a roster file first:\n"
                "     python pull_fpl.py\n"
                "     python run_2627.py --players fpl_players.csv --gw-from 1 --gw-to 6"
            ) from e
    # build roster + signals from the same JSON
    tmp = os.path.join(_SCRATCH, "_bootstrap.json")
    with open(tmp, "w", encoding="utf-8") as f: json.dump(raw, f)
    roster = load_fpl_bootstrap(tmp)
    sig = sg.load_fpl_signals(tmp)
    return roster, sig, tmp


def resolve_understat(season, offline=False):
    if offline:
        log("Understat skipped (offline)", used=False); return None
    try:
        # Preseason, the CURRENT season exists as an Understat key but contains
        # no players yet. An empty result is useless as a prior and previously
        # crashed downstream, so treat it as a miss and step back a season —
        # last season's shot volumes are exactly the prior we want anyway.
        for s in [str(season), str(int(season) - 1)]:
            u = sg.load_understat_shots(s)
            if u is not None and len(u) and "time" in getattr(u, "columns", []):
                log(f"Understat shots season {s} ({len(u)} players)")
                return u
            log(f"Understat season {s} has no player data yet "
                f"({'season not started' if s == str(season) else 'unexpected'})",
                used=False)
        return None
    except ModuleNotFoundError:
        # Worth fixing rather than shrugging off: Understat supplies TRUE
        # non-penalty xG, which is the one thing the bootstrap-derived prior
        # approximates. One pip install materially sharpens attacking rates.
        log("Understat needs a package -> run:  py -m pip install understatapi "
            "  (adds true non-penalty xG; without it npxG stays approximated)",
            used=False)
        return None
    except Exception as e:
        log(f"Understat unavailable ({type(e).__name__}) -> FPL npxGI prior only", used=False)
        return None


def resolve_clubelo(offline=False):
    if offline:
        log("ClubElo skipped (offline)", used=False); return None
    try:
        e = sg.load_clubelo("today")
        log(f"ClubElo snapshot ({len(e)} clubs, {e.attrs.get('clubelo_date','?')})")
        return e
    except Exception as ex:
        # surface the real cause — "RuntimeError" alone is undiagnosable
        log(f"ClubElo unavailable -> {ex}", used=False)
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
                   out_path=None, fixture_odds=True, outright_odds=None,
                   market_weight=0.5, all_players=False):
    if out_path is None:
        os.makedirs(_OUT_DIR, exist_ok=True)
        out_path = os.path.join(_OUT_DIR, "projection_2627_run.csv")
    fd_code, us_year = current_season()
    understat_season = understat_season or us_year
    print("="*74); print(f"FPL 2026/27 projection  |  horizon GW{gw_from}-{gw_to}  |  S={S} draws"
                          f"{'  |  OFFLINE' if offline else '  |  auto-pulling free sources'}")
    print("="*74); print("Data sources:")

    # 1. results (auto-download latest) + ClubElo -> team model
    e0 = resolve_results(results_path, offline)
    elo = resolve_clubelo(offline or no_clubelo)
    # 1a. MARKET TEAM PRIOR. Season outright odds are the only forward-looking
    #     team signal available all year round, and crucially they cover the
    #     promoted clubs, which have no top-flight data of any kind. Converted to
    #     a pseudo-Elo frame so the existing ClubElo blend consumes it unchanged.
    if outright_odds:
        try:
            import market_odds as mo
            oelo = mo.outright_to_elo(outright_odds)
            elo = oelo if elo is None else (
                pd.concat([elo[["team", "Elo"]], oelo]).groupby("team", as_index=False).Elo.mean())
            log(f"outright market odds -> team prior ({len(oelo)} clubs, {outright_odds})")
        except Exception as e:
            log(f"outright odds unusable ({type(e).__name__}: {e})", used=False)
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
            import history as Hy, json as _json
            cachep = os.path.join(_SCRATCH, "history_hyperparams.json")
            if os.path.exists(cachep):
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
    bs_path = None
    if players_csv:
        from roster import load_pull_csv
        roster, fsig = load_pull_csv(players_csv)
        log(f"roster+signals from pull_fpl CSV: {players_csv} ({len(roster)} players)")
    else:
        roster, fsig, bs_path = resolve_bootstrap(None if offline else bootstrap_path,
                                                  offline=offline)
    if transfers:
        roster = apply_transfers(roster, transfers); log(f"applied {len(transfers)} transfer overrides")

    # 2b. PRIOR EVIDENCE. The rate/availability posteriors and the cold-start
    #     price calibration both need last season's per-player totals. Prefer the
    #     legacy per-gameweek panel if it happens to be present; otherwise rebuild
    #     an equivalent aggregate from the bootstrap payload we already have, so
    #     the pipeline runs anywhere with no extra data files.
    import roster as rst
    agg = None
    if os.path.exists(LOCAL_HIST):
        log(f"player priors from per-gameweek panel: {LOCAL_HIST}")
    elif players_csv:
        try:
            agg = rst.season_aggregate_from_pull_csv(players_csv)
            log(f"player priors rebuilt from pull_fpl CSV ({len(agg)} players, season totals)")
        except Exception as e:
            log(f"could not rebuild priors from {players_csv} ({type(e).__name__}: {e})", used=False)
    elif bs_path:
        try:
            agg = rst.season_aggregate_from_bootstrap(bs_path)
            played = int((agg.minutes > 0).sum())
            log(f"player priors rebuilt from FPL bootstrap season totals "
                f"({len(agg)} players, {played} with minutes) — npxG approximated, "
                f"Understat layer refines it")
        except Exception as e:
            log(f"could not rebuild priors from bootstrap ({type(e).__name__}: {e})", used=False)
    if agg is None and not os.path.exists(LOCAL_HIST):
        raise FileNotFoundError(
            "No prior-evidence source available: the per-gameweek panel "
            f"({LOCAL_HIST}) is absent and no bootstrap/CSV roster was reachable. "
            "Run with network access, or pass --players fpl_players.csv "
            "(generate it with: python pull_fpl.py).")

    players, _ = build_players_2627(roster, agg=agg)
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
                os.makedirs(_OUT_DIR, exist_ok=True)
                kd.to_csv(os.path.join(_OUT_DIR, "kickoff_dates_2627.csv"), index=False)
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
    if ushots is not None and len(ushots):
        before = players.npxgi_alpha.copy()
        players = sg.apply_understat_prior(players, ushots)
        changed = int((players.npxgi_alpha != before).sum())
        if changed:
            log(f"applied Understat shot-volume prior to {changed} players "
                f"(true non-penalty xG replaces the approximation)")
        else:
            log("Understat loaded but matched 0 players by name", used=False)
    if props_path:
        try:
            props = pd.read_csv(props_path); od = dict(zip(props.iloc[:,0], props.iloc[:,1]))
            players = sg.fuse_prop_odds(players, od); log(f"fused {len(od)} player-prop odds (PAID)")
        except Exception as e:
            log(f"props file unreadable ({e})", used=False)

    # 4c. PER-FIXTURE MARKET ODDS. Highest-resolution team signal there is:
    #     actual bookmaker prices on the exact fixtures being projected.
    market_map = {}
    if fixture_odds and not offline:
        try:
            import market_odds as mo
            from schedule_2627 import schedule as _sched
            fx = mo.load_fixture_odds()
            if len(fx):
                _, slong = _sched()
                market_map = mo.fixture_odds_to_market_map(fx, slong)
                in_win = {k: v for k, v in market_map.items()
                          if gw_from <= k[0] <= gw_to}
                market_map = in_win
                log(f"per-fixture market odds: {len(fx)} EPL fixtures priced, "
                    f"{len(market_map)//2 if market_map else 0} inside GW{gw_from}-{gw_to} "
                    f"(blend weight {market_weight})")
            else:
                log("per-fixture odds not posted yet (bookmakers price EPL ~1-2 "
                    "weeks out) -> team model + Elo/outright prior only", used=False)
        except Exception as e:
            log(f"fixture odds unavailable ({type(e).__name__}: {e})", used=False)

    # 5. project + rank
    res, fixframe = project(players, tm, tsamp, gw_from, gw_to, S=S,
                            market=market_map, market_weight=market_weight,
                            return_fixtures=True)
    res = res.drop_duplicates("player")
    res["cost_m"] = np.where(res["cost"] > 30, res["cost"] / 10.0, res["cost"])
    res["pts_per_m"] = res["mean"] / res["cost_m"].clip(lower=3.8)
    res["status"] = "projected"

    # 5b. ALL PLAYERS. `project` necessarily skips anyone whose club has no
    #     2026/27 fixtures. Right now that is every player still listed at a
    #     relegated club, because the FPL API has not rolled over to 26/27 yet.
    #     Dropping them silently makes the export look complete when it isn't,
    #     so carry them through with an explicit status instead.
    if all_players:
        done = set(res.player)
        miss = players[~players.web_name.isin(done)].copy()
        if len(miss):
            sched_teams = set(fixframe.team) if len(fixframe) else set()
            extra = pd.DataFrame({
                "id": miss.get("id", -1), "player": miss.web_name, "pos": miss.pos,
                "team": miss.team, "own": miss.own, "cost": miss.cost, "nfix": 0,
                "mean": 0.0, "sd": 0.0, "p5": 0.0, "p25": 0.0, "median": 0.0,
                "p75": 0.0, "p95": 0.0, "p_ceiling": 0.0, "p_floor": 1.0,
                "pts_per_fix": 0.0,
                "exp_starts": 0.0, "cold_start": miss.get("cold_start", False),
            })
            extra["cost_m"] = np.where(extra["cost"] > 30, extra["cost"] / 10.0, extra["cost"])
            extra["pts_per_m"] = 0.0
            extra["status"] = np.where(
                extra.team.isin(sched_teams), "no fixtures in window",
                "club not in 2026/27 Premier League (relegated / not yet updated)")
            res = pd.concat([res, extra], ignore_index=True)
            log(f"included {len(extra)} unprojected players (no 26/27 fixtures) "
                f"— see the `status` column")

    res = res.sort_values(["mean", "own"], ascending=False).reset_index(drop=True)
    res.insert(0, "rank", np.where(res["mean"] > 0, res.index + 1, np.nan))
    res["adj_delta"] = 0.0          # <- your manual override column
    res["adj_total"] = res["mean"]
    res["gw_from"] = gw_from; res["gw_to"] = gw_to

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    COLS = ["rank", "player", "pos", "team", "status", "cost_m", "own", "nfix",
            "exp_starts", "mean", "sd", "p5", "p25", "median", "p75", "p95",
            "pts_per_fix", "pts_per_m", "p_ceiling", "p_floor", "cold_start",
            "adj_delta", "adj_total", "gw_from", "gw_to", "id"]
    res[[c for c in COLS if c in res.columns]].round(3).to_csv(out_path, index=False)

    # companion exports: per-fixture context and team strength, both editable
    fix_path = os.path.join(out_dir or ".", f"fixtures_gw{gw_from}_{gw_to}.csv")
    if len(fixframe):
        fixframe.round(4).to_csv(fix_path, index=False)
    team_path = os.path.join(out_dir or ".", "team_strength_2627.csv")
    try:
        import market_odds as mo
        mo.write_team_inputs(team_path, tsamp, market_map)
    except Exception:
        team_path = None

    # 6. report
    live = res[res.nfix > 0]
    print("\n" + "="*74); print(f"TOP {top} by projected points (GW{gw_from}-{gw_to}, 90% CI)"); print("="*74)
    cols = ["player","pos","team","cost_m","nfix","mean","p5","p95","pts_per_m","own"]
    print(live.head(top)[cols].round(2).to_string(index=False))
    print("\nBest value (pts per £m, >=half the window's fixtures):")
    val = live[live.nfix >= max(1, (gw_to-gw_from+1)//2)].sort_values("pts_per_m", ascending=False)
    print(val.head(10)[["player","pos","team","cost_m","mean","pts_per_m"]].round(2).to_string(index=False))
    print("\nHighest ceiling (best captain, by p95):")
    cap = live.sort_values("p95", ascending=False)
    print(cap.head(6)[["player","pos","team","mean","p95","p_ceiling"]].round(2).to_string(index=False))

    print("\n" + "-"*74)
    print(f"Players projected : {len(live)}"
          + (f"   (+{len(res)-len(live)} carried with no 26/27 fixtures)" if len(res) > len(live) else ""))
    print(f"Cold-start priors : {int(res.cold_start.fillna(False).sum())} "
          f"(no PL history — price/position prior, widest uncertainty)")
    nteam = live.team.nunique() if len(live) else 0
    if nteam < 20:
        missing = sorted(set(fixframe.team) - set(live.team)) if len(fixframe) else []
        print(f"WARNING: only {nteam}/20 clubs have players. Missing: {', '.join(missing) or 'n/a'}")
        print("         The FPL API is still serving the previous season's squads, so")
        print("         promoted-club players do not exist yet. Re-run after the game")
        print("         rolls over to 26/27 for a complete board.")
    print("-"*74)
    print(f"\nSaved projections : {out_path}")
    if len(fixframe):
        print(f"Saved fixtures    : {fix_path}")
    if team_path:
        print(f"Saved team model  : {team_path}")
    print("\nThe projections CSV has an `adj_delta` column for your own manual "
          "adjustments\n(injury news, eye-test, differential punts) — edit it and "
          "recompute `adj_total`.")
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
    # --- betting market ----------------------------------------------------
    ap.add_argument("--no-fixture-odds", action="store_true",
                    help="skip live per-fixture betting odds (football-data fixtures.csv)")
    ap.add_argument("--outright-odds", default=None, metavar="CSV",
                    help="season outright odds CSV (title/top4/relegation) -> team prior. "
                         "Create a blank one with --write-templates")
    ap.add_argument("--market-weight", type=float, default=0.5, metavar="W",
                    help="0..1 weight on market odds vs the model's own team "
                         "expectation, per priced fixture (default 0.5)")
    ap.add_argument("--all-players", action="store_true",
                    help="include players with no 26/27 fixtures in the export, "
                         "flagged in the `status` column, instead of dropping them")
    ap.add_argument("--write-templates", action="store_true",
                    help="write editable input templates (outright odds) and exit")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: ../outputs/projection_2627_run.csv next to src/)")
    a = ap.parse_args()
    if a.write_templates:
        import market_odds as mo
        os.makedirs(_OUT_DIR, exist_ok=True)
        p = mo.write_outright_template(os.path.join(_OUT_DIR, "outright_odds.csv"))
        print(f"Wrote editable template: {p}")
        print("\nFill in ANY ONE column for the 20 clubs (decimal odds), save, then:")
        print("  python run_2627.py --outright-odds ../outputs/outright_odds.csv "
              "--gw-from 1 --gw-to 6")
        print("\n  title_odds       odds to win the league        e.g. 2.50")
        print("  top4_odds        odds to finish top four        e.g. 1.80")
        print("  relegation_odds  odds to be relegated           e.g. 4.00")
        print("  predicted_finish your own 1-20 forecast instead of odds")
        return
    af_fx = [int(x) for x in a.af_lineup_fixtures.split(',')] if a.af_lineup_fixtures else None
    run_projection(a.gw_from, a.gw_to, a.top, a.bootstrap, a.understat, a.lineups,
                   a.props, None, a.draws, a.demo_signals, a.offline,
                   a.no_understat, a.no_clubelo, a.results,
                   a.apifootball_key, a.af_season, a.af_budget, af_fx, a.fm_export,
                   a.history_seasons, a.fpl_strength, a.players,
                   a.out, not a.no_fixture_odds, a.outright_odds,
                   a.market_weight, a.all_players)


if __name__ == "__main__":
    main()
