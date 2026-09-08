import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
export_team_projections.py — the team layer, made auditable
============================================================
The player board consumes team strength as a black box: a lambda arrives per fixture
and the simulation composes points around it. This export opens that box, because the
team layer is where the betting-odds signal actually enters and it is the single
documented largest predictive gain in the project (PROJECT_KNOWLEDGE §6.1).

Two tables:

  team_projections_season.csv   one row per club — the full provenance chain from
    published decimal odds -> de-vigged probabilities -> market strength z -> market
    Elo -> blend with ClubElo -> the fitted posterior attack/defence on the model's
    own log scale. Every intermediate is a column, so a disagreement with the model
    can be localised to the step that caused it rather than argued about in the
    aggregate. Carries the press/PPDA covariate and the DefCon xGA environment too,
    since those are team-level quantities that condition player DefCon.

  team_projections_gw.csv       one row per team-gameweek — opponent, venue, the
    posterior lambda for and against with credible intervals, and the derived match
    outcome distribution (win/draw/loss, clean sheet, both-teams-to-score).

PROBABILITIES ARE POSTERIOR-PREDICTIVE, NOT PLUG-IN
---------------------------------------------------
P(clean sheet) is computed as mean_over_draws(exp(-lambda_against)), NOT
exp(-mean(lambda_against)). These are not the same number: exp(-x) is convex, so by
Jensen the plug-in version is biased DOWNWARD — it understates every clean sheet.
The engine itself gets this right (it draws `rng.poisson(lam_against) == 0` per
posterior draw, bayes_model.py:413), so the plug-in form would also have put this
export out of step with the board it is meant to explain. Both are emitted
(`p_clean_sheet` and `p_clean_sheet_plugin`) so the size of the gap is visible.

Match outcomes come from independent Poisson draws per posterior sample, which is
the same generative assumption the engine makes. That ignores goal-level dependence
within a match (Dixon-Coles low-score correction); it is not corrected here because
the engine does not correct it either, and introducing it in the export only would
make the two disagree. Flagged, not silently patched.

Run:  python scripts/export_team_projections.py
Env:  GW_HI (10), DRAWS (3000), MARKET_ODDS=off, MARKET_WEIGHT (0.6)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import core_insights as ci, bayes_model
import market_odds as mo
import press_index as px
from bayes_model import TeamModel
from schedule_2627 import schedule, PROMOTED

GW_HI = int(_os.environ.get("GW_HI", "10"))
S = int(_os.environ.get("DRAWS", "3000"))
MARKET_WEIGHT = float(_os.environ.get("MARKET_WEIGHT", "0.6"))
USE_MARKET = _os.environ.get("MARKET_ODDS") != "off"

OUT_SEASON = _os.path.join(config.OUTPUTS, "team_projections_season.csv")
OUT_GW = _os.path.join(config.OUTPUTS, f"team_projections_gw1_{GW_HI}.csv")


def odds_provenance():
    """The betting layer, one row per club, every intermediate retained."""
    odds = mo.ODDS_2627
    p_title = mo._devig({t: 1.0 / o["title"] for t, o in odds.items()})
    p_rel_raw = mo._devig({t: 1.0 / o["releg"] for t, o in odds.items()})
    # the relegation market is de-vigged to a 3-team field, matching implied_strength
    p_rel = {t: min(v * 3.0, 0.99) for t, v in p_rel_raw.items()}
    z = mo.implied_strength(odds)
    melo = mo.market_elo_frame(odds=odds).set_index("team")["Elo"]
    return pd.DataFrame({
        "team": list(odds.keys()),
        "title_odds_dec": [odds[t]["title"] for t in odds],
        "releg_odds_dec": [odds[t]["releg"] for t in odds],
        "p_title_devig": [p_title[t] for t in odds],
        "p_relegation": [p_rel[t] for t in odds],
        "market_strength_z": [float(z[t]) for t in odds],
        "market_elo": [float(melo[t]) for t in odds],
    })


def build():
    _d26, t26, _ = ci.load(base=config.repo("2026-2027"))
    elo_base = ci.to_elo_frame(t26)
    pclub = ci.promoted_prior_from_elo(t26)["per_club"]

    elo = elo_base
    if USE_MARKET:
        elo = mo.blended_elo(elo_base, market_weight=MARKET_WEIGHT)
        print(f"[team] betting-odds Elo blended at weight {MARKET_WEIGHT}")
    else:
        print("[team] MARKET_ODDS=off — ClubElo only")

    # Same E0 the board fitted on. With INSEASON=on the board appends 26/27 xG matches
    # to the stack, and a team layer fitted without them would describe a different
    # posterior than the projections this file is published alongside.
    _e0 = config.E0_RECON
    if _os.environ.get("INSEASON", "on").lower() in ("on", "1", "true"):
        import inseason as _ins
        _mx = _ins.played(upto_gw=int(_os.environ.get("INSEASON_UPTO", "38")),
                          require_xg=True, verbose=False)
        if len(_mx) >= _ins.MIN_XG_MATCHES:
            _e0, _n = _ins.stack_e0(
                _mx, weight=int(_os.environ.get("INSEASON_W_MATCH", int(_ins.W_MATCH))),
                weight_promoted=int(_os.environ.get("INSEASON_W_PROMOTED",
                                                    int(_ins.W_MATCH_PROMOTED))),
                verbose=True)
    tm = TeamModel(promoted_per_club=pclub).fit(e0_path=_e0,
                                                clubelo=elo, clubelo_weight=0.45)
    bayes_model.rng = np.random.default_rng(7)
    ts = tm.sample_2627(S=S)
    idx, mu, home, A, D = ts["idx"], ts["mu"], ts["home"], ts["att"], ts["dfn"]

    # ---------- per team-gameweek ----------
    _, long = schedule()
    orng = np.random.default_rng(11)     # one stream, so the table is reproducible
    rows = []
    for _, r in long[long.gameweek <= GW_HI].iterrows():
        if r.team not in idx or r.opp not in idx:
            continue
        gw = int(r.gameweek)
        he = bayes_model._home_effect(home, gw, bool(r.is_home))
        ho = bayes_model._home_effect(home, gw, not bool(r.is_home))
        lf = np.exp(mu + he + A[:, idx[r.team]] - D[:, idx[r.opp]])   # (S,)
        la = np.exp(mu + ho + A[:, idx[r.opp]] - D[:, idx[r.team]])
        # match outcome: independent Poisson per posterior draw (engine's assumption)
        g_for = orng.poisson(lf)
        g_ag = orng.poisson(la)
        rows.append({
            "team": r.team, "gw": gw, "opponent": r.opp, "is_home": int(bool(r.is_home)),
            "lam_for": float(lf.mean()),
            "lam_for_p5": float(np.percentile(lf, 5)),
            "lam_for_p95": float(np.percentile(lf, 95)),
            "lam_against": float(la.mean()),
            "lam_against_p5": float(np.percentile(la, 5)),
            "lam_against_p95": float(np.percentile(la, 95)),
            # posterior-predictive: integrate exp(-lam) over draws, do NOT plug in the mean
            "p_clean_sheet": float(np.exp(-la).mean()),
            "p_clean_sheet_plugin": float(np.exp(-la.mean())),
            "p_concede_2plus": float((g_ag >= 2).mean()),
            "p_win": float((g_for > g_ag).mean()),
            "p_draw": float((g_for == g_ag).mean()),
            "p_loss": float((g_for < g_ag).mean()),
            "p_btts": float(((g_for > 0) & (g_ag > 0)).mean()),
            "home_discount_applied": int(gw <= 3),
        })
    G = pd.DataFrame(rows)
    G["exp_league_pts"] = 3 * G["p_win"] + G["p_draw"]

    # ---------- per team, season layer ----------
    strength = pd.DataFrame({
        "team": list(idx.keys()),
        "att_strength": [float(A[:, idx[t]].mean()) for t in idx],
        "att_sd": [float(A[:, idx[t]].std()) for t in idx],
        "def_strength": [float(D[:, idx[t]].mean()) for t in idx],
        "def_sd": [float(D[:, idx[t]].std()) for t in idx],
    })
    strength["att_rank"] = strength["att_strength"].rank(ascending=False).astype(int)
    strength["def_rank"] = strength["def_strength"].rank(ascending=False).astype(int)

    T = odds_provenance().merge(strength, on="team", how="outer")
    T = T.merge(elo_base.rename(columns={"Elo": "clubelo_base"}), on="team", how="left")
    T = T.merge(elo.rename(columns={"Elo": "blended_elo"}), on="team", how="left")
    T["market_weight"] = MARKET_WEIGHT if USE_MARKET else 0.0
    T["promoted"] = T["team"].isin(PROMOTED).astype(int)

    # press / PPDA covariate — conditions the MID/FWD DefCon (CBIRT) channel
    T["ppda_2526"] = T["team"].map(px.ppda_2526)
    T["ppda_2627"] = T["team"].map(px.ppda_2627)
    T["press_factor"] = T["team"].map(lambda c: px.press_factor(c, "2627"))
    T["press_is_new_manager_estimate"] = T["team"].isin(px.REGIME_PRESS_CLUBS).astype(int)

    # DefCon xGA environment over the window (the DEF/CBIT channel's conditioner)
    agg = G.groupby("team").agg(
        xga_env=("lam_against", "mean"),
        xg_env=("lam_for", "mean"),
        exp_pts=("exp_league_pts", "sum"),
        exp_cs=("p_clean_sheet", "sum"),
        n_fix=("gw", "count"),
        n_home=("is_home", "sum")).reset_index()
    T = T.merge(agg, on="team", how="left")

    # DefCon OPPONENT category — how much defensive work this club hands the opposition.
    # Distinct from `xga_env`, which is how much work the club does for ITSELF. Measured
    # within player on 25/26 (`defcon_team`), empirical-Bayes shrunk, and carried only
    # for positions whose split-half reliability clears the gate — DEF passes, MID does
    # not, so only DEF is joined. A club with no 25/26 rating (promoted) stays NaN
    # rather than being silently defaulted to neutral.
    try:
        import defcon_team as dct
        r = dct.ratings_for_board("2025-2026", positions=("Defender",))
        # Filter on the column, not on row 0 — if a second position is ever added here
        # an .iloc[0] check would silently ship whatever the first one happened to be.
        r = r[r["usable"].astype(bool)] if not r.empty else r
        if not r.empty:
            r = r[["team", "shrunk", "hit_shrunk", "category"]].rename(columns={
                "shrunk": "defcon_conceded_actions",
                "hit_shrunk": "defcon_conceded_hit",
                "category": "defcon_opponent_category"})
            T = T.merge(r, on="team", how="left")
        else:
            print("[team] DefCon opponent rating failed its reliability gate — "
                  "column omitted rather than shipped unusable")
    except Exception as e:                                    # noqa: BLE001
        print(f"[team] DefCon opponent rating unavailable ({e}); column omitted")

    T = T.sort_values("att_strength", ascending=False).reset_index(drop=True)
    return T, G


def main():
    T, G = build()
    T.round(4).to_csv(OUT_SEASON, index=False)
    G.round(4).to_csv(OUT_GW, index=False)

    jens = (G["p_clean_sheet"] - G["p_clean_sheet_plugin"])
    print(f"[team] wrote {len(T)} clubs -> {OUT_SEASON}")
    print(f"[team] wrote {len(G)} team-gameweeks -> {OUT_GW}")
    print(f"[team] Jensen gap on P(CS): mean +{jens.mean():.4f}, max +{jens.max():.4f} "
          f"(plug-in understates clean sheets)")

    print("\n" + "=" * 104)
    print(f"TEAM LAYER — betting-odds provenance and fitted strength (GW1-{GW_HI})")
    print("=" * 104)
    cols = ["team", "title_odds_dec", "releg_odds_dec", "market_strength_z", "market_elo",
            "clubelo_base", "blended_elo", "att_strength", "def_strength", "xg_env",
            "xga_env", "exp_pts", "exp_cs"]
    print(T[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
