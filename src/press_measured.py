"""
press_measured.py — replace judgment PPDA with measured PPDA as the season accrues.
===================================================================================
`press_index` carries two hardcoded tables: PPDA_2526 (a retrospective description of
last season) and PPDA_2627 (a manager-aware *forecast*, [JUDGMENT], low confidence for
the ten REGIME_PRESS_CLUBS). `defcon_env` turns the ratio of the two into the CBIRT
scale factor for every MID/FWD DefCon rate. So the forecast is load-bearing, and until
now nothing ever revised it against what the clubs actually did.

This module measures PPDA from 26/27 results and blends it into the judgment prior with
a weight that grows with the number of matches played. At GW1 the weight is 2%; the
prior is not displaced until the sample earns it. No call site changes: `press_index`
consults this module from inside `press_factor`.


WHY THE WEIGHT IS WHAT IT IS
----------------------------
Two estimates of a club's true 26/27 press level, both noisy, combined by inverse
variance. In log space (PPDA is a ratio and enters `defcon_env` multiplicatively, so
logs are the natural scale) the posterior mean is

    w = n / (n + k),      k = sigma2_within / tau2_prior

with `n` the club's matches played. One constant governs the whole switchover, and it
is estimable rather than assumed. Two independent estimates of it, and they disagree:

  (a) Understat PPDA, 12 seasons, 9120 team-matches (studies/press_switchover.py):
      sigma2_w = 0.1993, sigma2_b = 0.0401 -> single-match reliability r1 = 0.168.
      Year-over-year persistence of team season PPDA r = 0.776 (11 transitions), so a
      previous-season prior carries tau2 = sigma2_b(1-r2) = 0.0160, giving k = 12.5.
      Backtested out-of-sample on 220 team-seasons — held-out truth, so the estimators
      cannot share noise with it — the optimum is k = 11.5 (MSE -27.5% vs prior-only),
      and the basin is flat from k = 9 to 15. Theory and backtest agree.

  (b) The SAME backtest run operationally, i.e. with the feed this module can actually
      compute (below) rather than Understat itself: optimum k = 44-50, and k = 12 LOSES
      3.6%. Only 25/26 is testable this way, and it is a favourable draw for the prior:
      the 2425->2526 persistence was r = 0.874, the highest of the eleven transitions.

The gap is the proxy's definition mismatch (season-level r = 0.937 against Understat,
so ~12% of between-club variance is not shared) plus, possibly, that single-season luck.
With one season of proxy data the two cannot be separated. **K_DEFAULT = 40 is the
conservative resolution: it is the value that is non-negative under BOTH backtests**
(+9% operational, ~+20% Understat-native). k = 12 is optimal under one and negative
under the other, so it is not available. This is a deliberate trade of expected gain for
robustness, and the honest statement is that the truth is somewhere in [12, 50].

Schedule confounding was tested and is not the explanation — opponent-adjusting the
measured feed for who a club has played moves the optimum only from k = 49.5 to 44.2.
The adjustment is applied anyway (it is principled and free) but it is not the story.

Resulting weight schedule at k = 40:

    GW1 0.02   GW3 0.07   GW5 0.11   GW10 0.20   GW19 0.32   GW38 0.49

which is the intended behaviour: the GW1 press table is ~92% noise (single-match
reliability 0.168 on Understat's own measure, 0.137 on this one), and a 2% weight is
what that is worth. The mechanism is self-limiting — it cannot do damage early, because
early it barely does anything.


THE FEED
--------
Understat's PPDA is the quantity `press_index` is denominated in, but Understat 26/27
needs a network scrape (`sd_ingest`), so it cannot be the in-season path. The offline
source is the FPL-Core-Insights repo, and PPDA has to be rebuilt from components.

Numerator comes from `matches.csv` (team level, complete), denominator from
`playermatchstats.csv`:

    ppda = opponent attempted passes / (tackles_won + interceptions
                                        + fouls_committed + recoveries)

Two constraints forced this exact specification, both found by checking rather than
assuming:

  - `tackles` (attempted) is **100% null in 26/27**. The column exists, so a proxy built
    on it passes every import and silently divides by the wrong denominator. Only
    `tackles_won` is populated. [VERIFIED 2026-08-28]
  - `recoveries` belongs in the denominator. It is the closest thing the repo has to
    Understat's "challenges", and adding it lifts the season-level correlation from
    0.878 to 0.937. It is also the action CBIRT is actually driven by, so the proxy is
    arguably closer to what `defcon_env` needs than Understat's own definition is.

Rows with `minutes_played == 0` are dropped before aggregating; on that subset all four
denominator columns are 100% complete in 26/27 GW1, whereas `accurate_passes` is not —
hence the numerator coming from `matches.csv` instead.

The proxy is on its own scale (season means 5.0-6.9 against Understat's 8.8-14.0), so it
is mapped onto the Understat scale by a linear fit in logs, `CALIB`, fitted on 25/26
where both exist. All 20 clubs, r = 0.937, residual sd 0.053.

[VERIFIED] the variance components, the persistence, both backtests, the calibration,
           and the null-column findings — all from data in this repo.
[JUDGMENT]  K_DEFAULT = 40 (the robust point in a [12, 50] range), and K_WEAK_PRIOR = 20
            for clubs whose prior is not a previous-season measurement.
"""
from __future__ import annotations
import os, glob, re
import numpy as np, pandas as pd
import config

# ---------------------------------------------------------------- constants

# log_understat_ppda = A + B * log_proxy_ppda.  Fitted on 25/26, 20 clubs, r = 0.937,
# residual sd 0.053.  Refit with fit_calibration().
CALIB_A, CALIB_B = 0.19356, 1.28669

# The PitchAPI feed, on the same Understat scale. Fitted the same way on 25/26, all 20
# clubs, using the RATIO OF SUMS (opponent_passes summed over defensive_actions summed)
# rather than a mean of per-match ratios - PPDA is a ratio, and PitchAPI is the only
# feed that publishes both parts, so the correct season statistic is available here and
# nowhere else. r = 0.9677, residual sd 0.0381, against the proxy's 0.9370 / 0.053.
# studies/press_feed_compare.py is the pre-registered test that earned the switch.
CALIB_PA_A, CALIB_PA_B = -0.35440, 1.16706

# w = n / (n + k).  See the module docstring for why 40 and not 12.
K_DEFAULT = 40.0
# Clubs whose 26/27 prior is not a previous-season measurement of that club — the
# promoted three (no PL season at all) and the regime clubs (a manager's press at a
# DIFFERENT club). A weaker prior earns a smaller k, i.e. a faster switch. The direction
# is evidenced — the Understat-native backtest gives k = 8.25 for promoted against 11.5
# for the rest — the magnitude is [JUDGMENT], and it is deliberately conservative.
K_WEAK_PRIOR = 20.0

# Below this many matches the two-way opponent adjustment is not identified (each club
# has played too few distinct opponents), so the raw mean is used instead. Immaterial in
# practice: the blend weight at 4 matches is 0.09.
MIN_GW_FOR_OPPONENT_ADJUST = 5

# Names differ between the FPL repo and the frame spelling the rest of the model uses.
# The first two are the clubs `regime_panel.PANEL_TO_FRAME` fixes; the promoted three
# joined 26/27 under their full names. An unmapped name does not raise here — it silently
# misses `PPDA_2627` and takes LEAGUE_PPDA as its prior, which looks like a plausible
# number — so `measured_table` asserts every measured club resolves to a known key.
REPO_TO_FRAME = {"Man Utd": "Man United", "Spurs": "Tottenham",
                 "Coventry City": "Coventry", "Hull City": "Hull",
                 "Ipswich Town": "Ipswich"}

_DEN_COLS = ("tackles_won", "interceptions", "fouls_committed", "recoveries")


# ---------------------------------------------------------------- the feed

def _gw_files(season, name, upto_gw=None):
    """Every By Gameweek/GW*/<name> under a season, optionally capped at a gameweek."""
    out = []
    for f in glob.glob(os.path.join(config.repo(season), "By Gameweek", "GW*", name)):
        m = re.search(r"GW(\d+)", f)
        if not m:
            continue
        gw = int(m.group(1))
        if upto_gw is None or gw <= upto_gw:
            out.append((gw, f))
    return sorted(out)


def _team_names(season):
    t = pd.read_csv(os.path.join(config.repo(season), "teams.csv"))
    return {k: REPO_TO_FRAME.get(v, v) for k, v in t.set_index("code")["name"].items()}


def team_match_ppda(season="2026-2027", upto_gw=None):
    """One row per team-match: the measured PPDA proxy.

    Only finished Premier League matches — the `-prem-` match_id filter is the same
    guard `inseason.played()` carries, and for the same reason: cup ties against
    lower-division opposition would otherwise enter a Premier League press estimate.

    The test seam is `measured_table(d=...)`, which takes this frame directly.
    """
    mf = _gw_files(season, "matches.csv", upto_gw)
    pf = _gw_files(season, "playermatchstats.csv", upto_gw)
    if not mf or not pf:
        return pd.DataFrame(columns=["match_id", "gw", "team", "opponent", "ppda"])
    matches = pd.concat([pd.read_csv(f) for _, f in mf], ignore_index=True)
    pms = pd.concat([pd.read_csv(f) for _, f in pf], ignore_index=True)
    names = _team_names(season)

    matches = matches.drop_duplicates(subset="match_id")
    matches = matches[(matches["finished"] == True) &
                      matches["match_id"].astype(str).str.contains("-prem-")]
    if matches.empty:
        return pd.DataFrame(columns=["match_id", "gw", "team", "opponent", "ppda"])

    num = []
    for side, opp in (("home", "away"), ("away", "home")):
        num.append(pd.DataFrame({
            "match_id": matches["match_id"].values,
            "gw": matches["gameweek"].values,
            "team": matches[f"{side}_team"].map(names).values,
            "opponent": matches[f"{opp}_team"].map(names).values,
            "opp_passes": matches[f"{opp}_passes"].values,
        }))
    num = pd.concat(num, ignore_index=True).dropna(subset=["team", "opponent"])

    # denominator: per-club defensive actions, players who actually played
    pms = pms.drop_duplicates(subset=["match_id", "player_id"])
    pms = pms[pms["match_id"].astype(str).str.contains("-prem-")]
    pms = pms[pms["minutes_played"] > 0]
    players = pd.read_csv(os.path.join(config.repo(season), "players.csv"))
    club = players["team_code"].map(names)
    pl = pd.DataFrame({"player_id": players["player_id"].values, "team": club.values})
    pms = pms.merge(pl, on="player_id", how="left").dropna(subset=["team"])

    missing = [c for c in _DEN_COLS if c not in pms.columns]
    if missing:
        raise KeyError(f"playermatchstats missing PPDA denominator columns: {missing}")
    den = pms.groupby(["match_id", "team"])[list(_DEN_COLS)].sum().sum(axis=1)
    den = den.rename("den").reset_index()

    d = num.merge(den, on=["match_id", "team"], how="inner")
    # the repo CSVs read back as object where a column is sparsely populated, and
    # np.log on an object Series fails with a bare AttributeError. Coerce, then drop.
    for c in ("opp_passes", "den"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["den"] > 0) & (d["opp_passes"] > 0)]
    d["ppda"] = (d["opp_passes"] / d["den"]).astype(float)
    return d.reset_index(drop=True)


# ------------------------------------------------------- estimate + blend

def _two_way(d):
    """Additive team + opponent effects on log PPDA, by iterative demeaning.

    A club's first few fixtures are not a balanced schedule, and PPDA depends on the
    opponent (you press a possession side differently). Removing the opponent effect
    makes the early estimate fixture-neutral. Worth little — it moved the backtest
    optimum from k=49.5 to 44.2 — but it is the right shape and costs nothing.
    """
    y = np.log(d["ppda"].to_numpy(float))
    mu = float(y.mean())
    tm, om = d["team"].to_numpy(), d["opponent"].to_numpy()
    te = {t: 0.0 for t in np.unique(tm)}
    oe = {t: 0.0 for t in np.unique(om)}
    for _ in range(60):
        r = y - mu - np.array([oe.get(o, 0.0) for o in om])
        te = pd.Series(r).groupby(tm).mean().to_dict()
        r2 = y - mu - np.array([te.get(t, 0.0) for t in tm])
        oe = pd.Series(r2).groupby(om).mean().to_dict()
    return {t: mu + te.get(t, 0.0) for t in te}


def calibrate(log_proxy):
    """Proxy log-PPDA -> Understat log-PPDA, the scale `press_index` is denominated in."""
    return CALIB_A + CALIB_B * np.asarray(log_proxy, dtype=float)


# ---------------------------------------------------------------- the PitchAPI feed
def _pitch_season(season):
    """'2026-2027' -> '2026/2027', the season spelling PitchAPI uses."""
    return season.replace("-", "/")


def pitchapi_ppda(season="2026-2027", upto_gw=None):
    """One row per team-match from PitchAPI: team, opponent, ppda, and the numerator and
    denominator behind it. Empty frame (never an exception) if the feed is unreachable.

    `upto_gw` is NOT supported here and is deliberately ignored rather than silently
    approximated: PitchAPI carries no FPL gameweek, only a date, and mapping dates onto
    gameweeks across blank and double weeks is a guess. Callers that need a gameweek cut
    should use the repo proxy, which is keyed on the gameweek directory.
    """
    try:
        import pitchapi as pa
    except Exception:                                             # noqa: BLE001
        return pd.DataFrame()
    try:
        d = pa.team_match_advanced(_pitch_season(season), verbose=False)
    except Exception as e:                                        # noqa: BLE001
        print(f"[press] PitchAPI feed unavailable ({type(e).__name__}: {e})")
        return pd.DataFrame()
    if not len(d) or "defending.ppda" not in d.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "match_id": d["match_id"], "gw": np.nan, "team": d["team"],
        "opponent": d["opp"], "ppda": pd.to_numeric(d["defending.ppda"], errors="coerce"),
        "num": pd.to_numeric(d["defending.opponent_passes"], errors="coerce"),
        "den": pd.to_numeric(d["defending.defensive_actions"], errors="coerce"),
    })
    return out.dropna(subset=["ppda"])


def resolve_feed(season="2026-2027", upto_gw=None, feed=None, verbose=True):
    """(frame, feed_name). `feed` is 'pitchapi', 'proxy', or None for auto.

    Auto prefers PitchAPI - it agrees with the scale `press_index` is denominated in at
    r = 0.9677 against the proxy's 0.9370 - and falls back to the proxy whenever PitchAPI
    is unreachable, which is the offline case and the no-key case. The fallback is LOUD:
    a board silently built on a different feed than the operator believes is exactly the
    failure this repo keeps finding.

    A gameweek cut forces the proxy, because PitchAPI cannot honour one (see above).
    """
    want = (feed or os.environ.get("PRESS_FEED", "auto")).lower()
    if want in ("proxy", "repo"):
        return team_match_ppda(season, upto_gw), "proxy"
    if upto_gw is not None and want != "pitchapi":
        if verbose:
            print("[press] gameweek cut requested -> proxy feed (PitchAPI has no gameweek)")
        return team_match_ppda(season, upto_gw), "proxy"
    d = pitchapi_ppda(season, upto_gw)
    if len(d):
        return d, "pitchapi"
    if want == "pitchapi":
        raise RuntimeError("PRESS_FEED=pitchapi but the PitchAPI feed returned nothing")
    if verbose:
        print("[press] PitchAPI unavailable -> falling back to the component proxy")
    return team_match_ppda(season, upto_gw), "proxy"


def measured_table(season="2026-2027", upto_gw=None, opponent_adjust=True, d=None,
                   check_names=True, feed=None, verbose=True):
    """{club: measured PPDA on the press_index scale}, {club: matches played}.

    Empty dicts when the season has no finished matches — the caller then sees weight 0
    for every club and the judgment prior stands untouched.

    Two feeds, two estimators, two calibrations, and they must not be crossed:

      pitchapi  ratio of sums, sum(opponent_passes) / sum(defensive_actions), which is
                the correct season statistic for a ratio and is computable only because
                PitchAPI publishes both parts.  CALIB_PA.
      proxy     mean of per-match log PPDA (or the two-way opponent adjustment), because
                the components are not separately recoverable per club.  CALIB.

    Applying one feed's calibration to the other's estimator silently shifts every club's
    press factor, so the constants are selected here from the resolved feed name rather
    than passed in by a caller who might not know which feed answered.
    """
    if d is None:
        d, feed_used = resolve_feed(season, upto_gw, feed, verbose=verbose)
    else:
        feed_used = feed or ("pitchapi" if "den" in getattr(d, "columns", []) else "proxy")
    if len(d) == 0:
        return {}, {}
    d = d.assign(ppda=pd.to_numeric(d["ppda"], errors="coerce"))
    d = d[d["ppda"] > 0]
    if len(d) == 0:
        return {}, {}
    n = d.groupby("team").size().to_dict()

    if feed_used == "pitchapi" and {"num", "den"} <= set(d.columns):
        g = d.groupby("team")
        ros = g["num"].sum() / g["den"].sum()
        logp = np.log(ros[ros > 0]).to_dict()
        a, b = CALIB_PA_A, CALIB_PA_B
    else:
        if opponent_adjust and d["gw"].nunique() >= MIN_GW_FOR_OPPONENT_ADJUST:
            logp = _two_way(d)
        else:
            logp = np.log(d["ppda"]).groupby(d["team"]).mean().to_dict()
        a, b = CALIB_A, CALIB_B

    if check_names:
        _assert_names(logp)
    if verbose:
        print(f"[press] feed={feed_used}, {len(logp)} clubs, "
              f"{min(n.values())}-{max(n.values())} matches each")
    return {t: float(np.exp(a + b * v)) for t, v in logp.items()}, n


def _assert_names(logp):
    """Every measured club must resolve to a `press_index` key.

    A club whose repo spelling is unmapped falls through `PPDA_2627.get(club, LEAGUE_PPDA)`
    to the league average — a number that looks entirely reasonable on a board and is
    wrong. It happened: the promoted three arrive as "Hull City"/"Coventry City"/
    "Ipswich Town". Fail loudly instead.
    """
    import press_index as px
    unknown = sorted(t for t in logp if t not in px.PPDA_2627)
    if unknown:
        raise KeyError(
            f"measured clubs absent from press_index.PPDA_2627: {unknown}. "
            f"Add the repo spelling to press_measured.REPO_TO_FRAME — an unmapped club "
            f"silently takes LEAGUE_PPDA as its prior.")


def blend_weight(n, k=None, weak_prior=False):
    """w = n/(n+k). Inverse-variance weight on the measured estimate."""
    if not n or n <= 0:
        return 0.0
    if k is None:
        k = K_WEAK_PRIOR if weak_prior else K_DEFAULT
    return float(n) / (float(n) + float(k))


def fit_calibration(season="2025-2026", understat_csv=None):
    """Refit CALIB against Understat season PPDA. Needs the sd cache; returns (a, b, r).

    Not run at import — the constants are frozen so a board is reproducible without the
    scrape cache present.
    """
    import sd_ingest as sd
    path = understat_csv or os.path.join(config.SD_CACHE, "understat_team", "2526.csv.gz")
    u = pd.read_csv(path)
    u["team"] = u["team"].map(sd.normalise_team)
    lu = np.log(u["ppda"]).groupby(u["team"]).mean()
    d = team_match_ppda(season)
    lp = np.log(d["ppda"]).groupby(d["team"]).mean()
    j = pd.concat([lp.rename("lp"), lu.rename("lu")], axis=1).dropna()
    b, a = np.polyfit(j["lp"], j["lu"], 1)
    return float(a), float(b), float(j["lp"].corr(j["lu"]))


# ---------------------------------------------------------------- selftest

def selftest():
    """Offline, on synthetic fixtures. Checks the shape of the blend, not its value."""
    # weight schedule
    assert blend_weight(0) == 0.0
    assert abs(blend_weight(1, k=40) - 1 / 41) < 1e-12
    ws = [blend_weight(n, k=40) for n in range(1, 39)]
    assert all(b > a for a, b in zip(ws, ws[1:])), "weight must be monotone in n"
    assert ws[0] < 0.03 and ws[-1] < 0.55, "k=40 must stay conservative all season"
    assert blend_weight(5, weak_prior=True) > blend_weight(5), "weak prior switches faster"

    # calibration maps the proxy range onto the press_index range
    assert 8.0 < float(np.exp(calibrate(np.log(5.0)))) < 11.0
    assert 12.0 < float(np.exp(calibrate(np.log(6.9)))) < 16.0

    # opponent adjustment recovers a planted team effect from an unbalanced schedule
    rng = np.random.default_rng(7)
    teams = [f"T{i}" for i in range(10)]
    true = {t: 0.25 * (i - 4.5) / 4.5 for i, t in enumerate(teams)}
    rows = []
    for gw in range(1, 11):
        order = list(rng.permutation(teams))
        for a, b in zip(order[::2], order[1::2]):
            for x, y in ((a, b), (b, a)):
                rows.append({"match_id": f"m{gw}{x}", "gw": gw, "team": x, "opponent": y,
                             "ppda": float(np.exp(2.4 + true[x] - 0.4 * true[y]
                                                  + rng.normal(0, 0.05)))})
    d = pd.DataFrame(rows)
    est = _two_way(d)
    got = np.array([est[t] for t in teams]); want = np.array([true[t] for t in teams])
    r = float(np.corrcoef(got, want)[0, 1])
    assert r > 0.95, f"two-way opponent adjustment did not recover team effects (r={r:.3f})"

    # measured_table end to end on the synthetic frame, and the empty-season contract
    tbl, n = measured_table(d=d, check_names=False)
    assert set(tbl) == set(teams) and all(v > 0 for v in tbl.values())
    assert all(v == 10 for v in n.values()), n
    e_tbl, e_n = measured_table(d=pd.DataFrame(columns=["team", "gw", "ppda", "opponent"]))
    assert e_tbl == {} and e_n == {}, "empty season must leave the prior untouched"

    # an unmapped club spelling must raise, not silently take LEAGUE_PPDA as its prior.
    # This is the bug the promoted three actually caused on the first live run.
    try:
        measured_table(d=d.assign(team=d["team"].replace({"T0": "Hull City"})))
    except KeyError as e:
        assert "Hull City" in str(e), e
    else:
        raise AssertionError("unmapped club name did not raise")

    # object dtype is what the repo CSVs actually hand back for sparse columns, and
    # np.log on it fails with a bare AttributeError deep in pandas. Must be coerced.
    tbl2, _ = measured_table(d=d.assign(ppda=d["ppda"].astype(object)), check_names=False)
    assert set(tbl2) == set(teams)
    print("press_measured selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)

    gw = None
    for a in sys.argv[1:]:
        if a.startswith("--gw="):
            gw = int(a.split("=")[1])
    d = team_match_ppda("2026-2027", gw)
    print(f"26/27 team-matches with a measured press: {len(d)}"
          f"  (gameweeks {sorted(d['gw'].unique()) if len(d) else '-'})")
    if len(d) == 0:
        print("no finished Premier League matches yet — the judgment prior stands.")
        sys.exit(0)
    import press_index as px
    tbl, n = measured_table("2026-2027", gw)
    rows = []
    for club in sorted(tbl, key=tbl.get):
        prior = px.ppda_2627(club)
        weak = club in px.REGIME_PRESS_CLUBS or club in px.PROMOTED_PRESS_CLUBS
        w = blend_weight(n[club], weak_prior=weak)
        rows.append((club, n[club], prior, tbl[club], w,
                     float(np.exp((1 - w) * np.log(prior) + w * np.log(tbl[club])))))
    print(f"\n{'club':16s}{'n':>3s}{'prior':>8s}{'measured':>10s}{'w':>7s}{'blended':>9s}   note")
    for club, nn, prior, meas, w, out in rows:
        note = "weak prior" if (club in px.REGIME_PRESS_CLUBS
                                or club in px.PROMOTED_PRESS_CLUBS) else ""
        print(f"{club:16s}{nn:>3d}{prior:>8.1f}{meas:>10.1f}{w:>7.2f}{out:>9.1f}   {note}")
