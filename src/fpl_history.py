from __future__ import annotations
import config
"""
fpl_history.py — deep player-gameweek history from vaastav/Fantasy-Premier-League
=================================================================================
`history.py` opens by stating that historical PL data "is MATCH RESULTS ONLY (no
player stats), so it cannot inform player priors". That was true of football-data.co.uk.
It is not true of vaastav/Fantasy-Premier-League, which carries player-GAMEWEEK rows
(minutes, points, price, ICT, bonus) from 2016/17. This module is the player-side
counterpart to `history.py`'s team-side hyperparameter work, and it exists to feed the
one layer the project measures as its largest validated gain: minutes.

WHY MINUTES AND NOT EVERYTHING
------------------------------
`multiseason_priors.to_priors` builds four priors. They do NOT backfill equally, and
pretending otherwise would quietly fabricate evidence:

  start_a/start_b  (Beta, minutes)   ALL 10 seasons.  <- the reason this module exists
  npxgi / xa       (Gamma)           see PENALTIES below; NOT wired in yet
  defcon           (Gamma)           2025/26 ONLY. Cannot be backfilled at all.

`defensive_contribution` appears in exactly one season because the STAT ITSELF is new
(FPL introduced DefCon scoring in 2025/26). Nine seasons of history buy nothing here,
and any attempt to proxy it from the tackles/CBI/recoveries columns runs into a second
problem: those exist for 2016/17-2018/19 and 2025/26 but are ABSENT for the six seasons
between. A proxy fitted on the old era and applied to the new one would be crossing a
definitional change with a six-season gap in the middle. Not attempted.

PENALTIES — why xG is not wired into the priors yet
---------------------------------------------------
`to_priors` is explicit that its Gamma is built on NON-penalty xG (`build_2425_panel`
does `npxg = xg - 0.79 * pen_attempts`). merged_gw.csv carries `penalties_missed` but
NOT `penalties_scored`, so penalty ATTEMPTS cannot be recovered from it, and
`expected_goals` here is penalty-INCLUSIVE. Feeding it in raw would inflate the npxGI
prior for every penalty taker — the exact players the model is most sensitive to. The
columns are loaded and flagged (`has_xg`), deliberately not consumed. Resolving this
needs the penalty split from Understat via a verified crosswalk (G5).

SCHEMA ERAS — the same file is three different files
----------------------------------------------------
  2016/17-2018/19  no `position`, no `team`, no `starts`, no xG; HAS tackles/CBI/recoveries
  2019/20-2021/22  no `position`/`team` until 2020/21; no xG; defensive columns GONE
  2022/23-2024/25  `starts` and `expected_goals`/`expected_assists` appear
  2025/26          `defensive_contribution` and the defensive columns return

Handled by deriving what is derivable and FLAGGING what is derived, never by silently
filling. `position` comes from players_raw.element_type when the column is absent;
`starts` falls back to minutes >= 60, which is what `multiseason_priors` and
`build_pms` already use as their own definition of a start, so the fallback is
consistent with the rest of the project rather than a new convention.

IDENTITY (G5)
-------------
merged_gw.csv keys on `element`, which FPL REASSIGNS every season — joining on it
across seasons would silently merge different players. players_raw.csv carries `code`,
FPL's permanent per-player id, and [VERIFIED 2026-08-09] it is exactly the repo's
`player_code`: all 841 of the 25/26 players intersect, spot-checked by name. So the
deep history joins to the existing model with NO fuzzy matching and no crosswalk —
G5 is satisfied by construction. `_load_season` reports the per-season match rate and
`load_history` refuses a season that falls below a floor.

LEAKAGE
-------
`xP`, `ep_this` and `ep_next` are dropped on read. They are FPL's own forward-looking
projections; `ep_this` in particular is not available pre-deadline in the form the file
records it, so any backtest using them scores partly against hindsight. (The known
all-zero GW35 xP bug is moot once the column is gone.)

Run:  python src/fpl_history.py --selftest     (offline, synthetic)
      python src/fpl_history.py --audit        (needs FPL_HISTORY)
"""
import os
import numpy as np, pandas as pd

# The 2026/27 directory exists but merged_gw.csv is empty until the season starts.
SEASONS = ("2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
           "2021-22", "2022-23", "2023-24", "2024-25", "2025-26")

# FPL's element_type, stable across every season.
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POS_NAME = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

# Never read into the panel — see LEAKAGE above.
LEAKY = ("xP", "ep_this", "ep_next")

# A season joining below this to players_raw means the id map is broken, not thin.
MIN_MATCH_RATE = 0.95

START_MINUTES = 60          # the project's existing definition of a start


def _read_csv(path, **kw):
    """vaastav's early files are latin-1, the rest utf-8, and nothing marks which.
    Try strict utf-8 first so a real mojibake problem surfaces rather than being
    silently mis-decoded into plausible-looking nonsense."""
    try:
        return pd.read_csv(path, encoding="utf-8", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kw)


def _id_to_code(season):
    """{element_id -> player_code} for ONE season, plus that season's position map.
    Must be rebuilt per season: FPL reassigns element ids annually."""
    raw = _read_csv(os.path.join(config.history(season), "players_raw.csv"))
    missing = {"id", "code", "element_type"} - set(raw.columns)
    if missing:
        raise RuntimeError(f"{season}/players_raw.csv is missing {sorted(missing)}; "
                           f"cannot establish player identity without it")
    code = dict(zip(raw["id"], raw["code"]))
    pos = {int(i): POS_MAP.get(int(t)) for i, t in zip(raw["id"], raw["element_type"])}
    return code, pos


def _load_season(season, verbose=True):
    path = os.path.join(config.history(season), "gws", "merged_gw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = _read_csv(path)
    if d.empty:
        raise RuntimeError(f"{season} merged_gw.csv is empty (season not started?)")
    d = d.drop(columns=[c for c in LEAKY if c in d.columns])

    code, pos_by_id = _id_to_code(season)
    d["player_code"] = d["element"].map(code)
    rate = d["player_code"].notna().mean()
    if rate < MIN_MATCH_RATE:
        raise RuntimeError(
            f"{season}: only {rate:.1%} of gameweek rows resolved to a player_code "
            f"(floor {MIN_MATCH_RATE:.0%}). The element->code map is broken; do NOT "
            f"fall back to name matching, which would merge distinct players.")
    d = d[d["player_code"].notna()].copy()
    d["player_code"] = d["player_code"].astype(int)

    # --- position: native column where present, else element_type ---
    if "position" in d.columns:
        d["pos"] = d["position"].map(lambda p: POS_NAME.get(str(p).upper()))
        d["pos_derived"] = False
    else:
        d["pos"] = d["element"].map(pos_by_id)
        d["pos_derived"] = True

    d["mins"] = pd.to_numeric(d.get("minutes"), errors="coerce").fillna(0.0)

    # --- starts: native from 2022/23, else the project's own mins>=60 rule ---
    if "starts" in d.columns:
        d["is_start"] = pd.to_numeric(d["starts"], errors="coerce").fillna(0) > 0
        d["starts_derived"] = False
    else:
        d["is_start"] = d["mins"] >= START_MINUTES
        d["starts_derived"] = True

    for c, out in (("total_points", "points"), ("value", "price"),
                   ("selected", "selected"), ("expected_goals", "xg_raw"),
                   ("expected_assists", "xa_raw")):
        d[out] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan

    d["season"] = season
    d["gw"] = pd.to_numeric(d.get("GW", d.get("round")), errors="coerce")
    d["has_xg"] = d["xg_raw"].notna().any()

    keep = ["player_code", "season", "gw", "pos", "mins", "is_start", "points",
            "price", "selected", "xg_raw", "xa_raw",
            "pos_derived", "starts_derived", "has_xg"]
    out = d[keep]
    if verbose:
        print(f"[history] {season}: {len(out):5d} player-GW rows, "
              f"{out.player_code.nunique():3d} players, match {rate:.1%}"
              f"{'  (pos derived)' if out['pos_derived'].iloc[0] else ''}"
              f"{'  (starts derived)' if out['starts_derived'].iloc[0] else ''}"
              f"{'  +xG' if out['has_xg'].iloc[0] else ''}")
    return out


def load_history(seasons=SEASONS, verbose=True) -> pd.DataFrame:
    """Player-gameweek panel across `seasons`, keyed on player_code."""
    return pd.concat([_load_season(s, verbose) for s in seasons], ignore_index=True)


def season_evidence(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the panel to one row per (player_code, season) — the shape
    `multiseason_priors` consumes as evidence."""
    g = panel.groupby(["player_code", "season"], dropna=False)
    ev = g.agg(pos=("pos", lambda s: s.dropna().iloc[-1] if s.notna().any() else None),
               mins=("mins", "sum"),
               starts=("is_start", "sum"),
               games=("mins", "size"),
               apps=("mins", lambda s: (s > 0).sum()),
               points=("points", "sum"),
               xg_raw=("xg_raw", "sum"),
               xa_raw=("xa_raw", "sum"),
               starts_derived=("starts_derived", "first")).reset_index()
    return ev


def season_weights(seasons, target_season_index=None, half_life=1.0):
    """Geometric recency weights: a season `k` back from the newest gets 0.5**(k/half_life).

    The existing two-season prior uses a flat older_weight=0.5 for the single season
    back, so half_life=1.0 reproduces that weight exactly at k=1 and then keeps
    decaying — which is the point. A player's 2016/17 minutes are not half as
    informative about 2026/27 as last season's; at half_life=1.0 they carry 0.2%.
    """
    order = list(seasons)
    n = len(order) if target_season_index is None else target_season_index + 1
    return {s: 0.5 ** ((n - 1 - i) / half_life) for i, s in enumerate(order)}


# ------------------------------------------------------------------ selftest
def _fake_season(season, n_players=40, n_gw=38, era="modern", seed=0):
    """Synthesises one season in the layout of a given era, so the selftest exercises
    the era branching without touching the network or the clone."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n_players + 1)
    rows = []
    for gw in range(1, n_gw + 1):
        for i in ids:
            m = int(rng.choice([0, 90, 75, 20], p=[.2, .5, .2, .1]))
            r = {"element": i, "minutes": m, "total_points": max(0, m // 30),
                 "value": 50 + i, "selected": 1000, "GW": gw, "xP": 3.0}
            if era in ("mid", "modern"):
                r["position"] = ["GK", "DEF", "MID", "FWD"][i % 4]
            if era == "modern":
                r["starts"] = int(m >= 60)
                r["expected_goals"] = 0.1
                r["expected_assists"] = 0.05
            rows.append(r)
    return pd.DataFrame(rows)


def selftest():
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        real_hist = config.HISTORY
        config.HISTORY = tmp
        specs = [("2016-17", "old"), ("2020-21", "mid"), ("2025-26", "modern")]
        for season, era in specs:
            os.makedirs(os.path.join(tmp, season, "gws"), exist_ok=True)
            gw = _fake_season(season, era=era)
            # 2016-17 is latin-1 on disk in the real dataset; reproduce that here so
            # the encoding fallback is genuinely exercised.
            enc = "latin-1" if season == "2016-17" else "utf-8"
            gw["name"] = "René Fake"
            gw.to_csv(os.path.join(tmp, season, "gws", "merged_gw.csv"),
                      index=False, encoding=enc)
            raw = pd.DataFrame({"id": np.arange(1, 41),
                                "code": 100000 + np.arange(1, 41),
                                "element_type": [(i % 4) + 1 for i in range(40)]})
            raw.to_csv(os.path.join(tmp, season, "players_raw.csv"), index=False)

        panel = load_history([s for s, _ in specs], verbose=False)
        assert len(panel) == 3 * 40 * 38, len(panel)
        assert panel["player_code"].min() >= 100001, "code join failed"
        assert not panel["player_code"].isna().any()

        # leakage columns must never survive the read
        assert "xP" not in panel.columns and "ep_this" not in panel.columns

        # era branching: derived where the column is absent, native where present
        by = panel.groupby("season")[["pos_derived", "starts_derived", "has_xg"]].first()
        assert by.loc["2016-17", "pos_derived"] and by.loc["2016-17", "starts_derived"]
        assert not by.loc["2020-21", "pos_derived"] and by.loc["2020-21", "starts_derived"]
        assert not by.loc["2025-26", "starts_derived"] and by.loc["2025-26", "has_xg"]
        assert not by.loc["2016-17", "has_xg"]
        assert panel["pos"].notna().all(), "position must resolve in every era"

        ev = season_evidence(panel)
        assert len(ev) == 3 * 40
        assert (ev["mins"] > 0).all()

        # a broken id map must raise, not silently drop players
        bad = os.path.join(tmp, "2020-21", "players_raw.csv")
        pd.DataFrame({"id": [999], "code": [1], "element_type": [1]}).to_csv(bad, index=False)
        try:
            _load_season("2020-21", verbose=False)
        except RuntimeError as e:
            assert "player_code" in str(e)
        else:
            raise AssertionError("a broken element->code map must raise")

        w = season_weights(["2023-24", "2024-25", "2025-26"], half_life=1.0)
        assert abs(w["2025-26"] - 1.0) < 1e-12 and abs(w["2024-25"] - 0.5) < 1e-12
        assert abs(w["2023-24"] - 0.25) < 1e-12

        print(f"SELFTEST OK: {len(panel)} player-GW rows across 3 schema eras "
              f"(latin-1 + utf-8), code join enforced, leakage columns dropped, "
              f"derived fields flagged, decay weights exact.")
    finally:
        config.HISTORY = real_hist
        shutil.rmtree(tmp, ignore_errors=True)


def audit():
    """Coverage report against the real clone."""
    panel = load_history()
    ev = season_evidence(panel)
    print(f"\ntotal {len(panel)} player-GW rows, {panel.player_code.nunique()} distinct players")
    t = ev.groupby("season").agg(players=("player_code", "nunique"),
                                 mins=("mins", "sum"),
                                 starts_derived=("starts_derived", "first"))
    t["mins"] = (t["mins"] / 1000).round(1)
    print("\nper season (mins in thousands):")
    print(t.to_string())


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    elif "--audit" in sys.argv:
        import warnings; warnings.filterwarnings("ignore")
        audit()
    else:
        print(__doc__)
