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

  A column appearing in the header is NOT the same as a column being populated.
  [VERIFIED 2026-09-16] In 2022/23 `starts`, `expected_goals` and `expected_assists`
  are a literal 0 on every row for GW1-15 — FPL began publishing them at GW16 — while
  minutes are fully recorded. Zero, not null, so a presence check passes them, and 2,818
  players with 60+ minutes read as non-starters. It is the only such block in the ten
  seasons. `empty_native_gws` finds it from the data rather than a hard-coded date:
  those gameweeks take the derived `starts` rule, flagged per ROW, and their xG is
  nulled. It biased every start-persistence number built on 22/23 upward before it was
  caught (docs/START_PERSISTENCE_2026-09-07.md).

  The derived rule is not unbiased, and its bias has a known sign. On the SAME rows in
  every native season, mins>=60 undercounts read starts by 0.017-0.020 absolute (about
  -6.5%): P(proxy|native) = 0.93, P(native|proxy) ~ 0.99 [VERIFIED 2026-09-17]. So 22/23
  GW1-15 start rates are slightly LOW, as in the 16/17-21/22 proxy seasons. Do NOT
  compare the proxied GW1-15 rate (0.332) with GW16+ native (0.298) and read it as proxy
  inflation — the two periods have different rosters (566 vs 783 rows a gameweek). NaN
  instead of the proxy would be worse: `season_evidence` counts games as rows, so a
  missing start summed as 0 would restore the original defect.

  Known latent gap: xG is nulled per row, but `season_evidence` sums it with skipna, so a
  22/23 player-season gets GW16+ xG against full-season minutes (xG per 90 LOW). No
  consumer reads xG from here today; fix with sum(min_count=1) and xG-exposure minutes
  before wiring it in.

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

# Native columns that can be present in the header before FPL populates them.
NATIVE_COLS = ("starts", "expected_goals", "expected_assists")


def empty_native_gws(d, col, gw_col="gw", mins_col="mins"):
    """Gameweeks in which native `col` is present but carries no information.

    A gameweek qualifies when the column sums to zero across every row while somebody
    played minutes in it. No real gameweek can do that: if matches were played, someone
    started them and someone generated xG. A blank gameweek (no minutes at all) does not
    qualify, so a fixture-free round is never mistaken for a missing column. Returns a
    sorted list of gameweek numbers; empty when the column is absent or fully populated.
    """
    if col not in d.columns:
        return []
    g = pd.DataFrame({"gw": d[gw_col],
                      "v": pd.to_numeric(d[col], errors="coerce").fillna(0).abs(),
                      "m": pd.to_numeric(d[mins_col], errors="coerce").fillna(0)})
    t = g.groupby("gw").agg(v=("v", "sum"), m=("m", "sum"))
    return sorted(int(x) for x in t.index[(t["v"] == 0) & (t["m"] > 0)])


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
    d["gw"] = pd.to_numeric(d.get("GW", d.get("round")), errors="coerce")

    # --- feed check: native columns present in the header but not yet populated ---
    gaps = {c: empty_native_gws(d, c) for c in NATIVE_COLS}

    # --- starts: native from 2022/23, else the project's own mins>=60 rule ---
    # `starts_derived` is per ROW. A season can be native for most gameweeks and derived
    # for the ones FPL had not populated (22/23 GW1-15); consumers that require read
    # starts must filter rows on it, not test the season.
    if "starts" in d.columns:
        missing = d["gw"].isin(gaps["starts"])
        native = pd.to_numeric(d["starts"], errors="coerce").fillna(0) > 0
        d["is_start"] = native.where(~missing, d["mins"] >= START_MINUTES)
        d["starts_derived"] = missing
    else:
        d["is_start"] = d["mins"] >= START_MINUTES
        d["starts_derived"] = True

    for c, out in (("total_points", "points"), ("value", "price"),
                   ("selected", "selected"), ("expected_goals", "xg_raw"),
                   ("expected_assists", "xa_raw")):
        d[out] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
        if gaps.get(c):
            d.loc[d["gw"].isin(gaps[c]), out] = np.nan     # a structural 0 is not a 0

    d["season"] = season
    d["has_xg"] = d["xg_raw"].notna().any()

    keep = ["player_code", "season", "gw", "pos", "mins", "is_start", "points",
            "price", "selected", "xg_raw", "xa_raw",
            "pos_derived", "starts_derived", "has_xg"]
    out = d[keep]
    if verbose:
        gap_note = "".join(
            f"  ({c} EMPTY in GW{min(g)}-{max(g)}: {len(g)} gws, "
            f"{'derived' if c == 'starts' else 'nulled'})"
            for c, g in gaps.items() if g)
        sd = out["starts_derived"]
        print(f"[history] {season}: {len(out):5d} player-GW rows, "
              f"{out['player_code'].nunique():3d} players, match {rate:.1%}"
              f"{'  (pos derived)' if out['pos_derived'].iloc[0] else ''}"
              f"{'  (starts derived)' if bool(sd.all()) else ''}"
              f"{'  +xG' if out['has_xg'].iloc[0] else ''}{gap_note}")
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
               # ANY, not first: a season derived for some gameweeks is not a season of
               # read starts, and `first` reported 22/23 by whichever row sorted first.
               starts_derived=("starts_derived", "any")).reset_index()
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

        # feed check: a native column present in the header but zero until GW16 (the real
        # 22/23 shape) must be detected from the data, derived per ROW, and xG nulled.
        # GW20 is a genuinely blank gameweek (no minutes, zero starts) and must NOT be
        # mistaken for a missing column.
        os.makedirs(os.path.join(tmp, "2022-23", "gws"), exist_ok=True)
        gap = _fake_season("2022-23", era="modern", seed=7)
        early = gap["GW"] <= 15
        gap.loc[early, ["starts", "expected_goals", "expected_assists"]] = 0
        blank = gap["GW"] == 20
        gap.loc[blank, ["minutes", "starts", "expected_goals", "expected_assists"]] = 0
        gap["name"] = "Fake"
        gap.to_csv(os.path.join(tmp, "2022-23", "gws", "merged_gw.csv"), index=False)
        raw.to_csv(os.path.join(tmp, "2022-23", "players_raw.csv"), index=False)
        g = _load_season("2022-23", verbose=False)
        e = g["gw"] <= 15
        assert g.loc[e, "starts_derived"].all(), "GW1-15 starts must be flagged derived"
        assert not g.loc[~e, "starts_derived"].any(), \
            "GW16+ is read, and a blank gameweek is not a missing column"
        assert g.loc[e, "is_start"].any(), \
            "a structural zero must NOT read as a non-start (the old behaviour)"
        assert (g.loc[e, "is_start"] == (g.loc[e, "mins"] >= START_MINUTES)).all()
        assert g.loc[e, "xg_raw"].isna().all() and g.loc[e, "xa_raw"].isna().all()
        assert g.loc[~e & (g["gw"] != 20), "xg_raw"].notna().all()
        assert bool(season_evidence(g)["starts_derived"].all()), \
            "a partly-derived season must not report itself as read"

        w = season_weights(["2023-24", "2024-25", "2025-26"], half_life=1.0)
        assert abs(w["2025-26"] - 1.0) < 1e-12 and abs(w["2024-25"] - 0.5) < 1e-12
        assert abs(w["2023-24"] - 0.25) < 1e-12

        print(f"SELFTEST OK: {len(panel)} player-GW rows across 3 schema eras "
              f"(latin-1 + utf-8), code join enforced, leakage columns dropped, "
              f"derived fields flagged, empty native gameweeks derived per row, "
              f"decay weights exact.")
    finally:
        config.HISTORY = real_hist
        shutil.rmtree(tmp, ignore_errors=True)


def audit():
    """Coverage report against the real clone."""
    panel = load_history()
    ev = season_evidence(panel)
    print(f"\ntotal {len(panel)} player-GW rows, {panel.player_code.nunique()} distinct players")
    t = ev.groupby("season").agg(players=("player_code", "nunique"),
                                 mins=("mins", "sum"))
    t["starts_derived_share"] = panel.groupby("season")["starts_derived"].mean().round(3)
    t["mins"] = (t["mins"] / 1000).round(1)
    print("\nper season (mins in thousands; starts_derived_share = fraction of rows whose "
          "start is inferred from minutes):")
    print(t.to_string())
    print("\nfeed check — native columns present but EMPTY (sum 0 while minutes played):")
    found = False
    for s in SEASONS:
        path = os.path.join(config.history(s), "gws", "merged_gw.csv")
        if not os.path.exists(path):
            continue
        raw = _read_csv(path)
        raw["gw"] = pd.to_numeric(raw.get("GW", raw.get("round")), errors="coerce")
        raw["mins"] = pd.to_numeric(raw.get("minutes"), errors="coerce").fillna(0.0)
        for c in NATIVE_COLS:
            g = empty_native_gws(raw, c)
            if g:
                found = True
                print(f"  {s}  {c:18s} {len(g):2d} gameweeks  GW{min(g)}-{max(g)}")
    if not found:
        print("  none")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    elif "--audit" in sys.argv:
        import warnings; warnings.filterwarnings("ignore")
        audit()
    else:
        print(__doc__)
