from __future__ import annotations
"""
defcon_series.py — the ONE place a DefCon count is read off the feed.
======================================================================
FPL scores DefCon on two different sums:

    DEF       CBIT   clearances + blocks + interceptions + tackles        threshold 10
    MID/FWD   CBIRT  the same, PLUS recoveries                          threshold 12

THE UPSTREAM DEFECT THIS EXISTS FOR  [VERIFIED 2026-09-10]
-----------------------------------------------------------
FPL-Core-Insights 2025-26 publishes a pre-summed column, `defensive_contributions` in
playermatchstats and `defensive_contribution` in player_gameweek_stats. For DEFENDERS in
GW2-GW10, and only those weeks, both carry CBIRT instead of CBIT: recoveries were added
to a defender's count. Against FPL's official value (vaastav merged_gw.csv, joined on
player_code) the per-match column mismatches 0.81-0.88 of DEF rows in GW2-10 and 0.00 in
every other week, with MID/FWD exact throughout. The per-gameweek file shows the same
pattern (DEF agreement 0.12-0.19 in GW2-10, 1.00 elsewhere).

The COMPONENTS are clean. In player_gameweek_stats, clearances_blocks_interceptions +
tackles equals FPL's official DEF DefCon in 100% of 3,904 rows. In playermatchstats,
clearances + blocks + interceptions + tackles equals it exactly in ~91% of rows, with a
mean signed error of +0.003 per match (Opta revisions, symmetric), and reproduces the
official P(hit 10) for 60+ minute defenders (0.267 vs 0.269). The contaminated column put
that at 0.356 and a defender's per-90 rate ~12% too high. Evidence and the per-gameweek
table: studies/defcon_source_audit.py.

So a DEF count is ALWAYS summed here from the components, and the pre-summed column is
never read for a defender, in any week, from either file. MID/FWD keep the upstream
column, which is exact for them. GK cannot score DefCon (bayes_model has no GK
threshold) and keeps the upstream value, which is carried but never paid.

WHY COMPONENTS AND NOT FPL'S OFFICIAL COLUMN FROM VAASTAV
----------------------------------------------------------
Both are correct in expectation. Components win on three counts that matter here:
  * SAME FEED, SAME GRAIN. The prior panel is per MATCH (it needs opponent and venue);
    vaastav is per fixture id, so using it means an element->code map per season plus a
    fixture->match_id crosswalk, two new joins that can fail quietly, for a gain the
    audit puts at zero bias.
  * ONE RULE FOR PRIOR AND IN-SEASON. In 26/27 the pre-summed column AND the per-match
    components are both 100% null through GW3; the only populated DefCon source is
    player_gameweek_stats, whose components are FPL-exact. A rule of "sum the
    components" applies to both files unchanged, so a future in-season update cannot
    drift from the prior it updates.
  * NO NEW DEPENDENCY. The prior build does not need FPL_HISTORY.

NULLS STAY NULL
---------------
A missing component makes the DEF count NaN, never 0, and a missing upstream value stays
NaN for MID/FWD. 2.4% of 25/26 league minutes (rising to 16% of rows in GW29) carry every
other stat but no defensive block; filling those with 0 counts real minutes as exposure
with no actions and dilutes the rate. Use `exposure()` for the matching denominator.

SEASONS
-------
DefCon did not exist before 2025-26 and FPL never scored it. The components DO exist in
24/25, so summing them there would silently manufacture a season of DEF DefCon evidence
that no prior, exposure or validation in this repo accounts for. `fpl_defcon` refuses
seasons before FIRST_SEASON rather than let that happen by accident; 24/25 DefCon is
NaN by construction.

Run:  python src/defcon_series.py --selftest
"""
import numpy as np
import pandas as pd

FIRST_SEASON = 2025                                   # 2025-26: DefCon introduced

# Per-match schema (playermatchstats) and per-gameweek schema (player_gameweek_stats).
DEF_PARTS_MATCH = ("clearances", "blocks", "interceptions", "tackles")
DEF_PARTS_GW = ("clearances_blocks_interceptions", "tackles")
RECOVERIES = "recoveries"
UPSTREAM = ("defensive_contributions", "defensive_contribution")

_DEF_LABELS = ("DEF", "Defender", 2)


def _season_start(season):
    """'2025-2026', '2025-26' or 2025 -> 2025."""
    return int(str(season)[:4])


def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce").astype(float)


def def_parts(d):
    """The component columns this frame's schema carries, per-match schema first."""
    for parts in (DEF_PARTS_MATCH, DEF_PARTS_GW):
        if all(c in d.columns for c in parts):
            return parts
    raise KeyError(f"no DefCon component columns: need {DEF_PARTS_MATCH} or {DEF_PARTS_GW}")


def cbit(d):
    """Clearances + blocks + interceptions + tackles. NaN if ANY part is missing."""
    parts = pd.concat([_num(d, c) for c in def_parts(d)], axis=1)
    return parts.sum(axis=1, min_count=parts.shape[1])


def is_defender(pos):
    return pd.Series(pos).isin(_DEF_LABELS).to_numpy()


def fpl_defcon(d, pos, season):
    """DefCon count under FPL's scoring rule, one value per row of `d`.

    `pos` is a per-row position label ('DEF'/'MID'/... or 'Defender'/'Midfielder'/...).
    DEF rows are summed from components; every other row takes the upstream column as
    published. NaN is preserved everywhere.
    """
    if _season_start(season) < FIRST_SEASON:
        raise ValueError(
            f"season {season}: DefCon did not exist before {FIRST_SEASON}-"
            f"{(FIRST_SEASON + 1) % 100:02d}. Summing components there would invent DEF "
            f"evidence the priors do not account for; leave the column NaN instead.")
    up = next((c for c in UPSTREAM if c in d.columns), None)
    out = _num(d, up) if up else pd.Series(np.nan, index=d.index, dtype=float)
    dmask = is_defender(pos)
    if dmask.any():
        out = out.where(~dmask, cbit(d))
    return out


def exposure(mins, dc):
    """Minutes that can carry a DefCon: zero where the count is missing, so a rate
    sum(dc)/sum(exposure) never counts unmeasured minutes as minutes with no actions."""
    m = pd.to_numeric(pd.Series(mins), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.where(pd.Series(dc).notna().to_numpy(), m, 0.0)


def assert_no_recoveries(d, dc, pos):
    """Postcondition on a built series: every finite DEF count equals CBIT exactly, so
    none can contain recoveries. Raises naming how many rows would have been CBIRT."""
    dmask = is_defender(pos)
    if not dmask.any():
        return True
    x = pd.Series(np.asarray(dc, dtype=float), index=d.index)[dmask]
    ref = cbit(d)[dmask]
    fin = x.notna()
    bad = fin & (x != ref)
    if bad.any():
        extra = ""
        if RECOVERIES in d.columns:
            rec = _num(d, RECOVERIES)[dmask]
            extra = f"; {int((bad & (x == ref + rec) & (rec > 0)).sum())} of them equal CBIT+recoveries"
        raise AssertionError(f"{int(bad.sum())} DEF DefCon values are not CBIT{extra}")
    if (x.isna() & ref.notna()).any():
        raise AssertionError("DEF DefCon is NaN where every component is present")
    return True


def selftest():
    rng = np.random.default_rng(0)
    n = 400
    pos = np.array(["DEF", "MID", "FWD", "GK"])[rng.integers(0, 4, n)]
    d = pd.DataFrame({c: rng.integers(0, 6, n).astype(float) for c in DEF_PARTS_MATCH})
    d[RECOVERIES] = rng.integers(1, 9, n).astype(float)
    base = d[list(DEF_PARTS_MATCH)].sum(axis=1)
    # Plant the upstream defect: everyone published as CBIRT. Correct for MID/FWD, the
    # GW2-10 fault for DEF. Recoveries are >= 1, so the two rules differ on every row.
    d["defensive_contributions"] = base + d[RECOVERIES]
    d.loc[pos == "GK", "defensive_contributions"] = 0.0
    dc = fpl_defcon(d, pos, "2025-2026")

    D = pos == "DEF"
    # 1. DEF is CBIT — never includes recoveries.
    assert (dc[D] == base[D]).all(), "DEF DefCon must equal CBIT"
    assert (dc[D] != d.loc[D, "defensive_contributions"]).all(), \
        "DEF DefCon must not pass the contaminated upstream value through"
    # 2. Recoveries are INVISIBLE to a defender: perturbing them moves no DEF count.
    d2 = d.copy()
    d2[RECOVERIES] = d2[RECOVERIES] + rng.integers(1, 50, n)
    assert (fpl_defcon(d2, pos, "2025-26")[D] == dc[D]).all(), \
        "changing recoveries changed a DEF DefCon count"
    # 3. MID/FWD keep the upstream column exactly (CBIRT is their rule); GK untouched.
    MF = np.isin(pos, ["MID", "FWD"])
    assert (dc[MF] == d.loc[MF, "defensive_contributions"]).all()
    assert (dc[pos == "GK"] == 0.0).all()
    # 4. The postcondition passes on a correct series and FIRES on the defect.
    assert assert_no_recoveries(d, dc, pos)
    try:
        assert_no_recoveries(d, d["defensive_contributions"], pos)
        raise RuntimeError("postcondition missed a CBIRT defender series")
    except AssertionError as e:
        assert "CBIT+recoveries" in str(e), e
    # 5. Nulls stay null: a missing component is NaN for DEF, never 0; upstream NaN
    #    stays NaN for MID; exposure drops exactly those minutes.
    d3 = d.copy()
    i_def = int(np.flatnonzero(D)[0]); i_mid = int(np.flatnonzero(pos == "MID")[0])
    d3.loc[i_def, "tackles"] = np.nan
    d3.loc[i_mid, "defensive_contributions"] = np.nan
    dc3 = fpl_defcon(d3, pos, "2025-2026")
    assert np.isnan(dc3[i_def]) and np.isnan(dc3[i_mid]), "a null must not become 0"
    ex = exposure(np.full(n, 90.0), dc3)
    assert ex[i_def] == 0.0 and ex[i_mid] == 0.0 and ex.sum() == 90.0 * (n - 2)
    assert assert_no_recoveries(d3, dc3, pos)
    # 6. The per-gameweek schema (player_gameweek_stats) takes the same rule.
    g = pd.DataFrame({"clearances_blocks_interceptions": [7.0, 3.0, 4.0],
                      "tackles": [4.0, 1.0, 2.0], RECOVERIES: [6.0, 5.0, 9.0],
                      "defensive_contribution": [17.0, 9.0, 15.0]})
    gdc = fpl_defcon(g, ["Defender", "Midfielder", "Defender"], 2026)
    assert list(gdc) == [11.0, 9.0, 6.0], list(gdc)
    # 7. Pre-DefCon seasons are refused, not silently summed.
    try:
        fpl_defcon(d, pos, "2024-2025")
        raise RuntimeError("24/25 must be refused")
    except ValueError:
        pass
    print("SELFTEST OK: DEF DefCon is CBIT from components and never includes recoveries "
          "(perturbing recoveries moves no DEF count; the postcondition fires on a CBIRT "
          "series); MID/FWD keep the upstream value; nulls stay null and leave exposure; "
          "per-gameweek schema handled; pre-25/26 seasons refused.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    print(__doc__)
