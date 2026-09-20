from __future__ import annotations
import os as _os, sys as _sys, glob as _glob
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
defcon_source_audit.py — is the repo's DefCon count the one FPL actually pays on?
================================================================================
The evidence behind `src/defcon_series.py`. Three series for every 25/26 player-match,
each compared with FPL's OFFICIAL `defensive_contribution` (vaastav merged_gw.csv):

  published   FPL-Core-Insights `defensive_contributions`, as shipped
  derived     `defcon_series.fpl_defcon`: CBIT from components for DEF, published for
              MID/FWD — what the pipeline uses since 2026-09-10
  cbirt       the components plus recoveries — the MID/FWD rule, to name the defect

Join: player_code (vaastav element -> code through that season's players_raw.csv) and
gameweek, restricted to player-gameweeks with ONE fixture so a per-match row and a
per-gameweek official value describe the same match. Rows with a null published value
are excluded from the equality rates (a null is not a wrong number; it is handled by
`defcon_series.exposure`), and reported separately.

Also checks FPL-Core-Insights `player_gameweek_stats.csv`, the per-GAMEWEEK file and the
only DefCon source populated in 26/27 so far: same defect in its pre-summed column, but
its components equal FPL's to the unit.

Run:  python studies/defcon_source_audit.py        (needs FPL_DATA and FPL_HISTORY)
Out:  studies/defcon_source_audit.csv              one row per gameweek x position
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import defcon_series as dcs

SEASON, HIST_SEASON = "2025-2026", "2025-26"
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "defcon_source_audit.csv")
POS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD"}


def _repo_frames(fname):
    out = []
    for g in sorted(_glob.glob(_os.path.join(config.repo(SEASON), "By Gameweek", "GW*"))):
        f = _os.path.join(g, fname)
        if not _os.path.exists(f):
            continue
        d = pd.read_csv(f)
        pl = pd.read_csv(_os.path.join(g, "players.csv"))[["player_id", "player_code", "position"]]
        key = "player_id" if "player_id" in d.columns else "id"
        d = d.drop(columns=[c for c in ("position", "player_code") if c in d.columns and key != c])
        d = d.merge(pl, left_on=key, right_on="player_id", how="left", suffixes=("", "_pl"))
        d["gw"] = int(_os.path.basename(g)[2:])
        out.append(d)
    return pd.concat(out, ignore_index=True)


def official():
    """FPL's own per-fixture values, keyed player_code x GW, single-fixture weeks only."""
    root = config.history(HIST_SEASON)
    raw = pd.read_csv(_os.path.join(root, "players_raw.csv"))[["id", "code"]]
    m = pd.read_csv(_os.path.join(root, "gws", "merged_gw.csv"))
    m = m.merge(raw, left_on="element", right_on="id", how="inner")
    m["n_fix"] = m.groupby(["code", "GW"])["fixture"].transform("count")
    m = m[m["n_fix"] == 1]
    return m.rename(columns={"code": "player_code", "GW": "gw",
                             "defensive_contribution": "official",
                             "clearances_blocks_interceptions": "o_cbi",
                             "tackles": "o_tackles", "recoveries": "o_rec"})[
        ["player_code", "gw", "official", "o_cbi", "o_tackles", "o_rec"]]


def main():
    off = official()
    d = _repo_frames("playermatchstats.csv")
    d = d[d["match_id"].astype(str).str.contains("-prem-", na=False)]
    d["mins"] = pd.to_numeric(d["minutes_played"], errors="coerce")
    d = d[d["mins"] > 0].copy()
    d["pos"] = d["position"].map(POS)
    d["published"] = pd.to_numeric(d["defensive_contributions"], errors="coerce")
    d["derived"] = dcs.fpl_defcon(d, d["pos"], SEASON)
    d["cbirt"] = dcs.cbit(d) + pd.to_numeric(d["recoveries"], errors="coerce")
    j = d.merge(off, on=["player_code", "gw"], how="inner")
    print(f"[audit] {len(d)} 25/26 league appearances; {len(j)} joined to FPL official "
          f"(single-fixture weeks)")
    nul = j["published"].isna()
    print(f"        published null on {nul.mean():.1%} of joined rows "
          f"({j.loc[nul, 'mins'].sum() / j['mins'].sum():.1%} of minutes) — excluded below")
    j = j[~nul].copy()

    rows = []
    for (gw, pos), g in j.groupby(["gw", "pos"]):
        rows.append({"gw": gw, "pos": pos, "n": len(g),
                     "published_mismatch": float((g["published"] != g["official"]).mean()),
                     "derived_mismatch": float((g["derived"] != g["official"]).mean()),
                     "published_eq_cbirt": float((g["published"] == g["cbirt"]).mean()),
                     "derived_minus_official": float((g["derived"] - g["official"]).mean())})
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)

    print("\n1. MISMATCH vs FPL OFFICIAL, by gameweek block and position")
    blk = np.select([R["gw"] == 1, R["gw"].between(2, 10)], ["GW1", "GW2-10"], "GW11-38")
    t = R.assign(blk=blk).groupby(["pos", "blk"]).apply(lambda g: pd.Series({
        "n": g["n"].sum(),
        "published": np.average(g["published_mismatch"], weights=g["n"]),
        "derived": np.average(g["derived_mismatch"], weights=g["n"])}))
    print(t.round(3).to_string())
    gw = R[R["pos"] == "DEF"].set_index("gw")["published_mismatch"]
    print("\n   DEF published mismatch per GW:", {int(k): round(v, 2) for k, v in gw.items()})

    D = j[j["pos"] == "DEF"]
    bad = D[D["gw"].between(2, 10) & (D["published"] != D["official"])]
    print(f"\n2. THE DEFECT: DEF mismatches in GW2-10 equal to FPL's own CBIRT "
          f"(official + official recoveries): "
          f"{(bad['published'] == bad['official'] + bad['o_rec']).mean():.1%}; "
          f"to CBIRT from the repo's components: {(bad['published'] == bad['cbirt']).mean():.1%}")
    print(f"   FPL official DEF == its own CBI + tackles: "
          f"{(D['official'] == D['o_cbi'] + D['o_tackles']).mean():.1%}")
    print(f"   derived == official exactly: {(D['derived'] == D['official']).mean():.1%};  "
          f"mean signed error {(D['derived'] - D['official']).mean():+.3f} per match")
    MF = j[j["pos"].isin(["MID", "FWD"])]
    print(f"   MID/FWD published == official: {(MF['published'] == MF['official']).mean():.1%}")

    print("\n3. WHAT IT DID — defenders")
    s = D[D["mins"] >= 60]
    blk2 = np.where(s["gw"].between(2, 10), "GW2-10", "other")
    for lab, m in (("all GWs", np.ones(len(s), bool)), ("GW2-10", blk2 == "GW2-10")):
        x = s[m]
        print(f"   P(hit 10 | 60+ min), {lab:7s}: published {(x['published'] >= 10).mean():.3f}"
              f"  derived {(x['derived'] >= 10).mean():.3f}  official {(x['official'] >= 10).mean():.3f}"
              f"   n={len(x)}")
    pp = D.groupby("player_code").agg(mins=("mins", "sum"), pub=("published", "sum"),
                                       der=("derived", "sum"), off=("official", "sum"))
    pp = pp[pp["mins"] >= 900]
    n90 = pp["mins"].sum() / 90.0
    infl = pp["pub"] / pp["off"] - 1
    print(f"   per 90, defenders with 900+ min (n={len(pp)}): published {pp['pub'].sum() / n90:.2f}"
          f"  derived {pp['der'].sum() / n90:.2f}  official {pp['off'].sum() / n90:.2f}")
    print(f"   per-player inflation of published over official: median {infl.median():+.1%}, "
          f"p90 {infl.quantile(0.9):+.1%}, max {infl.max():+.1%}")

    print("\n4. THE PER-GAMEWEEK FILE (player_gameweek_stats) — the 26/27 in-season source")
    g = _repo_frames("player_gameweek_stats.csv")
    g = g[pd.to_numeric(g["minutes"], errors="coerce") > 0]
    g["pos"] = g["position"].map(POS)
    G = g[g["pos"] == "DEF"].copy()
    G["derived"] = dcs.fpl_defcon(G, G["pos"], SEASON)
    agg = off.groupby(["player_code", "gw"], as_index=False)["official"].sum()
    G = G.merge(agg, on=["player_code", "gw"], how="inner")
    inb = G["gw"].between(2, 10)
    print(f"   DEF published == official: GW2-10 {(G.loc[inb, 'defensive_contribution'] == G.loc[inb, 'official']).mean():.1%}"
          f", other GWs {(G.loc[~inb, 'defensive_contribution'] == G.loc[~inb, 'official']).mean():.1%}")
    print(f"   DEF derived (cbi + tackles) == official: {(G['derived'] == G['official']).mean():.1%}"
          f"  n={len(G)}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
