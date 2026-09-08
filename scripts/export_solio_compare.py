import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
export_solio_compare.py — every Solio section, not just the projections.
=========================================================================
The board consumes ONE of Solio's nine published tables (top-projected players, blended
into GW1). The other eight are pulled, parsed and then discarded, which throws away the
only independent benchmark this project has for several layers it has never externally
checked:

  6  Best clean sheet odds        team Proj. G Against and CS%  -> our lam_against
  9  Best attacking fixtures      team Proj. G For              -> our lam_for
  8  Highest projected DefCon     DefCon% and Proj. P (DefCon)  -> our defcon_ev
  4  Highest projected goals      prG and Proj. P (Goals)       -> our goal channel
  5  Highest projected assists    prA and Proj. P (Assists)     -> our assist channel
  7  Highest projected bonus      Proj. P (Bonus)               -> our bonus channel
  2  Best captain picks           captain projection            -> our ep_cap
  3  Highest-leverage differentials

WHY THE COMPONENT TABLES MATTER MORE THAN THE TOTALS
-----------------------------------------------------
Two models agreeing on a player's total says little about whether they agree for the
same reasons. Sections 4, 5, 7 and 8 decompose the projection into goals, assists, bonus
and DefCon, so a disagreement can be attributed to a CHANNEL rather than left as an
unexplained gap. `PROJECT_KNOWLEDGE §5` records the team clean-sheet engine as validated
against Solio at GA r=0.89 / CS r=0.93 — sections 6 and 9 are what that check was run
on, and re-running it every fetch turns a one-off validation into a monitor.

The DefCon comparison is the genuinely new one: `defcon_env` has never been benchmarked
against anything external.

Run:  python scripts/export_solio_compare.py
Out:  outputs/solio_sections.xlsx  +  outputs/solio_team_compare.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import solio_ensemble as se

TEAM_NORM = {
    "MUN": "Man United", "MCI": "Man City", "ARS": "Arsenal", "LIV": "Liverpool",
    "CHE": "Chelsea", "TOT": "Tottenham", "NEW": "Newcastle", "AVL": "Aston Villa",
    "BOU": "Bournemouth", "BRE": "Brentford", "BHA": "Brighton", "CRY": "Crystal Palace",
    "EVE": "Everton", "FUL": "Fulham", "LEE": "Leeds", "NFO": "Nott'm Forest",
    "SUN": "Sunderland", "COV": "Coventry", "HUL": "Hull", "IPS": "Ipswich",
    "Man Utd": "Man United", "Spurs": "Tottenham", "Nott'm Forest": "Nott'm Forest",
}
def _team(x): return TEAM_NORM.get(str(x).strip(), str(x).strip())


def _num(s):
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace("[£m%]", "", regex=True).str.strip(),
        errors="coerce")


def main():
    md = se.fetch_solio()
    sol = se.parse_solio(md)
    secs = sol.get("sections", {})
    gw = sol.get("gameweek")
    print(f"[solio] gameweek {gw}; {len(secs)} sections parsed")
    for k, v in secs.items():
        print(f"    {k[:58]:58s} {len(v):3d} rows")

    OUT = config.OUTPUTS
    # ---------- team-level: our lambdas vs theirs ----------
    tf = pd.read_csv(_os.path.join(OUT, "team_projections_gw1_10.csv"))
    ours = tf[tf.gw == 1][["team", "lam_for", "lam_against", "p_clean_sheet"]].copy()

    def find(*keys):
        for name, tab in secs.items():
            low = name.lower()
            if all(k in low for k in keys):
                return tab.copy()
        return None

    cs = find("clean", "sheet")
    at = find("attacking", "fixtures")
    rows = ours.copy()
    if cs is not None and len(cs):
        c = cs.rename(columns={cs.columns[1]: "Team"})
        col_ga = [x for x in cs.columns if "against" in str(x).lower()]
        col_cs = [x for x in cs.columns if "cs" in str(x).lower()]
        c["team"] = c["Team"].map(_team)
        if col_ga: c["solio_ga"] = _num(c[col_ga[0]])
        if col_cs: c["solio_cs"] = _num(c[col_cs[0]]) / 100.0
        rows = rows.merge(c[[x for x in ["team", "solio_ga", "solio_cs"]
                             if x in c.columns]], on="team", how="left")
    if at is not None and len(at):
        a = at.rename(columns={at.columns[1]: "Team"})
        col_gf = [x for x in at.columns if "g for" in str(x).lower()]
        a["team"] = a["Team"].map(_team)
        if col_gf:
            a["solio_gf"] = _num(a[col_gf[0]])
            rows = rows.merge(a[["team", "solio_gf"]], on="team", how="left")

    print("\n" + "=" * 92)
    print("TEAM LAYER vs SOLIO — GW1")
    print("=" * 92)
    for ourcol, theircol, label in [("lam_for", "solio_gf", "goals for"),
                                    ("lam_against", "solio_ga", "goals against"),
                                    ("p_clean_sheet", "solio_cs", "clean sheet %")]:
        if theircol in rows.columns:
            d = rows[[ourcol, theircol]].dropna()
            if len(d) >= 3:
                r = d[ourcol].corr(d[theircol])
                mae = (d[ourcol] - d[theircol]).abs().mean()
                bias = (d[ourcol] - d[theircol]).mean()
                print(f"  {label:16s} n={len(d):2d}  r={r:.3f}  MAE={mae:.3f}  "
                      f"bias(ours-theirs)={bias:+.3f}")
    rows.round(3).to_csv(_os.path.join(OUT, "solio_team_compare.csv"), index=False)
    if "solio_ga" in rows.columns:
        rows["ga_gap"] = rows["lam_against"] - rows["solio_ga"]
        print("\n  largest goals-against disagreements:")
        big = rows.reindex(rows["ga_gap"].abs().sort_values(ascending=False).index)
        print(big.head(6)[["team", "lam_against", "solio_ga", "ga_gap"]]
              .round(3).to_string(index=False))

    # ---------- player component channels ----------
    det = pd.read_csv(_os.path.join(OUT, "projection_detail_gw1_10.csv"))
    d1 = det[det.gw == 1][["player", "team", "pos", "mean", "defcon_ev", "cs_ev"]].copy()
    dc = find("defcon")
    comp = None
    if dc is not None and len(dc):
        x = dc.copy()
        pcol = [c for c in x.columns if str(c).strip().lower() == "player"]
        dcol = [c for c in x.columns if "defcon" in str(c).lower()
                and "proj" in str(c).lower()]
        if pcol and dcol:
            x["player"] = x[pcol[0]].astype(str).str.strip()
            x["solio_defcon"] = _num(x[dcol[0]])
            comp = d1.merge(x[["player", "solio_defcon"]], on="player", how="inner")
            d = comp[["defcon_ev", "solio_defcon"]].dropna()
            if len(d) >= 3:
                print(f"\n  DEFCON channel  n={len(d)}  "
                      f"r={d['defcon_ev'].corr(d['solio_defcon']):.3f}  "
                      f"MAE={(d['defcon_ev']-d['solio_defcon']).abs().mean():.3f}  "
                      f"bias={(d['defcon_ev']-d['solio_defcon']).mean():+.3f}")
                print("  (first external benchmark of the DefCon layer)")

    # ---------- write every section ----------
    xl = _os.path.join(OUT, "solio_sections.xlsx")
    with pd.ExcelWriter(xl, engine="xlsxwriter") as w:
        rows.round(4).to_excel(w, sheet_name="Team compare", index=False)
        if comp is not None:
            comp.round(3).to_excel(w, sheet_name="DefCon compare", index=False)
        for name, tab in secs.items():
            sh = "".join(ch for ch in str(name) if ch.isalnum() or ch == " ")[:28] or "s"
            tab.to_excel(w, sheet_name=sh, index=False)
    print(f"\n[solio] wrote {xl}")
    print(f"[solio] wrote {_os.path.join(OUT, 'solio_team_compare.csv')}")


if __name__ == "__main__":
    main()
