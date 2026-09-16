import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
export_solio_compare.py — every Solio section, not just the projections.
=========================================================================
The board consumes ONE of Solio's nine published tables (top-projected players, blended
into the board's gameweek). The other eight are pulled, parsed and then discarded, which
throws away the only independent benchmark this project has for several layers it has
never externally checked:

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

The per-player channels are the part of this script with no other home; the DefCon one
is the genuinely new comparison, since `defcon_env` has never been benchmarked against
anything external. The team layer is now covered twice over — see below.

TWO TEAM-LAYER COMPARISONS, NOT ONE
-----------------------------------
Since 2026-09-11 `fixture_market.attach` joins the market's own per-fixture lambda onto
`team_projections_gw1_38.csv` as `mkt_lam_for` / `mkt_lam_against` / `mkt_p_clean_sheet`.
Those columns do NOT replace the comparison built here from the published markdown. This
script prints both, because they answer different questions:

  vs PUBLISHED  Solio's sections 6 and 9 as a reader sees them, for the gameweek the
                page is currently showing. This is the instrument `PROJECT_KNOWLEDGE §5`
                validated on, so it is the one that keeps that number comparable over
                time. Its weakness is worth naming rather than leaving in the footnotes:
                both tables are TOP-TEN lists SELECTED ON the quantity being compared —
                section 6 is the ten lowest goals-against, section 9 the ten highest
                goals-for — so r is computed over a range-restricted, endogenously
                selected subset and is not the twenty-team correlation §5's headline
                implies. n is printed beside every figure.
  vs STORED     `mkt_*`: every priced club, no selection, per fixture rather than per
                team. But it is the last PRE-DEADLINE snapshot, deliberately frozen at
                the information cut of `predictions/`, so once a deadline passes it and
                the live page describe different moments — and once the page turns over,
                different gameweeks. The better statistic and the worse monitor.

Where both are present, read the stored columns for calibration and the published ones
for continuity with §5. Neither is independent of the model in the way a naive reading
suggests: `market_odds.py` already blends the outright market into the team layer at
MARKET_WEIGHT=0.6, so some agreement is by construction (`fixture_market` §"DISPLAY AND
AUDIT ONLY").

CLEAN SHEETS ARE COMPARED PLUG-IN
---------------------------------
Solio's published CS% is plug-in exp(-lambda) — `fixture_market` §"CLEAN SHEETS ARE
PLUG-IN" records its csProb agreeing with the plug-in to within 0.003 on every stored
snapshot. Our `p_clean_sheet` is posterior-PREDICTIVE, the mean over draws of
exp(-lambda), which Jensen puts strictly above the plug-in. Both comparisons therefore
score against `p_clean_sheet_plugin`. Until 2026-09-12 this script used
`p_clean_sheet`, which read the Jensen gap as a disagreement with Solio.

Run:  python scripts/export_solio_compare.py
Out:  outputs/solio_sections.xlsx  +  outputs/solio_team_compare.csv
"""
import datetime as _dt
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
    # The markdown spells the club out where the board abbreviates it. Without this the
    # club silently drops out of every merge below instead of failing loudly.
    "Leeds United": "Leeds",
}
def _team(x): return TEAM_NORM.get(str(x).strip(), str(x).strip())


def _num(s):
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace("[£m%]", "", regex=True).str.strip(),
        errors="coerce")


# ------------------------------------------------------------------ which export
def covering(prefix, gw, outputs=None):
    """The most recently written `<prefix>gw1_N.csv` whose horizon reaches gameweek `gw`.

    Returns (path, n, mtime) or None.

    Both exports this script reads name themselves after their own GW_HI, which is an
    env knob rather than a constant: `export_team_projections` defaulted to 10 until
    2026-09-10 and defaults to 38 now, while `export_projection_detail` still defaults
    to 10. A hardcoded filename therefore reads a stale leftover or nothing at all —
    which is exactly what happened here. This script read `team_projections_gw1_10.csv`,
    last written 2026-09-07, while `team_projections_gw1_38.csv` was being rebuilt daily.

    `gw_explorer.team_layer` resolves the same family of files by taking the NARROWEST
    export covering the horizon, and this is that pattern with one deliberate change.
    There, the requirement is a board spanning GW_LO..GW_HI, so coverage is a real
    constraint and a wider export is the same run extended rather than a better one.
    Here the requirement is a SINGLE gameweek, which nearly every export satisfies, so
    "narrowest" would have re-selected the September-7 leftover — the very staleness
    being fixed. Coverage is the filter, RECENCY is the discriminator, and narrowness is
    only the tie-break between two files written in the same second.
    """
    outputs = outputs or config.OUTPUTS
    if not _os.path.isdir(outputs):
        return None
    found = []
    for fn in sorted(_os.listdir(outputs)):
        if not (fn.startswith(prefix + "gw1_") and fn.endswith(".csv")):
            continue
        try:
            n = int(fn[len(prefix) + len("gw1_"):-len(".csv")])
        except ValueError:
            continue
        if n >= gw:
            p = _os.path.join(outputs, fn)
            found.append((_os.path.getmtime(p), -n, p, n))
    if not found:
        return None
    mtime, _, path, n = max(found)
    return path, n, _dt.datetime.fromtimestamp(mtime)


def _read_covering(prefix, gw, what, rebuild):
    """`covering`, then read it and keep only gameweek `gw`. Refuses rather than
    degrades: an absent export means the printed comparison would be silently missing a
    side, and a wrong-gameweek one means it would be silently comparing two weeks."""
    hit = covering(prefix, gw)
    if hit is None:
        raise SystemExit(
            f"[solio] no {prefix}gw1_N.csv in {config.OUTPUTS} covers GW{gw}, so the "
            f"{what} cannot be compared.  {rebuild}")
    path, n, mtime = hit
    df = pd.read_csv(path)
    df["gw"] = pd.to_numeric(df["gw"], errors="coerce")
    if gw not in set(df["gw"].dropna().astype(int)):
        raise SystemExit(
            f"[solio] {_os.path.basename(path)} names GW1-{n} but holds no GW{gw} rows; "
            f"rebuild it before comparing.  {rebuild}")
    print(f"[solio] {what}: {_os.path.basename(path)} (GW1-{n}, written "
          f"{mtime:%Y-%m-%d %H:%M})")
    return df[df["gw"] == gw].copy()


# ------------------------------------------------------- per-fixture -> per-team
def _sum(s):
    """Sum across a team's fixtures, or NaN if any term is missing.

    `Series.sum()` skips NaN, which would report half a priced double gameweek as the
    whole of it — a silent understatement rather than an absent number."""
    v = pd.to_numeric(s, errors="coerce")
    return float(v.sum()) if len(v) and v.notna().all() else np.nan


def _any_cs(s):
    """P(at least one clean sheet across the gameweek's fixtures) — the quantity Solio
    section 6 states it publishes. Identical to the single value in a single gameweek."""
    v = pd.to_numeric(s, errors="coerce")
    if not len(v) or not v.notna().all():
        return np.nan
    return float(1.0 - np.prod(1.0 - v.to_numpy(dtype=float)))


_AGG = [("lam_for", _sum), ("lam_against", _sum),
        ("p_clean_sheet_plugin", _any_cs), ("p_clean_sheet", _any_cs),
        ("mkt_lam_for", _sum), ("mkt_lam_against", _sum),
        ("mkt_p_clean_sheet", _any_cs)]


def per_team(tf):
    """Collapse the per-team-FIXTURE export to the per-team-GAMEWEEK shape Solio
    publishes ("across all Gameweek fixtures"). A no-op in a single gameweek; in a
    double, where the export carries two rows for a club and the markdown one, it is the
    difference between comparing a total with a total and a total with a half."""
    out = []
    for team, g in tf.groupby("team", sort=True):
        rec = {"team": team, "fixtures": int(len(g))}
        if {"opponent", "is_home"} <= set(g.columns):
            rec["opponents"] = ", ".join(
                f"{'vs' if int(h) else '@'} {o}"
                for o, h in zip(g["opponent"], g["is_home"]))
        for col, how in _AGG:
            if col in g.columns:
                rec[col] = how(g[col])
        for col in ("mkt_source", "mkt_as_of"):
            if col in g.columns:
                vals = sorted(set(g[col].dropna().astype(str)))
                rec[col] = "+".join(vals) if vals else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def _mirrored(rows, priced, ours_a, theirs_a, ours_b, theirs_b, tol=5e-4):
    """True when the goals-for and goals-against comparisons are the SAME set of
    (ours, theirs) pairs relabelled.

    Every fixture contributes one club's `for` as the other's `against`, so once both
    sides of every fixture are in the frame the two rows below are one statistic printed
    twice — identical r, MAE and bias, by construction rather than by agreement. That is
    the normal case for the stored market columns, where all twenty clubs are priced,
    and it must be said out loud: read as two independent checks it doubles the apparent
    evidence. The published tables are top-ten selections and are not mirrored."""
    a = sorted(zip(rows.loc[priced, ours_a], rows.loc[priced, theirs_a]))
    b = sorted(zip(rows.loc[priced, ours_b], rows.loc[priced, theirs_b]))
    return len(a) == len(b) and len(a) > 0 and all(
        abs(x[0] - y[0]) < tol and abs(x[1] - y[1]) < tol for x, y in zip(a, b))


def _score(rows, pairs, title, notes=()):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    for ln in notes:
        print(f"  {ln}")
    scored = 0
    for ourcol, theircol, label in pairs:
        if ourcol not in rows.columns or theircol not in rows.columns:
            continue
        d = rows[[ourcol, theircol]].dropna()
        if len(d) < 3:
            print(f"  {label:16s} n={len(d):2d}  (too few paired clubs to score)")
            continue
        scored += 1
        r = d[ourcol].corr(d[theircol])
        mae = (d[ourcol] - d[theircol]).abs().mean()
        bias = (d[ourcol] - d[theircol]).mean()
        print(f"  {label:16s} n={len(d):2d}  r={r:.3f}  MAE={mae:.3f}  "
              f"bias(ours-theirs)={bias:+.3f}")
    if not scored:
        print("  nothing paired — the columns compared are absent or empty.")


def main():
    md = se.fetch_solio()
    sol = se.parse_solio(md)
    secs = sol.get("sections", {})
    gw = sol.get("gameweek")
    print(f"[solio] gameweek {gw}; {len(secs)} sections parsed")
    for k, v in secs.items():
        print(f"    {k[:58]:58s} {len(v):3d} rows")
    if gw is None:
        raise SystemExit("[solio] the feed carries no gameweek number, so nothing can be "
                         "aligned to it and no comparison is printed.")

    OUT = config.OUTPUTS
    # ---------- team-level: our lambdas vs theirs ----------
    # Solio's gameweek, not GW1. Until 2026-09-12 this filtered `tf.gw == 1` against
    # whatever gameweek the page happened to be showing, so every r / MAE / bias below
    # was one gameweek of ours scored against another of theirs.
    tf = _read_covering("team_projections_", gw, "team layer",
                        f"GW_HI={max(gw, 38)} python scripts/export_team_projections.py "
                        f"writes it.")
    csc = "p_clean_sheet_plugin"
    if csc not in tf.columns:
        csc = "p_clean_sheet"
        print("  [warn] this export predates p_clean_sheet_plugin, so the clean-sheet "
              "row scores our posterior-predictive column against a plug-in one and "
              "reads the Jensen gap as disagreement.")
    ours = per_team(tf)

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
        col_ga = [x for x in at.columns if "g against" in str(x).lower()]
        a["team"] = a["Team"].map(_team)
        take = ["team"]
        if col_gf:
            a["solio_gf"] = _num(a[col_gf[0]]); take.append("solio_gf")
        # Section 9 republishes goals-against for ITS top ten, which is a different ten
        # from section 6's. Taking both widens the goals-against sample beyond either
        # table's own selection without mixing in a second source: where a club appears
        # in both, the two figures agree exactly.
        if col_ga:
            a["solio_ga9"] = _num(a[col_ga[0]]); take.append("solio_ga9")
        rows = rows.merge(a[take], on="team", how="left")
    if "solio_ga9" in rows.columns:
        if "solio_ga" not in rows.columns:
            rows["solio_ga"] = np.nan
        rows["solio_ga"] = rows["solio_ga"].fillna(rows["solio_ga9"])

    n_teams = len(rows)
    _score(rows,
           [("lam_for", "solio_gf", "goals for"),
            ("lam_against", "solio_ga", "goals against"),
            (csc, "solio_cs", "clean sheet %")],
           f"TEAM LAYER vs SOLIO PUBLISHED — GW{gw}",
           notes=[f"sections 6 and 9 of the live page against {n_teams} clubs of ours; "
                  f"clean sheets scored on {csc}.",
                  "both sections are top-ten tables selected ON the compared quantity, "
                  "so n < 20 and r is range-restricted."])

    if "mkt_lam_for" in rows.columns and rows["mkt_lam_for"].notna().any():
        priced = rows["mkt_lam_for"].notna()
        srcs = sorted(set("+".join(rows.loc[priced, "mkt_source"].dropna().astype(str))
                          .split("+")) - {""})
        asof = rows.loc[priced, "mkt_as_of"].dropna().astype(str)
        notes = [f"mkt_* from fixture_market: {int(priced.sum())} of {n_teams} "
                 f"clubs priced, source {'/'.join(srcs) or 'unknown'}"
                 f"{', as of ' + max(asof) if len(asof) else ''}.",
                 "unselected and per fixture, but frozen at the last pre-deadline "
                 "snapshot, so it need not match the live page above."]
        if _mirrored(rows, priced, "lam_for", "mkt_lam_for",
                     "lam_against", "mkt_lam_against"):
            notes.append("both sides of every priced fixture are present, so the two "
                         "goals rows are the same pairs relabelled: one statistic, not "
                         "two agreeing ones.")
        _score(rows,
               [("lam_for", "mkt_lam_for", "goals for"),
                ("lam_against", "mkt_lam_against", "goals against"),
                (csc, "mkt_p_clean_sheet", "clean sheet %")],
               f"TEAM LAYER vs STORED MARKET — GW{gw}", notes=notes)
    else:
        print("\n  [note] no populated mkt_* columns in this export, so the unselected "
              "model-vs-market comparison is unavailable. `fixture_market.attach` "
              "writes them from GW_HI=38 python scripts/export_team_projections.py")

    rows.round(3).to_csv(_os.path.join(OUT, "solio_team_compare.csv"), index=False)
    if "solio_ga" in rows.columns and rows["solio_ga"].notna().any():
        rows["ga_gap"] = rows["lam_against"] - rows["solio_ga"]
        print("\n  largest goals-against disagreements (vs published):")
        big = rows.reindex(rows["ga_gap"].abs().sort_values(ascending=False).index)
        cols = [c for c in ("team", "opponents", "lam_against", "solio_ga", "ga_gap")
                if c in big.columns]
        print(big.head(6)[cols].round(3).to_string(index=False))

    # ---------- player component channels ----------
    # The same two corrections as the team block: the export is resolved rather than
    # named, and it is filtered to Solio's gameweek rather than to GW1.
    det = _read_covering("projection_detail_", gw, "player channels",
                         f"GW_HI={max(gw, 10)} python "
                         f"scripts/export_projection_detail.py writes it.")
    keep = [c for c in ("player", "team", "pos", "mean", "defcon_ev", "cs_ev")
            if c in det.columns]
    d1 = det[keep].copy()
    dc = find("defcon")
    comp = None
    if dc is not None and len(dc):
        x = dc.copy()
        pcol = [c for c in x.columns if str(c).strip().lower() == "player"]
        tcol = [c for c in x.columns if str(c).strip().lower() == "team"]
        dcol = [c for c in x.columns if "defcon" in str(c).lower()
                and "proj" in str(c).lower()]
        if pcol and dcol:
            x["player"] = x[pcol[0]].astype(str).str.strip()
            x["solio_defcon"] = _num(x[dcol[0]])
            # Solio publishes SURNAMES, and 17 of them are shared by two or three
            # players in the current league. Joining on the name alone silently pairs
            # our man with theirs AND with his namesake: at GW5 their Fofana (CHE, DEF)
            # matched Sunderland's Fofana too, and their Palacios (IPS) matched Fulham's,
            # so a 15-row table produced 17 pairs, two of them a ~1.0 projection scored
            # against a ~0.1 one. Their table carries the club, so key on it. The repo
            # rule is to join on `player_code`; Solio publishes no id, and the club is
            # the strongest disambiguator that both sides actually have.
            keys = ["player"]
            if tcol and "team" in d1.columns:
                x["team"] = x[tcol[0]].map(_team)
                keys.append("team")
            else:
                print("\n  [warn] Solio's DefCon table carries no Team column, so the "
                      "join is on surname alone and namesakes will pair wrongly.")
            comp = d1.merge(x[keys + ["solio_defcon"]], on=keys, how="inner")
            if len(comp) > len(x):
                print(f"\n  [warn] {len(comp)} pairs from {len(x)} Solio rows — the join "
                      f"is still duplicating; treat the figures below as unreliable.")
            d = comp[["defcon_ev", "solio_defcon"]].dropna()
            if len(d) >= 3:
                print(f"\n  DEFCON channel  GW{gw}  n={len(d)}  "
                      f"r={d['defcon_ev'].corr(d['solio_defcon']):.3f}  "
                      f"MAE={(d['defcon_ev']-d['solio_defcon']).abs().mean():.3f}  "
                      f"bias={(d['defcon_ev']-d['solio_defcon']).mean():+.3f}")
                print("  (first external benchmark of the DefCon layer; Solio's table is "
                      "its own top fifteen, so this too is a selected sample)")
            else:
                print(f"\n  DEFCON channel  GW{gw}  n={len(d)} — too few names matched to "
                      f"score, of {len(x)} in Solio's table.")

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
