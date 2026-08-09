from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
setpiece_study.py — run spec §2.1/§2.2 on real Understat data.
==============================================================
Measures how reliably each shot component is observed, then how persistent it is
across seasons once the attenuation from that measurement error is undone. Applies
the pre-committed decision rule from `setpiece.py` and writes the two deliverable
CSVs.

Two deliberate choices, both recorded in docs/SOCCERDATA_FINDINGS.md:

1. **Keyed on the Understat player id, not `player_code`.** G5 governs anything that
   feeds the FPL model; this study feeds a decision, not the model. Going through the
   crosswalk would inject match error into a persistence estimate for no benefit —
   and the crosswalk's fuzzy rows are unverified by construction. When §2.3 ships a
   prior, THAT must join on `player_code`.

2. **Understat xG is used raw, uncalibrated.** Both quantities here are invariant to
   the affine calibration map: split-half reliability is a correlation, and an AR(1)
   on log rates absorbs a scale factor into the intercept (log(b*x) = log b + log x),
   leaving the slope unchanged. Calibration binds on anything that feeds LEVELS into
   the player layer — §2.3's prior and §2.4's penalty xG — not on this.

Run:  python studies/setpiece_study.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest, setpiece as sp

SEASONS = ["2425", "2526"]
OUT_REL = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        "setpiece_reliability.csv")
OUT_AR1 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        "component_ar1.csv")


def load(drop_gk=True):
    """Shots + a minutes panel, both keyed on the Understat player id.

    Goalkeepers are excluded. This is a ROLE exclusion, not an outcome one: a keeper's
    open-play xG is a structural zero, and 48 of them sat in the >=900-minute panel with
    identical floored values in both seasons. Left in, they contribute ~7% of pairs with
    perfect persistence and bias the AR(1) upward — an artefact of squad composition, not
    a fact about shooting. Outfield players with genuine zeros are KEPT (dropping those
    would be selection on the outcome).
    """
    shots = sd_ingest.understat_shots(SEASONS)
    pm = sd_ingest.understat_player_match(SEASONS)
    shots = shots.rename(columns={"understat_player_id": "player_code"})
    pm = pm.rename(columns={"understat_player_id": "player_code"})
    if drop_gk:
        gk = set(pm.loc[pm["position"] == "GK", "player_code"].unique())
        n0 = pm["player_code"].nunique()
        pm = pm[~pm["player_code"].isin(gk)]
        shots = shots[~shots["player_code"].isin(gk)]
        print(f"[study] dropped {len(gk)} goalkeepers of {n0} players (role exclusion)")
    minutes = pm[["player_code", "season", "match_id", "minutes"]]
    return shots, minutes


def main():
    shots, minutes = load()
    print(f"[study] {len(shots)} shots, {len(minutes)} player-matches, "
          f"seasons {SEASONS}")

    # Two DIFFERENT partitions of the same shots: the four-way split for reporting,
    # the two-way split for the decision. They must be run separately — merging the
    # dicts would let `set_piece_all` swallow the `from_corner` shots, leaving that
    # component empty and its statistics degenerate.
    print("\n=== §2.1 split-half reliability (matches split, Spearman-Brown) ===")
    rel = pd.concat([sp.split_half_reliability(shots, minutes, n_splits=200,
                                               components=part)
                     for part in (sp.COMPONENTS, sp.TWO_COMPONENT)],
                    ignore_index=True)
    rel = rel.drop_duplicates(["season", "component"]).sort_values(["component", "season"])
    print(rel.to_string(index=False))
    rel.to_csv(OUT_REL, index=False)
    print(f"-> {OUT_REL}")

    # one reliability per component, pooled across seasons for the AR(1) correction
    rel_mean = rel.groupby("component")["reliability"].mean().to_dict()
    rel_se = (rel.groupby("component")["reliability"]
              .agg(lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0).to_dict())

    print("\n=== §2.2 persistence, measured and corrected ===")
    panels = {name: sp.component_panel(shots, minutes, components=part)
              for name, part in (("four_way", sp.COMPONENTS),
                                 ("two_way", sp.TWO_COMPONENT))}
    parts = []
    for panel in panels.values():
        a = sp.component_ar1(panel, rel_mean, reliability_se=rel_se)
        if not a.empty:
            parts.append(sp.corrected_ar1(a, panel, n_sims=120))
    if not parts:
        print("no component had enough season pairs — nothing to decide")
        return
    ar1 = pd.concat(parts, ignore_index=True).drop_duplicates("component")
    panel = panels["two_way"]
    cols = ["component", "n_pairs", "rho_measured", "se_measured", "reliability",
            "rho_disattenuated", "rho_corrected", "ci_lo_sim", "ci_hi_sim",
            "sim_saturated"]
    impossible = ar1[ar1["rho_disattenuated"] >= 1.0]["component"].tolist()
    if impossible:
        print(f"!! ratio correction returned rho >= 1 for {impossible} — impossible for a "
              f"stationary process, and a direct demonstration of its upward bias")
    print(ar1[cols].round(4).to_string(index=False))
    ar1.to_csv(OUT_AR1, index=False)
    print(f"-> {OUT_AR1}")

    print("\n=== pre-committed decision rule ===")
    # median nineties among the players the AR(1) is actually fitted on (>=900 min in
    # both seasons), not the whole roster — w0 is a function of exposure, so it has to
    # be evaluated at the exposure of the estimation sample.
    est = panel[panel["minutes"] >= sp.MIN_MINUTES_SEASON]
    n90 = float(np.median(est.groupby(["player_code", "season"])["minutes"].first() / 90.0))
    w0 = sp.implied_pooled_persistence(n90)
    print(f"median player-season nineties = {n90:.1f} -> pooled prior implies "
          f"w0 = {w0:.3f}")
    for comp in ("set_piece_all", "open_play"):
        d = sp.decide(ar1, n90_median=n90, component=comp)
        if "ci" not in d:
            print(f"  {comp}: {d['verdict'].upper()}")
            print(f"      {d['note']}")
            continue
        print(f"  {comp}: rho={d['rho']:.3f} CI=({d['ci'][0]:.3f}, {d['ci'][1]:.3f}) "
              f"vs w0={d['w0_pooled']:.3f}  ->  {d['verdict'].upper()}")
        print(f"      {d['note']}")


if __name__ == "__main__":
    main()
