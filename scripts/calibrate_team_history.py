import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
calibrate_team_history.py — estimate TeamModel's hyperparameters from 31 seasons
=================================================================================
Writes data/team_hyperparams.json, which `bayes_model` loads at import. Run this only
when you want to REFRESH the calibration (it downloads football-data.co.uk and refits
every season); the committed JSON is what the model actually reads, so the normal
pipeline needs no network and no re-fitting.

Replaces the values `bayes_model` previously hard-coded as, in its own words, "educated
guesses". Validated first on simulated data with known parameters
(studies/test_history.py), where the IV estimator recovers `revert` to within 0.022 while
naive OLS is off by 0.158.

Run:  python scripts/calibrate_team_history.py
      python scripts/calibrate_team_history.py --dry-run    (print, do not write)
"""
import warnings; warnings.filterwarnings("ignore")
import json, datetime
import history as H


def build(write=True):
    res, rat, ha, prom = H.calibrate_all(start=1993, end=2025, verbose=False)
    kw = H.to_model_kwargs(res)
    out = {
        "_provenance": {
            "generated": datetime.date.today().isoformat(),
            "source": "football-data.co.uk E0, seasons 1993/94-2025/26",
            "n_seasons": int(rat.season.nunique()),
            "n_reversion_pairs": int(res["n_pairs"]),
            "n_promoted_team_seasons": int(res["n_promoted"]),
            "estimator": "IV-corrected reversion (errors-in-variables); see history.py",
            "validated_by": "studies/test_history.py",
        },
        "home_prior": list(kw["home_prior"]),
        "promoted_att": list(kw["promoted_att"]),
        "promoted_def": list(kw["promoted_def"]),
        "revert": float(kw["revert"]),
        "season_sd": float(kw["season_sd"]),
        # Home advantage is suppressed in the opening matchdays. Measured on 12
        # Understat seasons (studies/early_season_goals.py): within-season log home
        # advantage is ~0.15 lower over matchdays 1-3 than the rest of the season,
        # 10/12 seasons down, robust to the baseline chosen (-0.152 vs md4+, -0.151 vs
        # md7+, -0.163 vs md20+). A SINGLE step, deliberately: matchdays 4-6 show no
        # significant discount (-0.071, CI -0.155..+0.023) and there is no monotone
        # trend across the season (slope CI spans zero), so a multi-step schedule would
        # be fitting noise. Not produced by calibrate_all — hardcoded here from the
        # study so it travels with the other hyperparameters.
        "home_early_discount": 0.152,
        "home_early_last_gw": 3,
        "home_adv_trend_per_season": float(res["home_adv_trend_per_season"]),
        "revert_att": float(res["revert_att"]),
        "revert_def": float(res["revert_def"]),
        "revert_ols_attenuated": float(res["revert_ols"]),
    }
    if write:
        with open(config.TEAM_HYPERPARAMS, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {config.TEAM_HYPERPARAMS}")
    return out


if __name__ == "__main__":
    o = build(write="--dry-run" not in _sys.argv)
    print("\n=== calibrated ===")
    for k, v in o.items():
        if k != "_provenance":
            print(f"  {k:26s} {v}")
    print("\n=== previous hard-coded guesses ===")
    print("  home_prior                 (0.26, 0.08)")
    print("  promoted_att               (-0.20, 0.30)")
    print("  promoted_def               (-0.22, 0.30)")
    print("  revert                     0.85")
    print("  season_sd                  0.15")
