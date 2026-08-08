import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
build_all.py — regenerate every reconstructed input and prior, in order.
Run once after cloning, before the analysis scripts. Assumes the FPL-Core-Insights
repo is extracted and REPO (below / in each script) points at its data dir, and
that the core modules (bayes_model, core_insights, roster, signals, build_pms,
multiseason_priors, starter_prior, ...) are importable on sys.path.

Produces:
  /tmp/pms_panel.pkl        25/26 per-match panel      (build_pms)
  /tmp/ms_priors.pkl        two-season pooled priors   (multiseason_priors, older_weight=0.5)
  /tmp/own_start_cal.pkl    ownership->start calibration (starter_prior)
  E0_recon.csv              team-model input           (reconstruct_e0)
  coldstart_hist.csv        cold-start calibration input (reconstruct_coldstart)
"""
import warnings; warnings.filterwarnings("ignore")
import sys
# sys.path.insert(0, "/path/to/core/modules")   # <- point at your module dir

import build_pms
print("[1/5] building 25/26 per-match panel ...")
build_pms.build().to_pickle(config.PMS_PANEL)

import multiseason_priors as ms
print("[2/5] building two-season priors (older_weight=0.5) ...")
ms.to_priors(ms.two_season_evidence(older_weight=0.5)).to_pickle(config.MS_PRIORS)

import starter_prior as sp
print("[3/5] calibrating ownership->start ...")
sp.calibrate_ownership_start()

print("[4/5] reconstructing E0 from repo Opta xG ...")
import reconstruct_e0; reconstruct_e0.build()

print("[5/5] reconstructing cold-start calibration input ...")
import reconstruct_coldstart; reconstruct_coldstart.build()

print(f"done. reconstructed inputs in {config.DATA}; pickles in {config.SCRATCH}.")
