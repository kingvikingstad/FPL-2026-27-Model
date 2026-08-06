# PATCHES — edits to existing modules

**Status (Aug 5 2026): historical/reference.** `src/final_ms.py`, as copied into this repo from
`fpl-core-engine`, already has all three edits below pre-applied — confirmed by reading the file
directly (loader points at `base=f'{REPO}/2026-2027'`, cold-start uses `coldstart_hist.csv`, the
team model fits on `E0_recon.csv`). This document is kept for reference and for anyone applying
the same integration to a different copy of the runner; no action is needed against the copy
already in this repo.

The new layers are additive, but three one-line edits to the existing runner make the
pipeline use the reconstructed inputs and the new priors. Apply these to your copies rather
than overwriting the files, to avoid clobbering unrelated local changes.

## `final_ms.py` (or your equivalent runner)

1. **Point the 26/27 loader at the repo** (loose CSVs live in the repo, not an uploads dir):
   ```python
   # before
   d26,t26,_ = ci.load(); sig = ci.to_signals(d26)
   # after
   d26,t26,_ = ci.load(base=f'{REPO}/2026-2027'); sig = ci.to_signals(d26)
   ```

2. **Use the reconstructed cold-start input** (defective source CSV not required):
   ```python
   # before
   cal = calibrate_cold_start()
   # after
   cal = calibrate_cold_start(hist_csv='coldstart_hist.csv')
   ```

3. **Use the reconstructed E0** for the team model (original odds-based E0 not required):
   ```python
   # before
   tm = TeamModel(promoted_per_club=pclub).fit(clubelo=elo, clubelo_weight=0.45)
   # after
   tm = TeamModel(promoted_per_club=pclub).fit(e0_path='E0_recon.csv',
                                               clubelo=elo, clubelo_weight=0.45)
   ```

## Adding the new priors to the frame build

After the priors/cold-start rows are assembled into `pl` and before `apply_availability`:
```python
import starter_prior as sp
cal_own = sp.calibrate_ownership_start()
pl = sp.apply_coldstart_depth(pl, cal_own)              # cold-start players only
pl = sp.apply_minutes_shrinkage(pl, cal_own, k_min=900) # thin/role-changed established players
pl = sg.apply_availability(pl, sig, lineups=lu)         # injuries + confirmed XIs override
```
`scripts/decision_v2.py` already wires this end to end; use it as the reference implementation.

## Optional: live feeds

- **Injuries / confirmed XIs:** set `APIFOOTBALL_KEY` (uses the existing `apifootball` module via
  `lineups.from_apifootball` / `merge_into_signals`).
- **Predicted / manual XIs:** set `LINEUPS_PATH` to a CSV (`team,player,role`) or JSON
  (`{team:{start:[...],bench:[...]}}`).
