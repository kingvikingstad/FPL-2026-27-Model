from __future__ import annotations
import config
"""
team_overrides.py — explicit manager judgment on team strength.
================================================================
A deliberate, labelled way to disagree with the model about a club, kept separate from
every fitted layer so it can never be mistaken for a measurement.

WHAT THE EVIDENCE IN THIS REPO ACTUALLY SAYS
---------------------------------------------
Read this before setting a value. The obvious reasons to mark a club down at the start of
a season have all been tested here, and all came back null:

  new_manager_debut      offseason hires vs prior-season strength: no bump and no
                         penalty inside GW1-6. Price them at prior strength.
  late_form_carryover    a weak finish does not predict a weak start once full-season
                         strength is controlled for (+0.0000 r2). A close-season manager
                         change does NOT reset carryover either.
  transfer_churn         heavy squad turnover costs a little across a whole season,
                         clustered CI (-0.113, -0.005), but it is NOT front-loaded
                         (Aug-Sep interaction t=-0.73).
  early_scoring_trends   prior position, last-10 form, offseason manager change and
                         squad turnover: four nulls on match totals, 31 seasons, with a
                         positive control that recovers a known effect.

So "new manager, sold their midfield, poor pre-season, will start slowly" is a
well-tested hypothesis that this project has repeatedly failed to find. That does not
make it wrong — the studies bound the effect at roughly 0.11 goals per match per SD and
cannot rule out something smaller — but it does mean an override here is a JUDGMENT
against the measured evidence, not an application of it.

It is still legitimate. The model is a prior, the manager watches the football, and the
betting snapshot behind team strength is dated 7 August. Overrides are for information
the model does not have. They are simply recorded as such.

UNITS
-----
Shifts are in LOG strength, the same scale `injury_impact` uses and the same scale the
team model samples on. Roughly: -0.05 is a 5% cut in expected goals, -0.10 is 10%. The
cap is deliberately tight; a club is not usually 20% different from what 31 seasons of
reversion say it is.

Set via the TEAM_OVERRIDE env var, e.g.  TEAM_OVERRIDE="Newcastle:-0.08,-0.05"
(club:att_shift,def_shift), or by editing OVERRIDES below.
"""
import os
import numpy as np
import pandas as pd

MAX_SHIFT = 0.15

# club -> (attack shift, defence shift) in log units. Negative = weaker.
OVERRIDES = {
    # 2026-08-20 [JUDGMENT, manager's call]. Ninth regime club: Jaissle appointed 5 Aug,
    # heavy midfield sales, poor pre-season. The repo's own studies find no early-season
    # penalty for any of those individually (see above), so this is a stated disagreement
    # with the model, applied openly rather than by quietly editing a prior.
    "Newcastle": (-0.08, -0.05),
}


def parse_env(val=None):
    """TEAM_OVERRIDE="Newcastle:-0.08,-0.05;Everton:-0.03,0" -> dict."""
    raw = val if val is not None else os.environ.get("TEAM_OVERRIDE", "")
    out = {}
    for part in str(raw).split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        club, nums = part.split(":", 1)
        try:
            bits = [float(x) for x in nums.split(",")]
        except ValueError:
            continue
        if bits:
            out[club.strip()] = (bits[0], bits[1] if len(bits) > 1 else 0.0)
    return out


def active(extra=None):
    """Committed overrides merged with anything supplied by env or argument."""
    d = dict(OVERRIDES)
    d.update(parse_env())
    d.update(extra or {})
    return d


def apply_to_samples(ts, overrides=None, max_shift=MAX_SHIFT, verbose=True):
    """Shift sampled attack/defence for the named clubs. Returns (ts, applied).

    Operates on the posterior draws, so the model's uncertainty is preserved — this
    moves the centre of a club's distribution and does not pretend to have learned
    something that narrows it.
    """
    ov = active(overrides)
    if not ov:
        if verbose:
            print("[team-override] none set")
        return ts, []
    idx = ts["idx"]
    applied = []
    for club, (da, dd) in ov.items():
        if club not in idx:
            if verbose:
                print(f"[team-override] '{club}' is not a club in this season — skipped")
            continue
        da = float(np.clip(da, -max_shift, max_shift))
        dd = float(np.clip(dd, -max_shift, max_shift))
        ts["att"][:, idx[club]] += da
        ts["dfn"][:, idx[club]] += dd
        applied.append((club, da, dd))
    if verbose and applied:
        print(f"[team-override] MANAGER JUDGMENT applied to {len(applied)} club(s) "
              f"— not a fitted effect:")
        for club, da, dd in applied:
            print(f"    {club:16s} att {da:+.3f}   def {dd:+.3f}   "
                  f"(~{abs(da)*100:.0f}% / {abs(dd)*100:.0f}% on expected goals)")
    return ts, applied


def selftest():
    idx = {"Newcastle": 0, "Arsenal": 1}
    ts = {"idx": idx, "att": np.zeros((4, 2)), "dfn": np.zeros((4, 2))}
    ts, ap = apply_to_samples(ts, {"Newcastle": (-0.08, -0.05)}, verbose=False)
    assert len(ap) == 1
    assert abs(ts["att"][0, 0] + 0.08) < 1e-9, "attack shift not applied"
    assert abs(ts["dfn"][0, 0] + 0.05) < 1e-9, "defence shift not applied"
    assert ts["att"][0, 1] == 0.0, "another club must be untouched"

    # cap holds
    ts2 = {"idx": idx, "att": np.zeros((2, 2)), "dfn": np.zeros((2, 2))}
    ts2, _ = apply_to_samples(ts2, {"Newcastle": (-5.0, 5.0)}, verbose=False)
    assert abs(ts2["att"][0, 0]) <= MAX_SHIFT + 1e-9, "cap not enforced"
    assert abs(ts2["dfn"][0, 0]) <= MAX_SHIFT + 1e-9

    # unknown club is skipped, not crashed on. NB `active()` merges the committed
    # OVERRIDES with what is passed, so the applied list is not empty here — the check
    # is that the UNKNOWN club specifically never appears.
    ts3 = {"idx": idx, "att": np.zeros((2, 2)), "dfn": np.zeros((2, 2))}
    ts3, ap3 = apply_to_samples(ts3, {"Nowhere FC": (-0.1, 0)}, verbose=False)
    assert "Nowhere FC" not in [c for c, _, _ in ap3], "unknown club must be skipped"

    e = parse_env("Newcastle:-0.08,-0.05;Everton:-0.03")
    assert e["Newcastle"] == (-0.08, -0.05)
    assert e["Everton"] == (-0.03, 0.0), "a single number means attack only"
    assert parse_env("") == {}
    assert parse_env("garbage") == {}

    print("SELFTEST OK: shifts applied to the named club only, cap enforced, unknown "
          "clubs skipped, env parsing handles one or two numbers and junk.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    print(__doc__)
    print("active overrides:", active())
