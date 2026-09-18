from __future__ import annotations
import config
"""
defcon_roles.py — centre-back vs full-back, for the DefCon prior ONLY.
=======================================================================
Measured on `studies/defcon_matchups.csv` (2,934 appearances of 60+ minutes, 25/26):

    CB       1,587.5 90s    9.811 DefCon per 90    hit rate 0.480
    FB       1,269.3 90s    7.063 DefCon per 90    hit rate 0.207
    all DEF  2,856.8 90s    8.590 per 90           hit rate 0.358

A single `PRIOR_DC["DEF"] = 7.6` cannot express a 2.3x difference in hit rate, and it is
not even the right pooled value — 7.6 sits below the measured 8.590 for defenders as a
whole, so every thin-history defender was being shrunk toward a target that is too low
regardless of role.

SCOPE — DELIBERATELY NARROW
----------------------------
Role is used for ONE thing: the mean of the DefCon Gamma prior. It does not touch
`pos`, which stays DEF for FPL purposes and continues to drive the DefCon threshold (10),
the clean-sheet multiplier, the goal multiplier and every other scoring rule. Nothing
downstream should be able to tell a CB from an FB except the DefCon rate they are shrunk
toward. `assert_scope` exists to make that testable rather than asserted.

WHAT THIS CAN AND CANNOT REACH  [measured 2026-08-20]
------------------------------------------------------
Of 191 defenders in the 26/27 squad, 108 (57%) carry a role from 25/26 appearances.
Every one of those 108 is an ESTABLISHED player, and all 65 cold-start defenders are
unlabelled — a player with no Premier League minutes has no measured average position to
classify. So the split does NOT reach the group whose prior dominates their posterior.

That is a real limitation, not a rounding error, and it caps the value of this layer:
  * for a CB with a full season of minutes, the likelihood swamps the prior and the role
    changes almost nothing;
  * the split bites hardest on labelled defenders with THIN history — a returning or
    rotated CB whose own record is too short to identify his rate;
  * for cold-start defenders the only available improvement is the corrected POOLED
    value (8.590 rather than 7.6), which this module also supplies.

Getting role onto cold-start players needs an external positional source (a lineup feed
carrying CB/RB/LB, or average-position data for a player's previous league). Until then
`role_for` returns None for them and they take the pooled prior.

WHERE THE LABELS COME FROM  [changed 2026-09-10]
-------------------------------------------------
`role_map()` reads `data/defcon_roles.csv` (config.DEFCON_ROLES), a FROZEN copy of the
labels, and never the study CSV. Until 2026-09-10 it read `studies/defcon_matchups.csv`
directly, which made a study file a live input to `ms_priors.pkl` that the manifest could
not see — and `test_all` rewrites that file on every run, so its mtime could not have been
tracked honestly even if it had been declared. Worse, a missing file returned `{}` without
a word, silently moving every labelled defender onto the pooled prior. The frozen file is
a declared `committed` node, and a missing one now fails the build.

Re-freeze only when the study's underlying data changes (it is 25/26 appearances, so in
practice never this season):  python src/defcon_roles.py --freeze

Run:  python src/defcon_roles.py --selftest
      python src/defcon_roles.py --check      (needs FPL_DATA)
      python src/defcon_roles.py --freeze     (study CSV -> data/defcon_roles.csv)
"""
import os
import numpy as np
import pandas as pd

STUDY = config.DEFCON_MATCHUPS_STUDY     # research input: measure(), freeze()
FROZEN = config.DEFCON_ROLES             # pipeline input: role_map()

# Measured per-90 rates. Recomputed by `measure()`; these are the committed values so a
# prior build does not silently depend on a study file being present.
RATE_CB = 9.811
RATE_FB = 7.063
RATE_DEF_POOLED = 8.590          # replaces the old 7.6, which was below the measurement
MIN_APPS = 3                     # appearances needed before a role label is trusted


def measure(path=None, min_mins=60):
    """Re-estimate the per-90 rates from the study file. Sum(dc)/sum(90s), not the mean
    of per-appearance counts — appearances differ in length and the rate is per 90."""
    p = path or STUDY
    if not os.path.exists(p):
        return {}
    M = pd.read_csv(p)
    M = M[M["mins"] >= min_mins]
    out = {}
    for role, g in M.groupby("role"):
        n90 = g["mins"].sum() / 90.0
        if n90 > 0:
            out[role] = {"rate90": float(g["dc"].sum() / n90),
                         "hit_rate": float(g["hit"].mean()),
                         "n_apps": int(len(g)), "n90": float(n90)}
    n90 = M["mins"].sum() / 90.0
    if n90 > 0:
        out["ALL"] = {"rate90": float(M["dc"].sum() / n90),
                      "hit_rate": float(M["hit"].mean()),
                      "n_apps": int(len(M)), "n90": float(n90)}
    return out


def derive_roles(path=None, min_apps=MIN_APPS):
    """Frame of player_code, role, n_role, n_apps — from per-appearance study rows.

    A player who appears under both labels takes his majority role; one with fewer than
    `min_apps` appearances is left unlabelled rather than classified on one or two games.
    Ties in `n` are broken by player_code then role, so the frozen file is deterministic.
    """
    M = pd.read_csv(path or STUDY).dropna(subset=["player_code", "role"])
    cnt = M.groupby(["player_code", "role"]).size().rename("n_role").reset_index()
    tot = cnt.groupby("player_code")["n_role"].sum().rename("n_apps")
    cnt = cnt.merge(tot, on="player_code")
    cnt = cnt[cnt["n_apps"] >= min_apps]
    best = (cnt.sort_values(["player_code", "n_role", "role"], ascending=[True, False, True])
               .drop_duplicates("player_code"))
    return best[["player_code", "role", "n_role", "n_apps"]].reset_index(drop=True)


def freeze(src=None, out=None, min_apps=MIN_APPS):
    """Write the frozen role file the pipeline reads. Returns its path."""
    out = out or FROZEN
    derive_roles(src, min_apps).to_csv(out, index=False)
    return out


def role_map(path=None):
    """player_code -> 'CB' | 'FB', from the FROZEN role file.

    Raises if the file is absent. The old behaviour — an empty map — moved every labelled
    defender onto the pooled prior without a word, which is a different model, not a
    degraded one.
    """
    p = path or FROZEN
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} is missing. It is a committed input to the DefCon prior; regenerate "
            f"with `python src/defcon_roles.py --freeze` (reads {STUDY}).")
    M = pd.read_csv(p).dropna(subset=["player_code", "role"])
    return dict(zip(M["player_code"], M["role"]))


def prior_rate(pos, role=None):
    """The DefCon rate a player should be shrunk toward. Role only refines DEF."""
    if pos != "DEF":
        return None
    if role == "CB":
        return RATE_CB
    if role == "FB":
        return RATE_FB
    return RATE_DEF_POOLED


def attach(players, path=None, verbose=True):
    """Add `dc_role` and `dc_prior_rate`. Adds no other column and changes none."""
    p = players.copy()
    rm = role_map(path)
    p["dc_role"] = p["player_code"].map(rm) if "player_code" in p.columns else None
    p["dc_prior_rate"] = [prior_rate(po, ro) for po, ro in
                          zip(p.get("pos", pd.Series(index=p.index)), p["dc_role"])]
    if verbose:
        d = p[p.get("pos") == "DEF"] if "pos" in p.columns else p
        lab = d["dc_role"].notna().sum()
        print(f"[defcon-roles] {lab}/{len(d)} defenders labelled "
              f"({lab/max(len(d),1):.0%}); CB {int((d.dc_role=='CB').sum())}, "
              f"FB {int((d.dc_role=='FB').sum())}, "
              f"unlabelled use the pooled {RATE_DEF_POOLED}")
    return p


def assert_scope(before, after, allowed=("dc_role", "dc_prior_rate",
                                         "defcon_alpha", "defcon_beta")):
    """Fail if role leaked into anything other than the DefCon prior.

    The whole point of this module is that a CB and an FB remain identical to the model
    in every respect except the DefCon rate they shrink toward. This makes that checkable
    instead of a claim in a docstring.
    """
    common = [c for c in before.columns if c in after.columns and c not in allowed]
    changed = []
    for c in common:
        b, a = before[c], after[c]
        try:
            if not b.equals(a):
                changed.append(c)
        except Exception:
            pass
    if changed:
        raise AssertionError(f"role leaked into non-DefCon columns: {changed}")
    return True


def selftest():
    import tempfile
    root = tempfile.mkdtemp()
    p = os.path.join(root, "dc.csv")
    rows = []
    # a CB scoring ~10 per 90 and an FB scoring ~7, both over many appearances
    for i in range(30):
        rows.append({"player_code": 1, "role": "CB", "dc": 10, "hit": 1, "mins": 90})
        rows.append({"player_code": 2, "role": "FB", "dc": 7, "hit": 0, "mins": 90})
    rows.append({"player_code": 3, "role": "CB", "dc": 9, "hit": 0, "mins": 90})  # thin
    pd.DataFrame(rows).to_csv(p, index=False)

    m = measure(p)
    assert abs(m["CB"]["rate90"] - 10.0) < 0.1, m["CB"]
    assert abs(m["FB"]["rate90"] - 7.0) < 0.1, m["FB"]

    # derive -> freeze -> read round-trips, and role_map never reads the study format
    fz = freeze(src=p, out=os.path.join(root, "roles.csv"), min_apps=3)
    rm = role_map(fz)
    assert rm[1] == "CB" and rm[2] == "FB"
    assert 3 not in rm, "a single appearance must not earn a role label"
    assert list(pd.read_csv(fz).columns) == ["player_code", "role", "n_role", "n_apps"]
    # a missing frozen file must FAIL, not silently return every defender to pooled
    try:
        role_map(os.path.join(root, "absent.csv"))
        raise AssertionError("role_map returned on a missing file")
    except FileNotFoundError:
        pass

    # prior_rate: role refines DEF only, and never touches other positions
    assert prior_rate("DEF", "CB") == RATE_CB
    assert prior_rate("DEF", "FB") == RATE_FB
    assert prior_rate("DEF", None) == RATE_DEF_POOLED
    assert prior_rate("MID", "CB") is None, "role must not affect midfielders"
    assert prior_rate("GK", None) is None
    assert RATE_CB > RATE_DEF_POOLED > RATE_FB, "pooled must sit between the roles"

    # attach adds exactly two columns and changes nothing else
    pl = pd.DataFrame({"player_code": [1, 2, 3], "pos": ["DEF", "DEF", "MID"],
                       "web_name": ["CBman", "FBman", "Mid"], "other": [1.0, 2.0, 3.0]})
    out = attach(pl, path=fz, verbose=False)
    assert set(out.columns) - set(pl.columns) == {"dc_role", "dc_prior_rate"}
    assert_scope(pl, out)
    assert out.loc[out.player_code == 1, "dc_prior_rate"].iloc[0] == RATE_CB
    assert out.loc[out.player_code == 2, "dc_prior_rate"].iloc[0] == RATE_FB
    assert pd.isna(out.loc[out.player_code == 3, "dc_prior_rate"].iloc[0]), \
        "a midfielder must get no DefCon role rate"

    # scope guard actually fires when something else changes
    bad = out.copy(); bad.loc[0, "other"] = 99.0
    try:
        assert_scope(pl, bad)
        raise AssertionError("scope guard failed to catch a leak")
    except AssertionError as e:
        assert "leaked" in str(e)

    print("SELFTEST OK: per-90 rates estimated as sum(dc)/sum(90s), majority role with a "
          "minimum-appearance floor, pooled rate between CB and FB, role confined to DEF "
          "and to the DefCon prior, scope guard catches leaks.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    if "--check" in sys.argv:
        import warnings; warnings.filterwarnings("ignore")
        m = measure()
        for k, v in m.items():
            print(f"  {k:4s} rate90 {v['rate90']:.3f}  hit {v['hit_rate']:.3f}  "
                  f"apps {v['n_apps']}")
        rm = role_map()
        fresh = dict(zip(*[derive_roles()[c] for c in ("player_code", "role")]))
        print(f"\n  role labels: {len(rm)} players frozen; the study now derives "
              f"{len(fresh)}, {'IDENTICAL' if fresh == rm else 'DIFFERENT — re-freeze?'}")
        sys.exit(0)
    if "--freeze" in sys.argv:
        out = freeze()
        print(f"[defcon-roles] froze {len(role_map(out))} labels -> {out}")
        sys.exit(0)
    print(__doc__)
