from __future__ import annotations
import config
"""
injury_impact.py — an unavailable starter weakens his TEAM, not just his own row.
=================================================================================
Availability currently acts only on the individual: an injured player's start prior
collapses and his projection goes with it. Nothing tells the team model that the side is
now worse. So with Saliba out, Arsenal's defence is still modelled at full strength and
every OTHER Arsenal defender keeps the same clean-sheet expectation — which is wrong in
the direction that matters most, because clean sheets are shared across the back line
and the goalkeeper.

WHAT THIS COMPUTES
------------------
Per club, the share of first-choice contribution missing, from the model's own priors —
not a new hand-set table:

    attacking loss  = sum over unavailable players of  npxgi_rate90 * exp_minutes/90
                      * baseline_p_start,  divided by the same sum over the whole squad
    defensive loss  = the same, restricted to GK/DEF, weighted by baseline minutes

Both are shares in [0, 1]. Attack strength is scaled by (1 - beta_att * att_loss) and
defensive strength by (1 - beta_def * def_loss), in log space, with a hard cap.

THE GUARD THIS HAS TO CLEAR, AND WHY IT ARGUABLY DOES
------------------------------------------------------
`PROJECT_KNOWLEDGE §3` requires any new team-level signal to beat the market before
entering `TeamModel`, or it double-counts what the odds already price. Betting markets
price known absences, so an injury adjustment is exactly the kind of thing that usually
IS already in the price.

The specific reason it is not double-counting here: the market-odds snapshot this model
blends is dated **7 August 2026**, and the injuries that move these numbers post-date it
— Saliba's back surgery confirmed 14 Aug, Timber 14 Aug, Doku's calf 16 Aug in the
Community Shield. The odds cannot contain information that did not exist when they were
taken. That argument holds only while the snapshot is stale; refresh the odds and this
adjustment should be re-derived or switched off, because then it WOULD double-count.

CALIBRATION IS UNFITTED [JUDGMENT]
-----------------------------------
`BETA_ATT` and `BETA_DEF` are not estimated. Estimating them needs a panel of injuries
with known team-strength effects, which this repo does not have. They are deliberately
small and hard-capped at `MAX_SHIFT` so the adjustment can move a fixture at the margin
but cannot rewrite the table, and the whole layer is OFF unless asked for. Treat the
magnitude as a prior, not a measurement.

Run:  python src/injury_impact.py --selftest
"""
import numpy as np
import pandas as pd

BETA_ATT = 0.45          # [JUDGMENT] share of attacking loss that reaches team strength
BETA_DEF = 0.40          # [JUDGMENT] same for defence
MAX_SHIFT = 0.12         # hard cap in log units — roughly a 12% swing, never more
MIN_LOSS = 0.03          # ignore trivial losses; avoids nudging every team every week


def contribution(players, availability=None):
    """Per-player contribution weights, from the priors the simulation already uses.

    `availability` maps player_code -> chance_play in [0, 1]. A player at 0.0 is fully
    missing, one at 0.5 counts as half missing. Uses the BASELINE start prior where
    available (`p_start_base`), so a prior already collapsed by the injury does not make
    the player look like he was never expected to play — which would zero the very loss
    this module is trying to measure.
    """
    p = players.copy()
    # The per-90 rates are DERIVED columns that only some frames carry: the board holds
    # the raw Gamma shapes (npxgi_alpha / npxgi_beta) and computes rates downstream in
    # the detail export. Requiring the derived column silently produced a zero attacking
    # weight for every player and killed the whole attacking channel — att_loss was
    # exactly 0.000 for all 20 clubs — so derive it here when it is absent.
    for rate, a, b in (("npxgi_rate90", "npxgi_alpha", "npxgi_beta"),
                       ("xa_rate90", "xa_alpha", "xa_beta")):
        if rate not in p.columns:
            if a in p.columns and b in p.columns:
                p[rate] = pd.to_numeric(p[a], errors="coerce") / \
                    pd.to_numeric(p[b], errors="coerce").replace(0, np.nan)
            else:
                p[rate] = np.nan
    if "exp_minutes" not in p.columns:
        p["exp_minutes"] = np.nan
    if not np.isfinite(pd.to_numeric(p["npxgi_rate90"], errors="coerce")).any():
        raise ValueError(
            "no usable attacking rate: supply npxgi_rate90 or npxgi_alpha/npxgi_beta")
    base = (p["p_start_base"] if "p_start_base" in p.columns
            else p["start_a"] / (p["start_a"] + p["start_b"]))
    mins = p["exp_minutes"].fillna(85.3) / 90.0
    att_rate = p["npxgi_rate90"].fillna(0) + 0.5 * p["xa_rate90"].fillna(0)
    p["_w_att"] = att_rate * mins * base.fillna(0)
    p["_w_def"] = np.where(p["pos"].isin(["GK", "DEF"]), mins * base.fillna(0), 0.0)
    if availability is not None:
        p["_avail"] = p["player_code"].map(availability).fillna(1.0).clip(0, 1)
    elif "chance_play" in p.columns:
        p["_avail"] = p["chance_play"].fillna(1.0).clip(0, 1)
    else:
        p["_avail"] = 1.0
    return p


def team_losses(players, availability=None, min_loss=MIN_LOSS):
    """Share of attacking and defensive contribution unavailable, per club."""
    p = contribution(players, availability)
    rows = []
    for team, g in p.groupby("team"):
        tot_a, tot_d = g["_w_att"].sum(), g["_w_def"].sum()
        miss = 1.0 - g["_avail"]
        la = float((g["_w_att"] * miss).sum() / tot_a) if tot_a > 0 else 0.0
        ld = float((g["_w_def"] * miss).sum() / tot_d) if tot_d > 0 else 0.0
        out = g.loc[miss > 0.5, "web_name"].tolist() if "web_name" in g.columns else []
        rows.append({"team": team, "att_loss": la if la >= min_loss else 0.0,
                     "def_loss": ld if ld >= min_loss else 0.0,
                     "att_loss_raw": la, "def_loss_raw": ld,
                     "players_out": ", ".join(map(str, out[:6]))})
    return pd.DataFrame(rows).sort_values("att_loss", ascending=False)


def apply_to_samples(ts, losses, beta_att=BETA_ATT, beta_def=BETA_DEF,
                     max_shift=MAX_SHIFT, verbose=True):
    """Shift the sampled team attack/defence by the availability loss.

    Operates on the posterior draws in place of refitting, so the team model's own
    uncertainty is preserved — this moves the centre of the distribution, it does not
    pretend to have learned something that narrows it.
    """
    idx = ts["idx"]
    A, D = ts["att"], ts["dfn"]
    moved = []
    for r in losses.itertuples():
        if r.team not in idx:
            continue
        da = float(np.clip(beta_att * r.att_loss, 0.0, max_shift))
        dd = float(np.clip(beta_def * r.def_loss, 0.0, max_shift))
        if da <= 0 and dd <= 0:
            continue
        A[:, idx[r.team]] -= da       # weaker attack
        D[:, idx[r.team]] -= dd       # weaker defence -> concedes more
        moved.append((r.team, da, dd, r.players_out))
    if verbose and moved:
        print(f"[injury-impact] {len(moved)} clubs weakened by unavailable starters "
              f"(beta_att={beta_att}, beta_def={beta_def}, cap={max_shift}):")
        for t, da, dd, who in sorted(moved, key=lambda x: -(x[1] + x[2]))[:8]:
            print(f"    {t:16s} att -{da:.3f}  def -{dd:.3f}   {who}")
    elif verbose:
        print("[injury-impact] no club cleared the minimum-loss threshold")
    return ts, moved


def selftest():
    pl = pd.DataFrame({
        "player_code": [1, 2, 3, 4, 5, 6],
        "web_name": ["Star", "Sub", "CB1", "CB2", "GK1", "OtherStar"],
        "team": ["A", "A", "A", "A", "A", "B"],
        "pos": ["FWD", "FWD", "DEF", "DEF", "GK", "FWD"],
        "npxgi_rate90": [0.8, 0.1, 0.05, 0.05, 0.0, 0.8],
        "xa_rate90": [0.2, 0.05, 0.02, 0.02, 0.0, 0.2],
        "exp_minutes": [88.0, 20.0, 88.0, 88.0, 90.0, 88.0],
        "start_a": [90.0, 10.0, 90.0, 90.0, 90.0, 90.0],
        "start_b": [10.0, 90.0, 10.0, 10.0, 10.0, 10.0],
    })
    # nobody out -> no loss anywhere
    L0 = team_losses(pl, availability={})
    assert L0["att_loss"].sum() == 0.0, "no injuries should mean no loss"

    # the star forward out -> A loses attack, B untouched
    L1 = team_losses(pl, availability={1: 0.0}).set_index("team")
    assert L1.loc["A", "att_loss"] > 0.5, f"star out should be a big loss: {L1.loc['A','att_loss']:.2f}"
    assert L1.loc["A", "def_loss"] == 0.0, "a forward should not weaken the defence"
    assert L1.loc["B", "att_loss"] == 0.0, "another club must be unaffected"

    # a centre-back out -> defensive loss, minimal attacking loss
    L2 = team_losses(pl, availability={3: 0.0}).set_index("team")
    assert L2.loc["A", "def_loss"] > 0.2, "CB out should weaken the defence"
    assert L2.loc["A", "att_loss"] < L1.loc["A", "att_loss"]

    # partial availability counts partially
    L3 = team_losses(pl, availability={1: 0.5}).set_index("team")
    assert 0 < L3.loc["A", "att_loss"] < L1.loc["A", "att_loss"]

    # samples move the right way and stay capped
    idx = {"A": 0, "B": 1}
    ts = {"idx": idx, "att": np.zeros((5, 2)), "dfn": np.zeros((5, 2))}
    big = pd.DataFrame([{"team": "A", "att_loss": 1.0, "def_loss": 1.0,
                         "players_out": "Star"}])
    ts, moved = apply_to_samples(ts, big, verbose=False)
    assert ts["att"][0, 0] < 0 and ts["dfn"][0, 0] < 0, "A should be weakened"
    assert abs(ts["att"][0, 0]) <= MAX_SHIFT + 1e-9, "shift must respect the cap"
    assert ts["att"][0, 1] == 0.0, "B must be untouched"

    # A frame carrying only the raw Gamma shapes must work identically. This is the
    # case the board actually passes, and getting it wrong zeroed the attacking channel
    # for every club without raising anything.
    shapes = pl.drop(columns=["npxgi_rate90", "xa_rate90"]).copy()
    shapes["npxgi_alpha"] = pl["npxgi_rate90"] * 40.0
    shapes["npxgi_beta"] = 40.0
    shapes["xa_alpha"] = pl["xa_rate90"] * 40.0
    shapes["xa_beta"] = 40.0
    Ls = team_losses(shapes, availability={1: 0.0}).set_index("team")
    assert Ls.loc["A", "att_loss"] > 0.5, \
        f"alpha/beta-only frame must still see the attacking loss, got {Ls.loc['A','att_loss']:.3f}"
    assert abs(Ls.loc["A", "att_loss"] - L1.loc["A", "att_loss"]) < 1e-9, \
        "derived rates must match supplied rates exactly"

    # and a frame with NO attacking information at all must fail loudly, not silently
    blind = pl.drop(columns=["npxgi_rate90", "xa_rate90"]).copy()
    try:
        team_losses(blind)
        raise AssertionError("a frame with no attacking rate must raise")
    except ValueError:
        pass

    print("SELFTEST OK: losses scale with contribution and availability, forwards hit "
          "attack only, defenders hit defence, partial doubts count partially, other "
          "clubs untouched, shift capped, raw Gamma shapes give identical answers, "
          "missing attacking data raises instead of silently zeroing.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    print(__doc__)
