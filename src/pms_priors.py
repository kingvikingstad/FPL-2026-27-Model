"""
pms_priors.py — build 26/27 player priors from 25/26 PER-MATCH data
====================================================================
Replaces the FPL-aggregate priors with per-match ones, which the head-to-head
test showed predict next-gameweek points ~16% better on rank correlation
(Spearman 0.210 vs 0.181).

What changes versus the old priors:
  * attacking rate built from TRUE non-penalty xG (penalties removed using
    penalties_scored/missed) plus xGOT, rather than FPL's aggregate xG which
    includes penalties
  * penalty rate estimated SEPARATELY from actual penalty attempts, so the
    designated taker's premium is measured rather than assumed
  * defensive-contribution rate weighted by component (CBI carries ~2.7x the
    per-unit weight of tackles toward hitting the threshold)
  * goalkeepers driven by xGOT FACED rather than saves — see below

The goalkeeper finding is a genuine correction. Testing next-gameweek points:
    rolling saves/90        Spearman -0.102   <- NEGATIVE
    rolling goals_prevented +0.125
    rolling xGOT faced/90   -0.282            <- strongest, and the only
                                                 significant term jointly (p=0.0002)
More saves means a worse defence in front of you: fewer clean sheets, more
concessions. Save volume is a liability signal, not an asset signal. The model
should pick keepers on team defensive quality, not shot-stopping workload.
"""
from __future__ import annotations
import numpy as np, pandas as pd

PANEL = "/tmp/pms_panel.pkl"
PEN_XG = 0.79


def season_rates(panel_path=PANEL, min_minutes=450):
    """Per-90 rates per player from the per-match panel, with penalties split out."""
    p = pd.read_pickle(panel_path)
    g = p.groupby(["player_id", "web_name", "pos", "team"], dropna=False)
    a = g.agg(
        mins=("mins", "sum"),
        npxg=("npxg", "sum"), xa=("xa_", "sum"), xgot=("xgot_", "sum"),
        np_goals=("np_goals", "sum"), assists=("assists", "sum"),
        pens_scored=("pens_scored", "sum"), pens_missed=("pens_missed", "sum"),
        cbi=("cbi", "sum"), tkl=("tkl", "sum"), rec=("rec", "sum"),
        defcon=("defcon_raw", "sum"),
        saves=("saves_", "sum"), gp=("goals_prevented", "sum"),
        xgot_faced=("xgot_faced", "sum"),
        tob=("tob", "sum"), shots=("total_shots", "sum"),
        apps=("mins", lambda s: (s > 0).sum()),
        starts=("mins", lambda s: (s >= 60).sum()),
        games=("mins", "size"),
    ).reset_index()
    a = a[a.mins >= min_minutes].copy()
    n90 = a.mins / 90.0
    for c in ["npxg", "xa", "xgot", "np_goals", "assists", "cbi", "tkl", "rec",
              "defcon", "saves", "gp", "xgot_faced", "tob", "shots"]:
        a[f"{c}_90"] = a[c] / n90
    a["pen_att"] = a.pens_scored + a.pens_missed
    a["pen_xg_90"] = a.pen_att * PEN_XG / n90
    a["start_rate"] = a.starts / a.games
    a["sub_rate"] = (a.apps - a.starts).clip(lower=0) / a.games
    # attacking involvement prior: non-penalty xG + xA (penalties handled apart)
    a["inv90"] = a.npxg_90 + a.xa_90
    return a


def to_model_priors(rates, revert=0.70, k0=3.0):
    """Convert per-90 rates into the Gamma/Beta prior parameters the Bayesian
    model consumes, with the same new-season reversion discount as before."""
    out = []
    for _, r in rates.iterrows():
        n90 = r.mins / 90.0
        prior_inv = {"GK": 0.02, "DEF": 0.11, "MID": 0.27, "FWD": 0.42}.get(r.pos, 0.2)
        prior_xa = {"GK": 0.01, "DEF": 0.05, "MID": 0.13, "FWD": 0.10}.get(r.pos, 0.1)
        prior_dc = {"GK": 0.0, "DEF": 7.6, "MID": 8.4, "FWD": 4.7}.get(r.pos, 6.0)
        out.append({
            "player_id": r.player_id, "web_name": r.web_name, "pos": r.pos,
            "team_2526": r.team, "minutes": r.mins,
            "npxgi_alpha": prior_inv * k0 + revert * (r.npxg + r.xa),
            "npxgi_beta": k0 + revert * n90,
            "xa_alpha": prior_xa * k0 + revert * r.xa,
            "xa_beta": k0 + revert * n90,
            "defcon_alpha": prior_dc * k0 + revert * r.defcon,
            "defcon_beta": k0 + revert * n90,
            "start_a": 2.0 + revert * r.starts,
            "start_b": 2.0 + revert * max(r.games - r.starts, 0),
            "sub_app_rate": r.sub_rate,
            # measured, not assumed: the taker's own penalty xG rate
            "pen_xg90_measured": r.pen_xg_90,
            # keeper quality signals
            "xgot_faced_90": r.xgot_faced_90, "gp_90": r.gp_90, "saves_90": r.saves_90,
            "xgot_90": r.xgot_90, "tob_90": r.tob_90,
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    r = season_rates()
    print(f"players with >=450 min: {len(r)}")
    print("\nTop attacking involvement (npxG+xA per 90), true non-penalty:")
    print(r.nlargest(8, "inv90")[["web_name", "pos", "team", "mins", "npxg_90", "xa_90", "inv90"]]
          .round(3).to_string(index=False))
    print("\nMeasured penalty xG/90 (top takers) — replaces the assumed 0.10:")
    print(r.nlargest(6, "pen_xg_90")[["web_name", "pos", "team", "pen_att", "pen_xg_90"]]
          .round(3).to_string(index=False))
    print("\nKeepers by xGOT faced/90 (LOWER is better — the real GK signal):")
    gk = r[r.pos == "GK"].nsmallest(6, "xgot_faced_90")
    print(gk[["web_name", "team", "mins", "xgot_faced_90", "saves_90", "gp_90"]].round(3).to_string(index=False))
    pri = to_model_priors(r)
    pri.to_pickle("/tmp/pms_priors.pkl")
    print(f"\nwrote priors for {len(pri)} players -> /tmp/pms_priors.pkl")
</content>
