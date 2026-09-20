from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
defcon_team_matchups.py — which TEAMS and which MATCHUPS produce DefCon?
========================================================================
PRE-REGISTRATION. Design, estimands, permutation baselines and decision rules fixed
before any coefficient was read.

WHAT THE EXISTING NULL DOES AND DOES NOT SAY
--------------------------------------------
`defcon_matchups` binned opponents into STRENGTH quartiles and found the DefCon rate
flat (0.339 / 0.328 / 0.416 / 0.329), concluding the supply hypothesis is wrong and the
opponent does not matter. The first half of that is right. The second does not follow:
strength is a SCALAR, and "how much defending this opponent forces" is a different
quantity that a strength bin cannot express. Two equally strong sides can differ a lot —
one plays through the middle and concedes no clearances, the other crosses relentlessly.

So this re-asks the question on opponent IDENTITY rather than opponent strength, which
is a strictly richer partition and is the thing a fixture planner actually has.

ESTIMANDS
---------
DC actions per eligible appearance (60+ minutes), and the positional threshold hit
(DEF 10, MID/FWD 12). Actions are the primary outcome — the threshold discards
information and is noisier — with the hit rate reported because it is what pays.

  OPPONENT effect   estimated WITHIN player. Every player faces many opponents inside a
                    season, so this is identified and is not a statement about who
                    happened to be on the pitch. THE quantity of interest.
  OWN-TEAM effect   NOT identified. A player belongs to one club, so club and player are
                    nested and player-demeaning annihilates the club term. Reported
                    descriptively and explicitly labelled composition-confounded.
  MATCHUP           own-team x opponent beyond additive.

THE DEGREES-OF-FREEDOM TRAP, HANDLED UP FRONT
----------------------------------------------
A club x opponent grouping is 380 cells on ~3,000 rows — under 8 rows per cell. Such a
grouping "explains" a large share of ANY variance, signal or not. Every R-squared below
is therefore reported against a PERMUTATION baseline: the same grouping applied after
shuffling the opponent labels. Only the excess over that baseline is evidence. Reporting
the raw number would manufacture a matchup effect out of arithmetic.

DECISION RULES (fixed before results)
--------------------------------------
D1  SHIP an opponent DefCon rating for a position only if its split-half reliability
    across the two halves of the season is >= 0.5. Ratings are empirical-Bayes shrunk;
    an unreliable position is returned flagged, never consumed.
D2  ADOPT a matchup (club x opponent) term only if its excess R-squared over the
    permutation baseline exceeds the opponent main effect's excess. Otherwise the
    interaction is degrees of freedom and is dropped.
D3  Do NOT adopt any own-team term from this design, whatever it looks like — it is not
    identified within one season. Recorded as a design limit, not a null.

Sample: 25/26, the only season carrying `defensive_contributions` (100% null before it;
never fill those with zero).

CORRECTED 2026-09-10. The DEF count is now CBIT summed from components
(`defcon_team.build_panel` -> `defcon_series`); the published column this study first ran
on is CBIRT for defenders in GW2-10. Section 5 re-runs the headline under both series.

Run:  python studies/defcon_team_matchups.py
      python studies/defcon_team_matchups.py --selftest
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import defcon_team as dt

SEASON = "2025-2026"
N_PERM = 200
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                    "defcon_team_matchups.csv")


BLOCK = ["club", "match_id"]   # one club's defenders in one match share a shock


def perm_r2(s, group_cols, value, n=N_PERM, seed=0, block=None):
    """R-squared of a grouping, against a null that permutes the OUTCOME.

    The obvious null — shuffling the opponent LABELS — is wrong here. The fixture
    calendar makes club x opponent cells almost exactly balanced (every club plays
    every other twice), and shuffling labels destroys that balance, producing more
    small cells and a HIGHER baseline R-squared than the real design. The baseline
    then exceeds the observed value and the comparison is meaningless.

    Permuting the outcome instead holds the design — and therefore every cell size —
    exactly fixed, and destroys only the association being tested. That is the null
    the excess should be measured against.

    `block`: permute whole BLOCKS, not rows. [VERIFIED 2026-09-10, stats-referee] A
    club's ~4 defenders in one match share that match's shock (game state, red card,
    rout), so rows are not exchangeable within a grouping that nests club-matches —
    opponent, club x opponent and match identity all do. Row-level permutation breaks
    that shared shock apart and hands every such grouping a null it beats by
    construction: the "club x opponent clears its null" leg of D2 was exactly this.
    With `block`, rows are collapsed to block means (weighted by rows) and the block
    outcomes are permuted, so the design is still fixed and the shock stays whole.
    Every grouping in `group_cols` must be constant within a block.
    """
    if block:
        keep = [c for c in group_cols if c not in block]
        s = s.groupby(block, observed=True).agg(
            _v=(value, "mean"), _w=(value, "size"),
            **{c: (c, "first") for c in keep}).reset_index()
        v, w = s["_v"].to_numpy(dtype=float), s["_w"].to_numpy(dtype=float)
    else:
        v, w = s[value].to_numpy(dtype=float), np.ones(len(s))
    codes = s.groupby(group_cols, observed=True).ngroup().to_numpy()
    ncell = codes.max() + 1

    def r2(vals, wts):
        mu = np.average(vals, weights=wts)
        V = np.average((vals - mu) ** 2, weights=wts)
        sums = np.bincount(codes, weights=vals * wts, minlength=ncell)
        cnts = np.bincount(codes, weights=wts, minlength=ncell)
        means = sums / np.maximum(cnts, 1e-12)
        return 1 - np.average((vals - means[codes]) ** 2, weights=wts) / V

    real = r2(v, w)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.permutation(len(v))
        vals.append(r2(v[i], w[i]))
    vals = np.array(vals)
    return real, vals.mean(), np.percentile(vals, [2.5, 97.5])


def main():
    panel = dt.build_panel(SEASON)
    if panel.empty:
        print("[study] no DefCon panel — check FPL_DATA"); return
    print("=" * 78)
    print("DEFCON BY TEAM AND MATCHUP — 25/26, pre-registered")
    print("=" * 78)
    print(f"[sample] {len(panel)} eligible appearances (60+ min), "
          f"{panel['player_code'].nunique()} players, {panel['club'].nunique()} clubs")
    print(panel.groupby("position").agg(n=("dc", "size"), mean_actions=("dc", "mean"),
                                        hit=("hit", "mean")).round(3).to_string())

    print("\n" + "=" * 78)
    print("1. VARIANCE DECOMPOSITION, AGAINST A PERMUTATION BASELINE")
    print("=" * 78)
    print("  A club x opponent grouping is 380 cells on ~3,000 rows. It explains a lot")
    print("  of any variance. Only the EXCESS over shuffled labels is evidence.")
    rowsout = []
    for pos in ("Defender", "Midfielder"):
        s = panel[panel["position"] == pos].copy()
        if len(s) < 200:
            continue
        print(f"\n  {pos} (n={len(s)}), outcome = DC actions:")
        V = s["dc"].var()
        pl = 1 - (s["dc"] - s.groupby("player_code")["dc"].transform("mean")).var() / V
        print(f"    {'player identity':28s} R2 {pl:6.3f}   (player is the unit; no "
              f"permutation)")
        # Everything below is measured on PLAYER-DEMEANED actions, so it is variance
        # explained beyond who happened to be on the pitch.
        s["dc_dm"] = s["dc"] - s.groupby("player_code")["dc"].transform("mean")
        # Row-level permutation is ANTI-CONSERVATIVE here (it splits a club-match's
        # shared shock); it is printed for the audit trail and decides nothing. The
        # verdicts use the club-match BLOCK null (see perm_r2).
        for lab, cols in (("opponent", ["opp"]), ("club x opponent", ["club", "opp"])):
            rr_, rb_, rc_ = perm_r2(s, cols, "dc_dm")
            real, base, ci = perm_r2(s, cols, "dc_dm", block=BLOCK)
            excess = real - base
            print(f"    {lab:28s} rows: R2 {rr_:6.3f} vs {rb_:6.3f} [{rc_[0]:.3f},{rc_[1]:.3f}]"
                  f" | BLOCKS: R2 {real:6.3f} vs {base:6.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
                  f"  EXCESS {excess:+.3f}{'  *' if real > ci[1] else ''}")
            rowsout.append({"position": pos, "term": lab, "r2": real,
                            "r2_permuted": base, "excess": excess,
                            "beats_null": bool(real > ci[1])})

    print("\n  D2, AS PRE-REGISTERED, WAS THE WRONG STATISTIC — replaced, and the")
    print("  original reported so the change is auditable.")
    print("  It compared the club x opponent excess against the opponent excess. But")
    print("  the club x opponent grouping CONTAINS the opponent main effect, so it is")
    print("  bound to be larger and the comparison cannot isolate an interaction.")
    print("  Correct test: strip the additive opponent effect first, then ask whether")
    print("  club x opponent explains anything in what is left.")
    for pos in ("Defender", "Midfielder"):
        e = {r["term"]: r["excess"] for r in rowsout if r["position"] == pos}
        if len(e) != 2:
            continue
        s = panel[panel["position"] == pos].copy()
        s["dc_dm"] = s["dc"] - s.groupby("player_code")["dc"].transform("mean")
        # residual of the ADDITIVE model: player effect and a common opponent effect out
        s["resid_add"] = s["dc_dm"] - s.groupby("opp")["dc_dm"].transform("mean")
        real, base, ci = perm_r2(s, ["club", "opp"], "resid_add", block=BLOCK)
        beats = bool(real > ci[1])
        rrow = perm_r2(s, ["club", "opp"], "resid_add")
        print(f"\n    {pos}:")
        print(f"      pre-registered (nested, misleading): opponent {e['opponent']:+.3f}"
              f" vs matchup {e['club x opponent']:+.3f}")
        print(f"      interaction on additive residuals, ROW null (anti-conservative): "
              f"R2 {rrow[0]:.3f} vs {rrow[1]:.3f} [{rrow[2][0]:.3f},{rrow[2][1]:.3f}]")
        print(f"      interaction on additive residuals, BLOCK null: R2 {real:.3f} vs "
              f"permuted {base:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  ->  "
              f"{'clears its null' if beats else 'inside the null'}")
        rowsout.append({"position": pos, "term": "interaction | additive", "r2": real,
                        "r2_permuted": base, "excess": real - base,
                        "beats_null": bool(real > ci[1])})

        # A club x opponent cell is TWO matches. So an "interaction" that clears the
        # permutation null may simply be match-level shock — a red card, a game state,
        # a rout — which is real variance and completely unpredictable. Two checks
        # separate a repeatable tactical matchup from that.
        rm, bm, cm = perm_r2(s, ["match_id"], "resid_add", block=BLOCK)
        print(f"      match identity on the same residual (BLOCKS):  R2 {rm:.3f} vs permuted "
              f"{bm:.3f} [{cm[0]:.3f},{cm[1]:.3f}]")
        if rm >= real:
            print("      -> match identity explains AT LEAST as much as club x opponent,")
            print("         so the interaction is match-level shock, not a matchup.")
        # do the two meetings of the same pair agree? that is the repeatability test
        s["_pair"] = s["club"].astype(str) + " v " + s["opp"].astype(str)
        cell = s.groupby(["_pair", "match_id"])["resid_add"].mean().reset_index()
        cell["k"] = cell.groupby("_pair").cumcount()
        wide = cell[cell["k"] < 2].pivot(index="_pair", columns="k",
                                         values="resid_add").dropna()
        rr = np.nan
        if len(wide) >= 20:
            rr = float(wide[0].corr(wide[1]))
            print(f"      REPEATABILITY across the two meetings of the same pair: "
                  f"r={rr:+.3f} on {len(wide)} pairs")
        repeats = bool(np.isfinite(rr) and rr > 0.2)
        not_shock = bool(rm < real)
        adopt = beats and not_shock and repeats
        print(f"      D2 VERDICT: {'ADOPT' if adopt else 'DROP'} — a matchup term needs "
              f"all three of: clears its null ({beats}), not subsumed by match identity "
              f"({not_shock}), repeats across the two meetings ({repeats}).")
        rowsout.append({"position": pos, "term": "matchup verdict", "r2": real,
                        "r2_permuted": base, "excess": real - base,
                        "beats_null": beats, "repeatability": rr, "adopted": adopt})

    print("\n" + "=" * 78)
    print("2. THE OPPONENT RATING — identified, shrunk, reliability-gated")
    print("=" * 78)
    keep = []
    for pos in ("Defender", "Midfielder"):
        r = dt.opponent_defcon_ratings(panel, pos)
        if r.empty:
            continue
        rel = r["reliability"].iloc[0]
        k = r["shrink_factor"].iloc[0]
        usable = bool(r["usable"].iloc[0])
        print(f"\n  {pos}: split-half reliability {rel:+.3f}, EB shrink factor {k:.2f}, "
              f"D1 -> {'SHIP' if usable else 'DO NOT SHIP'}")
        print("    (the two are computed independently; agreement is a check on the "
              "noise model)")
        show = r[["team", "n", "raw", "shrunk", "hit_shrunk", "category"]]
        print(show.round(3).head(5).to_string(index=False))
        print("    ...")
        print(show.round(3).tail(5).to_string(index=False))
        sw = r["hit_shrunk"].max() - r["hit_shrunk"].min()
        print(f"    hit-probability swing across opponents: {sw:.3f} "
              f"= {2*sw:.2f} DefCon points per match between the extremes")
        r["season"] = SEASON
        keep.append(r)

    print("\n" + "=" * 78)
    print("3. IS THIS JUST OPPONENT STRENGTH RE-DISCOVERED?")
    print("=" * 78)
    print("  It should not be - the earlier study found the rate flat in strength.")
    dsum = panel.groupby("opp").agg(conceded=("hit", "mean")).reset_index()
    # opponent strength proxy: goals that opponent scores per match, from the panel's
    # own match keys, so no extra source is introduced
    import team_explosiveness as te
    g = te.team_match_goals(SEASON)
    strength = g.groupby("club")["g"].mean().rename("opp_goals_per_match")
    for pos in ("Defender", "Midfielder"):
        r = dt.opponent_defcon_ratings(panel, pos)
        if r.empty:
            continue
        j = r.set_index("team")[["shrunk"]].join(strength).dropna()
        print(f"  {pos}: corr(DefCon permissiveness, opponent attacking output) = "
              f"{j['shrunk'].corr(j['opp_goals_per_match']):+.3f}")
    print("  A modest positive correlation means permissive opponents are also the")
    print("  better attacking sides — so DefCon and clean sheets still point the SAME")
    print("  way, and there is still no 'tough fixture, DefCon floor' compensation.")

    print("\n" + "=" * 78)
    print("4. OWN TEAM — NOT IDENTIFIED (D3), reported so the reason is on record")
    print("=" * 78)
    for pos in ("Defender",):
        o = dt.own_team_profile(panel, pos)
        print(f"  {pos}: raw club means range "
              f"{o['mean_actions'].min():.2f} to {o['mean_actions'].max():.2f}, "
              f"split-half r={o['reliability_raw'].iloc[0]:+.3f}")
        print("  That reliability is high and MEANINGLESS as a team effect: within a")
        print("  season a player has exactly one club, so this is largely 'does the club")
        print("  field the same players'. Demeaning by player would remove the club term")
        print("  entirely. Separating them needs players who changed club WITH DefCon")
        print("  measured on both sides - i.e. a second season. 26/27 has 2 gameweeks.")
        print("  The structural route (condition on xGA/press, as `defcon_env` does)")
        print("  stays the right one because it uses a mechanism, not a club label.")

    print("\n" + "=" * 78)
    print("5. WHAT THE UPSTREAM DEFECT DID  (corrected 2026-09-10)")
    print("=" * 78)
    print("  Until 2026-09-10 the panel counted the PUBLISHED column, which for defenders in")
    print("  GW2-10 is CBIRT (recoveries included). Same design, both series:")
    pub = _published(panel)
    for pos in ("Defender", "Midfielder"):
        a = dt.opponent_defcon_ratings(panel, pos)
        b = dt.opponent_defcon_ratings(pub, pos)
        if a.empty or b.empty:
            continue
        j = a.set_index("team")[["shrunk", "hit_shrunk", "category"]].join(
            b.set_index("team")[["shrunk", "hit_shrunk", "category"]], rsuffix="_pub")
        sw = lambda r: r["hit_shrunk"].max() - r["hit_shrunk"].min()
        print(f"\n  {pos}:  {'':12s} {'published':>10s} {'corrected':>10s}")
        print(f"    split-half r   {b['reliability'].iloc[0]:+10.3f} {a['reliability'].iloc[0]:+10.3f}")
        print(f"    EB shrink      {b['shrink_factor'].iloc[0]:10.2f} {a['shrink_factor'].iloc[0]:10.2f}")
        print(f"    hit swing      {sw(b):10.3f} {sw(a):10.3f}")
        print(f"    D1 verdict     {'SHIP' if b['usable'].iloc[0] else 'NO':>10s} "
              f"{'SHIP' if a['usable'].iloc[0] else 'NO':>10s}")
        print(f"    rank corr of shrunk effects, published vs corrected: "
              f"{j['shrunk'].corr(j['shrunk_pub'], method='spearman'):+.3f}; "
              f"categories unchanged for {int((j['category'].astype(str) == j['category_pub'].astype(str)).sum())}/{len(j)} clubs")
        moved = j[j["category"].astype(str) != j["category_pub"].astype(str)]
        if len(moved):
            print("    moved: " + ", ".join(f"{t} {r.category_pub}->{r.category}"
                                         for t, r in moved.iterrows()))
        mb, ma = _matchup_legs(pub, pos), _matchup_legs(panel, pos)
        print(f"    D2 legs        club x opp R2 / match-identity R2 / repeatability r")
        print(f"      published    {mb[0]:.3f} / {mb[1]:.3f} / {mb[2]:+.3f}")
        print(f"      corrected    {ma[0]:.3f} / {ma[1]:.3f} / {ma[2]:+.3f}")

    print("\n  A defect confined to GW2-10 is a TIME effect. Match identity partly encodes the")
    print("  gameweek, so under the published series it absorbs the inflated block; and a")
    print("  club-opponent pair met once inside GW2-10 and once outside it shows opposite")
    print("  residuals in its two meetings. Both legs of the old matchup null were")
    print("  exposed to the defect — compare the two rows above.")

    if keep:
        pd.concat(keep, ignore_index=True).to_csv(OUT, index=False)
        print(f"\n-> {OUT}")


def _matchup_legs(panel, pos):
    """(club x opponent R2, match-identity R2, two-meeting repeatability) on additive
    residuals — the three numbers behind the D2 verdict in section 1."""
    s = panel[panel["position"] == pos].copy()
    s["dc_dm"] = s["dc"] - s.groupby("player_code")["dc"].transform("mean")
    s["resid_add"] = s["dc_dm"] - s.groupby("opp")["dc_dm"].transform("mean")
    real = perm_r2(s, ["club", "opp"], "resid_add", n=1)[0]      # row level, as first
    rm = perm_r2(s, ["match_id"], "resid_add", n=1)[0]           # recorded (audit only)
    s["_pair"] = s["club"].astype(str) + " v " + s["opp"].astype(str)
    cell = s.groupby(["_pair", "match_id"])["resid_add"].mean().reset_index()
    cell["k"] = cell.groupby("_pair").cumcount()
    wide = cell[cell["k"] < 2].pivot(index="_pair", columns="k", values="resid_add").dropna()
    return real, rm, float(wide[0].corr(wide[1]))


def _published(panel):
    """The same panel with `dc`/`hit` rebuilt from the PUBLISHED upstream column — the
    series the study used before 2026-09-10. For the comparison in section 5 only."""
    import glob as _glob
    root = config.repo(SEASON)
    raw = []
    for f in sorted(_glob.glob(_os.path.join(root, "By Gameweek", "GW*", "playermatchstats.csv"))):
        d = pd.read_csv(f, usecols=["player_id", "match_id", "defensive_contributions"])
        p = pd.read_csv(_os.path.join(_os.path.dirname(f), "players.csv"),
                        usecols=["player_id", "player_code"])
        raw.append(d.merge(p, on="player_id", how="left"))
    raw = pd.concat(raw, ignore_index=True).drop_duplicates(["player_code", "match_id"])
    out = panel.drop(columns=["dc", "hit"]).merge(
        raw[["player_code", "match_id", "defensive_contributions"]],
        on=["player_code", "match_id"], how="left")
    out["dc"] = pd.to_numeric(out["defensive_contributions"], errors="coerce")
    out = out[out["dc"].notna()].copy()
    out["hit"] = (out["dc"] >= out["thr"]).astype(float)
    return out


def selftest():
    """Offline. The permutation baseline must expose a fake matchup effect."""
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(20)]
    rows = []
    for p in range(150):
        club = teams[p % 20]
        base = rng.normal(8, 2.0)
        for gw in range(1, 39):
            opp = teams[(p * 7 + gw) % 20]
            if opp == club:
                continue
            rows.append(dict(player_code=p, club=club, opp=opp,
                             dc=base + rng.normal(0, 2.0)))
    s = pd.DataFrame(rows)   # NO opponent or matchup effect at all
    s["dc_dm"] = s["dc"] - s.groupby("player_code")["dc"].transform("mean")
    real_o, base_o, ci_o = perm_r2(s, ["opp"], "dc_dm", n=80)
    real_m, base_m, ci_m = perm_r2(s, ["club", "opp"], "dc_dm", n=80)
    assert real_o <= ci_o[1], f"null opponent effect flagged: {real_o:.4f} > {ci_o[1]:.4f}"
    assert real_m <= ci_m[1], f"null matchup effect flagged: {real_m:.4f} > {ci_m[1]:.4f}"
    assert real_m > 0.05, ("a 380-cell grouping should show a large raw R2 even with no "
                           f"signal, got {real_m:.3f} - the trap must be visible")

    # ...and a PLANTED opponent effect must still clear its null, or the test is vacuous
    eff = {t: v for t, v in zip(teams, np.linspace(-1.5, 1.5, 20))}
    s2 = s.copy()
    s2["dc"] = s2["dc"] + s2["opp"].map(eff)
    s2["dc_dm"] = s2["dc"] - s2.groupby("player_code")["dc"].transform("mean")
    real2, base2, ci2 = perm_r2(s2, ["opp"], "dc_dm", n=80)
    assert real2 > ci2[1], f"planted opponent effect missed: {real2:.4f} vs {ci2[1]:.4f}"

    # ...and a SHARED club-match shock with no matchup effect must fool the row null
    # (the trap the block null exists for) and not the block null.
    rows3 = []
    for ci_, club in enumerate(teams):
        for gw in range(1, 39):
            opp = teams[(ci_ + gw) % 20]
            if opp == club:
                continue
            shock = rng.normal(0, 2.0)
            for k in range(4):
                rows3.append(dict(player_code=ci_ * 4 + k, club=club, opp=opp,
                                  match_id=f"{club}-{gw}", dc=8 + shock + rng.normal(0, 1.0)))
    s3 = pd.DataFrame(rows3)
    s3["dc_dm"] = s3["dc"] - s3.groupby("player_code")["dc"].transform("mean")
    r_row = perm_r2(s3, ["club", "opp"], "dc_dm", n=80)
    r_blk = perm_r2(s3, ["club", "opp"], "dc_dm", n=80, block=BLOCK)
    assert r_row[0] > r_row[2][1], "a shared club-match shock should beat the ROW null"
    assert r_blk[0] <= r_blk[2][1], (
        f"block null flagged a pure club-match shock as a matchup: {r_blk[0]:.3f}")
    print(f"SELFTEST OK: with no planted effect club x opponent raw R2 is {real_m:.3f} "
          f"but sits inside its permutation null [{ci_m[0]:.3f},{ci_m[1]:.3f}] - the "
          f"degrees-of-freedom trap is caught; a planted opponent effect still clears "
          f"its null ({real2:.3f} > {ci2[1]:.3f}); a shared club-match shock beats the "
          f"row null ({r_row[0]:.3f} > {r_row[2][1]:.3f}) but not the block null "
          f"({r_blk[0]:.3f} <= {r_blk[2][1]:.3f}).")


if __name__ == "__main__":
    if "--selftest" in _sys.argv:
        selftest(); _sys.exit(0)
    main()
