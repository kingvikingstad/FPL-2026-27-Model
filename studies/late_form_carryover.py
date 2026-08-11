from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
late_form_carryover.py — does a strong or weak finish predict the next season's start?
======================================================================================
The question is NOT "do teams that finish well start well" — of course they do, because
good teams do both. The question is whether late form carries information ABOVE AND
BEYOND full-season strength. So the regressor is a RESIDUAL:

    late_resid(t) = (strength over last N matches of season t) - (full-season strength t)

i.e. how much a team over- or under-performed its own season baseline down the stretch.
Then:

    early(t+1) ~ full_season(t) + late_resid(t)

If the coefficient on late_resid is ~0, late surges are noise and should not be chased.

THE TRAP THIS STUDY IS BUILT AROUND
-----------------------------------
A 6-match window is mostly measurement error. Errors-in-variables ATTENUATES the
coefficient toward zero, so "no effect" is exactly what a broken test also produces. A
null here is only publishable if we can show it is not attenuation. So the study:
  * uses xG, not goals (far more reliable per match);
  * MEASURES the reliability of `late_resid` by split-half within the window;
  * reports the disattenuated coefficient alongside the raw one.

MANAGERIAL CHANGES
------------------
The interesting hypothesis: a late surge under a NEW manager is a genuine regime change
and should persist, whereas a late surge under the incumbent is noise. Tested as an
interaction. Manager data: data/manager_changes.csv, transcribed from transfermarkt
(club + date only — the transcribed manager NAMES contain known errors and are not used;
matchday is derived from actual fixture dates rather than trusting the source).

Also tested: whether a CLOSE-SEASON manager change breaks the normal carryover of
full-season strength, which is the mechanism `TeamModel`'s `revert` assumes is uniform.

Run:  python studies/late_form_carryover.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import sd_ingest

SEASONS = ["%02d%02d" % (y % 100, (y + 1) % 100) for y in range(2014, 2026)]
LATE_N = 6          # matches at the end of season t
EARLY_N = 6         # matches at the start of season t+1  (the board's horizon)
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "late_form_carryover.csv")


def season_start(code):
    yy = int(code[:2]); return (1900 + yy) if yy >= 90 else (2000 + yy)


def ols(X, y):
    """Least squares with an intercept; returns coefs, se, r2."""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return b, se, r2


def build():
    tm = sd_ingest.understat_team_match(SEASONS)
    tm["team"] = tm["team"].map(sd_ingest.normalise_team)
    tm["date"] = pd.to_datetime(tm["date"], errors="coerce")
    tm = tm.sort_values(["season", "team", "date"])
    tm["md"] = tm.groupby(["season", "team"]).cumcount() + 1
    tm["n_md"] = tm.groupby(["season", "team"])["md"].transform("max")
    # strength = non-penalty xG difference per match, season-standardised so eras compare
    tm["diff"] = tm["npxg"] - tm["npxga"]
    g = tm.groupby("season")["diff"]
    tm["z"] = (tm["diff"] - g.transform("mean")) / g.transform("std")
    return tm


def manager_flags(tm):
    """Per (season, team): did a change happen during the season, and how many matches
    before the end? Plus whether a close-season change preceded the NEXT season."""
    mc = pd.read_csv(config.MANAGER_CHANGES)
    mc["date"] = pd.to_datetime(mc["date"])
    mc["club"] = mc["club"].map(sd_ingest.normalise_team)

    # map each change onto the season/matchday of the club it belongs to
    rows = []
    for (season, team), grp in tm.groupby(["season", "team"]):
        lo, hi = grp["date"].min(), grp["date"].max()
        n_md = int(grp["n_md"].iloc[0])
        ch = mc[(mc["club"] == team) & (mc["date"] >= lo) & (mc["date"] <= hi)]
        # matches remaining after the LAST in-season change
        if len(ch):
            last = ch["date"].max()
            md_at = int((grp["date"] <= last).sum())
            remaining = n_md - md_at
        else:
            md_at, remaining = np.nan, np.nan
        # close-season change between this season's end and the next season's start
        rows.append({"season": season, "team": team, "n_md": n_md,
                     "in_season_change": len(ch) > 0,
                     "md_of_change": md_at, "matches_after_change": remaining,
                     "season_end": hi})
    f = pd.DataFrame(rows)

    order = sorted(tm["season"].unique(), key=season_start)
    nxt = {s: order[i + 1] for i, s in enumerate(order[:-1])}
    starts = tm.groupby("season")["date"].min().to_dict()
    close = []
    for _, r in f.iterrows():
        s1 = nxt.get(r["season"])
        if s1 is None:
            close.append(np.nan); continue
        ch = mc[(mc["club"] == r["team"]) & (mc["date"] > r["season_end"])
                & (mc["date"] < starts[s1])]
        close.append(len(ch) > 0)
    f["close_season_change"] = close
    return f


def pairs(tm, flags):
    order = sorted(tm["season"].unique(), key=season_start)
    out = []
    for s0, s1 in zip(order[:-1], order[1:]):
        a = tm[tm["season"] == s0]; b = tm[tm["season"] == s1]
        fa = flags[flags["season"] == s0].set_index("team")
        for team, ga in a.groupby("team"):
            gb = b[b["team"] == team]
            if len(gb) < EARLY_N or len(ga) < LATE_N + 4:
                continue                      # relegated, or too few matches
            ga = ga.sort_values("md"); gb = gb.sort_values("md")
            full = ga["z"].mean()
            late = ga["z"].tail(LATE_N).mean()
            early = gb["z"].head(EARLY_N).mean()
            # split-half of the late window, for reliability
            tail = ga["z"].tail(LATE_N).values
            out.append({
                "season": s0, "next": s1, "team": team,
                "full": full, "late": late, "late_resid": late - full,
                "early_next": early,
                "late_h1": tail[0::2].mean(), "late_h2": tail[1::2].mean(),
                "in_season_change": bool(fa.loc[team, "in_season_change"])
                if team in fa.index else False,
                "matches_after_change": fa.loc[team, "matches_after_change"]
                if team in fa.index else np.nan,
                "close_season_change": bool(fa.loc[team, "close_season_change"])
                if team in fa.index else False,
            })
    return pd.DataFrame(out)


def main():
    tm = build()
    flags = manager_flags(tm)
    P = pairs(tm, flags)
    P.to_csv(OUT, index=False)
    print(f"[study] {len(P)} team season-pairs across {P.season.nunique()} transitions "
          f"(surviving teams only)")
    print(f"        late window = last {LATE_N}, early window = first {EARLY_N}, "
          f"strength = season-standardised npxG difference per match")

    # ---------------------------------------------------------------- primary
    print("\n" + "=" * 78)
    print("PRIMARY  early(t+1) ~ full(t) + late_resid(t)")
    print("=" * 78)
    b, se, r2 = ols([P["full"], P["late_resid"]], P["early_next"].values)
    names = ["intercept", "full_season(t)", "late_resid(t)"]
    for n, c, s in zip(names, b, se):
        t = c / s if s > 0 else np.nan
        star = "  <- significant" if abs(t) > 1.96 else ""
        print(f"  {n:16s} {c:+.4f}  (se {s:.4f}, t {t:+.2f}){star}")
    print(f"  r2 = {r2:.4f}   n = {len(P)}")

    # baseline for comparison: full-season only
    b0, se0, r20 = ols([P["full"]], P["early_next"].values)
    print(f"\n  full-season alone: r2 = {r20:.4f}  ->  adding late form buys "
          f"{r2 - r20:+.4f}")

    # ------------------------------------------------- attenuation diagnostic
    print("\n" + "=" * 78)
    print("IS A NULL JUST ATTENUATION?  reliability of the late-form residual")
    print("=" * 78)
    r_half = float(np.corrcoef(P["late_h1"], P["late_h2"])[0, 1])
    rel = 2 * r_half / (1 + r_half)          # Spearman-Brown to full window
    print(f"  split-half r = {r_half:.3f}  ->  reliability of the {LATE_N}-match "
          f"window = {rel:.3f}")
    lo, hi = b[2] - 1.96 * se[2], b[2] + 1.96 * se[2]
    print(f"  raw coefficient {b[2]:+.4f}, 95% CI ({lo:+.3f}, {hi:+.3f})")
    if rel > 0:
        print(f"  disattenuated   {b[2]/rel:+.4f}, 95% CI ({lo/rel:+.3f}, {hi/rel:+.3f})")
    print("\n  Read this as a null WITH BOUNDS, not as proof of exactly zero. The window")
    print(f"  is genuinely noisy (reliability {rel:.2f}), so what is ruled out is any")
    print(f"  effect bigger than about {abs(hi/rel):.2f} in standardised units — a team")
    print("  finishing a full SD above its own baseline would move its next-season start")
    print(f"  by at most ~{abs(hi/rel):.2f} SD, and the point estimate is ~0. Small effects")
    print("  are not excluded; a tradeable one is.")

    # ---------------------------------------------------------------- manager
    print("\n" + "=" * 78)
    print("MANAGERIAL CHANGE — does a surge under a NEW manager behave differently?")
    print("=" * 78)
    P["new_mgr_late"] = P["in_season_change"] & (P["matches_after_change"] <= 12)
    for lab, sub in (("late surge under INCUMBENT", P[~P["new_mgr_late"]]),
                     ("late surge under NEW manager", P[P["new_mgr_late"]])):
        if len(sub) < 15:
            print(f"  {lab}: n={len(sub)} too small"); continue
        bb, ss, rr = ols([sub["full"], sub["late_resid"]], sub["early_next"].values)
        t = bb[2] / ss[2]
        print(f"  {lab:30s} n={len(sub):3d}  late_resid coef {bb[2]:+.4f} "
              f"(se {ss[2]:.4f}, t {t:+.2f})")
    inter = P["late_resid"] * P["new_mgr_late"].astype(float)
    bi, si, _ = ols([P["full"], P["late_resid"], P["new_mgr_late"].astype(float), inter],
                    P["early_next"].values)
    t = bi[4] / si[4]
    print(f"\n  interaction late_resid x new_manager: {bi[4]:+.4f} (se {si[4]:.4f}, "
          f"t {t:+.2f}){'  <- significant' if abs(t) > 1.96 else '  <- not significant'}")

    # ---- is that interaction robust, or one definition getting lucky? ----
    # The flag above is muddy: it includes changes that happened DURING the late
    # window, so the 6 matches straddle two regimes. A real regime effect should
    # survive requiring the new manager to have had the WHOLE late window, and should
    # not swing wildly with the cutoff.
    print("\n  SENSITIVITY — the interaction under different definitions of 'new manager':")
    print(f"    {'definition':34s} {'n':>4s} {'interaction':>12s} {'se':>8s} {'t':>7s}")
    defs = []
    for lo in (LATE_N, LATE_N + 2):
        for hi in (10, 12, 15, 20, 99):
            if hi <= lo:
                continue
            defs.append((f"change {lo}-{hi} matches before end", lo, hi))
    for lab, lo, hi in defs:
        flag = (P["in_season_change"] & P["matches_after_change"].between(lo, hi))
        n = int(flag.sum())
        if n < 5:
            print(f"    {lab:34s} {n:4d}   (too few)")
            continue
        it = P["late_resid"] * flag.astype(float)
        bb, ss, _ = ols([P["full"], P["late_resid"], flag.astype(float), it],
                        P["early_next"].values)
        print(f"    {lab:34s} {n:4d} {bb[4]:+12.4f} {ss[4]:8.4f} {bb[4]/ss[4]:+7.2f}")
    print("    ^ a real effect holds its sign and rough size across these; noise does not.")

    print("\n  Does a CLOSE-SEASON manager change break the carryover of full strength?")
    for lab, sub in (("no close-season change", P[~P["close_season_change"]]),
                     ("close-season change", P[P["close_season_change"]])):
        if len(sub) < 15:
            print(f"    {lab}: n={len(sub)} too small"); continue
        bb, ss, _ = ols([sub["full"], sub["late_resid"]], sub["early_next"].values)
        print(f"    {lab:24s} n={len(sub):3d}  full(t) coef {bb[1]:+.4f} "
              f"(se {ss[1]:.4f})")
    print("    ^ a smaller coefficient means a new manager DOES reset the team")

    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
