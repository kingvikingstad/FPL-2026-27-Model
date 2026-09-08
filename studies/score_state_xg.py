import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
score_state_xg.py — should team strength be estimated from xG in a NEUTRAL game state?
=======================================================================================
A side two goals up stops attacking; a side chasing throws bodies forward. Raw match xG
mixes those regimes together, so a team's xG total partly measures the scorelines it
happened to be in rather than how good it is. The fix usually proposed is to estimate
strength only from periods when the game was level, or close.

This is the general form of the question `red_card_matches.py` answered for a special
case. Red cards distort 11% of matches; SCORE state distorts all of them. That study
found the red-card contamination real but four times too small to pay for the 11% of
evidence it cost. The same trade-off applies here and is the whole question: filtering to
level minutes buys a cleaner signal and pays for it in sample size.

PRE-REGISTRATION  [fixed 2026-08-27, before any estimate was produced]
-----------------------------------------------------------------------
PRIMARY (two comparisons against the same baseline, Bonferroni alpha = 0.025)

  H  Team strength estimated from xG in a restricted score state predicts the rest of
     the season better than strength estimated from raw total xG.

  estimators  from a team's FIRST k matches of a season, xG difference per 90:
                RAW    all minutes
                LEVEL  only while the score was level
                CLOSE  only while the lead was within one goal
  target      that team's actual goal difference per match over the REMAINDER
  validation  leave-one-season-out across the 12 cached seasons
  metric      MAE (primary), Spearman (secondary)
  decision    adopt a restricted estimator only if its MAE beats RAW and the
              season-clustered 95% CI on the difference excludes zero

  Note the restricted estimators are NOT handicapped by a matched-size control here, and
  should not be: unlike the red-card test, this is not "drop some matches at random", it
  is two different estimators of the same quantity computed from the same matches. The
  precision cost of using fewer minutes is a real property of the estimator and belongs
  inside the comparison.

SCORE STATE IS RECONSTRUCTED, NOT SUPPLIED
--------------------------------------------
No cached source carries score state. It is rebuilt from goal timings in
`understat_shots`: every shot has a minute, so the goals define a step function and each
shot is attributed to the state in force when it was taken. Minutes per state come from
the same partition.

  OWN GOALS ARE FILED UNDER THE CONCEDING TEAM. Verified, not assumed: reconstructing
  final scores for the 38 own-goal matches in 2025-26 and checking against
  football-data's actual results gives 38/38 correct when the own goal is credited to
  the opponent and 1/38 when it is taken at face value. Getting this backwards would
  corrupt the scoreline in about 10% of matches.

DATA
----
`.cache/soccerdata/understat_shots` — 12 seasons, 116,448 shots, 2014-15..2025-26. Chosen
over the repo's own `xg_by_minute.csv`, which covers 25/26 and 26/27 only; twelve seasons
are needed to hold one out. If this test passes, APPLYING it in-season needs only the
repo's `shots.csv`, which carries minute and outcome for the current season.

RESULT  [2026-08-27] — filtering on score state makes the estimate WORSE
-------------------------------------------------------------------------
116,448 shots, 4,560 matches, 12 seasons. Teams spend 45.6% of match time level and
81.3% within one goal.

  first k matches -> rest-of-season goal difference per match, MAE (Spearman)

     k      raw            level                   close
     5   0.4383 (0.512)  0.4905 (0.359) WORSE   0.4493 (0.505) no difference
     8   0.3973 (0.677)  0.5170 (0.458) WORSE   0.4288 (0.637) WORSE
    12   0.3982 (0.704)  0.4819 (0.562) WORSE   0.4185 (0.684) no difference
    19   0.3885 (0.743)  0.4783 (0.626) WORSE   0.4188 (0.710) WORSE

**0 of 8 restricted estimators beat raw xG.** Six are significantly worse and two are
indistinguishable. Nothing here is close to the pre-registered bar.

WHY, AND IT IS NOT JUST SAMPLE SIZE
-------------------------------------
The obvious explanation is that level state is 45.6% of the time, so the estimator is
built on less than half the evidence. That is part of it — CLOSE keeps 81.3% and loses
much less than LEVEL does, exactly as a precision story predicts.

But precision alone does not explain the SPEARMAN collapse. Rank agreement falls 0.743
to 0.626 at k=19, where sample size is least binding. Noise degrades MAE; it does not
systematically reorder teams. Something is being removed that carried signal.

That something is the state itself. **Being ahead is not a nuisance variable, it is an
outcome of being good.** Strong teams spend more time leading, and they generate a large
share of their xG in exactly the states this filter discards. Conditioning on score state
therefore conditions on a consequence of the quantity being estimated — it strips signal,
not contamination. The cleaner-looking sample is cleaner because the informative part has
been removed.

VERDICT: no model change. `inseason.stack_e0` continues to write RAW match xG.

RELATION TO THE RED-CARD RESULT
---------------------------------
`red_card_matches.py` asked the same question for a special case and found the
contamination real but four times too small to pay for the evidence it cost. This is the
general case, and it goes further: the correction is not merely unprofitable, it is
harmful, because score state is endogenous to team quality in a way a red card is not.

Together they close the "non-replicable game state" line of attack. The premise is
descriptively true — game state does distort xG — and both available corrections make
the model worse. Recorded so the idea is not rediscovered and re-implemented.

Run:  python studies/score_state_xg.py
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd

K_GRID = [5, 8, 12, 19]
MATCH_END = 96.0                 # nominal full time including stoppage
MIN_REST = 8                     # remainder must be long enough to be a target
MIN_STATE_MIN = 200.0            # a team needs this much time in-state to be estimated
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "score_state_xg.csv")


def _yr(s):
    s = str(s)
    return 2000 + int(s[:2]) if int(s[:2]) < 50 else 1900 + int(s[:2])


def load_shots():
    """Every shot with its match, team, minute, xg, and whether it was a goal."""
    frames = []
    for f in sorted(glob.glob(_os.path.join(config.SD_CACHE, "understat_shots", "*.gz"))):
        d = pd.read_csv(f)
        rest = d["game"].astype(str).str.split(" ", n=1).str[1]
        d["date"] = pd.to_datetime(d["game"].astype(str).str.split(" ", n=1).str[0],
                                   errors="coerce")
        d["home"] = rest.str.split("-").str[0]
        d["away"] = rest.str.split("-").str[-1]
        d["is_goal"] = d["result"].isin(["Goal", "Own Goal"])
        # an Own Goal row names the CONCEDING team — see the module docstring
        d["scoring_team"] = np.where(
            d["result"].eq("Own Goal"),
            np.where(d["team"] == d["home"], d["away"], d["home"]),
            d["team"])
        d["yr"] = d["season"].map(_yr)
        frames.append(d[["yr", "match_id", "date", "home", "away", "team",
                         "scoring_team", "minute", "xg", "is_goal", "result"]])
    return pd.concat(frames, ignore_index=True)


def match_states(sub):
    """(intervals, goal_steps) for one match, from the HOME team's perspective.

    intervals: list of (start, end, home_lead). Shots and minutes are attributed to the
    interval containing them.
    """
    goals = sub[sub.is_goal].sort_values("minute")
    steps = []
    hg = ag = 0
    for _, g in goals.iterrows():
        if g["scoring_team"] == g["home"]:
            hg += 1
        else:
            ag += 1
        steps.append((float(g["minute"]), hg - ag))
    intervals, prev, lead = [], 0.0, 0
    for m, ld in steps:
        m = min(max(m, 0.0), MATCH_END)
        if m > prev:
            intervals.append((prev, m, lead))
        prev, lead = m, ld
    if prev < MATCH_END:
        intervals.append((prev, MATCH_END, lead))
    return intervals


def team_match_panel(shots):
    """One row per team-match: xG for/against and minutes, split by score state."""
    rows = []
    for (yr, mid), sub in shots.groupby(["yr", "match_id"], sort=False):
        home = sub["home"].iloc[0]; away = sub["away"].iloc[0]
        iv = match_states(sub)
        if not iv:
            continue
        lead_at = np.zeros(int(MATCH_END) + 2, dtype=int)
        mins = {}
        for a, b, ld in iv:
            for t in range(int(np.floor(a)), min(int(np.ceil(b)), int(MATCH_END) + 1)):
                lead_at[t] = ld
            mins[ld] = mins.get(ld, 0.0) + (b - a)
        rec = {}
        for side, opp in ((home, away), (away, home)):
            sign = 1 if side == home else -1
            m_level = mins.get(0, 0.0)
            m_close = sum(v for k, v in mins.items() if abs(k) <= 1)
            r = {"yr": yr, "match_id": mid, "team": side, "date": sub["date"].iloc[0],
                 "min_level": m_level, "min_close": m_close, "min_all": MATCH_END}
            for tag, keep in (("raw", None), ("level", 0), ("close", 1)):
                f = ag = 0.0
                for _, s in sub.iterrows():
                    t = int(min(max(s["minute"], 0), MATCH_END))
                    ld = lead_at[t] * sign          # lead from THIS team's perspective
                    if keep is not None and abs(ld) > keep:
                        continue
                    if s["team"] == side:
                        f += float(s["xg"])
                    else:
                        ag += float(s["xg"])
                r[f"xgf_{tag}"] = f
                r[f"xga_{tag}"] = ag
            rec[side] = r
        rows.extend(rec.values())
    return pd.DataFrame(rows)


def season_results(shots):
    """Actual goals for/against per team-match, for the target."""
    rows = []
    for (yr, mid), sub in shots.groupby(["yr", "match_id"], sort=False):
        home = sub["home"].iloc[0]; away = sub["away"].iloc[0]
        g = sub[sub.is_goal]
        hg = int((g["scoring_team"] == home).sum())
        ag = int((g["scoring_team"] == away).sum())
        rows.append({"yr": yr, "match_id": mid, "team": home, "gf": hg, "ga": ag})
        rows.append({"yr": yr, "match_id": mid, "team": away, "gf": ag, "ga": hg})
    return pd.DataFrame(rows)


def build(panel, res, k):
    """Per team-season: three estimators from the first k matches, and the target."""
    d = panel.merge(res, on=["yr", "match_id", "team"], how="left")
    d = d.sort_values(["yr", "team", "date"])
    rows = []
    for (yr, team), g in d.groupby(["yr", "team"]):
        if len(g) < k + MIN_REST:
            continue
        a, b = g.iloc[:k], g.iloc[k:]
        if a["min_level"].sum() < MIN_STATE_MIN:
            continue
        rows.append({
            "yr": yr, "team": team,
            "raw": (a["xgf_raw"].sum() - a["xga_raw"].sum()) / (a["min_all"].sum() / 90.0),
            "level": (a["xgf_level"].sum() - a["xga_level"].sum())
                     / max(a["min_level"].sum() / 90.0, 1e-9),
            "close": (a["xgf_close"].sum() - a["xga_close"].sum())
                     / max(a["min_close"].sum() / 90.0, 1e-9),
            "level_share": a["min_level"].sum() / a["min_all"].sum(),
            "target": float((b["gf"] - b["ga"]).mean()),
        })
    return pd.DataFrame(rows)


def evaluate(D):
    """Leave-one-season-out linear fit of the target on each estimator."""
    out = {}
    for est in ("raw", "level", "close"):
        preds, tgts, yrs = [], [], []
        for yr in sorted(D.yr.unique()):
            tr, te = D[D.yr != yr], D[D.yr == yr]
            if len(tr) < 40 or len(te) < 8:
                continue
            X = np.column_stack([np.ones(len(tr)), tr[est].values])
            beta = np.linalg.pinv(X.T @ X) @ X.T @ tr["target"].values
            Xe = np.column_stack([np.ones(len(te)), te[est].values])
            preds.append(Xe @ beta); tgts.append(te["target"].values)
            yrs.append(np.full(len(te), yr))
        if not tgts:
            continue
        out[est] = (np.concatenate(preds), np.concatenate(tgts), np.concatenate(yrs))
    return out


def main():
    print("[data] reconstructing score state from goal timings ...")
    shots = load_shots()
    print(f"[data] {len(shots):,} shots, {shots.match_id.nunique():,} matches, "
          f"{shots.yr.nunique()} seasons")
    panel = team_match_panel(shots)
    res = season_results(shots)
    lvl = panel["min_level"].sum() / panel["min_all"].sum()
    cls = panel["min_close"].sum() / panel["min_all"].sum()
    print(f"[data] share of match time spent LEVEL {lvl:.1%}, within one goal {cls:.1%}")

    from scipy import stats as st
    rows = []
    for k in K_GRID:
        D = build(panel, res, k)
        if len(D) < 60:
            print(f"\n  k={k}: insufficient data ({len(D)})")
            continue
        ev = evaluate(D)
        if "raw" not in ev:
            continue
        print("\n" + "=" * 78)
        print(f"FIRST {k} MATCHES -> rest-of-season goal difference per match   "
              f"n={len(D)} team-seasons")
        print("=" * 78)
        p0, y0, yr0 = ev["raw"]
        mae0 = float(np.abs(p0 - y0).mean())
        print(f"  {'estimator':10s}{'MAE':>9s}{'Spearman':>11s}{'vs raw':>10s}"
              f"{'95% CI':>24s}")
        print(f"  {'raw':10s}{mae0:>9.4f}{st.spearmanr(p0, y0)[0]:>11.4f}"
              f"{'-':>10s}{'-':>24s}")
        for est in ("level", "close"):
            if est not in ev:
                continue
            p, y, yr = ev[est]
            mae = float(np.abs(p - y).mean())
            per = pd.DataFrame({"d": np.abs(p - y) - np.abs(p0 - y0), "yr": yr}) \
                    .groupby("yr")["d"].mean()
            m = float(per.mean()); se = float(per.std(ddof=1) / np.sqrt(len(per)))
            lo, hi = m - 1.96 * se, m + 1.96 * se
            flag = ("BETTER" if hi < 0 else "WORSE" if lo > 0 else "no difference")
            print(f"  {est:10s}{mae:>9.4f}{st.spearmanr(p, y)[0]:>11.4f}{m:>+10.4f}"
                  f"   ({lo:+.4f}, {hi:+.4f})  {flag}")
            rows.append({"k": k, "estimator": est, "n": len(D), "mae_raw": mae0,
                         "mae": mae, "diff": m, "lo": lo, "hi": hi, "verdict": flag})
    if rows:
        R = pd.DataFrame(rows)
        R.to_csv(OUT, index=False)
        better = R[R.verdict == "BETTER"]
        print("\n" + "=" * 78)
        print(f"VERDICT: {len(better)}/{len(R)} restricted estimators beat raw xG")
        if len(better):
            print(f"  {sorted(set(better.estimator))} at k={sorted(set(better.k))}")
        else:
            print("  raw total xG stands; score-state filtering does not pay")
        print("=" * 78)
        print(f"[wrote] {OUT}")
    return rows


if __name__ == "__main__":
    main()
