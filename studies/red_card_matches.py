import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
red_card_matches.py — should matches containing a red card be dropped from rate priors?
========================================================================================
The argument for dropping them: a sending-off creates a game state (11 v 10) that will
not recur, so attacking and creative rates measured in those matches are contaminated and
should not inform a player's prior.

The argument against: dropping them costs 11.0% of all minutes, and a player's rate is
already a heavily pooled two-season quantity in which one match is roughly 3% of exposure.

Both are plausible. This measures it.

THE DESIGN POINT THAT MAKES OR BREAKS THIS TEST
------------------------------------------------
Excluding red-card matches removes information as well as contamination. A naive
comparison of "all matches" against "red-card matches excluded" therefore confounds two
things, and would mostly measure that less data is worse. Any such test is rigged toward
the null regardless of whether the contamination is real.

So the comparison is against a MATCHED-SIZE RANDOM CONTROL: drop exactly as many matches
per player, chosen at random. If red-card matches carry a distortion worth removing,
dropping them must beat dropping the same number of arbitrary ones. Sample size is then
identical by construction and the only difference is WHICH matches went.

PRE-REGISTRATION  [fixed 2026-08-26, before any estimate was produced]
-----------------------------------------------------------------------
H (PRIMARY, one test per channel, two channels, Bonferroni alpha = 0.025)

  Estimating a player's rate with red-card matches EXCLUDED predicts his rate in
  held-out clean matches better than excluding an equal number of matches at random.

  design      per player-season, split his matches at random into halves
                estimate  half A, three ways: all matches / reds dropped /
                          random-equal-number dropped
                target    half B, red-card matches ALWAYS excluded, so the thing being
                          predicted is uncontaminated by construction
  channels    npxG/90 and xA/90, penalties stripped
  metric      MAE against the clean target, averaged over many random splits
  decision    recommend dropping red-card matches only if `reds dropped` beats
              `random dropped` and the split-clustered 95% CI on the difference
              excludes zero
  context     also report `all matches` — if it beats both, the exclusion costs more in
              precision than it buys in cleanliness, whatever the contamination

SUPPORTING (descriptive, does not decide): a within-player comparison of rates in
red-card matches against the same player's other matches. If those are indistinguishable
the primary test has nothing to find, and that is worth knowing separately.

DATA
----
`.cache/soccerdata/understat_player` — player-MATCH rows, which is the only cached source
carrying `red_cards` per match. 2 seasons (2024-25, 2025-26), 23,057 player-matches, 85 of
760 matches containing at least one red card (11.2%). A match is flagged if EITHER side
was reduced, since the game state is distorted for both.

RESULT  [2026-08-26] — the contamination is REAL and four times too small to act on
------------------------------------------------------------------------------------
23,057 player-matches, 2 seasons; 85 of 760 matches contain a red card (11.2%), carrying
11.0% of all minutes. 120 random splits, ~689 players each.

SUPPORTING first, because it frames everything: rates in red-card matches are
indistinguishable from the same player's other matches.

    npxG/90   0.1335 in red-card matches vs 0.1341 otherwise   paired p = 0.924
    xA/90     0.1063                  vs 0.1027               paired p = 0.510

PRIMARY, matched sample size (clean target):

                            npxG/90     xA/90
    all matches kept        0.06790    0.06140
    red-card matches out    0.06964    0.06300
    random equal number     0.07013    0.06324
    reds minus random      -0.00049   -0.00024
    95% CI            (-.00071,-.00026)  (-.00042,-.00006)

So dropping red-card matches DOES beat dropping the same number at random, in both
channels, with CIs excluding zero. There is a real distributional difference. The
pre-registered primary rule, read alone, says "drop".

It is the wrong thing to read alone, and the design said so in advance. Dropping reds
costs +0.00174 MAE against keeping everything, while the targeting gains only 0.00049 —
**the benefit is 28% of the cost.** Removing 11% of the evidence to clean a distortion
this small makes the estimate worse, not better. Against the decision-relevant target
(all of half B, since 11% of a player's NEXT matches will also contain a red card) the
same holds: +0.00194 and +0.00175 worse, and for xA the targeting effect is no longer
distinguishable from zero at all (CI -0.00032 to +0.00005).

VERDICT: KEEP red-card matches. The mechanism exists; acting on it is net negative.

TWO NOTES ON WHAT THIS DOES AND DOES NOT SAY
---------------------------------------------
* It does NOT say game state is irrelevant. Red cards distort 11% of matches; SCORE
  state distorts all of them, and the model ignores that too. If non-replicable game
  state is the worry, red cards are the visible special case, not the important one.
* `project()` does not model cards as a SCORING event either — `RED_PTS`/`YELLOW_PTS`
  live in `fpl_xp_model`, which the Bayesian board does not use. So there was never
  anything to remove on that side, only on the rate-estimation side tested here.

Run:  python studies/red_card_matches.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

SEASONS = ("2425", "2526")
MIN_MATCHES = 12          # need enough matches to split and still estimate
MIN_HALF_N90 = 3.0        # a half must carry real minutes
N_SPLITS = 120
SEED = 11
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "red_card_matches.csv")


def load():
    """Player-match rows with penalties stripped and red-card matches flagged."""
    frames = []
    for s in SEASONS:
        f = _os.path.join(config.SD_CACHE, "understat_player", f"{s}.csv.gz")
        if not _os.path.exists(f):
            continue
        d = pd.read_csv(f)
        sf = _os.path.join(config.SD_CACHE, "understat_shots", f"{s}.csv.gz")
        if _os.path.exists(sf):
            sh = pd.read_csv(sf)
            pen = (sh[sh["situation"] == "Penalty"]
                   .groupby(["match_id", "understat_player_id"], as_index=False)
                   .agg(pen_shots=("xg", "size"), pen_xg=("xg", "sum")))
            d = d.merge(pen, on=["match_id", "understat_player_id"], how="left")
        for c in ("pen_shots", "pen_xg"):
            if c not in d.columns:
                d[c] = 0.0
        d[["pen_shots", "pen_xg"]] = d[["pen_shots", "pen_xg"]].fillna(0.0)
        d["np_xg"] = (d["xg"] - d["pen_xg"]).clip(lower=0)
        # a match is contaminated if EITHER side went down to ten
        reds = d.groupby("match_id")["red_cards"].sum()
        d["red_match"] = d["match_id"].map(reds).fillna(0) > 0
        frames.append(d[["season", "match_id", "understat_player_id", "player_name",
                         "minutes", "np_xg", "xa", "key_passes", "red_cards",
                         "red_match"]])
    d = pd.concat(frames, ignore_index=True)
    return d[d["minutes"] > 0].copy()


def descriptive(d):
    """Within player-season: rates in red-card matches vs the same player's others."""
    rows = []
    for (s, pid), g in d.groupby(["season", "understat_player_id"]):
        a = g[g.red_match]
        b = g[~g.red_match]
        if a["minutes"].sum() < 90 or b["minutes"].sum() < 450:
            continue
        rows.append({
            "season": s, "pid": pid,
            "n90_red": a["minutes"].sum() / 90.0, "n90_other": b["minutes"].sum() / 90.0,
            "npxg90_red": a["np_xg"].sum() / (a["minutes"].sum() / 90.0),
            "npxg90_other": b["np_xg"].sum() / (b["minutes"].sum() / 90.0),
            "xa90_red": a["xa"].sum() / (a["minutes"].sum() / 90.0),
            "xa90_other": b["xa"].sum() / (b["minutes"].sum() / 90.0),
        })
    R = pd.DataFrame(rows)
    if not len(R):
        return R
    from scipy import stats as st
    print(f"\n  n={len(R)} player-seasons with >=1 full match of red-card exposure "
          f"and >=5 matches without")
    for ch, lab in (("npxg90", "npxG/90"), ("xa90", "xA/90")):
        diff = R[f"{ch}_red"] - R[f"{ch}_other"]
        t = st.ttest_rel(R[f"{ch}_red"], R[f"{ch}_other"])
        print(f"    {lab:8s} in red-card matches {R[f'{ch}_red'].mean():.4f}  "
              f"vs other {R[f'{ch}_other'].mean():.4f}   "
              f"paired diff {diff.mean():+.4f}  t={t.statistic:+.2f}  p={t.pvalue:.3f}")
    return R


def experiment(d, rng):
    """One random half-split; returns per-player MAE contributions for each estimator."""
    rows = []
    for (s, pid), g in d.groupby(["season", "understat_player_id"]):
        if len(g) < MIN_MATCHES:
            continue
        g = g.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        h = len(g) // 2
        A, B = g.iloc[:h], g.iloc[h:]
        # TWO targets, because they answer different questions and only one of them is
        # the FPL question.
        #   CLEAN target  = half B with red-card matches removed. Asks "what is this
        #                   player's rate in an 11-v-11 match".
        #   ALL target    = half B entire. Asks "what will he do in his NEXT match",
        #                   which is the quantity the model projects — and 11% of next
        #                   matches will themselves contain a red card.
        # Scoring only against the clean target quietly rigs the test: the reds-dropped
        # estimator is then drawn from the same population as the target and the control
        # is not, so it wins on distribution matching alone, contamination or no.
        Bc = B[~B.red_match]
        if Bc["minutes"].sum() / 90.0 < MIN_HALF_N90 or B["minutes"].sum() / 90.0 < MIN_HALF_N90:
            continue
        n90c = Bc["minutes"].sum() / 90.0
        n90b = B["minutes"].sum() / 90.0
        tgt_x = Bc["np_xg"].sum() / n90c
        tgt_a = Bc["xa"].sum() / n90c
        tgt_x_all = B["np_xg"].sum() / n90b
        tgt_a_all = B["xa"].sum() / n90b

        A_nored = A[~A.red_match]
        n_drop = len(A) - len(A_nored)
        if A_nored["minutes"].sum() / 90.0 < MIN_HALF_N90:
            continue
        # matched-size random control: drop the SAME NUMBER of matches, at random
        if n_drop > 0:
            keep = rng.permutation(len(A))[: len(A) - n_drop]
            A_rand = A.iloc[np.sort(keep)]
        else:
            A_rand = A
        if A_rand["minutes"].sum() / 90.0 < MIN_HALF_N90:
            continue

        def rate(fr, col):
            n = fr["minutes"].sum() / 90.0
            return fr[col].sum() / n if n > 0 else np.nan

        rows.append({
            "pid": pid, "season": s, "n_drop": n_drop,
            "x_all": rate(A, "np_xg"), "x_nored": rate(A_nored, "np_xg"),
            "x_rand": rate(A_rand, "np_xg"), "x_tgt": tgt_x,
            "a_all": rate(A, "xa"), "a_nored": rate(A_nored, "xa"),
            "a_rand": rate(A_rand, "xa"), "a_tgt": tgt_a,
            "x_tgtall": tgt_x_all, "a_tgtall": tgt_a_all,
        })
    return pd.DataFrame(rows)


def main():
    d = load()
    n_m = d.groupby(["season", "match_id"]).red_match.first()
    print(f"[data] {len(d):,} player-matches over {d.season.nunique()} seasons; "
          f"{int(n_m.sum())} of {len(n_m)} matches contain a red card "
          f"({n_m.mean():.1%})")
    print(f"[data] minutes in red-card matches: "
          f"{d.loc[d.red_match,'minutes'].sum()/d['minutes'].sum():.1%}")

    print("\n" + "=" * 78)
    print("SUPPORTING — are rates actually different in red-card matches?")
    print("=" * 78)
    descriptive(d)

    print("\n" + "=" * 78)
    print("PRIMARY — matched-size comparison (Bonferroni alpha = 0.025)")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    per_split = []
    for i in range(N_SPLITS):
        e = experiment(d, rng)
        if not len(e):
            continue
        row = {"split": i, "n": len(e)}
        for pre, lab in (("x", "npxg"), ("a", "xa")):
            for est in ("all", "nored", "rand"):
                row[f"{lab}_{est}"] = float(np.mean(np.abs(e[f"{pre}_{est}"]
                                                           - e[f"{pre}_tgt"])))
                row[f"{lab}_{est}_ALL"] = float(np.mean(np.abs(e[f"{pre}_{est}"]
                                                              - e[f"{pre}_tgtall"])))
        per_split.append(row)
    S = pd.DataFrame(per_split)
    if not len(S):
        raise SystemExit("no usable splits")

    out = []
    for lab, name in (("npxg", "npxG/90"), ("xa", "xA/90")):
        a_, n_, r_ = S[f"{lab}_all"], S[f"{lab}_nored"], S[f"{lab}_rand"]
        diff = n_ - r_                       # negative => dropping reds is better
        m = float(diff.mean())
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        lo, hi = m - 1.96 * se, m + 1.96 * se
        print(f"\n  {name}   {len(S)} random splits, ~{int(S.n.mean())} players each")
        print(f"    all matches kept          MAE {a_.mean():.5f}")
        print(f"    red-card matches dropped  MAE {n_.mean():.5f}")
        print(f"    random equal number       MAE {r_.mean():.5f}   <- the fair control")
        print(f"    reds-dropped minus random {m:+.5f}   95% CI ({lo:+.5f}, {hi:+.5f})")
        # TWO-STAGE. Stage 1 asks whether red-card matches are specially contaminated,
        # holding sample size fixed. Stage 2 asks whether acting on that is worth the
        # 11% of evidence it costs. Only both together are a recommendation, and
        # reporting stage 1 alone would read as one.
        cost = float(n_.mean() - a_.mean())
        targeted = ("yes" if hi < 0 else "reversed" if lo > 0 else "no")
        worth_it = cost < 0
        verdict = ("DROP" if (hi < 0 and worth_it) else "KEEP")
        print(f"    -> contamination specific to red-card matches: {targeted}")
        print(f"    -> RECOMMENDATION: {verdict} red-card matches")
        print(f"    cost of the exclusion against keeping everything: {cost:+.5f} MAE")
        # the decision-relevant target: predict the NEXT match, red cards and all
        a2, n2, r2 = S[f"{lab}_all_ALL"], S[f"{lab}_nored_ALL"], S[f"{lab}_rand_ALL"]
        d2 = n2 - r2
        m2 = float(d2.mean()); se2 = float(d2.std(ddof=1) / np.sqrt(len(d2)))
        lo2, hi2 = m2 - 1.96 * se2, m2 + 1.96 * se2
        print(f"    -- against the ALL-matches target (the FPL question) --")
        print(f"    all matches kept          MAE {a2.mean():.5f}")
        print(f"    red-card matches dropped  MAE {n2.mean():.5f}")
        print(f"    random equal number       MAE {r2.mean():.5f}")
        print(f"    reds-dropped minus random {m2:+.5f}   95% CI ({lo2:+.5f}, {hi2:+.5f})")
        v2 = ("targeting reds helps" if hi2 < 0 else
              "targeting reds HURTS" if lo2 > 0 else "no difference")
        print(f"    -> {v2}; keeping everything vs dropping reds: "
              f"{float(n2.mean() - a2.mean()):+.5f} MAE")
        out.append({"channel": name, "splits": len(S), "mae_all": float(a_.mean()),
                    "mae_nored": float(n_.mean()), "mae_rand": float(r_.mean()),
                    "diff": m, "lo": lo, "hi": hi, "verdict": verdict,
                    "cost_vs_all": cost,
                    "mae_all_ALLtgt": float(a2.mean()),
                    "mae_nored_ALLtgt": float(n2.mean()),
                    "mae_rand_ALLtgt": float(r2.mean()),
                    "diff_ALLtgt": m2, "lo_ALLtgt": lo2, "hi_ALLtgt": hi2})
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(f"\n[wrote] {OUT}")
    return out


if __name__ == "__main__":
    main()
