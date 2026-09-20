from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
start_persistence_followup.py — the three tests `start_persistence.py` owed
===========================================================================
`studies/start_persistence.py` reported three things it could not support, and the
2026-09-16 stats-referee review named each. This file tests them. It adds no new data:
same loader, same within-player-season permutation null, same scopes.

  T1  THE NON-START CURVE HAS NO NULL. The study printed P(start | k consecutive
      non-starts) = 0.290 / 0.126 / 0.054 at k = 1/3/8 and called being dropped "far
      more informative than being picked" `[VERIFIED]`. That was raw — exactly the
      frailty-confounded object the study's own introduction warns against, since a long
      non-start run selects players who rarely start. The claim was withdrawn. T1 runs the
      permutation null for the NON-START state and asks whether the asymmetry survives it.

  T2  z WAS OVERSTATED. `z = (h_obs - h_null) / sd(h_null)` uses only the permutation
      spread and ignores the sampling variance of `h_obs`. T2 replaces z with a cluster
      bootstrap over `player_code` in which the null is RECOMPUTED inside every resample,
      so the interval covers both sources.

  T3  IS ONE LAG ENOUGH? The within-player lag-1 gap (+0.436) is persistence, not
      identified state dependence: within-season drift in a player's role inflates it. A
      first-order Markov chain in the start indicator predicts NO effect of the state two
      matches back once the last match is known; slow drift predicts one. T3 measures the
      lag-2-given-lag-1 contrast, net of the same permutation null.

      What T3 can and cannot do: a non-zero lag-2 contrast rules out first-order Markov as
      a sufficient description. It does NOT separate drift from genuine higher-order state
      dependence — both predict it — so neither outcome identifies "trust". It is an
      estimator-design test: if one lag suffices, a two-parameter Markov start model would
      do and long memory is unnecessary; if it does not, memory beyond the last match is
      real, which is the premise `start_recency`/the forgetting filter rest on.

      INSTRUMENT CORRECTED before any real-data number was read. T3 was first written
      against the permutation null, like T1. A synthetic calibration check (`calibrate`,
      run below on every invocation) showed that is invalid: on a first-order chain with
      p(stay)=0.9 the OBSERVED lag-2 contrast is ~0.00, as theory requires, but the
      permutation null sits at +0.157, because permuting destroys first-order dependence
      as well as higher-order structure and the pooled contrast then picks up count
      heterogeneity. Net-of-permutation would have declared first-order-by-construction
      data "first-order inadequate". T3 therefore uses a MARKOV null that preserves what
      is being conditioned on: each player-season's own (p01, p11), Jeffreys-smoothed
      (a=0.5), simulated from his observed first state for his own length. The permutation
      null stays for T1/T2, where holding the count and destroying all order IS the
      comparator the question asks for. No real-data lag-2 statistic had been computed
      when this change was made; the calibration used simulated sequences only.

PRE-REGISTERED — WRITTEN AND COMMITTED BEFORE THE STUDY WAS RUN
---------------------------------------------------------------
UNIT      player-match, native `starts`, 22/23 from GW16 (fpl_history.empty_native_gws),
          ordered by kickoff; streak resets each season. PRIMARY scope `native`;
          `proxy` (mins>=60, 16/17-21/22) is a REPLICATION; `nogap` is a declared
          sensitivity and is non-decision-bearing (it conditions on the outcome path).

NULL      within-player-season permutation of the outcome sequence: each player-season's
          start COUNT is held exactly, only the order is destroyed. The outcome is
          permuted, not labels (project guard). It conditions on the full-season count,
          which is hindsight; every statement below is therefore "beyond the player's own
          season rate", never "beyond a forecaster's prior".

ESTIMANDS on the primary scope, each reported as observed, null mean, and net = obs - null
  E1  h_S(k) = P(start at t+1 | exactly k consecutive starts through t)
  E2  h_N(k) = P(start at t+1 | exactly k consecutive non-starts through t)
  E3  LO_S(k), LO_N(k): the same nets on the LOG-ODDS scale, logit(obs) - logit(null).
      Log-odds is the DECISION scale for the asymmetry, because h_S ~ 0.8 and h_N ~ 0.1
      and an absolute difference between them is a statement about where they sit, not
      about how much each run tells you.
  E4  ASYM(k) = (-LO_N(k)) - LO_S(k).  Positive = a run of non-starts moves the
      probability further, in log-odds, than a run of starts of the same length.
  E5  D1 = P(start | started last) - P(start | did not), pooled over transitions, net of
      null. Pooled, not the study's within-player mean, so it is not restricted to the
      rotation-zone player-seasons that mean required (>=5 of each state).
  E6  L2_S = P(y | S_{t-1}=1, S_t=1) - P(y | S_{t-1}=0, S_t=1), net of the MARKOV null.
      L2_N = P(y | S_{t-1}=1, S_t=0) - P(y | S_{t-1}=0, S_t=0), net of the MARKOV null.
      (E1-E5 use the permutation null; only E6 uses the Markov null, for the reason in T3.)

CALIBRATION, a gate on T3 and not a result: the Markov null must recover ~0 on synthetic
  first-order sequences and must NOT mask a real second-order effect. `calibrate()` runs
  both before the real data is touched and prints them. Pass band: |net L2| <= 0.02 on the
  first-order panel AND net L2 >= 0.05 on a second-order panel (p(stay)=0.95 after two
  like states, 0.70 otherwise). If either fails, T3 reports INSTRUMENT NOT CALIBRATED and
  makes no claim.

INTERVALS 95% cluster bootstrap over `player_code` (a player's matches are not
          independent), N_BOOT resamples, with the permutation null recomputed from
          N_PERM_INNER draws inside each resample. Point estimates use N_PERM draws.
          Compute budget, not a decision rule: N_PERM=150, N_BOOT=150, N_PERM_INNER=8,
          and for the costlier first-order conditional null N_MARKOV=100 point / 2 inner.
          Both nulls are expectations whose MC error is small next to the bootstrap spread.

DECISION RULES, fixed here before any number was read
  R1 (T1/E4, the asymmetry claim)  SUPPORTED if ASYM(k) >= 0.10 log-odds with a 95% CI
     excluding 0 at BOTH k=3 and k=6. REVERSED if ASYM(k) <= -0.10 with CI excluding 0 at
     both. Otherwise NOT SHOWN, and the withdrawn claim stays withdrawn. 0.10 in log-odds
     is about a 10% relative change in the odds — the smallest asymmetry that would change
     how a non-start run is read against a start run.
  R2 (T1, is the non-start curve informative at all)  INFORMATIVE if the net h_N(k) is
     <= -0.02 with CI excluding 0 for at least k = 1, 2 and 3. Otherwise the curve is
     frailty and selection only.
  R3 (T3/E6, against the MARKOV null)  FIRST-ORDER INADEQUATE if |net L2_S| >= 0.02 with
     CI excluding 0, or the same for L2_N. FIRST-ORDER ADEQUATE if both CIs lie entirely
     inside (-0.02, +0.02) — an equivalence claim, not a failure to reject. Otherwise
     INCONCLUSIVE. 0.02 is the same margin the parent study used to call an excess
     decision-relevant. Conditional on the calibration gate above passing.
  A result that fails its rule is recorded as a null in the CSV and in
  docs/START_PERSISTENCE_2026-09-07.md. Nothing here turns a flag on; `INSEASON_LAM`
  stays off either way, and no streak covariate is built whatever T3 says (this module's
  standing rule against streak/momentum predictors is not up for revision here).

POWER, stated before the run: at native k=3 the start cell holds ~3.1k transitions and the
  non-start cell ~4.9k, so a 0.10 log-odds difference is ~4x the cluster-bootstrap spread
  seen in the parent study at those cells; k=6 is thinner (~1.6k / ~3.4k) and is the
  binding cell for R1. D1 and the lag-2 contrasts run on ~100k transitions, where the
  0.02 margin is far above the noise, so R3 can return a genuine equivalence rather than
  "not significant". If a CI at k=6 turns out wider than 0.10, R1 returns NOT SHOWN for
  lack of power and that is the honest outcome, not a reason to move the margin.

Run:  python studies/start_persistence_followup.py   (~17 min over three scopes; the
      first-order conditional null is the cost, and it is the slowest study in the suite)
Out:  studies/start_persistence_followup.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import start_persistence as sp

KS = (1, 2, 3, 4, 5, 6, 8, 10)
KMAX = sp.KMAX
N_PERM = 150                  # permutation draws for the point estimate
N_BOOT = 150                  # cluster-bootstrap resamples
N_PERM_INNER = 8              # permutation draws inside each resample
N_MARKOV = 100                # first-order conditional draws for the point estimate
N_MARKOV_INNER = 2            # ...and inside each resample (this null is the expensive one)
MIN_CELL = 100
MARGIN_ASYM = 0.10            # log-odds, R1
MARGIN_NET = 0.02             # probability, R2 and R3
OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                    "start_persistence_followup.csv")
rng = np.random.default_rng(20260917)


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def lag_counts(v, seg):
    """P(y | state two back, state one back) — numerator and denominator for the four
    combinations, over positions with BOTH a predecessor and a successor in the same
    player-season. Index = 2*prev + cur."""
    n = len(v)
    same_prev = np.zeros(n, dtype=bool); same_prev[1:] = seg[1:] == seg[:-1]
    same_next = np.zeros(n, dtype=bool); same_next[:-1] = seg[1:] == seg[:-1]
    valid = same_prev & same_next
    prev = np.zeros(n, dtype=np.int64); prev[1:] = v[:-1]
    nxt = np.zeros(n, dtype=float); nxt[:-1] = v[1:]
    code = 2 * prev + v.astype(np.int64)
    den = np.bincount(code[valid], minlength=4).astype(float)[:4]
    num = np.bincount(code[valid], weights=nxt[valid], minlength=4)[:4]
    return num, den


def lag1_counts(v, seg):
    """P(y | state one back), for E5."""
    n = len(v)
    same_next = np.zeros(n, dtype=bool); same_next[:-1] = seg[1:] == seg[:-1]
    nxt = np.zeros(n, dtype=float); nxt[:-1] = v[1:]
    cur = v.astype(np.int64)
    den = np.bincount(cur[same_next], minlength=2).astype(float)[:2]
    num = np.bincount(cur[same_next], weights=nxt[same_next], minlength=2)[:2]
    return num, den


def runs_of(v):
    """Run-length encoding of one binary sequence: (states, lengths)."""
    x = np.asarray(v, dtype=np.int8)
    cut = np.flatnonzero(np.diff(x)) + 1
    starts = np.concatenate(([0], cut))
    lengths = np.diff(np.concatenate((starts, [len(x)])))
    return x[starts], lengths


def markov_spec(arrs):
    """Per sequence: the alternating run STATES, and how many units each state has to
    distribute over its runs. These are the first-order chain's sufficient statistics —
    the transition counts n00/n01/n10/n11 are fixed by (states pattern, totals)."""
    spec = []
    for v in arrs:
        st, ln = runs_of(v)
        tot = {s: int(ln[st == s].sum()) for s in (0, 1)}
        cnt = {s: int((st == s).sum()) for s in (0, 1)}
        spec.append((st, tot, cnt))
    return spec


def _composition(n, r, rng_):
    """Uniform random composition of n into r positive parts (stars and bars)."""
    if r <= 1:
        return np.array([n], dtype=np.int64)
    cuts = np.sort(rng_.choice(n - 1, r - 1, replace=False)) + 1
    return np.diff(np.concatenate(([0], cuts, [n])))


def markov_draw(spec, rng_):
    """One draw from the FIRST-ORDER CONDITIONAL null: each sequence keeps its own
    transition counts exactly (same run states, same totals per state) and only the run
    LENGTHS are re-drawn uniformly. No parameters are estimated, so the null injects no
    heterogeneity of its own — which is what disqualified the parametric version."""
    out, seg = [], []
    for i, (st, tot, cnt) in enumerate(spec):
        parts = {s: _composition(tot[s], cnt[s], rng_) for s in (0, 1) if cnt[s]}
        idx = {0: 0, 1: 0}
        ln = np.empty(len(st), dtype=np.int64)
        for j, s in enumerate(st):
            ln[j] = parts[int(s)][idx[int(s)]]
            idx[int(s)] += 1
        v = np.repeat(st, ln)
        out.append(v)
        seg.append(np.full(len(v), i))
    return np.concatenate(out).astype(np.int8), np.concatenate(seg)


def markov_null_L2(arrs, n_sim, rng_=None):
    """Mean lag-2 contrasts under the first-order conditional null — what a one-lag chain
    with each player's own transition counts implies."""
    rng_ = rng_ or rng
    spec = markov_spec(arrs)
    accS = accN = 0.0
    for _ in range(n_sim):
        v, seg = markov_draw(spec, rng_)
        num, den = lag_counts(v, seg)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where(den > 0, num / den, np.nan)
        accS += q[3] - q[1]
        accN += q[2] - q[0]
    return accS / n_sim, accN / n_sim


def statistics(v, seg):
    """Every observed statistic in one pass-set: hazards by streak length for both
    states, the lag-1 gap, and the two lag-2 contrasts."""
    ns, ds, nn, dn = sp.profile_flat(v, seg, KMAX)
    out = {"h_S": sp._h(ns, ds), "n_S": ds, "h_N": sp._h(nn, dn), "n_N": dn}
    num, den = lag1_counts(v, seg)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(den > 0, num / den, np.nan)
    out["D1"] = p[1] - p[0]
    num2, den2 = lag_counts(v, seg)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(den2 > 0, num2 / den2, np.nan)
    out["L2_S"] = q[3] - q[1]           # cur = start; prev start vs prev non-start
    out["L2_N"] = q[2] - q[0]           # cur = non-start; prev start vs prev non-start
    out["n_L2_S"] = den2[3] + den2[1]
    out["n_L2_N"] = den2[2] + den2[0]
    return out


def null_mean(v, seg, n_perm):
    """Mean of every statistic over `n_perm` within-player-season permutations."""
    acc = None
    for _ in range(n_perm):
        s = statistics(sp.permute_flat(v, seg), seg)
        if acc is None:
            acc = {k: np.asarray(val, dtype=float) * 0.0 for k, val in s.items()}
        for k, val in s.items():
            acc[k] = acc[k] + np.asarray(val, dtype=float)
    return {k: val / n_perm for k, val in acc.items()}


def nets(obs, nul, markov):
    """Every net quantity the decision rules read. Hazards and the lag-1 gap are net of
    the PERMUTATION null; the lag-2 contrasts are net of the MARKOV null (see T3)."""
    out = {}
    for st in ("S", "N"):
        out[f"net_h_{st}"] = obs[f"h_{st}"] - nul[f"h_{st}"]
        out[f"LO_{st}"] = _logit(obs[f"h_{st}"]) - _logit(nul[f"h_{st}"])
    out["ASYM"] = (-out["LO_N"]) - out["LO_S"]
    out["net_D1"] = obs["D1"] - nul["D1"]
    out["net_L2_S"] = obs["L2_S"] - markov[0]
    out["net_L2_N"] = obs["L2_N"] - markov[1]
    return out


def bootstrap(seqs, n_boot=N_BOOT, n_perm_inner=N_PERM_INNER):
    """Cluster bootstrap over player_code with the null recomputed inside each resample."""
    by_pc = {}
    for s in seqs:
        by_pc.setdefault(s[0], []).append(s[1])
    codes = np.array(list(by_pc.keys()))
    keys = None
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(codes, size=len(codes), replace=True)
        arrs = [a for c in pick for a in by_pc[c]]
        v = np.concatenate(arrs).astype(np.int8)
        seg = np.repeat(np.arange(len(arrs)), [len(a) for a in arrs])
        d = nets(statistics(v, seg), null_mean(v, seg, n_perm_inner),
                 markov_null_L2(arrs, N_MARKOV_INNER))
        if keys is None:
            keys = list(d.keys())
        draws.append(np.concatenate([np.atleast_1d(np.asarray(d[k], dtype=float))
                                     for k in keys]))
    return keys, np.vstack(draws)


def _slice(keys, M, name, kmax=KMAX):
    """Columns of the bootstrap matrix belonging to one statistic."""
    off = 0
    for k in keys:
        width = (kmax + 1) if k in ("net_h_S", "net_h_N", "LO_S", "LO_N", "ASYM") else 1
        if k == name:
            return M[:, off:off + width]
        off += width
    raise KeyError(name)


def calibrate(n_seq=800, n_gw=38, n_sim=40, seed=99):
    """Gate on T3, run before the real data is touched: the Markov null must recover ~0 on
    a first-order panel and must not hide a real second-order effect."""
    r = np.random.default_rng(seed)

    def panel(second_order):
        arrs = []
        for _ in range(n_seq):
            v = [int(r.integers(0, 2))]
            v.append(v[-1] if r.random() < 0.9 else 1 - v[-1])
            for _ in range(n_gw - 2):
                p = (0.95 if v[-1] == v[-2] else 0.70) if second_order else 0.90
                v.append(v[-1] if r.random() < p else 1 - v[-1])
            arrs.append(np.array(v, dtype=np.int8))
        return arrs

    out = {}
    for lab, so in (("first_order", False), ("second_order", True)):
        arrs = panel(so)
        v = np.concatenate(arrs)
        seg = np.repeat(np.arange(len(arrs)), [len(a) for a in arrs])
        o = statistics(v, seg)
        mS, mN = markov_null_L2(arrs, n_sim)
        out[lab] = (o["L2_S"] - mS, o["L2_N"] - mN, o["L2_S"], mS)
    ok = (abs(out["first_order"][0]) <= MARGIN_NET
          and out["second_order"][0] >= 0.05)
    return ok, out


def run_scope(seqs, label, rows, primary, t3_ok):
    v, seg = sp.flatten(seqs)
    obs = statistics(v, seg)
    nul = null_mean(v, seg, N_PERM)
    net = nets(obs, nul, markov_null_L2([s[1] for s in seqs], N_MARKOV))
    keys, M = bootstrap(seqs)

    def ci(name, idx=None):
        col = _slice(keys, M, name)
        c = col[:, idx] if idx is not None else col[:, 0]
        return np.nanpercentile(c, 2.5), np.nanpercentile(c, 97.5)

    print(f"\n--- {label} ---")
    print(f"  {'k':>3} {'n_S':>6} {'h_S':>6} {'null':>6} {'net':>7} {'LO_S':>6} "
          f"| {'n_N':>6} {'h_N':>6} {'null':>6} {'net':>7} {'95% CI':>16} {'LO_N':>6} "
          f"| {'ASYM':>6} {'95% CI':>16}")
    for k in KS:
        if obs["n_S"][k] < MIN_CELL or obs["n_N"][k] < MIN_CELL:
            continue
        lo_n, hi_n = ci("net_h_N", k)
        lo_a, hi_a = ci("ASYM", k)
        print(f"  {k:>3} {int(obs['n_S'][k]):>6} {obs['h_S'][k]:>6.3f} "
              f"{nul['h_S'][k]:>6.3f} {net['net_h_S'][k]:>+7.3f} {net['LO_S'][k]:>+6.2f} "
              f"| {int(obs['n_N'][k]):>6} {obs['h_N'][k]:>6.3f} {nul['h_N'][k]:>6.3f} "
              f"{net['net_h_N'][k]:>+7.3f} {f'[{lo_n:+.3f},{hi_n:+.3f}]':>16} "
              f"{net['LO_N'][k]:>+6.2f} | {net['ASYM'][k]:>+6.2f} "
              f"{f'[{lo_a:+.2f},{hi_a:+.2f}]':>16}")
        rows.append(dict(scope=label, stat="hazard", k=k, n_S=int(obs["n_S"][k]),
                         n_N=int(obs["n_N"][k]), h_S=obs["h_S"][k], null_S=nul["h_S"][k],
                         net_S=net["net_h_S"][k], LO_S=net["LO_S"][k],
                         h_N=obs["h_N"][k], null_N=nul["h_N"][k],
                         net_N=net["net_h_N"][k], net_N_lo=lo_n, net_N_hi=hi_n,
                         LO_N=net["LO_N"][k], asym=net["ASYM"][k],
                         asym_lo=lo_a, asym_hi=hi_a))

    # R1 — asymmetry, decided at k = 3 and 6
    def r1():
        ok_pos, ok_neg = [], []
        for k in (3, 6):
            if obs["n_S"][k] < MIN_CELL or obs["n_N"][k] < MIN_CELL:
                return "NOT SHOWN (thin cell)"
            lo, hi = ci("ASYM", k)
            a = net["ASYM"][k]
            ok_pos.append(a >= MARGIN_ASYM and lo > 0)
            ok_neg.append(a <= -MARGIN_ASYM and hi < 0)
        return ("SUPPORTED" if all(ok_pos) else
                "REVERSED" if all(ok_neg) else "NOT SHOWN")

    # R2 — is the non-start curve informative beyond the player's own rate
    def r2():
        for k in (1, 2, 3):
            lo, hi = ci("net_h_N", k)
            if not (net["net_h_N"][k] <= -MARGIN_NET and hi < 0):
                return "NOT INFORMATIVE"
        return "INFORMATIVE"

    # R3 — is one lag enough
    def r3():
        if not t3_ok:
            return "INSTRUMENT NOT CALIBRATED — no claim"
        verd = []
        for nm in ("net_L2_S", "net_L2_N"):
            lo, hi = ci(nm)
            val = net[nm]
            if abs(val) >= MARGIN_NET and (lo > 0 or hi < 0):
                verd.append("INADEQUATE")
            elif lo > -MARGIN_NET and hi < MARGIN_NET:
                verd.append("ADEQUATE")
            else:
                verd.append("INCONCLUSIVE")
        return ("FIRST-ORDER INADEQUATE" if "INADEQUATE" in verd else
                "FIRST-ORDER ADEQUATE" if all(x == "ADEQUATE" for x in verd) else
                "INCONCLUSIVE")

    d1lo, d1hi = ci("net_D1")
    print(f"  E5 pooled lag-1 gap, net of null: {net['net_D1']:+.3f} "
          f"[{d1lo:+.3f},{d1hi:+.3f}]  (observed {obs['D1']:+.3f}, null {nul['D1']:+.3f})")
    for nm, lab in (("net_L2_S", "E6 lag-2 | started last"),
                    ("net_L2_N", "E6 lag-2 | benched last")):
        lo, hi = ci(nm)
        print(f"  {lab}: {net[nm]:+.3f} [{lo:+.3f},{hi:+.3f}]")
    v1, v2, v3 = r1(), r2(), r3()
    tag = "" if primary else "   (not decision-bearing on this scope)"
    print(f"  R1 asymmetry (log-odds, k=3 and 6): {v1}{tag}")
    print(f"  R2 non-start curve informative    : {v2}{tag}")
    print(f"  R3 one lag enough                 : {v3}{tag}")

    for nm, lab, key in (("net_D1", "lag1", "D1"),
                         ("net_L2_S", "lag2_given_start", "L2_S"),
                         ("net_L2_N", "lag2_given_nonstart", "L2_N")):
        lo, hi = ci(nm)
        rows.append(dict(scope=label, stat=lab, obs=float(obs[key]),
                         null=float(nul[key]), net=float(net[nm]),
                         net_lo=lo, net_hi=hi))
    rows.append(dict(scope=label, stat="verdict", asym=net["ASYM"][3],
                     r1=v1, r2=v2, r3=v3,
                     n_S=int(obs["n_S"][3]), n_N=int(obs["n_N"][3])))
    return net


def main():
    print("=" * 78)
    print("START PERSISTENCE — FOLLOW-UP TESTS T1/T2/T3 (pre-registered in this file)")
    print("=" * 78)
    rows = []
    print("\nCALIBRATION of the T3 instrument (synthetic, before any real data)")
    t3_ok, cal = calibrate()
    for lab, (nS, nN, obsS, nulS) in cal.items():
        print(f"  {lab:13s} observed L2_S {obsS:+.3f}, markov null {nulS:+.3f} "
              f"-> net {nS:+.3f} (L2_N net {nN:+.3f})")
        rows.append(dict(scope="calibration", stat=lab, obs=obsS, null=nulS, net=nS))
    print(f"  gate: |net| <= {MARGIN_NET} on first-order AND net >= 0.05 on "
          f"second-order  ->  {'PASS' if t3_ok else 'FAIL'}")
    rows.append(dict(scope="calibration", stat="gate", r3="PASS" if t3_ok else "FAIL"))

    nat = sp.sequences(sp.build(sp.NATIVE))
    nogap = [s for s in nat if not sp.has_absence(s[2])]
    prox = sp.sequences(sp.build(sp.PROXY))
    print(f"\nnative {len(nat)} player-seasons / "
          f"{sum(len(x[1]) for x in nat)} player-matches; "
          f"nogap {len(nogap)}; proxy {len(prox)}")
    print(f"N_PERM={N_PERM}, N_BOOT={N_BOOT} with {N_PERM_INNER} inner permutations")
    for label, seqs, primary in (("native", nat, True), ("proxy", prox, False),
                                 ("nogap", nogap, False)):
        run_scope(seqs, label, rows, primary, t3_ok)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
