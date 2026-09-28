from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
start_dependence.py — a start predictive that carries serial dependence (PROJECT_KNOWLEDGE §6.12)
================================================================================================
`start_forgetting.py`'s dispersion gate found the installed start predictive far too confident
over a horizon: coverage of the central 80% Beta-Binomial interval for a ten-match start count
is 0.530 against 0.80, because start sequences are serially correlated (§5) and the predictive
treats matches as exchangeable given p.

WHERE THE DEFECT REACHES [VERIFIED 2026-09-28, main session + research-preregistrar]
`project()` holds p across fixtures WITHIN one call. The canonical board (gw_board) calls it once
per gameweek and `_player_rng` is keyed on gw, so p is re-drawn every week. The under-dispersion
therefore reaches only double gameweeks (none on the 26/27 fixture list as of 28 Sep: 760
team-gameweeks, all single) and the non-canonical `project(lo, hi)` callers (run_final_board,
decision_v2, roster.__main__), none of which feeds a decision. Per-gameweek percentiles, the
captaincy tail, lock_team and P(haul) read one gameweek, where a start is Bernoulli(E[p]) and the
Beta's spread is irrelevant. The horizon solver (`solver.solve_horizon`) maximises summed MEANS,
which no dispersion fix changes. A consumer that summed canonical draws across weeks would see
ZERO serial start dependence — a larger defect than §6.12's, latent for the same reason.

VERDICT: record, do not build. This file holds the rule so it predates any look.

PRE-REGISTERED — WRITTEN BEFORE ANY OUTCOME WAS READ   (research-preregistrar, 2026-09-28)
------------------------------------------------------------------------------------------
STATUS    DORMANT. Filed now so the rule predates any look; NOT to be run until ACTIVATION
          holds. The finding it answers (PROJECT_KNOWLEDGE §6.12) has no decision consumer on
          the canonical path: gw_board calls project(gw,gw) per week and re-draws p_start per
          week, so a Beta's spread is inert at H=1; the horizon solver maximises summed MEANS.
ACTIVATION  run when ANY of: (a) a consumer sums per-player draws across >= 2 gameweeks or
          prices risk over a horizon (a risk term in solver.solve_horizon, a chip/EV-at-risk
          tool); (b) a project(lo,hi) output is registered in src/manifest.py AND read by a
          decision script. A DGW does NOT activate this study — see SCOPE.
PRIOR LOOK  disclosed: H=10 coverage 0.530/0.539/0.562 (start_forgetting, stand-in prior).
          No H=6 number, no per-position number, no Markov fit has been read.

ESTIMAND  coverage of the central 80% predictive interval for S_H = number of starts in the
          next H league matches, per (player, season, origin k), held out by season; and the
          shape of that miss (lower vs upper tail). Comparison: installed exchangeable
          Beta-Binomial vs two dependence-carrying arms that share its per-match mean exactly.
H         6 — scripts/run_solver.py SOLVER_HORIZON default (GW1..6), the horizon of the only
          multi-week planner. If the activating consumer's horizon differs, H is set to THAT
          horizon in a dated amendment written BEFORE the run; never chosen from a sweep.
UNIT      player-season x origin k, k in {3,5,8,12,20,30}, requiring >= H matches after k.
          Loader: start_persistence.build; 22/23 observed from GW16 only
          (fpl_history.empty_native_gws). Join on player_code. League matches only.
FOLDS     LOSO over the season-start folds 23/24, 24/25, 25/26; 22/23 supplies priors only.
          Free parameters are fitted on the two training seasons, scored on the third.
MEAN      every arm's per-match predictive mean is the flat update at kappa=4 on the
          previous-season stand-in prior (start_forgetting arm A), m_k = a_k/(a_k+b_k).
          The stand-in, not the production prior, because the production prior's LEVEL is
          biased (§6.9: 0.548 vs 0.424) and a widened interval would absorb a level error and
          score it as dispersion.
ARMS      A INSTALLED  S_H ~ BetaBinomial(H, a_k, b_k).
          B WIDENED    S_H ~ BetaBinomial(H, c*m_k, c*(1-m_k)), c = min(a_k+b_k, c_pos);
                       one c_pos per position, fitted. Mean preserved exactly; equivalent to
                       rescaling (start_a, start_b) in project() at fixed ratio.
          C CHAIN      p ~ Beta(a_k, b_k); a two-state chain with STATIONARY probability p
                       and lag-1 persistence phi_pos in [0,1) (P(1|1)=p+phi(1-p),
                       P(1|0)=p(1-phi)); one phi_pos per position, fitted. The INITIAL STATE
                       IS DRAWN FROM Bernoulli(p), NOT set to the observed y_k.
          (Filter-forward is excluded: at lam=1 it is arm A; at lam<1 it moves the mean.)
FIT       c_pos, phi_pos by maximising the held-in log predictive of S_H. Coverage is never
          the fit objective.
ENDPOINT  randomised PIT u = F(S-1) + V*f(S), V~U(0,1), on held-out windows.
          PRIMARY: coverage = P(0.10 <= u <= 0.90).
          TAILS:   P(u<0.10) and P(u>0.90).
          LEVEL:   mean(u).
          Measured before the XI constraint and the availability override, as in
          start_forgetting.

DECISION RULE, fixed now
  P0 VALIDITY (precondition, on arm A): mean(u) in [0.45, 0.55] in every fold. If it
     fails, the study is INVALID — dispersion is not separable from level — not null.
  An arm PASSES only if, in 3 of 3 held-out folds:
     (1) coverage in [0.75, 0.85];
     (2) each tail in [0.05, 0.15];
     (3) coverage strictly closer to 0.80 than arm A's.
  GUARDRAIL: single-match Brier of B and C must EQUAL A's to 1e-9 (the mean is shared by
     construction). A difference is a bug, halts the run, and is not a result.
  B and C both pass -> choose the higher pooled held-out log score of S_H. If the gap is
     under 2 SE by a player-season block bootstrap, choose B (one line in project()).
  Exactly one passes -> that arm.
  Neither passes -> NULL: dependence is real (§5) but neither one-parameter-per-position
     form calibrates it; nothing is wired.
  PASS ACTION: ship behind FPL_START_DEP, OFF; default it on only after an A/B on the
     activating consumer. If that consumer sums canonical per-GW draws, the arm must be
     implemented as one p (and chain) per player per WINDOW across project calls. Per-week
     re-draws (current _player_rng keying) would discard it. This must not reintroduce
     the 2026-09-08 comonotone-stream bug: share the p/chain draw only, never the uniforms.

POWER     about 450 player-seasons/fold [CHECK before the run]. Origins within a
          player-season overlap, so n_eff is taken as player-seasons. Per-fold coverage SE
          is about 0.019. A calibrated arm clears the band 3/3 with P about 0.97; true
          coverage 0.74 clears with P about 0.03; arm A (about 0.53) is about 14 SE out.
          Informative in both directions. Nailed and never-starting players dominate n and
          carry little PIT information, so the rotation zone m_k in [0.15, 0.85] is REPORTED
          with its own n. It is non-gating: its folds are too small to gate at this band.

GUARDS    Not a rotation multiplier: no fixture-conditional term. Not mean reversion: the
          mean is fixed. Not a streak predictor: arm C's initial state is drawn from the
          stationary law, so E[y_t] = m_k for every t in the window. Conditioning the chain
          on the observed y_k would be a lag-1 state predictor on the MEAN — the untested
          streak_k line (PROJECT_KNOWLEDGE §7), which needs its own registration and a mean
          endpoint — and is forbidden here. Benching asymmetry is a tested null, so phi is
          symmetric by construction. Not a team-level signal, so beats_the_market does not
          apply, and this is a second-moment claim in any case.
SCOPE     Says nothing about DGWs. Two fixtures about 3 days apart are not the
          week-apart sequences phi is fitted on; their dependence may be negative and is
          untested-adjacent to GW27+ congestion (knockout kickoffs + FA Cup, not in the feed).
IF IT FAILS  no src/ path exists before a pass, so nothing is deleted. This file and its CSV
          are the record; the null is written into PROJECT_KNOWLEDGE §6.12, and the §7
          entry states the form tested. An INVALID (P0) outcome is recorded as blocked on
          §6.9's level fix.

RESULT
------
(dormant — not run)
"""


def main():
    print("start_dependence: DORMANT — pre-registered 2026-09-28, not run.")
    print("Activates when a consumer sums per-player draws across >= 2 gameweeks or prices")
    print("horizon risk, or a project(lo,hi) output is registered AND read by a decision")
    print("script. See the module docstring; nothing here reads outcome data.")


if __name__ == "__main__":
    main()
