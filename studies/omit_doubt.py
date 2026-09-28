from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
omit_doubt.py — omitted AND flagged: is the team-news omission shrink too weak?
===============================================================================
GW5 (INTEGRATION_LOG, "fpl.page moved to RSC", 18 Sep 2026): João Pedro was named in fpl.page's
doubts line, written up as unlikely, omitted from its XI, and at chance_of_playing 0.75 on the
FPL feed. His start prior moved only 0.629 -> 0.515 and the injury ceiling did nothing, because
0.515 < 0.75. It was the board's largest single soft spot at that deadline (74.4% owned). Three
consistent signals were handled as one editorial silence.

CORRECTIONS TO THE RECORD [VERIFIED 2026-09-28, research-preregistrar + main session]
  * The constant that moved him was NOT OMIT_CONFIDENCE = 0.35. gw_board calls
    `apply_consensus` whenever `consensus()` returns rows, which it does with a single
    source; `apply_soft` (where 0.35 lives) is only the fallback. Single-source omission in
    `apply_consensus` is conf 0.20 (0.40 when weighted sources > 1.2), target 0.06
    (predicted_xi.py:547): 0.20*0.06 + 0.80*0.629 = 0.515, the logged value exactly.
  * No locked board holds team news: the GW5 primary lock has team_news_sources = 0, and the
    rebuild that had it was never locked. No pre-team-news prior sits beside a posterior.
  * The doubts line is parsed (`fplpage.parse_blocks`) but never persisted, and raw article
    HTML is not archived, so the D-cell label below cannot be recovered from stored files.
  * `p_band` exists only for players the XI NAMES; it cannot inform an omission.
  * Core-Insights `By Gameweek/GW*/` availability fields are POST-deadline (GW5 first
    committed 19 Sep 01:26, after the 18 Sep 17:30 deadline; news_added 100% null). The
    deadline state is recoverable from the season-level playerstats.csv git history
    (committed ~3x/day), which needs a full-history clone — the live clone is shallow.

STATUS    REGISTERED, NOT RUNNABLE. The instrument must first capture doubt flags, raw
          payloads and the pre-team-news prior per player (INSTRUMENT, below).

PRE-REGISTERED — WRITTEN BEFORE ANY OUTCOME WAS READ
-----------------------------------------------------
Written 2026-09-28 by research-preregistrar. No realised start or minute was opened; the
only data read were file headers, the GW5 predicted-XI file, and pre-deadline FPL
availability snapshots (status / chance_of_playing), used to size the cells.

QUESTION  When a predicted XI omits a player who is ALSO flagged — named in the source's
          doubts line, or FPL chance_of_playing_next_round in {25,50,75} at the deadline —
          is the installed omission shrink too weak? Separately (A2): is the graphic's
          ring band a better target than 0.97 for players the XI names?

INSTALLED The live path is `apply_consensus`, NOT `apply_soft`: gw_board calls apply_soft
          only when consensus() is empty. Single-source omission: conf 0.20 (0.40 when
          weighted sources > 1.2), target 0.06; then `apply_injury_ceiling` caps at
          chance_play. João Pedro GW5: 0.20*0.06 + 0.80*0.629 = 0.515. OMIT_CONFIDENCE
          = 0.35 governs only the fallback.

UNIT      (player_code, gw) at a club whose XI resolved to exactly 11, from a source
          captured before the deadline, with every doubt name in that club block resolved
          within club (one unresolved -> the whole club-gw is excluded: it would mislabel
          D as U). Blank gw excluded; double gw scores the first fixture only.

CELLS     N named. U omitted, unflagged. D omitted AND (doubt line OR FPL chance in
          {25,50,75} at the last Core-Insights commit before the deadline). R omitted with
          chance 0: already pinned by the ceiling, reported only.

ENDPOINT  y = started the first league fixture of the gw, finished matches only, via the
          loader score_gw uses; id -> player_code through that season's players.csv;
          never joined on name. PRIMARY: mean log score (p clipped to [0.005, 0.995]) of
          the start probability AFTER apply_consensus + apply_injury_ceiling, BEFORE the
          XI constraint. The constraint redistributes mass across teammates, so scoring
          after it would credit a D-cell change with its effect on named teammates.
          Post-constraint score and Brier are reported and gate nothing.

ARMS      Every arm reads the same logged pre-team-news (start_a, start_b), the same XI
          resolution and the same ceiling.
          A0  installed.
          A1  A0 with its own confidence c_D for cell D; target 0.06, strength formula
              unchanged. U and N are untouched by construction, so the Saka/Guéhi
              protection for UNEXPLAINED omissions is preserved. c_D is the only fitted
              parameter, on the grid {0.20, 0.30, ..., 0.90}.
          A2  named cell only: target = p_band in place of 0.97, confidence as installed.
              No fitted parameter.
          A3  NOT an arm. Multiplying, or adding log-odds of, the FPL chance and the
              source omission assumes the two are independent given the outcome; both
              read the same press conference, so it double counts. The joint signal is
              carried by A1's parameter fitted on the joint cell; the ceiling stays a cap.

FOLDS     FIT: 25/26 as currently served, plus 26/27 GW1-5; flags from git history.
          CONFIRM: 26/27 GW6+ captured prospectively (raw payload archived, captured_at <
          deadline, prior logged in the ledger). A season split, because served articles
          may carry post-publication edits: any such leak makes D look more predictive
          and pushes c_D toward over-confidence, which prospective confirmation then
          penalises. It can produce a false NULL, never a false ADOPT. Leave-one-gw-out is
          not used: an injured player recurs in D across consecutive gws. FALLBACK if the
          fit set has < 150 D cases: GroupKFold(5) on player_code inside the confirmation
          set, and the minimum below doubles to 400.

DECISION RULE (A1), evaluated ONCE, when the confirmation set first holds n_D >= 200
  (counted from the ledger alone, with no outcome read), or at GW38, whichever is first.
  ADOPT c_D only if ALL hold:
   (1) held-out log-score gain of A1 over A0 on D is > 0 and its 95% player-cluster
       bootstrap CI excludes 0;
   (2) calibration in the large on D: |mean p_A1 - mean y| <= |mean p_A0 - mean y|;
   (3) GUARDRAIL for the asymmetric cost: on D restricted to p_hist >= 0.60 (the large
       projections whose deletion is the expensive error), A1's log score is not below
       A0's (point estimate);
   (4) pooled log score over N+U+D not worse (mechanical; stated so a population cannot
       be chosen after the fact).
  (1) fails -> NULL, c_D is not wired. (1) passes and (3) fails -> NULL FOR SHIPPING.
  n_D < 200 at GW38 -> INCONCLUSIVE; installed stays; carried to 27/28 with this rule
  unchanged.
  NOT used as a guardrail: calibration of the "omitted but started" group. It conditions
  on y = 1, so any downward move worsens it by construction.

DECISION RULE (A2): the same gates (1)-(3) on N with band in {70-84, 50-69, <50},
  n >= 150, guardrail subset p_hist >= 0.60. A separate decision on a separate
  population with no shared parameter.

POWER     D is sized at 8-15 per gw [JUDGMENT], from pre-deadline FPL flags (8-16 per week
          at chance 25-75 across the league, 26/27 GW1-5) and about 4-5 doubt names per
          club. With A0 near 0.45 on D, n = 200 gives 80% power for a true rate of 0.25;
          a true rate of 0.30 needs about 400. Powered for a João Pedro-sized miss, not a
          subtle one. A null at n >= 200 therefore rules out a gap of 0.20 or more.

IF IT FAILS  A1: no code path exists outside this file; record the null in PROJECT_
          KNOWLEDGE §5 and §7. A2: delete BAND_P and the p_band column from fplpage (keep
          the band label as provenance); record in the fplpage docstring and §7. The doubt
          capture and the ledger stay: they are instrument, not model.

GUARDS    Conditions on contemporaneous availability news about the named player: no
          fixture term, no past-start sequence, no style term, player-level signal. Not a
          rotation multiplier, not congestion, not benching asymmetry, not a streak_k term.

INSTRUMENT — data capture, no model change, may ship ahead of the result
-------------------------------------------------------------------------
Acceptance: a board A/B bit-identical on `mean`, plus offline selftests.
  1. fplpage.scrape persists `doubt_flag` on XI rows, and writes non-XI doubt names to a
     separate gw{N}_fplpage_doubts.csv OUTSIDE the predicted_xi_gw* pattern (a
     role="doubt" row would be mapped to "bench" by predicted_xi.load). A loud check on
     the 400-character cap of the doubts regex (fplpage.py:258).
  2. Archive the raw article payload and the graphic bytes per capture under a new
     config.TEAM_NEWS directory, stamped with capture time.
  3. A team-news ledger appended by gw_board whenever PRED_XI applies: gw, run_ts,
     deadline, player_code, team, start_a/start_b before team news, weighted
     n_start/n_sources, doubt_flag, band, p_band, chance_play, p after consensus, p after
     ceiling — for EVERY player at a covered club.
  4. Register team_news/ and the ledger in src/manifest.py as committed, non-regenerable
     stores.
  5. A separate full-history Core-Insights clone at a new config path (never deepen the
     live clone doctor watches); read pre-deadline flags with `git show`.
  6. A selftest fixture for the 25/26 heading format ("ARSENAL (89%)", no colon) before
     any retro scrape [CHECK].

RESULT
------
(not run — blocked on INSTRUMENT)
"""


def main():
    print("omit_doubt: REGISTERED 2026-09-28, not runnable — blocked on INSTRUMENT (doubt")
    print("capture, raw payload archive, team-news ledger with the pre-team-news prior).")
    print("See the module docstring; nothing here reads outcome data.")


if __name__ == "__main__":
    main()
