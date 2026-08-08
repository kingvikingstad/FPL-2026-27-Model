# Reduced-Form Baseline: Minutes × Team Strength × Opponent

*Handoff note for the FPL 2026/27 main build. Companion to `PROJECT_KNOWLEDGE_2627.md`.
Purpose: specify a nested reduced-form baseline, the test that decides whether the component
layer earns its complexity, and the trap to avoid in evaluating it.*

---

## 1. The proposal

Rather than estimating each scoring component separately, build a projection driven directly by:

- predicted **starts / minutes** per gameweek,
- **own team strength**,
- **opponent strength**.

Rationale: a quality player, on a good team, playing 90 minutes, in an attacking position, will
accumulate chances by construction. If those three inputs carry most of the signal, the component
GLM layer may be expensive complexity.

## 2. Verdict — it is the existing structure minus one term

The current system already factorises multiplicatively. Written out:

```
E[pts_i] = E[min_i] × [ λ_team(own, opp) × share_i + CS_terms(own, opp) + bonus_i ]
```

Minutes, own strength and opponent strength identify `E[min_i]`, `λ_team` and `CS_terms`. They do
**not** identify `share_i` — the fraction of team attacking output routed through the player.
Haaland and Rodri: same club, same fixture, same 90 minutes, ~10× difference in share. Team-strength
precision cannot recover it; it is orthogonal to every proposed input.

So the proposal is not an alternative model. It is the current model with `share_i` shrunk to a
positional/depth prior. That is a legitimate bias–variance choice, not an error — but it must be
framed and tested as such.

## 3. Where the reduced form is (near) complete

**GK and much of DEF.** Clean-sheet points are a pure function of team × opponent × starts, with no
player-level term. Already validated internally: the CS engine matches Solio at **GA corr 0.89,
CS% corr 0.93**, bias ≈ 0 (−0.05 GA, +0.03 CS%), with no player input. This is the same finding as
the settled rule *"pick keepers on team defence"* and *"save volume is negatively correlated with
points."*

Residual player-level content for DEF: defensive contributions (defcon), set-piece and open-play
attacking threat for full-backs, BPS accrual. Call it ~70% team-determined, not 100%.

## 4. Supporting evidence from our own results

The largest validated gain in the project came from the **minutes layer**, not the event layer.
Evidence-weighted minutes shrinkage (`w_hist = minutes/(minutes+900)`, `starter_prior.py`) moved:

| Metric | Before | After |
| --- | --- | --- |
| Pearson vs Solio | 0.727 | 0.779 |
| Spearman | 0.482 | 0.609 |
| MAE | 0.75 | 0.70 |
| DEF gap | −1.07 | −0.86 |

Nothing done to the component GLMs comes close to a +0.13 Spearman move. Combined with the settled
finding that **minutes/availability is ~32% of single-GW variance**, this is genuine support for the
proposal's premise and the reason it is worth formalising rather than dismissing.

## 5. Where it breaks — predicting well ≠ deciding well

If `share_i` is shrunk to a positional prior, every Arsenal midfielder projects alike. But nearly
every real decision is **within-team or between two similar teams**: which Arsenal defender, which
of two £7.5m midfielders, Haaland or Mbeumo as captain. Differential edge lives entirely in `share`.

A reduced-form model can be well-calibrated in aggregate and near-useless for the choices that
actually generate rank. This is the binding objection, and it dictates the evaluation design in §7.

## 6. Specification — nested, so the test is free

Fit as a Poisson fixed-effects model with a minutes offset:

```
log λ_i = log(min_i / 90) + att_own − dfn_opp + home + u_i
```

- `att`, `dfn`, `home` — Dixon–Coles team layer, already fit in `bayes_model.TeamModel`.
- `u_i` — player attacking-share random effect, shrunk toward a position × depth-chart prior.
- Minutes enter as an **offset**, not a covariate: exposure, not a coefficient to estimate.
- Setting `u_i ≡ 0` yields the reduced form exactly. Nesting means one estimation gives both models.

Compose to points through the scoring rules as always — never regress total points directly. Apply
`BPS_2627_MULT = {GK:1.019, DEF:0.896, MID:1.045, FWD:1.041}` at the composition step. Join on
`player_code`.

### Minutes target — model starts, not E[minutes]

Model `P(start)` and `E[min | start]` **separately**. Do not model conditional-mean minutes.

FPL minutes points are a step function (1 pt at ≥1', 2 pts at ≥60') and attacking/CS accrual is
roughly proportional to minutes. `E[min] = 45` means something entirely different if it is "always
plays 45" versus "starts half the time" — identical mean, different point distribution, very
different captaincy tail. The current pipeline already does this; preserve it in the reduced form.

Existing inputs to reuse: ownership-aware cold-start depth prior (`apply_coldstart_depth`),
evidence-weighted shrinkage (`apply_minutes_shrinkage`), then `apply_availability` for injuries and
confirmed XIs. Order of operations unchanged — priors set the baseline, team news overrides.
Ownership is the strongest predictor after minutes (Spearman ~0.48 point-in-time, **~0.68 for
start-rate specifically**) and is not otherwise a projection input, so it stays clean information for
the start-probability layer.

## 7. Evaluation — the trap, and how to avoid it

**Do not evaluate on pooled cross-sectional correlation.** Pooled rank correlation is dominated by
between-team variance, which *both* models capture identically. The reduced form will look almost as
good as the full model on pooled metrics while being materially worse at the actual decision.

Evaluate the incremental value of freeing `u_i` on:

1. **Within-team rank correlation** — Spearman computed *within* club, then averaged across clubs.
   This isolates the share term. Primary metric.
2. **Within-price-band rank correlation** — same logic for the "which £7.5m mid" decision.
3. **Pairwise captaincy accuracy** — of the top-N candidates, how often is the higher-projected
   player the higher scorer.
4. **Tail calibration** — P(haul) coverage, since captaincy is a tail problem and the share term is
   what generates right-tail mass.

Report pooled Pearson/MAE too, but only as a sanity check, never as the decision statistic.

### Estimation constraint (be honest about this)

Full walk-forward validation is **not available offline** — it requires 2023-24 season data, which
is the same blocker holding up joint estimation of `older_weight` and shrinkage K. Until then:

- **Now:** cross-sectional comparison against the Solio feed (`solio_ensemble.py`; current baseline
  Pearson 0.72, MAE 0.72 across 30 GW1 players). Weak evidence — a single gameweek cross-section of
  a competitor model is not ground truth, and we have already ruled out recalibrating on it.
- **After GW1:** live scored data. Accumulate and re-test each week.
- **Ideal:** acquire 2023-24 and run proper walk-forward, which resolves this *and* the `older_weight`
  / Isak parameter sensitivity in the same fit.

Do not conclude from a single cross-section. Log the result as provisional.

## 8. Decision rule (set before looking at results)

- If freeing `u_i` adds **little within-team rank correlation** → real finding. It justifies moving
  effort from event modelling to the minutes prior, where measured returns have been far higher, and
  argues for harder shrinkage on the share term in early gameweeks when it is estimated on thin data.
- If freeing `u_i` adds **substantial within-team rank correlation** → the component layer is
  earning its complexity, and the reduced form's value is as a permanent nested baseline plus a
  regularisation target for cold-start players.

Either outcome is worth having documented. This is a diagnostic, not a proposed replacement.

## 9. What to add to the build

1. `reduced_form.py` — fit the nested spec above; expose `u_i` on/off as a flag (off-by-default
   convention for new components).
2. Within-team / within-band evaluation harness (new metrics, not in the current benchmark set).
3. Wire into the existing A/B pattern used by `decision_v2.py` and `validate_shrinkage.py`.
4. Report the incremental within-team Spearman alongside the pooled metrics in the standard output.

## 10. Standing caveats to carry

- Team strength is still **last-season Opta xG + ClubElo**, not live odds. Until the odds feed lands
  (`betting_odds_ingest.py` / `ab_market_vs_recon.py`), the reduced form inherits stale-prior risk
  on the heavy-turnover clubs (Alonso/Chelsea, Iraola/Liverpool, De Zerbi/Spurs, promoted trio). The
  reduced form is *more* exposed to this than the component model, because team strength does
  proportionally more of its work.
- A single round of priced fixtures is under-identified (~20 λ for ~40 att/dfn params). Any
  market-driven fit needs multi-GW stacking.
- Rotation multipliers and mean-reversion remain tested-null. Do not reintroduce them here.
