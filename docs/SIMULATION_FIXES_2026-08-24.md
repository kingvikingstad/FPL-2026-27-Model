# Two fixes inside `project()` — 2026-08-24

Both found while validating the in-season update path. Both are in the Monte Carlo
composition step, both were invisible in the mean, and both changed what the posterior
draws actually mean.

---

## 1. Per-player random streams — A/B comparisons were not clean

### The defect

`project()` drew every player from one shared module-level generator,
`bayes_model.rng`. numpy's `beta`, `gamma` and `poisson` use rejection sampling, so the
number of raw bits a draw consumes depends on the **parameter values**, not just the call
count. Changing one player therefore shifted the stream for every player simulated after
him.

There was also an outright data-dependent call: `pen_goals` is drawn only when
`pen_xg90 > 0`, so granting a penalty duty consumed an extra draw per fixture.

Measured before the fix — perturbing a single Arsenal player's `start_a`:

| players whose projection moved | at Arsenal | at other clubs | mean \|Δ\| elsewhere |
|---|---|---|---|
| **48 / 604** | 29 | 19 | 0.080 |

### Why it survived this long

The board was still reproducible given *identical* input, so `test_all`'s determinism
check passed and nothing looked wrong. What was broken was every **A/B**: a measured
difference was the true effect **plus a re-randomisation term** of roughly the same size
as the small effects this project spends its time trying to detect. The isolation checks
that did pass (e.g. predicted-XI gameweek isolation, bitwise 0.000) only exercised cases
where the player frame was untouched, so they could not catch it.

### The fix

`bayes_model._player_rng(p, seed)` seeds an independent `Generator` per player from
`[PROJECT_SEED, player_key]`, where the key is `player_code` — the project's stable join
key — falling back to a CRC32 of `web_name|team`. Python's built-in `hash()` is salted
per process and would break reproducibility across runs, so it is not used. `project()`
gained an explicit `seed=` argument.

**Team-level correlation is unaffected.** It lives in `tsamp`, drawn once outside
`project()` and indexed identically for every player, so draw *s* still means the same
team-strength world for everyone. What becomes independent across players is exactly what
the model already assumes is conditionally independent given team strength.

### After

| property | before | after |
|---|---|---|
| perturb one player | 48 moved | **1 moved** |
| grant a penalty duty | many moved | **1 moved** |
| identical input | reproduces | reproduces |
| **shuffle the player frame** | changed projections | **0 moved** |
| **drop a player** | changed survivors | **0 moved** |

The last two are new properties that did not hold before and are worth having on their
own: filtering or reordering the player frame no longer perturbs the board.

**Distribution unchanged**, as required — league total mean 1795.2 → 1796.5 at S=3000
(+0.07%), per-player mean |Δ| 0.061, correlation 0.996. This is a re-randomisation, not a
re-specification.

Locked down by `tests/test_rng_isolation.py`, which deliberately uses varied priors and
includes a penalty taker — identical parameters would consume identical bits and the old
code would pass.

---

## 2. Clean sheet and goals conceded were drawn twice

### The defect

```python
cs   = (rng.poisson(lam_against[f]) == 0) & played60     # one realisation
conc =  rng.poisson(lam_against[f])                      # a DIFFERENT one
```

Two independent draws of the same quantity. A simulation path could therefore award a
defender **clean-sheet points and a two-goal concession penalty in the same match** — a
state that cannot occur.

### Why it was invisible

`E[csp]` and `E[concp]` are each individually correct, and expectation is linear, so
`mean` was **unbiased**. What was wrong was their **joint** distribution: `sd`, `p5`/`p95`
and the captaincy tail — which is precisely what the posterior draws are retained for
(PROJECT_KNOWLEDGE §3: "captaincy is a tail problem"). The two components carry positive
covariance (conceding nothing gives you the clean sheet *and* spares you the penalty), and
throwing it away understated defender and goalkeeper variance.

### After

One realisation drives both. Measured with the RNG fix held constant, S=3000, GW2:

| | n | mean total | sd (mean) | p95 (mean) | cs_ev total |
|---|---|---|---|---|---|
| GK | 67 | 162.9 → 163.0 | **2.245 → 2.418** | 6.41 → 6.51 | 71.8 → 71.8 |
| DEF | 201 | 644.8 → 644.9 | **3.138 → 3.250** | 8.66 → 8.67 | 186.5 → 186.5 |
| MID | 268 | 768.0 → 768.1 | 2.989 → 2.989 | 8.84 → 8.83 | 51.3 → 51.3 |

Exactly the predicted signature: means and `cs_ev` unchanged, spread up for the positions
that concede, and **MID unchanged to three decimals** — a clean control, since concession
points apply only to GK and DEF.

---

## Consequence for anything already measured

Every A/B run before 2026-08-24 carries the re-randomisation term from §1. It is
unbiased, so conclusions resting on large effects stand unchanged. Conclusions resting on
differences of order 0.05–0.10 points per player are less certain than they were reported
to be, and are worth re-running now that the comparison is clean — in particular the
smaller board-level A/Bs in `scripts/decision_v2.py` and `scripts/sweep_older_weight.py`.

Nothing in `studies/` is affected: those are fitted on historical panels, not on
`project()` output.

Any tail-based result computed before today — captaincy P(haul) rankings especially —
also carries §2, which understated GK/DEF spread by roughly 3–8%.
