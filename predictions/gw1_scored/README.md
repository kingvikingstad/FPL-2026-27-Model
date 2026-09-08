# GW1 scored plans — frozen

The `hybrid` (52) and `model` (60) rows in `data/my_results.csv` were scored against
these two files. `scripts/plan_constrained.py` overwrites `outputs/plans/*_squad.csv`
on every run, and re-running it does NOT reproduce these squads — the inputs have moved
on. Without this copy the GW1 ledger entry would have no provenance.

Both carry the `bench_order` column added on 2026-08-26; the model track's autosub
(Amad blanked -> Ndiaye +9) is only resolvable because of it.

Do not regenerate these. Freeze a new pair per gameweek.
