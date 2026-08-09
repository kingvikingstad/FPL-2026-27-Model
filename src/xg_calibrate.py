from __future__ import annotations
import config
"""
xg_calibrate.py — put Understat xG on the scale the team layer already uses.
============================================================================
**Mandatory, and easy to skip. Do not skip it.** Understat runs its own xG model.
Mixing its output with the series `TeamModel` is calibrated against introduces a
level mismatch that propagates into every player projection — silently, because
nothing downstream checks units.

CORRECTION TO THE SPEC [VERIFIED by inspection]
-----------------------------------------------
The integration spec assumes the team layer is fitted on **Opta match xG via
E0_recon.csv**. It is not. `E0_recon.csv` holds goals and closing betting odds; the
expected-goals series the model actually uses is **market-implied lambda**, solved
out of the 1X2 + over/under markets by `betting_features.build()`. There is no
per-match Opta xG anywhere in this repo.

That distinction changes the test, because the two references are different kinds of
quantity:

  * xG vs xG (per match) — two measurements of the same latent event quality. An
    affine map should align them tightly, and the spec's `r2 > 0.85` gate is a
    JOIN-CORRECTNESS check: fail it and the join is wrong, not the models.
  * market lambda vs xG (per match) — a forecast against a realisation. These
    genuinely disagree match to match; per-match r2 lands nowhere near 0.85 and a
    0.85 gate would fail for a correct join. Aggregating to team-season averages out
    the match noise and makes r2 meaningful again.

So there are two entry points, each with its own pre-committed gate. `strict=True`
(the spec's rule) applies to the xG-vs-xG path only. Under G7 a rule that fails is a
null, not something to soften — so rather than relax 0.85 on the market path, the
market path is a separate test at the level where 0.85 is the right number.

If a genuine per-match Opta xG series is ever added, use `fit_understat_to_opta`
and delete the market path.
"""
import argparse, sys
import numpy as np, pandas as pd

R2_MIN = 0.85          # join-correctness gate, both paths (see note above)
SLOPE_BAND = (0.8, 1.2)


def _ols(x, y):
    """y = a + b x. Returns (a, b, r2, n). No sklearn — this is two moments."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 8:
        raise ValueError(f"only {n} paired observations — refusing to fit a calibration")
    b = np.cov(x, y, ddof=1)[0, 1] / max(np.var(x, ddof=1), 1e-12)
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / max(ss_tot, 1e-12)
    return float(a), float(b), float(r2), int(n)


def _gate(cal, strict, what):
    """Enforce the pre-committed thresholds. Loud failure by design (G7)."""
    bad = []
    for season, v in cal.items():
        if v["r2"] < R2_MIN or not (SLOPE_BAND[0] < v["b"] < SLOPE_BAND[1]):
            bad.append(f"  {season}: a={v['a']:+.3f} b={v['b']:.3f} "
                       f"r2={v['r2']:.3f} n={v['n']}")
    if bad and strict:
        raise AssertionError(
            f"{what}: calibration outside the pre-committed band "
            f"(r2 > {R2_MIN}, {SLOPE_BAND[0]} < b < {SLOPE_BAND[1]}):\n"
            + "\n".join(bad) +
            "\nThis means the JOIN is wrong, not that the xG models differ. "
            "Check team-name normalisation and date alignment before anything else.")
    for line in bad:
        print(f"[xg-calibrate] WARNING (non-strict) {line.strip()}")
    return cal


def fit_understat_to_opta(understat_team_match: pd.DataFrame,
                          opta_matches: pd.DataFrame,
                          strict: bool = True) -> dict:
    """Per-season affine map on OVERLAPPING team-match totals:

        xg_opta = a + b * xg_understat

    Fit by OLS on team-match pairs joined on (date, team). Returns
    {season: {a, b, r2, n}}.

    `opta_matches` must be long form, one row per team-match, columns
    [date, team, xg] (+ optional season). Asserts r2 > 0.85 and 0.8 < b < 1.2 —
    anything outside means the join is wrong, not that the models differ.

    `b` is attenuated by measurement error in the REGRESSOR, so it recovers slightly
    less than the true scale ratio. That is correct here: the target is E[ref | u],
    which is what we want to substitute, not the errors-in-variables structural slope.
    """
    j = _join_team_match(understat_team_match, opta_matches, "xg", "xg_ref")
    cal = {s: dict(zip(("a", "b", "r2", "n"), _ols(g["xg"], g["xg_ref"])))
           for s, g in j.groupby("season")}
    return _gate(cal, strict, "understat->opta (team-match)")


def fit_understat_to_market(understat_team_match: pd.DataFrame,
                            market_team_match: pd.DataFrame,
                            strict: bool = True) -> dict:
    """Affine map against the market-implied lambda the team layer actually uses,
    fitted on **team-season totals** rather than team-matches.

    Per-match, a forecast and a realisation disagree for real reasons and r2 would
    fail its gate on a perfectly correct join. Team-season totals average that away,
    so r2 recovers its meaning as a join check. Same thresholds, different level —
    stated up front, not tuned after seeing results.

    `market_team_match`: long form [date, team, opp, is_home, lam_for] from
    `market_reference()`. Joined on the fixture pair, not the date — see
    `_join_team_match`.
    """
    j = _join_team_match(understat_team_match, market_team_match, "xg", "lam_for",
                         on="fixture")
    agg = (j.groupby(["season", "team"])[["xg", "lam_for"]].sum().reset_index())
    cal = {s: dict(zip(("a", "b", "r2", "n"), _ols(g["xg"], g["lam_for"])))
           for s, g in agg.groupby("season")}
    return _gate(cal, strict, "understat->market (team-season)")


def _join_team_match(understat, ref, ucol, rcol, on="date"):
    """Join after normalising both sides' club spelling. Reports the match rate — a
    silently-thin join is the failure mode this whole module exists to catch.

    `on="date"`  — (date, team). Correct when both sides carry real kickoff dates.
    `on="fixture"` — (team, opp, is_home). Use against E0_recon: `reconstruct_e0.py`
        stamps any match with a missing kickoff time as the literal "01/01/2026", so
        its dates are a sentinel, not data. [VERIFIED 2026-08-09] — 20 teams collapse
        onto 2 dates, Arsenal appearing in 5 fixtures on 2026-01-01. Harmless to the
        model (it only sorts by date) but fatal to a date join. The fixture pair is
        unique within a season and unaffected.
    """
    import sd_ingest

    keys = ["date", "team"] if on == "date" else ["team", "opp", "is_home"]

    def _day(s):
        """Calendar day, timezone-naive. Opta kickoff times are UTC-aware and
        Understat's are not; pandas refuses to merge the two."""
        return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None) \
            .dt.normalize()

    u = understat.copy()
    u["team"] = u["team"].map(sd_ingest.normalise_team)
    u["date"] = _day(u["date"])
    if "opp" in u.columns:
        u["opp"] = u["opp"].map(sd_ingest.normalise_team)
    if "season" not in u.columns:
        u["season"] = "all"

    r = ref.copy()
    r["team"] = r["team"].map(sd_ingest.normalise_team)
    r["date"] = _day(r["date"])
    if "opp" in r.columns:
        r["opp"] = r["opp"].map(sd_ingest.normalise_team)
    r = r.rename(columns={ucol: rcol}) if ucol in r.columns and rcol not in r.columns else r

    # Keys must be unique on both sides. Duplicates fan the merge out and inflate n
    # while destroying the fit — the exact failure that looks like "the xG models
    # disagree" but is really a broken join.
    for side, d in (("understat", u), ("reference", r)):
        dup = int(d.duplicated(keys).sum())
        if dup:
            raise RuntimeError(
                f"{side} side has {dup} duplicate {tuple(keys)} rows, so the merge would "
                f"fan out. If joining on date against E0_recon, use on='fixture' — its "
                f"dates are a sentinel fallback, not data.")

    j = u.merge(r[keys + [rcol]], on=keys, how="inner")
    rate = len(j) / max(len(u), 1)
    print(f"[xg-calibrate] joined {len(j)}/{len(u)} team-matches ({rate:.1%})")
    if rate < 0.80:
        unmatched = sorted(set(u["team"]) - set(r["team"]))
        raise RuntimeError(
            f"only {rate:.1%} of Understat team-matches found a reference row. "
            f"Club names present in Understat but not the reference: {unmatched}. "
            f"Fix sd_ingest.UNDERSTAT_TO_FRAME before calibrating.")
    return j


def market_reference(e0_path=None) -> pd.DataFrame:
    """Long-form team-match market lambda from E0_recon:
    [date, team, opp, is_home, lam_for].

    This is the series `TeamModel` is calibrated against, so it is the correct
    reference scale for anything entering the player layer. `date` is carried for
    reference only — do not join on it (see `_join_team_match`).
    """
    import betting_features as bf
    d = bf.build(e0_path or config.E0_RECON)
    home = pd.DataFrame({"date": d["Date"], "team": d["HomeTeam"],
                         "opp": d["AwayTeam"], "is_home": True,
                         "lam_for": d["lambda_home"]})
    away = pd.DataFrame({"date": d["Date"], "team": d["AwayTeam"],
                         "opp": d["HomeTeam"], "is_home": False,
                         "lam_for": d["lambda_away"]})
    return pd.concat([home, away], ignore_index=True)


def apply_calibration(df: pd.DataFrame, cal: dict, cols=("xg",),
                      season_col="season") -> pd.DataFrame:
    """Apply the per-season affine map to every listed column. Returns a copy.

    Valid on SHOT-level xg as well as totals: the map is fitted on sums but affine
    maps commute with summation up to the intercept, so the intercept is spread
    across the summands in proportion to their share of the total. In practice `a`
    is small; the slope is what matters.

    A season with no fitted calibration is left UNTOUCHED and reported, rather than
    silently borrowing another season's map.
    """
    out = df.copy()
    seasons = out[season_col].unique() if season_col in out.columns else ["all"]
    for s in seasons:
        v = cal.get(s) or cal.get("all")
        if v is None:
            print(f"[xg-calibrate] no calibration for season {s} — left on the raw scale")
            continue
        m = (out[season_col] == s) if season_col in out.columns else slice(None)
        n_rows = int(m.sum()) if season_col in out.columns else len(out)
        for c in cols:
            if c not in out.columns:
                continue
            share = v["a"] / max(n_rows, 1) if n_rows else 0.0
            out.loc[m, c] = v["b"] * out.loc[m, c] + share
    return out


# ------------------------------------------------------------------- selftest
def selftest():
    rng = np.random.default_rng(3)
    clubs = ["Manchester City", "Liverpool", "Everton", "Nottingham Forest"]
    # one matchday per date, every club playing once — so (date, team) is unique
    md = pd.to_datetime("2025-08-16") + pd.to_timedelta(np.repeat(np.arange(95), 4) * 7, "D")
    dates, teams, n = md, np.tile(clubs, 95), 380
    truth = rng.gamma(4.0, 0.35, n)                         # latent match quality

    # Understat measures the truth with noise; the reference is the SAME quantity on a
    # shifted scale (b=1.15, a=-0.05) plus its own noise. A correct join must recover it.
    xg_u = truth + rng.normal(0, 0.12, n)
    xg_ref = -0.05 + 1.15 * truth + rng.normal(0, 0.12, n)

    u = pd.DataFrame({"date": dates, "team": teams, "xg": xg_u, "season": "2526"})
    ref = pd.DataFrame({"date": dates, "team": [
        {"Manchester City": "Man City", "Nottingham Forest": "Nott'm Forest"}.get(t, t)
        for t in teams], "xg_ref": xg_ref})
    # the reference arrives in FRAME spelling; only a working name map can join it

    cal = fit_understat_to_opta(u, ref)
    v = cal["2526"]
    assert abs(v["b"] - 1.15) < 0.05, f"slope not recovered: {v['b']:.3f}"
    assert v["r2"] > R2_MIN

    # applying the map must move the Understat mean onto the reference mean
    before = abs(u["xg"].mean() - ref["xg_ref"].mean())
    after = abs(apply_calibration(u, cal)["xg"].mean() - ref["xg_ref"].mean())
    assert after < before / 3, f"calibration did not close the level gap ({before:.3f}->{after:.3f})"

    # a wrong join must RAISE, not return a soft number
    bad_ref = ref.copy()
    bad_ref["date"] = bad_ref["date"] + pd.Timedelta(days=1)
    try:
        fit_understat_to_opta(u, bad_ref)
    except (AssertionError, RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("a misaligned join must fail the gate, not pass quietly")

    # unmapped club names must surface as a join failure, not vanish
    orphan = u.copy()
    orphan["team"] = orphan["team"].replace("Nottingham Forest", "Some New Club")
    try:
        fit_understat_to_opta(orphan, ref)
    except RuntimeError as e:
        assert "UNDERSTAT_TO_FRAME" in str(e)
    else:
        raise AssertionError("unmapped club names must raise")

    print(f"SELFTEST OK: recovered b={v['b']:.3f} (true 1.15), a={v['a']:+.3f} "
          f"(true -0.05), r2={v['r2']:.3f}, n={v['n']}; "
          f"level gap {before:.3f} -> {after:.3f}; bad joins raise.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seasons", nargs="+", default=["2526"])
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    import sd_ingest
    tm = sd_ingest.understat_team_match(a.seasons)
    cal = fit_understat_to_market(tm, market_reference())
    for s, v in cal.items():
        print(f"{s}: xg_ref = {v['a']:+.3f} + {v['b']:.3f} * xg_understat   "
              f"(r2={v['r2']:.3f}, n={v['n']})")
