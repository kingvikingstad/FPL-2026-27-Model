import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
test_xi_constraint.py — a club starts eleven players, not fifteen.
==================================================================
Start priors are built per player and never see each other, so nothing enforced the
hardest constraint in the sport. Measured on the GW1 locked board, every one of the 20
clubs exceeded it and the league expected 295.1 starters against a structural 220.

`starter_prior.apply_xi_constraint` removes the surplus with a per-club shift in log-odds.
These are the properties that make that shift the right one rather than merely a rescaling.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import starter_prior as sp


def main():
    ok = True

    def check(cond, label, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {label}" + (f"   {detail}" if detail else ""))
        ok &= bool(cond)

    rs = np.random.default_rng(4)
    rows = []
    for club, n in (("A", 30), ("B", 22), ("C", 18)):
        for i in range(n):
            pstart = float(rs.beta(2, 2))
            k = float(rs.uniform(6, 60))
            rows.append({"web_name": f"{club}{i}", "team": club,
                         "start_a": pstart * k, "start_b": (1 - pstart) * k})
    d = pd.DataFrame(rows)
    p_before = d.start_a / (d.start_a + d.start_b)
    out, rep = sp.apply_xi_constraint(d, verbose=False)
    p_after = out.start_a / (out.start_a + out.start_b)

    for club, g in out.groupby("team"):
        tot = float((g.start_a / (g.start_a + g.start_b)).sum())
        check(abs(tot - 11.0) < 1e-6, f"{club} sums to eleven", f"{tot:.6f}")

    check(bool(((p_after > 0) & (p_after < 1)).all()),
          "every probability stays inside (0, 1)")

    for club, g in out.groupby("team"):
        i = d.index[d.team == club]
        before = p_before.loc[i].values
        after = p_after.loc[i].values
        check(bool((np.argsort(before) == np.argsort(after)).all()),
              f"{club} preserves the ordering of players")
        # relative ODDS between any pair must be untouched by a pure log-odds shift
        ob = before / (1 - before); oa = after / (1 - after)
        ratio = (oa / ob)
        check(float(ratio.max() - ratio.min()) < 1e-8,
              f"{club} preserves every pairwise odds ratio",
              f"spread {ratio.max()-ratio.min():.2e}")

    # Incidence: the correction must fall on the FRINGE, not on nailed starters. Measured
    # WITHIN one club being corrected downward — pooling across clubs that move in
    # opposite directions cancels the effect and tests nothing.
    down = rep[rep.before > rep.after + 1e-9]["team"].tolist()
    check(len(down) > 0, "at least one club is corrected downward")
    club = down[0]
    i = d.index[d.team == club]
    moved = (p_before.loc[i] - p_after.loc[i]).abs()
    nailed = p_before.loc[i] > 0.9
    fringe = (p_before.loc[i] > 0.25) & (p_before.loc[i] < 0.6)
    check(nailed.sum() > 0 and fringe.sum() > 0, f"{club} has both bands to compare",
          f"nailed {int(nailed.sum())}, fringe {int(fringe.sum())}")
    check(moved[nailed].mean() < moved[fringe].mean(),
          f"{club}: nailed starters move less than fringe players",
          f"{moved[nailed].mean():.3f} vs {moved[fringe].mean():.3f}")

    # prior STRENGTH is confidence, not mean — it must survive untouched
    check(np.allclose((out.start_a + out.start_b).values,
                      (d.start_a + d.start_b).values),
          "prior strength (a+b) is unchanged")

    # UNDER eleven is corrected too — the constraint is an equality. A squad of twelve
    # all at p=0.1 expects one starter; eleven of them must play.
    small = pd.DataFrame({"web_name": [f"S{i}" for i in range(12)], "team": ["S"] * 12,
                          "start_a": [1.0] * 12, "start_b": [9.0] * 12})   # sums to 1.2
    s_out, _ = sp.apply_xi_constraint(small, verbose=False)
    sp_after = s_out.start_a / (s_out.start_a + s_out.start_b)
    check(abs(float(sp_after.sum()) - 11.0) < 1e-6,
          "a club under eleven is corrected upward", f"{sp_after.sum():.6f}")
    # ... but a club with fewer than eleven listed players cannot be solved
    tiny = pd.DataFrame({"web_name": [f"T{i}" for i in range(8)], "team": ["T"] * 8,
                         "start_a": [5.0] * 8, "start_b": [5.0] * 8})
    t_out, _ = sp.apply_xi_constraint(tiny, verbose=False)
    check(np.allclose(t_out[["start_a", "start_b"]].values,
                      tiny[["start_a", "start_b"]].values),
          "a club with fewer than eleven listed players is skipped")

    print("\n" + ("XI CONSTRAINT PROPERTIES HOLD" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    _sys.exit(main())
