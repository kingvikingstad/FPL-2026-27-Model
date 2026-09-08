import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
track.py — the shell: enter your squad, compare it to the model, keep the score.
=================================================================================
    python scripts/track.py --init            create data/my_squad.csv to fill in
    python scripts/track.py                   validate + compare against the proposal
    python scripts/track.py --gw 2            a specific gameweek
    python scripts/track.py --result 70 68    record what each squad actually scored
    python scripts/track.py --ledger          the running record

The comparison is projection-vs-projection, which is cheap talk until results land.
`--result` is the part that matters: it builds a paired record of your picks against the
model's on the same fixtures, which is the only way either gets held to account.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import squad_tracker as st

SOLVER_INPUTS = _os.path.join(config.OUTPUTS, "fpl_2627_solver_inputs.csv")
# three tracked squads
HYBRID = _os.path.join(config.PLANS, "plan_constrained_squad.csv")
MODEL = _os.path.join(config.PLANS, "plan_unconstrained_squad.csv")


def board_for(gw):
    d = pd.read_csv(SOLVER_INPUTS).rename(columns={"model_pts": "ep", "cost": "price"})
    d = d[d.gw == gw].dropna(subset=["ep", "price", "pos", "team"])
    return d.drop_duplicates("player_code")


def main():
    args = _sys.argv[1:]
    gw = int(args[args.index("--gw") + 1]) if "--gw" in args else 1
    bb = "--bench-boost" in args

    if "--init" in args:
        p = st.SQUAD_FILE
        if _os.path.exists(p) and "--force" not in args:
            print(f"{p} already exists — pass --force to overwrite it")
            return
        st.template(gw).to_csv(p, index=False)
        print(f"wrote a blank sheet to {p}")
        print("Fill in `player` for all 15, set in_xi for 11, one is_captain, "
              "one is_vice, then re-run without --init.")
        return

    if "--ledger" in args:
        L = st.ledger()
        if L.empty:
            print("no results recorded yet — use --result MY MODEL after a gameweek")
            return
        show = ["gw", "my_points", "hybrid_points", "model_points",
                "vs_hybrid", "vs_model", "hybrid_vs_model"]
        print(L[[c for c in show if c in L.columns]].round(1).to_string(index=False))
        n = len(L)
        print(f"\n  gameweeks recorded : {n}")
        for lab, col in (("yours ", "my_points"), ("hybrid", "hybrid_points"),
                         ("model ", "model_points")):
            s, m = L[col].sum(), L[col].mean()
            k = int(L[col].notna().sum())
            print(f"  {lab}  total {s:7.1f}   mean {m:6.1f}   ({k}/{n} weeks recorded)")
        if n >= 2:
            print(f"  weeks you beat the hybrid : {(L['vs_hybrid'] > 0).sum()}/"
                  f"{int(L['vs_hybrid'].notna().sum())}")
            print(f"  weeks you beat the model  : {(L['vs_model'] > 0).sum()}/"
                  f"{int(L['vs_model'].notna().sum())}")
        if n < 5:
            print("\n  Too few gameweeks to mean anything yet. Single-gameweek FPL "
                  "scores are mostly variance;\n  a handful of weeks cannot separate "
                  "skill from noise in either direction.")
        return

    if "--result" in args:
        i = args.index("--result")
        vals = []
        for a in args[i + 1:i + 4]:
            try:
                vals.append(float(a))
            except ValueError:
                break
        if len(vals) < 1:
            print("usage: --result MINE [HYBRID] [MODEL]")
            return
        mine_p = vals[0]
        hyb_p = vals[1] if len(vals) > 1 else None
        mod_p = vals[2] if len(vals) > 2 else None
        st.append_result(gw, mine_p, hyb_p, mod_p)
        parts = [f"you {mine_p:.0f}"]
        if hyb_p is not None: parts.append(f"hybrid {hyb_p:.0f}")
        if mod_p is not None: parts.append(f"model {mod_p:.0f}")
        print(f"recorded GW{gw}: " + ", ".join(parts))
        if hyb_p is None or mod_p is None:
            print("  (unrecorded tracks stored as blank, not zero)")
        print(f"  -> {st.RESULTS_FILE}")
        return

    # ---------- validate and compare ----------
    board = board_for(gw)
    entered = st.load_squad(gw=gw)
    if entered.empty:
        print(f"no rows for GW{gw} in {st.SQUAD_FILE}")
        return
    codes, problems = st.resolve_names(entered["player"], board)
    entered["player_code"] = codes
    for p in problems:
        print(f"  !! {p}")
    unresolved = entered[entered.player_code.isna()]
    if len(unresolved):
        print(f"\n{len(unresolved)} name(s) unresolved — fix them and re-run.")
        return

    mine = entered.merge(board, on="player_code", how="left", suffixes=("", "_b"))
    bad = st.validate(mine)
    print("=" * 88)
    print(f"YOUR SQUAD — GW{gw}" + ("   [bench boost]" if bb else ""))
    print("=" * 88)
    if bad:
        print("  RULE PROBLEMS:")
        for b in bad:
            print(f"    !! {b}")
        print()
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    s = mine.assign(_o=mine["pos"].map(order)).sort_values(
        ["in_xi", "_o", "ep"], ascending=[False, True, False])
    for _, r in s.iterrows():
        mark = "C" if r["is_captain"] else ("v" if r["is_vice"] else
                                            (" " if r["in_xi"] else "b"))
        print(f"  {mark} {r['pos']:3s} {r['display_name']:22s} {r['team']:15s} "
              f"£{r['price']:4.1f}m  {r['ep']:5.2f}")
    ms = st.score_squad(mine, bench_boost=bb)
    print(f"\n  cost £{mine['price'].sum():.1f}m   projected "
          f"{ms['total_ep']:.2f}  (XI {ms['xi_ep']:.2f} + captain {ms['captain_ep']:.2f}"
          + (f" + bench {ms['bench_ep']:.2f}" if bb else "") + ")")

    def load_track(path, label):
        if not _os.path.exists(path):
            print(f"\n  (no {label} squad at {path} — run scripts/plan_constrained.py)")
            return None, None
        t = pd.read_csv(path)
        keep = [c for c in t.columns if c not in ("ep", "pos", "price", "team",
                                                  "display_name")]
        t = t[keep].merge(board[["player_code", "display_name", "pos", "price", "ep"]],
                          on="player_code", how="left")
        return t, st.score_squad(t, bench_boost=bb)

    hyb, hs = load_track(HYBRID, "hybrid")
    mod, mods = load_track(MODEL, "unconstrained model")

    print("\n" + "=" * 88)
    print("THREE SQUADS, SAME GAMEWEEK")
    print("=" * 88)
    print(f"  {'squad':<34}{'projected':>10}{'vs yours':>10}   captain")
    def capname(t):
        if t is None or "is_captain" not in t:
            return "?"
        c = t.loc[t.is_captain, "display_name"]
        return c.iloc[0] if len(c) else "?"
    print(f"  {'yours':<34}{ms['total_ep']:>10.2f}{'':>10}   {capname(mine)}")
    if hs:
        print(f"  {'hybrid (your constraints)':<34}{hs['total_ep']:>10.2f}"
              f"{ms['total_ep'] - hs['total_ep']:>+10.2f}   {capname(hyb)}")
    if mods:
        print(f"  {'model (unconstrained)':<34}{mods['total_ep']:>10.2f}"
              f"{ms['total_ep'] - mods['total_ep']:>+10.2f}   {capname(mod)}")
    if hs and mods:
        print(f"\n  cost of your constraints (hybrid - model): "
              f"{hs['total_ep'] - mods['total_ep']:+.2f}")
        print(f"  cost of your own picks   (yours  - hybrid): "
              f"{ms['total_ep'] - hs['total_ep']:+.2f}")
        print("  Those are two different decisions and they are worth keeping apart:\n"
              "  the first is what your stated preferences cost, the second is what your\n"
              "  deviations from them cost.")

    for t, label in ((hyb, "hybrid"), (mod, "unconstrained model")):
        if t is None:
            continue
        c = st.compare(mine, t)
        print(f"\n  vs {label}: {c['shared']}/15 shared")
        if len(c["only_mine"]):
            print("    only yours   : " + ", ".join(
                f"{r.display_name} ({r.ep:.2f})" for r in c["only_mine"].itertuples()))
        if len(c["only_proposed"]):
            print(f"    only {label:<8}: " + ", ".join(
                f"{r.display_name} ({r.ep:.2f})" for r in c["only_proposed"].itertuples()))

    print("\n  A projected gap is not a verdict. Record the real scores with "
          "--result once\n  the gameweek is over; that is the comparison that counts.")


if __name__ == "__main__":
    main()
