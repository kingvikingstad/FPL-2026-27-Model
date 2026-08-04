"""
ab_market_vs_recon.py — A/B the team model under the backward-looking E0_recon
vs the market-implied E0 (from betting_odds_ingest). Closes the loop on the
stale-Spurs-prior finding: shows which teams the market reprices, and rechecks
the clean-sheet fixtures flagged as overstated (Chelsea GW8 vs Spurs, Everton
GW4 at Spurs).

Mirrors the fit path in cs_fixtures.py exactly, so it's the reference wiring for
run_2627.py too. Requires the repo (core_insights, bayes_model, schedule_2627);
if absent it prints the run instruction and exits 0. `--selftest` validates the
diff/CS math offline with a synthetic posterior (no repo needed).

Usage on a networked box with the repo:
  python betting_odds_ingest.py --source oddsapi --fixtures odds.json --out /tmp/E0_market.csv
  python ab_market_vs_recon.py --recon /tmp/E0_recon.csv --market /tmp/E0_market.csv \
         --clubelo-recon 0.45 --clubelo-market 0.20
"""
from __future__ import annotations
import argparse, sys, numpy as np, pandas as pd

REPO = "/home/claude/repo/FPL-Core-Insights-main/data"
S = 6000
# fixtures to re-check (the ones flagged as leaning on the stale Spurs prior)
CS_RECHECK = [("Chelsea", "Tottenham", True, 8), ("Everton", "Tottenham", False, 4)]


def _fit(e0_path, clubelo_weight):
    import core_insights as ci, bayes_model
    from bayes_model import TeamModel
    d26, t26, _ = ci.load(base=f"{REPO}/2026-2027")
    elo = ci.to_elo_frame(t26); pclub = ci.promoted_prior_from_elo(t26)["per_club"]
    tm = TeamModel(promoted_per_club=pclub).fit(
        e0_path=e0_path, clubelo=elo, clubelo_weight=clubelo_weight)
    bayes_model.rng = np.random.default_rng(7)
    return tm.sample_2627(S=S)


def _team_strength(ts):
    """posterior-mean attack/defence per team from a sample_2627 dict."""
    idx = ts["idx"]
    att = {t: float(ts["att"][:, i].mean()) for t, i in idx.items()}
    dfn = {t: float(ts["dfn"][:, i].mean()) for t, i in idx.items()}
    return att, dfn


def _cs_prob(ts, team, opp, is_home):
    idx, mu, home = ts["idx"], ts["mu"], ts["home"]
    A, D = ts["att"], ts["dfn"]
    if team not in idx or opp not in idx:
        return np.nan, np.nan
    ti, oi = idx[team], idx[opp]
    hopp = 0.0 if is_home else home          # home edge accrues to opponent if they're home
    lam_against = np.exp(mu + hopp + A[:, oi] - D[:, ti])
    return float(np.mean(np.exp(-lam_against))), float(lam_against.mean())


def diff_report(ts_a, ts_b, label_a="recon", label_b="market"):
    aA, aD = _team_strength(ts_a); bA, bD = _team_strength(ts_b)
    teams = sorted(set(aA) & set(bA))
    rows = [{"team": t, f"att_{label_a}": aA[t], f"att_{label_b}": bA[t],
             "d_att": bA[t] - aA[t], f"dfn_{label_a}": aD[t], f"dfn_{label_b}": bD[t],
             "d_dfn": bD[t] - aD[t]} for t in teams]
    df = pd.DataFrame(rows)
    print("=" * 74)
    print(f"TEAM-STRENGTH REPRICING  ({label_b} - {label_a})   [att: +=more attacking]")
    print("=" * 74)
    print(df.reindex(df.d_att.abs().sort_values(ascending=False).index)
            .round(3).to_string(index=False))
    if "Tottenham" in set(df.team):
        s = df[df.team == "Tottenham"].iloc[0]
        print(f"\nSPURS focus: att {s['att_'+label_a]:+.3f} -> {s['att_'+label_b]:+.3f} "
              f"(Δ{s.d_att:+.3f}), dfn {s['dfn_'+label_a]:+.3f} -> {s['dfn_'+label_b]:+.3f} "
              f"(Δ{s.d_dfn:+.3f})")

    print("\n" + "=" * 74)
    print("CLEAN-SHEET RECHECK — fixtures flagged as leaning on the stale Spurs prior")
    print("=" * 74)
    for team, opp, is_home, gw in CS_RECHECK:
        csa, laa = _cs_prob(ts_a, team, opp, is_home)
        csb, lab = _cs_prob(ts_b, team, opp, is_home)
        ven = "H" if is_home else "A"
        print(f"GW{gw:2d} {team:8s} vs {opp:9s} ({ven}) | "
              f"P(CS) {label_a} {csa:.3f} -> {label_b} {csb:.3f} (Δ{csb-csa:+.3f}) | "
              f"xGA {laa:.2f} -> {lab:.2f}")
    return df


# ------------------------------------------------------------------ selftest
def _synthetic_ts(att_map, dfn_map, mu=0.1, home=0.25, S=4000, seed=1):
    rng = np.random.default_rng(seed)
    teams = list(att_map); idx = {t: i for i, t in enumerate(teams)}
    A = np.column_stack([rng.normal(att_map[t], 0.02, S) for t in teams])
    D = np.column_stack([rng.normal(dfn_map[t], 0.02, S) for t in teams])
    return {"idx": idx, "mu": np.full(S, mu), "home": np.full(S, home),
            "att": A, "dfn": D}


def selftest():
    # recon: Spurs a weak-ish attack (+0.05). market: Spurs repriced up (+0.30).
    base_att = {"Tottenham": 0.05, "Chelsea": 0.45, "Everton": -0.10}
    base_dfn = {"Tottenham": 0.30, "Chelsea": 0.40, "Everton": 0.35}
    mkt_att = dict(base_att, Tottenham=0.30)
    ts_a = _synthetic_ts(base_att, base_dfn)
    ts_b = _synthetic_ts(mkt_att, base_dfn)
    df = diff_report(ts_a, ts_b)
    d = df[df.team == "Tottenham"].iloc[0].d_att
    assert d > 0.2, "Spurs attack should rise materially"
    # CS vs Spurs must FALL when Spurs attack rises
    csa, _ = _cs_prob(ts_a, "Chelsea", "Tottenham", True)
    csb, _ = _cs_prob(ts_b, "Chelsea", "Tottenham", True)
    assert csb < csa, "CS vs a stronger Spurs must drop"
    print(f"\nSELFTEST OK: Spurs Δatt {d:+.3f}; Chelsea-vs-Spurs P(CS) {csa:.3f}->{csb:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="/tmp/E0_recon.csv")
    ap.add_argument("--market", default="/tmp/E0_market.csv")
    ap.add_argument("--clubelo-recon", type=float, default=0.45)
    ap.add_argument("--clubelo-market", type=float, default=0.20)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    try:
        import core_insights  # noqa: F401
    except Exception:
        print("Repo modules not importable in this environment.\n"
              "Run on a networked box with FPL-Core-Insights checked out, e.g.:\n"
              "  python ab_market_vs_recon.py --recon /tmp/E0_recon.csv --market /tmp/E0_market.csv")
        sys.exit(0)
    print(f">> fit A: {a.recon} (clubelo_weight={a.clubelo_recon})")
    ts_a = _fit(a.recon, a.clubelo_recon)
    print(f">> fit B: {a.market} (clubelo_weight={a.clubelo_market})")
    ts_b = _fit(a.market, a.clubelo_market)
    diff_report(ts_a, ts_b).to_csv("/mnt/user-data/outputs/ab_team_strength.csv", index=False)
    print("\nfull att/dfn diff -> ab_team_strength.csv")
