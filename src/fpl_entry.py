from __future__ import annotations
import config
"""
fpl_entry.py — pull a manager's live squad from the FPL site.
==============================================================
`squad_tracker` reads `data/my_squad.csv`, which is hand-edited and keyed on `web_name`.
That is the one join this project forbids everywhere else, tolerable there only because a
human types it once a week. This module removes the need: the FPL API publishes any
entry's picks, so the squad can be READ rather than transcribed, and mapped straight onto
`player_code`.

    entry/{id}/                     name, bank, squad value, overall points and rank
    entry/{id}/event/{gw}/picks/    the fifteen, their order, captain, vice, active chip

Both are PUBLIC — no login, no token, nothing of yours leaves this machine except the
entry id, which is already public on your own team page. Auth-only endpoints
(`my-team/{id}/`, which carries per-player SELLING price) are deliberately not touched;
see the caveat below for what that costs.

ELEMENT ID IS NOT STABLE — MAP IT
----------------------------------
`picks[].element` is the season's element id, which FPL reassigns between seasons and
which this project never joins on. `bootstrap-static` carries `elements[].code`, the
stable `player_code` used everywhere else, so every pick is mapped through it before it
touches anything. A pick that fails to map is reported, never silently dropped.

WHAT THE PUBLIC FEED CANNOT TELL YOU
-------------------------------------
SELLING PRICE. FPL sells a player for his purchase price plus half of any rise, so a
player who has gone up 0.3 sells for 0.1 less than he is listed at. That number lives
behind the authenticated `my-team` endpoint. Everything here uses CURRENT price, so a
transfer budget computed from it is optimistic for anyone holding risers. `entry_history`
does give the true aggregate — `value` is the sum of selling prices and `bank` the cash —
so the TOTAL is right even though the split across players is not, and that is the number
the budget constraint actually needs.

Run:  python src/fpl_entry.py --team-id 1234567
      python src/fpl_entry.py --team-id 1234567 --write-tracker   (also rewrite my_squad.csv)
Out:  data/my_squad_live.csv
"""
import json
import os
import urllib.request

import numpy as np
import pandas as pd

API = "https://fantasy.premierleague.com/api"
LIVE_SQUAD = os.path.join(config.DATA, "my_squad_live.csv")
ID_FILE = os.path.join(config.DATA, "fpl_entry_id.txt")
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def resolve_team_id(team_id=None):
    """Explicit argument, then FPL_TEAM_ID, then the remembered id."""
    if team_id:
        return int(team_id)
    env = os.environ.get("FPL_TEAM_ID", "").strip()
    if env.isdigit():
        return int(env)
    if os.path.exists(ID_FILE):
        t = open(ID_FILE).read().strip()
        if t.isdigit():
            return int(t)
    return None


def remember(team_id):
    with open(ID_FILE, "w") as fh:
        fh.write(str(int(team_id)))


def current_gw(bootstrap=None):
    """The gameweek whose picks are the ones you currently hold: the last one whose
    deadline has passed. Before GW1 there is none."""
    b = bootstrap or _get(f"{API}/bootstrap-static/")
    cur = [e["id"] for e in b.get("events", []) if e.get("is_current")]
    if cur:
        return int(cur[0])
    past = [e["id"] for e in b.get("events", []) if e.get("finished")]
    return int(max(past)) if past else None


def fetch(team_id=None, gw=None, verbose=True):
    """The live squad as a frame, plus the entry's metadata.

    Returns (squad_df, meta). Raises with a readable message rather than a stack trace
    for the two things that actually go wrong: a wrong id, and a gameweek the entry has
    no picks for."""
    tid = resolve_team_id(team_id)
    if tid is None:
        raise SystemExit(
            "No FPL team id. Pass --team-id, set FPL_TEAM_ID, or write it to "
            f"{ID_FILE}.\nIt is the number in your own team URL: "
            "fantasy.premierleague.com/entry/<THIS>/event/1")
    boot = _get(f"{API}/bootstrap-static/")
    gw = int(gw) if gw else current_gw(boot)
    if gw is None:
        raise SystemExit("The season has no completed gameweek yet — no picks to read.")
    try:
        entry = _get(f"{API}/entry/{tid}/")
    except Exception as e:
        raise SystemExit(f"Could not read entry {tid} ({type(e).__name__}). "
                         f"Check the id on your team page.")
    try:
        picks = _get(f"{API}/entry/{tid}/event/{gw}/picks/")
    except Exception as e:
        raise SystemExit(f"Entry {tid} has no picks for GW{gw} ({type(e).__name__}). "
                         f"A team created mid-season has none for earlier weeks.")

    el = {e["id"]: e for e in boot["elements"]}
    teams = {t["id"]: t["name"] for t in boot["teams"]}
    import core_insights as ci
    rows, unmapped = [], []
    for p in picks["picks"]:
        e = el.get(p["element"])
        if e is None:
            unmapped.append(p["element"]); continue
        rows.append({
            "gw": gw,
            "player_code": int(e["code"]),          # stable key; never the element id
            "web_name": e["web_name"],
            "pos": POS.get(e["element_type"]),
            "team": ci.norm_team(teams.get(e["team"], "")),
            "now_cost": e["now_cost"] / 10.0,
            "slot": p["position"],                  # 1-11 XI, 12-15 bench order
            "in_xi": p["position"] <= 11,
            "is_captain": bool(p["is_captain"]),
            "is_vice": bool(p["is_vice_captain"]),
            "multiplier": p["multiplier"],
            "bench_order": (p["position"] - 11) if p["position"] > 11 else np.nan,
        })
    sq = pd.DataFrame(rows).sort_values("slot").reset_index(drop=True)
    hist = picks.get("entry_history", {}) or {}
    meta = {
        "team_id": tid,
        "entry_name": entry.get("name"),
        "manager": " ".join(x for x in (entry.get("player_first_name"),
                                        entry.get("player_last_name")) if x),
        "gw": gw,
        "bank": (hist.get("bank", entry.get("last_deadline_bank")) or 0) / 10.0,
        "squad_value": (hist.get("value", entry.get("last_deadline_value")) or 0) / 10.0,
        "gw_points": hist.get("points"),
        "overall_points": entry.get("summary_overall_points"),
        "overall_rank": entry.get("summary_overall_rank"),
        "transfers_made": hist.get("event_transfers"),
        "transfer_cost": hist.get("event_transfers_cost"),
        "active_chip": picks.get("active_chip"),
        "unmapped": unmapped,
    }
    if verbose:
        print(f"[fpl-entry] {meta['entry_name']} (id {tid}) — GW{gw}: "
              f"{len(sq)} picks, {int(sq['in_xi'].sum())} in the XI")
        print(f"[fpl-entry] squad value {meta['squad_value']:.1f} + bank "
              f"{meta['bank']:.1f} = {meta['squad_value'] + meta['bank']:.1f} budget"
              + (f"; chip: {meta['active_chip']}" if meta["active_chip"] else ""))
        if unmapped:
            print(f"[fpl-entry] WARNING {len(unmapped)} picks did not map to a "
                  f"player_code: {unmapped}")
    return sq, meta


def write(sq, meta, path=None, write_tracker=False, verbose=True):
    """Persist the live squad. Writes `data/my_squad_live.csv` (player_code-keyed) and,
    optionally, the name-keyed `data/my_squad.csv` that `squad_tracker` reads."""
    path = path or LIVE_SQUAD
    sq.to_csv(path, index=False)
    if verbose:
        print(f"[fpl-entry] wrote {path}")
    if write_tracker:
        import squad_tracker as st
        t = pd.DataFrame({
            "gw": sq["gw"], "player": sq["web_name"],
            "in_xi": np.where(sq["in_xi"], "yes", "no"),
            "is_captain": np.where(sq["is_captain"], "yes", "no"),
            "is_vice": np.where(sq["is_vice"], "yes", "no"),
            "bench_order": sq["bench_order"],
        })
        t.to_csv(st.SQUAD_FILE, index=False)
        if verbose:
            print(f"[fpl-entry] wrote {st.SQUAD_FILE} (squad_tracker format)")
    return path


def load(path=None):
    """The last fetched squad, or None. Lets the explorer build offline."""
    path = path or LIVE_SQUAD
    if not os.path.exists(path):
        return None, None
    sq = pd.read_csv(path)
    meta_p = os.path.splitext(path)[0] + "_meta.json"
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else None
    return sq, meta


def selftest():
    """Offline. Exercises the mapping and the writer on a synthetic bootstrap; the
    network path is not selftested because a selftest that needs the internet is not
    one."""
    import tempfile
    boot_elements = [{"id": 7, "code": 111111, "web_name": "A.Player",
                      "element_type": 2, "team": 1, "now_cost": 55}]
    el = {e["id"]: e for e in boot_elements}
    picks = [{"element": 7, "position": 3, "multiplier": 1,
              "is_captain": False, "is_vice_captain": True},
             {"element": 99, "position": 12, "multiplier": 0,
              "is_captain": False, "is_vice_captain": False}]
    rows, unmapped = [], []
    for p in picks:
        e = el.get(p["element"])
        if e is None:
            unmapped.append(p["element"]); continue
        rows.append({"gw": 2, "player_code": int(e["code"]), "web_name": e["web_name"],
                     "pos": POS[e["element_type"]], "team": "X",
                     "now_cost": e["now_cost"] / 10.0, "slot": p["position"],
                     "in_xi": p["position"] <= 11, "is_captain": p["is_captain"],
                     "is_vice": p["is_vice_captain"], "multiplier": p["multiplier"],
                     "bench_order": (p["position"] - 11) if p["position"] > 11 else np.nan})
    sq = pd.DataFrame(rows)
    assert len(sq) == 1 and unmapped == [99], "unmapped picks must be reported, not dropped"
    assert sq["player_code"].iloc[0] == 111111, "must key on code, not element id"
    assert sq["pos"].iloc[0] == "DEF" and sq["now_cost"].iloc[0] == 5.5
    assert bool(sq["in_xi"].iloc[0]) is True
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sq.csv")
        write(sq, {}, path=p, verbose=False)
        back = pd.read_csv(p)
        assert len(back) == 1 and back["player_code"].iloc[0] == 111111
    # resolve_team_id precedence
    old = os.environ.pop("FPL_TEAM_ID", None)
    try:
        assert resolve_team_id(42) == 42
        os.environ["FPL_TEAM_ID"] = "77"
        assert resolve_team_id() == 77
        assert resolve_team_id(42) == 42, "explicit argument must beat the environment"
    finally:
        os.environ.pop("FPL_TEAM_ID", None)
        if old is not None:
            os.environ["FPL_TEAM_ID"] = old
    print("fpl_entry selftest OK")


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Pull your live FPL squad.")
    ap.add_argument("--team-id", default=None, help="the number in your team URL")
    ap.add_argument("--gw", type=int, default=None, help="default: the current gameweek")
    ap.add_argument("--write-tracker", action="store_true",
                    help="also rewrite data/my_squad.csv for squad_tracker")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    sq, meta = fetch(a.team_id, a.gw)
    write(sq, meta, write_tracker=a.write_tracker)
    with open(os.path.splitext(LIVE_SQUAD)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    if a.team_id:
        remember(a.team_id)
    print(sq[["slot", "web_name", "pos", "team", "now_cost", "in_xi",
              "is_captain"]].to_string(index=False))
