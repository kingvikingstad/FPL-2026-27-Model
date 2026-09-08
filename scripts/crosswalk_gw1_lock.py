import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config
"""
crosswalk_gw1_lock.py — give the GW1 lock the stable key it was written without.
================================================================================

`predictions/gw1_board_locked_2026-08-21.csv` carries `player`, `pos`, `team` and
`cost`. It has no `player_code` and no `id`, because it predates the column. It is
also the project's ONLY true pre-deadline lock, and therefore the only row of
`scoring_ledger.csv` that is a forecast in the strict sense.

THE FAILURE THIS EXISTS TO STOP
-------------------------------
`score_gw` falls back to joining that board on `player` + `team`. CLAUDE.md's hard
guard says never to do this, and the guard is right: FPL rewrites `web_name` during
the season. When a second Sangaré arrived, Ibrahim Sangaré became `I.Sangaré`, and
the GW1 lock — frozen with the old spelling — stopped matching him.

The consequence is not a crash. It is a score that DECAYS:

    scored 2026-08-26   n = 577   r = 0.5305
    scored 2026-09-08   n = 561   r = 0.5283      same board, same gameweek

Sixteen players fell out of the model's only forecast because a name string moved
underneath it, and the ledger recorded the drift as if it were a result. Left alone
the number keeps sliding, and the one piece of evidence for PROJECT_KNOWLEDGE §6.6
becomes less reproducible every week.

WHAT THIS WRITES, AND WHY IT IS A SIDECAR
-----------------------------------------
A file mapping (player, team) -> player_code for the rows of that lock, written
beside it. The lock is NOT modified: `predictions/` is the one directory that is
never regenerated, and a locked board edited after the fact is not a forecast. An
annotation that lives in a separate file cannot change what the board predicted.

Resolution is by exact (player, team) against `outputs/fpl_2627_players.csv`, a
roster snapshot from 2026-08-28 that carries BOTH the contemporaneous `player`
spelling and `player_code`. Being contemporaneous is the whole point — resolving
against today's roster is the very thing that fails.

    577 / 584 resolve exactly, with no duplicate expansion.
      1 / 584 resolves as a recorded web_name change (Sangaré -> I.Sangaré).
      6 / 584 do not resolve: absent from the 08-28 roster at that price, so a
              transfer or a price move sits between the two snapshots and no
              evidence in this tree distinguishes them. They are emitted with a
              null code and a reason rather than guessed at.

Every row carries `resolution`, so a later reader can drop the manual one and
re-derive the strict-subset answer without re-reading this docstring.

Run:  python scripts/crosswalk_gw1_lock.py
      python scripts/crosswalk_gw1_lock.py --selftest
"""
import pandas as pd

LOCK = _os.path.join(config.PREDICTIONS, "gw1_board_locked_2026-08-21.csv")
ROSTER = _os.path.join(config.OUTPUTS, "fpl_2627_players.csv")
# In a SUBDIRECTORY, not beside the lock. `score_gw._locked()` globs
# predictions/gw*_board_locked_*.csv, and a sidecar named after its lock matches that
# pattern — on 2026-09-08 it did, and the scorer picked the crosswalk as the board and
# died on a missing `mean`. glob does not cross a directory separator, so one level
# down is structurally out of reach rather than excluded by a filter someone must
# remember to keep in sync.
XWALK = _os.path.join(config.PREDICTIONS, "crosswalks")
OUT = _os.path.join(XWALK, "gw1_board_locked_2026-08-21.player_code.csv")

# A web_name FPL rewrote after the lock was taken. Keyed on the lock's spelling plus
# club so it cannot collide with the other Sangaré, who is at Brentford and is a
# different player. Recorded here rather than applied silently: this is the one row of
# the crosswalk that rests on judgment, and it is the row a sceptic should drop first.
MANUAL = {
    ("Sangaré", "Nott'm Forest"): (210462, "webname-change: Sangaré -> I.Sangaré"),
}


def build(lock=None, roster=None):
    """Return the crosswalk frame. Pure — takes frames or reads the real files."""
    L = pd.read_csv(LOCK) if lock is None else lock.copy()
    R = pd.read_csv(ROSTER) if roster is None else roster.copy()

    # Bracket indexing throughout: this frame crosses a module boundary into score_gw.
    keep = R[["player", "team", "player_code"]].drop_duplicates(["player", "team"])
    m = L[["player", "team", "pos", "cost"]].merge(keep, on=["player", "team"], how="left")
    m["resolution"] = m["player_code"].notna().map({True: "roster-exact", False: ""})

    for i in m.index[m["player_code"].isna()]:
        key = (m.at[i, "player"], m.at[i, "team"])
        if key in MANUAL:
            m.at[i, "player_code"], m.at[i, "resolution"] = MANUAL[key]
    m.loc[m["resolution"] == "", "resolution"] = "unresolved: absent from 2026-08-28 roster at this price"
    return m


def main():
    m = build()
    _os.makedirs(XWALK, exist_ok=True)
    m.to_csv(OUT, index=False)
    ok = int(m["player_code"].notna().sum())
    print(f"[crosswalk] {len(m)} lock rows -> {ok} resolved, {len(m) - ok} unresolved")
    print(m["resolution"].value_counts().to_string())
    print(f"[wrote] {OUT}")
    return 0


def selftest():
    """Offline, on synthetic fixtures. Verifies the two properties that matter: an
    exact match resolves, and a name the roster no longer spells the same way does NOT
    silently attach to some other player at the club."""
    lock = pd.DataFrame({
        "player": ["Saka", "Sangaré", "Ghost"],
        "team": ["Arsenal", "Nott'm Forest", "Arsenal"],
        "pos": ["MID", "MID", "FWD"], "cost": [10.0, 5.0, 4.5]})
    roster = pd.DataFrame({
        "player": ["Saka", "I.Sangaré", "Odegaard"],
        "team": ["Arsenal", "Nott'm Forest", "Arsenal"],
        "player_code": [223340, 210462, 184029]})
    m = build(lock, roster)

    assert len(m) == 3, f"row count changed: {len(m)}"          # no duplicate expansion
    r = dict(zip(m["player"], m["player_code"]))
    assert r["Saka"] == 223340, r
    # the renamed player resolves ONLY through MANUAL, never by falling through to
    # whoever else happens to be at the club
    assert r["Sangaré"] == 210462, r
    assert pd.isna(r["Ghost"]), "an absent player must stay null, not borrow a code"
    res = dict(zip(m["player"], m["resolution"]))
    assert res["Saka"] == "roster-exact", res
    assert res["Sangaré"].startswith("webname-change"), res
    assert res["Ghost"].startswith("unresolved"), res
    # and the codes must be unique, or the join downstream fans out
    got = m["player_code"].dropna()
    assert got.is_unique, "duplicate player_code in crosswalk"
    print("crosswalk_gw1_lock selftest ok — exact, renamed and absent all behave")
    return 0


if __name__ == "__main__":
    _sys.exit(selftest() if "--selftest" in _sys.argv else main())
