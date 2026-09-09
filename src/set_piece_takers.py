from __future__ import annotations
import config
"""
set_piece_takers.py — projected set-piece duty from Fantasy Football Scout
==========================================================================
Closes the coverage gap measured in studies/penalty_assignment.py. FPL's own
`penalties_order` is the more PRECISE signal (85.7% precision against 45.0% for measured
history) but it covers only ~39% of the penalties actually taken, because roughly six
clubs carry no declared first-choice taker at all. At ~16 points a season per settled
taker, that missing coverage was the largest unexploited edge in the projection.

This supplies a projected taker for all 20 clubs. It does NOT replace the FPL flag where
one exists — FPL's is official and in-season, FFS's is a pre-season projection. Precedence
is FPL first, FFS only to fill a hole, which keeps the more precise signal on top and uses
the wider one exactly where the model previously had nothing.

MANUAL OVERRIDES ON TOP OF THE PROJECTION
------------------------------------------
`data/set_piece_takers.csv` is the FFS 26/27 projection with one deliberate correction:

  LIVERPOOL PEN — Isak promoted to #1 ahead of Szoboszlai (2026-08-20, manager's call).
    The two sources disagreed from the start. FFS projected Szoboszlai; Rotowire, Goal
    and several Premier League set-piece guides all listed Isak. The disagreement was
    flagged rather than auto-resolved because a first-choice taker is worth roughly 16
    points a season, and it has now been settled in Isak's favour.

Overrides belong in this file, not in code, so the resolver and its within-club matching
apply to them identically. Anything changed here should be recorded above with a date and
a reason, because a silent edit to a scraped data file is indistinguishable from a
scraping error six weeks later.

THE PRE-SEASON JUSTIFICATION HAS NOW LAPSED  [2026-08-24]
----------------------------------------------------------
This projection overrode FPL because, at pre-season, FPL's `penalties_order` was a stale
carryover and left roughly six clubs blank. After GW1 neither is true. Measured on the
2026-08-24 snapshot, FPL declares a #1 for **20/20 clubs on all three roles**, and ranks
more players than the projection does:

                        FPL in-season      this projection
    penalties           20/20, 66 ranked   20/20, 51 rows
    direct free kicks   20/20, 56 ranked   18/20, 30 rows
    corners / indirect  20/20, 81 ranked   20/20, 49 rows

So the module's own stated rule — "FPL first, FFS only to fill a hole" — now points the
other way, and the runners default to `FPL_SETPIECE=fpl`. Where the two disagree, FPL's
in-season declaration wins:

    Bournemouth   Kroupi Jr   (projection said Kluivert)
    Fulham        Gonzalo Garcia (projection said Muniz, who is no longer ranked at all)
    Liverpool     Szoboszlai  (projection said Isak — see the manual override above)

That last one REVERSES the 2026-08-20 manager's call, and it is now settled by direct
observation rather than by anyone's declaration: **Szoboszlai took Liverpool's penalty in
the 90th minute against Newcastle and scored it** (xG 0.79, xGOT 0.99).

CORRECTION [2026-08-24]: an earlier version of this note said no Liverpool penalty had
been awarded in GW1 and that nothing directly settled the question. That was wrong. It
came from looking in `player_gameweek_stats.csv`, the FPL element summary, which carries
`penalties_saved` and `penalties_missed` but NO `penalties_scored` — a converted penalty
is recorded there only as a goal. So a miss-only detector finds misses.

The right file is `By Gameweek/GW*/playermatchstats.csv`, the fotmob per-match file, which
DOES carry `penalties_scored` alongside `penalties_missed` and shows both GW1 penalties
directly. `observed_takers()` reads that, and cross-checks `shots.csv`
(`situation == "penalty"`) for situation-level detail. GW1 contained TWO penalties, not
the one the first method found. Set `FPL_SETPIECE=override` to restore the projection,
including Isak.

NAME RESOLUTION IS THE WHOLE RISK
---------------------------------
The source is a web page read through an LLM, and every scrape in this project has
contained at least one error. Names are matched WITHIN CLUB against the season's actual
squad, so a wrong name cannot silently attach to a player at another club, and `resolve()`
reports the match rate and lists every failure rather than dropping it quietly. Nothing is
matched fuzzily across clubs.

Run:  python src/set_piece_takers.py --check     (needs FPL_DATA)
"""
import os
import unicodedata
import pandas as pd

ROLES = ("PEN", "FK", "CORNER")


# Letters NFKD does NOT decompose — they are distinct characters, not letter+diacritic.
# Without these, 'Ødegaard' and 'Groß' never match 'Odegaard' and 'Gross'.
_FOLD = {"ø": "o", "æ": "ae", "ß": "ss", "đ": "d", "ð": "d", "ł": "l",
         "þ": "th", "œ": "oe", "ı": "i"}


def _key(s):
    """Casefold, fold non-decomposing letters, strip accents, and treat '.' as a space
    so 'N.Williams' and 'Kroupi.Jr' tokenise the way a reader would expect."""
    s = str(s).lower().replace(".", " ")
    s = "".join(_FOLD.get(c, c) for c in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join("".join(c for c in s if c.isalnum() or c == " ").split())


def _tokens(k):
    """Name tokens worth matching on, ignoring generational suffixes."""
    return [t for t in k.split() if t not in ("jr", "sr", "ii", "iii")]


def load(path=None) -> pd.DataFrame:
    d = pd.read_csv(path or config.SET_PIECE_TAKERS)
    bad = set(d["role"]) - set(ROLES)
    if bad:
        raise RuntimeError(f"unknown roles {sorted(bad)}; expected {ROLES}")
    d["key"] = d["name"].map(_key)
    return d


def resolve(players: pd.DataFrame, path=None, verbose=True) -> pd.DataFrame:
    """Attach player_code/web_name by matching name WITHIN club.

    `players` needs columns: web_name, team, player_code.
    Returns the taker table with `player_code` filled where resolved.
    """
    t = load(path)
    p = players.copy()
    p["key"] = p["web_name"].map(_key)
    p["last"] = p["key"].map(lambda k: (_tokens(k) or [""])[-1])

    out = []
    for club, g in t.groupby("club"):
        squad = p[p["team"] == club]
        if squad.empty:
            for _, r in g.iterrows():
                out.append({**r.to_dict(), "player_code": None, "why": "club not in squad list"})
            continue
        exact = dict(zip(squad["key"], squad["player_code"]))
        # every fallback requires UNIQUENESS inside the club, so an ambiguous surname
        # resolves to nothing rather than to the wrong player
        uniq = lambda s: s.iloc[0] if len(s) == 1 else None
        bylast = squad.groupby("last")["player_code"].agg(uniq).to_dict()
        bykey = squad.groupby("key")["player_code"].agg(uniq).to_dict()
        for _, r in g.iterrows():
            toks = _tokens(r["key"])
            code, why = exact.get(r["key"]), "exact"
            if code is None and toks:
                code = bylast.get(toks[-1])
                why = "surname, unique in club"
            if code is None:
                # the source may carry a fuller name than FPL's short one
                # ("Amad Diallo" -> "Amad"); require an exact whole-name hit on a token
                hits = {bykey[t] for t in toks if bykey.get(t) is not None}
                if len(hits) == 1:
                    code, why = hits.pop(), "token matches full FPL name"
            if code is None:
                why = "unresolved"
            out.append({**r.to_dict(), "player_code": code, "why": why})
    R = pd.DataFrame(out)

    ok = R["player_code"].notna()
    if verbose:
        print(f"[set-piece] resolved {ok.sum()}/{len(R)} entries ({ok.mean():.0%}) "
              f"across {R.club.nunique()} clubs")
        miss = R[~ok]
        if len(miss):
            print(f"[set-piece] UNRESOLVED ({len(miss)}) — these are dropped, not guessed:")
            for _, r in miss.iterrows():
                print(f"    {r['club']:16s} {r['role']:6s} #{r['order']}  {r['name']}")
        pen1 = R[(R.role == "PEN") & (R.order == 1) & ok]
        print(f"[set-piece] first-choice penalty takers resolved for "
              f"{pen1.club.nunique()}/20 clubs")
    return R


def pen1_codes(players: pd.DataFrame, path=None, mode="override", verbose=False,
               observed=None, upto_gw=None):
    """`player_code`s of the FIRST-CHOICE penalty taker at each club.

    Use this instead of reading `pen_order == 1` off the name-keyed signals frame. 15
    web_names in the 26/27 squad belong to players at two different clubs, and one of
    them — 'Palmer' — is a first-choice taker at Chelsea while Ipswich's projected taker
    is Clarke. A name-keyed lookup cannot separate them, so Ipswich was handed TWO
    order-1 takers and simulated roughly double the penalty xG it should. `player_code`
    is the project's stable join key and resolves it exactly.

    mode="override"/"fill" use the projection; mode="off" (alias "fpl") returns FPL's own
    declared `penalties_order == 1`, also keyed on player_code. "off" is a poor name for
    that — it means the PROJECTION is off, not the set-piece layer — so "fpl" is accepted
    and is what the runners now pass.
    """
    p = players.copy()
    mode_l = str(mode).lower()
    if mode_l in ("off", "0", "fpl", "observed"):
        po = pd.to_numeric(p.get("penalties_order"), errors="coerce")
        codes = set(p.loc[po == 1, "player_code"].dropna())
        if mode_l != "observed":
            return codes
        # OBSERVATION OUTRANKS DECLARATION. A declared order is what a club says will
        # happen; a penalty on the pitch is what did. Where a club has actually taken one
        # this season, its most frequent taker replaces the declared #1 for that club.
        # injected by the caller in tests; read from the season otherwise
        if observed is not None:
            obs = observed
        else:
            try:
                # Bounded by `upto_gw` for the same reason press_factor is: unbounded,
                # this reads penalties taken in the very gameweek a rebuilt board is
                # meant to be forecasting.
                obs = observed_takers(upto_gw=upto_gw, verbose=False)
            except Exception as e:
                if verbose:
                    print(f"[set-piece] observed takers unavailable ({type(e).__name__}); "
                          f"using declared order")
                return codes
        if not len(obs):
            return codes
        club_of = dict(zip(p["player_code"], p["team"]))
        changed = []
        for club, g in obs.groupby("team"):
            top = g.sort_values("taken", ascending=False).iloc[0]
            drop = {c for c in codes if club_of.get(c) == club}
            if drop == {top["player_code"]}:
                continue                       # observation agrees with the declaration
            codes -= drop
            codes.add(top["player_code"])
            was = [p.loc[p.player_code == c, "web_name"].iloc[0] for c in drop
                   if (p.player_code == c).any()]
            changed.append((club, ", ".join(was) or "(none)", top["web_name"],
                            int(top["taken"])))
        if verbose and changed:
            print(f"[set-piece] observation overrode the declared #1 at "
                  f"{len(changed)} club(s):")
            for club, was, now, n in changed:
                print(f"    {club:14s} {was} -> {now} ({n} taken)")
        return codes
    R = resolve(p, path, verbose=verbose)
    R = R[R["player_code"].notna()]
    codes = set(R.loc[(R.role == "PEN") & (R.order == 1), "player_code"])
    if str(mode).lower() == "fill":
        po = pd.to_numeric(p.get("penalties_order"), errors="coerce")
        covered = set(R.loc[R.role == "PEN", "club"])
        keep = p.loc[(po == 1) & (~p["team"].isin(covered)), "player_code"].dropna()
        codes |= set(keep)
    return codes


def apply_to_signals(sig: pd.DataFrame, players: pd.DataFrame, path=None,
                     mode="override", verbose=True) -> pd.DataFrame:
    """Apply projected set-piece duty to the live signals frame.

    mode="override" (default) — the projection wins. Correct for a PRE-SEASON board:
        FPL's `penalties_order` at this point is largely carried over from last season,
        while the projection is made for the season being modelled. This is what changes
        player values, so every disagreement is logged rather than applied silently.
    mode="fill" — FPL's flag wins and the projection only covers clubs it leaves blank.
        The conservative option; keeps the more precise signal on top.

    [MEASURED 2026-08-11 on the 26/27 squad] the two sources agree on 14 of 20 clubs,
    disagree on 5, and FPL is blank for 1. The disagreements move real penalty EV, which
    is worth ~16 points a season per taker — so this is not a cosmetic switch.
    """
    R = resolve(players, path, verbose=verbose)
    R = R[R["player_code"].notna()]
    code_to_name = dict(zip(players["player_code"], players["web_name"]))
    R["web_name"] = R["player_code"].map(code_to_name)
    R["skey"] = R["web_name"].str.lower().str.strip()
    # Club lookup for EVERY player in the squad, not just the ones the projection lists.
    # Building it from R (the taker table) was a bug: a player absent from the projection
    # got club None, so `t in covered` was never true and they silently KEPT their stale
    # FPL flag — the exact demotion this mode exists to perform. It left three clubs with
    # two order-1 penalty takers (Liverpool: Szoboszlai AND Isak), double-counting penalty
    # xG. Names shared by players at two clubs are left out deliberately: `sig` is keyed on
    # name alone and cannot disambiguate them, so the conservative action is to keep the
    # existing flag rather than risk clearing the wrong player's duty.
    _pk = players["web_name"].str.lower().str.strip()
    _clubs = pd.DataFrame({"k": _pk, "team": players["team"]}).drop_duplicates()
    _nclub = _clubs.groupby("k")["team"].nunique()
    club_of = _clubs[_clubs["k"].map(_nclub) == 1].set_index("k")["team"].to_dict()
    _ambiguous = int((_nclub > 1).sum())

    s = sig.copy()
    if "name" not in s.columns:
        return s
    s["_k"] = s["name"].str.lower().str.strip()
    col = {"PEN": "pen_order", "FK": "fk_order", "CORNER": "corner_order"}

    changed = {}
    for role, c in col.items():
        if c not in s.columns:
            s[c] = pd.NA
        want = dict(zip(R.loc[R.role == role, "skey"], R.loc[R.role == role, "order"]))
        if mode == "fill":
            new = [want.get(k, v) if pd.isna(v) else v for k, v in zip(s["_k"], s[c])]
        else:
            # the projection is authoritative: anyone it does not list for this role in a
            # club it DOES cover loses the duty, otherwise stale FPL flags would survive
            covered = set(R.loc[R.role == role, "club"])
            teams = s["_k"].map(lambda k: club_of.get(k))
            new = [want.get(k) if (want.get(k) is not None or t in covered) else v
                   for k, v, t in zip(s["_k"], s[c], teams)]
        before = s[c].copy()
        s[c] = new
        changed[c] = int((before.fillna(-1) != s[c].fillna(-1)).sum())

    if verbose:
        print(f"[set-piece] mode={mode}; rows changed: "
              + ", ".join(f"{k} {v}" for k, v in changed.items()))
        if _ambiguous:
            print(f"[set-piece] {_ambiguous} web_names are shared by two clubs — their "
                  f"existing flags are left alone (sig cannot disambiguate on name)")
        if mode == "override":
            for c in col.values():
                dup = s.loc[s[c] == 1, "_k"].map(club_of).value_counts()
                dup = dup[dup > 1]
                assert dup.empty, (f"{c}: clubs with more than one order-1 taker after "
                                   f"override: {dup.to_dict()}")
        if mode == "override" and "pen_order" in sig.columns:
            fpl1 = set(sig.loc[sig["pen_order"] == 1, "name"].str.lower().str.strip())
            ffs1 = set(R.loc[(R.role == "PEN") & (R.order == 1), "skey"])
            lost, gained = sorted(fpl1 - ffs1), sorted(ffs1 - fpl1)
            if lost or gained:
                print(f"[set-piece] penalty duty REASSIGNED ({len(lost)} lost, "
                      f"{len(gained)} gained)")
                print(f"[set-piece]   lost   : {', '.join(lost)}")
                print(f"[set-piece]   gained : {', '.join(gained)}")
    return s.drop(columns=["_k"])


def observed_takers(season="2026-2027", upto_gw=None, base=None, verbose=True):
    """Who has ACTUALLY taken a penalty this season, from shot-level data.

    Stronger evidence than any declared list. A club's `penalties_order` is what the club
    (or FPL) says will happen; a penalty on the pitch is what did.

    Source: `By Gameweek/GW*/playermatchstats.csv` (fotmob per-match), which carries BOTH
    `penalties_scored` and `penalties_missed` keyed on player_id.

    Not `player_gameweek_stats.csv`. That is the FPL element summary and has
    `penalties_saved` / `penalties_missed` but no `penalties_scored`, because a converted
    penalty is recorded there only as a goal — so it finds misses and nothing else. That
    mistake is why an earlier version of this module reported one GW1 penalty when there
    were two, and named the wrong club as unsettled.

    `shots.csv` (`situation == "penalty"`) is read as a cross-check and to supply anything
    playermatchstats lacks; it carries situation and placement but its `player_id` is
    sometimes null and `player_name` is empty throughout, so it is the secondary source.

    Returns one row per (club, taker), most frequent taker first.
    """
    import glob
    import core_insights as ci
    base = base or config.repo(season)
    roster = pd.read_csv(os.path.join(base, "players.csv"))
    teams = pd.read_csv(os.path.join(base, "teams.csv"))
    roster = roster.merge(teams[["code", "name"]], left_on="team_code",
                          right_on="code", how="left")
    roster["team"] = roster["name"].map(ci.norm_team)
    look = roster[["player_id", "player_code", "web_name", "team"]]

    rows, extra = [], 0
    for d in sorted(glob.glob(os.path.join(base, "By Gameweek", "GW*"))):
        gw = os.path.basename(d)[2:]
        if upto_gw is not None and gw.isdigit() and int(gw) > int(upto_gw):
            continue
        f = os.path.join(d, "playermatchstats.csv")
        if os.path.exists(f):
            pm = pd.read_csv(f)
            cols = [c for c in ("penalties_scored", "penalties_missed") if c in pm.columns]
            if cols:
                hit = pm[(pm[cols].fillna(0) > 0).any(axis=1)]
                for _, r in hit.iterrows():
                    m = look[look["player_id"] == r["player_id"]]
                    if not len(m):
                        continue
                    h = m.iloc[0]
                    sc = int(r.get("penalties_scored", 0) or 0)
                    ms = int(r.get("penalties_missed", 0) or 0)
                    for _ in range(sc):
                        rows.append({"gw": gw, "player_code": h["player_code"],
                                     "web_name": h["web_name"], "team": h["team"],
                                     "outcome": "goal", "scored": True})
                    for _ in range(ms):
                        rows.append({"gw": gw, "player_code": h["player_code"],
                                     "web_name": h["web_name"], "team": h["team"],
                                     "outcome": "missed", "scored": False})
        # cross-check against shot data; only ADD takers playermatchstats did not name
        sf = os.path.join(d, "shots.csv")
        if os.path.exists(sf):
            sh = pd.read_csv(sf)
            if "situation" in sh.columns:
                for _, r in sh[sh["situation"] == "penalty"].iterrows():
                    if pd.isna(r.get("player_id")):
                        continue
                    m = look[look["player_id"] == r["player_id"]]
                    if not len(m):
                        continue
                    h = m.iloc[0]
                    if any(x["gw"] == gw and x["player_code"] == h["player_code"]
                           for x in rows):
                        continue
                    extra += 1
                    rows.append({"gw": gw, "player_code": h["player_code"],
                                 "web_name": h["web_name"], "team": h["team"],
                                 "outcome": r.get("outcome"),
                                 "scored": r.get("outcome") == "goal"})

    if not rows:
        if verbose:
            print("[set-piece] no penalties observed yet this season")
        return pd.DataFrame(columns=["team", "player_code", "web_name", "taken", "scored"])
    R = pd.DataFrame(rows)
    agg = (R.groupby(["team", "player_code", "web_name"], as_index=False)
             .agg(taken=("outcome", "size"), scored=("scored", "sum"))
             .sort_values(["team", "taken"], ascending=[True, False]))
    if verbose:
        print(f"[set-piece] observed {int(agg.taken.sum())} penalties by "
              f"{len(agg)} players at {agg.team.nunique()} clubs"
              + (f"; {extra} found only in shot data" if extra else ""))
        for _, r in agg.iterrows():
            print(f"    {r['team']:14s} {r['web_name']:16s} taken {int(r['taken'])}, "
                  f"scored {int(r['scored'])}")
    return agg

def selftest():
    """Offline, on synthetic gameweek folders. Covers the observation-over-declaration
    precedence and the reason it exists: a SCORED penalty is invisible in
    penalties_missed / penalties_saved, so a miss-only detector finds the wrong takers."""
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp()
    gw = os.path.join(tmp, "By Gameweek", "GW1")
    os.makedirs(gw)
    pd.DataFrame({"player_id": [1, 2, 3, 4], "player_code": [101, 102, 103, 104],
                  "web_name": ["Alpha", "Bravo", "Charlie", "Delta"],
                  "team_code": [10, 10, 20, 20]}).to_csv(
        os.path.join(tmp, "players.csv"), index=False)
    pd.DataFrame({"code": [10, 20], "name": ["Arsenal", "Everton"]}).to_csv(
        os.path.join(tmp, "teams.csv"), index=False)
    # Alpha SCORES a penalty, Charlie MISSES one. The FPL element summary records only
    # the miss, which is the trap this function exists to avoid; the fotmob per-match
    # file records both.
    pd.DataFrame({"player_id": [1, 2, 3, 4], "match_id": ["m1"] * 2 + ["m2"] * 2,
                  "penalties_scored": [1, 0, 0, 0],
                  "penalties_missed": [0, 0, 1, 0]}).to_csv(
        os.path.join(gw, "playermatchstats.csv"), index=False)
    # Delta appears ONLY in shot data — the cross-check must pick him up
    pd.DataFrame({
        "match_id": ["m1", "m1", "m2"], "shot_index": [1, 2, 1], "minute": [10, 20, 30],
        "is_home": [True, True, False], "player_id": [1.0, 2.0, 4.0],
        "situation": ["penalty", "regular", "penalty"],
        "outcome": ["goal", "miss", "goal"], "xg": [0.79, 0.05, 0.79],
    }).to_csv(os.path.join(gw, "shots.csv"), index=False)
    pd.DataFrame({"id": [1, 2, 3, 4], "penalties_missed": [0, 0, 1, 0],
                  "penalties_saved": [0, 0, 0, 0], "minutes": [90, 90, 90, 90]}).to_csv(
        os.path.join(gw, "player_gameweek_stats.csv"), index=False)

    obs = observed_takers(base=tmp, verbose=False)
    got = dict(zip(obs.web_name, obs.taken))
    assert got == {"Alpha": 1, "Charlie": 1, "Delta": 1}, f"observed takers wrong: {got}"
    assert int(obs.loc[obs.web_name == "Alpha", "scored"].iloc[0]) == 1
    assert int(obs.loc[obs.web_name == "Charlie", "scored"].iloc[0]) == 0,         "a missed penalty is taken but not scored"
    # THE POINT: penalties_missed alone finds Charlie and misses Alpha entirely, which is
    # exactly the error this replaced.
    fpl_only = pd.read_csv(os.path.join(gw, "player_gameweek_stats.csv"))
    miss_only = set(fpl_only.loc[fpl_only.penalties_missed > 0, "id"])
    assert miss_only == {3}, "the FPL summary sees only the miss"
    assert "Alpha" in set(obs.web_name), "the SCORED penalty must still be recovered"
    # and the shot-data cross-check adds a taker playermatchstats never named
    assert "Delta" in set(obs.web_name), "shot data must supplement, not be ignored"

    # precedence: observation replaces the declared #1 at that club only
    players = pd.DataFrame({
        "player_code": [101, 102, 103, 104],
        "web_name": ["Alpha", "Bravo", "Charlie", "Delta"],
        "team": ["Arsenal", "Arsenal", "Everton", "Everton"],
        "penalties_order": [2, 1, 2, 1]})           # declares Bravo and Delta
    declared = pen1_codes(players, mode="fpl", verbose=False)
    assert declared == {102, 104}, declared
    seen = pen1_codes(players, mode="observed", verbose=False, observed=obs)
    assert 101 in seen and 103 in seen, f"observation should replace declared #1s: {seen}"

    # a club with no observed penalty keeps its declared taker
    players2 = players.copy()
    players2.loc[len(players2)] = [105, "Echo", "Hull", 1]
    seen2 = pen1_codes(players2, mode="observed", verbose=False, observed=obs)
    assert 105 in seen2, "an unobserved club must keep its declared #1"
    # an EMPTY observation frame must leave the declared order alone
    seen3 = pen1_codes(players, mode="observed", verbose=False,
                       observed=obs.iloc[0:0])
    assert seen3 == declared, "no observations must fall back to the declared order"

    # no observations at all -> falls back to the declared order unchanged
    empty = os.path.join(tempfile.mkdtemp(), "e")
    os.makedirs(os.path.join(empty, "By Gameweek"))
    shutil.copy(os.path.join(tmp, "players.csv"), empty)
    shutil.copy(os.path.join(tmp, "teams.csv"), empty)
    assert not len(observed_takers(base=empty, verbose=False))

    print("SELFTEST OK: scored penalties recovered from playermatchstats (the FPL summary "
          "sees only misses), shot data supplements it, observation "
          "overrides the declared #1 only at clubs that have taken one.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    if "--check" not in sys.argv:
        print(__doc__); sys.exit(0)
    import warnings; warnings.filterwarnings("ignore")
    import core_insights as ci
    d26, t26, _ = ci.load()
    d26 = d26.rename(columns={"team_name": "team"}) if "team_name" in d26.columns else d26
    R = resolve(d26[["web_name", "team", "player_code"]].drop_duplicates())
    print("\nfirst-choice penalty takers by club:")
    pen1 = R[(R.role == "PEN") & (R.order == 1)].sort_values("club")
    for _, r in pen1.iterrows():
        mark = "ok " if pd.notna(r["player_code"]) else "MISS"
        print(f"  {mark} {r['club']:16s} {r['name']}")
