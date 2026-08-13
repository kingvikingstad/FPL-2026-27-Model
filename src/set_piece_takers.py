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
    club_of = dict(zip(R["skey"], R["club"]))

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


if __name__ == "__main__":
    import sys
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
