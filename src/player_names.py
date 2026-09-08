from __future__ import annotations
import config
"""
player_names.py — one naming convention, applied everywhere.
=============================================================
FPL's `web_name` is a display name, not an identifier. In the 26/27 squad **15 surnames
belong to 32 different players**, spread across clubs:

    Wilson    Brentford / Coventry / Leeds        Phillips  Hull / Man City / Tottenham
    Palmer    Chelsea / Ipswich                   James     Chelsea / Leeds
    Johnson   Everton / Ipswich                   Dasilva   Brentford / Coventry
    Davies, Gomez, Henderson, Hughes, Kamara, King, Martinez, Patterson, Sangaré

This has already caused three separate defects in this project: a penalty override that
cleared the wrong player's duty, an availability join that could rule out the wrong
player, and a projection export that fanned out on a name-keyed merge. The fix is not
better matching — it is to stop using names as keys and to give humans a name that is
actually unambiguous.

THE CONVENTION
--------------
`player_code`   the identifier. Stable across seasons, never reused. ALL joins use this.
`display_name`  what a human reads. Minimal disambiguation — you only pay for it where
                it is needed, so 552 of 584 players keep their plain `web_name`:
                  1. `web_name` if it is unique league-wide          -> "Haaland"
                  2. else first initial + surname + club             -> "C. Palmer (CHE)"
                  3. else full first name + surname + club           -> "Brennan Johnson (EVE)"
                     (needed for Brennan/Ben Johnson and Josh/Jay Dasilva, where the
                     initial collides too)
`unique_label`  always carries the club: "Haaland (MCI)", "C. Palmer (CHE)". Guaranteed
                unique, consistent in shape, and therefore the safe thing to put in a
                spreadsheet lookup or a pivot row.

STABILITY, HONESTLY STATED
--------------------------
`display_name` is computed against the CURRENT squad, so it is not stable across
transfer windows: if a second Palmer arrives, the first one's display name grows a
qualifier. `unique_label` is stable unless the player changes club. Only `player_code`
is stable full stop — which is exactly why it, and not either of these, is the join key.

Run:  python src/player_names.py --selftest
      python src/player_names.py --check     (needs FPL_DATA)
"""
import numpy as np
import pandas as pd

# fallback three-letter codes, used only if teams.csv is not passed in
_TLA = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Chelsea": "CHE", "Coventry": "COV", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Hull": "HUL", "Ipswich": "IPS", "Leeds": "LEE",
    "Liverpool": "LIV", "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Sunderland": "SUN", "Tottenham": "TOT",
}


def team_codes(teams=None) -> dict:
    """team -> three-letter code, preferring the dataset's own `short_name`."""
    if teams is not None and {"team", "short_name"} <= set(teams.columns):
        m = dict(zip(teams["team"], teams["short_name"]))
        return {k: (v if isinstance(v, str) and v else _TLA.get(k, str(k)[:3].upper()))
                for k, v in m.items()}
    return dict(_TLA)


def _initial(first):
    s = str(first or "").strip()
    return s[:1].upper() if s else ""


def canonical(players: pd.DataFrame, teams: pd.DataFrame = None,
              name_col="web_name", team_col="team") -> pd.DataFrame:
    """Add `display_name`, `unique_label`, `full_name`, `name_ambiguous`.

    Needs [name_col, team_col] and, to disambiguate, ideally `first_name` /
    `second_name` and `player_code`. Degrades safely: with no first names available the
    club qualifier alone is used, which still yields a unique label because no club in
    the 26/27 squad has two players sharing a web_name.
    """
    p = players.copy()
    tla = team_codes(teams)
    code = p[team_col].map(lambda t: tla.get(t, str(t)[:3].upper()))

    first = p["first_name"] if "first_name" in p.columns else pd.Series("", index=p.index)
    second = p["second_name"] if "second_name" in p.columns else pd.Series("", index=p.index)
    p["full_name"] = (first.fillna("").astype(str).str.strip() + " " +
                      second.fillna("").astype(str).str.strip()).str.strip()
    p.loc[p["full_name"] == "", "full_name"] = p[name_col].astype(str)

    # identity for "is this the same person?" — player_code when present, else name+club
    ident = (p["player_code"] if "player_code" in p.columns
             else p[name_col].astype(str) + "|" + p[team_col].astype(str))

    n_per_name = ident.groupby(p[name_col]).transform("nunique")
    p["name_ambiguous"] = n_per_name > 1

    init = first.map(_initial)
    cand2 = np.where(init != "", init + ". " + p[name_col].astype(str),
                     p[name_col].astype(str))
    cand2 = pd.Series(cand2, index=p.index)
    n_per_cand2 = ident.groupby(cand2).transform("nunique")

    disp = p[name_col].astype(str).copy()
    lvl2 = p["name_ambiguous"] & (n_per_cand2 == 1)
    disp[lvl2] = cand2[lvl2] + " (" + code[lvl2] + ")"
    lvl3 = p["name_ambiguous"] & (n_per_cand2 > 1)
    disp[lvl3] = (first.fillna("").astype(str).str.strip()[lvl3] + " " +
                  p[name_col].astype(str)[lvl3] + " (" + code[lvl3] + ")").str.strip()
    p["display_name"] = disp

    base = np.where(p["name_ambiguous"],
                    disp.str.replace(r"\s*\([A-Z]{3}\)$", "", regex=True),
                    p[name_col].astype(str))
    p["unique_label"] = pd.Series(base, index=p.index) + " (" + code + ")"
    p["team_code"] = code
    return p


def check(players: pd.DataFrame) -> dict:
    """Uniqueness audit — call it and assert on it rather than assuming."""
    n = len(players)
    return {
        "players": n,
        "distinct_web_name": players["web_name"].nunique() if "web_name" in players else None,
        "distinct_display_name": players["display_name"].nunique(),
        "distinct_unique_label": players["unique_label"].nunique(),
        "ambiguous_players": int(players["name_ambiguous"].sum()),
        "display_name_is_unique": bool(players["display_name"].nunique() == n),
        "unique_label_is_unique": bool(players["unique_label"].nunique() == n),
    }


def selftest():
    df = pd.DataFrame({
        "player_code": [1, 2, 3, 4, 5, 6, 7],
        "web_name": ["Haaland", "Palmer", "Palmer", "Johnson", "Johnson", "Wilson", "Wilson"],
        "first_name": ["Erling", "Cole", "Alex", "Brennan", "Ben", "Callum", ""],
        "second_name": ["Haaland", "Palmer", "Palmer", "Johnson", "Johnson", "Wilson", "Wilson"],
        "team": ["Man City", "Chelsea", "Ipswich", "Everton", "Ipswich", "Brentford", "Leeds"],
    })
    out = canonical(df)
    g = dict(zip(out.player_code, out.display_name))

    assert g[1] == "Haaland", f"unique name should stay plain, got {g[1]}"
    assert g[2] == "C. Palmer (CHE)", g[2]
    assert g[3] == "A. Palmer (IPS)", g[3]
    # initial collides (Brennan/Ben) -> fall through to the full first name
    assert g[4] == "Brennan Johnson (EVE)", g[4]
    assert g[5] == "Ben Johnson (IPS)", g[5]
    # missing first name must not crash, and must still disambiguate by club
    assert g[6] == "C. Wilson (BRE)", g[6]
    assert "(LEE)" in g[7], g[7]

    st = check(out)
    assert st["display_name_is_unique"], "display_name must be unique"
    assert st["unique_label_is_unique"], "unique_label must be unique"
    assert out.loc[out.player_code == 1, "unique_label"].iloc[0] == "Haaland (MCI)"

    # degrade safely with no first names at all
    bare = df.drop(columns=["first_name", "second_name"])
    ob = canonical(bare)
    assert ob["unique_label"].nunique() == len(bare), "club qualifier alone must suffice"

    print("SELFTEST OK: unique names untouched, initials used where they resolve, full "
          "first names where they do not, missing first names handled, labels unique.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    if "--check" in sys.argv:
        import warnings; warnings.filterwarnings("ignore")
        import core_insights as ci
        d, t, _ = ci.load(base=config.repo("2026-2027"))
        out = canonical(d, t)
        st = check(out)
        for k, v in st.items():
            print(f"  {k:26s} {v}")
        print("\nevery player whose name needed disambiguating:")
        amb = out[out.name_ambiguous].sort_values(["web_name", "team"])
        print(amb[["web_name", "full_name", "team", "pos", "display_name",
                   "unique_label"]].to_string(index=False))
        sys.exit(0)
    print(__doc__)
