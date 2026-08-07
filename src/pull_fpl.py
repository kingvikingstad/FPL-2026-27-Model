#!/usr/bin/env python3
"""
pull_fpl.py — one command to grab the 2026/27 FPL data and make it uploadable
=============================================================================
Run this on YOUR machine (it needs normal internet, no key, no login):

    python pull_fpl.py

It writes three files next to itself:

    fpl_players.csv     <- SMALL (~150 KB). This is the one to upload.
    fpl_teams.csv       <- 20 teams + FPL's own strength ratings.
    bootstrap_raw.json  <- full raw payload, if you want the belt-and-braces copy.

`fpl_players.csv` carries everything the model needs in one flat table: roster
(name, team, position, price, ownership), availability (status, chance of
playing, news), set-piece and penalty order, FPL's own ep_next, and last
season's totals where present. Uploading it is far lighter than the ~2 MB JSON.

Then either:
  * upload fpl_players.csv (+ fpl_teams.csv) to the chat, or
  * run locally:  python run_2627.py --gw-from 1 --gw-to 6 --players fpl_players.csv

Sanity check printed at the end tells you WHICH SEASON the API returned, so you
can confirm it has actually rolled over to 26/27 before trusting the output.
"""
import json, sys, urllib.request

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
HDRS = {"User-Agent": "Mozilla/5.0"}          # FPL rejects bare urllib UA sometimes


def main():
    try:
        req = urllib.request.Request(URL, headers=HDRS)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"Fetch failed: {type(e).__name__}: {e}")
        print("If this is blocked, open the URL in your browser and save the page "
              "as bootstrap_raw.json, then re-run:  python pull_fpl.py --file bootstrap_raw.json")
        sys.exit(1)
    write_all(data)


def write_all(data):
    import csv
    teams = {t["id"]: t for t in data["teams"]}
    etype = {e["id"]: e["singular_name_short"] for e in data["element_types"]}

    # ---- players ----------------------------------------------------------
    cols = ["id", "web_name", "first_name", "second_name", "team", "pos",
            "price", "own", "status", "chance_next", "chance_this", "news",
            "ep_next", "pen_order", "fk_order", "corner_order",
            "minutes", "starts", "goals", "assists", "clean_sheets", "bonus",
            "xg", "xa", "xgi", "xgc", "defensive_contribution", "cbi",
            "recoveries", "tackles", "saves", "total_points"]
    rows = []
    for e in data["elements"]:
        rows.append({
            "id": e["id"], "web_name": e["web_name"],
            "first_name": e.get("first_name", ""), "second_name": e.get("second_name", ""),
            "team": teams[e["team"]]["name"], "pos": etype[e["element_type"]],
            "price": e["now_cost"] / 10.0, "own": e.get("selected_by_percent"),
            "status": e.get("status"),
            "chance_next": e.get("chance_of_playing_next_round"),
            "chance_this": e.get("chance_of_playing_this_round"),
            "news": (e.get("news") or "").replace("\n", " "),
            "ep_next": e.get("ep_next"),
            "pen_order": e.get("penalties_order"),
            "fk_order": e.get("direct_freekicks_order"),
            "corner_order": e.get("corners_and_indirect_freekicks_order"),
            "minutes": e.get("minutes"), "starts": e.get("starts"),
            "goals": e.get("goals_scored"), "assists": e.get("assists"),
            "clean_sheets": e.get("clean_sheets"), "bonus": e.get("bonus"),
            "xg": e.get("expected_goals"), "xa": e.get("expected_assists"),
            "xgi": e.get("expected_goal_involvements"),
            "xgc": e.get("expected_goals_conceded"),
            "defensive_contribution": e.get("defensive_contribution"),
            "cbi": e.get("clearances_blocks_interceptions"),
            "recoveries": e.get("recoveries"), "tackles": e.get("tackles"),
            "saves": e.get("saves"), "total_points": e.get("total_points"),
        })
    with open("fpl_players.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    # ---- teams ------------------------------------------------------------
    tcols = ["id", "name", "short_name", "strength", "ovr_home", "ovr_away",
             "att_home", "att_away", "def_home", "def_away"]
    trows = [{"id": t["id"], "name": t["name"], "short_name": t["short_name"],
              "strength": t["strength"],
              "ovr_home": t["strength_overall_home"], "ovr_away": t["strength_overall_away"],
              "att_home": t["strength_attack_home"], "att_away": t["strength_attack_away"],
              "def_home": t["strength_defence_home"], "def_away": t["strength_defence_away"]}
             for t in data["teams"]]
    with open("fpl_teams.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tcols); w.writeheader(); w.writerows(trows)

    with open("bootstrap_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

    # ---- WHICH SEASON did we actually get? --------------------------------
    url = data.get("game_config", {}).get("settings", {}).get("static_content_url", "")
    season = url.rstrip("/").split("/")[-1] if url else "unknown"
    names = {t["name"] for t in data["teams"]}
    promoted = {"Coventry", "Hull", "Ipswich"}
    hit = sorted(n for n in names if any(p in n for p in promoted))
    nxt = next((e for e in data["events"] if e.get("is_next")), None)

    print(f"wrote fpl_players.csv ({len(rows)} players), fpl_teams.csv ({len(trows)} teams), bootstrap_raw.json")
    print(f"\n--- SEASON CHECK ---")
    print(f"  static_content_url season : {season}")
    print(f"  promoted clubs present    : {hit if hit else 'NONE  <-- still last season'}")
    print(f"  next gameweek             : {nxt['name'] + ' ' + nxt['deadline_time'] if nxt else 'none flagged'}")
    if season.startswith("2026") or hit:
        print("  => looks like 2026/27. Good to go.")
    else:
        print("  => still serving 2025/26. The game hasn't rolled over on the public API yet.")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        with open(sys.argv[2], encoding="utf-8") as f:
            write_all(json.load(f))
    else:
        main()
