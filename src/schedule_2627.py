"""2026/27 Premier League schedule (all 380 fixtures), parsed from
premierleague.com. Team names normalised to match the 25/26 data files.
Promoted for 26/27: Coventry, Hull, Ipswich. Relegated: Burnley, West Ham, Wolves.
"""
import pandas as pd

# full club name -> short name used across the model
NAME = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "AFC Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton & Hove Albion": "Brighton", "Chelsea": "Chelsea",
    "Coventry City": "Coventry", "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds", "Liverpool": "Liverpool", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
}
PROMOTED = ["Coventry", "Hull", "Ipswich"]

# each gameweek: list of (home, away) using full names
_GW = {
1:[("Arsenal","Coventry City"),("Hull City","Manchester United"),("Everton","Crystal Palace"),("Ipswich Town","Sunderland"),("Nottingham Forest","Leeds United"),("Brentford","Tottenham Hotspur"),("Brighton & Hove Albion","Aston Villa"),("Manchester City","AFC Bournemouth"),("Newcastle United","Liverpool"),("Fulham","Chelsea")],
2:[("Crystal Palace","Manchester City"),("Liverpool","Nottingham Forest"),("AFC Bournemouth","Everton"),("Coventry City","Hull City"),("Tottenham Hotspur","Newcastle United"),("Chelsea","Brighton & Hove Albion"),("Leeds United","Brentford"),("Sunderland","Fulham"),("Manchester United","Ipswich Town"),("Aston Villa","Arsenal")],
3:[("Ipswich Town","Liverpool"),("Newcastle United","AFC Bournemouth"),("Brentford","Sunderland"),("Brighton & Hove Albion","Leeds United"),("Fulham","Crystal Palace"),("Manchester City","Coventry City"),("Nottingham Forest","Tottenham Hotspur"),("Hull City","Aston Villa"),("Everton","Manchester United"),("Arsenal","Chelsea")],
4:[("AFC Bournemouth","Brentford"),("Aston Villa","Nottingham Forest"),("Chelsea","Hull City"),("Crystal Palace","Ipswich Town"),("Liverpool","Fulham"),("Tottenham Hotspur","Everton"),("Sunderland","Arsenal"),("Coventry City","Brighton & Hove Albion"),("Manchester United","Manchester City"),("Leeds United","Newcastle United")],
5:[("Brentford","Chelsea"),("Tottenham Hotspur","Aston Villa"),("Brighton & Hove Albion","Arsenal"),("Everton","Ipswich Town"),("Leeds United","Crystal Palace"),("Manchester City","Sunderland"),("Newcastle United","Hull City"),("Nottingham Forest","Coventry City"),("AFC Bournemouth","Liverpool"),("Fulham","Manchester United")],
6:[("Arsenal","Leeds United"),("Aston Villa","Brentford"),("Chelsea","AFC Bournemouth"),("Coventry City","Newcastle United"),("Crystal Palace","Nottingham Forest"),("Hull City","Everton"),("Ipswich Town","Fulham"),("Liverpool","Manchester City"),("Manchester United","Tottenham Hotspur"),("Sunderland","Brighton & Hove Albion")],
7:[("AFC Bournemouth","Sunderland"),("Brentford","Liverpool"),("Brighton & Hove Albion","Crystal Palace"),("Everton","Chelsea"),("Fulham","Hull City"),("Leeds United","Manchester United"),("Manchester City","Ipswich Town"),("Newcastle United","Aston Villa"),("Nottingham Forest","Arsenal"),("Tottenham Hotspur","Coventry City")],
8:[("Arsenal","Everton"),("Aston Villa","Manchester City"),("Chelsea","Tottenham Hotspur"),("Coventry City","Fulham"),("Crystal Palace","Newcastle United"),("Hull City","Brentford"),("Ipswich Town","Nottingham Forest"),("Liverpool","Brighton & Hove Albion"),("Manchester United","AFC Bournemouth"),("Sunderland","Leeds United")],
9:[("AFC Bournemouth","Leeds United"),("Aston Villa","Fulham"),("Brentford","Nottingham Forest"),("Chelsea","Manchester United"),("Coventry City","Sunderland"),("Hull City","Ipswich Town"),("Liverpool","Arsenal"),("Manchester City","Brighton & Hove Albion"),("Newcastle United","Everton"),("Tottenham Hotspur","Crystal Palace")],
10:[("Arsenal","Hull City"),("Brighton & Hove Albion","Brentford"),("Crystal Palace","Liverpool"),("Everton","Coventry City"),("Fulham","Newcastle United"),("Ipswich Town","AFC Bournemouth"),("Leeds United","Tottenham Hotspur"),("Manchester United","Aston Villa"),("Nottingham Forest","Manchester City"),("Sunderland","Chelsea")],
11:[("AFC Bournemouth","Nottingham Forest"),("Aston Villa","Sunderland"),("Brentford","Everton"),("Chelsea","Leeds United"),("Coventry City","Crystal Palace"),("Hull City","Brighton & Hove Albion"),("Liverpool","Manchester United"),("Manchester City","Fulham"),("Newcastle United","Arsenal"),("Tottenham Hotspur","Ipswich Town")],
12:[("Arsenal","Manchester City"),("Brighton & Hove Albion","Newcastle United"),("Crystal Palace","Hull City"),("Everton","Liverpool"),("Fulham","AFC Bournemouth"),("Ipswich Town","Aston Villa"),("Leeds United","Coventry City"),("Manchester United","Brentford"),("Nottingham Forest","Chelsea"),("Sunderland","Tottenham Hotspur")],
13:[("AFC Bournemouth","Brighton & Hove Albion"),("Aston Villa","Everton"),("Brentford","Arsenal"),("Chelsea","Crystal Palace"),("Coventry City","Ipswich Town"),("Hull City","Nottingham Forest"),("Liverpool","Sunderland"),("Manchester City","Leeds United"),("Newcastle United","Manchester United"),("Tottenham Hotspur","Fulham")],
14:[("AFC Bournemouth","Hull City"),("Aston Villa","Crystal Palace"),("Brentford","Manchester City"),("Chelsea","Liverpool"),("Everton","Fulham"),("Leeds United","Ipswich Town"),("Manchester United","Coventry City"),("Newcastle United","Sunderland"),("Nottingham Forest","Brighton & Hove Albion"),("Tottenham Hotspur","Arsenal")],
15:[("Arsenal","AFC Bournemouth"),("Brighton & Hove Albion","Everton"),("Coventry City","Aston Villa"),("Crystal Palace","Manchester United"),("Fulham","Brentford"),("Hull City","Tottenham Hotspur"),("Ipswich Town","Newcastle United"),("Liverpool","Leeds United"),("Manchester City","Chelsea"),("Sunderland","Nottingham Forest")],
16:[("AFC Bournemouth","Coventry City"),("Arsenal","Manchester United"),("Brentford","Newcastle United"),("Brighton & Hove Albion","Ipswich Town"),("Chelsea","Aston Villa"),("Leeds United","Fulham"),("Liverpool","Tottenham Hotspur"),("Manchester City","Hull City"),("Nottingham Forest","Everton"),("Sunderland","Crystal Palace")],
17:[("Aston Villa","Leeds United"),("Coventry City","Chelsea"),("Crystal Palace","Arsenal"),("Everton","Sunderland"),("Fulham","Brighton & Hove Albion"),("Hull City","Liverpool"),("Ipswich Town","Brentford"),("Manchester United","Nottingham Forest"),("Newcastle United","Manchester City"),("Tottenham Hotspur","AFC Bournemouth")],
18:[("Aston Villa","Liverpool"),("Coventry City","Brentford"),("Crystal Palace","AFC Bournemouth"),("Everton","Manchester City"),("Fulham","Arsenal"),("Hull City","Leeds United"),("Ipswich Town","Chelsea"),("Manchester United","Sunderland"),("Newcastle United","Nottingham Forest"),("Tottenham Hotspur","Brighton & Hove Albion")],
19:[("AFC Bournemouth","Aston Villa"),("Arsenal","Ipswich Town"),("Brentford","Crystal Palace"),("Brighton & Hove Albion","Manchester United"),("Chelsea","Newcastle United"),("Leeds United","Everton"),("Liverpool","Coventry City"),("Manchester City","Tottenham Hotspur"),("Nottingham Forest","Fulham"),("Sunderland","Hull City")],
20:[("Arsenal","Brentford"),("Brighton & Hove Albion","AFC Bournemouth"),("Crystal Palace","Chelsea"),("Everton","Aston Villa"),("Fulham","Tottenham Hotspur"),("Ipswich Town","Coventry City"),("Leeds United","Manchester City"),("Manchester United","Newcastle United"),("Nottingham Forest","Hull City"),("Sunderland","Liverpool")],
21:[("AFC Bournemouth","Ipswich Town"),("Aston Villa","Manchester United"),("Brentford","Brighton & Hove Albion"),("Chelsea","Sunderland"),("Coventry City","Everton"),("Hull City","Arsenal"),("Liverpool","Crystal Palace"),("Manchester City","Nottingham Forest"),("Newcastle United","Fulham"),("Tottenham Hotspur","Leeds United")],
22:[("Arsenal","Newcastle United"),("Brighton & Hove Albion","Manchester City"),("Crystal Palace","Tottenham Hotspur"),("Everton","Brentford"),("Fulham","Aston Villa"),("Ipswich Town","Hull City"),("Leeds United","Chelsea"),("Manchester United","Liverpool"),("Nottingham Forest","AFC Bournemouth"),("Sunderland","Coventry City")],
23:[("AFC Bournemouth","Fulham"),("Aston Villa","Ipswich Town"),("Brentford","Manchester United"),("Chelsea","Nottingham Forest"),("Coventry City","Leeds United"),("Hull City","Crystal Palace"),("Liverpool","Everton"),("Manchester City","Arsenal"),("Newcastle United","Brighton & Hove Albion"),("Tottenham Hotspur","Sunderland")],
24:[("Arsenal","Liverpool"),("Brighton & Hove Albion","Hull City"),("Crystal Palace","Coventry City"),("Everton","Newcastle United"),("Fulham","Manchester City"),("Ipswich Town","Tottenham Hotspur"),("Leeds United","AFC Bournemouth"),("Manchester United","Chelsea"),("Nottingham Forest","Brentford"),("Sunderland","Aston Villa")],
25:[("Aston Villa","AFC Bournemouth"),("Coventry City","Liverpool"),("Crystal Palace","Brentford"),("Everton","Leeds United"),("Fulham","Nottingham Forest"),("Hull City","Sunderland"),("Ipswich Town","Arsenal"),("Manchester United","Brighton & Hove Albion"),("Newcastle United","Chelsea"),("Tottenham Hotspur","Manchester City")],
26:[("AFC Bournemouth","Crystal Palace"),("Arsenal","Fulham"),("Brentford","Coventry City"),("Brighton & Hove Albion","Tottenham Hotspur"),("Chelsea","Ipswich Town"),("Leeds United","Aston Villa"),("Liverpool","Hull City"),("Manchester City","Newcastle United"),("Nottingham Forest","Manchester United"),("Sunderland","Everton")],
27:[("Aston Villa","Chelsea"),("Coventry City","AFC Bournemouth"),("Crystal Palace","Sunderland"),("Everton","Nottingham Forest"),("Fulham","Leeds United"),("Hull City","Manchester City"),("Ipswich Town","Brighton & Hove Albion"),("Manchester United","Arsenal"),("Newcastle United","Brentford"),("Tottenham Hotspur","Liverpool")],
28:[("AFC Bournemouth","Tottenham Hotspur"),("Arsenal","Crystal Palace"),("Brentford","Ipswich Town"),("Brighton & Hove Albion","Fulham"),("Chelsea","Coventry City"),("Leeds United","Hull City"),("Liverpool","Aston Villa"),("Manchester City","Everton"),("Nottingham Forest","Newcastle United"),("Sunderland","Manchester United")],
29:[("AFC Bournemouth","Newcastle United"),("Aston Villa","Hull City"),("Chelsea","Arsenal"),("Coventry City","Manchester City"),("Crystal Palace","Fulham"),("Leeds United","Brighton & Hove Albion"),("Liverpool","Ipswich Town"),("Manchester United","Everton"),("Sunderland","Brentford"),("Tottenham Hotspur","Nottingham Forest")],
30:[("Arsenal","Sunderland"),("Brentford","AFC Bournemouth"),("Brighton & Hove Albion","Coventry City"),("Everton","Tottenham Hotspur"),("Fulham","Liverpool"),("Hull City","Chelsea"),("Ipswich Town","Crystal Palace"),("Manchester City","Manchester United"),("Newcastle United","Leeds United"),("Nottingham Forest","Aston Villa")],
31:[("AFC Bournemouth","Manchester City"),("Aston Villa","Brighton & Hove Albion"),("Chelsea","Fulham"),("Coventry City","Arsenal"),("Crystal Palace","Everton"),("Leeds United","Nottingham Forest"),("Liverpool","Newcastle United"),("Manchester United","Hull City"),("Sunderland","Ipswich Town"),("Tottenham Hotspur","Brentford")],
32:[("Arsenal","Aston Villa"),("Brentford","Leeds United"),("Brighton & Hove Albion","Chelsea"),("Everton","AFC Bournemouth"),("Fulham","Sunderland"),("Hull City","Coventry City"),("Ipswich Town","Manchester United"),("Manchester City","Crystal Palace"),("Newcastle United","Tottenham Hotspur"),("Nottingham Forest","Liverpool")],
33:[("AFC Bournemouth","Arsenal"),("Aston Villa","Coventry City"),("Brentford","Fulham"),("Chelsea","Manchester City"),("Everton","Brighton & Hove Albion"),("Leeds United","Liverpool"),("Manchester United","Crystal Palace"),("Newcastle United","Ipswich Town"),("Nottingham Forest","Sunderland"),("Tottenham Hotspur","Hull City")],
34:[("Arsenal","Tottenham Hotspur"),("Brighton & Hove Albion","Nottingham Forest"),("Coventry City","Manchester United"),("Crystal Palace","Aston Villa"),("Fulham","Everton"),("Hull City","AFC Bournemouth"),("Ipswich Town","Leeds United"),("Liverpool","Chelsea"),("Manchester City","Brentford"),("Sunderland","Newcastle United")],
35:[("AFC Bournemouth","Manchester United"),("Brentford","Aston Villa"),("Brighton & Hove Albion","Sunderland"),("Everton","Hull City"),("Fulham","Ipswich Town"),("Leeds United","Arsenal"),("Manchester City","Liverpool"),("Newcastle United","Coventry City"),("Nottingham Forest","Crystal Palace"),("Tottenham Hotspur","Chelsea")],
36:[("Arsenal","Nottingham Forest"),("Aston Villa","Newcastle United"),("Chelsea","Everton"),("Coventry City","Tottenham Hotspur"),("Crystal Palace","Brighton & Hove Albion"),("Hull City","Fulham"),("Ipswich Town","Manchester City"),("Liverpool","Brentford"),("Manchester United","Leeds United"),("Sunderland","AFC Bournemouth")],
37:[("AFC Bournemouth","Chelsea"),("Brentford","Hull City"),("Brighton & Hove Albion","Liverpool"),("Everton","Arsenal"),("Fulham","Coventry City"),("Leeds United","Sunderland"),("Manchester City","Aston Villa"),("Newcastle United","Crystal Palace"),("Nottingham Forest","Ipswich Town"),("Tottenham Hotspur","Manchester United")],
38:[("Arsenal","Brighton & Hove Albion"),("Aston Villa","Tottenham Hotspur"),("Chelsea","Brentford"),("Coventry City","Nottingham Forest"),("Crystal Palace","Leeds United"),("Hull City","Newcastle United"),("Ipswich Town","Everton"),("Liverpool","AFC Bournemouth"),("Manchester United","Fulham"),("Sunderland","Man City" if False else "Manchester City")],
}

def schedule():
    rows = []
    for gw, games in _GW.items():
        for h, a in games:
            rows.append({"gameweek": gw, "home": NAME[h], "away": NAME[a]})
    df = pd.DataFrame(rows)
    assert len(df) == 380, len(df)
    assert df.groupby("gameweek").size().eq(10).all()
    teams = sorted(set(df.home) | set(df.away))
    assert len(teams) == 20, teams
    # long form: one row per team-fixture with venue
    home = df.rename(columns={"home": "team", "away": "opp"}).assign(is_home=1)
    away = df.rename(columns={"away": "team", "home": "opp"}).assign(is_home=0)
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "gameweek"])
    return df, long

if __name__ == "__main__":
    df, long = schedule()
    print("fixtures:", len(df), "teams:", long.team.nunique())
    print("promoted present:", [t for t in PROMOTED if t in set(long.team)])
    print(df.head(10).to_string(index=False))
</content>
