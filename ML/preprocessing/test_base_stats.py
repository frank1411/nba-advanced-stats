import requests

HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com/',
}

url = "https://stats.nba.com/stats/leaguedashteamstats"
params = {
    'LeagueID': '00',
    'Season': '2025-26',
    'SeasonType': 'Regular Season',
    'MeasureType': 'Base', # Cambiamos de Advanced a Base
    'PerMode': 'PerGame',
    'PlusMinus': 'N',
    'PaceAdjust': 'N',
    'Rank': 'N',
}

print("Testing MeasureType=Base...")
try:
    response = requests.get(url, params=params, headers=HEADERS, timeout=10)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Base Stats Success!")
    else:
        print(f"❌ Failed: {response.text[:200]}")
except Exception as e:
    print(f"❌ Exception: {e}")
