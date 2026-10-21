import requests
import time

HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Origin': 'https://www.nba.com',
    'Connection': 'keep-alive',
    'Referer': 'https://www.nba.com/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'sec-ch-ua': '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"'
}

session = requests.Session()
# Intentar obtener cookies de la página principal primero
print("Obteniendo cookies de nba.com...")
try:
    session.get("https://www.nba.com", headers={'User-Agent': HEADERS['User-Agent']}, timeout=10)
except:
    pass


def test_nba_api(with_cache_params=True, season_type='Regular Season'):
    url = "https://stats.nba.com/stats/leaguedashteamstats"
    params = {
        'LeagueID': '00',
        'Season': '2025-26',
        'SeasonType': season_type,
        'MeasureType': 'Advanced',
        'PerMode': 'Per100Possessions',
        'PlusMinus': 'N',
        'PaceAdjust': 'Y',
        'Rank': 'N',
        'Outcome': '',
        'Location': '',
        'Month': '0',
        'SeasonSegment': '',
        'DateFrom': '',
        'DateTo': '',
        'OpponentTeamID': '0',
        'VsConference': '',
        'VsDivision': '',
        'GameSegment': '',
        'Period': '0',
        'LastNGames': '0',
        'GameScope': '',
        'PlayerExperience': '',
        'PlayerPosition': '',
        'StarterBench': '',
        'TwoWay': '0',
        'DraftYear': '',
        'DraftPick': '',
        'College': '',
        'Country': '',
        'Height': '',
        'Weight': ''
    }
    
    if with_cache_params:
        params.update({
            'ts': str(int(time.time())),
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        })
    
    print(f"Testing {season_type} (Cache Params: {with_cache_params})...")
    try:
        response = session.get(url, params=params, headers=HEADERS, timeout=15)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Success!")
            return True
        else:
            print(f"❌ Failed: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

print("--- NBA API Diagnostics ---")
# 1. Original (con params de cache)
success = test_nba_api(with_cache_params=True)

# 2. Sin params de cache
if not success:
    print("\nRetrying without cache parameters...")
    success = test_nba_api(with_cache_params=False)

# 3. Con SeasonType='Playoffs'
if not success:
    print("\nRetrying with SeasonType='Playoffs'...")
    success = test_nba_api(with_cache_params=False, season_type='Playoffs')

if success:
    print("\nFound a working configuration!")
else:
    print("\nAll attempts failed. The NBA API might be down or blocking this IP.")
