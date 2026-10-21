import requests
import json

def test_nba_api():
    url = "https://stats.nba.com/stats/leaguedashteamstats"
    
    # Headers exactly as in the Postman collection
    headers = {
        "Host": "stats.nba.com",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/"
    }
    
    params = {
        "LeagueID": "00",
        "Season": "2024-25", # Changed to 2024-25 as 25-26 might not be active yet or valid for some endpoints
        "SeasonType": "Regular Season",
        "MeasureType": "Advanced",
        "PerMode": "Per100Possessions",
        "PlusMinus": "N",
        "PaceAdjust": "Y",
        "Rank": "N"
    }
    
    print(f"📡 Testing {url}...")
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print(f"✅ Status Code: {response.status_code}")
        if response.status_code == 200:
            print("🎉 Success!")
            # print(json.dumps(response.json(), indent=2)[:500])
        else:
            print(f"❌ Error Response: {response.text[:500]}")
    except Exception as e:
        print(f"💥 Exception: {e}")

if __name__ == "__main__":
    test_nba_api()
