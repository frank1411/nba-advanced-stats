#!/usr/bin/env python3
"""Quick test to verify The Odds API endpoints"""
import os
import requests

API_KEY = os.environ.get('ODDS_API_KEY')
if not API_KEY:
    raise SystemExit("❌ Exporta ODDS_API_KEY en ~/.zshrc")

# Test: Get NBA odds
url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
params = {
    'api_key': API_KEY,
    'regions': 'us',
    'markets': 'h2h,spreads,totals',
    'oddsFormat': 'decimal',
    'dateFormat': 'iso'
}
resp = requests.get(url, params=params)
print(f"Status: {resp.status_code}")
print(f"Remaining requests: {resp.headers.get('x-requests-remaining', 'N/A')}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Games found: {len(data)}")
    if data:
        print(f"First game: {data[0]['home_team']} vs {data[0]['away_team']}")
