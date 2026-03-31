import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

file_path = Path('ML/data/raw/nba_games_raw.parquet')

if file_path.exists():
    df = pd.read_parquet(file_path)
    
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df_sorted = df.sort_values('GAME_DATE', ascending=False)
        
        print("\n--- Últimos 9 Partidos Actualizados ---")
        recent_games = df_sorted.head(9)
        
        for _, row in recent_games.iterrows():
            date_str = row['GAME_DATE'].strftime('%Y-%m-%d')
            home = row.get('TEAM_ABBREVIATION_HOME', 'N/A')
            away = row.get('TEAM_ABBREVIATION_AWAY', 'N/A')
            pts_home = row.get('PTS_HOME', 'N/A')
            pts_away = row.get('PTS_AWAY', 'N/A')
            print(f"📅 {date_str} | 🏠 {home} ({pts_home}) vs ✈️ {away} ({pts_away})")
