import time
import random
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

# Configuración
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Crear directorios si no existen
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Constantes
SEASONS = [
    '2018-19', '2019-20', '2020-21', 
    '2021-22', '2022-23', '2023-24', '2024-25'
]

# Headers mejorados para la API de la NBA
HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Origin': 'https://www.nba.com',
    'Connection': 'keep-alive',
    'Referer': 'https://www.nba.com/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'sec-ch-ua': '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"'
}

class NBADataFetcher:
    def __init__(self):
        self.base_url = 'https://stats.nba.com/stats/'
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.retry_count = 0
        self.max_retries = 5
        self.timeout = 15  # segundos

    def make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Realiza una petición a la API de la NBA con manejo de errores."""
        url = f"{self.base_url}{endpoint}"
        
        # Añadir timestamp para evitar caché
        params_with_ts = params.copy()
        params_with_ts.update({
            'ts': str(int(time.time())),
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        })
        
        for attempt in range(self.max_retries):
            try:
                # Rotar user-agent
                user_agents = [
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5.1 Safari/605.1.15',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/126.0'
                ]
                
                current_headers = HEADERS.copy()
                current_headers['User-Agent'] = user_agents[attempt % len(user_agents)]
                
                response = self.session.get(
                    url, 
                    params=params_with_ts,
                    headers=current_headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Verificar si la respuesta es válida
                if not data or 'resultSets' not in data:
                    raise ValueError("Respuesta inválida de la API")
                
                # Pausa aleatoria entre 1-3 segundos
                time.sleep(random.uniform(1, 3))
                return data
                
            except (requests.exceptions.RequestException, ValueError) as e:
                wait_time = min(2 ** attempt, 30)  # Backoff exponencial con máximo 30 segundos
                
                if attempt == self.max_retries - 1:
                    print(f"❌ Error después de {self.max_retries} intentos en {endpoint}: {str(e)}")
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"Código de estado: {e.response.status_code}")
                        print(f"Respuesta: {e.response.text[:500]}")
                    return None
                    
                print(f"⚠️  Intento {attempt + 1}/{self.max_retries} fallido. Reintentando en {wait_time} segundos...")
                time.sleep(wait_time)
    
    def get_season_games(self, season: str, season_type: str = 'Regular Season') -> Optional[pd.DataFrame]:
        """Obtiene todos los partidos de una temporada y tipo de temporada."""
        print(f"\n📅 Procesando temporada: {season} - {season_type}")
        print("-" * 50)
        
        endpoint = 'leaguegamefinder'
        params = {
            'LeagueID': '00',  # NBA
            'Season': season,
            'SeasonType': season_type,
            'PlayerOrTeam': 'T',  # Team stats
            'Sorter': 'DATE',
            'Direction': 'DESC'
        }
        
        data = self.make_request(endpoint, params)
        if not data:
            return None
            
        # Procesar los datos
        games = data['resultSets'][0]['rowSet']
        columns = data['resultSets'][0]['headers']
        df = pd.DataFrame(games, columns=columns)
        
        # Filtrar solo partidos de temporada regular y playoffs
        df = df[df['GAME_DATE'].notna()].copy()
        
        # Convertir fecha
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Asegurarse de que tenemos los datos necesarios
        required_columns = ['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_ABBREVIATION', 
                          'MATCHUP', 'PTS', 'WL', 'SEASON_ID']
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            print(f"❌ Faltan columnas requeridas: {missing}")
            return None
            
        print(f"✅ Se encontraron {len(df)} partidos para {season} - {season_type}")
        return df

    def process_games_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa los datos de los partidos para tener un formato más limpio."""
        if df.empty:
            return pd.DataFrame()
            
        # Crear copia para no modificar el original
        df_processed = df.copy()
        
        # Determinar local y visitante
        df_processed['IS_HOME'] = df_processed['MATCHUP'].str.contains('vs.').astype(int)
        
        # Separar en local y visitante
        home_games = df_processed[df_processed['IS_HOME'] == 1].copy()
        away_games = df_processed[df_processed['IS_HOME'] == 0].copy()
        
        # Renombrar columnas para el merge
        home_games = home_games.rename(columns={
            'TEAM_ID': 'TEAM_ID_HOME',
            'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION_HOME',
            'PTS': 'PTS_HOME',
            'WL': 'WL_HOME'
        })
        
        away_games = away_games.rename(columns={
            'TEAM_ID': 'TEAM_ID_AWAY',
            'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION_AWAY',
            'PTS': 'PTS_AWAY',
            'WL': 'WL_AWAY'
        })
        
        # Unir los datos
        merged = pd.merge(
            home_games,
            away_games,
            on=['GAME_ID', 'GAME_DATE', 'SEASON_ID'],
            suffixes=('', '_y')
        )
        
        # Seleccionar columnas relevantes
        final_columns = [
            'GAME_ID', 'GAME_DATE', 'SEASON_ID',
            'TEAM_ID_HOME', 'TEAM_ABBREVIATION_HOME', 'PTS_HOME', 'WL_HOME',
            'TEAM_ID_AWAY', 'TEAM_ABBREVIATION_AWAY', 'PTS_AWAY', 'WL_AWAY'
        ]
        
        # Asegurarse de que todas las columnas existan
        final_columns = [col for col in final_columns if col in merged.columns]
        merged = merged[final_columns]
        
        # Ordenar por fecha
        merged = merged.sort_values('GAME_DATE')
        
        return merged

def main():
    # Inicializar el fetcher
    fetcher = NBADataFetcher()
    
    # Lista para almacenar todos los partidos
    all_games = []
    
    # Obtener partidos para cada temporada
    for season in SEASONS:
        # Primero temporada regular
        regular_season = fetcher.get_season_games(season, 'Regular Season')
        if regular_season is not None:
            all_games.append(regular_season)
        
        # Luego playoffs
        playoffs = fetcher.get_season_games(season, 'Playoffs')
        if playoffs is not None:
            all_games.append(playoffs)
        
        # Pequeña pausa entre temporadas
        time.sleep(2)
    
    if not all_games:
        print("❌ No se pudieron obtener datos de los partidos")
        return
        
    # Combinar todos los datos
    all_games_df = pd.concat(all_games, ignore_index=True)
    
    # Procesar los datos
    processed_df = fetcher.process_games_data(all_games_df)
    
    if processed_df.empty:
        print("❌ No se pudieron procesar los datos de los partidos")
        return
    
    # Guardar los datos
    output_file = RAW_DATA_DIR / 'nba_games_raw.parquet'
    processed_df.to_parquet(output_file, index=False)
    print(f"\n✅ Datos guardados en: {output_file}")
    
    # Mostrar resumen
    print("\n📊 Resumen de datos:")
    print(f"Total de partidos: {len(processed_df)}")
    print(f"Temporadas: {processed_df['SEASON_ID'].nunique()}")
    print(f"Equipos únicos (local): {processed_df['TEAM_ABBREVIATION_HOME'].nunique()}")
    print(f"Equipos únicos (visitante): {processed_df['TEAM_ABBREVIATION_AWAY'].nunique()}")
    print(f"Rango de fechas: {processed_df['GAME_DATE'].min().date()} a {processed_df['GAME_DATE'].max().date()}")

if __name__ == "__main__":
    main()