import time
import random
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

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

# Ubicaciones de los estadios de la NBA (latitud, longitud)
NBA_STADIUMS = {
    'ATL': (33.7573, -84.3963),    # State Farm Arena
    'BOS': (42.3662, -71.0621),    # TD Garden
    'BKN': (40.6826, -73.9754),    # Barclays Center
    'CHA': (35.2251, -80.8392),    # Spectrum Center
    'CHI': (41.8806, -87.6742),    # United Center
    'CLE': (41.4965, -81.6881),    # Rocket Mortgage FieldHouse
    'DAL': (32.7903, -96.8103),    # American Airlines Center
    'DEN': (39.7487, -105.008),    # Ball Arena
    'DET': (42.6966, -83.2453),    # Little Caesars Arena
    'GSW': (37.7679, -122.3876),   # Chase Center
    'HOU': (29.7508, -95.3621),    # Toyota Center
    'IND': (39.7639, -86.1557),    # Gainbridge Fieldhouse
    'LAC': (34.0430, -118.2673),   # Crypto.com Arena
    'LAL': (34.0430, -118.2673),   # Crypto.com Arena
    'MEM': (35.1380, -90.0504),    # FedExForum
    'MIA': (25.7814, -80.1890),    # FTX Arena
    'MIL': (43.0451, -87.9173),    # Fiserv Forum
    'MIN': (44.9795, -93.2761),    # Target Center
    'NOP': (29.9490, -90.0821),    # Smoothie King Center
    'NYK': (40.7505, -73.9934),    # Madison Square Garden
    'OKC': (35.4634, -97.5151),    # Paycom Center
    'ORL': (28.5392, -81.3836),    # Amway Center
    'PHI': (39.9012, -75.1720),    # Wells Fargo Center
    'PHX': (33.4457, -112.0712),   # Footprint Center
    'POR': (45.5316, -122.6668),   # Moda Center
    'SAC': (38.5802, -121.4997),   # Golden 1 Center
    'SAS': (29.4270, -98.4375),    # AT&T Center
    'TOR': (43.6435, -79.3791),    # Scotiabank Arena
    'UTA': (40.7683, -111.9011),   # Vivint Arena
    'WAS': (38.8981, -77.0209)     # Capital One Arena
}

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

# Ubicaciones de los estadios de la NBA (latitud, longitud)
NBA_STADIUMS = {
    'ATL': (33.7573, -84.3963),    # State Farm Arena
    'BOS': (42.3662, -71.0621),    # TD Garden
    'BKN': (40.6826, -73.9754),    # Barclays Center
    'CHA': (35.2251, -80.8392),    # Spectrum Center
    'CHI': (41.8806, -87.6742),    # United Center
    'CLE': (41.4965, -81.6881),    # Rocket Mortgage FieldHouse
    'DAL': (32.7903, -96.8103),    # American Airlines Center
    'DEN': (39.7487, -105.008),    # Ball Arena
    'DET': (42.6966, -83.2453),    # Little Caesars Arena
    'GSW': (37.7679, -122.3876),   # Chase Center
    'HOU': (29.7508, -95.3621),    # Toyota Center
    'IND': (39.7639, -86.1557),    # Gainbridge Fieldhouse
    'LAC': (34.0430, -118.2673),   # Crypto.com Arena
    'LAL': (34.0430, -118.2673),   # Crypto.com Arena (mismo que LAC)
    'MEM': (35.1380, -90.0504),    # FedExForum
    'MIA': (25.7814, -80.1890),    # FTX Arena
    'MIL': (43.0451, -87.9173),    # Fiserv Forum
    'MIN': (44.9795, -93.2761),    # Target Center
    'NOP': (29.9490, -90.0821),    # Smoothie King Center
    'NYK': (40.7505, -73.9934),    # Madison Square Garden
    'OKC': (35.4634, -97.5151),    # Paycom Center
    'ORL': (28.5392, -81.3836),    # Amway Center
    'PHI': (39.9012, -75.1720),    # Wells Fargo Center
    'PHX': (33.4457, -112.0712),   # Footprint Center
    'POR': (45.5316, -122.6668),   # Moda Center
    'SAC': (38.5802, -121.4997),   # Golden 1 Center
    'SAS': (29.4270, -98.4375),    # AT&T Center
    'TOR': (43.6435, -79.3791),    # Scotiabank Arena
    'UTA': (40.7683, -111.9011),   # Vivint Arena
    'WAS': (38.8982, -77.0209)     # Capital One Arena
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
    
    def get_season_games(self, season='2024-25', season_type='Regular Season'):
        """
        Obtiene todos los partidos de una temporada y tipo específico.
        
        Args:
            season (str): Temporada en formato 'YYYY-YY'
            season_type (str): Tipo de temporada ('Regular Season', 'Playoffs', 'Pre Season', 'All-Star', 'IST' o 'ALL')
                               Si es 'ALL', obtiene todos los partidos disponibles sin filtrar.
        """
        endpoint = 'leaguegamefinder'
        
        params = {
            'LeagueID': '00',
            'Season': season,
            'PlayerOrTeam': 'T', # T = Team, P = Player
            'Sorter': 'DATE',
            'Direction': 'DESC'
        }
        
        # Solo agregar SeasonType si no es 'ALL'
        if season_type and season_type != 'ALL':
            params['SeasonType'] = season_type
            
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

    def get_team_stats(self, season: str, season_type: str = 'Regular Season') -> Optional[pd.DataFrame]:
        """Obtiene estadísticas de equipo, incluyendo eficiencia ofensiva y defensiva."""
        print(f"📊 Obteniendo estadísticas avanzadas para {season} - {season_type}...")
        
        # Primero, obtener estadísticas básicas
        endpoint = 'leaguedashteamstats'
        params = {
            'LeagueID': '00',  # NBA
            'Season': season,
            'SeasonType': season_type,
            'MeasureType': 'Advanced',  # Cambiado a Advanced para obtener más métricas
            'PerMode': 'Per100Possessions',  # Mejor para estadísticas de ritmo
            'PlusMinus': 'N',
            'PaceAdjust': 'Y',  # Ajustar por ritmo
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
        
        data = self.make_request(endpoint, params)
        if not data or 'resultSets' not in data or len(data['resultSets']) == 0:
            print(f"❌ No se pudieron obtener las estadísticas de equipo para {season} - {season_type}")
            return None
            
        # Procesar los datos
        stats = data['resultSets'][0]['rowSet']
        columns = data['resultSets'][0]['headers']
        df = pd.DataFrame(stats, columns=columns)
        
        # Seleccionar solo las columnas necesarias
        keep_columns = [
            'TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT',
            'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE'
        ]
        
        # Filtrar columnas que existen
        keep_columns = [col for col in keep_columns if col in df.columns]
        df = df[keep_columns]
        
        # Agregar información de temporada
        df['SEASON'] = int(season.split('-')[0])
        df['SEASON_STR'] = season
        df['SEASON_TYPE'] = season_type
        
        return df
        
    def calculate_rolling_averages(self, df: pd.DataFrame, team_id_col: str, stats_columns: list, window: int = 10) -> pd.DataFrame:
        """
        Calcula promedios móviles para las estadísticas especificadas.
        
        Args:
            df: DataFrame con los datos de los partidos
            team_id_col: Nombre de la columna que contiene el ID del equipo
            stats_columns: Lista de columnas para las que se calcularán los promedios
            window: Número de partidos a incluir en el promedio móvil
            
        Returns:
            DataFrame con las columnas de promedios móviles añadidas
        """
        # Asegurarse de que los datos estén ordenados por fecha
        df = df.sort_values('GAME_DATE')
        
        # Crear una copia para no modificar el original
        result = df.copy()
        
        # Calcular promedios móviles para cada equipo y estadística
        for team_id in df[team_id_col].unique():
            team_mask = df[team_id_col] == team_id
            team_games = df[team_mask].copy()
            
            # Para cada estadística, calcular el promedio móvil
            for stat in stats_columns:
                if stat in team_games.columns:
                    # Calcular promedio móvil excluyendo el partido actual
                    rolling_mean = team_games[stat].shift(1).rolling(window=window, min_periods=1).mean()
                    result.loc[team_mask, f'{stat}_L10'] = rolling_mean.values
        
        return result
    
    def process_games_data(self, df: pd.DataFrame, team_stats: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Procesa los datos de los partidos para tener un formato más limpio."""
        if df.empty:
            return pd.DataFrame()
            
        # Crear copia para no modificar el original
        df_processed = df.copy()
        
        # Determinar local y visitante por cada partido (maneja casos mixtos de cancha neutral)
        def mark_home_team(group):
            # Si hay al menos un 'vs.', usamos la lógica estándar
            if group['MATCHUP'].str.contains('vs.').any():
                group['IS_HOME'] = group['MATCHUP'].str.contains('vs.').astype(int)
            else:
                # Caso cancha neutral: ambos teams tienen '@'
                # El equipo después del '@' se considera "local" para efectos de la base de datos
                matchup = group['MATCHUP'].iloc[0]
                if ' @ ' in matchup:
                    home_team_abbr = matchup.split(' @ ')[1].strip()
                    group['IS_HOME'] = (group['TEAM_ABBREVIATION'] == home_team_abbr).astype(int)
                    
                    # Si aún no hay local (raro), asignamos al primero
                    if group['IS_HOME'].sum() == 0:
                        group.iloc[0, group.columns.get_loc('IS_HOME')] = 1
                else:
                    # No hay '@' ni 'vs.', asignamos al primero
                    group['IS_HOME'] = 0
                    group.iloc[0, group.columns.get_loc('IS_HOME')] = 1
            return group
            
        df_processed = df_processed.groupby('GAME_ID', group_keys=False).apply(mark_home_team)
        
        # Separar en local y visitante
        home_games = df_processed[df_processed['IS_HOME'] == 1].copy()
        away_games = df_processed[df_processed['IS_HOME'] == 0].copy()
        
        # Renombrar columnas para el merge
        home_games = home_games.rename(columns={
            'TEAM_ID': 'TEAM_ID_HOME',
            'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION_HOME',
            'PTS': 'PTS_HOME',
            'WL': 'WL_HOME',
            'OFF_RATING': 'OFF_RATING_HOME',
            'DEF_RATING': 'DEF_RATING_HOME',
            'NET_RATING': 'NET_RATING_HOME',
            'PACE': 'PACE_HOME'
        })
        
        away_games = away_games.rename(columns={
            'TEAM_ID': 'TEAM_ID_AWAY',
            'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION_AWAY',
            'PTS': 'PTS_AWAY',
            'WL': 'WL_AWAY',
            'OFF_RATING': 'OFF_RATING_AWAY',
            'DEF_RATING': 'DEF_RATING_AWAY',
            'NET_RATING': 'NET_RATING_AWAY',
            'PACE': 'PACE_AWAY'
        })
        
        # Unir los datos
        merged = pd.merge(
            home_games,
            away_games,
            on=['GAME_ID', 'GAME_DATE', 'SEASON_ID'],
            suffixes=('', '_y')
        )
        
        # Función para convertir SEASON_ID a año de temporada
        def get_season_year(season_id):
            # Los primeros dos dígitos indican el tipo de temporada (2=regular, 4=playoffs)
            # Los últimos cuatro dígitos son el año de inicio (ej: 2018 para 2018-19)
            season_type = int(str(season_id)[0])
            year = int(str(season_id)[1:])
            return year, 'Playoffs' if season_type == 4 else 'Regular Season'
            
        # Aplicar la función a SEASON_ID
        season_info = merged['SEASON_ID'].apply(get_season_year)
        merged['SEASON'] = season_info.apply(lambda x: x[0])
        merged['SEASON_TYPE'] = season_info.apply(lambda x: x[1])
        
        # Formatear la temporada como YYYY-YY (ej: 2018-19)
        merged['SEASON_STR'] = merged['SEASON'].apply(
            lambda y: f"{y}-{str(y+1)[-2:]}"
        )
        
        # Si tenemos estadísticas de equipo, unirlas
        if team_stats is not None and not team_stats.empty:
            # Crear una copia de las estadísticas de equipo
            team_stats_copy = team_stats.copy()
            
            # Primero, filtrar las columnas que necesitamos
            team_stats_cols = ['TEAM_ID', 'SEASON', 'SEASON_STR', 'SEASON_TYPE', 
                             'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 
                             'W_PCT', 'GP', 'W', 'L']
            
            # Filtrar solo las columnas que existen
            available_cols = [col for col in team_stats_cols if col in team_stats_copy.columns]
            team_stats_filtered = team_stats_copy[available_cols].copy()
            
            # Crear copias para home y away
            team_stats_home = team_stats_filtered.copy()
            team_stats_away = team_stats_filtered.copy()
            
            # Renombrar columnas para home
            rename_home = {
                'OFF_RATING': 'OFF_RATING_HOME',
                'DEF_RATING': 'DEF_RATING_HOME',
                'NET_RATING': 'NET_RATING_HOME',
                'PACE': 'PACE_HOME',
                'W_PCT': 'W_PCT_HOME',
                'GP': 'GP_HOME',
                'W': 'W_HOME',
                'L': 'L_HOME'
            }
            team_stats_home = team_stats_home.rename(columns=rename_home)
            
            # Renombrar columnas para away
            rename_away = {
                'OFF_RATING': 'OFF_RATING_AWAY',
                'DEF_RATING': 'DEF_RATING_AWAY',
                'NET_RATING': 'NET_RATING_AWAY',
                'PACE': 'PACE_AWAY',
                'W_PCT': 'W_PCT_AWAY',
                'GP': 'GP_AWAY',
                'W': 'W_AWAY',
                'L': 'L_AWAY'
            }
            team_stats_away = team_stats_away.rename(columns=rename_away)
            
            # Unir estadísticas del equipo local
            merged = pd.merge(
                merged,
                team_stats_home,
                left_on=['TEAM_ID_HOME', 'SEASON', 'SEASON_STR', 'SEASON_TYPE'],
                right_on=['TEAM_ID', 'SEASON', 'SEASON_STR', 'SEASON_TYPE'],
                how='left'
            )
            
            # Eliminar columna duplicada
            if 'TEAM_ID' in merged.columns:
                merged = merged.drop(columns=['TEAM_ID'])
            
            # Unir estadísticas del equipo visitante
            merged = pd.merge(
                merged,
                team_stats_away,
                left_on=['TEAM_ID_AWAY', 'SEASON', 'SEASON_STR', 'SEASON_TYPE'],
                right_on=['TEAM_ID', 'SEASON', 'SEASON_STR', 'SEASON_TYPE'],
                how='left'
            )
            
            # Eliminar columna duplicada
            if 'TEAM_ID' in merged.columns:
                merged = merged.drop(columns=['TEAM_ID'])
            
            # Eliminar columnas duplicadas
            merged = merged.loc[:, ~merged.columns.duplicated()]
        
        # Calcular diferencial de puntos
        if 'PTS_HOME' in merged.columns and 'PTS_AWAY' in merged.columns:
            merged['POINT_DIFF'] = merged['PTS_HOME'] - merged['PTS_AWAY']
            
            # Calcular diferencial de puntos para cada equipo
            merged['HOME_POINT_DIFF'] = merged['PTS_HOME'] - merged['PTS_AWAY']
            merged['AWAY_POINT_DIFF'] = merged['PTS_AWAY'] - merged['PTS_HOME']
        
        # Seleccionar columnas relevantes
        final_columns = [
            'GAME_ID', 'GAME_DATE', 'SEASON', 'SEASON_STR', 'SEASON_TYPE',
            'TEAM_ID_HOME', 'TEAM_ABBREVIATION_HOME', 'PTS_HOME', 'WL_HOME',
            'OFF_RATING_HOME', 'DEF_RATING_HOME', 'NET_RATING_HOME', 'PACE_HOME',
            'W_PCT_HOME', 'GP_HOME', 'W_HOME', 'L_HOME', 'HOME_POINT_DIFF',
            'TEAM_ID_AWAY', 'TEAM_ABBREVIATION_AWAY', 'PTS_AWAY', 'WL_AWAY',
            'OFF_RATING_AWAY', 'DEF_RATING_AWAY', 'NET_RATING_AWAY', 'PACE_AWAY',
            'W_PCT_AWAY', 'GP_AWAY', 'W_AWAY', 'L_AWAY', 'AWAY_POINT_DIFF',
            'POINT_DIFF'
        ]
        
        # Asegurarse de que todas las columnas existan
        final_columns = [col for col in final_columns if col in merged.columns]
        merged = merged[final_columns]
        
        # Ordenar por fecha
        merged = merged.sort_values('GAME_DATE')
        
        # Calcular promedios de los últimos 10 partidos para cada equipo
        print("📊 Calculando promedios de los últimos 10 partidos...")
        
        # Lista de estadísticas para las que calcularemos promedios
        stats_columns = ['PTS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'POINT_DIFF']
        
        # Columnas especiales que tienen nombres diferentes
        special_columns = {
            'POINT_DIFF': {
                'home': 'HOME_POINT_DIFF',
                'away': 'AWAY_POINT_DIFF'
            }
        }
        
        # Crear copia del DataFrame para no modificar el original
        result_df = merged.copy()
        
        # Para cada equipo, calcular sus promedios de los últimos 10 partidos
        for team_id in set(merged['TEAM_ID_HOME'].unique()).union(set(merged['TEAM_ID_AWAY'].unique())):
            # Filtrar partidos donde el equipo jugó (como local o visitante)
            team_matches = merged[
                (merged['TEAM_ID_HOME'] == team_id) | (merged['TEAM_ID_AWAY'] == team_id)
            ].copy()
            
            # Ordenar por fecha
            team_matches = team_matches.sort_values('GAME_DATE')
            
            # Crear un identificador único para cada fila
            team_matches['row_id'] = range(len(team_matches))
            
            # Para cada estadística, calcular el promedio móvil
            for stat in stats_columns:
                # Determinar si el equipo es local o visitante para cada partido
                is_home = team_matches['TEAM_ID_HOME'] == team_id
                
                # Obtener los valores de la estadística para este equipo
                if stat in special_columns:
                    # Para columnas especiales como POINT_DIFF
                    stat_values = np.where(
                        is_home,
                        team_matches[special_columns[stat]['home']],
                        team_matches[special_columns[stat]['away']]
                    )
                else:
                    # Para columnas estándar
                    stat_values = np.where(
                        is_home,
                        team_matches[f'{stat}_HOME'],
                        team_matches[f'{stat}_AWAY']
                    )
                
                # Calcular el promedio móvil (últimos 10 partidos, excluyendo el actual)
                rolling_mean = pd.Series(stat_values).shift(1).rolling(window=10, min_periods=1).mean()
                
                # Asignar los promedios a las filas correspondientes
                for idx, row_id in enumerate(team_matches['row_id']):
                    if is_home.iloc[idx]:
                        result_df.loc[result_df['GAME_ID'] == team_matches.iloc[idx]['GAME_ID'], 
                                    f'{stat}_L10_HOME'] = rolling_mean.iloc[idx]
                    else:
                        result_df.loc[result_df['GAME_ID'] == team_matches.iloc[idx]['GAME_ID'], 
                                    f'{stat}_L10_AWAY'] = rolling_mean.iloc[idx]
        
        # Reemplazar el DataFrame original con el que tiene los promedios
        merged = result_df
        
        return merged

    def calculate_rest_days_and_b2b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula los días de descanso y partidos B2B (Back-to-Back) para cada equipo.
        
        Args:
            df: DataFrame con los datos de los partidos
            
        Returns:
            DataFrame con columnas adicionales para días de descanso y B2B
        """
        print("📅 Calculando días de descanso y partidos B2B...")
        
        # Hacer una copia para no modificar el original
        df = df.copy()
        
        # Asegurarse de que la columna de fecha sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df['GAME_DATE']):
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Ordenar por equipo y fecha
        df = df.sort_values(['TEAM_ID_HOME', 'GAME_DATE'])
        
        # Inicializar columnas para equipos locales
        df['DAYS_REST_HOME'] = np.nan
        df['IS_B2B_HOME'] = False
        
        # Calcular días de descanso y B2B para equipos locales
        for team_id in df['TEAM_ID_HOME'].unique():
            team_mask = df['TEAM_ID_HOME'] == team_id
            team_games = df[team_mask].copy()
            
            # Calcular días desde el partido anterior
            team_games['DAYS_REST'] = team_games['GAME_DATE'].diff().dt.days - 1
            
            # El primer partido de la temporada tendrá NaN, lo rellenamos con 3 (promedio de descanso)
            team_games['DAYS_REST'] = team_games['DAYS_REST'].fillna(3)
            
            # Marcar partidos B2B (días de descanso = 0)
            team_games['IS_B2B'] = team_games['DAYS_REST'] == 0
            
            # Actualizar el DataFrame original
            df.loc[team_mask, 'DAYS_REST_HOME'] = team_games['DAYS_REST'].values
            df.loc[team_mask, 'IS_B2B_HOME'] = team_games['IS_B2B'].values
        
        # Repetir para equipos visitantes
        df = df.sort_values(['TEAM_ID_AWAY', 'GAME_DATE'])
        
        # Inicializar columnas para equipos visitantes
        df['DAYS_REST_AWAY'] = np.nan
        df['IS_B2B_AWAY'] = False
        
        for team_id in df['TEAM_ID_AWAY'].unique():
            team_mask = df['TEAM_ID_AWAY'] == team_id
            team_games = df[team_mask].copy()
            
            # Calcular días desde el partido anterior
            team_games['DAYS_REST'] = team_games['GAME_DATE'].diff().dt.days - 1
            
            # El primer partido de la temporada tendrá NaN, lo rellenamos con 3 (promedio de descanso)
            team_games['DAYS_REST'] = team_games['DAYS_REST'].fillna(3)
            
            # Marcar partidos B2B (días de descanso = 0)
            team_games['IS_B2B'] = team_games['DAYS_REST'] == 0
            
            # Actualizar el DataFrame original
            df.loc[team_mask, 'DAYS_REST_AWAY'] = team_games['DAYS_REST'].values
            df.loc[team_mask, 'IS_B2B_AWAY'] = team_games['IS_B2B'].values
        
        # Ordenar por fecha para el resultado final
        df = df.sort_values('GAME_DATE')
        
        return df

    def calculate_travel_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula las distancias de viaje para cada equipo entre partidos consecutivos.
        
        Args:
            df: DataFrame con los datos de los partidos
            
        Returns:
            DataFrame con columnas adicionales para distancias de viaje
        """
        print("✈️ Calculando distancias de viaje...")
        
        # Hacer una copia para no modificar el original
        df = df.copy()
        
        # Asegurarse de que la columna de fecha sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df['GAME_DATE']):
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Ordenar por equipo y fecha
        df = df.sort_values(['TEAM_ID_HOME', 'GAME_DATE'])
        
        # Inicializar columnas para distancias de viaje
        df['TRAVEL_DISTANCE_HOME'] = 0.0
        df['TRAVEL_DISTANCE_AWAY'] = 0.0
        
        # Función para calcular la distancia entre dos puntos (fórmula del semiverseno)
        def haversine(lat1, lon1, lat2, lon2):
            """
            Calcula la distancia en millas entre dos puntos en la Tierra.
            Utiliza la fórmula del semiverseno (Haversine formula).
            """
            from math import radians, sin, cos, sqrt, asin
            
            # Radio de la Tierra en millas
            R = 3958.8
            
            # Convertir grados a radianes
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Diferencia de latitud y longitud
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            # Fórmula del semiverseno
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            
            # Distancia en millas
            distance = R * c
            return distance
        
        # Función para calcular la distancia de viaje para un equipo
        def calculate_team_travel(team_games, team_id_col, team_abbr_col, is_home=True):
            # Ordenar por fecha
            team_games = team_games.sort_values('GAME_DATE')
            
            # Inicializar columna de distancia
            distances = [0.0] * len(team_games)
            
            # Para cada partido a partir del segundo
            for i in range(1, len(team_games)):
                # Obtener las ubicaciones de los partidos consecutivos
                prev_game = team_games.iloc[i-1]
                curr_game = team_games.iloc[i]
                
                # Determinar las ubicaciones de los partidos
                if is_home:
                    # Para equipos locales, comparamos con el partido anterior como visitante o local
                    if prev_game['TEAM_ID_HOME'] == curr_game['TEAM_ID_HOME']:
                        # Partido anterior como local
                        prev_abbr = prev_game['TEAM_ABBREVIATION_HOME']
                        prev_loc = 'HOME'
                    else:
                        # Partido anterior como visitante
                        prev_abbr = prev_game['TEAM_ABBREVIATION_AWAY']
                        prev_loc = 'AWAY'
                    
                    curr_abbr = curr_game['TEAM_ABBREVIATION_HOME']
                    curr_loc = 'HOME'
                else:
                    # Para equipos visitantes, comparamos con el partido anterior como local o visitante
                    if prev_game['TEAM_ID_AWAY'] == curr_game['TEAM_ID_AWAY']:
                        # Partido anterior como visitante
                        prev_abbr = prev_game['TEAM_ABBREVIATION_AWAY']
                        prev_loc = 'AWAY'
                    else:
                        # Partido anterior como local
                        prev_abbr = prev_game['TEAM_ABBREVIATION_HOME']
                        prev_loc = 'HOME'
                    
                    curr_abbr = curr_game['TEAM_ABBREVIATION_AWAY']
                    curr_loc = 'AWAY'
                
                # Obtener coordenadas
                # Si el equipo jugó en casa en el partido anterior, usamos su estadio
                if prev_loc == 'HOME':
                    coords1 = NBA_STADIUMS.get(prev_abbr)
                else:
                    # Si fue visitante, usamos el estadio del oponente
                    opp_abbr = prev_abbr if prev_abbr != curr_abbr else prev_game['TEAM_ABBREVIATION_HOME']
                    coords1 = NBA_STADIUMS.get(opp_abbr)
                
                lat1, lon1 = coords1 if coords1 else (0, 0)
                
                # Para el partido actual, usamos el estadio del partido
                if curr_loc == 'HOME':
                    coords2 = NBA_STADIUMS.get(curr_abbr)
                else:
                    coords2 = NBA_STADIUMS.get(curr_game['TEAM_ABBREVIATION_HOME'])
                
                lat2, lon2 = coords2 if coords2 else (0, 0)
                
                # Calcular distancia solo si las coordenadas son válidas
                if lat1 != 0 and lon1 != 0 and lat2 != 0 and lon2 != 0:
                    distance = haversine(lat1, lon1, lat2, lon2)
                    distances[i] = distance
            
            return distances
        
        # Calcular distancias para equipos locales
        for team_id in df['TEAM_ID_HOME'].unique():
            # Filtrar partidos donde el equipo jugó (como local o visitante)
            team_games = df[(df['TEAM_ID_HOME'] == team_id) | (df['TEAM_ID_AWAY'] == team_id)].copy()
            
            if len(team_games) > 1:
                # Ordenar por fecha
                team_games = team_games.sort_values('GAME_DATE')
                
                # Calcular distancias para este equipo
                for i in range(1, len(team_games)):
                    prev_game = team_games.iloc[i-1]
                    curr_game = team_games.iloc[i]
                    
                    # Determinar si el equipo es local o visitante en cada partido
                    is_home_prev = prev_game['TEAM_ID_HOME'] == team_id
                    is_home_curr = curr_game['TEAM_ID_HOME'] == team_id
                    
                    # Obtener las ubicaciones de los partidos
                    if is_home_prev:
                        prev_abbr = prev_game['TEAM_ABBREVIATION_HOME']
                        prev_loc = NBA_STADIUMS.get(prev_abbr)
                    else:
                        prev_opp_abbr = prev_game['TEAM_ABBREVIATION_HOME']
                        prev_loc = NBA_STADIUMS.get(prev_opp_abbr)
                    
                    if is_home_curr:
                        curr_abbr = curr_game['TEAM_ABBREVIATION_HOME']
                        curr_loc = NBA_STADIUMS.get(curr_abbr)
                    else:
                        curr_opp_abbr = curr_game['TEAM_ABBREVIATION_HOME']
                        curr_loc = NBA_STADIUMS.get(curr_opp_abbr)
                    
                    # Calcular distancia solo si ambos estadios están definidos
                    if prev_loc and curr_loc:
                        distance = haversine(prev_loc[0], prev_loc[1], curr_loc[0], curr_loc[1])
                    else:
                        distance = 0.0
                    
                    # Asignar la distancia al partido actual
                    game_id = curr_game['GAME_ID']
                    if is_home_curr:
                        df.loc[df['GAME_ID'] == game_id, 'TRAVEL_DISTANCE_HOME'] = distance
                    else:
                        df.loc[df['GAME_ID'] == game_id, 'TRAVEL_DISTANCE_AWAY'] = distance
        
        return df

def main():
    # Inicializar el fetcher
    fetcher = NBADataFetcher()
    
    # Lista para almacenar todos los partidos
    all_games = []
    
    # Temporadas a procesar
    start_year = 2018
    end_year = 2025  # Incluir hasta la temporada 2025-26
    
    # Procesar cada temporada
    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        
        # Procesar temporada regular y playoffs
        for season_type in ['Regular Season', 'Playoffs']:
            print(f"\n🔄 Procesando {season} - {season_type}...")
            
            # Obtener estadísticas de equipo primero
            print(f"📊 Obteniendo estadísticas de equipo...")
            team_stats = fetcher.get_team_stats(season, season_type)
            
            # Obtener partidos
            print(f"🏀 Obteniendo datos de partidos...")
            games = fetcher.get_season_games(season, season_type)
            
            if games is not None and not games.empty:
                print(f"🔄 Procesando datos...")
                processed_games = fetcher.process_games_data(games, team_stats)
                if processed_games is not None and not processed_games.empty:
                    # Calcular días de descanso y B2B
                    processed_games = fetcher.calculate_rest_days_and_b2b(processed_games)
                    
                    # Calcular distancias de viaje
                    processed_games = fetcher.calculate_travel_distances(processed_games)
                    
                    all_games.append(processed_games)
                    print(f"✅ Datos procesados: {len(processed_games)} partidos")
            
            # Pequeña pausa entre temporadas para no sobrecargar la API
            time.sleep(2)
    
    # Combinar todos los partidos
    if all_games:
        df = pd.concat(all_games, ignore_index=True)
        
        # Guardar los datos
        output_dir = Path(__file__).parent.parent / "data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "nba_games_raw.parquet"
        df.to_parquet(output_path, index=False)
        
        # Mostrar resumen
        print("\n✅ Datos guardados en:", output_path)
        print("\n📊 Resumen de datos:")
        print(f"Total de partidos: {len(df):,}")
        print(f"Temporadas: {df['SEASON_STR'].nunique()}")
        print(f"Equipos únicos (local): {df['TEAM_ABBREVIATION_HOME'].nunique()}")
        print(f"Equipos únicos (visitante): {df['TEAM_ABBREVIATION_AWAY'].nunique()}")
        print(f"Rango de fechas: {df['GAME_DATE'].min().date()} a {df['GAME_DATE'].max().date()}")
        
        # Mostrar columnas disponibles
        print("\n📋 Columnas disponibles:")
        print("\n".join([f"- {col}" for col in df.columns]))
    else:
        print("❌ No se encontraron partidos para procesar")

if __name__ == "__main__":
    main()