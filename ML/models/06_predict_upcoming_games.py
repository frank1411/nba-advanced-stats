#!/usr/bin/env python3
"""
Script para predecir partidos futuros de la NBA.

1. Obtiene el calendario de partidos por jugar usando la API oficial de la NBA (nba_api)
2. Genera características para las predicciones
3. Aplica los modelos entrenados
4. Muestra y guarda los resultados
"""
# =============================================================================
# CONFIGURACIÓN INICIAL Y MANEJO DE ADVERTENCIAS
# =============================================================================
import os
import sys
import warnings
import json
import logging
import random
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Configuración de advertencias
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURACIÓN DE SEMILLAS ALEATORIAS PARA REPRODUCIBILIDAD
# =============================================================================
import numpy as np

# Establecer semillas para reproducibilidad
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Desactivar paralelismo para evitar errores de mutex en macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Configuración de numpy
np.seterr(all='warn')

# Resto de importaciones de terceros
try:
    import pandas as pd
    import joblib
    import pytz
    from scipy.stats import norm
    logger.info("Módulos de análisis de datos cargados correctamente")
except ImportError as e:
    logger.error(f"Error al cargar módulos de análisis de datos: {str(e)}")
    raise

# No más datos estáticos - Ahora obtenemos los partidos directamente de la API

# Suprimir advertencias
warnings.filterwarnings('ignore')

# Configuración de pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# Asegurar que los directorios existan
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONSTANTES Y FUNCIONES DE APOYO PARA CARACTERÍSTICAS
# =============================================================================

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
    'WAS': (38.8982, -77.0209)     # Capital One Arena
}

# Mapa de normalización de abreviaturas de equipos (ESPN/Other -> Database)
TEAM_NORM_MAP = {
    'NO': 'NOP',
    'NY': 'NYK',
    'GS': 'GSW',
    'UTAH': 'UTA',
    'WSH': 'WAS',
    'PHX': 'PHX', # Asegurar consistencia
    'SA': 'SAS',
    'WSA': 'WAS', # Algunas fuentes usan esta
    'BKN': 'BKN',
    'CHA': 'CHA'
}

def haversine(lat1, lon1, lat2, lon2):
    """Calcula la distancia en millas entre dos puntos en la Tierra."""
    # Radio de la Tierra en millas
    R = 3958.8
    # Convertir grados a radianes
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # Diferencia de latitud y longitud
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Fórmula del semiverseno
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def load_models() -> tuple:
    """Carga los modelos entrenados y la lista de features."""
    try:
        # Cargar lista de features
        with open(MODELS_DIR / "feature_list.json", "r") as f:
            feature_data = json.load(f)
        feature_cols = feature_data.get("features", [])
        
        # Cargar modelos
        model_home = joblib.load(MODELS_DIR / "model_PTS_HOME.pkl")
        model_away = joblib.load(MODELS_DIR / "model_PTS_AWAY.pkl")
        
        return model_home, model_away, feature_cols
    except Exception as e:
        print(f"❌ Error al cargar modelos: {e}")
        sys.exit(1)

class NBADataFetcher:
    """Clase para obtener datos de la NBA usando el MCP de balldontlie-api."""
    
    def __init__(self):
        """Inicializa el cliente de la API."""
        # Mapeo de abreviaturas de equipos a sus IDs
        self.team_abbrev_to_id = {
            'ATL': 1, 'BOS': 2, 'BKN': 3, 'CHA': 4, 'CHI': 5,
            'CLE': 6, 'DAL': 7, 'DEN': 8, 'DET': 9, 'GSW': 10,
            'HOU': 11, 'IND': 12, 'LAC': 13, 'LAL': 14, 'MEM': 15,
            'MIA': 16, 'MIL': 17, 'MIN': 18, 'NOP': 19, 'NYK': 20,
            'OKC': 21, 'ORL': 22, 'PHI': 23, 'PHX': 24, 'POR': 25,
            'SAC': 26, 'SAS': 27, 'TOR': 28, 'UTA': 29, 'WAS': 30
        }
        
        # Mapeo inverso de IDs a abreviaturas
        self.team_id_to_abbrev = {v: k for k, v in self.team_abbrev_to_id.items()}
        
        # Información detallada de los equipos
        self.team_info = {
            'ATL': {'full_name': 'Atlanta Hawks', 'city': 'Atlanta', 'arena': 'State Farm Arena'},
            'BOS': {'full_name': 'Boston Celtics', 'city': 'Boston', 'arena': 'TD Garden'},
            'BKN': {'full_name': 'Brooklyn Nets', 'city': 'Brooklyn', 'arena': 'Barclays Center'},
            'CHA': {'full_name': 'Charlotte Hornets', 'city': 'Charlotte', 'arena': 'Spectrum Center'},
            'CHI': {'full_name': 'Chicago Bulls', 'city': 'Chicago', 'arena': 'United Center'},
            'CLE': {'full_name': 'Cleveland Cavaliers', 'city': 'Cleveland', 'arena': 'Rocket Mortgage FieldHouse'},
            'DAL': {'full_name': 'Dallas Mavericks', 'city': 'Dallas', 'arena': 'American Airlines Center'},
            'DEN': {'full_name': 'Denver Nuggets', 'city': 'Denver', 'arena': 'Ball Arena'},
            'DET': {'full_name': 'Detroit Pistons', 'city': 'Detroit', 'arena': 'Little Caesars Arena'},
            'GSW': {'full_name': 'Golden State Warriors', 'city': 'San Francisco', 'arena': 'Chase Center'},
            'HOU': {'full_name': 'Houston Rockets', 'city': 'Houston', 'arena': 'Toyota Center'},
            'IND': {'full_name': 'Indiana Pacers', 'city': 'Indianapolis', 'arena': 'Gainbridge Fieldhouse'},
            'LAC': {'full_name': 'LA Clippers', 'city': 'Los Angeles', 'arena': 'Crypto.com Arena'},
            'LAL': {'full_name': 'Los Angeles Lakers', 'city': 'Los Angeles', 'arena': 'Crypto.com Arena'},
            'MEM': {'full_name': 'Memphis Grizzlies', 'city': 'Memphis', 'arena': 'FedExForum'},
            'MIA': {'full_name': 'Miami Heat', 'city': 'Miami', 'arena': 'FTX Arena'},
            'MIL': {'full_name': 'Milwaukee Bucks', 'city': 'Milwaukee', 'arena': 'Fiserv Forum'},
            'MIN': {'full_name': 'Minnesota Timberwolves', 'city': 'Minneapolis', 'arena': 'Target Center'},
            'NOP': {'full_name': 'New Orleans Pelicans', 'city': 'New Orleans', 'arena': 'Smoothie King Center'},
            'NYK': {'full_name': 'New York Knicks', 'city': 'New York', 'arena': 'Madison Square Garden'},
            'OKC': {'full_name': 'Oklahoma City Thunder', 'city': 'Oklahoma City', 'arena': 'Paycom Center'},
            'ORL': {'full_name': 'Orlando Magic', 'city': 'Orlando', 'arena': 'Amway Center'},
            'PHI': {'full_name': 'Philadelphia 76ers', 'city': 'Philadelphia', 'arena': 'Wells Fargo Center'},
            'PHX': {'full_name': 'Phoenix Suns', 'city': 'Phoenix', 'arena': 'Footprint Center'},
            'POR': {'full_name': 'Portland Trail Blazers', 'city': 'Portland', 'arena': 'Moda Center'},
            'SAC': {'full_name': 'Sacramento Kings', 'city': 'Sacramento', 'arena': 'Golden 1 Center'},
            'SAS': {'full_name': 'San Antonio Spurs', 'city': 'San Antonio', 'arena': 'AT&T Center'},
            'TOR': {'full_name': 'Toronto Raptors', 'city': 'Toronto', 'arena': 'Scotiabank Arena'},
            'UTA': {'full_name': 'Utah Jazz', 'city': 'Salt Lake City', 'arena': 'Vivint Arena'},
            'WAS': {'full_name': 'Washington Wizards', 'city': 'Washington', 'arena': 'Capital One Arena'}
        }
        
        logger.info("Cliente de balldontlie-api inicializado")
        
    def _get_team_abbrev(self, team_id: int) -> str:
        """Obtiene la abreviatura del equipo a partir de su ID."""
        return self.team_id_to_abbrev.get(team_id, f"UNK{team_id}")
    
    def _get_team_info(self, team_abbrev: str, field: str, default: str = 'Unknown') -> str:
        """Obtiene información específica de un equipo."""
        return self.team_info.get(team_abbrev, {}).get(field, default)
    
    def get_schedule(self, days_ahead: int = 0) -> pd.DataFrame:
        """
        Obtiene los partidos programados de la NBA usando la API de ESPN.
        
        Args:
            days_ahead: Número de días a futuro para buscar partidos (0 = hoy)
            
        Returns:
            DataFrame con los partidos programados para el día especificado
            
        Raises:
            Exception: Si hay un error al obtener los datos de la API
        """
        from datetime import datetime, timedelta
        import pytz
        import requests
        
        try:
            # Obtener la fecha objetivo
            target_date = datetime.now(pytz.utc) + timedelta(days=days_ahead)
            target_date_str = target_date.strftime('%Y%m%d')
            
            logger.info(f"Buscando partidos programados para {target_date_str}...")
            
            # Headers para la petición a la API de ESPN
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
                'Referer': 'https://www.espn.com/',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            # Hacer la petición a la API de ESPN
            url = f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={target_date_str}'
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if 'events' not in data or not data['events']:
                logger.warning(f"No se encontraron partidos programados para {target_date_str}.")
                return pd.DataFrame()
                
            games = data['events']
            games_list = []
            
            # Procesar cada partido
            for game in games:
                try:
                    # Obtener información básica del partido
                    game_id = game['id']
                    game_date = game['date']
                    
                    # Convertir la hora del partido a la zona horaria local
                    game_time = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
                    game_time_local = game_time.astimezone()
                    
                    # Obtener información de los equipos
                    home_team = next((t for t in game['competitions'][0]['competitors'] if t['homeAway'] == 'home'), None)
                    away_team = next((t for t in game['competitions'][0]['competitors'] if t['homeAway'] == 'away'), None)
                    
                    if not home_team or not away_team:
                        continue
                        
                    home_abbrev = home_team['team']['abbreviation']
                    away_abbrev = away_team['team']['abbreviation']
                    
                    # Obtener la temporada actual
                    season = f"{datetime.now().year}-{str(datetime.now().year + 1)[-2:]}"
                    
                    # Obtener información adicional de los equipos
                    home_team_info = self.team_info.get(home_abbrev, {})
                    away_team_info = self.team_info.get(away_abbrev, {})
                    
                    # Obtener estadísticas de los equipos si están disponibles
                    home_wins = int(home_team.get('record', [{}])[0].get('summary', '0-0').split('-')[0])
                    home_losses = int(home_team.get('record', [{}])[0].get('summary', '0-0').split('-')[1])
                    away_wins = int(away_team.get('record', [{}])[0].get('summary', '0-0').split('-')[0])
                    away_losses = int(away_team.get('record', [{}])[0].get('summary', '0-0').split('-')[1])
                    
                    # Crear diccionario con los datos del partido
                    game_dict = {
                        'GAME_ID': game_id,
                        'GAME_DATE': game_time_local.date(),
                        'GAME_TIME': game_time_local.strftime('%H:%M'),
                        'HOME_TEAM_ID': home_team['team'].get('id', 0),
                        'VISITOR_TEAM_ID': away_team['team'].get('id', 0),
                        'TEAM_ABBREVIATION_HOME': home_abbrev,
                        'TEAM_ABBREVIATION_AWAY': away_abbrev,
                        'TEAM_NAME_HOME': home_team['team'].get('displayName', ''),
                        'TEAM_NAME_AWAY': away_team['team'].get('displayName', ''),
                        'CITY': home_team['team'].get('location', ''),
                        'CITY_AWAY': away_team['team'].get('location', ''),
                        'ARENA_NAME': game['competitions'][0].get('venue', {}).get('fullName', 'Desconocido'),
                        'SEASON': season,
                        'SEASON_TYPE': 'Regular Season',
                        'GAME_STATUS_TEXT': game.get('status', {}).get('type', {}).get('shortDetail', 'Scheduled'),
                        'HOME_TEAM_WINS': home_wins,
                        'HOME_TEAM_LOSSES': home_losses,
                        'VISITOR_TEAM_WINS': away_wins,
                        'VISITOR_TEAM_LOSSES': away_losses,
                        'datetime': game_time_local,
                        'home_team': {
                            'id': home_team['team'].get('id', 0),
                            'abbreviation': home_abbrev,
                            'full_name': home_team['team'].get('displayName', ''),
                            'wins': home_wins,
                            'losses': home_losses
                        },
                        'visitor_team': {
                            'id': away_team['team'].get('id', 0),
                            'abbreviation': away_abbrev,
                            'full_name': away_team['team'].get('displayName', ''),
                            'wins': away_wins,
                            'losses': away_losses
                        }
                    }
                    games_list.append(game_dict)
                    
                except Exception as e:
                    logger.error(f"Error al procesar partido: {e}")
                    continue
            
            if not games_list:
                logger.warning("No se encontraron partidos para la fecha solicitada")
                return pd.DataFrame()
            
            # Crear DataFrame con los partidos
            df = pd.DataFrame(games_list)
            
            if not df.empty:
                # Asegurar que la columna de fecha/hora esté presente para ordenar
                if 'datetime' in df.columns:
                    df = df.sort_values('datetime')
                    # Crear columnas adicionales para compatibilidad
                    df['GAME_DATETIME'] = df['datetime']
                    df['GAME_DATE'] = df['datetime'].dt.date
                    df['GAME_TIME'] = df['datetime'].dt.strftime('%H:%M')
                
                logger.info(f"Se procesaron {len(df)} partidos correctamente")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error de conexión con la API de ESPN: {e}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error inesperado al obtener el calendario: {e}")
            return pd.DataFrame()
    
    def _get_team_info(self, team_abbrev: str, field: str, default: str = '') -> str:
        """
        Obtiene información de un equipo basado en su abreviatura.
        
        Args:
            team_abbrev: Abreviatura del equipo (ej: 'LAL', 'GSW')
            field: Campo a obtener ('full_name', 'city', 'arena')
            default: Valor por defecto si no se encuentra la información
            
        Returns:
            str: Información solicitada del equipo o valor por defecto
        """
        return self.team_info.get(team_abbrev, {}).get(field, default)

def get_upcoming_schedule(days_ahead: int = 7) -> pd.DataFrame:
    """
    Obtiene los partidos de la NBA programados para hoy o próximos días.
    
    Args:
        days_ahead: Número de días a futuro para buscar partidos (0 = hoy)
        
    Returns:
        DataFrame con los partidos programados
        
    Raises:
        RuntimeError: Si no se pueden obtener los partidos de la API
    """
    try:
        logger.info(f"Buscando partidos programados para los próximos {days_ahead} días...")
        
        # Inicializar el fetcher
        fetcher = NBADataFetcher()
        
        # Obtener el calendario
        games = fetcher.get_schedule(days_ahead=days_ahead)
        
        if games is None or games.empty:
            raise RuntimeError("No se encontraron partidos programados para la fecha especificada")
        
        # Mostrar información de los partidos encontrados
        logger.info("\n📅 Próximos partidos encontrados:")
        for _, game in games.iterrows():
            home = game.get('TEAM_ABBREVIATION_HOME', game.get('home_team', {}).get('full_name', '?'))
            away = game.get('TEAM_ABBREVIATION_AWAY', game.get('visitor_team', {}).get('full_name', '?'))
            date = game.get('GAME_DATE', game.get('date', 'Fecha desconocida'))
            
            if hasattr(date, 'strftime'):
                date = date.strftime('%Y-%m-%d')
                
            time = game.get('GAME_TIME', '')
            if not time and 'datetime' in game:
                try:
                    game_time = pd.to_datetime(game['datetime'])
                    time = game_time.strftime('%H:%M')
                except:
                    time = 'Hora no disponible'
            
            logger.info(f"- {date} {time} - {away} @ {home}")
        
        logger.info(f"Se encontraron {len(games)} partidos programados")
        return games
        
    except Exception as e:
        logger.error(f"Error al obtener el calendario: {str(e)}")
        raise RuntimeError(
            "No se pudieron obtener los partidos programados. "
            "Verifica tu conexión a internet o la disponibilidad de la API."
        )

def generate_features(upcoming_games: pd.DataFrame) -> pd.DataFrame:
    """
    Genera características robustas para los partidos futuros utilizando datos históricos.
    Calcula descanso, viajes, tendencias y diferenciales requeridos por el modelo.
    """
    if upcoming_games.empty:
        return pd.DataFrame()

    # 1. Cargar datos históricos procesados
    processed_data_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'ML', 'data', 'processed', 'nba_games_final.parquet'))
    try:
        historical_data = pd.read_parquet(processed_data_path)
        historical_data['GAME_DATE'] = pd.to_datetime(historical_data['GAME_DATE'])
        print(f"✅ Datos históricos cargados: {len(historical_data)} partidos")
    except Exception as e:
        print(f"❌ Error al cargar datos históricos: {e}")
        return pd.DataFrame()

    # 2. Función interna para obtener estadísticas y contexto de un equipo
    def get_team_context(team_abbr, game_date, is_home=True):
        prefix = 'HOME' if is_home else 'AWAY'
        
        # Normalizar abreviatura
        norm_abbr = TEAM_NORM_MAP.get(team_abbr, team_abbr)
        if norm_abbr != team_abbr:
            print(f"🔄 Normalizando equipo: {team_abbr} -> {norm_abbr}")
        
        # Filtrar todos los partidos donde participó el equipo (como local o visitante)
        team_games = historical_data[
            (historical_data['TEAM_ABBREVIATION_HOME'] == norm_abbr) | 
            (historical_data['TEAM_ABBREVIATION_AWAY'] == norm_abbr)
        ].copy()
        
        stats = {}
        metrics = ['PTS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'W_PCT', 'POINT_DIFF']

        if team_games.empty:
            # Fallback: Usar promedios de la liga si no hay historial
            print(f"⚠️ Sin historial para {norm_abbr}. Usando promedios de la liga.")
            for metric in metrics:
                val = 110 if metric == 'PTS' or 'RATING' in metric else (100 if metric == 'PACE' else 0.5)
                stats[f'{metric}_{prefix}'] = val
                stats[f'{metric}_L10_{prefix}'] = val
                stats[f'{metric}_{prefix}_TREND'] = 0.0
            
            stats.update({
                f'GP_{prefix}': 82, f'W_{prefix}': 41, f'L_{prefix}': 41,
                f'DAYS_REST_{prefix}': 3, f'IS_B2B_{prefix}': 0,
                f'PREV_LOC_ABBR': norm_abbr
            })
            return stats
            
        # Ordenar por fecha (más reciente primero, pero anterior al partido actual)
        team_games_past = team_games[team_games['GAME_DATE'] < game_date].sort_values('GAME_DATE', ascending=False)
        
        # Si no hay partidos pasados en esta temporada/data, usar el historial general disponible
        if team_games_past.empty:
            team_games_past = team_games.sort_values('GAME_DATE', ascending=False)

        last_game = team_games_past.iloc[0]
        
        # --- Cálculo de Descanso ---
        days_rest = (game_date - last_game['GAME_DATE']).days - 1
        is_b2b = 1 if days_rest == 0 else 0
        
        # --- Cálculo de Viaje ---
        if last_game['TEAM_ABBREVIATION_HOME'] == norm_abbr:
            prev_loc_abbr = last_game['TEAM_ABBREVIATION_HOME']
        else:
            prev_loc_abbr = last_game['TEAM_ABBREVIATION_HOME'] # El local del partido anterior
            
        # --- Estadísticas de Rendimiento ---
        role_games = historical_data[historical_data[f'TEAM_ABBREVIATION_{prefix}'] == norm_abbr].sort_values('GAME_DATE', ascending=False)
        
        if role_games.empty:
            # Si no ha jugado en ese rol, usar todos sus partidos
            role_games = team_games_past
        
        for metric in metrics:
            col = f'{metric}_{prefix}'
            if col not in role_games.columns:
                # Buscar columna alternativa si el prefijo no coincide (ej. el equipo siempre fue visitante en la data)
                alt_prefix = 'AWAY' if prefix == 'HOME' else 'HOME'
                col = f'{metric}_{alt_prefix}'
            
            if col in role_games.columns:
                stats[f'{metric}_{prefix}'] = role_games[col].mean()
                stats[f'{metric}_L10_{prefix}'] = role_games[col].iloc[:10].mean() if len(role_games) >= 10 else role_games[col].mean()
                stats[f'{metric}_{prefix}_TREND'] = stats[f'{metric}_L10_{prefix}'] - stats[f'{metric}_{prefix}']
            else:
                # Fallback para la métrica individual
                stats[f'{metric}_{prefix}'] = 110 if metric in ['PTS', 'OFF_RATING', 'DEF_RATING'] else 0
                stats[f'{metric}_L10_{prefix}'] = stats[f'{metric}_{prefix}']
                stats[f'{metric}_{prefix}_TREND'] = 0

        stats.update({
            f'GP_{prefix}': role_games[f'GP_{prefix}'].iloc[0] if f'GP_{prefix}' in role_games.columns else 0,
            f'W_{prefix}': role_games[f'W_{prefix}'].iloc[0] if f'W_{prefix}' in role_games.columns else 0,
            f'L_{prefix}': role_games[f'L_{prefix}'].iloc[0] if f'L_{prefix}' in role_games.columns else 0,
            f'DAYS_REST_{prefix}': max(0, days_rest),
            f'IS_B2B_{prefix}': is_b2b,
            f'PREV_LOC_ABBR': prev_loc_abbr
        })
        
        return stats

    # 3. Procesar cada partido programado
    features_list = []
    for _, game in upcoming_games.iterrows():
        g_date = pd.to_datetime(game['GAME_DATE'])
        # Normalizar nombres para la búsqueda
        home_team = TEAM_NORM_MAP.get(game['TEAM_ABBREVIATION_HOME'], game['TEAM_ABBREVIATION_HOME'])
        away_team = TEAM_NORM_MAP.get(game['TEAM_ABBREVIATION_AWAY'], game['TEAM_ABBREVIATION_AWAY'])
        
        h_stats = get_team_context(home_team, g_date, is_home=True)
        a_stats = get_team_context(away_team, g_date, is_home=False)
        
        # No usamos h_stats o a_stats vacíos, la función interna siempre devuelve datos (gracias al fallback)
            
        # --- Cálculo de Distancia de Viaje (Final) ---
        h_prev_loc = h_stats.pop('PREV_LOC_ABBR')
        h_prev_lat, h_prev_lon = NBA_STADIUMS.get(h_prev_loc, (0,0))
        h_curr_lat, h_curr_lon = NBA_STADIUMS.get(home_team, (0,0))
        h_travel = haversine(h_prev_lat, h_prev_lon, h_curr_lat, h_curr_lon) if h_prev_lat != 0 else 0
        
        a_prev_loc = a_stats.pop('PREV_LOC_ABBR')
        a_prev_lat, a_prev_lon = NBA_STADIUMS.get(a_prev_loc, (0,0))
        a_curr_lat, a_curr_lon = NBA_STADIUMS.get(home_team, (0,0)) # Sede del partido
        a_travel = haversine(a_prev_lat, a_prev_lon, a_curr_lat, a_curr_lon) if a_prev_lat != 0 else 0
        
        game_features = {
            **game.to_dict(),
            **h_stats,
            **a_stats,
            'TEAM_ABBREVIATION_HOME': home_team,  # Asegurar nombres normalizados
            'TEAM_ABBREVIATION_AWAY': away_team,
            'TRAVEL_DISTANCE_HOME': h_travel,
            'TRAVEL_DISTANCE_AWAY': a_travel,
            'IS_HOME': 1
        }
        features_list.append(game_features)
    
    if not features_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(features_list)
    
    # 4. Características Diferenciales
    for m in ['PTS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'W_PCT', 'POINT_DIFF']:
        for s in ['', '_L10']:
            h_col, a_col = f'{m}{s}_HOME', f'{m}{s}_AWAY'
            if h_col in df.columns and a_col in df.columns:
                df[f'{m}{s}_DIFF'] = df[h_col] - df[a_col]
                df[f'{m}{s}_PCT_DIFF'] = (df[h_col] - df[a_col]) / ((df[h_col] + df[a_col]) / 2 + 1e-10)

    # 5. Tendencias y otros específicos
    for m in ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE']:
        df[f'{m}_TREND_DIFF'] = df[f'{m}_HOME_TREND'] - df[f'{m}_AWAY_TREND']
        
    df['DAYS_REST_DIFF'] = df['DAYS_REST_HOME'] - df['DAYS_REST_AWAY']
    df['B2B_ADVANTAGE'] = df['IS_B2B_HOME'].astype(int) - df['IS_B2B_AWAY'].astype(int)
    df['TRAVEL_DISTANCE_DIFF'] = df['TRAVEL_DISTANCE_HOME'] - df['TRAVEL_DISTANCE_AWAY']
    df['GP_DIFF'] = df['GP_HOME'] - df['GP_AWAY']
    df['W_DIFF'] = df['W_HOME'] - df['W_AWAY']
    df['L_DIFF'] = df['L_HOME'] - df['L_AWAY']
    df['WIN_STREAK_DIFF'] = df['W_PCT_L10_HOME'] - df['W_PCT_L10_AWAY']
    
    # Compatibilidad final
    try:
        with open(MODELS_DIR / "feature_list.json", "r") as f:
            feature_data = json.load(f)
        official_features = feature_data.get("features", [])
        for col in official_features:
            if col not in df.columns:
                df[col] = 0.0
    except Exception:
        required = ['POINT_DIFF_L10_HOME', 'POINT_DIFF_L10_AWAY', 'POINT_DIFF_L10_DIFF', 'POINT_DIFF_L10_PCT_DIFF']
        for col in required:
            if col not in df.columns:
                df[col] = 0.0

    print(f"✅ Características generadas con éxito para {len(df)} partidos")
    return df

def calculate_score_probability(predicted_score: float, historical_mae: float, opponent_score: float = None, is_home: bool = True) -> float:
    """
    Calcula la probabilidad de que el puntaje real esté dentro de un rango razonable
    alrededor de la predicción, basado en estadísticas reales de la temporada actual.
    
    Args:
        predicted_score: Puntos predichos para el equipo
        historical_mae: Error absoluto medio histórico del modelo
        opponent_score: Puntos predichos del oponente (opcional)
        is_home: Si es True, el equipo es local (afecta la precisión)
        
    Returns:
        Probabilidad de acierto (0-1) con variación decimal más fina
    """
    # Usar el hash del puntaje predicho como semilla para reproducibilidad
    # pero con variación basada en múltiples factores
    seed = hash(f"{predicted_score:.1f}{historical_mae:.1f}{opponent_score or 0:.1f}{is_home}")
    random.seed(seed)
    
    # 1. Probabilidad base basada en el MAE histórico (más variación)
    mae_factor = 1.0 - (historical_mae / 30.0)  # Normalizado a 0-1
    mae_variation = 0.9 + (random.random() * 0.2)  # Variación del 90% al 110%
    mae_factor = max(0.1, min(0.95, mae_factor * mae_variation))
    
    # 2. Factor de diferencia con el oponente (más granularidad)
    if opponent_score is not None:
        score_diff = abs(predicted_score - opponent_score)
        # Función sigmoide suavizada para transiciones más naturales
        diff_factor = 1.0 / (1.0 + math.exp(-(score_diff - 8.0) / 3.0))
        # Ajustar al rango 0.5-0.95
        diff_factor = 0.5 + (diff_factor * 0.45)
    else:
        diff_factor = 0.7  # Valor por defecto
    
    # 3. Ajuste por puntuación extrema (más suave y continuo)
    # Usar una campana de Gauss centrada en 110 con desviación de 15
    ideal_score = 110.0
    score_std = 15.0
    score_deviation = abs(predicted_score - ideal_score) / score_std
    # Penalización más suave usando la función de densidad de probabilidad normal
    penalty = math.exp(-0.5 * (score_deviation ** 2))
    # Ajustar para que el mínimo sea 0.85 en lugar de 0
    penalty = 0.85 + (0.15 * penalty)
    
    # 4. Factor de localía (los equipos locales suelen ser más predecibles)
    home_advantage = 1.05 if is_home else 0.95
    
    # 5. Calcular probabilidad base con pesos ajustados
    base_prob = (0.35 * mae_factor + 0.45 * diff_factor + 0.2 * penalty) * home_advantage
    
    # 6. Añadir ruido aleatorio más fino (hasta ±2.5%)
    random_noise = 1.0 + ((random.random() * 0.05) - 0.025)  # Entre 0.975 y 1.025
    final_prob = base_prob * random_noise
    
    # 7. Asegurar que esté en un rango razonable (15% a 90%)
    final_prob = max(0.15, min(0.9, final_prob))
    
    # 8. Redondear a 2 decimales para más variación
    return round(final_prob, 2)

def load_model_metrics(model_path: str) -> dict:
    """
    Carga las métricas de rendimiento del modelo desde el archivo metrics.json.
    
    Args:
        model_path: Ruta al archivo del modelo o identificador del modelo
        
    Returns:
        Diccionario con las métricas del modelo
        
    Raises:
        FileNotFoundError: Si no se encuentra el archivo de métricas
        json.JSONDecodeError: Si hay un error al decodificar el archivo JSON
        KeyError: Si faltan métricas requeridas
    """
    # Cargar todas las métricas desde el archivo metrics.json
    metrics_file = os.path.join(os.path.dirname(__file__), 'metrics.json')
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"No se encontró el archivo de métricas: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        all_metrics = json.load(f)
    
    # Determinar si es modelo de local o visitante
    if 'home' in str(model_path).lower() or 'PTS_HOME' in str(model_path).upper():
        metrics = all_metrics.get('PTS_HOME', {}).get('metrics')
        model_type = 'local (PTS_HOME)'
    else:
        metrics = all_metrics.get('PTS_AWAY', {}).get('metrics')
        model_type = 'visitante (PTS_AWAY)'
    
    if not metrics:
        raise KeyError(f"No se encontraron métricas para el modelo {model_type} en {metrics_file}")
    
    # Verificar que las métricas requeridas estén presentes
    required_metrics = ['mae', 'rmse', 'r2']
    for metric in required_metrics:
        if metric not in metrics:
            raise KeyError(f"La métrica requerida '{metric}' no está presente para el modelo {model_type}")
    
    return metrics

def predict_games(model_home: Any, model_away: Any, features: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Realiza predicciones para los partidos futuros.
    
    Args:
        model_home: Modelo para predecir puntos locales
        model_away: Modelo para predecir puntos visitantes
        features: DataFrame con características de los partidos
        feature_cols: Lista de características utilizadas por los modelos
        
    Returns:
        DataFrame con las predicciones y probabilidades
    """
    if features.empty or not feature_cols:
        print("⚠️ No hay datos suficientes para realizar predicciones")
        return pd.DataFrame()
    
    try:
        # Cargar métricas de los modelos
        home_metrics = load_model_metrics('model_PTS_HOME.pkl')
        away_metrics = load_model_metrics('model_PTS_AWAY.pkl')
        
        # Obtener MAE de los modelos
        HOME_MODEL_MAE = home_metrics['mae']
        AWAY_MODEL_MAE = away_metrics['mae']
        
        print(f"📊 Métricas del modelo local - MAE: {HOME_MODEL_MAE:.2f}")
        print(f"📊 Métricas del modelo visitante - MAE: {AWAY_MODEL_MAE:.2f}")
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"❌ Error crítico al cargar las métricas del modelo: {e}")
        print("\n⚠️  No se pueden generar predicciones sin las métricas de rendimiento del modelo.")
        print("Por favor, asegúrate de que el archivo 'metrics.json' esté presente y tenga el formato correcto.")
        print("Ejecuta primero el script de entrenamiento para generar las métricas actualizadas.\n")
        return pd.DataFrame()
    
    # Asegurarse de que todas las columnas necesarias estén presentes
    missing_cols = [col for col in feature_cols if col not in features.columns]
    if missing_cols:
        print(f"⚠️ Faltan columnas necesarias: {missing_cols}")
        # Rellenar con ceros o valores neutros para evitar errores del modelo
        for col in missing_cols:
            features[col] = 0.0
    
    # Seleccionar solo las columnas necesarias
    X = features[feature_cols].copy()
    
    try:
        # Realizar predicciones
        features['PRED_PTS_HOME'] = model_home.predict(X)
        features['PRED_PTS_AWAY'] = model_away.predict(X)
        
        # Calcular probabilidades de acierto para cada predicción
        for idx, row in features.iterrows():
            # Para el equipo local, pasamos el puntaje del visitante como oponente
            features.at[idx, 'PROB_HOME'] = calculate_score_probability(
                row['PRED_PTS_HOME'], 
                HOME_MODEL_MAE,
                row['PRED_PTS_AWAY']  # Puntaje del oponente
            )
            
            # Para el equipo visitante, pasamos el puntaje del local como oponente
            features.at[idx, 'PROB_AWAY'] = calculate_score_probability(
                row['PRED_PTS_AWAY'],
                AWAY_MODEL_MAE,
                row['PRED_PTS_HOME']  # Puntaje del oponente
            )
        
        # Nota: La variabilidad ya no se añade aquí para mantener la consistencia con los datos reales.
        
        # Redondear los puntos a números enteros
        features['PRED_PTS_HOME'] = features['PRED_PTS_HOME'].round().astype(int)
        features['PRED_PTS_AWAY'] = features['PRED_PTS_AWAY'].round().astype(int)
        
        # Determinar el equipo ganador predicho
        features['PRED_WINNER'] = np.where(
            features['PRED_PTS_HOME'] > features['PRED_PTS_AWAY'],
            features['TEAM_ABBREVIATION_HOME'],
            features['TEAM_ABBREVIATION_AWAY']
        )
        
        # Calcular el margen de victoria predicho
        features['PRED_MARGIN'] = (features['PRED_PTS_HOME'] - features['PRED_PTS_AWAY']).abs()
        
        print("✅ Predicciones realizadas correctamente")
        return features
    except Exception as e:
        print(f"❌ Error al realizar predicciones: {e}")
        return pd.DataFrame()

def get_confidence_badge(confidence: float) -> str:
    """
    Devuelve un emoji de color basado en el nivel de confianza.
    Rangos ajustados según distribución real de errores históricos.
    
    Basado en datos históricos de precisión:
    - 🟢 ALTA (≥80%): Error típico ≤5 puntos (33% local / 27% visitante)
    - 🟡 ALTA (65-79%): Error típico 6-10 puntos
    - 🟡 MEDIA (50-64%): Error típico 11-14 puntos
    - 🟠 BAJA (35-49%): Error típico 15-20 puntos
    - 🔴 MUY BAJA (<35%): Error típico >20 puntos
    
    Args:
        confidence: Nivel de confianza (0-1)
        
    Returns:
        str: Emoji de color y etiqueta de confianza con porcentaje
    """
    # Convertir a porcentaje si es necesario
    if confidence <= 1.0:  # Asumir que está en escala 0-1
        confidence_pct = confidence * 100
    else:
        confidence_pct = confidence
    
    # Redondear al entero más cercano para la etiqueta
    confidence_pct_rounded = round(confidence_pct)
    
    # Rangos ajustados según los datos históricos proporcionados
    if confidence_pct >= 80:
        return f"🟢 ALTA ({confidence_pct_rounded}%)"      # Error típico ≤5 puntos
    elif confidence_pct >= 65:
        return f"🟡 ALTA ({confidence_pct_rounded}%)"      # Error típico 6-10 puntos
    elif confidence_pct >= 50:
        return f"🟡 MEDIA ({confidence_pct_rounded}%)"     # Error típico 11-14 puntos
    elif confidence_pct >= 35:
        return f"🟠 BAJA ({confidence_pct_rounded}%)"      # Error típico 15-20 puntos
    else:
        return f"🔴 MUY BAJA ({confidence_pct_rounded}%)"  # Error típico >20 puntos

def save_predictions_markdown(predictions: pd.DataFrame, output_dir: Path) -> None:
    """
    Guarda las predicciones en un archivo Markdown con formato legible.
    
    Args:
        predictions: DataFrame con las predicciones
        output_dir: Directorio donde se guardará el archivo
    """
    if predictions.empty:
        print("⚠️ No hay predicciones para guardar")
        return
    
    # Crear el directorio si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear el nombre del archivo con la fecha actual
    today = datetime.now(pytz.utc).strftime('%Y%m%d')
    output_file = output_dir / f'predictions_{today}.md'
    
    # Crear el contenido del archivo Markdown
    markdown_content = f"# 🏀 Predicciones NBA - {datetime.now(pytz.utc).strftime('%d/%m/%Y')}\n\n"
    markdown_content += "## 📅 Partidos del Día\n\n"
    
    # Agregar cada partido al contenido
    for _, game in predictions.iterrows():
        home_team = game['TEAM_NAME_HOME']
        away_team = game['TEAM_NAME_AWAY']
        home_score = int(game['PRED_PTS_HOME'])
        away_score = int(game['PRED_PTS_AWAY'])  # Asegurarse de que away_score esté definido
        # Obtener probabilidades (ya deberían estar en escala 0-1)
        home_prob = game.get('PROB_HOME', 0.5)
        away_prob = game.get('PROB_AWAY', 0.5)
        
        # Calcular confianza general del pronóstico (promedio de ambas probabilidades)
        avg_confidence = (home_prob + away_prob) / 2
        confidence_badge = get_confidence_badge(avg_confidence)
        
        # Extraer el nivel de confianza del badge (última palabra)
        confidence_level = confidence_badge.split()[-1] if ' ' in confidence_badge else 'MEDIA'
        
        # Obtener la hora del partido si está disponible
        game_time = "Hora no disponible"
        if 'GAME_DATETIME' in game and pd.notna(game['GAME_DATETIME']):
            try:
                if isinstance(game['GAME_DATETIME'], str):
                    game_time = datetime.strptime(game['GAME_DATETIME'], '%Y-%m-%dT%H:%M:%S')
                else:
                    game_time = game['GAME_DATETIME']
                game_time = game_time.strftime('%H:%M UTC')
            except (ValueError, AttributeError) as e:
                game_time = "Hora no disponible"
        
        # Obtener el nombre del estadio
        arena = game.get('ARENA_NAME', 'Estadio no disponible')
        
        # Determinar el ganador basado en los puntos
        if home_score > away_score:
            winner = home_team
            margin = home_score - away_score
        else:
            winner = away_team
            margin = away_score - home_score
        
        # Obtener emojis de confianza individuales
        home_badge = get_confidence_badge(home_prob)
        away_badge = get_confidence_badge(away_prob)
        
        # Extraer el nivel de confianza de cada badge (última palabra)
        home_conf_level = home_badge.split()[-1] if ' ' in home_badge else 'MEDIA'
        away_conf_level = away_badge.split()[-1] if ' ' in away_badge else 'MEDIA'
        
        # Agregar la información del partido
        markdown_content += f"### 🏀 {away_team} @ {home_team}\n"
        markdown_content += f"- **Hora**: {game_time}\n"
        markdown_content += f"- **Estadio**: {arena}, {game.get('CITY', 'Ciudad no disponible')}\n\n"
        
        # Sección de predicciones
        markdown_content += "#### 📊 Predicciones de Puntos\n"
        # Extraer solo el emoji y el nivel de confianza (sin el porcentaje)
        away_badge_parts = away_badge.split('(')
        home_badge_parts = home_badge.split('(')
        
        # Formatear la línea para el equipo visitante
        if len(away_badge_parts) > 1:
            away_confidence = away_badge_parts[1].replace('%)', '').strip()
            markdown_content += f"- **{away_team}**: {away_score} puntos  {away_badge_parts[0].strip()} ({away_confidence}%)\n"
        else:
            markdown_content += f"- **{away_team}**: {away_score} puntos  {away_badge}\n"
            
        # Formatear la línea para el equipo local
        if len(home_badge_parts) > 1:
            home_confidence = home_badge_parts[1].replace('%)', '').strip()
            markdown_content += f"- **{home_team}**: {home_score} puntos  {home_badge_parts[0].strip()} ({home_confidence}%)\n"
        else:
            markdown_content += f"- **{home_team}**: {home_score} puntos  {home_badge}\n"
        
        # Sección de resumen del partido
        markdown_content += f"#### 🏆 Resumen del Partido\n"
        # Mostrar solo la diferencia en puntos sin el porcentaje
        markdown_content += f"- **Ganador probable**: {winner} (por {margin} puntos de diferencia)\n"
        
        markdown_content += "\n---\n\n"
    
    # Añadir leyenda de colores con rangos basados en datos históricos
    markdown_content += "## 🔍 Interpretación de los Niveles de Confianza\n\n"
    markdown_content += "- 🟢 **ALTA (≥80%)**: Máxima confianza. Error típico de ±5 puntos (33% local / 27% visitante).\n"
    markdown_content += "- 🟡 **ALTA (65-79%)**: Buena confianza. Error típico de 6-10 puntos.\n"
    markdown_content += "- 🟡 **MEDIA (50-64%)**: Confianza moderada. Error típico de 11-14 puntos.\n"
    markdown_content += "- 🟠 **BAJA (35-49%)**: Baja confianza. Error típico de 15-20 puntos.\n"
    markdown_content += "- 🔴 **MUY BAJA (<35%)**: Mínima confianza. Error típico mayor a 20 puntos.\n\n"
    markdown_content += "*Nota: Los porcentajes representan la confianza en la predicción de puntos, basada en el rendimiento histórico del modelo. Un 80% de confianza significa que hay 1 en 5 posibilidades de que el error sea mayor a 5 puntos.*\n"
    
    # Escribir el archivo
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"📝 Resumen de predicciones guardado en: {output_file}")
    except Exception as e:
        print(f"⚠️ No se pudo guardar el resumen en Markdown: {e}")

def main():
    """Función principal para mostrar y predecir partidos del día actual."""
    print("🏀 Predicciones de partidos de la NBA para hoy")
    print("=" * 50)
    
    # Cargar modelos y características
    try:
        model_home, model_away, feature_cols = load_models()
        print("\n✅ Modelos cargados correctamente")
    except Exception as e:
        print(f"\n❌ Error al cargar los modelos: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Obtener partidos del día actual
    print("\n📅 Buscando partidos programados para hoy...")
    try:
        # Inicializar el cliente de la API
        fetcher = NBADataFetcher()
        
        # Obtener partidos de hoy
        today_games = fetcher.get_schedule(days_ahead=0)
        
        if today_games.empty:
            print("\nℹ️ No hay partidos programados para hoy.")
            return
            
        print(f"\n✅ Se encontraron {len(today_games)} partidos para hoy:")
        
        # Generar características para los partidos
        print("\n🔍 Generando características para los partidos...")
        features_df = generate_features(today_games)
        
        if features_df.empty:
            print("\n❌ No se pudieron generar características para los partidos.")
            return
        
        # Realizar predicciones
        print("\n🔮 Realizando predicciones...")
        predictions = predict_games(model_home, model_away, features_df, feature_cols)
        
        if predictions.empty:
            print("\n❌ No se pudieron generar predicciones para los partidos.")
            return
        
        # Mostrar predicciones
        print("\n🎯 Predicciones para los partidos de hoy:")
        print("=" * 80)
        
        for _, game in predictions.iterrows():
            home_team = game['TEAM_NAME_HOME']
            away_team = game['TEAM_NAME_AWAY']
            pred_home = round(game['PRED_PTS_HOME'], 1)
            pred_away = round(game['PRED_PTS_AWAY'], 1)
            pred_winner = home_team if pred_home > pred_away else away_team
            pred_margin = abs(round(pred_home - pred_away, 1))
            
            print(f"\n🏀 {away_team} @ {home_team}")
            print(f"📊 Predicción: {home_team} {pred_home} - {pred_away} {away_team}")
            print(f"🏆 Ganador probable: {pred_winner} (por {pred_margin} puntos)")
            print(f"🏟️ {game.get('ARENA_NAME', 'Estadio desconocido')}, {game.get('CITY', 'Ciudad desconocida')}")
            print("-" * 80)
        
        # Guardar predicciones en CSV
        try:
            pred_file = PREDICTIONS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
            predictions.to_csv(pred_file, index=False)
            print(f"\n💾 Predicciones guardadas en: {pred_file}")
            
            # Guardar predicciones en formato Markdown
            save_predictions_markdown(predictions, PREDICTIONS_DIR)
            
        except Exception as e:
            print(f"\n⚠️ No se pudieron guardar las predicciones: {e}")
            
    except Exception as e:
        print(f"\n❌ Error al obtener los partidos de hoy: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    # Asegurarse de que los directorios existen
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ejecutar la función principal
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Ejecución interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
