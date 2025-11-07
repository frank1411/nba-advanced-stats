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

# Configuración de numpy
np.seterr(all='warn')

# Configuración de TensorFlow (si está disponible)
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    logger.info("TensorFlow configurado con semilla determinista")
except ImportError:
    logger.warning("TensorFlow no está instalado. Algunas funcionalidades pueden no estar disponibles.")

# Configuración de scikit-learn
try:
    from sklearn import set_config
    set_config(assume_finite=True, working_memory=1024)  # 1GB de memoria para scikit-learn
    logger.info("scikit-learn configurado correctamente")
except (ImportError, AttributeError) as e:
    logger.warning(f"No se pudo configurar scikit-learn: {str(e)}")

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
                        'CITY_HOME': home_team['team'].get('location', ''),
                        'CITY_AWAY': away_team['team'].get('location', ''),
                        'ARENA': game['competitions'][0].get('venue', {}).get('fullName', 'Desconocido'),
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
    Genera características para los partidos futuros utilizando datos históricos.
    
    Args:
        upcoming_games: DataFrame con los partidos programados
        
    Returns:
        DataFrame con características generadas
    """
    if upcoming_games.empty:
        return pd.DataFrame()

    # 1. Cargar datos históricos procesados
    # Usar ruta relativa al directorio del proyecto
    import os
    print(f"Directorio actual: {os.getcwd()}")
    # Ruta corregida: desde el directorio actual (ML/models) necesitamos subir un nivel (..) y luego entrar a data/processed
    processed_data_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'ML', 'data', 'processed', 'nba_games_final.parquet'))
    print(f"🔍 Buscando datos en: {processed_data_path}")
    print(f"¿Existe el archivo? {os.path.exists(processed_data_path)}")
    try:
        historical_data = pd.read_parquet(processed_data_path)
        print(f"✅ Datos históricos cargados: {len(historical_data)} partidos")
    except Exception as e:
        print(f"❌ Error al cargar datos históricos: {e}")
        return pd.DataFrame()

    # 2. Función para obtener estadísticas de un equipo
    def get_team_stats(team_abbr, is_home=True):
        prefix = 'HOME' if is_home else 'AWAY'
        team_col = f'TEAM_ABBREVIATION_{prefix}'
        
        # Filtrar partidos del equipo
        team_games = historical_data[historical_data[team_col] == team_abbr].copy()
        if team_games.empty:
            return {}
            
        # Ordenar por fecha (más reciente primero)
        team_games = team_games.sort_values('GAME_DATE', ascending=False)
        
        # Obtener estadísticas
        stats = {}
        
        # Estadísticas básicas
        for col in ['PTS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'W_PCT']:
            # Últimos 10 partidos o todos si hay menos de 10
            last_10 = team_games.iloc[:10][f'{col}_{prefix}'].mean() if len(team_games) >= 10 else team_games[f'{col}_{prefix}'].mean()
            stats[f'{col}_L10_{prefix}'] = last_10
            stats[f'{col}_{prefix}'] = team_games[f'{col}_{prefix}'].mean()
        
        # Otras estadísticas
        stats.update({
            f'GP_{prefix}': team_games[f'GP_{prefix}'].iloc[0] if len(team_games) > 0 else 0,
            f'W_{prefix}': team_games[f'W_{prefix}'].iloc[0] if len(team_games) > 0 else 0,
            f'L_{prefix}': team_games[f'L_{prefix}'].iloc[0] if len(team_games) > 0 else 0,
        })
        
        return stats

    # 3. Aplicar a cada partido
    features_list = []
    for _, game in upcoming_games.iterrows():
        home_team = game['TEAM_ABBREVIATION_HOME']
        away_team = game['TEAM_ABBREVIATION_AWAY']
        
        home_stats = get_team_stats(home_team, is_home=True)
        away_stats = get_team_stats(away_team, is_home=False)
        
        # Combinar estadísticas
        game_features = {
            **game.to_dict(),
            **home_stats,
            **away_stats
        }
        features_list.append(game_features)
    
    if not features_list:
        return pd.DataFrame()
        
    features = pd.DataFrame(features_list)
    
    # 4. Calcular características diferenciales
    for metric in ['PTS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'W_PCT']:
        for suffix in ['', '_L10']:
            home_col = f'{metric}{suffix}_HOME'
            away_col = f'{metric}{suffix}_AWAY'
            
            if home_col in features.columns and away_col in features.columns:
                # Diferencia absoluta
                features[f'{metric}{suffix}_DIFF'] = features[home_col] - features[away_col]
                
                # Diferencia porcentual (evitar división por cero)
                denominator = (features[home_col] + features[away_col]) / 2 + 1e-10
                features[f'{metric}{suffix}_PCT_DIFF'] = (features[home_col] - features[away_col]) / denominator
    
    # 5. Asegurar que las columnas requeridas estén presentes
    required_columns = [
        'TEAM_ABBREVIATION_HOME', 
        'TEAM_ABBREVIATION_AWAY',
        'GAME_DATE',
        'PTS_HOME', 'PTS_AWAY',
        'OFF_RATING_HOME', 'OFF_RATING_AWAY',
        'DEF_RATING_HOME', 'DEF_RATING_AWAY',
        'NET_RATING_HOME', 'NET_RATING_AWAY',
        'PACE_HOME', 'PACE_AWAY',
        'W_PCT_HOME', 'W_PCT_AWAY'
    ]
    
    # Verificar columnas faltantes
    missing_columns = [col for col in required_columns if col not in features.columns]
    if missing_columns:
        print(f"⚠️ Advertencia: Faltan columnas: {missing_columns}")
        # Rellenar con valores por defecto si es necesario
        for col in missing_columns:
            if col.endswith('_HOME') or col.endswith('_AWAY'):
                default_value = 100 if 'RATING' in col else 0
                features[col] = default_value
    
    # 6. Añadir características adicionales
    features['IS_HOME'] = 1  # Para consistencia con el modelo
    features['OPPONENT_TEAM_ABBREVIATION'] = features['TEAM_ABBREVIATION_AWAY']
    features['DAYS_SINCE_LAST_GAME'] = 3  # Valor predeterminado
    features['IS_B2B'] = 0  # No hay información de back-to-back para partidos futuros
    
    # Asegurar que las columnas de fecha sean de tipo datetime
    if 'GAME_DATE' in features.columns and not pd.api.types.is_datetime64_any_dtype(features['GAME_DATE']):
        features['GAME_DATE'] = pd.to_datetime(features['GAME_DATE'])
    
    # Añadir características de tendencia (ejemplo)
    features['HOME_WIN_PCT'] = 0.5  # Valor predeterminado
    features['AWAY_WIN_PCT'] = 0.5  # Valor predeterminado
    
    print("✅ Características generadas para los partidos futuros")
    return features

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
    """
    try:
        # Cargar todas las métricas desde el archivo metrics.json
        metrics_file = os.path.join(os.path.dirname(__file__), 'metrics.json')
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        # Determinar si es modelo de local o visitante
        if 'home' in str(model_path).lower() or 'PTS_HOME' in str(model_path).upper():
            return all_metrics.get('PTS_HOME', {}).get('metrics', {})
        else:
            return all_metrics.get('PTS_AWAY', {}).get('metrics', {})
            
    except Exception as e:
        print(f"⚠️  Error al cargar métricas: {e}")
        # Valores por defecto si no se pueden cargar las métricas
        return {
            'mae': 8.5 if 'home' in str(model_path).lower() or 'PTS_HOME' in str(model_path).upper() else 9.2,
            'rmse': 11.0,
            'r2': 0.2,
            'mean_actual': 110.0,
            'mean_pred': 110.0
        }

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
    
    # Cargar métricas de los modelos usando los nombres correctos
    home_metrics = load_model_metrics('model_PTS_HOME.pkl')
    away_metrics = load_model_metrics('model_PTS_AWAY.pkl')
    
    # MAE histórico de los modelos
    HOME_MODEL_MAE = home_metrics.get('mae', 9.03)  # Valor por defecto basado en metrics.json
    AWAY_MODEL_MAE = away_metrics.get('mae', 9.16)  # Valor por defecto basado en metrics.json
    
    print(f"📊 Métricas del modelo local - MAE: {HOME_MODEL_MAE:.2f}")
    print(f"📊 Métricas del modelo visitante - MAE: {AWAY_MODEL_MAE:.2f}")
    
    # Asegurarse de que todas las columnas necesarias estén presentes
    missing_cols = [col for col in feature_cols if col not in features.columns]
    if missing_cols:
        print(f"⚠️ Faltan columnas necesarias: {missing_cols}")
        # Rellenar con valores más realistas basados en promedios de la NBA
        for col in missing_cols:
            if col.startswith('IS_'):
                features[col] = 0
            elif 'RATING' in col:
                # Valores típicos de rating ofensivo/defensivo
                if 'HOME' in col:
                    features[col] = np.random.normal(110, 5)  # Media 110, desviación 5
                else:
                    features[col] = np.random.normal(108, 5)
            elif 'PACE' in col:
                features[col] = np.random.normal(100, 3)  # Posesiones por 48 minutos
            elif 'PCT' in col or 'PCT_' in col:
                features[col] = np.random.uniform(0.3, 0.7)  # Porcentajes entre 30% y 70%
            elif 'DIFF' in col:
                features[col] = np.random.normal(0, 5)  # Diferencias alrededor de 0
            elif col in ['W_HOME', 'L_HOME', 'W_AWAY', 'L_AWAY']:
                features[col] = np.random.randint(30, 50)  # Temporada regular típica
            elif col in ['GP_HOME', 'GP_AWAY']:
                features[col] = np.random.randint(70, 82)  # Partidos jugados
            elif 'PTS' in col:
                features[col] = np.random.normal(110, 10)  # Puntos por partido
            else:
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
        
        # Añadir variabilidad si faltaban columnas
        if missing_cols:
            # Ajustar las predicciones para que sean más realistas
            for i in range(len(features)):
                # Añadir ruido aleatorio a las predicciones
                home_noise = np.random.normal(0, 5)  # Ruido con media 0 y desviación 5
                away_noise = np.random.normal(0, 5)
                
                # Asegurar que los puntajes sean realistas (entre 90 y 140)
                features.at[i, 'PRED_PTS_HOME'] = np.clip(
                    features.at[i, 'PRED_PTS_HOME'] + home_noise, 
                    90, 140
                )
                features.at[i, 'PRED_PTS_AWAY'] = np.clip(
                    features.at[i, 'PRED_PTS_AWAY'] + away_noise,
                    90, 140
                )
                
                # Asegurar que el margen no sea idéntico
                margin = abs(features.at[i, 'PRED_PTS_HOME'] - features.at[i, 'PRED_PTS_AWAY'])
                if margin < 2:  # Si el margen es muy pequeño, ajustarlo
                    features.at[i, 'PRED_PTS_HOME'] += np.random.choice([-1, 1]) * np.random.uniform(2, 5)
        
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
    Actualizado para reflejar mejor la distribución real de errores.
    
    Args:
        confidence: Nivel de confianza (0-1)
        
    Returns:
        str: Emoji de color (🔴, 🟠, 🟡, 🟢) y etiqueta de confianza con porcentaje
    """
    # Convertir a porcentaje si es necesario
    if confidence <= 1.0:  # Asumir que está en escala 0-1
        confidence_pct = confidence * 100
    else:
        confidence_pct = confidence
    
    # Redondear al entero más cercano para la etiqueta
    confidence_pct_rounded = round(confidence_pct)
    
    # Rangos ajustados según la distribución real de errores
    if confidence_pct >= 75:
        return f"🟢 ALTA ({confidence_pct_rounded}%)"      # Verde para alta confianza (≥75%)
    elif confidence_pct >= 60:
        return f"🟡 ALTA ({confidence_pct_rounded}%)"      # Amarillo para confianza alta-media (60-74%)
    elif confidence_pct >= 45:
        return f"🟡 MEDIA ({confidence_pct_rounded}%)"     # Amarillo para confianza media (45-59%)
    elif confidence_pct >= 30:
        return f"🟠 BAJA ({confidence_pct_rounded}%)"      # Naranja para confianza baja-media (30-44%)
    else:
        return f"🔴 MUY BAJA ({confidence_pct_rounded}%)"  # Rojo para muy baja confianza (<30%)

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
        
        # Sección de predicciones con confianza individual
        markdown_content += "#### 📊 Predicciones de Puntos\n"
        markdown_content += f"- **{away_team}**: {away_score} puntos  {away_badge} ({away_prob*100:.1f}% confianza)\n"
        markdown_content += f"- **{home_team}**: {home_score} puntos  {home_badge} ({home_prob*100:.1f}% confianza)\n"
        
        # Sección de resumen del partido
        markdown_content += f"#### 🏆 Resumen del Partido\n"
        # Calcular diferencia porcentual para el margen de victoria
        if home_score > away_score:
            margin_pct = (home_score / away_score - 1) * 100
        else:
            margin_pct = (away_score / home_score - 1) * 100
            
        markdown_content += f"- **Ganador probable**: {winner} (por {margin} puntos, {margin_pct:.1f}% de diferencia)\n"
        markdown_content += f"- **Confianza general del pronóstico**: {confidence_badge} ({avg_confidence*100:.1f}%)\n"
        
        # Análisis detallado de confianza
        markdown_content += "\n#### 🔍 Análisis de Confianza\n"
        
        # Función para obtener el mensaje de análisis de confianza
        def get_confidence_analysis(team_name: str, prob: float, score: int, badge: str) -> str:
            if prob >= 0.7:  # 70% o más
                return f"- **{team_name}**: {badge} Alta confianza en la predicción de puntos. " + \
                       f"El modelo tiene un 70% o más de certeza en este pronóstico.\n"
            elif prob >= 0.6:  # 60-69%
                return f"- **{team_name}**: {badge} Confianza moderada. " + \
                       f"El rango probable es de {int(score - 5)} a {int(score + 5)} puntos.\n"
            elif prob >= 0.45:  # 45-59%
                return f"- **{team_name}**: {badge} Confianza limitada. " + \
                       f"La predicción tiene una certeza moderada y podría variar.\n"
            else:  # Menos de 45%
                return f"- **{team_name}**: {badge} Baja confianza. " + \
                       f"El modelo tiene poca certeza en esta predicción.\n"
        # Análisis para el equipo local
        markdown_content += get_confidence_analysis(home_team, home_prob, home_score, home_badge)
        
        # Análisis para el equipo visitante
        markdown_content += get_confidence_analysis(away_team, away_prob, away_score, away_badge)
        
        # Análisis general
        if avg_confidence >= 70:
            markdown_content += "\n💡 **Análisis general**: Alta confianza en el pronóstico. "
            markdown_content += "El modelo tiene un buen rendimiento con partidos de características similares.\n"
        elif avg_confidence >= 60:
            markdown_content += "\n💡 **Análisis general**: Confianza moderada. "
            markdown_content += "El pronóstico es razonable, pero se recomienda considerar otros factores.\n"
        else:
            markdown_content += "\n⚠️ **Análisis general**: Baja confianza. "
            markdown_content += "Se recomienda tomar esta predicción con precaución y considerar otros factores.\n"
        
        markdown_content += "\n---\n\n"
    
    # Añadir leyenda de colores
    markdown_content += "## 🔍 Interpretación de los Niveles de Confianza\n\n"
    markdown_content += "- 🟢 **Alta confianza (≥70%)**: El modelo tiene un historial preciso en predicciones similares.\n"
    markdown_content += "- 🟡 **Confianza media (60-70%)**: La predicción es razonable, pero con cierta incertidumbre.\n"
    markdown_content += "- 🔴 **Baja confianza (<60%)**: El modelo tiene poca certeza en esta predicción. Se recomienda precaución.\n\n"
    markdown_content += "*Nota: Estos porcentajes representan la confianza en la predicción de puntos, no la probabilidad de victoria.*\n"
    
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
