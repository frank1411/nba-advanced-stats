"""
Módulo para generar predicciones de partidos de la NBA.

Este módulo carga modelos entrenados y genera predicciones para próximos partidos.
"""
import logging
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import norm

# Configuración de logging
logger = logging.getLogger(__name__)

# Configuración de rutas
BASE_DIR = Path(__file__).parents[2]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR.parent / "data"  # Ajustar según la estructura de directorios
PREDICTIONS_DIR = DATA_DIR / "predictions"

# Asegurar que los directorios existan
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Desviación estándar para el cálculo de probabilidades (basado en RMSE del modelo)
DEFAULT_STD_DEV = 11.5

def load_models() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Cargar modelos entrenados y sus metadatos.
    
    Returns:
        Tuple con:
        - Diccionario de modelos cargados
        - Metadatos de los modelos
    """
    # Cargar metadatos
    metadata_path = MODELS_DIR / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontraron metadatos del modelo en {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Cargar modelos
    models = {}
    for target in metadata['feature_metadata']['target_cols']:
        model_path = MODELS_DIR / f"model_{target}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo para {target} en {model_path}")
        
        models[target] = joblib.load(model_path)
        logger.info(f"Modelo cargado: {model_path}")
    
    return models, metadata

def get_upcoming_games(days_ahead: int = 7) -> pd.DataFrame:
    """
    Obtener los próximos partidos de la NBA utilizando la API oficial.
    
    Args:
        days_ahead: Número de días hacia adelante para buscar partidos
        
    Returns:
        DataFrame con los próximos partidos programados
    """
    from nba_api.stats.endpoints import leaguegamefinder, teamgamelog
    from nba_api.stats.static import teams
    from datetime import datetime, timedelta
    
    try:
        logger.info("Obteniendo próximos partidos de la API de la NBA...")
        
        # Obtener la fecha actual y la fecha límite
        today = datetime.today().date()
        end_date = today + timedelta(days=days_ahead)
        
        # Formatear fechas para la API
        date_from = today.strftime('%m/%d/%Y')
        date_to = end_date.strftime('%m/%d/%Y')
        
        # Obtener todos los partidos programados en el rango de fechas
        gamefinder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=date_from,
            date_to_nullable=date_to,
            league_id_nullable='00'  # 00 es el ID de la NBA
        )
        
        # Obtener los datos en un DataFrame
        games = gamefinder.get_data_frames()[0]
        
        if games.empty:
            logger.warning(f"No se encontraron partidos programados entre {date_from} y {date_to}")
            return pd.DataFrame()
        
        # Filtrar solo partidos futuros y eliminar duplicados
        games = games.drop_duplicates(subset=['GAME_ID'])
        
        # Crear un diccionario de equipos para mapear IDs a abreviaturas
        nba_teams = {team['id']: team['abbreviation'] for team in teams.get_teams()}
        
        # Procesar los partidos
        upcoming_games = []
        
        for _, game in games.iterrows():
            # Obtener detalles del partido
            game_id = game['GAME_ID']
            game_date = datetime.strptime(game['GAME_DATE'], '%Y-%m-%d').date()
            
            # Verificar si es un partido futuro
            if game_date < today:
                continue
            
            # Obtener equipos
            team_id = game['TEAM_ID']
            team_abbr = nba_teams.get(team_id, 'UNK')
            
            # El campo MATCHUP tiene el formato 'LAL @ BOS' o 'LAL vs. BOS'
            matchup = game['MATCHUP']
            
            # Extraer equipos del campo MATCHUP
            parts = matchup.split()
            if len(parts) >= 3:
                if parts[1] == 'vs.' or parts[1] == '@':
                    away_team = parts[0]
                    home_team = parts[2]
                else:
                    # Si el formato no es el esperado, saltar este partido
                    continue
                
                # Asegurarse de que los equipos coincidan con el ID del equipo
                if team_abbr != away_team and team_abbr != home_team:
                    # Si el ID del equipo no coincide con ninguno en el partido, usar el ID
                    if team_abbr == 'UNK':
                        if parts[1] == 'vs.':
                            home_team = f"UNK_{game['TEAM_ID']}"
                        else:  # @
                            away_team = f"UNK_{game['TEAM_ID']}"
                
                # Agregar a la lista de partidos
                upcoming_games.append({
                    'GAME_ID': game_id,
                    'GAME_DATE': game_date,
                    'HOME_TEAM': home_team,
                    'AWAY_TEAM': away_team,
                    'SEASON': game['SEASON_ID'][1:5],  # Ej: '22022' -> '2022'
                    'SEASON_TYPE': 'Playoffs' if 'Playoffs' in game['SEASON_ID'] else 'Regular'
                })
        
        if not upcoming_games:
            logger.warning(f"No se encontraron partidos futuros entre {date_from} y {date_to}")
            return pd.DataFrame()
        
        # Convertir a DataFrame y eliminar duplicados (cada partido aparece dos veces, uno por equipo)
        upcoming_df = pd.DataFrame(upcoming_games).drop_duplicates(subset=['GAME_ID'])
        
        # Ordenar por fecha
        upcoming_df = upcoming_df.sort_values('GAME_DATE').reset_index(drop=True)
        
        logger.info(f"Se encontraron {len(upcoming_df)} partidos programados")
        return upcoming_df
        
    except Exception as e:
        logger.error(f"Error al obtener próximos partidos: {str(e)}", exc_info=True)
        return pd.DataFrame()

def prepare_prediction_features(upcoming_games: pd.DataFrame, historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara características para los próximos partidos basadas en datos históricos.
    
    Args:
        upcoming_games: DataFrame con los próximos partidos a predecir.
        historical_data: DataFrame con los datos históricos de partidos.
        
    Returns:
        DataFrame con las características generadas para los próximos partidos.
    """
    if upcoming_games.empty or historical_data.empty:
        logger.warning("Datos de entrada vacíos. No se pueden generar características.")
        return pd.DataFrame()
    
    try:
        logger.info("Generando características para los próximos partidos...")
        
        # Hacer una copia para no modificar el original
        features = upcoming_games.copy()
        
        # Asegurar que las fechas estén en formato datetime
        if not pd.api.types.is_datetime64_any_dtype(features['GAME_DATE']):
            features['GAME_DATE'] = pd.to_datetime(features['GAME_DATE'])
        
        if not pd.api.types.is_datetime64_any_dtype(historical_data['GAME_DATE']):
            historical_data['GAME_DATE'] = pd.to_datetime(historical_data['GAME_DATE'])
        
        # Inicializar columnas de características
        feature_columns = [
            'HOME_WIN_PCT', 'AWAY_WIN_PCT', 'HOME_LAST_5', 'AWAY_LAST_5',
            'HOME_POINTS_FOR', 'HOME_POINTS_AGAINST', 'AWAY_POINTS_FOR', 
            'AWAY_POINTS_AGAINST', 'DAYS_REST_HOME', 'DAYS_REST_AWAY'
        ]
        
        for col in feature_columns:
            features[col] = 0.0
        
        # Para cada partido futuro, calcular características basadas en datos históricos
        for idx, game in features.iterrows():
            home_team = game['HOME_TEAM']
            away_team = game['AWAY_TEAM']
            game_date = game['GAME_DATE']
            
            # Filtrar datos históricos hasta la fecha del partido
            hist_up_to_game = historical_data[historical_data['GAME_DATE'] < game_date]
            
            # Calcular estadísticas para el equipo local
            home_games = hist_up_to_game[
                (hist_up_to_game['HOME_TEAM'] == home_team) | 
                (hist_up_to_game['AWAY_TEAM'] == home_team)
            ].sort_values('GAME_DATE', ascending=False)
            
            if not home_games.empty:
                # Calcular porcentaje de victorias
                home_wins = home_games[
                    ((home_games['HOME_TEAM'] == home_team) & (home_games['HOME_TEAM_WINS'] == 1)) |
                    ((home_games['AWAY_TEAM'] == home_team) & (home_games['HOME_TEAM_WINS'] == 0))
                ].shape[0]
                
                features.at[idx, 'HOME_WIN_PCT'] = home_wins / len(home_games) if len(home_games) > 0 else 0.5
                
                # Últimos 5 partidos (últimos 5 resultados: 1=victoria, 0=derrota)
                last_5 = []
                for _, row in home_games.head(5).iterrows():
                    if (row['HOME_TEAM'] == home_team and row['HOME_TEAM_WINS'] == 1) or \
                       (row['AWAY_TEAM'] == home_team and row['HOME_TEAM_WINS'] == 0):
                        last_5.append(1)
                    else:
                        last_5.append(0)
                features.at[idx, 'HOME_LAST_5'] = sum(last_5) / len(last_5) if last_5 else 0.5
                
                # Promedio de puntos anotados y recibidos
                home_points_for = []
                home_points_against = []
                
                for _, row in home_games.head(10).iterrows():  # Usar últimos 10 partidos
                    if row['HOME_TEAM'] == home_team:
                        home_points_for.append(row['PTS_HOME'])
                        home_points_against.append(row['PTS_AWAY'])
                    else:
                        home_points_for.append(row['PTS_AWAY'])
                        home_points_against.append(row['PTS_HOME'])
                
                features.at[idx, 'HOME_POINTS_FOR'] = np.mean(home_points_for) if home_points_for else 100
                features.at[idx, 'HOME_POINTS_AGAINST'] = np.mean(home_points_against) if home_points_against else 100
                
                # Días de descanso
                last_game = home_games.iloc[0] if not home_games.empty else None
                if last_game is not None:
                    last_game_date = last_game['GAME_DATE']
                    if isinstance(last_game_date, str):
                        last_game_date = pd.to_datetime(last_game_date)
                    days_rest = (game_date - last_game_date).days - 1
                    features.at[idx, 'DAYS_REST_HOME'] = max(0, min(days_rest, 7))  # Cap entre 0 y 7 días
            
            # Calcular estadísticas para el equipo visitante (similar al local)
            away_games = hist_up_to_game[
                (hist_up_to_game['HOME_TEAM'] == away_team) | 
                (hist_up_to_game['AWAY_TEAM'] == away_team)
            ].sort_values('GAME_DATE', ascending=False)
            
            if not away_games.empty:
                # Calcular porcentaje de victorias
                away_wins = away_games[
                    ((away_games['HOME_TEAM'] == away_team) & (away_games['HOME_TEAM_WINS'] == 1)) |
                    ((away_games['AWAY_TEAM'] == away_team) & (away_games['HOME_TEAM_WINS'] == 0))
                ].shape[0]
                
                features.at[idx, 'AWAY_WIN_PCT'] = away_wins / len(away_games) if len(away_games) > 0 else 0.5
                
                # Últimos 5 partidos
                last_5 = []
                for _, row in away_games.head(5).iterrows():
                    if (row['HOME_TEAM'] == away_team and row['HOME_TEAM_WINS'] == 1) or \
                       (row['AWAY_TEAM'] == away_team and row['HOME_TEAM_WINS'] == 0):
                        last_5.append(1)
                    else:
                        last_5.append(0)
                features.at[idx, 'AWAY_LAST_5'] = sum(last_5) / len(last_5) if last_5 else 0.5
                
                # Promedio de puntos anotados y recibidos
                away_points_for = []
                away_points_against = []
                
                for _, row in away_games.head(10).iterrows():  # Usar últimos 10 partidos
                    if row['HOME_TEAM'] == away_team:
                        away_points_for.append(row['PTS_HOME'])
                        away_points_against.append(row['PTS_AWAY'])
                    else:
                        away_points_for.append(row['PTS_AWAY'])
                        away_points_against.append(row['PTS_HOME'])
                
                features.at[idx, 'AWAY_POINTS_FOR'] = np.mean(away_points_for) if away_points_for else 100
                features.at[idx, 'AWAY_POINTS_AGAINST'] = np.mean(away_points_against) if away_points_against else 100
                
                # Días de descanso
                last_game = away_games.iloc[0] if not away_games.empty else None
                if last_game is not None:
                    last_game_date = last_game['GAME_DATE']
                    if isinstance(last_game_date, str):
                        last_game_date = pd.to_datetime(last_game_date)
                    days_rest = (game_date - last_game_date).days - 1
                    features.at[idx, 'DAYS_REST_AWAY'] = max(0, min(days_rest, 7))  # Cap entre 0 y 7 días
            
            # Calcular características adicionales basadas en el enfrentamiento directo
            head_to_head = hist_up_to_game[
                ((hist_up_to_game['HOME_TEAM'] == home_team) & (hist_up_to_game['AWAY_TEAM'] == away_team)) |
                ((hist_up_to_game['HOME_TEAM'] == away_team) & (hist_up_to_game['AWAY_TEAM'] == home_team))
            ].sort_values('GAME_DATE', ascending=False)
            
            if not head_to_head.empty:
                # Porcentaje de victorias en enfrentamientos directos
                home_wins_h2h = 0
                for _, row in head_to_head.head(5).iterrows():  # Últimos 5 enfrentamientos
                    if (row['HOME_TEAM'] == home_team and row['HOME_TEAM_WINS'] == 1) or \
                       (row['AWAY_TEAM'] == home_team and row['HOME_TEAM_WINS'] == 0):
                        home_wins_h2h += 1
                
                features.at[idx, 'H2H_WIN_PCT'] = home_wins_h2h / min(5, len(head_to_head))
            else:
                features.at[idx, 'H2H_WIN_PCT'] = 0.5  # Sin historial, asumir empate
        
        # Calcular diferencias entre equipos
        features['WIN_PCT_DIFF'] = features['HOME_WIN_PCT'] - features['AWAY_WIN_PCT']
        features['LAST_5_DIFF'] = features['HOME_LAST_5'] - features['AWAY_LAST_5']
        features['POINTS_FOR_DIFF'] = features['HOME_POINTS_FOR'] - features['AWAY_POINTS_FOR']
        features['POINTS_AGAINST_DIFF'] = features['HOME_POINTS_AGAINST'] - features['AWAY_POINTS_AGAINST']
        features['DAYS_REST_DIFF'] = features['DAYS_REST_HOME'] - features['DAYS_REST_AWAY']
        
        # Asegurarse de que todas las columnas necesarias estén presentes
        required_columns = [
            'HOME_WIN_PCT', 'AWAY_WIN_PCT', 'WIN_PCT_DIFF',
            'HOME_LAST_5', 'AWAY_LAST_5', 'LAST_5_DIFF',
            'HOME_POINTS_FOR', 'AWAY_POINTS_FOR', 'POINTS_FOR_DIFF',
            'HOME_POINTS_AGAINST', 'AWAY_POINTS_AGAINST', 'POINTS_AGAINST_DIFF',
            'DAYS_REST_HOME', 'DAYS_REST_AWAY', 'DAYS_REST_DIFF',
            'H2H_WIN_PCT'
        ]
        
        # Añadir columnas faltantes con valores por defecto
        for col in required_columns:
            if col not in features.columns:
                if 'DIFF' in col:
                    features[col] = 0.0
                elif 'HOME' in col:
                    features[col] = 0.5
                elif 'AWAY' in col:
                    features[col] = 0.5
                else:
                    features[col] = 0.0
        
        logger.info(f"Características generadas para {len(features)} partidos")
        return features[['GAME_ID'] + required_columns]
    
    except Exception as e:
        logger.error(f"Error al generar características: {str(e)}", exc_info=True)
        return pd.DataFrame()

def generate_markdown_predictions(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Genera un archivo Markdown con las predicciones formateadas.
    
    Args:
        predictions_df: DataFrame con las predicciones
        output_dir: Directorio de salida para el archivo
    """
    try:
        # Crear copia para no modificar el DataFrame original
        df = predictions_df.copy()
        
        # Ordenar por diferencia de puntos (de mayor a menor)
        df['PRED_POINT_DIFF_ABS'] = df['PRED_POINT_DIFF'].abs()
        df_sorted = df.sort_values(['GAME_DATE', 'PRED_POINT_DIFF_ABS'], ascending=[True, False])
        
        # Formatear fechas
        df_sorted['GAME_DATE'] = pd.to_datetime(df_sorted['GAME_DATE']).dt.strftime('%d/%m/%Y')
        
        # Crear contenido Markdown
        md_content = "# 🏀 Predicciones NBA - {}\n\n".format(
            datetime.now().strftime('%d/%m/%Y %H:%M')
        )
        
        # Agrupar por fecha
        for date, group in df_sorted.groupby('GAME_DATE'):
            md_content += f"## 📅 {date}\n\n"
            
            # Ordenar por probabilidad de victoria local (de mayor a menor)
            group = group.sort_values('IMPLIED_HOME_WIN_PROB', ascending=False)
            
            for _, row in group.iterrows():
                home_team = row['HOME_TEAM']
                away_team = row['AWAY_TEAM']
                home_pts = round(row['PRED_PTS_HOME'], 1)
                away_pts = round(row['PRED_PTS_AWAY'], 1)
                prob = row['IMPLIED_HOME_WIN_PROB'] * 100
                diff = row['PRED_POINT_DIFF_ABS']
                
                # Determinar si es favorito local o visitante
                if row['PRED_POINT_DIFF'] > 0:
                    favorite = f"**{home_team}**"
                    underdog = away_team
                    favorite_pts = home_pts
                    underdog_pts = away_pts
                else:
                    favorite = f"**{away_team}**"
                    underdog = home_team
                    favorite_pts = away_pts
                    underdog_pts = home_pts
                
                # Añadir partido al markdown
                md_content += (
                    f"### 🏆 {home_team} vs {away_team}\n"
                    f"- **Predicción**: {favorite} {favorite_pts} - {underdog_pts} {underdog}\n"
                    f"- **Diferencial**: {abs(row['PRED_POINT_DIFF']):.1f} puntos\n"
                    f"- **Prob. victoria local**: {prob:.1f}%\n"
                    f"- **Confianza**: {'Alta' if diff > 8 else 'Media' if diff > 4 else 'Baja'}\n\n"
                )
            
            md_content += "---\n\n"
        
        # Guardar archivo
        output_path = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.md"
        with open(output_path, 'w') as f:
            f.write(md_content)
            
        logger.info(f"Archivo de predicciones guardado en: {output_path}")
        
    except Exception as e:
        logger.error(f"Error al generar archivo Markdown: {str(e)}")
        raise

def generate_predictions() -> pd.DataFrame:
    """
    Generar predicciones para los próximos partidos.
    
    Returns:
        DataFrame con las predicciones
    """
    try:
        logger.info("Iniciando generación de predicciones...")
        
        # Cargar modelos y metadatos
        models, metadata = load_models()
        
        # Cargar datos históricos para generar características
        historical_path = DATA_DIR / "processed" / "nba_games_final.parquet"
        if not historical_path.exists():
            raise FileNotFoundError(f"No se encontraron datos históricos en {historical_path}")
        
        logger.info(f"Cargando datos históricos desde {historical_path}...")
        historical = pd.read_parquet(historical_path)
        
        # Asegurar que las columnas necesarias estén presentes
        required_columns = ['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 'PTS_HOME', 'PTS_AWAY', 'HOME_TEAM_WINS']
        for col in required_columns:
            if col not in historical.columns:
                raise ValueError(f"Columna requerida no encontrada en datos históricos: {col}")
        
        # Obtener próximos partidos
        logger.info("Obteniendo próximos partidos...")
        upcoming = get_upcoming_games()
        
        if upcoming.empty:
            logger.warning("No hay partidos próximos para predecir")
            return pd.DataFrame()
        
        # Preparar características para predicción
        logger.info("Generando características para los próximos partidos...")
        features = prepare_prediction_features(upcoming, historical)
        
        if features.empty:
            logger.error("No se pudieron generar características para los partidos")
            return pd.DataFrame()
        
        # Obtener las columnas de características esperadas por el modelo
        feature_cols = metadata.get('feature_metadata', {}).get('feature_cols', [])
        
        if not feature_cols:
            # Si no hay columnas definidas, usar todas las numéricas excepto GAME_ID
            feature_cols = [col for col in features.columns 
                          if col != 'GAME_ID' and features[col].dtype in ['int64', 'float64']]
            
            logger.warning(f"No se encontraron columnas de características en los metadatos. Usando {len(feature_cols)} columnas numéricas.")
        
        # Verificar que tengamos todas las características necesarias
        missing_cols = set(feature_cols) - set(features.columns)
        if missing_cols:
            logger.warning(f"Faltan {len(missing_cols)} características. Se usarán valores predeterminados.")
            for col in missing_cols:
                if 'DIFF' in col:
                    features[col] = 0.0
                elif 'HOME' in col:
                    features[col] = 0.5
                elif 'AWAY' in col:
                    features[col] = 0.5
                else:
                    features[col] = 0.0
        
        # Asegurar el orden correcto de las columnas
        X_pred = features[feature_cols].copy()
        
        logger.info(f"Datos preparados para predicción. Forma: {X_pred.shape}")
        
        # Hacer predicciones para cada objetivo
        predictions = {}
        for target, model in models.items():
            predictions[f'PRED_{target}'] = model.predict(X_pred)
        
        # Combinar predicciones con datos de los partidos
        for key, values in predictions.items():
            upcoming[key] = values
        
        # Calcular diferencia de puntos predicha
        if 'PRED_PTS_HOME' in upcoming.columns and 'PRED_PTS_AWAY' in upcoming.columns:
            upcoming['PRED_POINT_DIFF'] = upcoming['PRED_PTS_HOME'] - upcoming['PRED_PTS_AWAY']
            
            # Calcular probabilidad implícita de victoria local
            std_dev = metadata.get('metrics', {}).get('PTS_HOME', {}).get('rmse', DEFAULT_STD_DEV)
            upcoming['IMPLIED_HOME_WIN_PROB'] = 1 - norm.cdf(
                0, 
                loc=upcoming['PRED_POINT_DIFF'], 
                scale=std_dev
            )
            
            # Determinar ganador predicho
            upcoming['PRED_WINNER'] = np.where(
                upcoming['PRED_POINT_DIFF'] > 0,
                upcoming['HOME_TEAM'],
                upcoming['AWAY_TEAM']
            )
        
        # Seleccionar columnas para el resultado final
        output_cols = [
            'GAME_DATE', 'GAME_ID',
            'HOME_TEAM', 'AWAY_TEAM', 'SEASON', 'SEASON_TYPE',
            'PRED_PTS_HOME', 'PRED_PTS_AWAY', 'PRED_POINT_DIFF',
            'IMPLIED_HOME_WIN_PROB', 'PRED_WINNER'
        ]
        
        # Filtrar columnas existentes
        output_cols = [col for col in output_cols if col in upcoming.columns]
        predictions_df = upcoming[output_cols].copy()
        
        # Guardar predicciones
        predictions_dir = DATA_DIR / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        # Guardar CSV
        output_file = predictions_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        upcoming.to_csv(output_file, index=False)
        logger.info(f"Predicciones guardadas en {output_file}")
        
        # Generar archivo Markdown
        generate_markdown_predictions(predictions_df, predictions_dir)
        
        return predictions_df
        
    except Exception as e:
        logger.error(f"Error al generar predicciones: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Configurar logging para ejecución directa
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("prediction.log"),
            logging.StreamHandler()
        ]
    )

    # Ejecutar generación de predicciones
    try:
        predictions = generate_predictions()
        print("\nPredicciones generadas exitosamente!")
        print(predictions[['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM', 
                         'PRED_PTS_HOME', 'PRED_PTS_AWAY', 'PRED_WINNER']])
    except Exception as e:
        logger.error(f"Error al ejecutar el script: {str(e)}")
        sys.exit(1)
