"""
Módulo para actualizar los datos históricos de partidos de la NBA.

Este módulo se encarga de obtener los datos más recientes de partidos de la NBA
y actualizar el conjunto de datos histórico.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
import time

# Configuración de logging
logger = logging.getLogger(__name__)

# Configuración de rutas
BASE_DIR = Path(__file__).parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Asegurar que los directorios existan
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def load_historical_data() -> pd.DataFrame:
    """
    Cargar datos históricos existentes o crear un DataFrame vacío.
    
    Returns:
        pd.DataFrame: DataFrame con los datos históricos o vacío si no existen.
    """
    historical_path = PROCESSED_DATA_DIR / "nba_games_final.parquet"
    if historical_path.exists():
        logger.info(f"Cargando datos históricos desde {historical_path}")
        return pd.read_parquet(historical_path)
    
    logger.info("No se encontraron datos históricos. Se creará un nuevo conjunto.")
    return pd.DataFrame()

def fetch_season_games(season: int) -> pd.DataFrame:
    """
    Obtener datos de partidos para una temporada específica.
    
    Args:
        season: Año de la temporada (ej: 2023 para la temporada 2022-2023)
        
    Returns:
        pd.DataFrame con los datos de los partidos
    """
    # TODO: Implementar la obtención de datos de la API de la NBA
    # Por ahora, devolvemos un DataFrame vacío
    logger.warning(f"Función fetch_season_games no implementada. Temporada: {season}")
    return pd.DataFrame()

def preprocess_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesar los datos de partidos.
    
    Args:
        df: DataFrame con datos de partidos en bruto
        
    Returns:
        pd.DataFrame con los datos preprocesados
    """
    if df.empty:
        return df
    
    # Hacer una copia para no modificar el original
    df = df.copy()
    
    # TODO: Implementar preprocesamiento de datos
    # - Limpieza de columnas
    # - Conversión de tipos de datos
    # - Cálculo de características adicionales
    
    return df

def update_historical_data(start_season: int = 2016, end_season: int = 2025) -> None:
    """
    Actualizar los datos históricos con partidos de temporadas recientes.
    
    Args:
        start_season: Primera temporada a incluir
        end_season: Última temporada a incluir
    """
    logger.info(f"Iniciando actualización de datos históricos (temporadas {start_season}-{end_season})")
    
    # Cargar datos históricos existentes
    historical = load_historical_data()
    
    # Obtener partidos de cada temporada
    all_new_games = []
    
    for season in range(start_season, end_season + 1):
        logger.info(f"Obteniendo datos para la temporada {season}")
        
        # Obtener datos de la temporada
        season_games = fetch_season_games(season)
        
        if season_games.empty:
            logger.warning(f"No se encontraron datos para la temporada {season}")
            continue
        
        # Preprocesar datos
        season_games = preprocess_games(season_games)
        
        # Agregar a la lista de nuevos partidos
        all_new_games.append(season_games)
        
        # Pequeña pausa para no sobrecargar la API
        time.sleep(1)
    
    if not all_new_games:
        logger.warning("No se encontraron nuevos partidos para agregar")
        return
    
    # Combinar todos los nuevos partidos
    new_games = pd.concat(all_new_games, ignore_index=True)
    
    # Si hay datos históricos, combinar con los nuevos
    if not historical.empty:
        # Filtrar duplicados
        if 'GAME_ID' in historical.columns and 'GAME_ID' in new_games.columns:
            new_games = new_games[~new_games['GAME_ID'].isin(historical['GAME_ID'])]
        
        # Combinar con datos históricos
        updated = pd.concat([historical, new_games], ignore_index=True)
    else:
        updated = new_games
    
    # Eliminar duplicados por si acaso
    if 'GAME_ID' in updated.columns:
        updated = updated.drop_duplicates(subset=['GAME_ID'])
    
    # Guardar datos actualizados
    output_path = PROCESSED_DATA_DIR / "nba_games_final.parquet"
    updated.to_parquet(output_path, index=False)
    
    logger.info(f"✅ Datos históricos actualizados. Total de partidos: {len(updated)}")
    logger.info(f"   - Partidos históricos: {len(historical) if not historical.empty else 0}")
    logger.info(f"   - Nuevos partidos agregados: {len(new_games)}")
    logger.info(f"   - Archivo guardado en: {output_path}")

if __name__ == "__main__":
    # Configurar logging para ejecución directa
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("update_historical.log"),
            logging.StreamHandler()
        ]
    )
    
    # Ejecutar actualización
    update_historical_data()
