"""
Módulo para entrenar modelos de predicción de la NBA.

Este módulo contiene funciones para entrenar y evaluar modelos de machine learning
para predecir resultados de partidos de la NBA.
"""
# =============================================================================
# CONFIGURACIÓN DE SEMILLAS ALEATORIAS PARA REPRODUCIBILIDAD
# =============================================================================
import os
import random
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

# Módulos estándar
import logging
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configuración de logging
logger = logging.getLogger(__name__)

# Configuración de rutas
BASE_DIR = Path(__file__).parents[1]  # Subimos un nivel menos
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"  # Ahora apunta a ML/data
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Asegurar que los directorios existan
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data() -> pd.DataFrame:
    """
    Cargar datos para entrenamiento.
    
    Returns:
        pd.DataFrame con los datos de entrenamiento
    """
    data_path = PROCESSED_DATA_DIR / "nba_games_final.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")
    
    logger.info(f"Cargando datos desde {data_path}")
    return pd.read_parquet(data_path)

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Preparar características para el entrenamiento.
    
    Args:
        df: DataFrame con los datos de partidos
        
    Returns:
        Tuple con:
        - DataFrame con características
        - Diccionario con metadatos de características
    """
    if df.empty:
        raise ValueError("El DataFrame de entrada está vacío")
    
    # Hacer una copia para no modificar el original
    df = df.copy()
    
    # Definir columnas objetivo
    target_cols = ['PTS_HOME', 'PTS_AWAY']
    
    # Definir columnas a excluir
    exclude_cols = {
        'GAME_ID', 'GAME_DATE', 'SEASON', 'SEASON_STR', 'SEASON_TYPE',
        'TEAM_ID_HOME', 'TEAM_ABBREVIATION_HOME', 'TEAM_NAME_HOME',
        'TEAM_ID_AWAY', 'TEAM_ABBREVIATION_AWAY', 'TEAM_NAME_AWAY',
        'PTS_HOME', 'PTS_AWAY', 'WL_HOME', 'WL_AWAY', 'TARGET',
        'POINT_DIFF_TARGET', 'PTS_DIFF', 'PTS_PCT_DIFF', 'HOME_TEAM_WINS'
    }
    
    # Obtener características numéricas
    feature_cols = [
        col for col in df.columns 
        if col not in exclude_cols 
        and df[col].dtype in ['int64', 'float64']
    ]
    
    # Filtrar filas con valores faltantes en características o objetivos
    required_cols = feature_cols + target_cols
    df_clean = df[required_cols].dropna()
    
    # Verificar que tengamos suficientes datos
    if len(df_clean) == 0:
        raise ValueError("No hay datos suficientes después de la limpieza")
    
    # Preparar metadatos de características
    feature_metadata = {
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'n_samples': len(df_clean),
        'n_features': len(feature_cols),
        'feature_dtypes': {col: str(df_clean[col].dtype) for col in feature_cols}
    }
    
    return df_clean[feature_cols], feature_metadata

def train_models(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2, 
                random_state: int = 42) -> Dict[str, Any]:
    """
    Entrenar modelos de regresión para predecir puntos por equipo.
    
    Args:
        X: DataFrame con características
        y: DataFrame con variables objetivo
        test_size: Proporción de datos para prueba
        random_state: Semilla para reproducibilidad
        
    Returns:
        Diccionario con modelos entrenados y métricas
    """
    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Entrenar un modelo para cada objetivo
    models = {}
    metrics = {}
    
    for target in y.columns:
        logger.info(f"Entrenando modelo para {target}...")
        
        # Crear y entrenar modelo
        # n_jobs=1 para evitar errores de mutex en macOS
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1
        )
        
        model.fit(X_train, y_train[target])
        models[target] = model
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        metrics[target] = {
            'mae': mean_absolute_error(y_test[target], y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test[target], y_pred)),
            'r2': r2_score(y_test[target], y_pred),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test)
        }
        
        logger.info(f"  - {target} - MAE: {metrics[target]['mae']:.2f}, "
                   f"RMSE: {metrics[target]['rmse']:.2f}, "
                   f"R²: {metrics[target]['r2']:.4f}")
    
    return {
        'models': models,
        'metrics': metrics,
        'feature_importances': {
            target: dict(zip(X.columns, model.feature_importances_))
            for target, model in models.items()
        }
    }

def convert_numpy_types(obj):
    """
    Convierte tipos de datos NumPy a tipos nativos de Python para serialización JSON.
    """
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def save_models_and_metadata(models: Dict, metadata: Dict, output_dir: Path) -> None:
    """
    Guardar modelos y metadatos en disco.
    
    Args:
        models: Diccionario con modelos entrenados
        metadata: Metadatos del entrenamiento
        output_dir: Directorio de salida
    """
    # Asegurar que el directorio existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar cada modelo
    for target, model in models.items():
        model_path = output_dir / f"model_{target}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Modelo guardado en {model_path}")
    
    # Convertir tipos NumPy a tipos nativos de Python
    metadata_serializable = convert_numpy_types(metadata)
    
    # Guardar metadatos
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_serializable, f, indent=2)
    
    logger.info(f"Metadatos guardados en {metadata_path}")
    
    # Guardar métricas de rendimiento en un archivo separado para facilitar la lectura
    metrics_path = output_dir / "model_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("Métricas de rendimiento de los modelos:\n")
        f.write("=" * 50 + "\n\n")
        
        for target, metrics in metadata_serializable.get('metrics', {}).items():
            f.write(f"Modelo: {target}\n")
            f.write(f"- Error Absoluto Medio (MAE): {metrics.get('mae', 'N/A'):.2f}\n")
            f.write(f"- Raíz del Error Cuadrático Medio (RMSE): {metrics.get('rmse', 'N/A'):.2f}\n")
            f.write(f"- R² Score: {metrics.get('r2', 'N/A'):.4f}\n")
            f.write(f"- Muestras de entrenamiento: {metrics.get('n_samples_train', 'N/A'):,}\n")
            f.write(f"- Muestras de prueba: {metrics.get('n_samples_test', 'N/A'):,}\n\n")
    
    logger.info(f"Métricas detalladas guardadas en {metrics_path}")

def train_model(test_season: int = 2025) -> None:
    """
    Función principal para entrenar los modelos.
    
    Args:
        test_season: Temporada a usar para pruebas
    """
    try:
        # Cargar datos
        df = load_data()
        
        # Verificar que la temporada de prueba existe
        if 'SEASON' in df.columns:
            seasons = sorted(df['SEASON'].unique())
            if test_season not in seasons:
                logger.warning(f"La temporada {test_season} no está en los datos. "
                             f"Temporadas disponibles: {seasons}")
                test_season = max(seasons)
                logger.info(f"Usando la temporada más reciente: {test_season}")
        
        # Preparar características
        X, feature_metadata = prepare_features(df)
        
        # Obtener variables objetivo
        y = df[feature_metadata['target_cols']]
        
        # Entrenar modelos
        results = train_models(X, y)
        
        # Preparar metadatos completos
        metadata = {
            'feature_metadata': feature_metadata,
            'metrics': results['metrics'],
            'feature_importances': results['feature_importances'],
            'training_date': pd.Timestamp.now().isoformat(),
            'test_season': test_season
        }
        
        # Guardar modelos y metadatos
        save_models_and_metadata(results['models'], metadata, MODELS_DIR)
        
        logger.info("✅ Entrenamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Configurar logging para ejecución directa
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    
    # Ejecutar entrenamiento
    train_model()
