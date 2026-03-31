#!/usr/bin/env python3
"""
Entrenamiento de modelos de regresión para predecir PTS_HOME y PTS_AWAY
usando estadísticas históricas y contextuales de la NBA.

- Entrada por defecto: ML/data/processed/nba_games_final.parquet
- Holdout por temporada más reciente (o parametrizable por --holdout-season)
- Modelos: RandomForestRegressor y XGBoost
- Salidas en ML/models/:
  - model_home.pkl, model_away.pkl (modelos entrenados)
  - metrics.json (métricas de evaluación)
  - feature_list.json (lista de features utilizadas)

Uso:
  PYTHONPATH=. python -m ML.models.train_model \
    --input ML/data/processed/nba_games_final.parquet \
    --holdout-season latest \
    --model rf,xgb
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
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Módulos de terceros
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de regresión para predecir puntos")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_DIR / "nba_games_final.parquet"),
        help="Ruta al parquet procesado con features",
    )
    parser.add_argument(
        "--holdout-season",
        type=str,
        default="latest",
        help="Temporada a usar como holdout (YYYY) o 'latest' para usar la más reciente",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf,xgb",
        help="Modelos a evaluar separados por coma: 'rf' (RandomForest), 'xgb' (XGBoost)",
    )
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    """Carga los datos y asegura los tipos correctos."""
    df = pd.read_parquet(path)
    
    # Asegurar tipos básicos
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    
    # Asegurar que las columnas de puntos sean numéricas
    for col in ["PTS_HOME", "PTS_AWAY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Obtiene las columnas de features y targets."""
    # Columnas objetivo
    target_cols = ["PTS_HOME", "PTS_AWAY"]
    
    # Columnas a excluir (fuga de información o no predictivas)
    exclude_cols = {
        # Identificadores y metadatos
        "GAME_ID", "GAME_DATE", "SEASON", "SEASON_STR", "SEASON_TYPE",
        "TEAM_ID_HOME", "TEAM_ABBREVIATION_HOME",
        "TEAM_ID_AWAY", "TEAM_ABBREVIATION_AWAY",
        
        # Targets y derivados
        "PTS_HOME", "PTS_AWAY", "POINT_DIFF", "HOME_POINT_DIFF", "AWAY_POINT_DIFF",
        "WL_HOME", "WL_AWAY", "TARGET", "POINT_DIFF_TARGET",
        
        # Columnas con fuga (contienen información del partido actual)
        "PTS_DIFF", "PTS_PCT_DIFF"
    }
    
    # Columnas de features (numéricas y no excluidas)
    feature_cols = [
        col for col in df.columns 
        if col not in exclude_cols 
        and df[col].dtype in ["int64", "float64"]
    ]
    
    return feature_cols, target_cols


def temporal_split(
    df: pd.DataFrame, holdout_season: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide los datos en conjuntos de entrenamiento y prueba usando split 80/20 en la temporada actual.
    
    Estrategia:
    - Entrenamiento: Todas las temporadas anteriores + 80% de la temporada actual
    - Prueba: 20% final de la temporada actual
    """
    if "SEASON" not in df.columns:
        raise ValueError("La columna SEASON es necesaria para el split temporal")
    if "GAME_DATE" not in df.columns:
        raise ValueError("La columna GAME_DATE es necesaria para el split temporal")

    # Convertir SEASON a numérico si es necesario
    df["SEASON"] = pd.to_numeric(df["SEASON"], errors='coerce')
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors='coerce')
    
    if holdout_season == "latest":
        current_season = int(df["SEASON"].max())
    else:
        current_season = int(holdout_season)

    # Obtener datos históricos (todas las temporadas anteriores)
    historical_df = df[df["SEASON"] < current_season].copy()
    
    # Obtener datos de la temporada actual y ordenar por fecha
    current_season_df = df[df["SEASON"] == current_season].copy()
    current_season_df = current_season_df.sort_values("GAME_DATE")
    
    # Split 80/20 en la temporada actual
    split_ratio = 0.80
    split_idx = int(len(current_season_df) * split_ratio)
    
    train_current = current_season_df.iloc[:split_idx]
    test_current = current_season_df.iloc[split_idx:]
    
    # Combinar históricos + 80% de temporada actual para entrenamiento
    train_df = pd.concat([historical_df, train_current], ignore_index=True)
    test_df = test_current.copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Split inválido: train({len(train_df)}), test({len(test_df)}) para temporada {current_season}"
        )

    print(f"✅ Split temporal mejorado (80/20):")
    print(f"   - Entrenamiento: SEASON {int(historical_df['SEASON'].min())}-{int(historical_df['SEASON'].max())} ({len(historical_df)} partidos)")
    print(f"                  + SEASON {current_season} primeros 80% ({len(train_current)} partidos)")
    print(f"                  = TOTAL: {len(train_df)} partidos")
    print(f"   - Prueba: SEASON {current_season} últimos 20% ({len(test_df)} partidos)")
    print(f"             Desde {test_df['GAME_DATE'].min().strftime('%Y-%m-%d')} hasta {test_df['GAME_DATE'].max().strftime('%Y-%m-%d')}")
    
    return train_df, test_df


def build_models(models_to_try: List[str]) -> Dict[str, object]:
    """Crea instancias de los modelos de regresión a evaluar."""
    models = {}
    
    if "rf" in models_to_try:
        # n_jobs=1 para evitar errores de mutex en macOS
        models["random_forest"] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            n_jobs=1,
            random_state=42
        )
    
    if "xgb" in models_to_try:
        # n_jobs=1 para evitar errores de mutex en macOS
        models["xgboost"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1
        )
    
    if not models:
        raise ValueError("Ningún modelo válido especificado. Usa 'rf' o 'xgb'.")
    
    return models


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula métricas de evaluación para regresión."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_actual": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
    }


def train_and_evaluate_models(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    models: Dict[str, object],
    target_name: str
) -> Tuple[object, dict]:
    """Entrena y evalúa modelos para un target específico."""
    print(f"\n🔍 Entrenando modelos para {target_name}")
    print("=" * 50)
    
    best_model = None
    best_metrics = None
    best_name = None
    
    for name, model in models.items():
        print(f"\n⚙️  Entrenando {name}...")
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Evaluación
        metrics = evaluate_regression(y_test, y_pred)
        
        print(f"  - MAE: {metrics['mae']:.2f} puntos")
        print(f"  - RMSE: {metrics['rmse']:.2f} puntos")
        print(f"  - R²: {metrics['r2']:.4f}")
        print(f"  - Puntos reales promedio: {metrics['mean_actual']:.1f}")
        print(f"  - Puntos predichos promedio: {metrics['mean_pred']:.1f}")
        
        if best_metrics is None or metrics["r2"] > best_metrics["r2"]:
            best_model = model
            best_metrics = metrics
            best_name = name
    
    print(f"\n🏆 Mejor modelo para {target_name}: {best_name} (R² = {best_metrics['r2']:.4f})")
    return best_model, best_metrics

def main():
    args = parse_args()
    
    print("🚀 Iniciando entrenamiento de modelos de regresión para predecir puntos")
    print("=" * 70)
    
    # Cargar datos
    print("📂 Cargando datos...")
    df = load_data(args.input)
    print(f"✅ Datos cargados: {len(df)} partidos")
    
    # Obtener columnas de features y targets
    feature_cols, target_cols = get_feature_columns(df)
    print(f"📊 {len(feature_cols)} features seleccionadas")
    
    # Split temporal
    print(f"\n🔄 Dividiendo datos por temporada...")
    train_df, test_df = temporal_split(df, args.holdout_season)
    print(f"   - Conjunto de entrenamiento: {len(train_df)} partidos")
    print(f"   - Conjunto de prueba: {len(test_df)} partidos")
    
    # Preparar datos
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # Inicializar modelos
    models_to_try = [m.strip() for m in args.model.split(",") if m.strip()]
    models = build_models(models_to_try)
    
    # Entrenar y evaluar modelos para cada target
    results = {}
    
    for target in target_cols:
        print(f"\n🎯 Target: {target}")
        print("-" * 50)
        
        # Entrenar modelos para este target
        best_model, metrics = train_and_evaluate_models(
            X_train, train_df[target], 
            X_test, test_df[target],
            models, target
        )
        
        # Guardar el mejor modelo
        model_path = MODELS_DIR / f"model_{target}.pkl"
        joblib.dump(best_model, model_path)
        print(f"💾 Modelo guardado en {model_path}")
        
        results[target] = {"metrics": metrics, "model": str(best_model.__class__.__name__)}
    
    # Guardar lista de features usadas
    with open(MODELS_DIR / "feature_list.json", "w") as f:
        json.dump({"features": feature_cols, "targets": target_cols}, f, indent=2)
    
    # Guardar métricas
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Entrenamiento completado")
    print(f"📊 Métricas guardadas en {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
