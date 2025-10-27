#!/usr/bin/env python3
"""
Script de inferencia para predecir puntos de equipos locales y visitantes.

- Carga los modelos entrenados (`model_PTS_HOME.pkl` y `model_PTS_AWAY.pkl`).
- Lee un dataset de entrada (Parquet o CSV) con las features necesarias.
- Genera predicciones de puntos para equipos locales y visitantes.
- Calcula la diferencia de puntos y la probabilidad implícita de victoria.

Uso:
  PYTHONPATH=. python -m ML.models.predict \
    --input ML/data/processed/nba_games_final.parquet \
    --output ML/models/predictions.csv
"""
import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predicción de puntos en partidos de la NBA")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_DIR / "nba_games_final.parquet"),
        help="Ruta al archivo de entrada (Parquet o CSV)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODELS_DIR / "predictions.csv"),
        help="Ruta del CSV de salida con predicciones",
    )
    parser.add_argument(
        "--std-dev",
        type=float,
        default=12.0,
        help="Desviación estándar asumida para el cálculo de probabilidades (ajustar según métricas de validación)",
    )
    return parser.parse_args()


def load_input(path: str) -> pd.DataFrame:
    """Carga el archivo de entrada y asegura los tipos correctos."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {p}")
    
    # Cargar datos
    if p.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(p)
    elif p.suffix.lower() in {".csv"}:
        df = pd.read_csv(p)
    else:
        raise ValueError("Formato no soportado. Use .parquet o .csv")
    
    # Asegurar tipos de columnas clave
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    
    return df


def load_models() -> Tuple[Dict[str, Any], Dict[str, Any], list]:
    """Carga los modelos entrenados y la lista de features."""
    # Cargar lista de features
    features_path = MODELS_DIR / "feature_list.json"
    if not features_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de features: {features_path}")
    
    with open(features_path, "r") as f:
        feature_data = json.load(f)
    
    feature_cols = feature_data.get("features", [])
    target_cols = feature_data.get("targets", ["PTS_HOME", "PTS_AWAY"])
    
    # Cargar modelos
    models = {}
    for target in target_cols:
        model_path = MODELS_DIR / f"model_{target}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo para {target} en {model_path}")
        models[target] = joblib.load(model_path)
    
    return models, feature_cols, target_cols


def calculate_win_probability(home_points: float, away_points: float, std_dev: float) -> float:
    """
    Calcula la probabilidad de victoria del equipo local basado en la diferencia de puntos predicha.
    
    Args:
        home_points: Puntos predichos para el equipo local
        away_points: Puntos predichos para el equipo visitante
        std_dev: Desviación estándar de la diferencia de puntos (ajustar según métricas)
        
    Returns:
        Probabilidad de victoria del equipo local (0-1)
    """
    point_diff = home_points - away_points
    return norm.cdf(point_diff / std_dev)


def main():
    args = parse_args()
    
    print("🚀 Iniciando predicción de puntos para partidos de la NBA")
    print("=" * 60)
    
    try:
        # Cargar modelos y features
        print("🔍 Cargando modelos entrenados...")
        models, feature_cols, target_cols = load_models()
        print(f"✅ Modelos cargados: {', '.join(models.keys())}")
        print(f"📊 Total de features: {len(feature_cols)}")
        
        # Cargar datos de entrada
        print(f"📂 Cargando datos de {args.input}...")
        df = load_input(args.input)
        print(f"✅ Datos cargados: {len(df)} partidos")
        
        # Verificar columnas requeridas
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            print(f"⚠️  Advertencia: Faltan {len(missing_features)} features. Se rellenarán con 0.")
            df[missing_features] = 0
        
        # Seleccionar features para predicción
        X = df[feature_cols].copy()
        
        # Hacer predicciones para cada target
        predictions = {}
        for target, model in models.items():
            print(f"🔮 Prediciendo {target}...")
            predictions[target] = model.predict(X)
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'GAME_ID': df.get('GAME_ID', range(len(df))),
            'GAME_DATE': df.get('GAME_DATE', ''),
            'SEASON': df.get('SEASON', ''),
            'SEASON_TYPE': df.get('SEASON_TYPE', ''),
            'HOME_TEAM': df.get('TEAM_ABBREVIATION_HOME', ''),
            'AWAY_TEAM': df.get('TEAM_ABBREVIATION_AWAY', ''),
            'PRED_PTS_HOME': predictions.get('PTS_HOME', [0] * len(df)),
            'PRED_PTS_AWAY': predictions.get('PTS_AWAY', [0] * len(df)),
        })
        
        # Calcular diferencia y probabilidad de victoria
        results['PRED_POINT_DIFF'] = results['PRED_PTS_HOME'] - results['PRED_PTS_AWAY']
        results['IMPLIED_HOME_WIN_PROB'] = results.apply(
            lambda x: calculate_win_probability(x['PRED_PTS_HOME'], x['PRED_PTS_AWAY'], args.std_dev), 
            axis=1
        )
        results['PRED_WINNER'] = np.where(
            results['PRED_PTS_HOME'] > results['PRED_PTS_AWAY'], 
            results['HOME_TEAM'], 
            results['AWAY_TEAM']
        )
        
        # Ordenar columnas
        col_order = [
            'GAME_ID', 'GAME_DATE', 'SEASON', 'SEASON_TYPE',
            'HOME_TEAM', 'AWAY_TEAM', 'PRED_WINNER',
            'PRED_PTS_HOME', 'PRED_PTS_AWAY', 'PRED_POINT_DIFF', 'IMPLIED_HOME_WIN_PROB'
        ]
        results = results[col_order]
        
        # Guardar resultados
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False, float_format='%.2f')
        
        # Mostrar resumen
        print("\n📊 Resumen de predicciones:")
        print(f"- Partidos predichos: {len(results)}")
        print(f"- Puntos promedio (local/visitante): {results['PRED_PTS_HOME'].mean():.1f} / {results['PRED_PTS_AWAY'].mean():.1f}")
        print(f"- Diferencia promedio: {results['PRED_POINT_DIFF'].mean():.1f} puntos")
        print(f"- Probabilidad media de victoria local: {results['IMPLIED_HOME_WIN_PROB'].mean()*100:.1f}%")
        print(f"\n✅ Predicciones guardadas en {out_path}")
        
    except Exception as e:
        print(f"\n❌ Error durante la predicción: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
