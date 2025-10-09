#!/usr/bin/env python3
"""
Script de backtesting para evaluar el rendimiento del modelo de predicción de la NBA.

Realiza una validación retrospectiva dividiendo los datos históricos en conjuntos
de entrenamiento y prueba basados en la temporada, y evalúa las predicciones.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    confusion_matrix
)

# Configuración de rutas
MODELS_DIR = Path(__file__).parent
ROOT_DIR = MODELS_DIR.parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

# Añadir el directorio raíz al path para importaciones
sys.path.append(str(ROOT_DIR))

def load_data(input_path: str) -> pd.DataFrame:
    """Carga los datos de partidos de la NBA."""
    print(f"📂 Cargando datos desde {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Asegurarse de que las columnas necesarias existen
    required_cols = [
        'GAME_DATE', 'SEASON', 'SEASON_TYPE', 'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY',
        'PTS_HOME', 'PTS_AWAY', 'TARGET'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
    
    # Convertir fechas
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Ordenar por fecha
    df = df.sort_values('GAME_DATE')
    
    return df

def temporal_split(df: pd.DataFrame, test_season) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide los datos en conjuntos de entrenamiento y prueba por temporada."""
    # Asegurarse de que SEASON sea numérico
    df = df.copy()
    df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
    
    if str(test_season).lower() == "latest":
        test_season = int(df['SEASON'].max())
    else:
        test_season = int(test_season)
    
    train_df = df[df['SEASON'] < test_season].copy()
    test_df = df[df['SEASON'] == test_season].copy()
    
    print(f"📊 División temporal:")
    print(f"   - Entrenamiento: {len(train_df)} partidos (hasta temporada {test_season-1})")
    print(f"   - Prueba: {len(test_df)} partidos (temporada {test_season})")
    
    return train_df, test_df

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
    """Evalúa las predicciones del modelo."""
    # Calcular métricas de regresión
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calcular métricas de clasificación binaria (si se proporcionan probabilidades)
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'mean_actual': float(np.mean(y_true)),
        'mean_pred': float(np.mean(y_pred))
    }
    
    if y_prob is not None:
        # Convertir a predicciones binarias (1 si predicho > 0.5, 0 en caso contrario)
        y_pred_binary = (y_prob > 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred_binary)
        
        # Calcular matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        metrics.update({
            'accuracy': float(accuracy),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'f1_score': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        })
    
    return metrics

def run_backtest(input_path: str, test_season: int = 2024) -> Dict:
    """Ejecuta el backtesting del modelo de predicción de la NBA."""
    # Cargar datos
    df = load_data(input_path)
    
    # Dividir datos
    train_df, test_df = temporal_split(df, test_season)
    
    if len(test_df) == 0:
        raise ValueError(f"No hay datos de prueba para la temporada {test_season}")
    
    # Cargar modelos entrenados
    model_home_path = MODELS_DIR / "model_PTS_HOME.pkl"
    model_away_path = MODELS_DIR / "model_PTS_AWAY.pkl"
    
    if not (model_home_path.exists() and model_away_path.exists()):
        raise FileNotFoundError("Los modelos entrenados no se encontraron. Ejecuta train_model.py primero.")
    
    model_home = joblib.load(model_home_path)
    model_away = joblib.load(model_away_path)
    
    # Cargar lista de features
    features_path = MODELS_DIR / "feature_list.json"
    with open(features_path, 'r') as f:
        feature_data = json.load(f)
    
    feature_cols = feature_data.get('features', [])
    
    # Preparar datos de prueba
    X_test = test_df[feature_cols].copy()
    
    # Hacer predicciones
    print("\n🔮 Generando predicciones...")
    test_df['PRED_PTS_HOME'] = model_home.predict(X_test)
    test_df['PRED_PTS_AWAY'] = model_away.predict(X_test)
    
    # Calcular diferencia de puntos predicha y real
    test_df['PRED_POINT_DIFF'] = test_df['PRED_PTS_HOME'] - test_df['PRED_PTS_AWAY']
    test_df['ACTUAL_POINT_DIFF'] = test_df['PTS_HOME'] - test_df['PTS_AWAY']
    
    # Calcular probabilidad implícita de victoria local
    # Usamos una distribución normal con desviación estándar de 11.5 (basado en RMSE del modelo)
    std_dev = 11.5
    test_df['IMPLIED_HOME_WIN_PROB'] = 1 - norm.cdf(0, loc=test_df['PRED_POINT_DIFF'], scale=std_dev)
    
    # Determinar el ganador predicho
    test_df['PRED_WINNER'] = np.where(
        test_df['PRED_POINT_DIFF'] > 0,
        test_df['TEAM_ABBREVIATION_HOME'],
        test_df['TEAM_ABBREVIATION_AWAY']
    )
    
    # Determinar el ganador real
    test_df['ACTUAL_WINNER'] = np.where(
        test_df['PTS_HOME'] > test_df['PTS_AWAY'],
        test_df['TEAM_ABBREVIATION_HOME'],
        test_df['TEAM_ABBREVIATION_AWAY']
    )
    
    # Calcular si la predicción fue correcta
    test_df['PREDICTION_CORRECT'] = (test_df['PRED_WINNER'] == test_df['ACTUAL_WINNER']).astype(int)
    
    # Evaluar predicciones de puntos
    print("\n📊 Evaluando predicciones de puntos:")
    home_metrics = evaluate_predictions(
        test_df['PTS_HOME'].values,
        test_df['PRED_PTS_HOME'].values
    )
    
    away_metrics = evaluate_predictions(
        test_df['PTS_AWAY'].values,
        test_df['PRED_PTS_AWAY'].values
    )
    
    # Evaluar predicciones de ganador
    print("\n🎯 Evaluando predicciones de ganador:")
    win_metrics = evaluate_predictions(
        (test_df['ACTUAL_WINNER'] == test_df['TEAM_ABBREVIATION_HOME']).astype(int).values,
        test_df['PRED_POINT_DIFF'].values,
        test_df['IMPLIED_HOME_WIN_PROB'].values
    )
    
    # Calcular precisión por rangos de confianza
    test_df['CONFIDENCE_BIN'] = pd.cut(
        test_df['IMPLIED_HOME_WIN_PROB'],
        bins=[0, 0.4, 0.6, 1.0],
        labels=['Baja confianza', 'Media confianza', 'Alta confianza'],
        include_lowest=True
    )
    
    accuracy_by_confidence = test_df.groupby('CONFIDENCE_BIN')['PREDICTION_CORRECT'].agg(
        ['count', 'mean']
    ).rename(columns={'count': 'n_games', 'mean': 'accuracy'})
    
    # Guardar resultados
    results = {
        'test_season': int(test_season),
        'n_games': len(test_df),
        'points_metrics': {
            'home': home_metrics,
            'away': away_metrics
        },
        'win_metrics': win_metrics,
        'accuracy_by_confidence': accuracy_by_confidence.to_dict('index'),
        'overall_accuracy': float(test_df['PREDICTION_CORRECT'].mean())
    }
    
    # Guardar predicciones detalladas
    output_cols = [
        'GAME_DATE', 'SEASON', 'SEASON_TYPE', 'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY',
        'PTS_HOME', 'PTS_AWAY', 'PRED_PTS_HOME', 'PRED_PTS_AWAY',
        'ACTUAL_POINT_DIFF', 'PRED_POINT_DIFF',
        'IMPLIED_HOME_WIN_PROB', 'PRED_WINNER', 'ACTUAL_WINNER',
        'PREDICTION_CORRECT', 'CONFIDENCE_BIN'
    ]
    
    # Asegurarse de que test_season sea un entero
    test_season_int = int(test_season) if isinstance(test_season, str) else test_season
    output_path = MODELS_DIR / f"backtest_results_{test_season_int-1}-{test_season_int}.csv"
    test_df[output_cols].to_csv(output_path, index=False)
    print(f"\n💾 Resultados detallados guardados en {output_path}")
    
    # Guardar métricas
    metrics_path = MODELS_DIR / f"backtest_metrics_{test_season_int-1}-{test_season_int}.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Métricas de evaluación guardadas en {metrics_path}")
    
    return results

def print_summary(results: Dict):
    """Imprime un resumen de los resultados del backtesting."""
    print("\n" + "="*80)
    print("📈 RESUMEN DE BACKTESTING")
    print("="*80)
    
    print(f"\n🏀 TEMPORADA DE PRUEBA: {results['test_season']-1}-{results['test_season']}")
    print(f"📊 TOTAL DE PARTIDOS: {results['n_games']}")
    
    print("\n🎯 PRECISIÓN GENERAL")
    print(f"   - Precisión: {results['overall_accuracy']*100:.1f}%")
    
    print("\n🎯 PRECISIÓN POR NIVEL DE CONFIANZA")
    for conf_level, metrics in results['accuracy_by_confidence'].items():
        print(f"   - {conf_level}: {metrics['accuracy']*100:.1f}% (n={metrics['n_games']})")
    
    print("\n📊 MÉTRICAS DE PUNTOS")
    print("   - Error absoluto medio (MAE):")
    print(f"     * Local: {results['points_metrics']['home']['mae']:.2f} puntos")
    print(f"     * Visitante: {results['points_metrics']['away']['mae']:.2f} puntos")
    
    print("\n   - Raíz del error cuadrático medio (RMSE):")
    print(f"     * Local: {results['points_metrics']['home']['rmse']:.2f} puntos")
    print(f"     * Visitante: {results['points_metrics']['away']['rmse']:.2f} puntos")
    
    print("\n🎲 MÉTRICAS DE CLASIFICACIÓN")
    print(f"   - Precisión: {results['win_metrics']['precision']*100:.1f}%")
    print(f"   - Sensibilidad (recall): {results['win_metrics']['recall']*100:.1f}%")
    print(f"   - Puntuación F1: {results['win_metrics']['f1_score']*100:.1f}%")
    
    print("\n✅ Backtesting completado")
    print("="*80 + "\n")

if __name__ == "__main__":
    import argparse
    from scipy.stats import norm
    
    parser = argparse.ArgumentParser(description='Backtesting del modelo de predicción de la NBA')
    parser.add_argument('--input', type=str, default=str(DATA_DIR / 'nba_games_final.parquet'),
                        help='Ruta al archivo de datos de entrada (parquet)')
    parser.add_argument('--test-season', type=str, default='latest',
                        help='Temporada a utilizar como conjunto de prueba (ej: 2024) o "latest"')
    
    args = parser.parse_args()
    
    try:
        results = run_backtest(args.input, args.test_season)
        print_summary(results)
    except Exception as e:
        print(f"\n❌ Error durante el backtesting: {str(e)}")
        sys.exit(1)
