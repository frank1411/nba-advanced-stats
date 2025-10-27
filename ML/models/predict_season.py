#!/usr/bin/env python3
"""
Script para generar predicciones para una temporada específica de la NBA.
"""
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import sys

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

def load_models() -> Tuple[object, object, List[str]]:
    """Carga los modelos entrenados y la lista de features."""
    try:
        model_home = joblib.load(MODELS_DIR / "model_PTS_HOME.pkl")
        model_away = joblib.load(MODELS_DIR / "model_PTS_AWAY.pkl")
        
        # Cargar la lista de features utilizadas en el entrenamiento
        feature_list = model_home.feature_names_in_.tolist()
        
        print(f"✅ Modelos cargados correctamente")
        print(f"📊 Número de características: {len(feature_list)}")
        
        return model_home, model_away, feature_list
    except Exception as e:
        print(f"❌ Error al cargar los modelos: {e}")
        sys.exit(1)

def load_data(season: int) -> pd.DataFrame:
    """Carga y filtra los datos para la temporada especificada."""
    try:
        input_path = PROCESSED_DATA_DIR / "nba_games_final.parquet"
        df = pd.read_parquet(input_path)
        
        # Filtrar por temporada
        season_df = df[df['SEASON'] == season].copy()
        
        print(f"📂 Datos cargados: {len(season_df)} partidos de la temporada {season}")
        return season_df
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")
        sys.exit(1)

def calculate_win_probability(home_points: float, away_points: float, std_dev: float = 12.0) -> float:
    """Calcula la probabilidad de victoria del equipo local."""
    point_diff = home_points - away_points
    return norm.cdf(point_diff / std_dev)

def main():
    # Temporada objetivo (2024-2025)
    target_season = 2024
    
    # Cargar modelos y características
    model_home, model_away, feature_cols = load_models()
    
    # Cargar datos de la temporada
    season_df = load_data(target_season)
    
    if len(season_df) == 0:
        print(f"⚠️  No se encontraron partidos para la temporada {target_season}")
        return
    
    # Verificar que todas las características necesarias estén presentes
    missing_cols = [col for col in feature_cols if col not in season_df.columns]
    if missing_cols:
        print(f"⚠️  Faltan {len(missing_cols)} columnas en los datos. Asegúrate de que todas las características estén presentes.")
        print("Primeras 5 columnas faltantes:", missing_cols[:5])
        return
    
    # Preparar características
    X = season_df[feature_cols].copy()
    
    # Hacer predicciones
    print("\n🔮 Generando predicciones...")
    season_df['PRED_PTS_HOME'] = model_home.predict(X)
    season_df['PRED_PTS_AWAY'] = model_away.predict(X)
    
    # Calcular diferencia y probabilidad de victoria
    season_df['PRED_POINT_DIFF'] = season_df['PRED_PTS_HOME'] - season_df['PRED_PTS_AWAY']
    season_df['PRED_WIN_PROB'] = season_df.apply(
        lambda x: calculate_win_probability(x['PRED_PTS_HOME'], x['PRED_PTS_AWAY']), 
        axis=1
    )
    
    # Determinar el ganador predicho
    season_df['PRED_WINNER'] = np.where(
        season_df['PRED_PTS_HOME'] > season_df['PRED_PTS_AWAY'],
        season_df['TEAM_ABBREVIATION_HOME'],
        season_df['TEAM_ABBREVIATION_AWAY']
    )
    
    # Guardar resultados
    output_cols = [
        'GAME_DATE', 'SEASON', 'SEASON_TYPE', 
        'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY',
        'PTS_HOME', 'PTS_AWAY', 'PRED_PTS_HOME', 'PRED_PTS_AWAY',
        'PRED_POINT_DIFF', 'PRED_WIN_PROB', 'PRED_WINNER'
    ]
    
    # Ordenar por fecha
    result_df = season_df[output_cols].sort_values('GAME_DATE')
    
    # Guardar en CSV
    output_path = MODELS_DIR / f'predictions_{target_season}.csv'
    result_df.to_csv(output_path, index=False)
    
    # Mostrar resumen
    print("\n📊 RESULTADOS DE LAS PREDICCIONES")
    print("=" * 50)
    print(f"📅 Temporada: {target_season}")
    print(f"🎯 Total de partidos: {len(result_df)}")
    print(f"🏆 Victorias locales predichas: {(result_df['PRED_PTS_HOME'] > result_df['PRED_PTS_AWAY']).sum()}")
    print(f"🏀 Puntos locales promedio (predichos): {result_df['PRED_PTS_HOME'].mean():.1f}")
    print(f"🏀 Puntos visitantes promedio (predichos): {result_df['PRED_PTS_AWAY'].mean():.1f}")
    print(f"📊 Probabilidad de victoria local promedio: {result_df['PRED_WIN_PROB'].mean()*100:.1f}%")
    print(f"\n💾 Resultados guardados en: {output_path}")

if __name__ == "__main__":
    from scipy.stats import norm  # Mover aquí para evitar dependencia innecesaria
    main()
