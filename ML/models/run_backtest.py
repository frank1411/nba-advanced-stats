#!/usr/bin/env python3
"""
Script para ejecutar el backtesting de los modelos de la NBA.
"""
import sys
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Configuración de rutas
MODELS_DIR = Path(__file__).parent
ROOT_DIR = MODELS_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

# Cargar datos
print("📂 Cargando datos...")
df = pd.read_parquet(DATA_DIR / "nba_games_final.parquet")

# Filtrar solo temporada 2024 para pruebas
test_season = 2024
test_df = df[df['SEASON'] == test_season].copy()
print(f"📊 Partidos en temporada {test_season}: {len(test_df)}")

# Cargar modelos
try:
    print("🔍 Cargando modelos...")
    model_home = joblib.load(MODELS_DIR / "model_PTS_HOME.pkl")
    model_away = joblib.load(MODELS_DIR / "model_PTS_AWAY.pkl")
    print("✅ Modelos cargados correctamente")
except Exception as e:
    print(f"❌ Error al cargar los modelos: {e}")
    sys.exit(1)

# Obtener características del modelo
feature_cols = model_home.feature_names_in_
print(f"📊 Número de características: {len(feature_cols)}")

# Verificar que todas las características estén presentes en los datos
missing_cols = [col for col in feature_cols if col not in test_df.columns]
if missing_cols:
    print(f"⚠️  Faltan {len(missing_cols)} columnas en los datos. Asegúrate de que todas las características estén presentes.")
    print("Primeras 5 columnas faltantes:", missing_cols[:5])
    sys.exit(1)

# Preparar datos de prueba
X_test = test_df[feature_cols].copy()

# Hacer predicciones
print("\n🔮 Generando predicciones...")
test_df['PRED_PTS_HOME'] = model_home.predict(X_test)
test_df['PRED_PTS_AWAY'] = model_away.predict(X_test)

# Calcular métricas
mae_home = mean_absolute_error(test_df['PTS_HOME'], test_df['PRED_PTS_HOME'])
mae_away = mean_absolute_error(test_df['PTS_AWAY'], test_df['PRED_PTS_AWAY'])
rmse_home = np.sqrt(mean_squared_error(test_df['PTS_HOME'], test_df['PRED_PTS_HOME']))
rmse_away = np.sqrt(mean_squared_error(test_df['PTS_AWAY'], test_df['PRED_PTS_AWAY']))

# Calcular precisión en predicción de ganadores
test_df['PRED_WINNER'] = np.where(test_df['PRED_PTS_HOME'] > test_df['PRED_PTS_AWAY'], 
                                 test_df['TEAM_ABBREVIATION_HOME'], 
                                 test_df['TEAM_ABBREVIATION_AWAY'])
test_df['ACTUAL_WINNER'] = np.where(test_df['PTS_HOME'] > test_df['PTS_AWAY'], 
                                   test_df['TEAM_ABBREVIATION_HOME'], 
                                   test_df['TEAM_ABBREVIATION_AWAY'])
accuracy = (test_df['PRED_WINNER'] == test_df['ACTUAL_WINNER']).mean() * 100

# Mostrar resultados
print("\n📊 RESULTADOS DEL BACKTESTING")
print("=" * 50)
print(f"📅 Temporada: {test_season}")
print(f"🎯 Total de partidos: {len(test_df)}")
print("\n🏀 Predicción de Puntos:")
print(f"  - Puntos locales (MAE): {mae_home:.2f}")
print(f"  - Puntos visitantes (MAE): {mae_away:.2f}")
print(f"  - RMSE local: {rmse_home:.2f}")
print(f"  - RMSE visitante: {rmse_away:.2f}")
print(f"\n🎯 Precisión en predicción de ganadores: {accuracy:.1f}%")

# Guardar resultados
test_df[['GAME_DATE', 'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY', 
         'PTS_HOME', 'PTS_AWAY', 'PRED_PTS_HOME', 'PRED_PTS_AWAY',
         'PRED_WINNER', 'ACTUAL_WINNER']].to_csv(MODELS_DIR / 'backtest_results.csv', index=False)
print(f"\n💾 Resultados guardados en: {MODELS_DIR / 'backtest_results.csv'}")
