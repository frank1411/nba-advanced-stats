import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
# Asegurar imports relativos desde el paquete ML
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ML.db import NBADatabase

def calculate_advanced_metrics(df):
    """
    Calcula métricas avanzadas basadas en los datos existentes

    Args:
        df: DataFrame con los datos de los partidos

    Returns:
        DataFrame con métricas avanzadas añadidas
    """
    # Hacer una copia para no modificar el DataFrame original
    df = df.copy()

    # 1. Calcular diferencia de rating ofensivo vs defensa rival
    df['HOME_NET_EFFICIENCY'] = df['OFF_RATING_HOME'] - df['DEF_RATING_AWAY']
    df['AWAY_NET_EFFICIENCY'] = df['OFF_RATING_AWAY'] - df['DEF_RATING_HOME']

    # 2. Calcular diferencia de ritmo
    df['PACE_DIFF'] = df['PACE_HOME'] - df['PACE_AWAY']

    # 3. Calcular tendencias recientes (últimos 5 partidos vs últimos 10)
    # Para equipos locales
    df['HOME_TREND_OFF'] = df['OFF_RATING_HOME'] - df['OFF_RATING_L10_HOME']
    df['HOME_TREND_DEF'] = df['DEF_RATING_HOME'] - df['DEF_RATING_L10_HOME']

    # Para equipos visitantes
    df['AWAY_TREND_OFF'] = df['OFF_RATING_AWAY'] - df['OFF_RATING_L10_AWAY']
    df['AWAY_TREND_DEF'] = df['DEF_RATING_AWAY'] - df['DEF_RATING_L10_AWAY']

    # 4. Calcular diferencia de tendencias entre equipos
    df['TREND_OFF_DIFF'] = df['HOME_TREND_OFF'] - df['AWAY_TREND_OFF']
    df['TREND_DEF_DIFF'] = df['HOME_TREND_DEF'] - df['AWAY_TREND_DEF']

    # 5. Calcular diferencia de eficiencia neta
    df['NET_EFFICIENCY_DIFF'] = df['HOME_NET_EFFICIENCY'] - df['AWAY_NET_EFFICIENCY']

    return df

def main():
    # Cargar datos desde SQLite
    print("Cargando datos desde SQLite...")
    with NBADatabase() as db:
        db.create_tables()
        df = db.get_raw_games()

    if df.empty:
        print("Error: No se encontraron datos en la base de datos.")
        return None

    # Calcular métricas avanzadas
    print("Calculando métricas avanzadas...")
    df = calculate_advanced_metrics(df)

    # Guardar datos procesados en SQLite
    with NBADatabase() as db:
        db.create_tables()
        # Insertar como tabla adicional para advanced stats
        df.to_sql("nba_advanced_stats", db._conn, if_exists="replace", index=False)
    print("Estadísticas avanzadas guardadas en SQLite")

    # Mostrar resumen de las nuevas columnas
    new_columns = [col for col in df.columns if 'NET_EFFICIENCY' in col or 'TREND' in col or 'DIFF' in col]
    print("\nNuevas métricas calculadas:")
    for col in new_columns:
        print(f"- {col}")

    return df

if __name__ == "__main__":
    df = main()
    if df is not None:
        print(f"Proceso completado. Se procesaron {len(df)} partidos.")
