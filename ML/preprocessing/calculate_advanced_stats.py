import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuración
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

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
    # Cargar datos en bruto
    raw_data_path = RAW_DATA_DIR / 'nba_games_raw.parquet'
    if not raw_data_path.exists():
        print("Error: No se encontraron datos en bruto. Ejecuta primero fetch_nba_data.py")
        return None
    
    print(f"Cargando datos desde {raw_data_path}...")
    df = pd.read_parquet(raw_data_path)
    
    # Calcular métricas avanzadas
    print("Calculando métricas avanzadas...")
    df = calculate_advanced_metrics(df)
    
    # Guardar datos procesados
    output_path = PROCESSED_DATA_DIR / 'nba_advanced_stats.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\nEstadísticas avanzadas guardadas en {output_path}")
    
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
