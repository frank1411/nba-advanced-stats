import pandas as pd
import numpy as np
from pathlib import Path

# Cargar los datos
data_path = Path("ML/data/raw/nba_games_raw.parquet")
df = pd.read_parquet(data_path)

# Función para mostrar información detallada del DataFrame
def analyze_data(df):
    print("🔍 ANÁLISIS DE CALIDAD DE DATOS")
    print("=" * 80)
    
    # 1. Información general
    print("\n📋 1. INFORMACIÓN GENERAL")
    print("-" * 40)
    print(f"Total de filas: {len(df):,}")
    print(f"Total de columnas: {len(df.columns)}")
    print(f"Rango de fechas: {df['GAME_DATE'].min().date()} a {df['GAME_DATE'].max().date()}")
    
    # 2. Valores nulos
    print("\n📊 2. VALORES NULOS")
    print("-" * 40)
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100
    null_report = pd.DataFrame({
        'Valores nulos': null_counts,
        'Porcentaje': null_percent.round(2)
    })
    print(null_report[null_report['Valores nulos'] > 0])
    
    # 3. Estadísticas descriptivas
    print("\n📈 3. ESTADÍSTICAS DESCRIPTIVAS")
    print("-" * 40)
    print(df.describe(include='all'))
    
    # 4. Conteo de partidos por temporada
    print("\n🏀 4. PARTIDOS POR TEMPORADA Y TIPO")
    print("-" * 40)
    df['SEASON_TYPE'] = df['SEASON_ID'].apply(lambda x: 'Playoffs' if 'Playoffs' in str(x) else 'Regular Season')
    season_counts = df.groupby(['SEASON_ID', 'SEASON_TYPE']).size().unstack(fill_value=0)
    print(season_counts)
    
    # 5. Equipos únicos
    print("\n🏆 5. EQUIPOS ÚNICOS")
    print("-" * 40)
    home_teams = set(df['TEAM_ABBREVIATION_HOME'].unique())
    away_teams = set(df['TEAM_ABBREVIATION_AWAY'].unique())
    all_teams = home_teams.union(away_teams)
    print(f"Total equipos únicos: {len(all_teams)}")
    print("\nEquipos que solo aparecen como local o visitante (si los hay):")
    print(f"Solo locales: {home_teams - away_teams}")
    print(f"Solo visitantes: {away_teams - home_teams}")
    
    # 6. Análisis de puntuaciones
    print("\n🎯 6. ANÁLISIS DE PUNTUACIONES")
    print("-" * 40)
    print("Puntos por partido (Local vs Visitante):")
    print(f"Media local: {df['PTS_HOME'].mean():.1f}")
    print(f"Media visitante: {df['PTS_AWAY'].mean():.1f}")
    print(f"Diferencia media: {df['PTS_HOME'].mean() - df['PTS_AWAY'].mean():.1f}")
    
    # 7. Partidos con puntuaciones atípicas
    print("\n🔍 7. PARTIDOS CON PUNTUACIONES ATÍPICAS")
    print("-" * 40)
    high_scoring = df[df['PTS_HOME'] + df['PTS_AWAY'] > 250]
    low_scoring = df[df['PTS_HOME'] + df['PTS_AWAY'] < 150]
    print(f"Partidos con más de 250 puntos totales: {len(high_scoring)}")
    print(f"Partidos con menos de 150 puntos totales: {len(low_scoring)}")
    
    # 8. Validación de resultados
    print("\n✅ 8. VALIDACIÓN DE RESULTADOS")
    print("-" * 40)
    df['RESULT_VALID'] = ((df['PTS_HOME'] > df['PTS_AWAY']) & (df['WL_HOME'] == 'W')) | \
                         ((df['PTS_HOME'] < df['PTS_AWAY']) & (df['WL_HOME'] == 'L'))
    invalid_results = df[~df['RESULT_VALID']]
    print(f"Partidos con resultados inconsistentes: {len(invalid_results)}")
    if not invalid_results.empty:
        print("\nEjemplo de partidos con resultados inconsistentes:")
        print(invalid_results[['GAME_DATE', 'TEAM_ABBREVIATION_HOME', 'PTS_HOME', 'WL_HOME', 
                              'TEAM_ABBREVIATION_AWAY', 'PTS_AWAY']].head())

# Ejecutar el análisis
analyze_data(df)
