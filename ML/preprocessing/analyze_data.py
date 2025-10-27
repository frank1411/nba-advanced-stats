import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# Configuración de estilo
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Cargar los datos
data_path = Path("ML/data/raw/nba_games_raw.parquet")
df = pd.read_parquet(data_path)

# Función para mostrar información detallada del DataFrame
def analyze_data(df):
    print("🔍 ANÁLISIS DE CALIDAD DE DATOS")
    print("=" * 80)
    
    # 1. Información general
    print("\n📋 1. INFORMACIÓN GENERAL")
    print("-" * 80)
    print(f"Total de filas: {len(df):,}")
    print(f"Total de columnas: {len(df.columns)}")
    print(f"Rango de fechas: {df['GAME_DATE'].min().date()} a {df['GAME_DATE'].max().date()}")
    
    # 2. Valores nulos
    print("\n📊 2. VALORES NULOS")
    print("-" * 80)
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100
    null_report = pd.DataFrame({
        'Valores nulos': null_counts,
        'Porcentaje': null_percent.round(2)
    })
    print(null_report[null_report['Valores nulos'] > 0])
    
    # 3. Análisis por temporada
    print("\n📅 3. ANÁLISIS POR TEMPORADA")
    print("-" * 80)
    
    # 3.1. Partidos por temporada y tipo
    df['SEASON_TYPE'] = df['SEASON_ID'].apply(lambda x: 'Playoffs' if 'Playoffs' in str(x) else 'Regular Season')
    season_summary = df.groupby(['SEASON_STR', 'SEASON_TYPE']).agg({
        'GAME_ID': 'count',
        'PTS_HOME': 'mean',
        'PTS_AWAY': 'mean',
    }).round(1)
    season_summary.columns = ['Partidos', 'PTS Local', 'PTS Visitante']
    print("\n📊 Resumen por temporada y tipo:")
    print(season_summary)
    
    # 3.2. Evolución de puntuación media
    season_avg = df.groupby('SEASON_STR').agg({
        'PTS_HOME': 'mean',
        'PTS_AWAY': 'mean'
    }).round(1)
    season_avg['Diferencia'] = season_avg['PTS_HOME'] - season_avg['PTS_AWAY']
    print("\n📈 Evolución de puntuación media por temporada:")
    print(season_avg)
    
    # 4. Equipos por temporada
    print("\n🏆 4. ANÁLISIS DE EQUIPOS POR TEMPORADA")
    print("-" * 80)
    
    # 4.1. Equipos por temporada
    teams_by_season = df.groupby('SEASON_STR').apply(
        lambda x: pd.Series({
            'Equipos Únicos': len(set(x['TEAM_ABBREVIATION_HOME'].unique()) | set(x['TEAM_ABBREVIATION_AWAY'].unique())),
            'Partidos por Equipo (avg)': (len(x) * 2) / len(set(x['TEAM_ABBREVIATION_HOME'].unique())),  # Aproximación
            'PTS Local (avg)': x['PTS_HOME'].mean().round(1),
            'PTS Visitante (avg)': x['PTS_AWAY'].mean().round(1)
        })
    )
    print("\n📊 Estadísticas por temporada:")
    print(teams_by_season)
    
    # 5. Análisis de puntuaciones
    print("\n🎯 5. ANÁLISIS DETALLADO DE PUNTUACIONES")
    print("-" * 80)
    
    # 5.1. Puntuaciones totales
    df['TOTAL_POINTS'] = df['PTS_HOME'] + df['PTS_AWAY']
    df['POINT_DIFF'] = df['PTS_HOME'] - df['PTS_AWAY']
    
    # 5.2. Resumen de puntuaciones por temporada
    points_summary = df.groupby('SEASON_STR').agg({
        'TOTAL_POINTS': ['mean', 'median', 'std', 'min', 'max'],
        'POINT_DIFF': 'mean'
    }).round(1)
    print("\n📊 Estadísticas de puntuación por temporada:")
    print(points_summary)
    
    # 5.3. Partidos con puntuaciones atípicas
    q1 = df['TOTAL_POINTS'].quantile(0.25)
    q3 = df['TOTAL_POINTS'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df['TOTAL_POINTS'] < lower_bound) | (df['TOTAL_POINTS'] > upper_bound)]
    print(f"\n🔍 Partidos con puntuaciones atípicas: {len(outliers):,} ({len(outliers)/len(df)*100:.1f}%)")
    
    # 6. Gráficos
    print("\n📊 Generando gráficos de análisis...")
    
    # Crear directorio para gráficos si no existe
    plots_dir = Path("ML/analysis/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 6.1. Evolución de puntuación media por temporada
    plt.figure(figsize=(12, 6))
    season_avg[['PTS_HOME', 'PTS_AWAY']].plot(kind='line', marker='o')
    plt.title('Evolución de Puntos por Partido (2018-2025)')
    plt.ylabel('Puntos por Partido')
    plt.xlabel('Temporada')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / 'puntos_por_temporada.png')
    
    # 6.2. Distribución de puntuaciones totales
    plt.figure(figsize=(12, 6))
    sns.histplot(df['TOTAL_POINTS'], bins=30, kde=True)
    plt.axvline(df['TOTAL_POINTS'].mean(), color='r', linestyle='--')
    plt.title('Distribución de Puntos Totales por Partido')
    plt.xlabel('Puntos Totales')
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(plots_dir / 'distribucion_puntos.png')
    
    # 6.3. Diferencia de puntos por temporada
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='SEASON_STR', y='POINT_DIFF')
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Diferencia de Puntos (Local - Visitante) por Temporada')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'diferencia_puntos_temporada.png')
    
    print(f"\n✅ Gráficos guardados en: {plots_dir}")
    
    # 7. Validación de resultados
    print("\n✅ 6. VALIDACIÓN DE RESULTADOS")
    print("-" * 80)
    df['RESULT_VALID'] = ((df['PTS_HOME'] > df['PTS_AWAY']) & (df['WL_HOME'] == 'W')) | \
                         ((df['PTS_HOME'] < df['PTS_AWAY']) & (df['WL_HOME'] == 'L'))
    invalid_results = df[~df['RESULT_VALID']]
    print(f"Partidos con resultados inconsistentes: {len(invalid_results)}")
    
    if not invalid_results.empty:
        print("\nEjemplo de partidos con resultados inconsistentes:")
        print(invalid_results[['SEASON_STR', 'GAME_DATE', 'TEAM_ABBREVIATION_HOME', 'PTS_HOME', 'WL_HOME', 
                              'TEAM_ABBREVIATION_AWAY', 'PTS_AWAY']].head())
    
    # 8. Recomendaciones
    print("\n📌 RECOMENDACIONES")
    print("-" * 80)
    print("1. Verificar partidos con puntuaciones atípicas")
    print("2. Analizar la disminución de puntos en ciertas temporadas")
    print("3. Considerar el efecto de la pandemia en las temporadas 2019-20 y 2020-21")
    print("4. Validar la consistencia de los equipos a lo largo de las temporadas")

# Ejecutar el análisis
analyze_data(df)
