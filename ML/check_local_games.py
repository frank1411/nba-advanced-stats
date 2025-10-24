#!/usr/bin/env python3
"""
Script para ver los próximos partidos de la NBA usando datos locales.
"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Archivo con los datos de los partidos
GAMES_FILE = PROCESSED_DATA_DIR / 'nba_games_final.parquet'

def load_games_data():
    """Carga los datos de los partidos desde el archivo local."""
    try:
        print(f"📂 Cargando datos desde {GAMES_FILE}...")
        df = pd.read_parquet(GAMES_FILE)
        
        # Asegurarse de que la columna de fecha sea datetime
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        print(f"✅ Datos cargados correctamente. Total de partidos: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")
        return None

def get_upcoming_games(days_ahead=7):
    """Obtiene los próximos partidos de la NBA."""
    # Cargar datos
    df = load_games_data()
    if df is None or df.empty:
        return
    
    # Filtrar partidos futuros
    today = datetime.now()
    future_date = today + timedelta(days=days_ahead)
    
    # Filtrar partidos futuros
    future_games = df[
        (df['GAME_DATE'] >= today) & 
        (df['GAME_DATE'] <= future_date)
    ]
    
    if future_games.empty:
        print(f"\nℹ️ No hay partidos programados en los próximos {days_ahead} días.")
        return
    
    # Ordenar por fecha y hora
    future_games = future_games.sort_values(['GAME_DATE', 'GAME_TIME'])
    
    # Mostrar resultados
    print(f"\n📅 Próximos partidos (hasta {future_date.strftime('%Y-%m-%d')}):")
    print("=" * 60)
    
    for date, games_on_date in future_games.groupby('GAME_DATE'):
        print(f"\n📅 {date.strftime('%Y-%m-%d')} ({date.strftime('%A')})")
        print("-" * 60)
        
        for _, game in games_on_date.iterrows():
            home_team = game.get('TEAM_ABBREVIATION_HOME', 'Equipo Local')
            away_team = game.get('TEAM_ABBREVIATION_AWAY', 'Equipo Visitante')
            game_time = game.get('GAME_TIME', 'Hora no disponible')
            
            # Intentar obtener la arena si está disponible
            arena = game.get('ARENA_NAME', 'Por determinar')
            
            print(f"🏀 {home_team} vs {away_team}")
            print(f"   ⏰ {game_time}")
            
            if arena != 'Por determinar':
                print(f"   🏟️  {arena}")
            
            # Mostrar información adicional si está disponible
            if 'SEASON_TYPE' in game:
                print(f"   🏆 {game['SEASON_TYPE']}")
            
            print()  # Línea en blanco entre partidos
    
    print(f"\n✅ Total de partidos encontrados: {len(future_games)}")

if __name__ == "__main__":
    get_upcoming_games(days_ahead=14)  # Mostrar los próximos 14 días
