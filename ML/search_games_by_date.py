#!/usr/bin/env python3
"""
Script interactivo para buscar partidos de la NBA por fecha.
Permite al usuario introducir una fecha y muestra los partidos programados.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

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
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
        
        print(f"✅ Datos cargados correctamente. Total de partidos: {len(df):,}")
        return df
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")
        return None

def search_games_by_date(df, date_str):
    """Busca partidos para una fecha específica."""
    try:
        # Convertir la cadena de fecha a objeto date
        search_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        print("❌ Formato de fecha inválido. Usa el formato YYYY-MM-DD (ej: 2024-10-22)")
        return
    
    # Filtrar partidos para la fecha especificada
    games_on_date = df[df['GAME_DATE'] == search_date]
    
    if games_on_date.empty:
        print(f"\nℹ️ No hay partidos programados para el {search_date}.")
        return
    
    # Mostrar resultados
    print(f"\n📅 Partidos del {search_date} ({search_date.strftime('%A')}):")
    print("=" * 60)
    
    # Ordenar por hora si está disponible
    if 'GAME_TIME' in games_on_date.columns:
        games_on_date = games_on_date.sort_values('GAME_TIME')
    
    for _, game in games_on_date.iterrows():
        home_team = game.get('TEAM_ABBREVIATION_HOME', 'Equipo Local')
        away_team = game.get('TEAM_ABBREVIATION_AWAY', 'Equipo Visitante')
        
        # Obtener la hora del partido si está disponible
        game_time = game.get('GAME_TIME', 'Hora no disponible')
        if not pd.isna(game_time) and str(game_time) != 'NaT':
            if hasattr(game_time, 'strftime'):  # Si es un objeto de tiempo
                game_time_str = game_time.strftime('%H:%M')
            else:
                game_time_str = str(game_time)
        else:
            game_time_str = 'Hora no disponible'
        
        # Obtener información adicional
        season = game.get('SEASON', 'Temporada no disponible')
        season_type = game.get('SEASON_TYPE', 'Tipo de temporada no disponible')
        
        print(f"\n🏀 {away_team} @ {home_team}")
        print(f"   ⏰ {game_time_str}")
        print(f"   🏆 {season} - {season_type}")
        
        # Mostrar puntuación si el partido ya se jugó
        if 'PTS_HOME' in game and 'PTS_AWAY' in game and not pd.isna(game['PTS_HOME']) and not pd.isna(game['PTS_AWAY']):
            print(f"   📊 Resultado: {away_team} {int(game['PTS_AWAY'])} - {int(game['PTS_HOME'])} {home_team}")
            
            # Mostrar ganador
            if 'WL_HOME' in game and pd.notna(game['WL_HOME']):
                if game['WL_HOME'] == 'W':
                    print(f"   🏆 Ganador: {home_team}")
                else:
                    print(f"   🏆 Ganador: {away_team}")
    
    print(f"\n✅ Total de partidos encontrados: {len(games_on_date)}")

def main():
    """Función principal del script."""
    # Cargar datos
    df = load_games_data()
    if df is None or df.empty:
        return
    
    print("\n🔍 Búsqueda de partidos por fecha")
    print("=" * 30)
    print("Instrucciones:")
    print("- Introduce una fecha en formato YYYY-MM-DD (ej: 2024-12-25)")
    print("- Escribe 'salir' para terminar")
    
    while True:
        # Solicitar fecha al usuario
        date_input = input("\n📅 Ingresa una fecha (YYYY-MM-DD): ").strip()
        
        # Verificar si el usuario quiere salir
        if date_input.lower() in ('salir', 'exit', 'q', 'quit'):
            print("👋 ¡Hasta luego!")
            break
        
        # Buscar partidos para la fecha especificada
        search_games_by_date(df, date_input)

if __name__ == "__main__":
    main()
