#!/usr/bin/env python3
"""
Script para verificar los próximos partidos de la NBA.
Utiliza la clase NBADataFetcher del módulo de preprocesamiento.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Añadir el directorio raíz al path para poder importar los módulos
sys.path.append(str(Path(__file__).parent.parent))

# Importar la clase NBADataFetcher usando importlib para manejar el nombre con número
import importlib.util
import sys

# Ruta al módulo
module_path = str(Path(__file__).parent.parent / 'preprocessing' / '01_fetch_nba_data.py')
module_name = 'fetch_nba_data'

# Cargar el módulo dinámicamente
try:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    fetch_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = fetch_module
    spec.loader.exec_module(fetch_module)
    NBADataFetcher = fetch_module.NBADataFetcher
except Exception as e:
    print(f"Error al importar NBADataFetcher: {e}")
    sys.exit(1)

def get_upcoming_games(days_ahead: int = 7):
    """Obtiene los próximos partidos de la NBA.
    
    Args:
        days_ahead: Número de días hacia adelante para buscar partidos
    """
    print("🏀 Obteniendo próximos partidos de la NBA...")
    
    # Crear instancia del fetcher
    fetcher = NBADataFetcher()
    
    # Usar la temporada 2024-25 para las pruebas
    season = "2024-25"
    
    print(f"🔍 Buscando partidos para la temporada {season}...")
    
    # Obtener partidos de la temporada regular
    print("🔄 Obteniendo partidos de temporada regular...")
    regular_season = fetcher.get_season_games(season=season, season_type='Regular Season')
    
    # Obtener partidos de playoffs (si los hay)
    print("🔄 Obteniendo partidos de playoffs...")
    playoffs = fetcher.get_season_games(season=season, season_type='Playoffs')
    
    # Combinar datos
    all_games = []
    if regular_season is not None and not regular_season.empty:
        all_games.append(regular_season)
    if playoffs is not None and not playoffs.empty:
        all_games.append(playoffs)
    
    if not all_games:
        print("❌ No se encontraron partidos para la temporada actual.")
        return
    
    # Combinar todos los partidos
    df = pd.concat(all_games, ignore_index=True)
    
    # Filtrar solo partidos futuros
    today_str = today.strftime('%Y-%m-%d')
    future_date = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    
    future_games = df[
        (df['GAME_DATE'] >= today_str) & 
        (df['GAME_DATE'] <= future_date)
    ]
    
    if future_games.empty:
        print(f"ℹ️ No hay partidos programados en los próximos {days_ahead} días.")
        return
    
    # Mostrar resultados
    print(f"\n📅 Próximos partidos (hasta {future_date}):")
    print("=" * 60)
    
    # Ordenar por fecha y hora
    future_games = future_games.sort_values(['GAME_DATE', 'GAME_TIME'])
    
    # Mostrar los partidos agrupados por fecha
    for date, games_on_date in future_games.groupby('GAME_DATE'):
        print(f"\n📅 {date}")
        print("-" * 60)
        
        for _, game in games_on_date.iterrows():
            home_team = game['TEAM_ABBREVIATION_HOME']
            away_team = game['TEAM_ABBREVIATION_AWAY']
            game_time = game['GAME_TIME']
            arena = game.get('ARENA_NAME', 'Por determinar')
            
            print(f"🏀 {home_team} vs {away_team}")
            print(f"   ⏰ {game_time} | 🏟️  {arena}")
            
            # Mostrar información adicional si está disponible
            if 'SEASON_TYPE' in game:
                print(f"   🏆 {game['SEASON_TYPE']}")
            
            print()  # Línea en blanco entre partidos
    
    print(f"\n✅ Total de partidos encontrados: {len(future_games)}")

if __name__ == "__main__":
    get_upcoming_games(days_ahead=7)
