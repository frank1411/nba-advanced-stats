#!/usr/bin/env python3
"""
Script para obtener los partidos de la NBA programados para hoy.
"""
import json
from datetime import datetime, timezone

def main():
    """Función principal."""
    print("🏀 Partidos de la NBA para hoy")
    print("=" * 50)
    
    # Obtener la fecha de hoy
    today = datetime.now(timezone.utc).date()
    date_str = today.strftime('%Y-%m-%d')
    
    print(f"\n📅 Buscando partidos para hoy ({date_str})...")
    
    try:
        # Obtener los partidos de hoy usando la herramienta del MCP
        from mcp import mcp0_nba_get_games
        result = mcp0_nba_get_games({
            'start_date': date_str,
            'end_date': date_str,
            'per_page': 100
        })
        
        if not result or 'data' not in result or not result['data']:
            print("\nℹ️ No hay partidos programados para hoy.")
            return
        
        games = result['data']
        print(f"\n✅ Se encontraron {len(games)} partidos para hoy:")
        print("-" * 50)
        
        # Mostrar información de los partidos
        for game in games:
            home_team = game['home_team']['full_name']
            away_team = game['visitor_team']['full_name']
            game_time = game.get('datetime', 'Hora no disponible')
            
            # Formatear la hora si está disponible
            if game_time and isinstance(game_time, str) and 'T' in game_time:
                try:
                    game_dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
                    game_time = game_dt.strftime('%H:%M UTC')
                except (ValueError, TypeError):
                    pass
            
            print(f"\n🏀 {away_team} @ {home_team}")
            print(f"🕒 Hora: {game_time}")
            print("-" * 50)
            
    except Exception as e:
        print(f"\n❌ Error al obtener los partidos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
