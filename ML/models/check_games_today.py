#!/usr/bin/env python3
"""
Script para verificar los partidos de la NBA programados para hoy.
"""
import json
import subprocess
from datetime import datetime
import pytz

def get_todays_games():
    """Obtiene los partidos de la NBA programados para hoy."""
    # Obtener la fecha de hoy
    today = datetime.now(pytz.utc).date()
    date_str = today.strftime('%Y-%m-%d')
    
    print(f"🏀 Buscando partidos de la NBA para hoy ({date_str})...")
    
    try:
        # Usar la herramienta del MCP para obtener los partidos de hoy
        from mcp import mcp0_nba_get_games
        result = mcp0_nba_get_games({
            'start_date': date_str,
            'end_date': date_str,
            'per_page': 100
        })
        
        if not result or 'data' not in result or not result['data']:
            print("\nℹ️ No hay partidos programados para hoy.")
            return []
        
        return result['data']
        
    except Exception as e:
        print(f"\n❌ Error al obtener los partidos: {e}")
        return []

def main():
    """Función principal."""
    print("🏀 Partidos de la NBA para hoy")
    print("=" * 50)
    
    # Obtener partidos de hoy
    games = get_todays_games()
    
    if not games:
        return
    
    print(f"\n✅ Se encontraron {len(games)} partidos para hoy:")
    print("-" * 50)
    
    # Mostrar información de los partidos
    for game in games:
        try:
            home_team = game['home_team']['full_name']
            away_team = game['visitor_team']['full_name']
            status = game.get('status', 'Hora por confirmar')
            
            print(f"\n🏀 {away_team} @ {home_team}")
            print(f"🕒 {status}")
            print("-" * 50)
            
        except Exception as e:
            print(f"\n⚠️ Error al procesar un partido: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Ejecución interrumpida por el usuario")
