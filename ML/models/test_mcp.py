#!/usr/bin/env python3
"""
Script de prueba para verificar el acceso a la función mcp0_nba_get_games.
"""
import json
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Intentar llamar a la función mcp0_nba_get_games
try:
    print("🔍 Intentando llamar a mcp0_nba_get_games...")
    result = mcp0_nba_get_games({
        'start_date': '2025-10-21',
        'end_date': '2025-10-28',
        'per_page': 5
    })
    print("✅ Éxito al llamar a la función")
    print(f"📊 Resultado: {json.dumps(result, indent=2)[:200]}...")
except NameError as e:
    print(f"❌ Error: {e}")
    print("La función mcp0_nba_get_games no está disponible globalmente.")
    
# Intentar listar las variables globales disponibles
try:
    print("\n🔍 Variables globales disponibles:")
    import builtins
    print("Funciones incorporadas:", [f for f in dir(builtins) if f.startswith('mcp')])
    print("Variables globales:", [k for k in globals().keys() if k.startswith('mcp')])
    print("Variables locales:", [k for k in locals().keys() if k.startswith('mcp')])
except Exception as e:
    print(f"Error al listar variables globales: {e}")
