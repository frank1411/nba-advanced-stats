#!/usr/bin/env python3
"""
Script para agregar columnas de betting tracking a la tabla nba_games_final.

Las nuevas columnas permiten rastrear:
- Líneas de Over/Under de casas de apuestas (LINE_OU_*)
- Pronósticos del modelo para OU (PREDICTION_OU_*)
- Resultados de los pronósticos (RESULT_OU_*)
- Nivel de confianza de cada pronóstico (CONFIDENCE_OU_*)

Uso:
    python ML/scripts/add_betting_columns.py
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ML.db import NBADatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Columnas de betting a agregar
BETTING_COLUMNS = [
    # Líneas de casas de apuestas (Over/Under)
    {"name": "LINE_OU_TOTAL", "type": "REAL", "default": None},
    {"name": "LINE_OU_HOME", "type": "REAL", "default": None},
    {"name": "LINE_OU_AWAY", "type": "REAL", "default": None},

    # Pronósticos del modelo
    {"name": "PREDICTION_OU_TOTAL", "type": "REAL", "default": None},
    {"name": "PREDICTION_OU_HOME", "type": "REAL", "default": None},
    {"name": "PREDICTION_OU_AWAY", "type": "REAL", "default": None},

    # Resultados ('Hit' / 'Miss')
    {"name": "RESULT_OU_TOTAL", "type": "TEXT", "default": None},
    {"name": "RESULT_OU_HOME", "type": "TEXT", "default": None},
    {"name": "RESULT_OU_AWAY", "type": "TEXT", "default": None},

    # Confianza del modelo (0-1)
    {"name": "CONFIDENCE_OU_TOTAL", "type": "REAL", "default": None},
    {"name": "CONFIDENCE_OU_HOME", "type": "REAL", "default": None},
    {"name": "CONFIDENCE_OU_AWAY", "type": "REAL", "default": None},
]


def main():
    """Agrega las columnas de betting a la tabla nba_games_final."""
    logger.info("=" * 60)
    logger.info("Agregando columnas de Betting Tracking")
    logger.info("=" * 60)

    with NBADatabase() as db:
        # Verificar que la tabla existe
        tables = db.list_tables()
        if "nba_games_final" not in tables:
            logger.error("No se encontró la tabla nba_games_final")
            logger.info("Ejecuta primero: python ML/scripts/migrate_parquet_to_sqlite.py")
            sys.exit(1)

        # Contar filas actuales
        count = db.get_game_count("final")
        logger.info(f"Tabla nba_games_final: {count} filas")

        # Agregar columnas
        added = db.add_columns("final", BETTING_COLUMNS)

        if added > 0:
            logger.info(f"✅ {added} columnas nuevas agregadas")
        else:
            logger.info("Todas las columnas ya existían. No se hicieron cambios.")

        # Verificar columnas finales
        info = db.table_info("final")
        logger.info(f"\nTabla nba_games_final actualizada:")
        logger.info(f"  Filas: {info['row_count']}")
        logger.info(f"  Columnas: {info['column_count']}")

        # Listar las columnas de betting
        betting_cols = [c for c in info["columns"] if "OU" in c[0] or "CONFIDENCE" in c[0]]
        logger.info(f"\nColumnas de betting:")
        for name, dtype in betting_cols:
            logger.info(f"  {name} ({dtype})")

        logger.info("\n" + "=" * 60)
        logger.info("✅ Columnas de betting agregadas exitosamente")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
