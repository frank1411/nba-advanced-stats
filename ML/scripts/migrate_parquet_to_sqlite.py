#!/usr/bin/env python3
"""
Script de migración de archivos .parquet a SQLite.

Lee los archivos parquet existentes y los inserta en la base de datos SQLite.
No borra los archivos originales (se mantienen como backup).

Uso:
    python ML/scripts/migrate_parquet_to_sqlite.py
"""
import sys
import logging
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en el path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ML.db import NBADatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Ejecuta la migración de parquet a SQLite."""
    raw_parquet = PROJECT_ROOT / "ML" / "data" / "raw" / "nba_games_raw.parquet"
    final_parquet = PROJECT_ROOT / "ML" / "data" / "processed" / "nba_games_final.parquet"

    logger.info("=" * 60)
    logger.info("Migración de Parquet a SQLite")
    logger.info("=" * 60)

    # Verificar que existen los archivos
    if not raw_parquet.exists():
        logger.error(f"No se encontró: {raw_parquet}")
        sys.exit(1)
    if not final_parquet.exists():
        logger.error(f"No se encontró: {final_parquet}")
        sys.exit(1)

    with NBADatabase() as db:
        # Crear tablas
        logger.info("Creando tablas e índices...")
        db.create_tables()

        # Migrar raw
        logger.info(f"\nMigrando raw: {raw_parquet}")
        raw_count = db.migrate_parquet_to_sqlite(
            str(raw_parquet), table="raw", if_exists="replace"
        )
        logger.info(f"✅ Raw migrado: {raw_count} filas")

        # Migrar final
        logger.info(f"\nMigrando final: {final_parquet}")
        final_count = db.migrate_parquet_to_sqlite(
            str(final_parquet), table="final", if_exists="replace"
        )
        logger.info(f"✅ Final migrado: {final_count} filas")

        # Validación
        logger.info("\n" + "=" * 60)
        logger.info("Validación")
        logger.info("=" * 60)

        db_raw_count = db.get_game_count("raw")
        db_final_count = db.get_game_count("final")

        logger.info(f"Raw - Parquet filas: {raw_count}, DB filas: {db_raw_count}")
        logger.info(f"Final - Parquet filas: {final_count}, DB filas: {db_final_count}")

        assert raw_count == db_raw_count, (
            f"Mismatch en raw: parquet={raw_count}, db={db_raw_count}"
        )
        assert final_count == db_final_count, (
            f"Mismatch en final: parquet={final_count}, db={db_final_count}"
        )

        # Verificar que las consultas funcionan
        sample = db.query("SELECT * FROM nba_games_final ORDER BY GAME_DATE LIMIT 5")
        logger.info(f"\nSample query: {len(sample)} filas, {len(sample.columns)} columnas")
        logger.info(f"Columnas: {list(sample.columns[:10])}...")

        seasons = db.get_seasons()
        logger.info(f"Temporadas disponibles: {seasons}")

        # Tabla info
        raw_info = db.table_info("raw")
        final_info = db.table_info("final")
        logger.info(f"\nTabla raw: {raw_info['row_count']} filas, {raw_info['column_count']} columnas")
        logger.info(f"Tabla final: {final_info['row_count']} filas, {final_info['column_count']} columnas")

        logger.info("\n" + "=" * 60)
        logger.info("✅ Migración completada exitosamente")
        logger.info("=" * 60)
        logger.info(f"Base de datos: {db.db_path}")
        logger.info("Los archivos .parquet originales NO han sido eliminados")


if __name__ == "__main__":
    main()
