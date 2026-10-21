#!/usr/bin/env python3
"""
Script principal para ejecutar el pipeline diario de datos de la NBA.

Flujo correcto:
  02 → Actualiza resultados reales de partidos jugados
  06 → Genera pronósticos para los próximos partidos
  08 → Obtiene líneas de la casa + recomendaciones (cruza con pronósticos de 06)
"""
import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def run_step(step_name, module_path):
    """Ejecuta un paso del pipeline importando dinámicamente el módulo."""
    try:
        logger.info(f"{'='*60}")
        logger.info(f"▶ Ejecutando: {step_name}")
        logger.info(f"{'='*60}")

        import importlib.util
        spec = importlib.util.spec_from_file_location(step_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

        logger.info(f"✅ {step_name} completado")
        return True
    except Exception as e:
        logger.error(f"❌ Error en {step_name}: {e}", exc_info=True)
        return False

def main():
    logger.info("=" * 80)
    logger.info("PIPELINE DIARIO NBA - PREDICCIONES")
    logger.info("=" * 80)

    steps = [
        ("02: Actualizar datos",        "ML/preprocessing/02_update_nba_data.py"),
        ("06: Generar pronósticos",     "ML/models/06_predict_upcoming_games.py"),
        ("08: Líneas + recomendaciones","ML/models/08_the_odds_api_recommendations.py"),
    ]

    for name, path in steps:
        success = run_step(name, path)
        if not success:
            logger.warning(f"⚠️  {name} falló, continuando con el siguiente paso...")

    logger.info("=" * 80)
    logger.info("PIPELINE DIARIO COMPLETADO")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
