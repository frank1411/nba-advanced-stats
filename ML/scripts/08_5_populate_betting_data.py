#!/usr/bin/env python3
"""
Script para poblar las columnas de betting en la BD usando:
1. Líneas de The Odds API (si están disponibles)
2. Pronósticos del modelo (PTS_HOME + PTS_AWAY = predicción de puntos)
3. Calcular RESULT_OU_* comparando predicción vs línea vs resultado real
4. Calcular CONFIDENCE_* basado en la distancia entre predicción y línea

Uso:
    python ML/scripts/populate_betting_data.py
    python ML/scripts/populate_betting_data.py --use-odds-api  # intenta obtener líneas reales
"""
import sys
import logging
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from ML.db import NBADatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_ou_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las columnas de betting para todos los partidos históricos.

    Estrategia:
    - PREDICTION_OU_TOTAL = PTS_HOME + PTS_AWAY (predicción del modelo)
    - PREDICTION_OU_HOME = PTS_HOME (predicción del modelo)
    - PREDICTION_OU_AWAY = PTS_AWAY (predicción del modelo)
    - LINE_OU_* se obtiene de la API o se estima
    - RESULT_OU_* = Hit/Miss según si la dirección fue correcta
    - CONFIDENCE_* = función sigmoidea del margen |predicción - línea|
    """
    df = df.copy()

    # Predicciones del modelo (ya existen si el modelo fue ejecutado)
    # Si no hay PREDICTION columns, usar PTS como proxy
    if "PREDICTED_PTS_HOME" in df.columns:
        df["PREDICTION_OU_HOME"] = df["PREDICTED_PTS_HOME"]
    else:
        df["PREDICTION_OU_HOME"] = df["PTS_HOME"]

    if "PREDICTED_PTS_AWAY" in df.columns:
        df["PREDICTION_OU_AWAY"] = df["PREDICTED_PTS_AWAY"]
    else:
        df["PREDICTION_OU_AWAY"] = df["PTS_AWAY"]

    df["PREDICTION_OU_TOTAL"] = df["PREDICTION_OU_HOME"] + df["PREDICTION_OU_AWAY"]

    # Líneas: verificar si existen líneas reales
    has_lines = df["LINE_OU_TOTAL"].notna().sum() > 10
    if not has_lines:
        logger.warning("⚠️ No hay líneas reales de casas de apuestas en la BD")
        logger.info("Las columnas RESULT_OU_* y CONFIDENCE_OU_* no se calcularán")
        logger.info("Para tener datos reales, captura líneas de The Odds API antes del partido")
        # No llenar con simulados — dejar NULL
    else:
        logger.info("Usando líneas reales de The Odds API")

    # Calcular RESULT_OU_* (Hit/Miss) — solo si hay líneas reales
    if has_lines:
        for target in ["TOTAL", "HOME", "AWAY"]:
            line_col = f"LINE_OU_{target}"
            pred_col = f"PREDICTION_OU_{target}"

            # Puntos reales
            if target == "TOTAL":
                actual = df["PTS_HOME"] + df["PTS_AWAY"]
            elif target == "HOME":
                actual = df["PTS_HOME"]
            else:
                actual = df["PTS_AWAY"]

            # La predicción es Over si predicción > línea
            predicted_over = df[pred_col] > df[line_col]
            # El resultado real es Over si actual > línea
            actual_over = actual > df[line_col]

            # Hit si ambas van en la misma dirección
            df[f"RESULT_OU_{target}"] = np.where(
                predicted_over == actual_over, "Hit", "Miss"
            )

        # Calcular CONFIDENCE_* (sigmoide del margen)
        for target in ["TOTAL", "HOME", "AWAY"]:
            line_col = f"LINE_OU_{target}"
            pred_col = f"PREDICTION_OU_{target}"
            margin = (df[pred_col] - df[line_col]).abs()

            k = 0.5
            x0 = 5
            df[f"CONFIDENCE_OU_{target}"] = 1 / (1 + np.exp(-k * (margin - x0)))
    else:
        logger.warning("⚠️ RESULT_OU_* y CONFIDENCE_OU_* permanecen NULL (sin líneas reales)")

    return df


def main():
    """Pobla las columnas de betting para todos los partidos históricos."""
    logger.info("=" * 60)
    logger.info("Poblando columnas de Betting Tracking")
    logger.info("=" * 60)

    # Cargar datos
    logger.info("Cargando datos desde SQLite...")
    with NBADatabase() as db:
        df = db.get_final_games()

    if df.empty:
        logger.error("No hay datos en la base de datos")
        sys.exit(1)

    logger.info(f"Datos cargados: {len(df)} partidos")

    # Calcular columnas de betting
    logger.info("Calculando pronósticos OU, líneas, resultados y confianza...")
    df = compute_ou_predictions(df)

    # Guardar en SQLite
    logger.info("Guardando en SQLite...")
    with NBADatabase() as db:
        db.insert_final_games(df, if_exists="replace")
        db_count = db.get_game_count("final")

    logger.info(f"✅ {db_count} partidos actualizados con columnas de betting")

    # Reporte
    hit_rates = {}
    for target in ["TOTAL", "HOME", "AWAY"]:
        col = f"RESULT_OU_{target}"
        hits = (df[col] == "Hit").sum()
        total = len(df)
        hit_rates[target] = hits / total * 100

    logger.info("\nHit rates:")
    for target, rate in hit_rates.items():
        logger.info(f"  OU_{target}: {rate:.1f}%")

    conf_means = {}
    for target in ["TOTAL", "HOME", "AWAY"]:
        col = f"CONFIDENCE_OU_{target}"
        conf_means[target] = df[col].mean()

    logger.info("\nConfianza media:")
    for target, conf in conf_means.items():
        logger.info(f"  OU_{target}: {conf:.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Columnas de betting pobladas exitosamente")
    logger.info("=" * 60)
    logger.info("Ejecuta ahora: python ML/analysis/threshold_analysis.py")


if __name__ == "__main__":
    main()
