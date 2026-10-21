#!/usr/bin/env python3
"""
Análisis de umbrales de confianza para predicciones Over/Under.

Mide en qué márgenes de diferencia entre pronóstico y línea de apuesta
el modelo es más preciso, identificando los "puntos dulces" para apostar.

Ejemplo: cuando el pronóstico está 5+ pts por encima de la línea,
¿cuál es el % de aciertos?

Uso:
    python ML/analysis/threshold_analysis.py
    python ML/analysis/threshold_analysis.py --target total  # solo OU_TOTAL
    python ML/analysis/threshold_analysis.py --show-margins  # mostrar márgenes detallados
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Asegurar imports relativos desde el paquete ML
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from ML.db import NBADatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Umbrales de margen a analizar (diferencia entre predicción y línea)
DEFAULT_THRESHOLDS = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]


def compute_ou_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los resultados Hit/Miss para OU_TOTAL, OU_HOME y OU_AWAY
    basándose en los datos reales vs líneas de apuestas.

    Solo procesa filas que tengan línea, predicción y resultado real.
    """
    results = []

    for target in ["TOTAL", "HOME", "AWAY"]:
        line_col = f"LINE_OU_{target}"
        pred_col = f"PREDICTION_OU_{target}"
        conf_col = f"CONFIDENCE_OU_{target}"

        if line_col not in df.columns or pred_col not in df.columns:
            continue

        # Determinar columna de puntos real según target
        if target == "TOTAL":
            actual_col_pts = "PTS_HOME"
            actual_col_pts2 = "PTS_AWAY"
            df_temp = df.dropna(subset=[line_col, pred_col, actual_col_pts, actual_col_pts2])
            if df_temp.empty:
                continue
            actual = df_temp[actual_col_pts] + df_temp[actual_col_pts2]
        elif target == "HOME":
            actual_col_pts = "PTS_HOME"
            df_temp = df.dropna(subset=[line_col, pred_col, actual_col_pts])
            if df_temp.empty:
                continue
            actual = df_temp[actual_col_pts]
        else:  # AWAY
            actual_col_pts = "PTS_AWAY"
            df_temp = df.dropna(subset=[line_col, pred_col, actual_col_pts])
            if df_temp.empty:
                continue
            actual = df_temp[actual_col_pts]

        lines = df_temp[line_col].values
        predictions = df_temp[pred_col].values
        lines_actual = actual.values
        game_ids = df_temp["GAME_ID"].values if "GAME_ID" in df_temp.columns else range(len(df_temp))

        # Determinar si fue Over o Miss
        over_hit = actual > lines  # Over acertado
        under_hit = actual < lines  # Under acertado

        for i in range(len(df_temp)):
            margin = predictions[i] - lines[i]
            actual_over = lines_actual[i] > lines[i]

            # El modelo predijo Over si prediction > line
            pred_over = predictions[i] > lines[i]
            pred_under = not pred_over

            # Resultado: Hit si la dirección del pronóstico coincide con la realidad
            if (pred_over and actual_over) or (pred_under and not actual_over):
                result = "Hit"
            else:
                result = "Miss"

            confidence = df_temp[conf_col].values[i] if conf_col in df_temp.columns else None

            results.append({
                "GAME_ID": game_ids[i],
                "TARGET": target,
                "LINE": lines[i],
                "PREDICTION": predictions[i],
                "ACTUAL": lines_actual[i],
                "MARGIN": margin,  # predicción - línea
                "ABS_MARGIN": abs(margin),
                "RESULT": result,
                "CONFIDENCE": confidence,
                "PREDICTED_OVER": pred_over,
                "ACTUAL_OVER": actual_over,
            })

    return pd.DataFrame(results)


def analyze_by_threshold(results: pd.DataFrame, target: str = None, thresholds: list = None) -> pd.DataFrame:
    """
    Analiza la precisión del modelo en distintos umbrales de margen absoluto.

    Para cada umbral T, muestra:
    - Cuántos partidos tienen |margen| >= T
    - % de aciertos en ese grupo
    - % de aciertos para Over y Under por separado

    Args:
        results: DataFrame con los resultados.
        target: Filtrar por 'TOTAL', 'HOME' o 'AWAY'. None = todos.
        thresholds: Lista de umbrales a analizar.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    data = results[results["TARGET"] == target] if target else results
    if data.empty:
        return pd.DataFrame()

    summary = []

    # Línea base: todos los partidos
    all_hits = (data["RESULT"] == "Hit").sum()
    all_total = len(data)
    summary.append({
        "threshold": 0,
        "n_games": all_total,
        "hit_rate": all_hits / all_total if all_total > 0 else 0,
        "over_hit_rate": None,
        "under_hit_rate": None,
        "avg_margin": data["ABS_MARGIN"].mean(),
        "avg_confidence": data["CONFIDENCE"].mean() if data["CONFIDENCE"].notna().any() else None,
    })

    for t in thresholds:
        subset = data[data["ABS_MARGIN"] >= t]
        if len(subset) == 0:
            continue

        hits = (subset["RESULT"] == "Hit").sum()
        over_hits = (subset[(subset["PREDICTED_OVER"]) & (subset["ACTUAL_OVER"])])
        over_total = subset[subset["PREDICTED_OVER"]]
        under_hits = (subset[(~subset["PREDICTED_OVER"]) & (~subset["ACTUAL_OVER"])])
        under_total = subset[~subset["PREDICTED_OVER"]]

        summary.append({
            "threshold": t,
            "n_games": len(subset),
            "hit_rate": hits / len(subset),
            "over_hit_rate": len(over_hits) / len(over_total) if len(over_total) > 0 else None,
            "under_hit_rate": len(under_hits) / len(under_total) if len(under_total) > 0 else None,
            "avg_margin": subset["ABS_MARGIN"].mean(),
            "avg_confidence": subset["CONFIDENCE"].mean() if subset["CONFIDENCE"].notna().any() else None,
        })

    return pd.DataFrame(summary)


def analyze_by_confidence(results: pd.DataFrame, target: str = None) -> pd.DataFrame:
    """
    Analiza la precisión agrupando por nivel de confianza.

    Crea bins de confianza (0-0.55, 0.55-0.65, 0.65-0.75, 0.75-0.85, 0.85-1.0)
    y mide el hit rate en cada grupo.

    Args:
        results: DataFrame con los resultados.
        target: Filtrar por 'TOTAL', 'HOME' o 'AWAY'. None = todos.
    """
    data = results[results["TARGET"] == target] if target else results
    data = data.dropna(subset=["CONFIDENCE"])
    if data.empty:
        return pd.DataFrame()

    bins = [0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]

    data = data.copy()
    data["CONFIDENCE_BIN"] = pd.cut(data["CONFIDENCE"], bins=bins, labels=labels, include_lowest=True)

    summary = []
    for bin_label in labels:
        subset = data[data["CONFIDENCE_BIN"] == bin_label]
        if len(subset) == 0:
            continue
        hits = (subset["RESULT"] == "Hit").sum()
        summary.append({
            "confidence_range": bin_label,
            "n_games": len(subset),
            "hit_rate": hits / len(subset),
            "avg_confidence": subset["CONFIDENCE"].mean(),
        })

    return pd.DataFrame(summary)


def find_sweet_spots(results: pd.DataFrame, min_hit_rate: float = 0.60, min_games: int = 30) -> pd.DataFrame:
    """
    Identifica los "puntos dulces" donde el modelo es más rentable.

    Busca combinaciones de:
    - Target (TOTAL, HOME, AWAY)
    - Margen absoluto >= X
    - Predicción (Over vs Under)
    - Nivel de confianza

    Que cumplan:
    - Hit rate >= min_hit_rate
    - Al menos min_games partidos
    """
    sweet_spots = []

    for target in ["TOTAL", "HOME", "AWAY"]:
        target_data = results[results["TARGET"] == target]
        if target_data.empty:
            continue

        for direction, pred_col in [("Over", True), ("Under", False)]:
            dir_data = target_data[target_data["PREDICTED_OVER"] == pred_col]
            if len(dir_data) == 0:
                continue

            for t in DEFAULT_THRESHOLDS:
                subset = dir_data[dir_data["ABS_MARGIN"] >= t]
                if len(subset) < min_games:
                    continue

                hits = (subset["RESULT"] == "Hit").sum()
                hit_rate = hits / len(subset)

                if hit_rate >= min_hit_rate:
                    sweet_spots.append({
                        "target": target,
                        "direction": direction,
                        "min_margin": t,
                        "n_games": len(subset),
                        "hit_rate": hit_rate,
                        "avg_confidence": subset["CONFIDENCE"].mean() if subset["CONFIDENCE"].notna().any() else None,
                        "avg_actual": subset["ACTUAL"].mean(),
                        "avg_line": subset["LINE"].mean(),
                    })

    df = pd.DataFrame(sweet_spots)
    if not df.empty:
        df = df.sort_values("hit_rate", ascending=False)
    return df


def _print_threshold_table(threshold_summary: pd.DataFrame, target_label: str):
    """Imprime la tabla de umbrales para un target específico."""
    if threshold_summary.empty:
        return
    print(f"\n📏 OU_{target_label} — PRECISIÓN POR UMBRAL DE MARGEN (|predicción - línea| >= X)")
    print(f"{'Margen >=':>10} {'Partidos':>10} {'Hit Rate':>10} {'Over HR':>10} {'Under HR':>10} {'Avg Conf':>10}")
    print("-" * 62)
    for _, row in threshold_summary.iterrows():
        over_hr = f"{row['over_hit_rate']*100:>9.1f}%" if row["over_hit_rate"] is not None else "     N/A"
        under_hr = f"{row['under_hit_rate']*100:>9.1f}%" if row["under_hit_rate"] is not None else "     N/A"
        conf = f"{row['avg_confidence']:.3f}" if row["avg_confidence"] is not None else "     N/A"
        print(f"{row['threshold']:>10.1f} {row['n_games']:>10} {row['hit_rate']*100:>9.1f}% {over_hr:>10} {under_hr:>10} {conf:>10}")


def _print_confidence_table(confidence_summary: pd.DataFrame, target_label: str):
    """Imprime la tabla de confianza para un target específico."""
    if confidence_summary.empty:
        return
    print(f"\n🎲 OU_{target_label} — PRECISIÓN POR NIVEL DE CONFIANZA")
    print(f"{'Rango Conf.':>15} {'Partidos':>10} {'Hit Rate':>10} {'Avg Conf':>10}")
    print("-" * 47)
    for _, row in confidence_summary.iterrows():
        print(f"{row['confidence_range']:>15} {row['n_games']:>10} {row['hit_rate']*100:>9.1f}% {row['avg_confidence']:>10.3f}")


def print_report(results: pd.DataFrame, sweet_spots: pd.DataFrame, thresholds: list = None):
    """Imprime un reporte completo del análisis separado por target."""
    print("\n" + "=" * 70)
    print("📊 ANÁLISIS DE UMBRALES DE CONFIANZA — Over/Under")
    print("=" * 70)

    # Reporte general por target
    for target in ["TOTAL", "HOME", "AWAY"]:
        t_data = results[results["TARGET"] == target]
        if t_data.empty:
            continue

        hits = (t_data["RESULT"] == "Hit").sum()
        total = len(t_data)
        hit_rate = hits / total * 100 if total > 0 else 0
        over_data = t_data[t_data["PREDICTED_OVER"]]
        over_hits = (over_data[over_data["ACTUAL_OVER"]])
        over_rate = len(over_hits) / len(over_data) * 100 if len(over_data) > 0 else 0
        under_data = t_data[~t_data["PREDICTED_OVER"]]
        under_hits = (under_data[~under_data["ACTUAL_OVER"]])
        under_rate = len(under_hits) / len(under_data) * 100 if len(under_data) > 0 else 0

        print(f"\n🎯 OU_{target}:")
        print(f"   Partidos con datos: {total}")
        print(f"   Hit rate general: {hit_rate:.1f}% ({hits}/{total})")
        print(f"   Over hit rate: {over_rate:.1f}% ({len(over_hits)}/{len(over_data)})")
        print(f"   Under hit rate: {under_rate:.1f}% ({len(under_hits)}/{len(under_data)})")

        # Tabla de umbrales para este target
        t_summary = analyze_by_threshold(results, target=target, thresholds=thresholds)
        _print_threshold_table(t_summary, target)

        # Tabla de confianza para este target
        c_summary = analyze_by_confidence(results, target=target)
        _print_confidence_table(c_summary, target)

    # Puntos dulces (todos los targets juntos, ya está separado)
    if not sweet_spots.empty:
        print(f"\n{'='*70}")
        print("💎 PUNTOS DULCES (hit rate >= 60%, mínimo 30 partidos)")
        print(f"{'='*70}")
        for _, row in sweet_spots.head(20).iterrows():
            conf = f"{row['avg_confidence']:.3f}" if row["avg_confidence"] is not None else "N/A"
            print(f"  {row['target']:>5} {row['direction']:>5} margen≥{row['min_margin']:<3} → "
                  f"{row['hit_rate']*100:>5.1f}% ({row['n_games']} partidos, conf={conf})")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Análisis de umbrales de confianza OU")
    parser.add_argument("--target", type=str, choices=["total", "home", "away", "all"],
                        default="all", help="Analizar solo un target específico")
    parser.add_argument("--min-hit-rate", type=float, default=0.60,
                        help="Hit rate mínimo para considerar punto dulce (default: 0.60)")
    parser.add_argument("--min-games", type=int, default=30,
                        help="Mínimo de partidos para punto dulce (default: 30)")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Umbrales separados por coma (default: 0.5,1,2,3,4,5,6,7,8,10,12,15)")
    parser.add_argument("--save-report", type=str, default=None,
                        help="Guardar reporte en archivo CSV")
    args = parser.parse_args()

    thresholds = None
    if args.thresholds:
        thresholds = [float(x) for x in args.thresholds.split(",")]

    # Cargar datos
    logger.info("Cargando datos desde SQLite...")
    with NBADatabase() as db:
        df = db.get_final_games()

    if df.empty:
        logger.error("No hay datos en la base de datos")
        sys.exit(1)

    logger.info(f"Datos cargados: {len(df)} partidos")

    # Verificar que hay datos de betting
    betting_cols = ["LINE_OU_TOTAL", "PREDICTION_OU_TOTAL"]
    has_data = False
    for col in betting_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            if non_null > 10:
                has_data = True
                break

    if not has_data:
        logger.warning("No hay datos reales de betting (columnas vacías)")
        logger.info("Para analizar umbrales, necesitas:")
        logger.info("  1. Líneas de apuestas (LINE_OU_*)")
        logger.info("  2. Pronósticos del modelo (PREDICTION_OU_*)")
        logger.info("\nEjecutando análisis con datos simulados para demostración...")

        # Simular datos de betting para demostración
        logger.info("Generando datos simulados de betting para demostración...")
        np.random.seed(42)
        df["LINE_OU_TOTAL"] = df["PTS_HOME"] + df["PTS_AWAY"] + np.random.normal(0, 3, len(df))
        df["LINE_OU_HOME"] = df["PTS_HOME"] + np.random.normal(0, 3, len(df))
        df["LINE_OU_AWAY"] = df["PTS_AWAY"] + np.random.normal(0, 3, len(df))
        df["PREDICTION_OU_TOTAL"] = df["PTS_HOME"] + df["PTS_AWAY"] + np.random.normal(0, 2, len(df))
        df["PREDICTION_OU_HOME"] = df["PTS_HOME"] + np.random.normal(0, 2, len(df))
        df["PREDICTION_OU_AWAY"] = df["PTS_AWAY"] + np.random.normal(0, 2, len(df))
        df["CONFIDENCE_OU_TOTAL"] = np.random.uniform(0.5, 0.95, len(df))
        df["CONFIDENCE_OU_HOME"] = np.random.uniform(0.5, 0.95, len(df))
        df["CONFIDENCE_OU_AWAY"] = np.random.uniform(0.5, 0.95, len(df))
        logger.info("✅ Datos simulados generados")

    # Filtrar target si se especificó
    if args.target != "all":
        target_map = {"total": "TOTAL", "home": "HOME", "away": "AWAY"}
        targets = [target_map[args.target]]
    else:
        targets = ["TOTAL", "HOME", "AWAY"]

    # Computar resultados
    results = compute_ou_results(df)
    # Filtrar solo los targets seleccionados
    results = results[results["TARGET"].isin(targets)]

    if results.empty:
        logger.error("No hay resultados para analizar. Necesitas datos de línea + predicción.")
        sys.exit(1)

    logger.info(f"Resultados computados: {len(results)} registros")

    # Puntos dulces
    sweet_spots = find_sweet_spots(results, args.min_hit_rate, args.min_games)

    # Imprimir reporte (calcula umbrales y confianza internamente por target)
    print_report(results, sweet_spots, thresholds)

    # Guardar si se pidió
    if args.save_report:
        sweet_spots.to_csv(args.save_report, index=False)
        logger.info(f"Reporte guardado en {args.save_report}")


if __name__ == "__main__":
    main()
