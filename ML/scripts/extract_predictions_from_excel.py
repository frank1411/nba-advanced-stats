#!/usr/bin/env python3
"""
Extrae predicciones del modelo desde el Excel 'Apuestas - NBA 2025 - 2026.xlsx'
(pestaña 'Jornada') y las inserta en la BD SQLite.

Columnas del Excel:
  D: Pro (pronóstico visitante)
  H: Pro2 (pronóstico local)
  K: Tot (pronóstico total)

Match con la BD por fecha + equipos.

Uso:
    python ML/scripts/extract_predictions_from_excel.py
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from ML.db import NBADatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Mapeo de nombres de equipos en el Excel a abreviaturas NBA
TEAM_EXCEL_TO_ABBREV = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'LA Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS',
}


def main():
    """Extrae predicciones del Excel y las inserta en la BD."""
    logger.info("=" * 60)
    logger.info("Extrayendo predicciones del modelo desde Excel")
    logger.info("=" * 60)

    excel_path = PROJECT_ROOT / "data" / "Apuestas - NBA 2025 - 2026.xlsx"
    if not excel_path.exists():
        logger.error(f"No se encontró: {excel_path}")
        sys.exit(1)

    # Leer la pestaña Jornada
    xls = pd.ExcelFile(str(excel_path))
    df_xls = pd.read_excel(xls, sheet_name='Jornada')
    logger.info(f"Leída hoja 'Jornada': {len(df_xls)} filas")

    # Filtrar solo filas con fecha válida y datos de predicción
    df_xls = df_xls.dropna(subset=['Fecha', 'Partido', 'Pro'])
    df_xls['Fecha'] = pd.to_datetime(df_xls['Fecha'])

    # Filtrar por rango de fechas
    df_xls = df_xls[
        (df_xls['Fecha'] >= '2025-10-21') &
        (df_xls['Fecha'] <= '2026-05-30')
    ]

    logger.info(f"Partidos con predicción en el rango: {len(df_xls)}")

    # Parsear partidos: "Houston Rockets @ Oklahoma City Thunder"
    records = []
    unmatched_parties = []

    for _, row in df_xls.iterrows():
        date = row['Fecha'].strftime('%Y-%m-%d')
        partido = row['Partido']

        if '@' not in partido:
            unmatched_parties.append(partido)
            continue

        parts = partido.split(' @ ')
        away_full = parts[0].strip()
        home_full = parts[1].strip()

        away_abbr = TEAM_EXCEL_TO_ABBREV.get(away_full, away_full)
        home_abbr = TEAM_EXCEL_TO_ABBREV.get(home_full, home_full)

        pred_away = row.get('Pro', None)
        pred_home = row.get('Pro2', None)
        pred_total = row.get('Tot', None)

        # Convertir a numérico
        try:
            pred_away = float(pred_away) if pd.notna(pred_away) else None
        except (ValueError, TypeError):
            pred_away = None
        try:
            pred_home = float(pred_home) if pd.notna(pred_home) else None
        except (ValueError, TypeError):
            pred_home = None
        try:
            pred_total = float(pred_total) if pd.notna(pred_total) else None
        except (ValueError, TypeError):
            pred_total = None

        if pred_away is None or pred_home is None:
            continue

        # Si no hay total, calcularlo
        if pred_total is None:
            pred_total = pred_away + pred_home

        records.append({
            'game_date': date,
            'away_team': away_abbr,
            'home_team': home_abbr,
            'pred_away': pred_away,
            'pred_home': pred_home,
            'pred_total': pred_total,
        })

    logger.info(f"Registros parseados: {len(records)}")
    if unmatched_parties:
        logger.warning(f"Partidos sin parsear ({len(unmatched_parties)}): {unmatched_parties[:5]}")

    # Cargar datos de la BD para hacer match
    with NBADatabase() as db:
        df = db.get_final_games()

    # Match por fecha + equipos
    matched = 0
    unmatched = 0

    for rec in records:
        date = rec['game_date']
        away = rec['away_team']
        home = rec['home_team']

        mask = (
            (df['GAME_DATE'] == pd.to_datetime(date)) &
            (df['TEAM_ABBREVIATION_AWAY'] == away) &
            (df['TEAM_ABBREVIATION_HOME'] == home)
        )

        match_idx = df[mask].index
        if len(match_idx) == 0:
            unmatched += 1
            if unmatched <= 10:
                logger.warning(f"No encontrado: {away} @ {home} en {date}")
            continue

        idx = match_idx[0]
        df.loc[idx, 'PREDICTION_OU_AWAY'] = rec['pred_away']
        df.loc[idx, 'PREDICTION_OU_HOME'] = rec['pred_home']
        df.loc[idx, 'PREDICTION_OU_TOTAL'] = rec['pred_total']
        matched += 1

    logger.info(f"\nResultado: {matched} partidos con predicción, {unmatched} no encontrados")

    # Calcular RESULT_OU_* y CONFIDENCE_OU_* para los que tienen línea + predicción
    valid_mask = df['LINE_OU_TOTAL'].notna() & df['PREDICTION_OU_TOTAL'].notna()
    if valid_mask.sum() > 0:
        import numpy as np
        logger.info(f"Calculando RESULT_OU_* y CONFIDENCE_OU_* para {valid_mask.sum()} partidos...")

        for target in ["TOTAL", "HOME", "AWAY"]:
            line_col = f"LINE_OU_{target}"
            pred_col = f"PREDICTION_OU_{target}"

            if target == "TOTAL":
                actual = df["PTS_HOME"] + df["PTS_AWAY"]
            elif target == "HOME":
                actual = df["PTS_HOME"]
            else:
                actual = df["PTS_AWAY"]

            valid = valid_mask & df[line_col].notna() & df[pred_col].notna()
            if valid.sum() == 0:
                continue

            predicted_over = df.loc[valid, pred_col] > df.loc[valid, line_col]
            actual_over = actual.loc[valid] > df.loc[valid, line_col]

            df.loc[valid, f"RESULT_OU_{target}"] = np.where(
                predicted_over == actual_over, "Hit", "Miss"
            )

            margin = (df.loc[valid, pred_col] - df.loc[valid, line_col]).abs()
            k = 0.5
            x0 = 5
            df.loc[valid, f"CONFIDENCE_OU_{target}"] = 1 / (1 + np.exp(-k * (margin - x0)))

    # Guardar en BD
    logger.info("Guardando en SQLite...")
    with NBADatabase() as db:
        db.insert_final_games(df, if_exists="replace")

    # Estadísticas finales
    pred_count = df['PREDICTION_OU_TOTAL'].notna().sum()
    line_count = df['LINE_OU_TOTAL'].notna().sum()
    both_count = valid_mask.sum()

    logger.info(f"\n📊 Resumen final:")
    logger.info(f"   Predicciones del modelo: {pred_count}")
    logger.info(f"   Líneas de casas: {line_count}")
    logger.info(f"   Con ambos (línea + predicción): {both_count}")

    if both_count > 0:
        for target in ["TOTAL", "HOME", "AWAY"]:
            col = f"RESULT_OU_{target}"
            hits = (df[col] == "Hit").sum()
            total = df[col].notna().sum()
            if total > 0:
                logger.info(f"   OU_{target}: {hits}/{total} ({hits/total*100:.1f}% hits)")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Predicciones del modelo insertadas en la BD")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
