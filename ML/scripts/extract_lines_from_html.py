#!/usr/bin/env python3
"""
Extrae líneas reales de casas de apuestas desde los archivos HTML de recomendaciones
y las inserta en la BD SQLite en las columnas LINE_OU_*.

Formato HTML esperado:
    <td class='team-cell'>Washington Wizards @ Miami Heat</td>
    <td class='casa-cell'>115.0 / 133.0 - <span class='tot-bold'>247.5</span></td>

Uso:
    python ML/scripts/extract_lines_from_html.py
"""
import sys
import re
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from bs4 import BeautifulSoup
from ML.db import NBADatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Mapeo de nombres completos de equipos a abreviaturas NBA
TEAM_FULL_TO_ABBREV = {
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


def parse_html_file(filepath: str) -> list:
    """
    Parsea un archivo HTML de recomendaciones y extrae las líneas.

    Returns:
        Lista de dicts con: date, away_team, home_team, line_away, line_home, line_total
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    rows = soup.select('tbody tr')

    # Extraer fecha del nombre del archivo
    basename = Path(filepath).stem  # recommendations_20260404
    date_str = basename.split('_')[-1]  # 20260404
    date_obj = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')

    records = []
    for row in rows:
        # Equipos: "Washington Wizards @ Miami Heat"
        team_cell = row.select_one('.team-cell')
        if not team_cell:
            continue
        team_text = team_cell.get_text(strip=True)
        parts = team_text.split(' @ ')
        if len(parts) != 2:
            continue
        away_full, home_full = parts[0].strip(), parts[1].strip()

        # Modelo (predicción): "111 / 115 - 226"
        model_cell = row.select_one('.model-cell')
        pred_away = pred_home = pred_total = None
        if model_cell:
            model_text = model_cell.get_text(strip=True)
            match_model = re.match(
                r'([\d.]+)\s*/\s*([\d.]+)\s*-\s*([\d.]+)',
                model_text
            )
            if match_model:
                pred_away = float(match_model.group(1))
                pred_home = float(match_model.group(2))
                pred_total = float(match_model.group(3))

        # Líneas de la casa: "115.0 / 133.0 - 247.5"
        casa_cell = row.select_one('.casa-cell')
        if not casa_cell:
            continue
        casa_text = casa_cell.get_text(strip=True)

        # Parsear: "away_line / home_line - total"
        match = re.match(
            r'([\d.]+)\s*/\s*([\d.]+)\s*-\s*([\d.]+)',
            casa_text
        )
        if not match:
            continue

        line_away = float(match.group(1))
        line_home = float(match.group(2))
        line_total = float(match.group(3))

        # Mapear nombres completos a abreviaturas
        away_abbr = TEAM_FULL_TO_ABBREV.get(away_full, away_full)
        home_abbr = TEAM_FULL_TO_ABBREV.get(home_full, home_full)

        records.append({
            'game_date': date_obj,
            'away_team': away_abbr,
            'home_team': home_abbr,
            'line_away': line_away,
            'line_home': line_home,
            'line_total': line_total,
            'pred_away': pred_away,
            'pred_home': pred_home,
            'pred_total': pred_total,
        })

    return records


def main():
    """Extrae líneas de HTML y las inserta en la BD."""
    logger.info("=" * 60)
    logger.info("Extrayendo líneas reales de archivos HTML")
    logger.info("=" * 60)

    # Encontrar todos los archivos HTML
    pred_dir = PROJECT_ROOT / "ML" / "data" / "predictions"
    html_files = sorted(pred_dir.glob("recommendations_*.html"))

    if not html_files:
        logger.error("No se encontraron archivos recommendations_*.html")
        sys.exit(1)

    logger.info(f"Encontrados {len(html_files)} archivos HTML")

    # Parsear todos los archivos
    all_records = []
    for f in html_files:
        try:
            records = parse_html_file(str(f))
            all_records.extend(records)
            if records:
                logger.info(f"  {f.name}: {len(records)} partidos")
        except Exception as e:
            logger.warning(f"Error parseando {f.name}: {e}")

    if not all_records:
        logger.error("No se extrajeron registros de los archivos HTML")
        sys.exit(1)

    lines_df = pd.DataFrame(all_records)
    logger.info(f"\nTotal de líneas extraídas: {len(lines_df)}")
    logger.info(f"Rango de fechas: {lines_df['game_date'].min()} a {lines_df['game_date'].max()}")

    # Cargar datos de la BD para hacer match
    with NBADatabase() as db:
        df = db.get_final_games()

    # Match por fecha + equipos
    matched = 0
    unmatched = 0
    with_pred = 0

    for _, row in lines_df.iterrows():
        date = row['game_date']
        away = row['away_team']
        home = row['home_team']

        # Buscar partido en la BD
        mask = (
            (df['GAME_DATE'] == pd.to_datetime(date)) &
            (df['TEAM_ABBREVIATION_AWAY'] == away) &
            (df['TEAM_ABBREVIATION_HOME'] == home)
        )

        match_idx = df[mask].index
        if len(match_idx) == 0:
            unmatched += 1
            logger.warning(f"No encontrado: {away} @ {home} en {date}")
            continue

        # Actualizar las columnas de línea
        idx = match_idx[0]
        df.loc[idx, 'LINE_OU_AWAY'] = row['line_away']
        df.loc[idx, 'LINE_OU_HOME'] = row['line_home']
        df.loc[idx, 'LINE_OU_TOTAL'] = row['line_total']

        # Actualizar predicciones del modelo (si están disponibles)
        if pd.notna(row.get('pred_away')):
            df.loc[idx, 'PREDICTION_OU_AWAY'] = row['pred_away']
            df.loc[idx, 'PREDICTION_OU_HOME'] = row['pred_home']
            df.loc[idx, 'PREDICTION_OU_TOTAL'] = row['pred_total']
            with_pred += 1

        matched += 1

    logger.info(f"\nResultado: {matched} partidos actualizados ({with_pred} con predicción), {unmatched} no encontrados")

    # No calcular predicciones desde puntos reales — usar las del HTML si existen
    has_pred = df['PREDICTION_OU_TOTAL'].notna()
    if has_pred.sum() > 0:
        logger.info(f"Partidos con predicción del modelo: {has_pred.sum()}")

    # Calcular RESULT_OU_* para los partidos que ahora tienen líneas Y predicciones
    valid_mask = df['LINE_OU_TOTAL'].notna() & df['PREDICTION_OU_TOTAL'].notna()
    if valid_mask.sum() > 0:
        logger.info(f"Calculando RESULT_OU_* y CONFIDENCE_OU_* para {valid_mask.sum()} partidos con líneas + predicción...")
        import numpy as np

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

            # Solo calcular donde hay línea Y predicción
            valid = valid_mask & df[line_col].notna() & df[pred_col].notna()
            if valid.sum() == 0:
                continue

            predicted_over = df.loc[valid, pred_col] > df.loc[valid, line_col]
            actual_over = actual.loc[valid] > df.loc[valid, line_col]

            df.loc[valid, f"RESULT_OU_{target}"] = np.where(
                predicted_over == actual_over, "Hit", "Miss"
            )

            # Confianza
            margin = (df.loc[valid, pred_col] - df.loc[valid, line_col]).abs()
            k = 0.5
            x0 = 5
            df.loc[valid, f"CONFIDENCE_OU_{target}"] = 1 / (1 + np.exp(-k * (margin - x0)))

    # Guardar en BD
    logger.info("Guardando en SQLite...")
    with NBADatabase() as db:
        db.insert_final_games(df, if_exists="replace")
        final_count = db.get_game_count("final")

    logger.info(f"✅ {final_count} partidos guardados en SQLite")

    # Estadísticas de líneas y predicciones
    lines_count = df['LINE_OU_TOTAL'].notna().sum()
    pred_count = df['PREDICTION_OU_TOTAL'].notna().sum()
    result_count = valid_mask.sum()
    logger.info(f"\n📊 Líneas reales: {lines_count} partidos")
    logger.info(f"   Predicciones del modelo: {pred_count} partidos")
    logger.info(f"   Con resultado evaluado: {result_count} partidos")

    if lines_count > 0:
        # Hit rates
        for target in ["TOTAL", "HOME", "AWAY"]:
            col = f"RESULT_OU_{target}"
            hits = (df[col] == "Hit").sum()
            total = df[col].notna().sum()
            if total > 0:
                logger.info(f"  OU_{target}: {hits}/{total} ({hits/total*100:.1f}% hits)")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Extracción de líneas completada")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
