import requests
import pandas as pd
import os
from datetime import datetime
import sys
from pathlib import Path

# Asegurar que el directorio raíz esté en el path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ML.db import NBADatabase

# Configuración
#API_KEY = '51ece9cbfb6b8d6c0e96264e09fc29a0'
API_KEY = '0cf108d7160e1d104f1580b009d0673e'
PREDICTIONS_FILE = f'ML/data/predictions/predictions_{datetime.now().strftime("%Y%m%d")}.csv'
OUTPUT_FILE = f'ML/data/predictions/recommendations_{datetime.now().strftime("%Y%m%d")}.md'


# Mapeo de nombres de equipos de la API a abreviaturas NBA
TEAM_API_TO_ABBREV = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL', 'LA Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
}


def save_lines_to_db(lines_records: list) -> int:
    """
    Guarda las líneas de casas de apuestas (LINE_OU_*) en la BD SQLite.

    Si un juego de HOY no existe en nba_games_final, crea un registro
    mínimo (placeholder) directamente en la tabla final con líneas y pronósticos.
    Los juegos de días futuros NO se guardan como placeholder.

    Args:
        lines_records: Lista de dicts con game_date, home_team, away_team,
                       line_home, line_away, line_total, pred_home, pred_away, pred_total

    Returns:
        Número de partidos actualizados/creados.
    """
    if not lines_records:
        print("⚠️ No hay líneas para guardar en la BD")
        return 0

    today = datetime.now().strftime('%Y-%m-%d')

    with NBADatabase() as db:
        df = db.get_final_games()

    updated = 0
    created = 0

    for rec in lines_records:
        date_str = rec['game_date']
        home = rec['home_team']
        away = rec['away_team']

        # Solo crear placeholder para juegos de HOY
        if date_str != today:
            continue

        mask = (
            (df['GAME_DATE'] == pd.to_datetime(date_str)) &
            (df['TEAM_ABBREVIATION_HOME'] == home) &
            (df['TEAM_ABBREVIATION_AWAY'] == away)
        )

        match_idx = df[mask].index

        if len(match_idx) == 0:
            # Juego de hoy: crear placeholder directamente en df final
            print(f"   📝 Creando placeholder para {away} @ {home} en {date_str}")
            game_id_str = f"placeholder_{date_str.replace('-', '')}_{home}_{away}"

            # Crear una fila con todas las columnas del df existente
            new_row = {col: None for col in df.columns}
            new_row['GAME_ID'] = game_id_str
            new_row['GAME_DATE'] = pd.to_datetime(date_str)
            new_row['TEAM_ABBREVIATION_HOME'] = home
            new_row['TEAM_ABBREVIATION_AWAY'] = away
            new_row['SEASON'] = 2025
            new_row['SEASON_STR'] = '2025-26'
            new_row['SEASON_TYPE'] = 'Playoffs'
            new_row['LINE_OU_HOME'] = rec['line_home']
            new_row['LINE_OU_AWAY'] = rec['line_away']
            new_row['LINE_OU_TOTAL'] = rec['line_total']

            if rec.get('pred_home') is not None:
                new_row['PREDICTION_OU_HOME'] = rec['pred_home']
                new_row['PREDICTION_OU_AWAY'] = rec['pred_away']
                new_row['PREDICTION_OU_TOTAL'] = rec['pred_total']

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            created += 1
        else:
            idx = match_idx[0]
            df.loc[idx, 'LINE_OU_HOME'] = rec['line_home']
            df.loc[idx, 'LINE_OU_AWAY'] = rec['line_away']
            df.loc[idx, 'LINE_OU_TOTAL'] = rec['line_total']

            if rec.get('pred_home') is not None:
                df.loc[idx, 'PREDICTION_OU_HOME'] = rec['pred_home']
                df.loc[idx, 'PREDICTION_OU_AWAY'] = rec['pred_away']
                df.loc[idx, 'PREDICTION_OU_TOTAL'] = rec['pred_total']

            updated += 1

    # Calcular RESULT_OU_* y CONFIDENCE_OU_* para partidos con línea + predicción + resultado real
    # Y RECOMMENDATION_OU_* para partidos con línea + predicción (sin resultado)
    import numpy as np
    valid_mask = df['LINE_OU_TOTAL'].notna() & df['PREDICTION_OU_TOTAL'].notna()
    has_result = valid_mask & df['PTS_HOME'].notna() & (df['PTS_HOME'] > 0)
    no_result = valid_mask & ~has_result

    # --- 1. RESULTADO (Hit/Miss) para partidos jugados ---
    if has_result.sum() > 0:
        print(f"   📊 Calculando resultados para {has_result.sum()} partidos jugados...")
        for target in ["TOTAL", "HOME", "AWAY"]:
            line_col = f"LINE_OU_{target}"
            pred_col = f"PREDICTION_OU_{target}"

            if target == "TOTAL":
                actual = df["PTS_HOME"] + df["PTS_AWAY"]
            elif target == "HOME":
                actual = df["PTS_HOME"]
            else:
                actual = df["PTS_AWAY"]

            valid = has_result & df[line_col].notna() & df[pred_col].notna()
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

    # --- 2. RECOMENDACIÓN (Over/Under) para partidos sin resultado ---
    if no_result.sum() > 0:
        print(f"   💡 Calculando recomendaciones para {no_result.sum()} próximos partidos...")
        for target in ["TOTAL", "HOME", "AWAY"]:
            line_col = f"LINE_OU_{target}"
            pred_col = f"PREDICTION_OU_{target}"
            rec_col = f"RECOMMENDATION_OU_{target}"

            valid_no = no_result & df[line_col].notna() & df[pred_col].notna()
            if valid_no.sum() == 0:
                continue

            predicted_over = df.loc[valid_no, pred_col] > df.loc[valid_no, line_col]
            df.loc[valid_no, rec_col] = np.where(predicted_over, "Over", "Under")

            # Guardar también la magnitud de la diferencia
            diff = (df.loc[valid_no, pred_col] - df.loc[valid_no, line_col]).abs()
            conf_col = f"CONFIDENCE_OU_{target}"
            k = 0.5
            x0 = 5
            df.loc[valid_no, conf_col] = 1 / (1 + np.exp(-k * (diff - x0)))

    # Resumen
    for target in ["TOTAL", "HOME", "AWAY"]:
        rec_col = f"RECOMMENDATION_OU_{target}"
        res_col = f"RESULT_OU_{target}"
        recs = df[rec_col].notna().sum()
        hits = (df[res_col] == "Hit").sum()
        total_res = df[res_col].notna().sum()
        if recs > 0:
            rec_summary = df[rec_col].value_counts().to_dict()
            print(f"   {rec_col}: {rec_summary}")
        if total_res > 0:
            print(f"   {res_col}: {hits}/{total_res} ({hits/total_res*100:.1f}% hits)")

    # Guardar en la BD
    with NBADatabase() as db:
        db.insert_final_games(df, if_exists="replace")

    print(f"✅ Líneas guardadas en BD: {updated} actualizados, {created} creados")
    return updated + created

def get_nba_odds():
    print("📡 Consultando The Odds API...")
    url = f'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
    params = {
        'api_key': API_KEY,
        'regions': 'us',
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'decimal',
        'dateFormat': 'iso'
    }
    #response = requests.get(url, params=params)
    response = requests.get(url, params=params, timeout=15)
    if response.status_code != 200:
        print(f"❌ Error API: {response.status_code}")
        return []
    return response.json()

def clean_team_name(name):
    # Diccionario de mapeo para asegurar compatibilidad perfecta
    mapping = {
        'LA Clippers': 'Los Angeles Clippers',
        'LA Lakers': 'Los Angeles Lakers',
        'Lakers': 'Los Angeles Lakers',
        'Clippers': 'Los Angeles Clippers',
        '76ers': 'Philadelphia 76ers',
        'Cavs': 'Cleveland Cavaliers',
        'Grizzlies': 'Memphis Grizzlies',
        'Timberwolves': 'Minnesota Timberwolves',
        'Trail Blazers': 'Portland Trail Blazers'
    }
    for short, full in mapping.items():
        if short in name: return full
    return name

def generate_report():
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"❌ No se encontró el archivo de predicciones: {PREDICTIONS_FILE}")
        return

    predictions = pd.read_csv(PREDICTIONS_FILE)
    api_data = get_nba_odds()

    # --- CONFIGURACIÓN DE UMBRALES ASIMÉTRICOS (FILTRO 3.1) ---
    # Mínimo 5 pts de diferencia para entrar en zona >52% efectividad histórica
    THR_HOME = 14.0       # Mercado Local: Umbral para Local (14 pts)
    THR_AWAY = 5.0        # Mercado Visitante: mínimo 5 pts (zona >57% hit rate)
    THR_TOTAL = 5.0       # Mercado Partido: mínimo 5 pts (zona >57% hit rate)

    report_md = f"# 🏀 NBA Smart Picks 3.1 (Precision Betting) - {datetime.now().strftime('%Y-%m-%d')}\n"
    report_md += f"*Filtros Asimétricos: L ±{THR_HOME} | V ±{THR_AWAY} | P ±{THR_TOTAL} | ⭐ Alta Estabilidad*\n\n"
    report_md += "| Encuentro (V - L) | Mercados O/U | Estabilidad | Ventaja (Pts) |\n"
    report_md += "| :--- | :--- | :---: | :---: |\n"

    rows_html = ""
    lines_records = []  # Para guardar en la BD

    for _, row in predictions.iterrows():
        home_pred, away_pred = row['TEAM_NAME_HOME'], row['TEAM_NAME_AWAY']
        pred_h, pred_v = round(row['PRED_PTS_HOME'], 1), round(row['PRED_PTS_AWAY'], 1)
        pred_tot = round(pred_h + pred_v, 1)

        # Obtener fecha del partido (si está disponible)
        game_date = row.get('GAME_DATE', None)
        if game_date is None or pd.isna(game_date):
            game_date = datetime.now().strftime('%Y-%m-%d')
        else:
            game_date = str(game_date).split('T')[0]  # Limpiar timestamp si existe

        # Obtener abreviaturas de equipos para match con BD
        home_abbr = TEAM_API_TO_ABBREV.get(clean_team_name(home_pred), clean_team_name(home_pred))
        away_abbr = TEAM_API_TO_ABBREV.get(clean_team_name(away_pred), clean_team_name(away_pred))

        h_clean = clean_team_name(home_pred)
        odds_match = next((g for g in api_data if h_clean in g['home_team'] or g['home_team'] in h_clean), None)

        if odds_match:
            try:
                books = {b['key']: b for b in odds_match['bookmakers']}
                book = books.get('fanduel', books.get('draftkings', odds_match['bookmakers'][0]))
                hnd_line, tot_line = "-", "-"
                for m in book['markets']:
                    if m['key'] == 'spreads': hnd_line = next(o['point'] for o in m['outcomes'] if o['name'] == odds_match['home_team'])
                    if m['key'] == 'totals': tot_line = m['outcomes'][0]['point']

                if hnd_line != "-" and tot_line != "-":
                    implied_h = round(((tot_line - hnd_line) / 2) * 2) / 2
                    implied_v = round(((tot_line + hnd_line) / 2) * 2) / 2

                    # Recolectar línea para guardar en la BD
                    lines_records.append({
                        'game_date': game_date,
                        'home_team': home_abbr,
                        'away_team': away_abbr,
                        'line_home': implied_h,
                        'line_away': implied_v,
                        'line_total': tot_line,
                        'pred_home': pred_h,
                        'pred_away': pred_v,
                        'pred_total': pred_tot,
                    })

                    green_picks = []
                    stability_tags = []

                    # 1. O/U Individual Local
                    diff_h = pred_h - implied_h
                    if abs(diff_h) >= THR_HOME:
                        label = "Over Local" if diff_h > 0 else "Under Local"
                        # Validar Estabilidad Local
                        is_stable = False
                        if (pred_h < 113 and diff_h > 0) or (pred_h >= 113 and diff_h < 0):
                            is_stable = True

                        icon = "⭐" if is_stable else ""
                        green_picks.append(f"🟢 **{label}** {icon} ({round(diff_h,1)})")
                        if is_stable: stability_tags.append("Local")

                    # 2. O/U Individual Visita
                    diff_v = pred_v - implied_v
                    if abs(diff_v) >= THR_AWAY:
                        label = "Over Visita" if diff_v > 0 else "Under Visita"
                        # Validar Estabilidad Visita
                        is_stable = False
                        if (pred_v < 110 and diff_v > 0) or (pred_v > 118 and diff_v < 0):
                            is_stable = True

                        icon = "⭐" if is_stable else ""
                        green_picks.append(f"🟢 **{label}** {icon} ({round(diff_v,1)})")
                        if is_stable: stability_tags.append("Visita")

                    # 3. O/U Partido
                    diff_tot = pred_tot - tot_line
                    if abs(diff_tot) >= THR_TOTAL:
                        label = "Over Partido" if diff_tot > 0 else "Under Partido"
                        # Validar Estabilidad Partido
                        is_stable = False
                        if (pred_tot < 229 and diff_tot > 0) or (232 <= pred_tot <= 238 and diff_tot < 0):
                            is_stable = True

                        icon = "⭐" if is_stable else ""
                        green_picks.append(f"🟢 **{label}** {icon} ({round(diff_tot,1)})")
                        if is_stable: stability_tags.append("Partido")

                    picks_md = "<br>".join(green_picks) if green_picks else ""
                    stab_md = " / ".join(stability_tags) if stability_tags else "-"
                    report_md += f"| **{away_pred} @ {home_pred}** | {picks_md} | {stab_md} | - |\n"

                    picks_html = " / ".join(green_picks).replace("**", "") if green_picks else ""
                    rows_html += f"""
                    <tr>
                        <td class='team-cell'>{away_pred} @ {home_pred}</td>
                        <td class='model-cell'>{pred_v} / {pred_h} - <span class='tot-bold'>{pred_tot}</span></td>
                        <td class='casa-cell'>{implied_v} / {implied_h} - <span class='tot-bold'>{tot_line}</span></td>
                        <td style='color: #059669; font-weight: bold;'>{picks_html}</td>
                    </tr>
                    """
            except Exception as e:
                print(f"⚠️ Error {home_pred}: {e}")

    # --- Guardar líneas en la BD ---
    if lines_records:
        print(f"\n💾 Guardando {len(lines_records)} líneas en SQLite...")
        save_lines_to_db(lines_records)

    # --- GENERACIÓN FINAL ---
    html_file = OUTPUT_FILE.replace('.md', '.html')
    final_html = f"""
    <!DOCTYPE html>
    <html lang="es"><head><meta charset="UTF-8"><title>NBA Smart Picks 3.0</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; padding: 40px; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); overflow: hidden; }}
        header {{ background: #1e293b; color: white; padding: 25px; text-align: center; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #f8fafc; padding: 18px; text-align: left; color: #64748b; font-size: 0.85rem; text-transform: uppercase; border-bottom: 2px solid #e2e8f0; }}
        td {{ padding: 20px; border-bottom: 1px solid #f1f5f9; font-size: 0.95rem; }}
        .team-cell {{ font-weight: 700; color: #0f172a; }}
        .tot-bold {{ font-weight: 700; color: #1e293b; }}
        tr:hover {{ background: #f8fafc; }}
        .no-data {{ padding: 40px; text-align: center; color: #94a3b8; font-style: italic; }}
    </style></head>
    <body><div class="container"><header><h1>🏀 NBA Smart Picks 3.1</h1><p>Filtros Asimétricos (L: {THR_HOME} / V: {THR_AWAY} / P: {THR_TOTAL})</p></header>
    <table><thead><tr><th>Partido (V @ L)</th><th>Modelo (V / L - Tot)</th><th>Casa (V / L - Tot)</th><th>Recomendaciones Verdes</th></tr></thead>
    <tbody>{rows_html if rows_html else '<tr><td colspan="4" class="no-data">No se encontraron señales de alta confianza para hoy.</td></tr>'}</tbody></table>
    <div style='padding: 20px; background: #f8fafc; font-size: 0.85rem;'>
        <b>Leyenda Filtro 3.1:</b> 🟢 Verde: Rentabilidad validada | ⭐: Rango de Estabilidad Validado | Umbrales: L {THR_HOME}, V {THR_AWAY}, P {THR_TOTAL}
    </div>
    </div></body></html>
    """

    with open(html_file, 'w', encoding='utf-8') as f: f.write(final_html)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f: f.write(report_md)

if __name__ == "__main__":
    generate_report()
