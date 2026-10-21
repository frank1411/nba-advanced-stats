"""
Script para hacer seguimiento de recomendaciones de apuestas vs resultados reales.

Este script lee los archivos de recomendaciones HTML, extrae las recomendaciones de apuestas
(Over/Under), busca los resultados reales en el dataset histórico y guarda el seguimiento.
"""

import pandas as pd
import os
import re
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
import json


# Mapeo de nombres completos de equipos a abreviaturas
TEAM_NAME_TO_ABBREV = {
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
    'Washington Wizards': 'WAS'
}


def parse_recommendations(recommendations_text: str) -> list:
    """
    Parsea el texto de recomendaciones para extraer las apuestas individuales.
    
    Args:
        recommendations_text: Texto con recomendaciones (ej: "🟢 Over Visita ⭐ (7.5) / 🟢 Over Partido ⭐ (10.5)")
        
    Returns:
        Lista de diccionarios con cada recomendación
    """
    recommendations = []
    
    # Dividir por '/', salto de línea y <br> (el HTML transforma <br> en \n)
    parts = re.split(r'[/\n]+', recommendations_text)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # 🔴 = NO jugar: registrar omitido (no es una apuesta a evaluar)
        if '🔴' in part:
            continue
        
        # Extraer umbral + token abreviado (formato 3.1: OV/OL/OP/UV/UL/UP)
        m = re.search(r'\b([OU])([VLP])\s*\(([-\d.]+)\)', part)
        if m:
            bet_type = 'Over' if m.group(1) == 'O' else 'Under'
            market = {'V': 'Visitante', 'L': 'Local', 'P': 'Partido'}[m.group(2)]
            threshold = float(m.group(3))
        else:
            # Formato legacy: "Over Visita (7.5)"
            bet_type = None
            market = None
            if 'Over' in part:
                bet_type = 'Over'
            elif 'Under' in part:
                bet_type = 'Under'
            if bet_type:
                if 'Visita' in part:
                    market = 'Visitante'
                elif 'Local' in part or 'Home' in part:
                    market = 'Local'
                elif 'Partido' in part:
                    market = 'Partido'
            if not bet_type or not market:
                # Moneyline: ML-L = favorito local, ML-V = favorito visitante
                ml_m = re.search(r'ML\s*-?\s*([LV])', part)
                if ml_m:
                    threshold_match = re.search(r'\(([-\d.]+)\)', part)
                    recommendations.append({
                        'bet_type': f"ML_{ml_m.group(1)}",
                        'market': 'ML',
                        'threshold': float(threshold_match.group(1)) if threshold_match else 0.0,
                        'has_star': '⭐' in part,
                        'full_text': part
                    })
                continue
            threshold_match = re.search(r'\(([-\d.]+)\)', part)
            threshold = float(threshold_match.group(1)) if threshold_match else 0.0
        
        # Verificar si tiene estrella (⭐)
        has_star = '⭐' in part
        
        recommendations.append({
            'bet_type': bet_type,
            'market': market,
            'threshold': threshold,
            'has_star': has_star,
            'full_text': part
        })
    
    return recommendations


def extract_betting_data_from_html(html_file: str) -> list:
    """
    Extrae los datos de apuestas del archivo HTML de recomendaciones.
    
    Args:
        html_file: Ruta al archivo HTML de recomendaciones
        
    Returns:
        Lista de diccionarios con los datos de apuestas extraídos
    """
    if not os.path.exists(html_file):
        print(f"❌ No se encontró el archivo: {html_file}")
        return []
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    betting_data = []
    
    # Buscar filas de la tabla
    table = soup.find('table')
    if not table:
        print("❌ No se encontró la tabla en el archivo HTML")
        return []
    
    tbody = table.find('tbody')
    if not tbody:
        print("❌ No se encontró tbody en la tabla")
        return []
    
    rows = tbody.find_all('tr')
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 4:
            # Extraer información
            partido_cell = cells[0].get_text(strip=True)
            model_cell = cells[1].get_text(strip=True)
            casa_cell = cells[2].get_text(strip=True)
            recomendaciones_cell = cells[3].get_text(strip=True)
            
            # Parsear partido
            if '@' in partido_cell:
                away_team, home_team = partido_cell.split(' @ ')
            else:
                continue
            
            # Parsear líneas de apuestas (casa)
            # Formato: "101.5 / 116.0 -217.5"
            casa_parts = casa_cell.split(' -')
            scores_part = casa_parts[0].strip() if len(casa_parts) > 0 else ""
            total_line = float(casa_parts[1].strip()) if len(casa_parts) > 1 else 0.0
            
            if '/' in scores_part:
                away_line, home_line = scores_part.split(' / ')
                try:
                    away_line = float(away_line.strip())
                    home_line = float(home_line.strip())
                except ValueError:
                    away_line = 0.0
                    home_line = 0.0
            else:
                away_line = 0.0
                home_line = 0.0
            
            # Parsear predicciones del modelo
            model_parts = model_cell.split(' -')
            model_scores_part = model_parts[0].strip() if len(model_parts) > 0 else ""
            model_total = float(model_parts[1].strip()) if len(model_parts) > 1 else 0.0
            
            if '/' in model_scores_part:
                away_pred, home_pred = model_scores_part.split(' / ')
                try:
                    away_pred = float(away_pred.strip())
                    home_pred = float(home_pred.strip())
                except ValueError:
                    away_pred = 0.0
                    home_pred = 0.0
            else:
                away_pred = 0.0
                home_pred = 0.0
            
            # Parsear recomendaciones de apuestas
            recommendations = parse_recommendations(recomendaciones_cell)
            
            betting_data.append({
                'away_team': away_team.strip(),
                'home_team': home_team.strip(),
                'away_line': away_line,
                'home_line': home_line,
                'total_line': total_line,
                'away_pred': away_pred,
                'home_pred': home_pred,
                'total_pred': model_total,
                'recommendations': recommendations
            })
    
    return betting_data


def load_historical_data(raw_data_file: str):
    """
    Carga el dataset histórico UNA sola vez por corrida.
    
    Orden de preferencia:
      1. SQLite (nba.db) — siempre actualizado por 02_update_nba_data.py
      2. Parquet — respaldo histórico (puede estar desactualizado)
    
    Returns:
        DataFrame histórico o None si no hay fuente disponible
    """
    try:
        import sys
        _pr = Path(__file__).parent.parent.parent
        if str(_pr) not in sys.path:
            sys.path.insert(0, str(_pr))
        from ML.db import NBADatabase
        with NBADatabase() as db:
            df = db.get_raw_games()
            if df is not None and not df.empty:
                print("📂 Datos históricos cargados desde SQLite (nba.db)")
                return df
    except Exception as e:
        print(f"⚠️ SQLite no disponible ({e}); intentando parquet...")
    
    if os.path.exists(raw_data_file):
        df = pd.read_parquet(raw_data_file)
        print(f"📂 Datos históricos cargados desde parquet: {raw_data_file}")
        return df
    
    print(f"❌ Sin fuente de datos históricos (ni SQLite ni {raw_data_file})")
    return None


def get_actual_results(date_str: str, betting_data: list, historical_df) -> list:
    """
    Busca los resultados reales de los partidos en el dataset histórico.
    
    Args:
        date_str: Fecha en formato YYYY-MM-DD
        betting_data: Lista de datos de apuestas con equipos
        historical_df: DataFrame histórico ya cargado (de load_historical_data)
        
    Returns:
        Lista de diccionarios con datos de apuestas y resultados reales
    """
    if historical_df is None or historical_df.empty:
        print("❌ No hay datos históricos para evaluar")
        return []

    df = historical_df
    
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    # Filtrar por fecha
    target_date = pd.to_datetime(date_str)
    df_date = df[df['GAME_DATE'] == target_date]
    
    if df_date.empty:
        print(f"⚠️ No se encontraron partidos para la fecha: {date_str}")
        # Agregar game_date a cada registro
        for bet_data in betting_data:
            bet_data['game_date'] = date_str
            # Marcar recomendaciones como no evaluadas
            evaluated_recommendations = []
            for rec in bet_data['recommendations']:
                rec_evaluated = rec.copy()
                rec_evaluated['was_correct'] = None
                rec_evaluated['difference'] = None
                evaluated_recommendations.append(rec_evaluated)
            bet_data['recommendations'] = evaluated_recommendations
            bet_data['actual_home'] = None
            bet_data['actual_away'] = None
            bet_data['actual_total'] = None
        return betting_data
    
    results = []
    
    for bet_data in betting_data:
        # Convertir nombres de equipos a abreviaturas
        home_abbrev = TEAM_NAME_TO_ABBREV.get(bet_data['home_team'], bet_data['home_team'])
        away_abbrev = TEAM_NAME_TO_ABBREV.get(bet_data['away_team'], bet_data['away_team'])
        
        # Buscar match
        match = df_date[
            (df_date['TEAM_ABBREVIATION_HOME'] == home_abbrev) &
            (df_date['TEAM_ABBREVIATION_AWAY'] == away_abbrev)
        ]
        
        if match.empty:
            # Intentar sin mapeo
            match = df_date[
                (df_date['TEAM_ABBREVIATION_HOME'].str.upper() == bet_data['home_team'].upper()) &
                (df_date['TEAM_ABBREVIATION_AWAY'].str.upper() == bet_data['away_team'].upper())
            ]
        
        if not match.empty:
            game = match.iloc[0]
            bet_data_with_result = bet_data.copy()
            bet_data_with_result['actual_home'] = game['PTS_HOME']
            bet_data_with_result['actual_away'] = game['PTS_AWAY']
            bet_data_with_result['actual_total'] = game['PTS_HOME'] + game['PTS_AWAY']
            bet_data_with_result['game_date'] = date_str
            
            # Evaluar cada recomendación
            evaluated_recommendations = []
            for rec in bet_data['recommendations']:
                rec_evaluated = rec.copy()
                
                # Determinar si la recomendación fue acertada
                if rec['market'] == 'Visitante':
                    # Over/Under Visitante: comparar resultado real vs línea
                    actual = bet_data_with_result['actual_away']
                    line = bet_data_with_result['away_line']
                    if rec['bet_type'] == 'Over':
                        rec_evaluated['was_correct'] = actual > line
                        rec_evaluated['difference'] = actual - line
                    else:  # Under
                        rec_evaluated['was_correct'] = actual < line
                        rec_evaluated['difference'] = line - actual
                
                elif rec['market'] == 'Local':
                    # Over/Under Local: comparar resultado real vs línea
                    actual = bet_data_with_result['actual_home']
                    line = bet_data_with_result['home_line']
                    if rec['bet_type'] == 'Over':
                        rec_evaluated['was_correct'] = actual > line
                        rec_evaluated['difference'] = actual - line
                    else:  # Under
                        rec_evaluated['was_correct'] = actual < line
                        rec_evaluated['difference'] = line - actual
                
                elif rec['market'] == 'Partido':
                    # Over/Under Partido: comparar total real vs línea
                    actual = bet_data_with_result['actual_total']
                    line = bet_data_with_result['total_line']
                    if rec['bet_type'] == 'Over':
                        rec_evaluated['was_correct'] = actual > line
                        rec_evaluated['difference'] = actual - line
                    else:  # Under
                        rec_evaluated['was_correct'] = actual < line
                        rec_evaluated['difference'] = line - actual
                
                elif rec['market'] == 'ML':
                    # Moneyline (sin línea de la casa): ¿ganó el favorito predicho?
                    # ML_L → favorito el local · ML_V → favorito el visitante
                    if rec['bet_type'] == 'ML_L':
                        rec_evaluated['was_correct'] = bet_data_with_result['actual_home'] > bet_data_with_result['actual_away']
                        rec_evaluated['difference'] = bet_data_with_result['actual_home'] - bet_data_with_result['actual_away']
                    elif rec['bet_type'] == 'ML_V':
                        rec_evaluated['was_correct'] = bet_data_with_result['actual_away'] > bet_data_with_result['actual_home']
                        rec_evaluated['difference'] = bet_data_with_result['actual_away'] - bet_data_with_result['actual_home']
                    else:
                        rec_evaluated['was_correct'] = None
                        rec_evaluated['difference'] = None
                
                evaluated_recommendations.append(rec_evaluated)
            
            bet_data_with_result['recommendations'] = evaluated_recommendations
            results.append(bet_data_with_result)
        else:
            print(f"⚠️ No se encontró resultado para: {bet_data['away_team']} @ {bet_data['home_team']}")
            bet_data_with_result = bet_data.copy()
            bet_data_with_result['actual_home'] = None
            bet_data_with_result['actual_away'] = None
            bet_data_with_result['actual_total'] = None
            bet_data_with_result['game_date'] = date_str
            
            # Marcar recomendaciones como no evaluadas
            evaluated_recommendations = []
            for rec in bet_data['recommendations']:
                rec_evaluated = rec.copy()
                rec_evaluated['was_correct'] = None
                rec_evaluated['difference'] = None
                evaluated_recommendations.append(rec_evaluated)
            
            bet_data_with_result['recommendations'] = evaluated_recommendations
            results.append(bet_data_with_result)
    
    return results


def flatten_tracking_data(results: list) -> list:
    """
    Convierte los datos anidados de seguimiento en una lista plana de registros.
    
    Args:
        results: Lista de resultados con recomendaciones anidadas
        
    Returns:
        Lista plana de registros individuales de apuestas
    """
    flat_records = []
    
    for result in results:
        for rec in result['recommendations']:
            record = {
                'game_date': result['game_date'],
                'away_team': result['away_team'],
                'home_team': result['home_team'],
                'bet_type': rec['bet_type'],
                'market': rec['market'],
                'threshold': rec['threshold'],
                'has_star': rec['has_star'],
                'away_line': result['away_line'],
                'home_line': result['home_line'],
                'total_line': result['total_line'],
                'away_pred': result['away_pred'],
                'home_pred': result['home_pred'],
                'total_pred': result['total_pred'],
                'actual_home': result['actual_home'],
                'actual_away': result['actual_away'],
                'actual_total': result['actual_total'],
                'was_correct': rec['was_correct'],
                'difference': rec['difference']
            }
            flat_records.append(record)
    
    return flat_records


def save_tracking_results(results: list, output_file: str):
    """
    Guarda los resultados del seguimiento en un archivo CSV.
    
    Args:
        results: Lista de resultados con predicciones vs realidad
        output_file: Ruta del archivo de salida
    """
    if not results:
        print("❌ No hay resultados para guardar")
        return
    
    # Aplanar los datos
    flat_records = flatten_tracking_data(results)
    df = pd.DataFrame(flat_records)
    
    if df.empty:
        print("ℹ️ Sin picks evaluados hoy (sin resultados aún) — no se actualiza el CSV")
        return
    # Verificar si el archivo ya existe
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            existing_df = pd.read_csv(output_file)
        except pd.errors.EmptyDataError:
            existing_df = None
    else:
        existing_df = None
    if existing_df is not None:
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=['game_date', 'home_team', 'away_team', 'market', 'bet_type', 'threshold'],
            keep='last'
        )
        combined_df.to_csv(output_file, index=False)
        print(f"✅ Resultados agregados al archivo existente: {output_file}")
    else:
        df.to_csv(output_file, index=False)
        print(f"✅ Nuevo archivo de seguimiento creado: {output_file}")
    
    # Mostrar resumen
    print(f"\n📊 Resumen del seguimiento:")
    print(f"Total de registros de apuestas: {len(df)}")
    
    # Accuracy por mercado
    for market in ['Visitante', 'Local', 'Partido', 'ML']:
        market_df = df[df['market'] == market]
        if market_df['was_correct'].notna().any():
            # Normaliza booleanos y strings ('True'/'False' al leer del CSV)
            evaluated = market_df['was_correct'].dropna()
            was_correct_bool = evaluated.astype(str).str.strip().str.lower().eq('true')
            correct = int(was_correct_bool.sum())
            total = int(len(evaluated))
            
            if total > 0:
                accuracy = (correct / total) * 100
                print(f"Accuracy {market}: {accuracy:.1f}% ({correct}/{total})")


def main():
    """Función principal para ejecutar el seguimiento."""
    print("🏀 Iniciando seguimiento de recomendaciones de apuestas vs resultados reales")
    print("=" * 70)
    
    # Configuración
    start_date = datetime(2026, 4, 4)  # Fecha de inicio: 04/04/2026
    end_date = datetime.now()
    
    raw_data_file = 'ML/data/raw/nba_games_raw.parquet'
    tracking_file = 'ML/data/predictions/betting_tracking.csv'
    recommendations_dir = 'ML/data/predictions/'
    
    # Cargar el dataset histórico UNA vez (antes se releía por cada fecha del rango)
    historical_df = load_historical_data(raw_data_file)
    if historical_df is None:
        print("❌ Sin datos históricos: no se puede hacer tracking")
        return
    historical_df['GAME_DATE'] = pd.to_datetime(historical_df['GAME_DATE'])
    
    print(f"📅 Rango de fechas: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
    print(f"📄 Directorio de recomendaciones: {recommendations_dir}")
    print(f"📄 Archivo de datos históricos: {raw_data_file}")
    print(f"📄 Archivo de seguimiento: {tracking_file}")
    print()
    
    # Buscar todos los archivos de recomendaciones en el rango
    all_results = []
    processed_files = 0
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        date_display = current_date.strftime('%Y-%m-%d')
        html_file = f'{recommendations_dir}recommendations_{date_str}.html'
        
        if os.path.exists(html_file):
            print(f"📖 Procesando archivo: {date_display}")
            
            # Paso 1: Extraer datos de apuestas del HTML
            betting_data = extract_betting_data_from_html(html_file)
            
            if betting_data:
                print(f"  ✅ Se extrajeron {len(betting_data)} partidos")
                
                # Paso 2: Buscar resultados reales y evaluar recomendaciones
                results = get_actual_results(date_display, betting_data, historical_df)
                
                if results:
                    total_recs = sum(len(r['recommendations']) for r in results)
                    print(f"  ✅ Se evaluaron {total_recs} recomendaciones")
                    all_results.extend(results)
                    processed_files += 1
                else:
                    print(f"  ⚠️ No se encontraron resultados para esta fecha")
            else:
                print(f"  ⚠️ No se encontraron datos de apuestas en este archivo")
        else:
            pass
        
        current_date += pd.Timedelta(days=1)
    
    print()
    print(f"📊 Resumen del procesamiento:")
    print(f"  - Archivos procesados: {processed_files}")
    total_recs = sum(len(r['recommendations']) for r in all_results)
    print(f"  - Total de recomendaciones evaluadas: {total_recs}")
    print()
    
    if not all_results:
        print("❌ No se encontraron resultados para guardar")
        return
    
    # Paso 3: Guardar seguimiento
    print("💾 Guardando resultados de seguimiento...")
    save_tracking_results(all_results, tracking_file)
    
    print()
    print("=" * 70)
    print("✅ Seguimiento completado exitosamente")


if __name__ == "__main__":
    main()
