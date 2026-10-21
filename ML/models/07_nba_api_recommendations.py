import requests
import json
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURACIÓN DE LA API ---
CLIENT_ID = os.environ.get('PARTNER_CLIENT_ID', '')
CLIENT_SECRET = os.environ.get('PARTNER_CLIENT_SECRET', '')
TOKEN_URL = "https://cpservm.com/gateway/token"
BASE_URL = "https://cpservm.com/gateway/marketing"
REF_ID = "152"
NBA_TOURNAMENT_ID = "13589"

# --- MAPEO DE EQUIPOS (API vs LOCAL) ---
# Si un equipo en la API tiene nombre diferente a nuestro CSV, agregarlo aquí
TEAM_MAPPING = {
    "L.A. Lakers": "Los Angeles Lakers",
    "L.A. Clippers": "LA Clippers",
    "Portland T. Blazers": "Portland Trail Blazers",
    "Philadelphia 76ers": "Philadelphia 76ers",
    "N.Y. Knicks": "New York Knicks"
}

def get_access_token():
    try:
        response = requests.post(TOKEN_URL, data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        })
        return response.json().get("access_token")
    except Exception as e:
        print(f"❌ Error obteniendo token: {e}")
        return None

def fetch_nba_odds(token):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    url = f"{BASE_URL}/datafeed/prematch/api/v2/sportevents?ref={REF_ID}&tournamentIds={NBA_TOURNAMENT_ID}&schemeOfGettingOddsOperations=GetAllOdds"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("items", [])
        return []
    except Exception as e:
        print(f"❌ Error consultando la API: {e}")
        return []

def analyze_value(pred, market_odds):
    recos = []
    
    # 1. Total del Partido (Type 9-Over / 10-Under)
    pred_total = pred['PRED_PTS_HOME'] + pred['PRED_PTS_AWAY']
    total_line = next((o for o in market_odds if o.get('type') == 9 and o.get('isCenter')), None)
    if total_line:
        line_val = total_line.get('parameter')
        diff = pred_total - line_val
        if abs(diff) >= 7.0:
            status = "🟢" if abs(diff) >= 16.0 else "🟡"
            type_str = "Over" if diff > 0 else "Under"
            recos.append(f"{status} **{type_str} Partido** (+{abs(diff):.1f})")

    # 2. Handicap (Type 7-Hnd Local / 8-Hnd Visita)
    pred_margin = pred['PRED_MARGIN'] # PRED_PTS_HOME - PRED_PTS_AWAY
    hnd_line = next((o for o in market_odds if o.get('type') == 7 and o.get('isCenter')), None)
    if hnd_line:
        line_val = hnd_line.get('parameter') # Ej: -4.5 para el local
        # Si pred_margin (+4) es mejor que line (-4.5), hay valor en el Local
        diff = pred_margin + line_val 
        if abs(diff) >= 2.5:
             status = "🟢" if abs(diff) >= 5.0 else "🟡"
             side = pred['TEAM_NAME_HOME'] if diff > 0 else pred['TEAM_NAME_AWAY']
             recos.append(f"{status} **{side} Hnd** (+{abs(diff):.1f})")

    # 3. Totales Individuales (11/12 Home, 13/14 Away)
    # Local
    ind_h_line = next((o for o in market_odds if o.get('type') == 11 and o.get('isCenter')), None)
    if ind_h_line:
        diff_h = pred['PRED_PTS_HOME'] - ind_h_line.get('parameter')
        if abs(diff_h) >= 14.0:
            status = "🟢" if abs(diff_h) >= 17.0 else "🟡"
            type_str = "Over" if diff_h > 0 else "Under"
            recos.append(f"{status} **{type_str} Ind. Local** (+{abs(diff_h):.1f})")

    # Visita
    ind_v_line = next((o for o in market_odds if o.get('type') == 13 and o.get('isCenter')), None)
    if ind_v_line:
        diff_v = pred['PRED_PTS_AWAY'] - ind_v_line.get('parameter')
        if abs(diff_v) >= 4.0:
            status = "🟢" if abs(diff_v) >= 7.0 else "🟡"
            type_str = "Over" if diff_v > 0 else "Under"
            recos.append(f"{status} **{type_str} Ind. Visita** (+{abs(diff_v):.1f})")

    return " / ".join(recos) if recos else "⚪ No se detectó valor claro"

def main():
    date_str = datetime.now().strftime("%Y%m%d")
    pred_path = f"ML/data/predictions/predictions_{date_str}.csv"
    
    if not os.path.exists(pred_path):
        print(f"❌ No se encontró el archivo de predicciones de hoy: {pred_path}")
        return

    print(f"🚀 Iniciando análisis inteligente para el {date_str}...")
    token = get_access_token()
    if not token: return
    
    api_games = fetch_nba_odds(token)
    if not api_games:
        print("ℹ️ La API no devolvió partidos de NBA para hoy (Tournament 13589).")
        # Generamos un reporte informando el retraso
        with open(f"ML/data/predictions/recommendations_{date_str}.md", "w") as f:
            f.write(f"# 🏀 Recomendaciones Inteligentes NBA - {date_str}\n\n")
            f.write("> [!WARNING]\n")
            f.write("> **Estado de la API:** El feed de la NBA está actualmente vacío. Esto puede deberse a que las cuotas aún no han sido publicadas por el proveedor para el `ref: 152`.\n\n")
            f.write("Vuelve a ejecutar este script en una hora para sincronizar las cuotas.")
        return

    predictions = pd.read_csv(pred_path)
    
    report_rows = []
    for _, pred in predictions.iterrows():
        # Buscar el juego en la API
        home_name = pred['TEAM_NAME_HOME']
        away_name = pred['TEAM_NAME_AWAY']
        
        # Mapeo inverso para buscar en la API si es necesario
        match = None
        for g in api_games:
            api_home = g.get('opponent1NameLocalization')
            api_away = g.get('opponent2NameLocalization')
            
            # Normalizar nombres para comparación
            api_home_mapped = TEAM_MAPPING.get(api_home, api_home)
            api_away_mapped = TEAM_MAPPING.get(api_away, api_away)
            
            if (api_home_mapped == home_name) or (api_away_mapped == away_name):
                match = g
                break
        
        if match:
            odds = match.get('oddsLocalization', [])
            reco = analyze_value(pred, odds)
            
            # Formatear datos de Casa
            h_line = next((o for o in odds if o.get('type') == 11 and o.get('isCenter')), {}).get('parameter', '-')
            v_line = next((o for o in odds if o.get('type') == 13 and o.get('isCenter')), {}).get('parameter', '-')
            t_line = next((o for o in odds if o.get('type') == 9 and o.get('isCenter')), {}).get('parameter', '-')
            hnd_val = next((o for o in odds if o.get('type') == 7 and o.get('isCenter')), {}).get('parameter', '-')
            
            row = (f"**{home_name} @ {away_name}**", 
                   f"{pred['PRED_PTS_HOME']:.0f} / {pred['PRED_PTS_AWAY']:.0f} / **{pred['PRED_PTS_HOME']+pred['PRED_PTS_AWAY']:.0f}** ({pred['PRED_MARGIN']:+.1f})",
                   f"±{h_line} / ±{v_line} / **{t_line}** / {hnd_val}",
                   reco)
            report_rows.append(row)

    # --- GENERACIÓN DEL REPORTE MD ---
    with open(f"ML/data/predictions/recommendations_{date_str}.md", "w") as f:
        f.write(f"# 🏆 Reporte Maestro de Valor NBA - {date_str}\n\n")
        f.write("| Encuentro (L/V) | Modelo (H/V/Tot/Mrg) | Casa (H±/V±/Tot/Hnd) | Recomendaciones (Valor) |\n")
        f.write("| :--- | :---: | :---: | :--- |\n")
        for r in report_rows:
            f.write(f"| {' | '.join(map(str, r))} |\n")
        
        if not report_rows:
             f.write("\n> No se pudieron vincular automáticamente los partidos de la API con las predicciones locales. Verifica el mapeo de nombres de equipos.")

    print(f"✅ Reporte generado en: ML/data/predictions/recommendations_{date_str}.md")

if __name__ == "__main__":
    main()
