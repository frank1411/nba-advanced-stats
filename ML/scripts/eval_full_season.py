import os
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime
from pathlib import Path

# Asegurar imports relativos desde el paquete ML
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ML.db import NBADatabase

# Configuración
PREDICTIONS_DIR = Path("ML/data/predictions")

TEAM_NAME_TO_ID = {
    'Atlanta Hawks': 1610612737, 'Boston Celtics': 1610612738,
    'Cleveland Cavaliers': 1610612739, 'New Orleans Pelicans': 1610612740,
    'Chicago Bulls': 1610612741, 'Dallas Mavericks': 1610612742,
    'Denver Nuggets': 1610612743, 'Golden State Warriors': 1610612744,
    'Houston Rockets': 1610612745, 'Los Angeles Clippers': 1610612746,
    'LA Clippers': 1610612746, 'Los Angeles Lakers': 1610612747,
    'LA Lakers': 1610612747, 'Miami Heat': 1610612748,
    'Milwaukee Bucks': 1610612749, 'Minnesota Timberwolves': 1610612750,
    'Brooklyn Nets': 1610612751, 'New York Knicks': 1610612752,
    'Orlando Magic': 1610612753, 'Indiana Pacers': 1610612754,
    'Philadelphia 76ers': 1610612755, 'Phoenix Suns': 1610612756,
    'Portland Trail Blazers': 1610612757, 'Sacramento Kings': 1610612758,
    'San Antonio Spurs': 1610612759, 'Oklahoma City Thunder': 1610612760,
    'Toronto Raptors': 1610612761, 'Utah Jazz': 1610612762,
    'Memphis Grizzlies': 1610612763, 'Washington Wizards': 1610612764,
    'Detroit Pistons': 1610612765, 'Charlotte Hornets': 1610612766
}

def clean_name(name):
    return name.strip()

def parse_recommendation(text):
    """Extrae picks del string de recomendaciones."""
    picks = []
    # Ejemplo: 🟢 Over Local  (14.0) / 🟢 Under Visita  (-6.0) / 🟢 Over Partido ⭐ (7.5)
    parts = text.split('/')
    for p in parts:
        p = p.strip()
        if not p: continue
        
        market = ""
        if "Local" in p: market = "Local"
        elif "Visita" in p: market = "Visita"
        elif "Partido" in p: market = "Partido"
        
        type_ = "Over" if "Over" in p else "Under"
        stable = "⭐" in p
        
        picks.append({
            "market": market,
            "type": type_,
            "stable": stable
        })
    return picks

def evaluate_season():
    print("🚀 Iniciando Auditoría Histórica de Smart Picks...")
    
    # 1. Cargar resultados reales desde SQLite
    print("📂 Cargando datos reales desde SQLite...")
    with NBADatabase() as db:
        df_results = db.get_raw_games()
    df_results['GAME_DATE'] = df_results['GAME_DATE'].dt.strftime('%Y%m%d')
    
    # 2. Buscar archivos de recomendaciones
    html_files = sorted(list(PREDICTIONS_DIR.glob("recommendations_*.html")))
    
    all_evals = []
    
    for html_path in html_files:
        date_str = html_path.stem.split('_')[1]
        print(f"📅 Procesando {date_str}...")
        
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
        table = soup.find('table')
        if not table: continue
        
        rows = table.find_all('tr')[1:] # Saltar header
        
        for row in rows:
            tds = row.find_all('td')
            if len(tds) < 4: continue
            
            matchup = tds[0].text.strip()
            casa_vals = tds[2].text.strip() # V / L - Tot (ej: 115.0 / 103.0 - 218.5)
            recs_text = tds[3].text.strip()
            
            try:
                # Parsear Matchup
                away_name, home_name = [clean_name(n) for n in matchup.split('@')]
                home_id = TEAM_NAME_TO_ID.get(home_name)
                away_id = TEAM_NAME_TO_ID.get(away_name)
                
                # Parsear Líneas de la Casa
                # Regex para encontrar números: 115.0 / 103.0 - 218.5
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", casa_vals)
                if len(nums) < 3: continue
                line_v, line_h, line_tot = float(nums[0]), float(nums[1]), float(nums[2])
                
                # Buscar resultado real
                # El juego puede estar en el día date_str o date_str -1 (depende de cuándo se generó)
                # Probamos con date_str
                game_real = df_results[
                    (df_results['GAME_DATE'] == date_str) & 
                    (df_results['TEAM_ID_HOME'] == home_id) & 
                    (df_results['TEAM_ID_AWAY'] == away_id)
                ]
                
                if game_real.empty:
                    # Intentar con date_str anterior? A veces las recomendaciones se generan la noche anterior
                    continue
                
                pts_h = game_real.iloc[0]['PTS_HOME']
                pts_v = game_real.iloc[0]['PTS_AWAY']
                pts_tot = pts_h + pts_v
                
                # Evaluar Picks
                picks = parse_recommendation(recs_text)
                for pick in picks:
                    hit = False
                    if pick['market'] == "Local":
                        if pick['type'] == "Over": hit = pts_h > line_h
                        else: hit = pts_h < line_h
                    elif pick['market'] == "Visita":
                        if pick['type'] == "Over": hit = pts_v > line_v
                        else: hit = pts_v < line_v
                    elif pick['market'] == "Partido":
                        if pick['type'] == "Over": hit = pts_tot > line_tot
                        else: hit = pts_tot < line_tot
                    
                    all_evals.append({
                        "date": date_str,
                        "matchup": matchup,
                        "market": pick['market'],
                        "type": pick['type'],
                        "stable": pick['stable'],
                        "line": line_h if pick['market']=="Local" else (line_v if pick['market']=="Visita" else line_tot),
                        "actual": pts_h if pick['market']=="Local" else (pts_v if pick['market']=="Visita" else pts_tot),
                        "hit": hit
                    })
                    
            except Exception as e:
                print(f"⚠️ Error procesando partido {matchup}: {e}")
                
    # 3. Reporte Final
    if not all_evals:
        print("❌ No se pudieron evaluar recomendaciones (posible desajuste de fechas o IDs).")
        return
        
    df_eval = pd.DataFrame(all_evals)
    
    print("\n" + "="*50)
    print("📊 RESULTADOS GLOBALES SMART PICKS")
    print("="*50)
    
    total_picks = len(df_eval)
    hits = df_eval['hit'].sum()
    print(f"Total Picks Evaluados: {total_picks}")
    print(f"Hits: {hits} | Misses: {total_picks - hits}")
    print(f"Hit Rate Global: {hits/total_picks:.2%}")
    
    print("\n📈 DESGLOSE POR MERCADO:")
    for m in ["Local", "Visita", "Partido"]:
        sub = df_eval[df_eval['market'] == m]
        if not sub.empty:
            m_hits = sub['hit'].sum()
            print(f"- {m}: {m_hits}/{len(sub)} ({m_hits/len(sub):.2%})")
            
    print("\n⭐ RENDIMIENTO 'STABLE' (ALTA CONFIANZA):")
    stable = df_eval[df_eval['stable'] == True]
    if not stable.empty:
        s_hits = stable['hit'].sum()
        print(f"Total Stable: {len(stable)}")
        print(f"Stable Hit Rate: {s_hits/len(stable):.2%}")
        
        for m in ["Local", "Visita", "Partido"]:
            sub_s = stable[stable['market'] == m]
            if not sub_s.empty:
                ms_hits = sub_s['hit'].sum()
                print(f"  - Stable {m}: {ms_hits}/{len(sub_s)} ({ms_hits/len(sub_s):.2%})")

    # Guardar CSV para análisis detallado
    output_path = PREDICTIONS_DIR / "evaluacion_historica_smart_picks.csv"
    df_eval.to_csv(output_path, index=False)
    print(f"\n✅ Detalle guardado en {output_path}")

if __name__ == "__main__":
    evaluate_season()
