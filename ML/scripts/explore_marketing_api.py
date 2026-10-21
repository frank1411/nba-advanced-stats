import requests
import json
import os
from datetime import datetime, timedelta

CLIENT_ID = os.environ.get('PARTNER_CLIENT_ID', '')
CLIENT_SECRET = os.environ.get('PARTNER_CLIENT_SECRET', '')
REF = "152"

TOKEN_URL = "https://cpservm.com/gateway/token"
BASE_URL = "https://cpservm.com/gateway/marketing"

def get_token():
    data = {"grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
    response = requests.post(TOKEN_URL, data=data)
    return response.json().get("access_token") if response.status_code == 200 else None

def explore():
    token = get_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    # 1. Obtener TODOS los eventos de Basketball (Sport ID 3)
    # Sin filtrar por torneo aún para ver qué hay disponible
    print("\n--- Listando Eventos de Basketball Generales ---")
    
    # Buscamos eventos que empiecen desde hoy
    now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    tomorrow = (datetime.utcnow() + timedelta(days=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    events_url = f"{BASE_URL}/datafeed/prematch/api/v2/sportevents?ref={REF}&SportIds=3&count=100"
    res = requests.get(events_url, headers=headers)
    
    if res.status_code == 200:
        events = res.json().get("items", [])
        print(f"Se encontraron {len(events)} eventos de basket en total.")
        
        nba_events = []
        for e in events:
            t_name = e.get('tournamentNameLocalization', '').upper()
            if "NBA" in t_name and e.get('opponent2NameLocalization'):
                nba_events.append(e)
        
        print(f"Total partidos NBA detectados: {len(nba_events)}")
        
        if nba_events:
            match = nba_events[0]
            print(f"✅ Ejemplo NBA: {match.get('opponent1NameLocalization')} vs {match.get('opponent2NameLocalization')} (ID: {match.get('sportEventId')})")
            print(f"Torneo: {match.get('tournamentNameLocalization')} (ID: {match.get('tournamentId')})")
            
            # 2. SEGUNDO INTENTO: Obtener mercados para este partido real
            m_id = match.get('sportEventId')
            print(f"\n--- Mercados para el partido real {m_id} ---")
            markets_url = f"{BASE_URL}/datafeed/prematch/api/v2/sportevents?ref={REF}&SportEventIds={m_id}&schemeOfGettingOddsOperations=GetAllOdds"
            res_m = requests.get(markets_url, headers=headers)
            if res_m.status_code == 200:
                data = res_m.json().get("items", [])
                if data:
                    odds = data[0].get("oddsLocalization", [])
                    print(f"Se encontraron {len(odds)} cuotas.")
                    # Guardar una muestra en un archivo para análisis manual
                    with open('/tmp/odds_sample.json', 'w') as f:
                        json.dump(odds, f, indent=2)
                    
                    # Buscar Handicaps (Spread) y Totales (Over/Under)
                    # Basado en el YAML: 
                    # 7/8 suele ser Handicap Principal (Spread)
                    # 9/10 suele ser Total Principal (Over/Under)
                    for o in odds:
                        name = o.get('name', '') or ''
                        if "total" in name.lower() or "handicap" in name.lower() or "over" in name.lower() or "under" in name.lower():
                            print(f"🎯 [{o.get('id')}] {name}: Value={o.get('value')} | Param={o.get('param')}")
        else:
            print("⚠️ No se encontraron partidos de la NBA activos en este momento.")
            # Si no hay NBA, imprimimos los torneos disponibles para ver qué pasa
            tournaments_present = set([e.get('tournamentNameLocalization') for e in events])
            print("Torneos disponibles en el feed actual:", tournaments_present)
    else:
        print(f"Error eventos: {res.text}")

if __name__ == "__main__":
    explore()
