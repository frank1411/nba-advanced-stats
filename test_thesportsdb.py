import requests
from datetime import datetime

def test_thesportsdb():
    """Prueba la API de TheSportsDB para obtener datos de la NBA"""
    print("=== PRUEBA DE THESPORTSDB API ===\n")
    
    # ID de la NBA en TheSportsDB
    NBA_LEAGUE_ID = '4387'  # ID de la NBA en TheSportsDB
    
    # Base URL de la API
    BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"
    
    # 1. Obtener información básica de la liga
    print("1. Obteniendo información de la liga NBA...")
    try:
        league_url = f"{BASE_URL}/lookupleague.php?id={NBA_LEAGUE_ID}"
        response = requests.get(league_url, timeout=10)
        response.raise_for_status()
        league_data = response.json()
        
        if 'leagues' in league_data and league_data['leagues']:
            league = league_data['leagues'][0]
            print(f"Liga: {league['strLeague']}")
            print(f"Deporte: {league['strSport']}")
            print(f"Temporada actual: {league.get('intFormedYear', 'No disponible')}")
            print(f"Sitio web: {league.get('strWebsite', 'No disponible')}\n")
        else:
            print("No se pudo obtener información de la liga.\n")
    except Exception as e:
        print(f"Error al obtener información de la liga: {e}\n")
    
    # 2. Obtener los próximos eventos (partidos) de la NBA
    print("2. Obteniendo próximos partidos de la NBA...")
    try:
        # Primero, intentamos obtener la temporada actual
        current_year = datetime.now().year
        events_url = f"{BASE_URL}/eventsseason.php?id={NBA_LEAGUE_ID}&s={current_year}-{current_year+1}"
        
        response = requests.get(events_url, timeout=10)
        response.raise_for_status()
        events_data = response.json()
        
        if 'events' in events_data and events_data['events']:
            print(f"\nPróximos partidos de la NBA ({len(events_data['events'])} encontrados):")
            
            # Mostrar los próximos 5 partidos
            for i, event in enumerate(events_data['events'][:5], 1):
                try:
                    event_date = datetime.strptime(event['dateEvent'], '%Y-%m-%d')
                    date_str = event_date.strftime('%Y-%m-%d')
                    
                    print(f"\nPartido {i}:")
                    print(f"{event['strAwayTeam']} vs {event['strHomeTeam']}")
                    print(f"Fecha: {date_str} {event.get('strTime', '')}")
                    print(f"Liga: {event.get('strLeague', 'NBA')}")
                    print(f"Estado: {event.get('strStatus', 'Programado')}")
                    
                    # Mostrar información adicional si está disponible
                    if 'strVenue' in event and event['strVenue']:
                        print(f"Estadio: {event['strVenue']}")
                    
                    # Mostrar marcador si el partido ya comenzó
                    if 'intAwayScore' in event and 'intHomeScore' in event:
                        print(f"Resultado: {event['strAwayTeam']} {event['intAwayScore']} - {event['intHomeScore']} {event['strHomeTeam']}")
                
                except (KeyError, ValueError) as e:
                    print(f"Error al procesar el partido: {e}")
                    continue
        else:
            print("No se encontraron partidos próximos en la temporada actual.")
            
            # Intentar con la temporada anterior por si acaso
            print("\nIntentando con la temporada anterior...")
            events_url = f"{BASE_URL}/eventsseason.php?id={NBA_LEAGUE_ID}&s={current_year-1}-{current_year}"
            response = requests.get(events_url, timeout=10)
            response.raise_for_status()
            events_data = response.json()
            
            if 'events' in events_data and events_data['events']:
                print(f"\nPartidos de la temporada {current_year-1}-{current_year} (últimos 5):")
                for i, event in enumerate(events_data['events'][-5:], 1):
                    try:
                        event_date = datetime.strptime(event['dateEvent'], '%Y-%m-%d')
                        date_str = event_date.strftime('%Y-%m-%d')
                        
                        print(f"\nPartido {i}:")
                        print(f"{event['strAwayTeam']} vs {event['strHomeTeam']}")
                        print(f"Fecha: {date_str}")
                        
                        if 'intAwayScore' in event and 'intHomeScore' in event:
                            print(f"Resultado: {event['strAwayTeam']} {event['intAwayScore']} - {event['intHomeScore']} {event['strHomeTeam']}")
                    
                    except (KeyError, ValueError) as e:
                        print(f"Error al procesar el partido: {e}")
                        continue
            else:
                print("No se encontraron partidos en la temporada anterior.")
    
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
    
    # 3. Obtener información de los equipos de la NBA
    print("\n3. Obteniendo información de los equipos de la NBA...")
    try:
        teams_url = f"{BASE_URL}/lookup_all_teams.php?id={NBA_LEAGUE_ID}"
        response = requests.get(teams_url, timeout=10)
        response.raise_for_status()
        teams_data = response.json()
        
        if 'teams' in teams_data and teams_data['teams']:
            print(f"\nEquipos de la NBA ({len(teams_data['teams'])} equipos):")
            
            # Mostrar información básica de los primeros 5 equipos
            for i, team in enumerate(teams_data['teams'][:5], 1):
                print(f"\nEquipo {i}: {team['strTeam']}")
                print(f"Abreviatura: {team.get('strTeamShort', 'N/A')}")
                print(f"Estadio: {team.get('strStadium', 'N/A')}")
                print(f"Ubicación: {team.get('strStadiumLocation', 'N/A')}")
                
                if 'strWebsite' in team and team['strWebsite']:
                    print(f"Sitio web: {team['strWebsite']}")
            
            if len(teams_data['teams']) > 5:
                print(f"\n... y {len(teams_data['teams']) - 5} equipos más.")
        else:
            print("No se pudieron obtener los equipos de la NBA.")
    
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión al obtener equipos: {e}")
    except Exception as e:
        print(f"Error inesperado al obtener equipos: {e}")

if __name__ == "__main__":
    test_thesportsdb()
