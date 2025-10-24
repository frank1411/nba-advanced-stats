import requests
from datetime import datetime, timedelta
import json

def get_nba_scoreboard(date_str=None):
    """Obtiene el scoreboard de la NBA para una fecha específica"""
    try:
        # Si no se proporciona fecha, usar hoy
        if date_str is None:
            date_str = datetime.today().strftime('%m/%d/%Y')
            
        print(f"\nObteniendo partidos para: {date_str}")
        
        # URL de la API de NBA Stats
        url = 'https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json'
        
        # Headers para simular una petición de navegador
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.nba.com/',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true'
        }
        
        # Hacer la petición con verificación SSL desactivada (solo para pruebas)
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        
        # Procesar la respuesta
        data = response.json()
        
        # Verificar si hay partidos
        if 'scoreboard' in data and 'games' in data['scoreboard']:
            games = data['scoreboard']['games']
            print(f"\nSe encontraron {len(games)} partidos:")
            
            for game in games:
                home_team = game['homeTeam']['teamName']
                away_team = game['awayTeam']['teamName']
                game_time = game.get('gameStatusText', 'Hora no disponible')
                game_clock = game.get('gameClock', '')
                game_status = game.get('gameStatus', '')
                
                print(f"\n{away_team} @ {home_team}")
                print(f"Estado: {game_status} {game_clock}")
                print(f"Hora: {game_time}")
                
                # Mostrar puntuación si el partido ya comenzó
                if 'homeTeam' in game and 'score' in game['homeTeam']:
                    print(f"Marcador: {away_team} {game['awayTeam']['score']} - {game['homeTeam']['score']} {home_team}")
            
            return True
        else:
            print("No se encontraron partidos para la fecha especificada.")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión: {str(e)}")
        return False
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        return False

def get_nba_schedule(season_year):
    """Obtiene el calendario de la NBA para una temporada específica"""
    try:
        print(f"\nObteniendo calendario para la temporada {season_year}-{season_year+1}...")
        
        # URL para obtener el calendario
        url = f'https://data.nba.net/data/10s/prod/v1/{season_year}/schedule.json'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://www.nba.com/'
        }
        
        # Desactivar advertencias de solicitud no segura
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Hacer la petición con verificación SSL desactivada (solo para pruebas)
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        data = response.json()
        
        if 'league' in data and 'standard' in data['league'] and 'game' in data['league']['standard']:
            games = data['league']['standard']
            print(f"\nTotal de partidos en la temporada: {len(games)}")
            
            # Mostrar los próximos 5 partidos
            print("\nPróximos partidos:")
            today = datetime.today()
            upcoming_games = []
            
            for game in games:
                try:
                    game_date = datetime.strptime(game['startTimeUTC'], '%Y-%m-%dT%H:%M:%S.%fZ')
                    if game_date >= today:
                        home_team = game['hTeam']['triCode']
                        away_team = game['vTeam']['triCode']
                        game_time = game_date.strftime('%Y-%m-%d %H:%M')
                        upcoming_games.append((game_date, f"{away_team} @ {home_team} - {game_time}"))
                except (KeyError, ValueError) as e:
                    continue
            
            # Ordenar por fecha y mostrar los próximos 5
            upcoming_games.sort()
            for _, game_info in upcoming_games[:5]:
                print(game_info)
            
            return True
        else:
            print("No se pudo obtener el calendario.")
            return False
            
    except Exception as e:
        print(f"Error al obtener el calendario: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== PRUEBA DE API DE LA NBA ===")
    
    # Probar el scoreboard del día
    print("\n1. Probando scoreboard del día...")
    get_nba_scoreboard()
    
    # Probar el calendario para la temporada actual (2024-2025)
    print("\n2. Probando calendario de la temporada...")
    get_nba_schedule(2024)
    
    print("\nPrueba completada.")
