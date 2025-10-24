from flask import Flask, jsonify, request
from nba_api.stats.endpoints import leaguegamefinder, commonteamroster
from nba_api.stats.static import teams
from datetime import datetime, timedelta, date
import pandas as pd
import logging
import os

# Configuración básica
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nba_api_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Cache para almacenar datos de equipos
try:
    nba_teams = {team['abbreviation']: team for team in teams.get_teams()}
    logger.info(f"Cargados {len(nba_teams)} equipos de la NBA")
except Exception as e:
    logger.error(f"Error al cargar equipos de la NBA: {e}")
    nba_teams = {}

def process_games(games_df):
    """Procesa los partidos para un formato más limpio."""
    if games_df.empty:
        return []
    
    processed = []
    for _, game in games_df.iterrows():
        try:
            # Extraer información del partido
            matchup = game['MATCHUP']
            parts = matchup.split()
            
            if len(parts) >= 3 and parts[1] in ['@', 'vs.']:
                home_team = parts[2] if parts[1] == '@' else parts[0]
                away_team = parts[0] if parts[1] == '@' else parts[2]
                
                # Obtener nombre completo de los equipos
                home_team_full = nba_teams.get(home_team, {}).get('full_name', home_team)
                away_team_full = nba_teams.get(away_team, {}).get('full_name', away_team)
                
                processed.append({
                    'GAME_ID': game['GAME_ID'],
                    'GAME_DATE': game['GAME_DATE'],
                    'TEAM_ABBREVIATION_HOME': home_team,
                    'TEAM_NAME_HOME': home_team_full,
                    'TEAM_ABBREVIATION_AWAY': away_team,
                    'TEAM_NAME_AWAY': away_team_full,
                    'SEASON': str(game.get('SEASON_ID', '22025'))[1:5],  # 22025 = temporada 2024-25
                    'SEASON_TYPE': 'Playoffs' if 'Playoffs' in str(game.get('SEASON_ID', '')) else 'Regular',
                    'ARENA_NAME': nba_teams.get(home_team, {}).get('arena', 'Unknown Arena'),
                    'CITY': nba_teams.get(home_team, {}).get('city', 'Unknown City')
                })
        except Exception as e:
            logger.error(f"Error procesando partido {game.get('GAME_ID', '')}: {e}")
            continue
    
    return processed

@app.route('/api/upcoming-games', methods=['GET'])
def get_upcoming_games():
    """Endpoint para obtener partidos programados usando los endpoints disponibles."""
    try:
        days_ahead = int(request.args.get('days_ahead', 7))
        season = request.args.get('season', '2025-26')
        
        # Obtener el año de inicio de la temporada
        try:
            start_year = int(season.split('-')[0])
        except (ValueError, IndexError):
            # Si no se puede extraer el año, usar el año actual
            start_year = datetime.now().year
            
        # Definir el rango de fechas de la temporada (octubre a junio del año siguiente)
        season_start = date(start_year, 10, 1)  # La temporada suele empezar en octubre
        season_end = date(start_year + 1, 6, 30)  # Y terminar en junio del año siguiente
        
        logger.info(f"Buscando partidos para los próximos {days_ahead} días...")
        
        # Obtener fechas
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # Asegurarse de que no nos salgamos de la temporada actual
        if today < season_start:
            today = season_start
            logger.info(f"Ajustando fecha de inicio al inicio de la temporada: {today}")
            
        if end_date > season_end:
            end_date = season_end
            logger.info(f"Ajustando fecha de fin al final de la temporada: {end_date}")
            
        logger.info(f"Temporada {season}: {season_start} a {season_end}")
        
        logger.info(f"Rango de fechas: {today.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # Importar las bibliotecas necesarias
            from nba_api.stats.endpoints import scoreboardv2, leaguegamefinder
            from nba_api.stats.static import teams
            
            # Obtener la lista de equipos para mapear IDs a abreviaturas
            nba_teams = teams.get_teams()
            team_id_to_abbr = {team['id']: team['abbreviation'] for team in nba_teams}
            
            logger.info(f"Equipos de la NBA cargados: {len(nba_teams)} equipos encontrados")
            
            # Lista para almacenar todos los partidos
            all_games = []
            
            # Intentar obtener el scoreboard para el rango de fechas
            try:
                # Obtener la lista de fechas en el rango
                date_range = [today + timedelta(days=x) for x in range((end_date - today).days + 1)]
                
                for game_date in date_range:
                    try:
                        logger.info(f"Buscando partidos para la fecha: {game_date.strftime('%Y-%m-%d')}...")
                        
                        # Formatear la fecha para la API (MM/DD/YYYY)
                        formatted_date = game_date.strftime('%m/%d/%Y')
                        
                        # Crear instancia de ScoreboardV2 con la fecha específica
                        sb = scoreboardv2.ScoreboardV2(
                            game_date=formatted_date,
                            league_id='00',  # NBA
                            day_offset='0',  # Sin desplazamiento
                            # Usar el año de la temporada actual para el scoreboard
                            season=season,
                            season_type='Regular'  # O 'Playoffs' si es necesario
                        )
                        
                        # Obtener los datos del scoreboard
                        scoreboard_data = sb.get_dict()
                        
                        # Procesar los partidos del scoreboard
                        if 'GameHeader' in scoreboard_data and scoreboard_data['GameHeader']:
                            for game in scoreboard_data['GameHeader']:
                                try:
                                    game_date_est = datetime.strptime(game['GAME_DATE_EST'].split('T')[0], '%Y-%m-%d').date()
                                    
                                    # Solo incluir partidos dentro del rango de fechas
                                    if today <= game_date_est <= end_date:
                                        game_info = {
                                            'GAME_ID': game.get('GAME_ID', ''),
                                            'GAME_DATE': game_date_est.strftime('%Y-%m-%d'),
                                            'HOME_TEAM_ID': game.get('HOME_TEAM_ID', ''),
                                            'HOME_TEAM_ABBREVIATION': team_id_to_abbr.get(int(game.get('HOME_TEAM_ID', 0)), 'UNK'),
                                            'VISITOR_TEAM_ID': game.get('VISITOR_TEAM_ID', ''),
                                            'VISITOR_TEAM_ABBREVIATION': team_id_to_abbr.get(int(game.get('VISITOR_TEAM_ID', 0)), 'UNK'),
                                            'GAME_STATUS_TEXT': game.get('GAME_STATUS_TEXT', 'Scheduled'),
                                            'SEASON': game.get('SEASON', season),
                                            'SEASON_TYPE': game.get('SEASON_TYPE', 'Regular')
                                        }
                                        all_games.append(game_info)
                                        logger.info(f"Partido encontrado: {game_info['VISITOR_TEAM_ABBREVIATION']} @ {game_info['HOME_TEAM_ABBREVIATION']} - {game_info['GAME_DATE']}")
                                except Exception as game_error:
                                    logger.error(f"Error al procesar partido: {str(game_error)}")
                    
                    except Exception as date_error:
                        logger.error(f"Error al obtener partidos para {formatted_date}: {str(date_error)}")
                    
                    logger.info(f"Se encontraron {len(all_games)} partidos en el scoreboard")
                
            except Exception as sb_error:
                logger.error(f"Error al obtener el scoreboardv2: {str(sb_error)}")
            
            # Si no se encontraron partidos, intentar con LeagueGameFinder
            if not all_games:
                logger.warning("No se encontraron partidos con el scoreboard. Intentando con LeagueGameFinder...")
                
                try:
                    # Crear un rango de años para la búsqueda (temporada actual)
                    season_start_year = int(season.split('-')[0])
                    season_end_year = season_start_year + 1
                    
                    # Buscar partidos para la temporada actual
                    logger.info(f"Buscando partidos para la temporada {season}...")
                    game_finder = leaguegamefinder.LeagueGameFinder(
                        league_id_nullable='00',  # NBA
                        season_type_nullable='Regular',  # Regular season
                        season=season,  # Especificar la temporada
                        date_from_nullable=today.strftime('%m/%d/%Y'),
                        date_to_nullable=end_date.strftime('%m/%d/%Y'),
                        timeout=30
                    )
                    
                    # Obtener datos
                    logger.info("Obteniendo DataFrames de partidos...")
                    games_df = game_finder.get_data_frames()[0]
                    
                    if not games_df.empty:
                        # Filtrar por fechas futuras
                        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE']).dt.date
                        games_df = games_df[games_df['GAME_DATE'] >= today]
                        
                        # Convertir a formato de diccionario
                        for _, row in games_df.iterrows():
                            game_info = {
                                'GAME_ID': row.get('GAME_ID', ''),
                                'GAME_DATE': row['GAME_DATE'].strftime('%Y-%m-%d'),
                                'HOME_TEAM_ID': row.get('TEAM_ID', ''),
                                'HOME_TEAM_ABBREVIATION': row.get('MATCHUP', '').split(' ')[0],
                                'VISITOR_TEAM_ID': row.get('OPPONENT_ID', ''),
                                'VISITOR_TEAM_ABBREVIATION': row.get('OPPONENT_TEAM_ABBREVIATION', ''),
                                'SEASON': row.get('SEASON_ID', season),
                                'SEASON_TYPE': 'Regular' if 'Regular' in str(row.get('SEASON_TYPE', '')) else 'Playoffs'
                            }
                            all_games.append(game_info)
                        
                        logger.info(f"Se encontraron {len(games_df)} partidos con LeagueGameFinder")
                
                except Exception as league_error:
                    logger.error(f"Error en LeagueGameFinder: {str(league_error)}")
            
            # Si aún no hay partidos, devolver un mensaje informativo
            if not all_games:
                logger.warning("No se encontraron partidos programados con ningún método")
                return jsonify({
                    'status': 'success',
                    'data': [],
                    'count': 0,
                    'message': 'No se encontraron partidos programados para las fechas solicitadas',
                    'debug': {
                        'date_range': f"{today.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                        'season': season,
                        'teams_loaded': len(nba_teams),
                        'method': 'ScoreboardV2 y LeagueGameFinder'
                    }
                })
            
            # Mostrar información sobre los datos disponibles
            if not games_df.empty:
                # Convertir fechas a datetime
                games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE']).dt.date
                
                # Filtrar por fechas solicitadas
                filtered_df = games_df[
                    (games_df['GAME_DATE'] >= today) & 
                    (games_df['GAME_DATE'] <= end_date)
                ]
                
                logger.info(f"Partidos en el rango de fechas: {len(filtered_df)}")
                
                # Mostrar las primeras filas para depuración
                if not filtered_df.empty:
                    logger.info("Primeros partidos en el rango:")
                    for _, row in filtered_df.head(3).iterrows():
                        logger.info(f"- {row.get('GAME_DATE', 'Sin fecha')}: {row.get('MATCHUP', 'Sin equipos')}")
                
                # Procesar partidos filtrados
                processed_games = process_games(filtered_df)
            else:
                # Si no hay datos, intentar con un rango más amplio
                logger.warning("No se encontraron partidos. Intentando con un rango de fechas más amplio...")
                
                # Obtener partidos de la temporada actual sin filtrar por fecha
                game_finder = leaguegamefinder.LeagueGameFinder(
                    league_id_nullable='00',
                    season_type_nullable='Regular',
                    season='2025-26',
                    timeout=30
                )
                
                games_df = game_finder.get_data_frames()[0]
                if not games_df.empty:
                    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE']).dt.date
                    processed_games = process_games(games_df)
                    logger.info(f"Se encontraron {len(processed_games)} partidos en la temporada 2025-26")
                else:
                    processed_games = []
                    logger.warning("No se encontraron partidos para la temporada 2025-26")
            
            logger.info(f"Se procesaron {len(processed_games)} partidos")
            
            return jsonify({
                'status': 'success',
                'data': processed_games,
                'count': len(processed_games),
                'debug': {
                    'date_range': f"{today.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    'raw_count': len(games_df) if 'games_df' in locals() else 0,
                    'processed_count': len(processed_games)
                }
            })
            
        except Exception as api_error:
            logger.error(f"Error en la API de la NBA: {str(api_error)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f'Error en la API de la NBA: {str(api_error)}',
                'debug': {
                    'date_range': f"{today.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    'error_type': type(api_error).__name__
                }
            }), 500
        
    except Exception as e:
        logger.error(f"Error al obtener partidos: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Endpoint para obtener la lista de equipos de la NBA."""
    try:
        return jsonify({
            'status': 'success',
            'data': [{'id': team['id'], 'abbreviation': team['abbreviation'], 'full_name': team['full_name']} 
                    for team in teams.get_teams()]
        })
    except Exception as e:
        logger.error(f"Error al obtener equipos: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud del servidor."""
    return jsonify({'status': 'healthy', 'time': datetime.now().isoformat()}), 200

if __name__ == '__main__':
    import argparse
    
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description='Servidor de la API de la NBA')
    parser.add_argument('--port', type=int, default=5001, help='Puerto para el servidor (predeterminado: 5001)')
    args = parser.parse_args()
    
    logger.info(f"Iniciando servidor NBA API en el puerto {args.port}...")
    app.run(host='0.0.0.0', port=args.port, debug=True)
