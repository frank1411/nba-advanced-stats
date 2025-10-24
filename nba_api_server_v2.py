from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime, timedelta, date
import logging
import os

# Configuración básica
app = Flask(__name__)
CORS(app)  # Habilita CORS para todos los dominios

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración de la API
BALLDONTLIE_API = 'https://www.balldontlie.io/api/v1'

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud del servidor."""
    return jsonify({
        'status': 'healthy',
        'service': 'NBA API Server',
        'version': '2.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'health': 'GET /api/health',
            'today_games': 'GET /api/today-games',
            'games_by_date': 'GET /api/games?date=YYYY-MM-DD',
            'upcoming_games': 'GET /api/upcoming-games?days_ahead=N'
        }
    })

@app.route('/api/today-games', methods=['GET'])
def get_today_games():
    """Endpoint para obtener los partidos de la NBA programados para hoy."""
    try:
        # Usar la fecha actual
        today = datetime.now().strftime('%Y-%m-%d')
        return get_games_by_date(today)
        
    except Exception as e:
        logger.error(f"Error en la API: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error al obtener los partidos de hoy: {str(e)}"
        }), 500

@app.route('/api/games', methods=['GET'])
def get_games():
    """Endpoint para obtener partidos de la NBA para una fecha específica."""
    try:
        date_str = request.args.get('date')
        if not date_str:
            return jsonify({
                'status': 'error',
                'message': 'Se requiere el parámetro date (formato: YYYY-MM-DD)'
            }), 400
            
        return get_games_by_date(date_str)
        
    except Exception as e:
        logger.error(f"Error en la API: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def get_games_by_date(date_str):
    """Función auxiliar para obtener partidos por fecha."""
    try:
        # Validar el formato de la fecha
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            current_date = datetime.now().date()
            
            # Verificar si la fecha es futura
            if target_date > current_date:
                return jsonify({
                    'status': 'success',
                    'date': date_str,
                    'games': [],
                    'count': 0,
                    'message': 'La fecha solicitada es futura. No hay datos disponibles aún.',
                    'is_future_date': True
                })
                
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Formato de fecha inválido. Use YYYY-MM-DD'
            }), 400
        
        logger.info(f"Solicitando partidos para la fecha: {date_str}")
        
        try:
            # Hacer la petición a la API de balldontlie
            url = f"{BALLDONTLIE_API}/games?dates[]={date_str}&per_page=100"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 404:
                return jsonify({
                    'status': 'success',
                    'date': date_str,
                    'games': [],
                    'count': 0,
                    'message': 'No se encontraron partidos para la fecha solicitada.'
                })
            raise
        
        # Procesar los partidos
        games = []
        for game in data.get('data', []):
            try:
                # Verificar que el partido sea de la NBA
                if game.get('home_team', {}).get('conference') not in ['East', 'West']:
                    continue
                    
                game_info = {
                    'gameId': game.get('id'),
                    'date': game.get('date'),
                    'time': game.get('time') or 'TBD',
                    'status': game.get('status'),
                    'period': game.get('period'),
                    'season': game.get('season'),
                    'postseason': game.get('postseason', False),
                    'homeTeam': {
                        'id': game.get('home_team', {}).get('id'),
                        'name': game.get('home_team', {}).get('full_name'),
                        'abbreviation': game.get('home_team', {}).get('abbreviation'),
                        'score': game.get('home_team_score', 0),
                        'conference': game.get('home_team', {}).get('conference'),
                        'division': game.get('home_team', {}).get('division')
                    },
                    'awayTeam': {
                        'id': game.get('visitor_team', {}).get('id'),
                        'name': game.get('visitor_team', {}).get('full_name'),
                        'abbreviation': game.get('visitor_team', {}).get('abbreviation'),
                        'score': game.get('visitor_team_score', 0),
                        'conference': game.get('visitor_team', {}).get('conference'),
                        'division': game.get('visitor_team', {}).get('division')
                    }
                }
                
                # Agregar información adicional si el partido ya comenzó
                if game_info['status'] != 'Scheduled':
                    game_info.update({
                        'time_remaining': game.get('time_remaining'),
                        'status_text': f"Q{game_info['period']} - {game_info['time_remaining']}" if game_info.get('time_remaining') else game_info['status']
                    })
                
                games.append(game_info)
                
            except Exception as e:
                logger.error(f"Error al procesar partido {game.get('id')}: {str(e)}")
                continue
        
        logger.info(f"Se encontraron {len(games)} partidos para la fecha {date_str}")
        
        return jsonify({
            'status': 'success',
            'date': date_str,
            'games': games,
            'count': len(games),
            'metadata': {
                'total_pages': data.get('meta', {}).get('total_pages', 1),
                'current_page': data.get('meta', {}).get('current_page', 1),
                'total_count': data.get('meta', {}).get('total_count', len(games))
            }
        })
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al conectar con la API: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error al conectar con el servidor de datos: {str(e)}"
        }), 502
    except Exception as e:
        logger.error(f"Error en get_games_by_date: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Servidor de la API de la NBA')
    parser.add_argument('--port', type=int, default=5000, help='Puerto para el servidor (predeterminado: 5000)')
    args = parser.parse_args()
    
    logger.info(f"🏀 Servidor NBA API v2.0 iniciado en el puerto {args.port}")
    logger.info(f"  Usando balldontlie.io API")
    logger.info(f"Endpoints disponibles:")
    logger.info(f"• GET http://localhost:{args.port}/api/health - Verificar estado del servidor")
    logger.info(f"• GET http://localhost:{args.port}/api/today-games - Partidos de hoy")
    logger.info(f"• GET http://localhost:{args.port}/api/games?date=YYYY-MM-DD - Partidos por fecha")
    logger.info(f"• GET http://localhost:{args.port}/api/upcoming-games?days_ahead=N - Próximos partidos (1-30 días)")
    logger.info(f"\n  Ejemplos:")
    logger.info(f"  • Obtener partidos de hoy: curl http://localhost:{args.port}/api/today-games")
    logger.info(f"  • Próximos 5 días: curl http://localhost:{args.port}/api/upcoming-games?days_ahead=5")
    logger.info(f"  • Partidos por fecha: curl http://localhost:{args.port}/api/games?date=2025-12-25")
    
    app.run(host='0.0.0.0', port=args.port, debug=True)
