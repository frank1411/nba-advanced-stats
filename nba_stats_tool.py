import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

def get_team_stats(season='2024-25', season_type='Regular Season'):
    """
    Obtiene estadísticas avanzadas de equipos de la NBA para la temporada 2024-25.
    
    Args:
        season (str): Temporada en formato 'YYYY-YY' (ej: '2024-25')
        season_type (str): 'Regular Season' o 'Playoffs'
        
    Returns:
        DataFrame: Estadísticas de los equipos
    """
    print(f"Obteniendo estadísticas para la temporada {season} ({season_type})...")
    
    try:
        # 1. Obtener estadísticas generales
        print("1/4 Obteniendo estadísticas generales...")
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Base',
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed='PerGame',
            league_id_nullable='00',
            timeout=30
        ).get_data_frames()[0]
        
        if team_stats.empty:
            print("No se recibieron datos de la API para estadísticas generales")
            return pd.DataFrame()
        
        # 2. Obtener estadísticas avanzadas
        print("2/4 Obteniendo estadísticas avanzadas...")
        adv_stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced',
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed='Per100Possessions',
            league_id_nullable='00',
            timeout=30
        ).get_data_frames()[0]
        
        # 3. Combinar estadísticas generales y avanzadas
        result = team_stats.merge(
            adv_stats[['TEAM_NAME', 'OFF_RATING', 'DEF_RATING', 'PACE']],
            on='TEAM_NAME',
            how='left'
        )
        
        # 4. Obtener estadísticas de local
        print("3/4 Obteniendo estadísticas de local...")
        try:
            home_stats = leaguedashteamstats.LeagueDashTeamStats(
                measure_type_detailed_defense='Base',
                season=season,
                season_type_all_star=season_type,
                location_nullable='Home',
                per_mode_detailed='PerGame',
                league_id_nullable='00',
                timeout=30
            ).get_data_frames()[0][['TEAM_NAME', 'W', 'GP']]
            
            # Renombrar columnas de local
            home_stats = home_stats.rename(columns={
                'W': 'victoriasCasa',
                'GP': 'juegosCasa'
            })
            
            # Calcular estadísticas de visitante
            result = result.merge(home_stats, on='TEAM_NAME', how='left')
            result['juegosFuera'] = result['GP'] - result['juegosCasa'].fillna(0)
            result['victoriasFuera'] = result['W'] - result['victoriasCasa'].fillna(0)
            
            # Asegurar que no haya valores negativos
            result['juegosFuera'] = result['juegosFuera'].clip(lower=0)
            result['victoriasFuera'] = result['victoriasFuera'].clip(lower=0)
            
            # Convertir a enteros
            for col in ['juegosCasa', 'victoriasCasa', 'juegosFuera', 'victoriasFuera']:
                result[col] = result[col].fillna(0).astype(int)
                
        except Exception as e:
            print(f"⚠️  No se pudieron obtener estadísticas de local/visitante: {e}")
            print("No se incluirán estadísticas de local/visitante en los resultados.")
        
        # 5. Seleccionar y renombrar columnas finales
        result = result.rename(columns={
            'TEAM_NAME': 'team',
            'OFF_RATING': 'offRating',
            'DEF_RATING': 'defRating',
            'PACE': 'pace',
            'PTS': 'puntosPromedio',
            'OPP_PTS': 'puntosPermitidos'
        })
        
        # Calcular puntos permitidos si no están disponibles
        if 'puntosPermitidos' not in result.columns or result['puntosPermitidos'].isnull().all():
            result['puntosPermitidos'] = result['puntosPromedio'] - result['PLUS_MINUS']
        
        # Columnas a mostrar (eliminamos GP, W, L, NET_RATING)
        final_columns = [
            'team', 'offRating', 'defRating', 'pace',
            'puntosPromedio', 'puntosPermitidos', 'juegosCasa',
            'victoriasCasa', 'juegosFuera', 'victoriasFuera'
        ]
        
        # Filtrar solo las columnas que existen
        final_columns = [col for col in final_columns if col in result.columns]
        result = result[final_columns]
        
        # Ordenar por ofRating (de mayor a menor)
        result = result.sort_values('offRating', ascending=False).reset_index(drop=True)
        
        # Formatear números
        float_cols = ['offRating', 'defRating', 'pace', 'puntosPromedio', 'puntosPermitidos']
        for col in float_cols:
            if col in result.columns:
                result[col] = result[col].round(1)
        
        return result
        
    except Exception as e:
        print(f"❌ Error al obtener los datos: {e}")
        return pd.DataFrame()

def print_stats(stats, title):
    """Muestra las estadísticas formateadas"""
    if stats is None or stats.empty:
        print("No se encontraron datos.")
        return
        
    print("\n" + "=" * 120)
    print(f"{title.upper()}")
    print("=" * 120)
    
    # Configurar opciones de visualización
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    
    # Mostrar la tabla
    print(stats.to_string(index=False))
    print("=" * 120 + "\n")

if __name__ == "__main__":
    # Obtener estadísticas de temporada regular
    regular_season = get_team_stats(season_type='Regular Season')
    print_stats(regular_season, "ESTADÍSTICAS NBA 2024-25 - TEMPORADA REGULAR")
    
    # Intentar obtener estadísticas de playoffs
    try:
        playoffs = get_team_stats(season_type='Playoffs')
        if not playoffs.empty:
            print_stats(playoffs, "ESTADÍSTICAS NBA 2024-25 - PLAYOFFS")
        else:
            print("\nℹ️  No hay datos disponibles para los playoffs de la temporada actual.")
    except Exception as e:
        print(f"\n⚠️  No se pudieron obtener los datos de playoffs: {e}")
