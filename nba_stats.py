import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

def get_advanced_team_stats(season_type='Regular Season'):
    """
    Obtiene estadísticas avanzadas de equipos de la NBA para la temporada 2024-25.
    
    Args:
        season_type (str): 'Regular Season' o 'Playoffs'.
    """
    season = '2024-25'
    print(f"Obteniendo estadísticas para la temporada {season} ({season_type})...\n")
    
    try:
        # Obtener estadísticas avanzadas de equipos
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced',
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed='Per100Possessions',
            league_id_nullable='00'  # 00 es para la NBA
        ).get_data_frames()[0]
        
        # Seleccionar y renombrar columnas relevantes (solo las solicitadas)
        columns_map = {
            'TEAM_NAME': 'Equipo',
            'DEF_RATING': 'Rating Defensivo',
            'NET_RATING': 'Net Rating',
            'PACE': 'Pace'
        }
        
        # Filtrar y ordenar columnas
        advanced_stats = team_stats[list(columns_map.keys())].rename(columns=columns_map)
        
        # Ordenar por Net Rating (de mayor a menor)
        advanced_stats = advanced_stats.sort_values('Net Rating', ascending=False).reset_index(drop=True)
        
        # Mostrar estadísticas
        tipo_temporada = "PLAYOFFS" if season_type == 'Playoffs' else "TEMPORADA REGULAR"
        print("\n" + "=" * 60)
        print(f"ESTADÍSTICAS AVANZADAS NBA 2024-25 - {tipo_temporada}")
        print("=" * 60 + "\n")
        
        # Mostrar tabla formateada
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        
        print(advanced_stats.to_string(index=False))
        
        return advanced_stats
        
    except Exception as e:
        print(f"Error al obtener los datos: {e}")
        return None

if __name__ == "__main__":
    # Mostrar estadísticas de temporada regular
    get_advanced_team_stats(season_type='Regular Season')
    
    # Mostrar estadísticas de playoffs (si están disponibles)
    print("\n" + "#" * 60)
    print("ESTADÍSTICAS DE PLAYOFFS (cuando estén disponibles)")
    print("#" * 60)
    try:
        get_advanced_team_stats(season_type='Playoffs')
    except Exception as e:
        print("\n⚠️  Los playoffs de la temporada 2024-25 aún no han comenzado o no hay datos disponibles.")
        print(f"Error: {e}")
