import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuración
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

class AdvancedStatsCalculator:
    def __init__(self, df):
        self.df = df.copy()
        
    def calculate_team_stats(self, team_games):
        """Calcula estadísticas acumuladas para un equipo"""
        stats = team_games.sort_values('GAME_DATE')
        
        # Estadísticas básicas
        stats['GAME_NUM'] = range(1, len(stats) + 1)
        
        # Calcular estadísticas acumuladas
        numeric_cols = ['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 
                       'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
        
        for col in numeric_cols:
            if col in stats.columns:
                stats[f'CUM_{col}'] = stats[col].expanding().sum()
        
        # Calcular estadísticas avanzadas
        stats['CUM_OFF_RATING'] = (stats['CUM_PTS'] / stats['CUM_POSSESSIONS'] * 100).fillna(0)
        stats['CUM_DEF_RATING'] = (stats['CUM_OPP_PTS'] / stats['CUM_POSSESSIONS'] * 100).fillna(0)
        stats['CUM_NET_RATING'] = stats['CUM_OFF_RATING'] - stats['CUM_DEF_RATING']
        
        # Calcular estadísticas de los últimos 10 partidos
        rolling_stats = stats.rolling(window=10, min_periods=1)
        
        stats['L10_OFF_RATING'] = (rolling_stats['PTS'].sum() / 
                                  rolling_stats['POSSESSIONS'].sum() * 100).fillna(0)
        stats['L10_DEF_RATING'] = (rolling_stats['OPP_PTS'].sum() / 
                                  rolling_stats['POSSESSIONS'].sum() * 100).fillna(0)
        
        return stats
    
    def calculate_possessions(self, game):
        """Calcula las posesiones para un partido"""
        # Fórmula simplificada para posesiones
        return (
            0.5 * ((game['FGA'] + 0.4 * game['FTA'] - 1.07 * 
                   (game['OREB'] / (game['OREB'] + game['DREB_OPP'])) * 
                   (game['FGA'] - game['FGM']) + game['TOV']) + 
                   (game['OPP_FGA'] + 0.4 * game['OPP_FTA'] - 1.07 * 
                   (game['OPP_OREB'] / (game['OPP_OREB'] + game['DREB'])) * 
                   (game['OPP_FGA'] - game['OPP_FGM']) + game['OPP_TOV']))
        )
    
    def process_games(self):
        """Procesa todos los partidos para calcular estadísticas avanzadas"""
        print("Procesando estadísticas avanzadas...")
        
        # Crear copia del DataFrame
        df = self.df.copy()
        
        # Calcular posesiones para cada partido
        df['POSSESSIONS'] = df.apply(self.calculate_possessions, axis=1)
        
        # Lista para almacenar datos procesados
        processed_games = []
        
        # Procesar cada equipo en cada temporada
        for (team_id, season), team_games in tqdm(
            df.groupby(['TEAM_ID', 'SEASON']), 
            desc="Procesando equipos"
        ):
            # Calcular estadísticas del equipo
            team_stats = self.calculate_team_stats(team_games)
            processed_games.append(team_stats)
        
        # Combinar todos los datos
        final_df = pd.concat(processed_games, ignore_index=True)
        
        return final_df

def main():
    # Cargar datos en bruto
    raw_data_path = RAW_DATA_DIR / 'nba_games_raw.parquet'
    if not raw_data_path.exists():
        print("Error: No se encontraron datos en bruto. Ejecuta primero fetch_nba_data.py")
        return
    
    print(f"Cargando datos desde {raw_data_path}...")
    df = pd.read_parquet(raw_data_path)
    
    # Procesar estadísticas avanzadas
    calculator = AdvancedStatsCalculator(df)
    processed_df = calculator.process_games()
    
    # Guardar datos procesados
    output_path = PROCESSED_DATA_DIR / 'nba_advanced_stats.parquet'
    processed_df.to_parquet(output_path, index=False)
    print(f"Datos procesados guardados en {output_path}")
    
    return processed_df

if __name__ == "__main__":
    df = main()
    if df is not None:
        print(f"Proceso completado. Se procesaron {len(df)} partidos.")
