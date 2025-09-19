import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm

# Configuración
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

class MasterDatasetCreator:
    def __init__(self, df):
        self.df = df.copy()
        self.teams_schedule = {}
        
    def calculate_rest_days(self, team_id, game_date):
        """Calcula los días de descanso de un equipo antes de un partido"""
        if team_id not in self.teams_schedule:
            return 0
            
        previous_games = [d for d in self.teams_schedule[team_id] if d < game_date]
        if not previous_games:
            return 0
            
        last_game = max(previous_games)
        return (game_date - last_game).days - 1
    
    def update_team_schedule(self, team_id, game_date):
        """Actualiza el calendario de partidos de un equipo"""
        if team_id not in self.teams_schedule:
            self.teams_schedule[team_id] = []
        self.teams_schedule[team_id].append(game_date)
    
    def create_features(self, game):
        """Crea características adicionales para el dataset"""
        features = {}
        
        # Identificadores y contexto
        features['game_id'] = game['GAME_ID']
        features['game_date'] = game['GAME_DATE']
        features['season'] = game['SEASON']
        
        # Equipos
        features['team_id_local'] = game['TEAM_ID_HOME']
        features['team_abbreviation_local'] = game['TEAM_ABBREVIATION_HOME']
        features['team_id_visitante'] = game['TEAM_ID_AWAY']
        features['team_abbreviation_visitante'] = game['TEAM_ABBREVIATION_AWAY']
        
        # Playoffs
        features['is_playoff'] = 1 if game['SEASON_TYPE'] == 'Playoffs' else 0
        
        # Targets
        features['puntos_local'] = game['PTS_HOME']
        features['puntos_visitante'] = game['PTS_AWAY']
        
        # Días de descanso
        game_date = pd.to_datetime(game['GAME_DATE'])
        
        features['L_dias_descanso'] = self.calculate_rest_days(
            game['TEAM_ID_HOME'], game_date)
        features['V_dias_descanso'] = self.calculate_rest_days(
            game['TEAM_ID_AWAY'], game_date)
            
        # Back-to-back
        features['L_es_b2b'] = 1 if features['L_dias_descanso'] == 0 else 0
        features['V_es_b2b'] = 1 if features['V_dias_descanso'] == 0 else 0
        
        # Actualizar calendario
        self.update_team_schedule(game['TEAM_ID_HOME'], game_date)
        self.update_team_schedule(game['TEAM_ID_AWAY'], game_date)
        
        return pd.Series(features)
    
    def create_dataset(self):
        """Crea el dataset final con todas las características"""
        print("Creando dataset maestro...")
        
        # Ordenar por fecha
        self.df = self.df.sort_values('GAME_DATE')
        
        # Aplicar la creación de características a cada partido
        features_list = []
        
        for _, game in tqdm(self.df.iterrows(), total=len(self.df), 
                           desc="Procesando partidos"):
            features = self.create_features(game)
            features_list.append(features)
        
        # Crear DataFrame con todas las características
        master_df = pd.DataFrame(features_list)
        
        # Calcular características diferenciales
        master_df['diff_off_rating'] = master_df['L_off_rating_L10'] - master_df['V_def_rating_L10']
        master_df['diff_def_rating'] = master_df['V_off_rating_L10'] - master_df['L_def_rating_L10']
        master_df['diff_pace'] = master_df['L_pace_L10'] - master_df['V_pace_L10']
        master_df['diff_efg_pct'] = master_df['L_efg_pct_L10'] - master_df['V_efg_pct_L10']
        master_df['diff_descanso'] = master_df['L_dias_descanso'] - master_df['V_dias_descanso']
        
        return master_df

def main():
    # Cargar datos procesados
    processed_data_path = PROCESSED_DATA_DIR / 'nba_advanced_stats.parquet'
    if not processed_data_path.exists():
        print("Error: No se encontraron datos procesados. Ejecuta primero calculate_advanced_stats.py")
        return
    
    print(f"Cargando datos procesados desde {processed_data_path}...")
    df = pd.read_parquet(processed_data_path)
    
    # Crear dataset maestro
    creator = MasterDatasetCreator(df)
    master_df = creator.create_dataset()
    
    # Guardar dataset final
    output_path = PROCESSED_DATA_DIR / 'nba_master_dataset.parquet'
    master_df.to_parquet(output_path, index=False)
    print(f"Dataset maestro guardado en {output_path}")
    
    # Mostrar resumen
    print("\nResumen del dataset:")
    print(f"- Número de partidos: {len(master_df)}")
    print(f"- Número de características: {len(master_df.columns)}")
    print("\nPrimeras filas del dataset:")
    print(master_df.head())
    
    return master_df

if __name__ == "__main__":
    df = main()
