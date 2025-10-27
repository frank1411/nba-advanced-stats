#!/usr/bin/env python3
"""
Script para analizar datos de la API de la NBA y verificar valores nulos por temporada.
"""
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import time
from datetime import datetime

class NBADataAnalyzer:
    """Clase para analizar datos de la API de la NBA."""
    
    def __init__(self):
        self.base_url = "https://stats.nba.com/stats/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'application/json, text/plain, */*',
            'x-nba-stats-origin': 'stats',
            'x-nba-stats-token': 'true',
            'Referer': 'https://www.nba.com/',
            'Connection': 'keep-alive',
        }
        self.seasons = [
            '2018-19', '2019-20', '2020-21', 
            '2021-22', '2022-23', '2023-24', '2024-25', '2025-26'
        ]
    
    def make_api_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Realiza una solicitud a la API de la NBA con reintentos."""
        url = f"{self.base_url}{endpoint}"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Intento {attempt + 1} fallado: {e}")
                if attempt == max_retries - 1:
                    print(f"Error después de {max_retries} intentos: {e}")
                    return None
                time.sleep(2 ** attempt)  # Espera exponencial
        return None
    
    def get_season_games(self, season: str, season_type: str = 'Regular Season') -> Optional[pd.DataFrame]:
        """Obtiene los partidos de una temporada específica."""
        print(f"\n{'='*50}")
        print(f"Obteniendo datos para la temporada {season} ({season_type})...")
        
        params = {
            'LeagueID': '00',
            'Season': season,
            'SeasonType': season_type,
            'PlayerOrTeam': 'T',  # Team stats
            'Sorter': 'DATE',
            'Direction': 'DESC',
        }
        
        data = self.make_api_request('leaguegamefinder', params)
        if not data or 'resultSets' not in data or not data['resultSets']:
            print(f"No se pudieron obtener datos para {season} - {season_type}")
            return None
        
        # Convertir a DataFrame
        games = data['resultSets'][0]['rowSet']
        headers = data['resultSets'][0]['headers']
        df = pd.DataFrame(games, columns=headers)
        
        # Agregar columnas de temporada y tipo de temporada
        df['SEASON'] = season
        df['SEASON_TYPE'] = season_type
        
        return df
    
    def analyze_nulls_by_season(self):
        """Analiza los valores nulos por temporada."""
        all_games = []
        
        # Obtener datos para todas las temporadas
        for season in self.seasons:
            # Obtener partidos de temporada regular
            reg_season = self.get_season_games(season, 'Regular Season')
            if reg_season is not None:
                all_games.append(reg_season)
            
            # Obtener partidos de playoffs (opcional)
            # playoffs = self.get_season_games(season, 'Playoffs')
            # if playoffs is not None:
            #     all_games.append(playoffs)
            
            # Pequeña pausa entre solicitudes
            time.sleep(1)
        
        if not all_games:
            print("No se pudieron obtener datos de ninguna temporada.")
            return
        
        # Combinar todos los datos
        df = pd.concat(all_games, ignore_index=True)
        
        # Mostrar información general
        print("\n" + "="*50)
        print("RESUMEN DE DATOS POR TEMPORADA")
        print("="*50)
        
        # Total de partidos por temporada
        print("\nTOTAL DE PARTIDOS POR TEMPORADA:")
        print(df.groupby('SEASON').size())
        
        # Columnas con valores nulos
        null_columns = df.columns[df.isnull().any()].tolist()
        print("\nCOLUMNAS CON VALORES NULOS:", null_columns)
        
        # Análisis de nulos por temporada
        print("\nANÁLISIS DE VALORES NULOS POR TEMPORADA:")
        for season in sorted(df['SEASON'].unique()):
            season_data = df[df['SEASON'] == season]
            total_games = len(season_data)
            
            print(f"\nTEMPORADA: {season} (Total partidos: {total_games})")
            
            # Porcentaje de nulos por columna
            null_percent = (season_data.isnull().sum() / total_games * 100).round(2)
            null_percent = null_percent[null_percent > 0].sort_values(ascending=False)
            
            if not null_percent.empty:
                print("\nPorcentaje de valores nulos por columna:")
                for col, percent in null_percent.items():
                    print(f"  - {col}: {percent}%")
            else:
                print("No se encontraron valores nulos en esta temporada.")
        
        # Guardar datos para análisis posterior
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"nba_games_analysis_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDatos guardados en: {output_file}")


def main():
    """Función principal."""
    analyzer = NBADataAnalyzer()
    analyzer.analyze_nulls_by_season()


if __name__ == "__main__":
    main()
