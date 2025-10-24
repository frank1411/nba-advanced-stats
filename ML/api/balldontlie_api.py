#!/usr/bin/env python3
"""
Módulo para interactuar con la API de balldontlie.io
Documentación: https://www.balldontlie.io/home.html#introduction
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import time

class BallDontLieAPI:
    """Cliente para la API de balldontlie.io"""
    
    BASE_URL = "https://www.balldontlie.io/api/v1"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Inicializa el cliente de la API.
        
        Args:
            cache_dir: Directorio para guardar caché de peticiones
        """
        self.session = requests.Session()
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Realiza una petición a la API con manejo de errores y reintentos.
        
        Args:
            endpoint: Punto final de la API (ej: 'games')
            params: Parámetros de la consulta
            
        Returns:
            Diccionario con la respuesta de la API
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        # Verificar caché si está habilitado
        cache_file = None
        if self.cache_dir:
            cache_key = f"{endpoint}_{str(params) if params else 'all'}.json"
            cache_file = self.cache_dir / cache_key
            if cache_file.exists():
                import json
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # Realizar la petición con reintentos
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Guardar en caché si es exitoso
                if cache_file:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(data, f, ensure_ascii=False, indent=2)
                
                return data
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Error al realizar la petición a {url}: {e}")
                time.sleep(1)  # Esperar antes de reintentar
    
    def get_games(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        days_ahead: int = 7,
        season: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Obtiene los partidos programados en un rango de fechas.
        
        Args:
            start_date: Fecha de inicio en formato 'YYYY-MM-DD'
            end_date: Fecha de fin en formato 'YYYY-MM-DD'
            days_ahead: Número de días a partir de hoy si no se especifican fechas
            season: Año de la temporada (ej: 2023 para la temporada 2023-24)
            
        Returns:
            DataFrame con los partidos programados
        """
        # Establecer fechas por defecto si no se especifican
        today = datetime.now().date()
        if not start_date:
            start_date = today.strftime('%Y-%m-%d')
        if not end_date:
            end_date = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        
        # Construir parámetros de la consulta
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'per_page': 100  # Máximo permitido por la API
        }
        
        if season:
            params['seasons[]'] = season
        
        # Realizar la petición
        data = self._make_request('games', params)
        
        # Procesar la respuesta
        games = data.get('data', [])
        
        if not games:
            return pd.DataFrame()
        
        # Convertir a DataFrame
        df = pd.DataFrame(games)
        
        # Renombrar columnas para mantener consistencia
        column_mapping = {
            'id': 'GAME_ID',
            'date': 'GAME_DATE',
            'home_team': 'HOME_TEAM',
            'visitor_team': 'AWAY_TEAM',
            'home_team_score': 'PTS_HOME',
            'visitor_team_score': 'PTS_AWAY',
            'status': 'GAME_STATUS',
            'season': 'SEASON',
            'period': 'PERIOD',
            'postseason': 'IS_PLAYOFF',
            'time': 'GAME_TIME'
        }
        
        # Aplicar el mapeo de columnas
        df = df.rename(columns={v: k for k, v in column_mapping.items() 
                              if v in df.columns})
        
        # Convertir fechas
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
        
        # Extraer información de los equipos
        if 'HOME_TEAM' in df.columns:
            df['TEAM_ABBREVIATION_HOME'] = df['HOME_TEAM'].apply(
                lambda x: x.get('abbreviation') if isinstance(x, dict) else None
            )
            df['TEAM_NAME_HOME'] = df['HOME_TEAM'].apply(
                lambda x: x.get('full_name') if isinstance(x, dict) else None
            )
            df = df.drop('HOME_TEAM', axis=1)
            
        if 'AWAY_TEAM' in df.columns:
            df['TEAM_ABBREVIATION_AWAY'] = df['AWAY_TEAM'].apply(
                lambda x: x.get('abbreviation') if isinstance(x, dict) else None
            )
            df['TEAM_NAME_AWAY'] = df['AWAY_TEAM'].apply(
                lambda x: x.get('full_name') if isinstance(x, dict) else None
            )
            df = df.drop('AWAY_TEAM', axis=1)
        
        return df
    
    def get_teams(self) -> pd.DataFrame:
        """Obtiene la lista de equipos de la NBA."""
        data = self._make_request('teams')
        return pd.DataFrame(data.get('data', []))
    
    def get_current_season(self) -> int:
        """Obtiene el año de la temporada actual."""
        today = datetime.now()
        # La temporada de la NBA generalmente comienza en octubre
        return today.year if today.month >= 10 else today.year - 1
