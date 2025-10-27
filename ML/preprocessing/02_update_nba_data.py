#!/usr/bin/env python3
"""
Script para actualizar los datos de la NBA con los partidos más recientes.

Este script se encarga de:
1. Cargar los datos existentes
2. Identificar la fecha del último partido registrado
3. Descargar solo los partidos nuevos
4. Actualizar el archivo de datos sin duplicados
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración de rutas
BASE_DIR = Path(__file__).parents[1]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "nba_games_raw.parquet"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "nba_games_final.parquet"

# Asegurar que existan los directorios
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

class NBADataUpdater:
    def __init__(self):
        """Inicializa el actualizador de datos de la NBA."""
        self.existing_data = None
        self.last_game_date = None
        
    def load_existing_data(self) -> bool:
        """Carga los datos existentes y establece la fecha del último partido."""
        try:
            if RAW_DATA_FILE.exists():
                self.existing_data = pd.read_parquet(RAW_DATA_FILE)
                
                # Convertir GAME_DATE a datetime si es necesario
                if 'GAME_DATE' in self.existing_data.columns:
                    if not pd.api.types.is_datetime64_any_dtype(self.existing_data['GAME_DATE']):
                        self.existing_data['GAME_DATE'] = pd.to_datetime(self.existing_data['GAME_DATE'])
                    
                    # Obtener la fecha del último partido
                    self.last_game_date = self.existing_data['GAME_DATE'].max()
                    logger.info(f"📅 Último partido registrado: {self.last_game_date}")
                else:
                    logger.warning("No se encontró la columna 'GAME_DATE' en los datos existentes")
                    
                return True
            else:
                logger.warning("No se encontró el archivo de datos existente. Se creará uno nuevo.")
                self.existing_data = pd.DataFrame()
                return False
                
        except Exception as e:
            logger.error(f"Error al cargar datos existentes: {e}")
            return False
    
    def fetch_new_data(self, season: str = '2025-26') -> Optional[pd.DataFrame]:
        """
        Descarga los datos de partidos nuevos desde la API de la NBA.
        
        Args:
            season: Temporada en formato 'YYYY-YY' (ej: '2025-26')
            
        Returns:
            DataFrame con los nuevos partidos o None si hay un error
        """
        try:
            # Importar usando importlib para manejar el nombre con número
            import importlib.util
            import sys
            
            # Ruta al módulo
            module_path = str(Path(__file__).parent / '01_fetch_nba_data.py')
            module_name = 'fetch_nba_data'
            
            # Cargar el módulo dinámicamente
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            fetch_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = fetch_module
            spec.loader.exec_module(fetch_module)
            
            # Obtener la clase NBADataFetcher
            NBADataFetcher = fetch_module.NBADataFetcher
            
            logger.info(f"🔍 Buscando datos para la temporada {season}...")
            fetcher = NBADataFetcher()
            
            # Descargar datos de la temporada regular
            logger.info("Buscando partidos de temporada regular...")
            regular_season = fetcher.get_season_games(season=season, season_type='Regular Season')
            
            # Descargar datos de playoffs si los hay
            logger.info("Buscando partidos de playoffs...")
            playoffs = fetcher.get_season_games(season=season, season_type='Playoffs')
            
            # Combinar datos
            if regular_season is not None and not regular_season.empty:
                if playoffs is not None and not playoffs.empty:
                    new_data = pd.concat([regular_season, playoffs], ignore_index=True)
                else:
                    new_data = regular_season
            elif playoffs is not None and not playoffs.empty:
                new_data = playoffs
            else:
                new_data = pd.DataFrame()
            
            # Filtrar por fecha si es necesario
            if not new_data.empty and self.last_game_date is not None:
                if 'GAME_DATE' in new_data.columns:
                    new_data['GAME_DATE'] = pd.to_datetime(new_data['GAME_DATE'])
                    new_data = new_data[new_data['GAME_DATE'] > self.last_game_date]
                    logger.info(f"Filtrados {len(new_data)//2} partidos nuevos después de {self.last_game_date}")
            
            if new_data is not None and not new_data.empty:
                logger.info(f"✅ Se encontraron {len(new_data)//2} partidos nuevos")
                return new_data
            else:
                logger.info("No se encontraron partidos nuevos")
                return None
                
        except Exception as e:
            logger.error(f"Error al descargar nuevos datos: {e}")
            return None
    
    def update_data(self, new_data: pd.DataFrame) -> bool:
        """
        Actualiza los datos existentes con los nuevos partidos.
        
        Args:
            new_data: DataFrame con los nuevos partidos
            
        Returns:
            bool: True si la actualización fue exitosa, False en caso contrario
        """
        try:
            if new_data is None or new_data.empty:
                logger.info("No hay datos nuevos para actualizar")
                return False
                
            # Asegurar que GAME_DATE sea datetime
            if 'GAME_DATE' in new_data.columns and not pd.api.types.is_datetime64_any_dtype(new_data['GAME_DATE']):
                new_data['GAME_DATE'] = pd.to_datetime(new_data['GAME_DATE'])
            
            # Filtrar partidos que ya existen (por si acaso)
            if not self.existing_data.empty and 'GAME_ID' in self.existing_data.columns and 'GAME_ID' in new_data.columns:
                existing_ids = set(self.existing_data['GAME_ID'].unique())
                new_data = new_data[~new_data['GAME_ID'].isin(existing_ids)]
                
                if new_data.empty:
                    logger.info("No hay partidos nuevos para agregar")
                    return False
            
            # Combinar datos
            updated_data = pd.concat([self.existing_data, new_data], ignore_index=True)
            
            # Eliminar duplicados por GAME_ID por si acaso
            if 'GAME_ID' in updated_data.columns:
                updated_data = updated_data.drop_duplicates(subset=['GAME_ID'], keep='last')
            
            # Ordenar por fecha
            if 'GAME_DATE' in updated_data.columns:
                updated_data = updated_data.sort_values('GAME_DATE')
            
            # Guardar copia de seguridad
            if RAW_DATA_FILE.exists():
                backup_file = RAW_DATA_DIR / f"nba_games_raw_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                self.existing_data.to_parquet(backup_file)
                logger.info(f"✅ Copia de seguridad guardada en: {backup_file}")
            
            # Guardar datos actualizados
            updated_data.to_parquet(RAW_DATA_FILE)
            logger.info(f"✅ Datos actualizados guardados en: {RAW_DATA_FILE}")
            logger.info(f"📊 Total de partidos en el archivo: {len(updated_data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error al actualizar los datos: {e}")
            return False

def main():
    """Función principal para ejecutar la actualización de datos."""
    logger.info("🚀 Iniciando actualización de datos de la NBA")
    
    # Inicializar el actualizador
    updater = NBADataUpdater()
    
    # Cargar datos existentes
    if not updater.load_existing_data() and not RAW_DATA_FILE.exists():
        logger.warning("No se encontraron datos existentes. Se creará un nuevo archivo con los datos más recientes.")
    
    # Obtener la temporada actual (asumiendo que estamos en la temporada 2025-26)
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # La temporada de la NBA generalmente va de octubre a junio
    if current_month >= 10:  # Si es octubre o después, estamos en la nueva temporada
        season = f"{current_year}-{str(current_year + 1)[2:]}"
    else:  # Si es antes de octubre, estamos en la segunda mitad de la temporada anterior
        season = f"{current_year - 1}-{str(current_year)[2:]}"
    
    # Descargar datos nuevos
    new_data = updater.fetch_new_data(season=season)
    
    # Actualizar datos si hay nuevos partidos
    if new_data is not None and not new_data.empty:
        success = updater.update_data(new_data)
        if success:
            logger.info("✅ Actualización completada exitosamente")
            
            # Procesar los datos actualizados (opcional, descomentar si es necesario)
            # from .process_final_dataset import process_data
            # process_data()
        else:
            logger.error("❌ Error al actualizar los datos")
    else:
        logger.info("ℹ️ No hay partidos nuevos para actualizar")
    
    logger.info("🏁 Proceso de actualización finalizado")

if __name__ == "__main__":
    main()
