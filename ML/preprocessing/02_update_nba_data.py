#!/usr/bin/env python3
"""
Script para actualizar los datos de la NBA con los partidos más recientes.

Este script se encarga de:
1. Cargar los datos existentes
2. Identificar la fecha del último partido registrado
3. Descargar solo los partidos nuevos
4. Actualizar el archivo de datos sin duplicados
5. Calcular RESULT_OU_* y CONFIDENCE_OU_* para partidos con predicción + línea + resultado
"""
import os
import numpy as np
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

from ML.db import NBADatabase

class NBADataUpdater:
    def __init__(self):
        """Inicializa el actualizador de datos de la NBA."""
        self.existing_data = None
        self.last_game_date = None
        
    def load_existing_data(self) -> bool:
        """Carga los datos existentes desde SQLite y establece la fecha del último partido."""
        try:
            with NBADatabase() as db:
                if db.get_game_count("raw") > 0:
                    self.existing_data = db.get_raw_games()

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
                    logger.warning("No hay datos en la base de datos. Se crearán nuevos registros.")
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
            
            # Obtener estadísticas de equipo primero
            logger.info("📊 Obteniendo estadísticas de equipos...")
            team_stats = fetcher.get_team_stats(season, 'Regular Season')
            
            # Procesar temporada regular
            logger.info("🏀 Buscando partidos de temporada regular...")
            regular_games = fetcher.get_season_games(season=season, season_type='Regular Season')
            
            # Procesar playoffs si los hay
            logger.info("🏆 Buscando partidos de playoffs...")
            playoff_games = fetcher.get_season_games(season=season, season_type='Playoffs')
            
            # Buscar todos los partidos para capturar Copa NBA y otros
            logger.info("🏆 Buscando partidos de Copa NBA y otros...")
            all_types_games = fetcher.get_season_games(season=season, season_type='ALL')
            
            # Combinar partidos
            all_games = []
            
            for games, season_type in [(regular_games, 'Regular Season'), (playoff_games, 'Playoffs'), (all_types_games, 'NBA Cup/Other')]:
                if games is not None and not games.empty:
                    # Filtrar partidos nuevos
                    if 'GAME_DATE' in games.columns:
                        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
                        if self.last_game_date is not None:
                            games = games[games['GAME_DATE'] > self.last_game_date]
                    
                    if not games.empty:
                        # Procesar partidos con estadísticas de equipo
                        logger.info(f"🔄 Procesando {len(games)/2} partidos de {season_type}...")
                        processed_games = fetcher.process_games_data(games, team_stats)
                        
                        if processed_games is not None and not processed_games.empty:
                            # Calcular días de descanso y B2B
                            processed_games = fetcher.calculate_rest_days_and_b2b(processed_games)
                            
                            # Calcular distancias de viaje
                            processed_games = fetcher.calculate_travel_distances(processed_games)
                            
                            all_games.append(processed_games)
                            logger.info(f"✅ Procesados {len(processed_games)} partidos de {season_type}")
            
            # Combinar todos los partidos
            if all_games:
                new_data = pd.concat(all_games, ignore_index=True)
                logger.info(f"✅ Total de equipos jugando: {len(new_data)}")
                return new_data
            else:
                logger.info("ℹ️ No se encontraron partidos nuevos")
                return pd.DataFrame()
                
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

            # Guardar datos actualizados en SQLite
            with NBADatabase() as db:
                db.create_tables()
                db.insert_raw_games(updated_data, if_exists="replace")

            logger.info(f"✅ Datos actualizados guardados en SQLite: {db.db_path}")
            logger.info(f"📊 Total de partidos: {len(updated_data)}")

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
    if not updater.load_existing_data():
        logger.warning("No se encontraron datos existentes. Se crearán nuevos registros con los datos más recientes.")
    
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

    # Calcular RESULT_OU_* y CONFIDENCE_OU_* para partidos con predicción + línea + resultado
    update_betting_results()

    logger.info("🏁 Proceso de actualización finalizado")


def update_betting_results():
    """
    Calcula RESULT_OU_* y CONFIDENCE_OU_* para partidos que ahora tienen:
    - Predicción del modelo (PREDICTION_OU_*)
    - Línea de la casa (LINE_OU_*)
    - Resultado real (PTS_HOME, PTS_AWAY)

    Se ejecuta después de actualizar datos para evaluar automáticamente
    los pronósticos de partidos ya jugados.
    """
    logger.info("=" * 50)
    logger.info("Calculando resultados de betting...")

    with NBADatabase() as db:
        df = db.get_final_games()

    # Necesitamos predicción + línea + resultado real
    has_pred = df['PREDICTION_OU_TOTAL'].notna()
    has_line = df['LINE_OU_TOTAL'].notna()
    has_result = df['PTS_HOME'].notna() & (df['PTS_HOME'] > 0)

    valid = has_pred & has_line & has_result

    if valid.sum() == 0:
        logger.info("No hay partidos con predicción + línea + resultado para evaluar")
        return

    logger.info(f"Evaluando {valid.sum()} partidos con datos completos...")

    for target in ["TOTAL", "HOME", "AWAY"]:
        line_col = f"LINE_OU_{target}"
        pred_col = f"PREDICTION_OU_{target}"

        if target == "TOTAL":
            actual = df["PTS_HOME"] + df["PTS_AWAY"]
        elif target == "HOME":
            actual = df["PTS_HOME"]
        else:
            actual = df["PTS_AWAY"]

        valid_target = valid & df[line_col].notna() & df[pred_col].notna()
        if valid_target.sum() == 0:
            continue

        predicted_over = df.loc[valid_target, pred_col] > df.loc[valid_target, line_col]
        actual_over = actual.loc[valid_target] > df.loc[valid_target, line_col]

        df.loc[valid_target, f"RESULT_OU_{target}"] = np.where(
            predicted_over == actual_over, "Hit", "Miss"
        )

        # Confianza basada en el margen |predicción - línea|
        margin = (df.loc[valid_target, pred_col] - df.loc[valid_target, line_col]).abs()
        k = 0.5
        x0 = 5
        df.loc[valid_target, f"CONFIDENCE_OU_{target}"] = 1 / (1 + np.exp(-k * (margin - x0)))

    # Guardar cambios
    with NBADatabase() as db:
        db.insert_final_games(df, if_exists="replace")

    # Resumen
    for target in ["TOTAL", "HOME", "AWAY"]:
        col = f"RESULT_OU_{target}"
        hits = (df[col] == "Hit").sum()
        total = df[col].notna().sum()
        if total > 0:
            logger.info(f"  OU_{target}: {hits}/{total} ({hits/total*100:.1f}% hits)")

if __name__ == "__main__":
    main()
