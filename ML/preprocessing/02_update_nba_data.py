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
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional

# Asegurar imports relativos desde el paquete ML
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from ML.db import NBADatabase

# 30 franquicias NBA: whitelist para descartar eventos con equipos no NBA
# (All-Star, exhibiciones con invitados: STP/STR/ANT/MEL/etc. — esos equipos
# no tienen stats de temporada; sus filas terminan imputadas con la media
# global contaminando el dataset)
VALID_NBA_TEAMS = {
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS',
}

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
    
    def _retry_api_call(self, func, label: str, max_attempts: int = 3, base_wait: float = 20.0):
        """
        Reintenta una llamada a la API al nivel del flujo (encima de los reintentos HTTP).
        
        Convenio: la función subyacente devuelve None cuando falló la API y un
        DataFrame (posiblemente vacío) cuando tuvo éxito sin datos.
        
        Args:
            func: callable sin argumentos a ejecutar
            label: nombre descriptivo para los logs
            max_attempts: intentos totales
            base_wait: espera base en segundos (backoff exponencial: base * 2^intento)
        
        Returns:
            Resultado exitoso (DataFrame, puede estar vacío) o None si falló todo
        """
        for attempt in range(max_attempts):
            try:
                result = func()
            except Exception as e:
                logger.warning(f"Excepción en {label} (intento {attempt + 1}/{max_attempts}): {e}")
                result = None
            
            if result is not None:
                return result
            
            if attempt == max_attempts - 1:
                break
            
            wait_time = base_wait * (2 ** attempt)
            logger.warning(f"⚠️ {label} falló (intento {attempt + 1}/{max_attempts}). Reintentando en {wait_time:.0f} s...")
            time.sleep(wait_time)
        
        logger.error(f"❌ {label} falló después de {max_attempts} intentos")
        return None
    
#     def fetch_new_data(self, season: str = '2025-26') -> Optional[pd.DataFrame]:
    def fetch_new_data(self, season: str = '2026-27') -> Optional[pd.DataFrame]:
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
            
            # Obtener estadísticas de equipo primero.
            # No es fatal: si falla, los partidos se guardan sin ratings (L10 se
            # calculará cuando haya estadísticas disponibles).
            logger.info("📊 Obteniendo estadísticas de equipos...")
            team_stats = self._retry_api_call(
                lambda: fetcher.get_team_stats(season, 'Regular Season'),
                f"estadísticas de equipos {season}"
            )
            if team_stats is None:
                logger.warning("⚠️ Continuando sin estadísticas de equipos")
            
            # Procesar temporada regular
            logger.info("🏀 Buscando partidos de temporada regular...")
            regular_games = self._retry_api_call(
                lambda: fetcher.get_season_games(season=season, season_type='Regular Season'),
                "búsqueda de temporada regular"
            )
            
            # Procesar playoffs si los hay
            logger.info("🏆 Buscando partidos de playoffs...")
            playoff_games = self._retry_api_call(
                lambda: fetcher.get_season_games(season=season, season_type='Playoffs'),
                "búsqueda de playoffs"
            )
            
            # Buscar todos los partidos para capturar Copa NBA y otros
            logger.info("🏆 Buscando partidos de Copa NBA y otros...")
            all_types_games = self._retry_api_call(
                lambda: fetcher.get_season_games(season=season, season_type='ALL'),
                "búsqueda de Copa NBA y otros"
            )
            
            # Combinar partidos
            all_games = []
            
            for games, season_type in [(regular_games, 'Regular Season'), (playoff_games, 'Playoffs'), (all_types_games, 'NBA Cup/Other')]:
                # None = la API falló en todos los reintentos: abortar como fallo
                # (main() lo registrará como ERROR y saldrá con código 1)
                if games is None:
                    logger.error(f"❌ Búsqueda de {season_type} falló; abortando la verificación de partidos nuevos")
                    return None

                if not games.empty:
                    # Descartar partidos de pretemporada: GAME_ID con prefijo '001'.
                    # Códigos de la API: 001=Pretemporada, 002=Regular Season,
                    # 003=NBA Cup/In-Season, 004=Playoffs, 005/006=otros.
                    # La pretemporada se ignora; la NBA Cup se conserva.
                    if 'GAME_ID' in games.columns:
                        before = len(games)
                        games = games[~games['GAME_ID'].astype(str).str.startswith('001')].copy()
                        dropped = before - len(games)
                        if dropped > 0:
                            logger.info(f"🗑️ Descartadas {dropped} filas de pretemporada en {season_type}")

                    # Descartar eventos con equipos no franquicia (All-Star,
                    # exhibiciones con invitados) — solo las 30 franquicias NBA
                    if 'TEAM_ABBREVIATION_HOME' in games.columns and 'TEAM_ABBREVIATION_AWAY' in games.columns:
                        before = len(games)
                        games = games[
                            games['TEAM_ABBREVIATION_HOME'].isin(VALID_NBA_TEAMS) &
                            games['TEAM_ABBREVIATION_AWAY'].isin(VALID_NBA_TEAMS)
                        ].copy()
                        dropped = before - len(games)
                        if dropped > 0:
                            logger.info(f"🗑️ Descartadas {dropped} filas de eventos sin franquicias NBA (All-Star/exhibiciones)")

                    # Filtrar partidos nuevos
                    if 'GAME_DATE' in games.columns:
                        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
                        if self.last_game_date is not None:
                            games = games[games['GAME_DATE'] > self.last_game_date]
                    
                    if not games.empty:
                        # Procesar partidos con estadísticas de equipo
                        logger.info(f"🔄 Procesando {int(len(games)/2)} partidos de {season_type}...")
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

            # Guardar con upsert incremental: solo las filas nuevas tocan la BD.
            # GAME_ID es PRIMARY KEY, así que INSERT OR REPLACE es seguro y
            # evita el rewrite completo de ~10k filas (riesgo si se interrumpe).
            # get_game_count va DENTRO del with: al salir del bloque la conexión
            # se cierra (self._conn = None) y hacer queries después es NoneType crash.
            with NBADatabase() as db:
                db.create_tables()
                inserted = db.bulk_upsert("nba_games_raw", new_data)
                total = db.get_game_count("raw")

            logger.info(f"✅ Datos guardados en SQLite")
            logger.info(f"📊 Partidos nuevos insertados: {inserted}")
            logger.info(f"📊 Total de partidos en BD: {total}")

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
    
    # Distinguir "la API falló" (None) de "no hubo datos nuevos" (DataFrame vacío)
    fetch_failed = new_data is None
    
    if fetch_failed:
        logger.error("❌ La API falló. NO se pudo verificar si hay partidos nuevos (esto NO es lo mismo que 'sin partidos nuevos')")
    elif new_data.empty:
        logger.info("ℹ️ No hay partidos nuevos para actualizar")
    else:
        success = updater.update_data(new_data)
        if success:
            logger.info("✅ Actualización completada exitosamente")
            
            # Procesar los datos actualizados (opcional, descomentar si es necesario)
            # from .process_final_dataset import process_data
            # process_data()
        else:
            logger.error("❌ Error al actualizar los datos")
            fetch_failed = True

    # Calcular RESULT_OU_* y CONFIDENCE_OU_* para partidos con predicción + línea + resultado
    update_betting_results()

    logger.info("🏁 Proceso de actualización finalizado")
    
    # Salir con código de error si la descarga o actualización falló,
    # para que el fallo sea detectable por scripts/usuarios que dependan de este paso
    if fetch_failed:
        sys.exit(1)


def update_betting_results():
    """
    Calcula RESULT_OU_* y CONFIDENCE_OU_* para partidos que ahora tienen:
    - Predicción del modelo (PREDICTION_OU_*) en tabla final
    - Línea de la casa (LINE_OU_*) en tabla final
    - Resultado real (PTS_HOME, PTS_AWAY) en tabla raw

    Se ejecuta después de actualizar datos para evaluar automáticamente
    los pronósticos de partidos ya jugados.

    IMPORTANTE: Los puntos reales se obtienen de la tabla raw porque la tabla
    final puede tener PTS como NaN para juegos de playoffs o placeholders.
    """
    logger.info("=" * 50)
    logger.info("Calculando resultados de betting...")

    with NBADatabase() as db:
        df_final = db.get_final_games()
        df_raw = db.get_raw_games()

    # Snapshot para detectar solo las filas realmente modificadas al final
    df_final_before = df_final.copy()

    # Obtener puntos reales de la tabla raw
    # Normalizar GAME_DATE en raw
    if 'GAME_DATE' in df_raw.columns:
        df_raw['GAME_DATE'] = pd.to_datetime(df_raw['GAME_DATE'])

    # Crear un diccionario de lookup: GAME_ID -> (PTS_HOME, PTS_AWAY)
    raw_lookup = {}
    if 'GAME_ID' in df_raw.columns:
        for _, row in df_raw.iterrows():
            gid = row['GAME_ID']
            pts_h = row.get('PTS_HOME')
            pts_a = row.get('PTS_AWAY')
            if pd.notna(pts_h) and pd.notna(pts_a) and pts_h > 0 and pts_a > 0:
                raw_lookup[gid] = (pts_h, pts_a)

    # También crear lookup por fecha + equipos para juegos sin GAME_ID match
    # (placeholders con GAME_ID tipo "placeholder_20260603_SAS_NYK")
    raw_date_team_lookup = {}
    for _, row in df_raw.iterrows():
        date = str(pd.to_datetime(row['GAME_DATE']).date())
        home = row.get('TEAM_ABBREVIATION_HOME', '')
        away = row.get('TEAM_ABBREVIATION_AWAY', '')
        pts_h = row.get('PTS_HOME')
        pts_a = row.get('PTS_AWAY')
        if pd.notna(pts_h) and pd.notna(pts_a) and pts_h > 0 and pts_a > 0:
            key = (date, home, away)
            raw_date_team_lookup[key] = (pts_h, pts_a)

    # Cruzar puntos reales de raw hacia final
    points_updated = 0
    for idx in df_final.index:
        gid = df_final.loc[idx, 'GAME_ID']
        game_date = df_final.loc[idx, 'GAME_DATE']

        # Intentar match por GAME_ID primero
        if gid in raw_lookup:
            pts_h, pts_a = raw_lookup[gid]
            if pd.isna(df_final.loc[idx, 'PTS_HOME']) or df_final.loc[idx, 'PTS_HOME'] == 0:
                df_final.loc[idx, 'PTS_HOME'] = pts_h
                df_final.loc[idx, 'PTS_AWAY'] = pts_a
                points_updated += 1
                continue

        # Intentar match por fecha + equipos
        date_str = str(pd.to_datetime(game_date).date())
        home = df_final.loc[idx, 'TEAM_ABBREVIATION_HOME']
        away = df_final.loc[idx, 'TEAM_ABBREVIATION_AWAY']
        key = (date_str, home, away)

        if key in raw_date_team_lookup:
            pts_h, pts_a = raw_date_team_lookup[key]
            if pd.isna(df_final.loc[idx, 'PTS_HOME']) or df_final.loc[idx, 'PTS_HOME'] == 0:
                df_final.loc[idx, 'PTS_HOME'] = pts_h
                df_final.loc[idx, 'PTS_AWAY'] = pts_a
                points_updated += 1

    if points_updated > 0:
        logger.info(f"✅ Se actualizaron {points_updated} juegos con puntos reales de la tabla raw")

    # Ahora evaluar predicciones con los puntos (ya actualizados si fue posible)
    has_pred = df_final['PREDICTION_OU_TOTAL'].notna()
    has_line = df_final['LINE_OU_TOTAL'].notna()
    has_result = df_final['PTS_HOME'].notna() & (df_final['PTS_HOME'] > 0)

    valid = has_pred & has_line & has_result

    if valid.sum() == 0:
        logger.info("No hay partidos con predicción + línea + resultado para evaluar")
        return

    logger.info(f"Evaluando {valid.sum()} partidos con datos completos...")

    new_evaluations = 0
    for target in ["TOTAL", "HOME", "AWAY"]:
        line_col = f"LINE_OU_{target}"
        pred_col = f"PREDICTION_OU_{target}"
        result_col = f"RESULT_OU_{target}"

        # Solo evaluar si no tiene ya un resultado
        already_has_result = df_final[result_col].notna()

        if target == "TOTAL":
            actual = df_final["PTS_HOME"] + df_final["PTS_AWAY"]
        elif target == "HOME":
            actual = df_final["PTS_HOME"]
        else:
            actual = df_final["PTS_AWAY"]

        valid_target = valid & df_final[line_col].notna() & df_final[pred_col].notna() & ~already_has_result
        if valid_target.sum() == 0:
            continue

        predicted_over = df_final.loc[valid_target, pred_col] > df_final.loc[valid_target, line_col]
        actual_over = actual.loc[valid_target] > df_final.loc[valid_target, line_col]

        df_final.loc[valid_target, result_col] = np.where(
            predicted_over == actual_over, "Hit", "Miss"
        )

        # Recalcular confianza basada en el margen |predicción - línea|
        margin = (df_final.loc[valid_target, pred_col] - df_final.loc[valid_target, line_col]).abs()
        k = 0.5
        x0 = 5
        df_final.loc[valid_target, f"CONFIDENCE_OU_{target}"] = 1 / (1 + np.exp(-k * (margin - x0)))

        new_evaluations += valid_target.sum()

    if new_evaluations == 0:
        logger.info("Todos los partidos con datos completos ya tienen RESULT_OU_* calculado")
    else:
        logger.info(f"✅ Se evaluaron {new_evaluations} predicciones nuevas")

    # Guardar solo las filas modificadas (upsert sobre PRIMARY KEY GAME_ID).
    # Evita el rewrite completo de ~10k filas: el resto de la tabla no se toca.
    changed_cols = [
        "PTS_HOME", "PTS_AWAY",
        "RESULT_OU_TOTAL", "RESULT_OU_HOME", "RESULT_OU_AWAY",
        "CONFIDENCE_OU_TOTAL", "CONFIDENCE_OU_HOME", "CONFIDENCE_OU_AWAY",
    ]
    try:
        eq = df_final[changed_cols].eq(df_final_before[changed_cols])
        both_nan = df_final[changed_cols].isna() & df_final_before[changed_cols].isna()
        diff_mask = ~(eq | both_nan).all(axis=1)
    except KeyError:
        # Columnas no presentes: fallback conservador, upsert de todo
        diff_mask = pd.Series(True, index=df_final.index)

    df_changed = df_final[diff_mask]
    if df_changed.empty:
        logger.info("💾 Sin filas modificadas en nba_games_final")
    else:
        with NBADatabase() as db:
            db.bulk_upsert("nba_games_final", df_changed)
        logger.info(f"💾 Upsert de {len(df_changed)} filas modificadas en nba_games_final")

    # Resumen global
    logger.info("\nResumen de accuracy:")
    for target in ["TOTAL", "HOME", "AWAY"]:
        col = f"RESULT_OU_{target}"
        hits = (df_final[col] == "Hit").sum()
        total = df_final[col].notna().sum()
        if total > 0:
            logger.info(f"  OU_{target}: {hits}/{total} ({hits/total*100:.1f}% hits)")

if __name__ == "__main__":
    main()
