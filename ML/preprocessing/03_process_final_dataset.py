"""
Script para procesar el dataset final de partidos de la NBA con características diferenciales.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import os
from pathlib import Path
import logging
# Asegurar imports relativos desde el paquete ML
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ML.db import NBADatabase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NBAFinalProcessor:
    """Clase para procesar el dataset final de partidos de la NBA."""
    
    def __init__(self, input_path: str = None, output_dir: str = None):
        """
        Inicializa el procesador de datos.

        Args:
            input_path: Ruta al archivo de entrada o "sqlite" para usar DB.
            output_dir: Directorio de salida (opcional).
        """
        # Obtener la ruta absoluta del directorio del script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Si input_path es "sqlite" o None, usar base de datos
        self.use_db = (input_path is None) or (input_path == "sqlite")

        # Establecer rutas por defecto si no se proporcionan
        default_input = os.path.join(script_dir, '..', '..', 'ML', 'data', 'raw', 'nba_games_raw.parquet')
        default_output = os.path.join(script_dir, '..', '..', 'ML', 'data', 'processed')

        self.input_path = input_path or default_input
        self.output_dir = output_dir or default_output
        self.df = None

        # Asegurar que exista el directorio de salida
        os.makedirs(self.output_dir, exist_ok=True)
        if self.use_db:
            logger.info("Fuente de datos: SQLite (nba.db)")
        else:
            logger.info(f"Directorio de entrada: {self.input_path}")
        logger.info(f"Directorio de salida: {self.output_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """Carga los datos desde SQLite o archivo de entrada."""
        if self.use_db:
            logger.info("Cargando datos desde SQLite...")
            with NBADatabase() as db:
                self.df = db.get_raw_games()
        else:
            logger.info(f"Cargando datos desde {self.input_path}")
            self.df = pd.read_parquet(self.input_path)
        logger.info(f"Datos cargados. Filas: {len(self.df)}, Columnas: {len(self.df.columns)}")
        self._recompute_l10_columns()
        return self.df
    
    def _recompute_l10_columns(self) -> None:
        """
        Recalcula las columnas *_L10_* desde los resultados del raw completo,
        replicando la fórmula exacta de 01_fetch (media móvil shift(1) de los
        10 juegos previos del equipo, ventana dentro de SEASON+SEASON_TYPE).

        Motivo: el fetch diario en temporada llega en lotes de 1-2 juegos →
        el rolling del fetch solo ve esos juegos y deja la mayoría in-season
        con L10 nulo, que luego 03 imputa con la media global en ~12 features.
        Recalcular aquí garantiza features íntegras e idempotentes en cada
        corrida, independientemente del tamaño del lote diario.

        Solo filas con resultados reales (todas las de raw); las filas
        placeholder de juegos futuros viven en nba_games_final y nunca
        pasan por aquí, así que sus features de pools (06) no se tocan.
        """
        df = self.df
        metrics = ['PTS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'POINT_DIFF']
        # POINT_DIFF: convención especial de 01 (columnas ya del lado del equipo)
        special_cols = {'POINT_DIFF': ('HOME_POINT_DIFF', 'AWAY_POINT_DIFF')}
        l10_cols = [f'{m}_L10_{s}' for m in metrics if f'{m}_L10_HOME' in df.columns
                    for s in ('HOME', 'AWAY')]
        missing = [c for c in l10_cols if c not in df.columns]
        if missing:
            logger.warning(f"Columnas L10 ausentes en raw, no se recalculan: {missing}")
            l10_cols = [c for c in l10_cols if c not in missing]
        if not l10_cols:
            return
        
        before_nulls = int(df[l10_cols].isna().sum().sum())
        before_notnull = int(df[l10_cols].notna().sum().sum())
        n_changed = 0
        
        # L10 por equipo dentro de cada (SEASON, SEASON_TYPE), igual que el
        # fetch histórico por lotses (los primeros juegos de temporada quedan
        # nulos legítimamente hasta min_periods)
        for (season, s_type), grp in df.groupby(['SEASON', 'SEASON_TYPE']):
            for team_id in grp['TEAM_ID_HOME'].dropna().unique():
                is_h = grp['TEAM_ID_HOME'] == team_id
                is_a = grp['TEAM_ID_AWAY'] == team_id
                tm = grp[is_h | is_a].sort_values('GAME_DATE')
                if tm.empty:
                    continue
                home_mask = (tm['TEAM_ID_HOME'] == team_id).values
                for metric in metrics:
                    h_col, a_col = special_cols.get(
                        metric, (f'{metric}_HOME', f'{metric}_AWAY'))
                    if h_col not in tm.columns or a_col not in tm.columns:
                        continue
                    values = np.where(home_mask, tm[h_col].values, tm[a_col].values)
                    rolling = (pd.Series(values).shift(1)
                               .rolling(window=10, min_periods=1).mean())
                    for pos, idx in enumerate(tm.index):
                        side = 'HOME' if home_mask[pos] else 'AWAY'
                        col = f'{metric}_L10_{side}'
                        new_v = rolling.iloc[pos]
                        old_v = df.at[idx, col]
                        old_na, new_na = pd.isna(old_v), pd.isna(new_v)
                        if (old_na and not new_na) or (not old_na and new_na) or \
                           (not old_na and not new_na and
                            abs(float(old_v) - float(new_v)) > 1e-6):
                            n_changed += 1
                        df.at[idx, col] = new_v
        
        after_nulls = int(df[l10_cols].isna().sum().sum())
        logger.info(
            f"Recalculo L10 (fórmula de 01): {n_changed} valores cambiados, "
            f"{before_notnull} no-nulos antes → {after_nulls - before_nulls:+d} nulls "
            f"({before_nulls} → {after_nulls})")
    
    def calculate_differential_features(self) -> pd.DataFrame:
        """
        Calcula características diferenciales entre equipos locales y visitantes.
        
        Returns:
            DataFrame con características diferenciales añadidas.
        """
        if self.df is None:
            raise ValueError("No se han cargado datos. Ejecuta load_data() primero.")
            
        logger.info("Calculando características diferenciales...")
        
        # Hacer una copia para no modificar el original
        df = self.df.copy()
        
        # 1. Características básicas del equipo
        basic_features = [
            'PTS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE',
            'W_PCT', 'GP', 'W', 'L', 'HOME_POINT_DIFF', 'AWAY_POINT_DIFF', 'POINT_DIFF'
        ]
        
        # 2. Características de los últimos 10 partidos
        l10_features = [
            'PTS_L10', 'OFF_RATING_L10', 'DEF_RATING_L10', 
            'NET_RATING_L10', 'PACE_L10', 'POINT_DIFF_L10'
        ]
        
        # 3. Características de descanso y viaje
        rest_features = [
            'DAYS_REST', 'IS_B2B', 'TRAVEL_DISTANCE'
        ]
        
        # Función para calcular diferencias porcentuales.
        # - Denominador con valor absoluto: sumas negativas (ej. NET_RATING con
        #   signos opuestos) invertían el signo del resultado.
        # - Clip ±10: con HOME≈AWAY≈0 el cociente explotaba a ±1e12. Los valores
        #   legítimos viven en [-2, 2], así que el clip no altera datos reales.
        def pct_diff(home, away):
            ratio = (home - away) / (np.abs((home + away) / 2) + 1e-10)
            return ratio.clip(-10, 10)
        
        # Calcular diferencias para características básicas
        for feat in basic_features:
            # Verificar si la característica tiene sufijo _HOME y _AWAY
            home_col = f'{feat}_HOME' if f'{feat}_HOME' in df.columns else feat
            away_col = f'{feat}_AWAY' if f'{feat}_AWAY' in df.columns else None
            
            # Si la columna de visitante no existe, verificar si es una columna general
            if away_col is None and feat in df.columns:
                # Si es una columna general, no podemos calcular diferencia
                continue
                
            # Calcular diferencia absoluta si ambas columnas existen
            if home_col in df.columns and away_col in df.columns:
                df[f'{feat}_DIFF'] = df[home_col] - df[away_col]
                
                # Diferencia porcentual (solo para valores numéricos, no conteos)
                if feat not in ['GP', 'W', 'L', 'HOME_POINT_DIFF', 'AWAY_POINT_DIFF', 'POINT_DIFF']:
                    df[f'{feat}_PCT_DIFF'] = pct_diff(df[home_col], df[away_col])
        
        # Calcular diferencias para características de últimos 10 partidos
        for feat in l10_features:
            home_col = f'{feat}_HOME'
            away_col = f'{feat}_AWAY'
            
            if home_col in df.columns and away_col in df.columns:
                # Diferencia absoluta
                df[f'{feat}_DIFF'] = df[home_col] - df[away_col]
                
                # Diferencia porcentual
                df[f'{feat}_PCT_DIFF'] = pct_diff(df[home_col], df[away_col])
        
        # Calcular diferencias para características de descanso y viaje
        for feat in rest_features:
            if feat in ['DAYS_REST', 'TRAVEL_DISTANCE']:
                home_col = f'{feat}_HOME'
                away_col = f'{feat}_AWAY'
                if home_col in df.columns and away_col in df.columns:
                    # Para días de descanso y distancia de viaje, calculamos la diferencia
                    df[f'{feat}_DIFF'] = df[home_col] - df[away_col]
            elif feat == 'IS_B2B':
                # Para B2B, codificamos como 1 si el local está en B2B y el visitante no, -1 si es al revés, 0 si ambos o ninguno
                if 'IS_B2B_HOME' in df.columns and 'IS_B2B_AWAY' in df.columns:
                    # Rellenar valores nulos con 0 antes de la conversión a enteros
                    df['IS_B2B_HOME'] = df['IS_B2B_HOME'].fillna(0).astype(int)
                    df['IS_B2B_AWAY'] = df['IS_B2B_AWAY'].fillna(0).astype(int)
                    df['B2B_ADVANTAGE'] = df['IS_B2B_HOME'] - df['IS_B2B_AWAY']
        
        # 4. Características de tendencia (últimos 5 partidos vs. temporada completa)
        for metric in ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE']:
            # Diferencia entre el rendimiento reciente y el de toda la temporada
            df[f'{metric}_HOME_TREND'] = df[f'{metric}_L10_HOME'] - df[f'{metric}_HOME']
            df[f'{metric}_AWAY_TREND'] = df[f'{metric}_L10_AWAY'] - df[f'{metric}_AWAY']
            df[f'{metric}_TREND_DIFF'] = df[f'{metric}_HOME_TREND'] - df[f'{metric}_AWAY_TREND']
        
        # 5. Características de rachas
        # Usamos el porcentaje de victorias en los últimos 10 partidos
        if 'W_PCT_L10_HOME' in df.columns and 'W_PCT_L10_AWAY' in df.columns:
            df['WIN_STREAK_DIFF'] = df['W_PCT_L10_HOME'] - df['W_PCT_L10_AWAY']
        else:
            # Si no hay datos de los últimos 10 partidos, usar el porcentaje general de victorias
            df['WIN_STREAK_DIFF'] = df['W_PCT_HOME'] - df['W_PCT_AWAY']
        
        logger.info(f"Se calcularon {len([c for c in df.columns if '_DIFF' in c or '_PCT_DIFF' in c or '_TREND' in c or 'STREAK' in c])} características diferenciales")
        
        # Actualizar el DataFrame
        self.df = df
        return df
    
    def add_target_variable(self) -> pd.DataFrame:
        """
        Añade la variable objetivo: 1 si gana el local, 0 si gana el visitante.
        """
        logger.info("Añadiendo variable objetivo...")
        
        # Verificar que las columnas necesarias existen
        if 'WL_HOME' not in self.df.columns:
            # Si no hay columna de resultado, intentamos inferirlo de los puntos
            if 'PTS_HOME' in self.df.columns and 'PTS_AWAY' in self.df.columns:
                logger.info("Inferiendo resultado del partido a partir de los puntos...")
                self.df['TARGET'] = (self.df['PTS_HOME'] > self.df['PTS_AWAY']).astype(int)
            else:
                raise ValueError("No se puede determinar el resultado del partido. Faltan columnas 'WL_HOME' o 'PTS_HOME'/'PTS_AWAY'")
        else:
            # Usar la columna WL_HOME directamente
            self.df['TARGET'] = (self.df['WL_HOME'] == 'W').astype(int)
        
        # Contar la distribución de la variable objetivo
        target_dist = self.df['TARGET'].value_counts(normalize=True) * 100
        logger.info(f"Distribución de la variable objetivo:\n{target_dist}")
        
        # Verificar si hay valores nulos en el objetivo
        if self.df['TARGET'].isnull().any():
            null_count = self.df['TARGET'].isnull().sum()
            logger.warning(f"¡Atención! Se encontraron {null_count} valores nulos en la variable objetivo.")
            # Eliminar filas con valores nulos en el objetivo
            self.df = self.df.dropna(subset=['TARGET'])
            logger.info(f"Se eliminaron {null_count} filas con valores nulos en la variable objetivo.")
        
        self.df['POINT_DIFF_TARGET'] = self.df['PTS_HOME'] - self.df['PTS_AWAY']
        
        logger.info(f"Variable objetivo añadida. Distribución: {self.df['TARGET'].value_counts().to_dict()}")
        
        return self.df
    
    def save_final_dataset(self, filename: str = "nba_games_final.parquet") -> str:
        """
        Guarda el dataset final en SQLite (y opcionalmente CSV para inspección).

        Preserva los datos de betting (LINE_OU_*, PREDICTION_OU_*, RECOMMENDATION_OU_*,
        RESULT_OU_*, CONFIDENCE_OU_*) de juegos históricos ya cargados en la BD,
        así como los placeholders del día actual.

        Args:
            filename: Nombre del archivo CSV de salida (la DB siempre se usa).

        Returns:
            Ruta al archivo CSV guardado.
        """
        if self.df is None:
            raise ValueError("No hay datos para guardar. Ejecuta load_data() y calculate_differential_features() primero.")

        # Columnas de betting que pueden existir en la BD pero no en el procesamiento
        betting_cols = [
            'LINE_OU_TOTAL', 'LINE_OU_HOME', 'LINE_OU_AWAY',
            'PREDICTION_OU_TOTAL', 'PREDICTION_OU_HOME', 'PREDICTION_OU_AWAY',
            'RECOMMENDATION_OU_TOTAL', 'RECOMMENDATION_OU_HOME', 'RECOMMENDATION_OU_AWAY',
            'RESULT_OU_TOTAL', 'RESULT_OU_HOME', 'RESULT_OU_AWAY',
            'CONFIDENCE_OU_TOTAL', 'CONFIDENCE_OU_HOME', 'CONFIDENCE_OU_AWAY',
        ]

        # Preservar datos de betting históricos y placeholders del día actual
        try:
            with NBADatabase() as db:
                existing_final = db.get_final_games()

            if 'GAME_DATE' in existing_final.columns and 'GAME_DATE' in self.df.columns and 'GAME_ID' in self.df.columns:
                existing_final['GAME_DATE'] = pd.to_datetime(existing_final['GAME_DATE'])

                # 1. Asegurar que self.df tenga TODAS las columnas de la BD
                for col in existing_final.columns:
                    if col not in self.df.columns:
                        self.df[col] = None

                # 2. Merge de columnas de betting de la BD hacia self.df por GAME_ID
                #    Solo para juegos que ya existen (match por GAME_ID)
                existing_with_betting = existing_final[
                    existing_final[betting_cols].notna().any(axis=1)
                ][['GAME_ID'] + [c for c in betting_cols if c in existing_final.columns]].copy()

                if not existing_with_betting.empty and 'GAME_ID' in self.df.columns:
                    # Solo mergear donde self.df no tiene datos de betting
                    betting_cols_in_df = [c for c in betting_cols if c in self.df.columns]
                    no_betting_mask = self.df[betting_cols_in_df].isna().all(axis=1)

                    if no_betting_mask.any():
                        merge_cols = [c for c in betting_cols if c in existing_with_betting.columns]
                        if merge_cols:
                            self.df = self.df.merge(
                                existing_with_betting,
                                on='GAME_ID',
                                how='left',
                                suffixes=('', '_existing')
                            )
                            # Llenar columnas de betting desde _existing donde estaban null
                            for c in merge_cols:
                                existing_col = f'{c}_existing'
                                if existing_col in self.df.columns:
                                    # Rellenaricios via máscara (evita FutureWarning
                                    # de downcasting del fillna en columnas object)
                                    fill_mask = self.df[c].isna()
                                    self.df.loc[fill_mask, c] = self.df.loc[fill_mask, existing_col]
                                    self.df = self.df.drop(columns=[existing_col])

                            logger.info(f"Preservadas columnas de betting para juegos existentes")

                # 3. Preservar placeholders del día actual (juegos de hoy aún no jugados)
                today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
                today_mask = existing_final['GAME_DATE'] == pd.to_datetime(today_str)
                not_in_processed = ~existing_final['GAME_ID'].isin(self.df['GAME_ID'])
                today_placeholders = existing_final[today_mask & not_in_processed].copy()

                if not today_placeholders.empty:
                    logger.info(f"Preservando {len(today_placeholders)} juegos de hoy ({today_str})")
                    for col in self.df.columns:
                        if col not in today_placeholders.columns:
                            today_placeholders[col] = None
                    today_placeholders = today_placeholders[self.df.columns]
                    self.df = pd.concat([self.df, today_placeholders], ignore_index=True)
        except Exception as e:
            logger.warning(f"No se pudieron preservar datos de betting: {e}")

        # Guardar en SQLite
        with NBADatabase() as db:
            db.create_tables()
            db.insert_final_games(self.df, if_exists="replace")
            logger.info(f"Dataset final guardado en SQLite: {db.db_path}")

        # Guardar también una versión CSV para fácil inspección
        csv_path = os.path.join(self.output_dir, filename.replace('.parquet', '.csv'))
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.df.to_csv(csv_path, index=False)

        # Guardar metadatos
        self._save_metadata(csv_path)

        logger.info(f"Tamaño del dataset: {len(self.df)} filas x {len(self.df.columns)} columnas")

        return csv_path
    
    def generate_final_dataset(self) -> str:
        """
        Procesa y guarda el dataset final.
        
        Returns:
            str: Ruta al archivo guardado.
        """
        logger.info("Iniciando generación del dataset final...")
        
        # Verificar que hay datos cargados
        if self.df is None or self.df.empty:
            raise ValueError("No hay datos cargados. Ejecuta load_data() primero.")
        
        # Verificar columnas mínimas requeridas
        required_columns = ['PTS_HOME', 'PTS_AWAY']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        # Calcular características diferenciales y añadir objetivo
        # (los métodos ya loguean sus propios mensajes)
        self.calculate_differential_features()
        self.add_target_variable()
        
        # Verificar valores nulos — imputar SOLO features.
        # No se imputan identificadores ni resultado/objetivo (PTS, WL, TARGET,
        # POINT_DIFF_TARGET): son datos hechos, no features del modelo.
        no_impute = {
            'GAME_ID', 'GAME_DATE', 'SEASON', 'SEASON_STR', 'SEASON_TYPE',
            'TEAM_ABBREVIATION_HOME', 'TEAM_ABBREVIATION_AWAY', 'WL_HOME', 'WL_AWAY',
            'PTS_HOME', 'PTS_AWAY', 'TARGET', 'POINT_DIFF_TARGET',
        }
        null_columns = [
            c for c in self.df.columns
            if self.df[c].isnull().any() and c not in no_impute
        ]
        if null_columns:
            logger.warning(f"Se encontraron valores nulos en las columnas: {null_columns}")
            # Rellenar numéricos con la media y categóricos con la moda
            for col in self.df.select_dtypes(include=['number']).columns:
                if col in no_impute or not self.df[col].isnull().any():
                    continue
                null_count = self.df[col].isnull().sum()
                mean_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(mean_val)
                logger.info(f"Rellenados {null_count} valores nulos en {col} con la media: {mean_val:.2f}")

            for col in self.df.select_dtypes(include=['object', 'category']).columns:
                if col in no_impute or not self.df[col].isnull().any():
                    continue
                null_count = self.df[col].isnull().sum()
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_val)
                logger.info(f"Rellenados {null_count} valores nulos en {col} con la moda: {mode_val}")
        
        # Guardar dataset final
        logger.info("Guardando dataset final...")
        return self.save_final_dataset()
    
    def _save_metadata(self, output_path: str):
        """Guarda metadatos sobre el dataset generado."""
        if self.df is None:
            return
            
        metadata = {
            'fecha_creacion': pd.Timestamp.now().isoformat(),
            'num_partidos': len(self.df),
            'temporadas': sorted(self.df['SEASON'].unique().tolist()),
            'equipos_locales': self.df['TEAM_ABBREVIATION_HOME'].nunique(),
            'equipos_visitantes': self.df['TEAM_ABBREVIATION_AWAY'].nunique(),
            'rango_fechas': {
                'inicio': self.df['GAME_DATE'].min().strftime('%Y-%m-%d'),
                'fin': self.df['GAME_DATE'].max().strftime('%Y-%m-%d')
            },
            'columnas': {
                'total': len(self.df.columns),
                'categorias': {
                    'identificadores': [c for c in self.df.columns if c in ['GAME_ID', 'GAME_DATE', 'SEASON', 'SEASON_STR', 'SEASON_TYPE']],
                    'equipos': [c for c in self.df.columns if '_HOME' in c or '_AWAY' in c],
                    'diferenciales': [c for c in self.df.columns if '_DIFF' in c],
                    'tendencias': [c for c in self.df.columns if 'TREND' in c or 'STREAK' in c],
                    'objetivo': [c for c in self.df.columns if c.startswith('TARGET')]
                }
            }
        }
        
        # Guardar metadatos en un archivo JSON
        import json
        metadata_path = output_path.replace('.csv', '_metadata.json').replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadatos guardados en {metadata_path}")

def main():
    """Función principal para ejecutar el procesamiento del dataset final.

    Flujo del pipeline:
      02_update_nba_data.py  → nuevos juegos de la API → BD (raw)
      03_process_final_dataset.py → features diferenciales → BD (final)
      (líneas de apuestas) → BD
      06_predict_upcoming_games.py → pronósticos → BD
    """
    try:
        # Obtener rutas absolutas
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, '..', '..', 'ML', 'data', 'processed')

        # FUENTE: SQLite (actualizado por 02_update_nba_data.py)
        # Ya no se usa el parquet como fuente primaria
        input_path = "sqlite"

        # Crear instancia del procesador
        processor = NBAFinalProcessor(input_path=input_path, output_dir=output_dir)

        # Cargar datos desde SQLite
        processor.load_data()

        # Calcular características diferenciales (generate_final_dataset
        # ya las ejecuta internamente; no duplicar el trabajo)
        # Generar dataset final (guarda en SQLite + CSV para inspección)
        output_path = processor.generate_final_dataset()

        logger.info(f"Procesamiento completado. Dataset guardado en: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error durante el procesamiento: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
