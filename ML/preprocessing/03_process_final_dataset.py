"""
Script para procesar el dataset final de partidos de la NBA con características diferenciales.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import os
from pathlib import Path
import logging

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
            input_path: Ruta al archivo de entrada (opcional).
            output_dir: Directorio de salida (opcional).
        """
        # Obtener la ruta absoluta del directorio del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Establecer rutas por defecto si no se proporcionan
        default_input = os.path.join(script_dir, '..', '..', 'ML', 'data', 'raw', 'nba_games_raw.parquet')
        default_output = os.path.join(script_dir, '..', '..', 'ML', 'data', 'processed')
        
        self.input_path = input_path or default_input
        self.output_dir = output_dir or default_output
        self.df = None
        
        # Asegurar que exista el directorio de salida
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Directorio de entrada: {self.input_path}")
        logger.info(f"Directorio de salida: {self.output_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """Carga los datos desde el archivo de entrada."""
        logger.info(f"Cargando datos desde {self.input_path}")
        self.df = pd.read_parquet(self.input_path)
        logger.info(f"Datos cargados. Filas: {len(self.df)}, Columnas: {len(self.df.columns)}")
        return self.df
    
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
        
        # Función para calcular diferencias porcentuales
        def pct_diff(home, away):
            return (home - away) / ((home + away) / 2 + 1e-10)  # Evitar división por cero
        
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
        Guarda el dataset final en formato Parquet.
        
        Args:
            filename: Nombre del archivo de salida.
            
        Returns:
            Ruta al archivo guardado.
        """
        if self.df is None:
            raise ValueError("No hay datos para guardar. Ejecuta load_data() y calculate_differential_features() primero.")
            
        # Asegurar que el directorio de salida existe
        output_path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Guardar el dataset
        self.df.to_parquet(output_path, index=False)
        
        # Guardar también una versión CSV para fácil inspección (opcional)
        csv_path = output_path.replace('.parquet', '.csv')
        self.df.to_csv(csv_path, index=False)
        
        # Guardar metadatos
        self._save_metadata(output_path)
        
        logger.info(f"Dataset final guardado en {output_path}")
        logger.info(f"Tamaño del dataset: {len(self.df)} filas x {len(self.df.columns)} columnas")
        
        return output_path
    
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
        
        # Calcular características diferenciales
        logger.info("Calculando características diferenciales...")
        self.calculate_differential_features()
        
        # Añadir variable objetivo
        logger.info("Añadiendo variable objetivo...")
        self.add_target_variable()
        
        # Verificar valores nulos
        null_columns = self.df.columns[self.df.isnull().any()].tolist()
        if null_columns:
            logger.warning(f"Se encontraron valores nulos en las columnas: {null_columns}")
            # Rellenar numéricos con la media y categóricos con la moda
            for col in self.df.select_dtypes(include=['number']).columns:
                if self.df[col].isnull().any():
                    mean_val = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(mean_val)
                    logger.info(f"Rellenados {self.df[col].isnull().sum()} valores nulos en {col} con la media: {mean_val:.2f}")
            
            for col in self.df.select_dtypes(include=['object', 'category']).columns:
                if self.df[col].isnull().any():
                    mode_val = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode_val)
                    logger.info(f"Rellenados {self.df[col].isnull().sum()} valores nulos en {col} con la moda: {mode_val}")
        
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
        metadata_path = output_path.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadatos guardados en {metadata_path}")

def main():
    """Función principal para ejecutar el procesamiento del dataset final."""
    try:
        # Obtener rutas absolutas
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, '..', '..', 'ML', 'data', 'raw', 'nba_games_raw.parquet')
        output_dir = os.path.join(script_dir, '..', '..', 'ML', 'data', 'processed')
        
        # Crear instancia del procesador con las rutas correctas
        processor = NBAFinalProcessor(input_path=input_path, output_dir=output_dir)
        
        # Cargar datos
        processor.load_data()
        
        # Calcular características diferenciales
        df_final = processor.calculate_differential_features()
        
        # Generar dataset final
        output_path = processor.generate_final_dataset()
        
        logger.info(f"Procesamiento completado. Dataset guardado en: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
