#!/usr/bin/env python3
"""
Script principal para ejecutar todo el pipeline de procesamiento de datos de la NBA.
"""
import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def run_fetch_data():
    """Ejecuta el script de descarga de datos."""
    try:
        logger.info("Iniciando descarga de datos de la NBA...")
        
        # Importar y ejecutar el script de descarga
        from ML.preprocessing.fetch_nba_data import main as fetch_main
        
        # Asegurar que el directorio de salida exista
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ML', 'data', 'raw')
        os.makedirs(output_dir, exist_ok=True)
        
        # Ejecutar la función principal con la ruta de salida correcta
        fetch_main()
        
        logger.info("Descarga de datos completada con éxito.")
        return True
    except Exception as e:
        logger.error(f"Error durante la descarga de datos: {str(e)}", exc_info=True)
        return False

def run_process_final_dataset():
    """Ejecuta el script de procesamiento del dataset final."""
    try:
        logger.info("Iniciando procesamiento del dataset final...")
        
        # Importar y ejecutar el script de procesamiento final
        from ML.preprocessing.process_final_dataset import NBAFinalProcessor
        
        # Definir rutas absolutas
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(script_dir, 'ML', 'data', 'raw', 'nba_games_raw.parquet')
        output_dir = os.path.join(script_dir, 'ML', 'data', 'processed')
        
        # Asegurar que exista el directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear instancia del procesador con las rutas correctas
        processor = NBAFinalProcessor(input_path=input_path, output_dir=output_dir)
        
        # Cargar y procesar los datos
        processor.load_data()
        df_final = processor.calculate_differential_features()
        output_path = processor.generate_final_dataset()
        
        logger.info(f"Procesamiento completado. Dataset guardado en: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error durante el procesamiento del dataset final: {str(e)}", exc_info=True)
        return False

def main():
    """Función principal que ejecuta todo el pipeline."""
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE PROCESAMIENTO DE DATOS DE LA NBA")
    logger.info("=" * 80)
    
    # 1. Ejecutar descarga de datos
    if not run_fetch_data():
        logger.error("Error en la descarga de datos. Deteniendo el pipeline.")
        return False
    
    # 2. Ejecutar procesamiento del dataset final
    if not run_process_final_dataset():
        logger.error("Error en el procesamiento del dataset final.")
        return False
    
    logger.info("=" * 80)
    logger.info("¡PIPELINE COMPLETADO CON ÉXITO!")
    logger.info("=" * 80)
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
