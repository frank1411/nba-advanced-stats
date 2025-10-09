#!/usr/bin/env python3
"""
Pipeline principal para el sistema de predicción de la NBA.

Este script orquesta todo el flujo de trabajo, desde la actualización de datos
del histórico hasta la generación de predicciones.
"""
import argparse
import logging
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

from ML.data.update_historical import update_historical_data
from ML.models.train import train_model
from ML.predict.generate import generate_predictions

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parsear argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Pipeline de predicción de la NBA')
    parser.add_argument('--update-historical', action='store_true',
                      help='Actualizar datos históricos')
    parser.add_argument('--retrain', action='store_true',
                      help='Reentrenar el modelo')
    parser.add_argument('--predict', action='store_true',
                      help='Generar predicciones')
    parser.add_argument('--all', action='store_true',
                      help='Ejecutar todo el pipeline')
    
    # Argumentos para la actualización de datos
    parser.add_argument('--start-season', type=int, default=2016,
                      help='Primera temporada a incluir (por defecto: 2016)')
    parser.add_argument('--end-season', type=int, default=2025,
                      help='Última temporada a incluir (por defecto: 2025)')
    
    # Argumentos para el entrenamiento
    parser.add_argument('--test-season', type=int, default=2025,
                      help='Temporada a usar para pruebas (por defecto: 2025)')
    
    return parser.parse_args()

def main():
    """Función principal."""
    args = parse_args()
    
    # Configurar parámetros
    data_params = {
        'start_season': args.start_season,
        'end_season': args.end_season,
        'test_season': args.test_season
    }
    
    # Ejecutar pasos solicitados
    if args.all or args.update_historical:
        logger.info("🔄 Actualizando datos históricos...")
        update_historical_data(
            start_season=data_params['start_season'],
            end_season=data_params['end_season']
        )
    
    if args.all or args.retrain:
        logger.info("🤖 Entrenando modelo...")
        train_model(test_season=data_params['test_season'])
    
    if args.all or args.predict:
        logger.info("🔮 Generando predicciones...")
        generate_predictions()
    
    logger.info("✅ Pipeline completado exitosamente")

if __name__ == "__main__":
    main()
