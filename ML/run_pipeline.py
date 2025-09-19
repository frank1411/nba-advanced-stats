#!/usr/bin/env python3
"""
Pipeline para la extracción y procesamiento de datos de la NBA
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent))

from preprocessing.fetch_nba_data import main as fetch_data
from preprocessing.calculate_advanced_stats import main as calculate_stats
from preprocessing.create_master_dataset import main as create_dataset

def main():
    print("=== Iniciando pipeline de procesamiento de datos de la NBA ===\n")
    
    # Paso 1: Obtener datos en bruto
    print("\n=== PASO 1: Obteniendo datos de la API de la NBA ===")
    raw_df = fetch_data()
    
    if raw_df is None or raw_df.empty:
        print("Error: No se pudieron obtener los datos de la API")
        return
    
    # Paso 2: Calcular estadísticas avanzadas
    print("\n=== PASO 2: Calculando estadísticas avanzadas ===")
    processed_df = calculate_stats()
    
    if processed_df is None or processed_df.empty:
        print("Error: No se pudieron calcular las estadísticas avanzadas")
        return
    
    # Paso 3: Crear dataset maestro
    print("\n=== PASO 3: Creando dataset maestro ===")
    master_df = create_dataset()
    
    if master_df is not None and not master_df.empty:
        print("\n¡Proceso completado exitosamente!")
    else:
        print("\nError: No se pudo crear el dataset maestro")

if __name__ == "__main__":
    main()
