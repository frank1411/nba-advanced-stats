#!/usr/bin/env python3
"""
Script para verificar que todos los archivos de modelo necesarios estén presentes.
"""
import sys
from pathlib import Path

# Configuración de rutas
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
REQUIRED_FILES = [
    "model_PTS_HOME.pkl",
    "model_PTS_AWAY.pkl",
    "best_pipeline.pkl",
    "feature_list.json"
]

def verify_files():
    """Verifica que todos los archivos requeridos existan."""
    print("🔍 Verificando archivos de modelo...")
    
    all_ok = True
    for filename in REQUIRED_FILES:
        file_path = MODELS_DIR / filename
        if file_path.exists():
            print(f"✅ {filename} encontrado")
        else:
            print(f"❌ {filename} NO encontrado")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Algunos archivos requeridos no se encontraron.")
        print("Por favor, asegúrate de que los siguientes archivos estén presentes:")
        for filename in REQUIRED_FILES:
            print(f"- {MODELS_DIR / filename}")
        sys.exit(1)
    
    print("\n✅ Todos los archivos requeridos están presentes")

if __name__ == "__main__":
    verify_files()
