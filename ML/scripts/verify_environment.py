#!/usr/bin/env python3
"""
Script para verificar la consistencia entre entornos de ejecución.
"""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
PREDICTIONS_DIR = DATA_DIR / "predictions"
REPORTS_DIR = BASE_DIR / "reports"

# Archivos clave a verificar
KEY_FILES = {
    "model_home": MODELS_DIR / "model_PTS_HOME.pkl",
    "model_away": MODELS_DIR / "model_PTS_AWAY.pkl",
    "data_processed": PROCESSED_DATA_DIR / "nba_games_final.parquet",
    "feature_list": MODELS_DIR / "feature_list.json",
    "metrics": MODELS_DIR / "metrics.json"
}

# Configurar logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('environment_verification.log')
    ]
)
logger = logging.getLogger(__name__)

def get_environment_info() -> Dict:
    """Obtiene información del entorno de ejecución."""
    return {
        "system": platform.system(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timestamp": datetime.now().isoformat(),
        "working_directory": str(Path.cwd()),
        "script_path": str(Path(__file__).resolve())
    }

def get_package_versions() -> Dict:
    """Obtiene las versiones de los paquetes clave."""
    try:
        from importlib.metadata import distributions, version
        packages = [
            'numpy', 'pandas', 'scikit-learn', 'joblib', 'pyarrow',
            'pytz', 'python-dateutil', 'scipy'
        ]
        versions = {}
        for dist in distributions():
            name = dist.metadata['Name']
            if name.lower() in [pkg.lower() for pkg in packages]:
                try:
                    versions[name] = version(name)
                except Exception:
                    versions[name] = dist.version
        return versions
    except ImportError:
        # Fallback para versiones antiguas de Python
        import pkg_resources
        fallback_packages = [
            'numpy', 'pandas', 'scikit-learn', 'joblib', 'pyarrow',
            'pytz', 'python-dateutil', 'scipy'
        ]
        return {pkg: pkg_resources.get_distribution(pkg).version 
                for pkg in fallback_packages if pkg in pkg_resources.working_set}

def calculate_file_hash(file_path: Path) -> str:
    """Calcula el hash MD5 de un archivo."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def verify_file_integrity() -> Dict:
    """Verifica la integridad de los archivos clave."""
    results = {}
    for name, path in KEY_FILES.items():
        try:
            if path.exists():
                file_info = {
                    "exists": True,
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "md5_hash": calculate_file_hash(path)
                }
            else:
                file_info = {"exists": False}
            results[name] = file_info
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

def load_and_validate_models() -> Dict:
    """Carga y valida los modelos."""
    results = {}
    try:
        # Verificar modelo de puntos locales
        model_home = joblib.load(KEY_FILES["model_home"])
        results["model_home"] = {
            "type": type(model_home).__name__,
            "n_estimators": getattr(model_home, 'n_estimators', 'N/A'),
            "features": list(getattr(model_home, 'feature_names_in_', []))
        }
        
        # Verificar modelo de puntos visitantes
        model_away = joblib.load(KEY_FILES["model_away"])
        results["model_away"] = {
            "type": type(model_away).__name__,
            "n_estimators": getattr(model_away, 'n_estimators', 'N/A'),
            "features": list(getattr(model_away, 'feature_names_in_', []))
        }
    except Exception as e:
        results["error"] = str(e)
    return results

def validate_data() -> Dict:
    """Valida los datos de entrada."""
    results = {}
    try:
        # Cargar datos procesados
        df = pd.read_parquet(KEY_FILES["data_processed"])
        
        # Estadísticas básicas
        results["data_stats"] = {
            "num_games": len(df),
            "date_range": {
                "min": str(df['GAME_DATE'].min()),
                "max": str(df['GAME_DATE'].max())
            },
            "teams_home": df['TEAM_ABBREVIATION_HOME'].nunique(),
            "teams_away": df['TEAM_ABBREVIATION_AWAY'].nunique(),
            "columns": list(df.columns)
        }
        
        # Verificar valores nulos
        null_counts = df.isnull().sum()
        results["null_values"] = {
            "total": int(null_counts.sum()),
            "top_missing": null_counts[null_counts > 0].to_dict()
        }
        
    except Exception as e:
        results["error"] = str(e)
    return results

def generate_verification_report() -> Dict:
    """Genera un informe completo de verificación."""
    report = {
        "environment": get_environment_info(),
        "packages": get_package_versions(),
        "file_integrity": verify_file_integrity(),
        "models": load_and_validate_models(),
        "data_validation": validate_data(),
        "verification_timestamp": datetime.now().isoformat()
    }
    return report

def save_report(report: Dict, output_dir: Path = None):
    """Guarda el informe en formato legible."""
    if output_dir is None:
        output_dir = Path.cwd()
    
    # Crear directorio si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar como JSON
    json_path = output_dir / "environment_verification.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Guardar como texto legible
    txt_path = output_dir / "environment_verification.txt"
    with open(txt_path, 'w') as f:
        f.write("=== Informe de Verificación de Entorno ===\n\n")
        f.write(f"Fecha de generación: {report['verification_timestamp']}\n")
        f.write(f"Sistema: {report['environment']['system']} {report['environment']['machine']}\n")
        f.write(f"Python: {report['environment']['python_version']}\n\n")
        
        f.write("=== Paquetes y Versiones ===\n")
        for pkg, ver in report['packages'].items():
            f.write(f"{pkg}: {ver}\n")
        
        f.write("\n=== Integridad de Archivos ===\n")
        for name, info in report['file_integrity'].items():
            f.write(f"\n{name}:\n")
            if 'error' in info:
                f.write(f"  ERROR: {info['error']}\n")
            elif info.get('exists'):
                f.write(f"  Tamaño: {info['size_mb']} MB\n")
                f.write(f"  Modificado: {info['modified']}\n")
                f.write(f"  MD5: {info['md5_hash']}\n")
            else:
                f.write("  NO ENCONTRADO\n")
        
        f.write("\n=== Validación de Modelos ===\n")
        if 'error' in report['models']:
            f.write(f"ERROR: {report['models']['error']}\n")
        else:
            for model_name, model_info in report['models'].items():
                if model_name == 'error':
                    continue
                f.write(f"\n{model_name}:\n")
                f.write(f"  Tipo: {model_info['type']}\n")
                f.write(f"  Número de estimadores: {model_info['n_estimators']}\n")
                f.write(f"  Número de características: {len(model_info['features']) if model_info['features'] is not None else 'N/A'}")
                if model_info['features']:
                    f.write("  \n  Primeras 5 características: " + ", ".join(model_info['features'][:5]) + "...")
        
        f.write("\n\n=== Validación de Datos ===\n")
        if 'error' in report['data_validation']:
            f.write(f"ERROR: {report['data_validation']['error']}\n")
        else:
            stats = report['data_validation']['data_stats']
            f.write(f"Número de partidos: {stats['num_games']}\n")
            f.write(f"Rango de fechas: {stats['date_range']['min']} a {stats['date_range']['max']}\n")
            f.write(f"Equipos (local/visitante): {stats['teams_home']}/{stats['teams_away']}\n")
            
            nulls = report['data_validation']['null_values']
            f.write(f"\nValores nulos totales: {nulls['total']}")
            if nulls['top_missing']:
                f.write("\nColumnas con valores nulos (primeras 5):\n")
                for i, (col, count) in enumerate(nulls['top_missing'].items()):
                    if i >= 5:  # Mostrar solo las primeras 5 columnas con valores nulos
                        f.write(f"  - ... y {len(nulls['top_missing']) - 5} columnas más\n")
                        break
                    f.write(f"  - {col}: {count} valores nulos\n")
    
    logger.info(f"Informe de verificación guardado en:\n- {json_path}\n- {txt_path}")
    return json_path, txt_path

def main():
    """Función principal."""
    logger.info("Iniciando verificación del entorno...")
    
    # Crear directorio de informes si no existe
    REPORTS_DIR.mkdir(exist_ok=True)
    
    # Generar informe
    report = generate_verification_report()
    
    # Guardar informe
    json_path, txt_path = save_report(report, REPORTS_DIR)
    
    # Mostrar resumen
    print("\n" + "="*80)
    print(" RESUMEN DE VERIFICACIÓN".center(80))
    print("="*80)
    print(f"\n📋 Entorno:")
    print(f"   Sistema: {report['environment']['system']} {report['environment']['machine']}")
    print(f"   Python: {report['environment']['python_version']}")
    print(f"   Directorio: {report['environment']['working_directory']}")
    
    print("\n🔍 Archivos clave:")
    for name, info in report['file_integrity'].items():
        status = "✅" if info.get('exists') else "❌"
        print(f"   {status} {name}: {'Encontrado' if info.get('exists') else 'No encontrado'}")
        if info.get('exists'):
            print(f"      Tamaño: {info['size_mb']} MB, MD5: {info['md5_hash'][:8]}...")
    
    print("\n📦 Paquetes principales:")
    for pkg in ['numpy', 'pandas', 'scikit-learn', 'joblib']:
        print(f"   {pkg}: {report['packages'].get(pkg, 'No instalado')}")
    
    print("\n📊 Datos:")
    if 'error' in report['data_validation']:
        print(f"   ❌ Error: {report['data_validation']['error']}")
    else:
        stats = report['data_validation']['data_stats']
        print(f"   Partidos: {stats['num_games']}")
        print(f"   Rango: {stats['date_range']['min']} a {stats['date_range']['max']}")
        print(f"   Equipos: {stats['teams_home']} locales, {stats['teams_away']} visitantes")
    
    print("\n" + "="*80)
    print(f"📝 Informes guardados en:\n   - {json_path}\n   - {txt_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
