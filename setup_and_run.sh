#!/bin/bash

# Cambiar al directorio del script
cd "$(dirname "$0")"

# Verificar si Python 3 está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado. Por favor instálalo primero."
    exit 1
fi

# Verificar si pip está instalado
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 no está instalado. Por favor instálalo primero."
    exit 1
fi

# Crear y activar entorno virtual (si no existe)
if [ ! -d "venv" ]; then
    echo "🔧 Creando entorno virtual..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Instalar dependencias
    echo "📦 Instalando dependencias..."
    pip3 install -r requirements.txt
else
    # Activar entorno virtual existente
    source venv/bin/activate
fi

# Verificar dependencias
echo "🔍 Verificando dependencias..."
pip3 install -r requirements.txt

# Ejecutar el script de estadísticas
echo "🏀 Iniciando la obtención de estadísticas de la NBA..."
python3 nba_stats_tool.py

echo "✅ Proceso completado. Los resultados se han guardado en la carpeta 'output/'."
