#!/bin/bash

# Colores para la salida
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Iniciando configuración del entorno NBA Advanced Stats...${NC}
"

# 1. Verificar Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "❌ Error: Python 3 no está instalado. Por favor instálalo primero."
    exit 1
fi

# 2. Verificar pip
if ! command -v pip3 &> /dev/null; then
    echo -e "❌ Error: pip3 no está instalado. Por favor instálalo primero."
    exit 1
fi

# 3. Crear/actualizar entorno virtual
echo -e "${GREEN}🔄 Configurando entorno virtual...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "✅ Entorno virtual creado en: ${PWD}/venv"
else
    echo -e "ℹ️  Usando entorno virtual existente en: ${PWD}/venv"
fi

# 4. Activar entorno virtual
echo -e "\n${GREEN}🚀 Activando entorno virtual...${NC}"
source venv/bin/activate

# 5. Actualizar pip
echo -e "\n${GREEN}🔄 Actualizando pip...${NC}"
pip install --upgrade pip

# 6. Instalar dependencias
echo -e "\n${GREEN}📦 Instalando dependencias...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Instalar dependencias directamente si no hay requirements.txt
    pip install pandas pyarrow python-dateutil pytz
    # Guardar las dependencias en requirements.txt
    pip freeze > requirements.txt
fi

# 7. Verificar instalación
echo -e "\n${GREEN}✅ Configuración completada con éxito!${NC}"
echo -e "\nPara activar el entorno manualmente, ejecuta:"
echo -e "${YELLOW}source venv/bin/activate${NC}"

# 8. Preguntar si se desea ejecutar el script
echo -e "\n¿Deseas ejecutar el script de actualización de datos ahora? [s/N]"
read -r response
if [[ "$response" =~ ^[sS]$ ]]; then
    echo -e "\n${GREEN}🚀 Ejecutando script de actualización...${NC}"
    cd ML/preprocessing
    python 02_update_nba_data.py
fi
