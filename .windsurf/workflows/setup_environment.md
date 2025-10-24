---
description: Configura el entorno de desarrollo para el proyecto NBA Advanced Stats
---
# Configuración del Entorno NBA Advanced Stats

Este workflow se encarga de configurar automáticamente el entorno de desarrollo para el proyecto NBA Advanced Stats, incluyendo la creación de un entorno virtual y la instalación de dependencias.

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- git (opcional, para clonar el repositorio)

## Pasos del Workflow

1. **Verificar Python**
   ```bash
   python3 --version
   ```

2. **Crear entorno virtual (si no existe)**
   ```bash
   # Navegar al directorio del proyecto
   cd /Users/franklinsantaella/CascadeProjects/nba_advanced_stats
   
   # Crear entorno virtual
   python3 -m venv venv
   ```

3. **Activar el entorno virtual**
   ```bash
   # En sistemas Unix/macOS
   source venv/bin/activate
   
   # En Windows
   # venv\Scripts\activate
   ```

4. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```
   
   Si no existe requirements.txt, instalar las dependencias directamente:
   ```bash
   pip install pandas pyarrow python-dateutil pytz
   ```

5. **Ejecutar el script de actualización**
   ```bash
   cd ML/preprocessing
   python 02_update_nba_data.py
   ```

## Uso Rápido

Para configurar todo el entorno con un solo comando:

```bash
# En la raíz del proyecto
bash setup.sh
```

## Notas Importantes

- El entorno virtual se guarda en la carpeta `venv/` dentro del proyecto
- No es necesario activar el entorno manualmente si usas el script `setup.sh`
- Las dependencias se instalarán automáticamente al ejecutar el script

## Solución de Problemas

Si encuentras algún error:
1. Verifica que Python 3.8+ esté instalado
2. Asegúrate de tener permisos de escritura en el directorio del proyecto
3. Si hay conflictos de dependencias, intenta:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```
