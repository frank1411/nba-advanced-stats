# NBA Advanced Stats Analytics

Una herramienta completa para el análisis de datos avanzados de la NBA, incluyendo temporada regular y playoffs. Este proyecto permite obtener, procesar y analizar estadísticas avanzadas de la NBA para múltiples temporadas.

## 📊 Características

- **Obtención de datos**: Descarga automática de datos de la API de la NBA
- **Procesamiento avanzado**: Cálculo de métricas avanzadas como:
  - Eficiencia ofensiva y defensiva (ORTG/DRTG)
  - Net Rating y Pace
  - Diferencial de puntos
  - Días de descanso y partidos consecutivos (B2B)
  - Distancias de viaje entre partidos
- **Generación de características**: Cálculo de características diferenciales entre equipos
- **Dataset final**: Generación de un dataset listo para análisis y modelado predictivo

## 🚀 Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/frank1411/nba-advanced-stats.git
   cd nba-advanced-stats
   ```

2. Crea y activa un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 🏀 Uso

### 1. Ejecución Programada (GitHub Actions)

El sistema está configurado para ejecutarse automáticamente todos los días a las 10:00 AM (hora de Venezuela, UTC-4) a partir del 20 de octubre de 2025. Las predicciones se generarán automáticamente y estarán disponibles en la sección de "Artifacts" de GitHub Actions.

Para ver las ejecuciones programadas y los resultados:
1. Ve a la pestaña "Actions" en tu repositorio de GitHub
2. Selecciona el workflow "Ejecución Diaria de Predicciones NBA"
3. Revisa las ejecuciones más recientes y sus resultados

### 2. Ejecución Manual

Para ejecutar manualmente el pipeline completo (descarga de datos + generación de características):

```bash
python run_pipeline.py
```

### 2. Ejecutar componentes individuales

#### Descargar datos de la NBA:
```bash
python -m ML.preprocessing.fetch_nba_data
```

#### Procesar dataset final con características diferenciales:
```bash
python -m ML.preprocessing.process_final_dataset
```

### Estructura del proyecto

```
nba-advanced-stats/
├── ML/
│   ├── data/
│   │   ├── raw/                 # Datos en bruto de la API
│   │   └── processed/           # Datos procesados y listos para análisis
│   └── preprocessing/
│       ├── fetch_nba_data.py    # Script para descargar datos de la NBA
│       └── process_final_dataset.py  # Script para generar características
├── run_pipeline.py             # Script principal para ejecutar todo el flujo
└── requirements.txt            # Dependencias del proyecto
```

## 📋 Requisitos

- Python 3.8+
- Conexión a Internet para acceder a la API de la NBA
- Dependencias principales:
  - pandas
  - numpy
  - requests
  - pyarrow (para archivos Parquet)
  - python-dateutil

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, lee las guías de contribución para más información.

## 📧 Contacto

Tu Nombre - [@tu_twitter](https://twitter.com/tu_twitter) - tu@email.com

Enlace del proyecto: [https://github.com/tu-usuario/nba-advanced-stats](https://github.com/tu-usuario/nba-advanced-stats)
