# NBA Advanced Stats - Machine Learning

Este proyecto tiene como objetivo crear un pipeline completo para la extracción, procesamiento y modelado de datos de la NBA con fines predictivos.

## Características principales

- Extracción de datos históricos de la NBA desde la temporada 2018-19 hasta la 2024-25
- Cálculo de estadísticas avanzadas y métricas de rendimiento
- Creación de un dataset maestro con características para modelado predictivo
- Pipeline automatizado para la actualización de datos

## Estructura del directorio

```
ML/
├── data/                   # Datos en bruto y procesados
│   ├── raw/               # Datos sin procesar de la API
│   └── processed/         # Datos procesados y listos para modelado
├── models/                # Modelos entrenados
├── notebooks/             # Jupyter notebooks para análisis
├── preprocessing/         # Scripts de procesamiento de datos
│   ├── fetch_nba_data.py         # Extracción de datos de la API
│   ├── calculate_advanced_stats.py # Cálculo de estadísticas avanzadas
│   └── create_master_dataset.py   # Creación del dataset final
├── utils/                 # Funciones auxiliares
├── run_pipeline.py        # Script principal para ejecutar todo el pipeline
└── requirements_ml.txt    # Dependencias del proyecto
```

## Requisitos

- Python 3.8+
- Dependencias listadas en `requirements_ml.txt`

## Instalación

1. Clona el repositorio
2. Crea un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements_ml.txt
   ```

## Uso

Para ejecutar todo el pipeline de procesamiento de datos:

```bash
python run_pipeline.py
```

### Scripts individuales

1. **Extraer datos de la API**:
   ```bash
   python -m preprocessing.fetch_nba_data
   ```

2. **Calcular estadísticas avanzadas**:
   ```bash
   python -m preprocessing.calculate_advanced_stats
   ```

3. **Crear dataset maestro**:
   ```bash
   python -m preprocessing.create_master_dataset
   ```

## Características del dataset

El dataset final (`nba_master_dataset.parquet`) incluye las siguientes características:

### Identificadores y contexto
- `game_id`, `game_date`, `season`
- IDs y abreviaturas de equipos local/visitante
- Indicador de partidos de playoffs

### Targets
- `puntos_local`, `puntos_visitante`

### Estadísticas de rendimiento
- Offensive/Defensive Rating (temporada y últimos 10 partidos)
- Pace y estadísticas de tiro
- Porcentajes de rebotes ofensivos/defensivos

### Características situacionales
- Días de descanso por equipo
- Indicadores de back-to-back

### Características diferenciales
- Diferencias en ratings ofensivos/defensivos
- Diferencias en ritmo de juego
- Diferencias en días de descanso

## Convenciones

- **Notebooks**: `[fecha]_[descripción].ipynb` (ej: `2024-09-18_eda_initial.ipynb`)
- **Modelos**: Guardar con nombre descriptivo y versión
- **Documentación**: Incluir docstrings y comentarios claros

## Contribución

1. Crea un fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT.
