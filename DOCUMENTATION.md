# Documentación del Sistema de Predicción de la NBA

## Tabla de Contenidos
1. [Visión General](#visión-general)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Flujo de Trabajo](#flujo-de-trabajo)
4. [Uso del Sistema](#uso-del-sistema)
5. [Mantenimiento y Actualización](#mantenimiento-y-actualización)
6. [Preguntas Frecuentes](#preguntas-frecuentes)

## Visión General

Este sistema está diseñado para predecir los resultados de partidos de la NBA utilizando técnicas de aprendizaje automático. El sistema sigue un flujo de trabajo completo que incluye la recopilación de datos, preprocesamiento, entrenamiento de modelos y generación de predicciones.

## Estructura del Proyecto

```
nba_advanced_stats/
├── ML/
│   ├── data/
│   │   ├── raw/                # Datos sin procesar
│   │   ├── processed/          # Datos procesados
│   │   └── predictions/        # Predicciones generadas
│   │
│   ├── models/                 # Modelos entrenados y metadatos
│   │   ├── model_PTS_HOME.pkl  # Modelo para puntos locales
│   │   ├── model_PTS_AWAY.pkl  # Modelo para puntos visitantes
│   │   └── training_metadata.json  # Metadatos del entrenamiento
│   │
│   ├── predict/                # Módulo de generación de predicciones
│   │   ├── __init__.py
│   │   └── generate.py
│   │
│   └── utils/                  # Utilidades (opcional)
│
├── scripts/

## Características Generadas

El sistema genera un conjunto completo de características para cada partido, que incluye:

### 1. Estadísticas de Rendimiento
- `HOME_WIN_PCT`: Porcentaje de victorias del equipo local en la temporada
- `AWAY_WIN_PCT`: Porcentaje de victorias del equipo visitante en la temporada
- `WIN_PCT_DIFF`: Diferencia en el porcentaje de victorias entre equipos

### 2. Rendimiento Reciente
- `HOME_LAST_5`: Promedio de victorias en los últimos 5 partidos (local)
- `AWAY_LAST_5`: Promedio de victorias en los últimos 5 partidos (visitante)
- `LAST_5_DIFF`: Diferencia en el rendimiento reciente

### 3. Eficiencia Ofensiva/Defensiva
- `HOME_POINTS_FOR`: Promedio de puntos anotados por el equipo local
- `AWAY_POINTS_FOR`: Promedio de puntos anotados por el equipo visitante
- `HOME_POINTS_AGAINST`: Promedio de puntos permitidos por el equipo local
- `AWAY_POINTS_AGAINST`: Promedio de puntos permitidos por el equipo visitante
- `POINTS_FOR_DIFF`: Diferencia en puntos anotados
- `POINTS_AGAINST_DIFF`: Diferencia en puntos permitidos

### 4. Factores de Contexto
- `DAYS_REST_HOME`: Días de descanso del equipo local
- `DAYS_REST_AWAY`: Días de descanso del equipo visitante
- `DAYS_REST_DIFF`: Diferencia en días de descanso
- `H2H_WIN_PCT`: Porcentaje de victorias en enfrentamientos directos
- `SEASON`: Temporada del partido
- `SEASON_TYPE`: Tipo de partido (Regular/Playoffs)
- `GAME_DATE`: Fecha del partido

Estas características se utilizan para alimentar los modelos de predicción y se generan automáticamente para los próximos partidos utilizando datos históricos.

## Uso del Sistema

### Requisitos Previos

- Python 3.8+
- Dependencias listadas en `requirements.txt`

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd nba_advanced_stats
   ```

2. Crear y activar un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Ejecución del Pipeline Completo

Para ejecutar todo el flujo de trabajo:

```bash
python scripts/run_pipeline.py --all
```

### Actualización de Datos Históricos

Para actualizar solo los datos históricos:

```bash
python scripts/run_pipeline.py --update-historical --start-season 2020 --end-season 2025
```

### Entrenamiento del Modelo

Para reentrenar los modelos con los datos actuales:

```bash
python scripts/run_pipeline.py --retrain --test-season 2025
```

### Generación de Predicciones

Para generar predicciones para los próximos partidos:

```bash
python scripts/run_pipeline.py --predict
```

## Mantenimiento y Actualización

### Actualización de Datos

El sistema está diseñado para actualizarse automáticamente con nuevos partidos. Para forzar una actualización manual:

```bash
python -m ML.data.update_historical
```

### Reentrenamiento del Modelo

Se recomienda reentrenar los modelos periódicamente (por ejemplo, semanalmente) para incorporar nuevos datos. Para forzar un reentrenamiento manual:

```bash
python -m ML.models.train
```

### Generación de Nuevas Predicciones

Para generar predicciones en cualquier momento:

```bash
python -m ML.predict.generate
```

## Preguntas Frecuentes

### ¿Cómo se manejan los partidos recién disputados?

Los partidos recién disputados se agregan automáticamente al conjunto de datos históricos la próxima vez que se ejecute el script de actualización. El sistema verifica la existencia de nuevos partidos comparando con los ya almacenados.

### ¿Con qué frecuencia debo reentrenar los modelos?

Se recomienda reentrenar los modelos al menos una vez por semana durante la temporada regular, y después de la temporada regular para incluir todos los partidos de playoffs.

### ¿Cómo puedo mejorar la precisión de las predicciones?

1. Añadir más características relevantes al conjunto de datos
2. Ajustar los hiperparámetros de los modelos
3. Probar diferentes algoritmos de aprendizaje automático
4. Aumentar la cantidad de datos de entrenamiento

### ¿Dónde se guardan las predicciones?

Las predicciones se guardan en `ML/data/predictions/` con un nombre que incluye la fecha y hora de generación.

### ¿Cómo interpreto las probabilidades de victoria?

La columna `IMPLIED_HOME_WIN_PROB` representa la probabilidad estimada de que el equipo local gane el partido, basada en la diferencia de puntos predicha y la desviación estándar de los errores del modelo.

## Solución de Problemas

### Error: No se encuentran los datos históricos

Asegúrate de que el archivo `nba_games_final.parquet` existe en `ML/data/processed/`. Si no existe, ejecuta primero la actualización de datos históricos.

### Error: Falta alguna dependencia

Instala todas las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

### Las predicciones no son precisas

1. Verifica que los datos estén actualizados
2. Revisa las métricas de entrenamiento en `ML/models/training_metadata.json`\.
3. Considera ajustar los parámetros del modelo o añadir más características relevantes.

## Licencia

[Incluir información sobre la licencia del proyecto]
