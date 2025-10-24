## 📋 Lista de Scripts del Proyecto NBA

### 1. **Extracción de Datos**
- **`ML/preprocessing/fetch_nba_data.py`**: Descarga datos de la API de la NBA
  - **Función**: Descarga datos históricos de partidos de la NBA desde la temporada 2018-19 hasta la actualidad
  - **Salida en consola**:
    - Progreso de descarga por temporada y tipo (Regular/Playoffs)
    - Estadísticas de equipos y partidos descargados
    - Resumen con total de partidos, equipos y rango de fechas
  - **Archivo generado**: `ML/data/raw/nba_games_raw.parquet`
  - **Contenido del archivo**:
    - Datos crudos de partidos con estadísticas básicas
    - Metadatos de equipos y jugadores
    - Fechas y ubicaciones de los partidos
- **`ML/data/update_historical.py`**: Actualiza datos históricos
  - **Función**: Actualiza incrementalmente los datos de partidos de la NBA para temporadas específicas
  - **Uso típico**: Ejecución periódica (ej: diaria) durante la temporada para mantener los datos actualizados
  - **Salida en consola**:
    - Progreso de actualización por temporada
    - Estadísticas de partidos nuevos encontrados
    - Resumen de la actualización
  - **Archivo generado**: `ML/data/processed/nba_games_final.parquet`
  - **Dependencias**:
    - Requiere `fetch_nba_data.py` para la lógica de descarga
    - Utiliza el mismo formato de datos que `fetch_nba_data.py`
  - **Características implementadas**:
    - Actualización incremental (solo descarga datos nuevos)
    - Manejo de errores y reintentos
    - Cálculo de días de descanso y distancias de viaje
    - Soporte para temporada regular y playoffs

### 2. **Procesamiento**
- **`ML/preprocessing/process_final_dataset.py`**: Procesa datos en crudo
  - **Función**: Transforma los datos crudos en características listas para el modelado
  - **Entrada principal**: `ML/data/raw/nba_games_raw.parquet`
  - **Salida principal**: `ML/data/processed/nba_games_final.parquet`
  - **Características generadas**:
    - Diferencias absolutas y porcentuales entre equipos
    - Tendencias de rendimiento reciente (últimos 10 partidos)
    - Estadísticas de rachas y momento de los equipos
    - Variables de contexto (días de descanso, viajes, B2B)
  - **Variable objetivo**:
    - Binaria: 1 si gana el local, 0 si gana el visitante
    - Numérica: Diferencia de puntos (PTS_HOME - PTS_AWAY)
  - **Métricas de salida**:
    - Distribución de la variable objetivo
    - Conteo de características generadas
    - Validación de calidad de datos
- **`ML/preprocessing/calculate_advanced_stats.py`**: Calcula estadísticas avanzadas
  - **Función**: Calcula métricas avanzadas basadas en ratings ofensivos, defensivos y ritmo de juego
  - **Entrada principal**: `ML/data/raw/nba_games_raw.parquet`
  - **Salida principal**: `ML/data/processed/nba_advanced_stats.parquet`
  
  ### Métricas Generadas
  
  1. **Eficiencia Ofensiva Neta**:
     - `HOME_NET_EFFICIENCY`: Eficiencia del equipo local vs defensa rival
     - `AWAY_NET_EFFICIENCY`: Eficiencia del equipo visitante vs defensa rival
     - `NET_EFFICIENCY_DIFF`: Diferencia en eficiencia neta entre equipos
  
  2. **Ritmo de Juego**:
     - `PACE_DIFF`: Diferencia en el ritmo de juego (posesiones/48min)
  
  3. **Tendencias Recientes**:
     - `HOME_TREND_OFF`: Tendencia ofensiva del equipo local (vs últimos 10 partidos)
     - `HOME_TREND_DEF`: Tendencia defensiva del equipo local
     - `AWAY_TREND_OFF`: Tendencia ofensiva del equipo visitante
     - `AWAY_TREND_DEF`: Tendencia defensiva del equipo visitante
     - `TREND_OFF_DIFF`: Diferencia en tendencias ofensivas
     - `TREND_DEF_DIFF`: Diferencia en tendencias defensivas
  
  4. **Otras Diferencias**:
     - `POINT_DIFF`: Diferencia de puntos en el partido
     - `POINT_DIFF_L10_HOME/AWAY`: Diferencia de puntos en últimos 10 partidos
  
  ### Procesamiento
  - Se procesaron 8,871 partidos
  - Datos guardados en formato Parquet para eficiencia
  - Incluye todas las columnas originales más las nuevas métricas
  
  ### Uso de las Métricas
  - Las diferencias positivas en `NET_EFFICIENCY_DIFF` indican ventaja del equipo local
  - `TREND_*` valores positivos indican mejora reciente en el rendimiento
  - `PACE_DIFF` positivo indica que el equipo local prefiere un ritmo más rápido
- **`ML/preprocessing/create_master_dataset.py`**: Crea el dataset maestro
  - **Función**: Consolida todas las características en un dataset final listo para modelado
  - **Entradas**:
    - `ML/data/processed/nba_advanced_stats.parquet` (datos con estadísticas avanzadas)
  - **Salida principal**: `ML/data/processed/nba_master_dataset.parquet`
  
  ### Proceso de Creación
  
  1. **Preparación de Datos**:
     - Ordena los partidos por fecha
     - Calcula días de descanso entre partidos para cada equipo
     - Identifica partidos back-to-back
  
  2. **Características Generadas**:
     - **Contexto del Partido**:
       - ID del partido y fecha
       - Temporada y tipo de partido (temporada regular/playoffs)
       - Equipos (local y visitante) con sus IDs y abreviaturas
     - **Métricas de Rendimiento**:
       - Puntos anotados por cada equipo
       - Diferencial de rating ofensivo/defensivo
       - Diferencia en ritmo de juego (PACE)
       - Eficiencia de tiro (eFG%)
     - **Factores de Contexto**:
       - Días de descanso para cada equipo
       - Indicadores de partidos back-to-back
       - Diferencias en días de descanso entre equipos
  
  3. **Características Diferenciales**:
     - `diff_off_rating`: Diferencia entre ofensiva local y defensa visitante
     - `diff_def_rating`: Diferencia entre ofensiva visitante y defensa local
     - `diff_pace`: Diferencia en ritmo de juego
     - `diff_net_rating`: Diferencia en rating neto entre equipos
     - `diff_descanso`: Diferencia en días de descanso
     - `home_trend_off`: Tendencia ofensiva del equipo local
     - `home_trend_def`: Tendencia defensiva del equipo local
     - `away_trend_off`: Tendencia ofensiva del equipo visitante
     - `away_trend_def`: Tendencia defensiva del equipo visitante
     - `net_efficiency_diff`: Diferencia en eficiencia neta
  
  4. **Salida de la Ejecución**:
     ```
     Cargando datos procesados desde ML/data/processed/nba_advanced_stats.parquet...
     Creando dataset maestro...
     Procesando partidos: 100%|█| 8871/8871 [00:01<00:00, 5447.25
     Dataset maestro guardado en ML/data/processed/nba_master_dataset.parquet
     
     Resumen del dataset:
     - Número de partidos: 8,871
     - Número de características: 84
     
     Primeras filas del dataset:
           game_id  ... net_efficiency_diff
     0  0021800001  ...                -2.3
     1  0021800002  ...                 7.3
     2  0021800013  ...                 0.7
     3  0021800009  ...                 2.3
     4  0021800006  ...                 2.0
     ```
  
  5. **Estructura del Dataset**:
     - **Total de partidos procesados**: 8,871
     - **Total de características generadas**: 84
     - **Formato de salida**: Parquet (optimizado para análisis de datos)
     - **Ubicación del archivo**: `ML/data/processed/nba_master_dataset.parquet`
     - **Listo para entrenamiento de modelos**
     - **Incluye variables objetivo para predicción**

### 3. **Modelos de ML**
  - **`ML/models/train.py`**: Entrena modelos de predicción
    - **Función principal**: `train_model()`
    - **Algoritmo principal**: Random Forest Regressor
    - **Características clave**:
      - Entrena modelos separados para predecir puntos del equipo local y visitante
      - Maneja características numéricas de rendimiento y contexto
      - Divide automáticamente los datos en conjuntos de entrenamiento y prueba
      - Evalúa el rendimiento con métricas estándar (MAE, RMSE, R²)
      - Guarda los modelos entrenados y sus metadatos
    - **Entrada principal**: `ML/data/processed/nba_games_final.parquet`
    - **Salidas**:
      - Modelos entrenados (`.pkl`)
      - Metadatos de entrenamiento (`.json`)
      - Métricas de rendimiento (`.txt`)
    - **Resultados del Entrenamiento**:
      - **Modelo PTS_HOME (Equipo Local)**:
        - MAE: 7.43 puntos
        - RMSE: 9.35 puntos
        - R²: 0.4413
      - **Modelo PTS_AWAY (Equipo Visitante)**:
        - MAE: 7.48 puntos
        - RMSE: 9.40 puntos
        - R²: 0.4516
    - **Interpretación de Métricas**:
      - **MAE**: Error promedio de ~7.5 puntos por partido
      - **R²**: Los modelos explican aproximadamente el 45% de la varianza
      - **RMSE**: Error típico de ~9.4 puntos, con penalización por errores grandes
    - **Archivos Generados**:
      - `model_PTS_HOME.pkl`: Modelo para predecir puntos locales
      - `model_PTS_AWAY.pkl`: Modelo para predecir puntos visitantes
      - `training_metadata.json`: Metadatos del entrenamiento
      - `model_metrics.txt`: Resumen de métricas en formato legible
    - **Consideraciones**:
      - Los modelos tienen un rendimiento similar para ambos equipos
      - El error promedio es consistente con la variabilidad típica en la NBA
      - Se utilizó la temporada más reciente (2024) para validación

- **`ML/models/train_model.py`**: Implementación del entrenamiento
  - **Propósito**: Entrenar modelos de regresión para predecir puntos de equipos locales (PTS_HOME) y visitantes (PTS_AWAY) usando estadísticas históricas y contextuales de la NBA.
  - **Características principales**:
    - Soporte para múltiples modelos (RandomForest, XGBoost)
    - División temporal de datos (holdout por temporada)
    - Evaluación con métricas estándar (MAE, RMSE, R²)
    - Manejo de datos temporales y validación cruzada
  - **Funcionalidades clave**:
    - `load_data()`: Carga y valida los datos de entrada
    - `get_feature_columns()`: Selecciona automáticamente las características relevantes
    - `temporal_split()`: Divide los datos por temporada para validación
    - `build_models()`: Configura los modelos a entrenar
    - `evaluate_regression()`: Calcula métricas de rendimiento
  - **Uso**:
    ```bash
    python -m ML.models.train_model \
      --input ML/data/processed/nba_games_final.parquet \
      --holdout-season latest \
      --model rf,xgb
    ```
  - **Parámetros de configuración**:
    - `--input`: Ruta al archivo de datos procesados
    - `--holdout-season`: Temporada a usar para validación ('latest' para la más reciente)
    - `--model`: Modelos a entrenar (rf: RandomForest, xgb: XGBoost)
  - **Salidas**:
    - Modelos entrenados (`.pkl`)
    - Métricas de evaluación (`.json`)
    - Lista de características utilizadas (`.json`)
  - **Consideraciones**:
    - Excluye automáticamente columnas con fuga de información
    - Maneja tipos de datos y conversiones necesarias
    - Optimizado para rendimiento con procesamiento en paralelo
- **`ML/models/backtest.py`**: Prueba los modelos con datos históricos
  - **Propósito**: Realizar validación retrospectiva de los modelos de predicción de la NBA para evaluar su rendimiento en datos históricos.
  - **Características principales**:
    - Validación temporal (time-based cross-validation)
    - Evaluación de predicciones de puntos y ganadores
    - Análisis de confianza de las predicciones
    - Métricas detalladas de rendimiento
  - **Funcionalidades clave**:
    - `load_data()`: Carga y prepara los datos de partidos
    - `temporal_split()`: Divide los datos por temporada
    - `evaluate_predictions()`: Calcula métricas de rendimiento
    - `run_backtest()`: Ejecuta el proceso completo de backtesting
  - **Métricas de evaluación**:
    - **Para predicciones de puntos**:
      - Error Absoluto Medio (MAE)
      - Raíz del Error Cuadrático Medio (RMSE)
      - Media de puntos reales vs predichos
    - **Para predicciones de ganadores**:
      - Precisión (Accuracy)
      - Matriz de confusión
      - Precisión, Recall y F1-Score
      - Precisión por rangos de confianza
  - **Uso**:
    ```bash
    python -m ML.models.backtest \
      --input ML/data/processed/nba_games_final.parquet \
      --test-season 2024
    ```
  - **Salidas**:
    - Métricas de rendimiento detalladas
    - Gráficos de rendimiento (si se habilita)
    - Resumen de precisión por rangos de confianza
  - **Consideraciones**:
    - Requiere modelos pre-entrenados (`model_PTS_HOME.pkl` y `model_PTS_AWAY.pkl`)
    - Asume una desviación estándar de 11.5 puntos para el cálculo de probabilidades
    - Genera un análisis detallado de la precisión en diferentes rangos de confianza
  - **Ejemplo de salida**:
    ```
    📊 Evaluando predicciones de puntos:
    - Puntos locales (MAE: 7.5, RMSE: 9.4)
    - Puntos visitantes (MAE: 7.6, RMSE: 9.5)
    
    🎯 Precisión en predicción de ganadores:
    - Precisión general: 68.5%
    - Precisión por confianza:
      * Baja confianza: 58.2%
      * Media confianza: 65.3%
      * Alta confianza: 72.1%
    ```

### 4. **Predicciones**
- **`ML/predict/generate.py`**: Genera predicciones para próximos partidos
- **`ML/models/predict.py`**: Implementación de la lógica de predicción

### 5. **Ejecución Principal**
- **`run_pipeline.py`**: Ejecuta todo el flujo (descarga + procesamiento)
- **`ML/run_pipeline.py`**: Versión alternativa del pipeline

### 6. **Utilidades**
- **`ML/preprocessing/analyze_data.py`**: Herramientas de análisis de datos
- **`nba_stats.py`**: Funciones auxiliares para estadísticas

### 7. **Configuración**
- **`requirements.txt`**: Dependencias del proyecto
- **`.github/workflows/`**: Configuración de CI/CD
