## 📋 Lista Completa de Scripts — NBA Advanced Stats

### 🔹 Extracción de Datos

| # | Script | Descripción | Fuente → Destino |
|---|---|---|---|
| 01 | `ML/preprocessing/01_fetch_nba_data.py` | Descarga datos históricos de la API de la NBA (2018 en adelante) | API NBA → **SQLite (`nba_games_raw`)** |
| 02 | `ML/preprocessing/02_update_nba_data.py` | Descarga **solo partidos nuevos** y los agrega a la BD; **calcula RESULT_OU_*** | API NBA → **SQLite (`nba_games_raw`) + betting results** |

### 🔹 Procesamiento

| # | Script | Descripción | Entrada → Salida |
|---|---|---|---|
| 03 | `ML/preprocessing/03_process_final_dataset.py` | Calcula features diferenciales (Diff, PctDiff, Trend) y variable TARGET | SQLite raw → **SQLite (`nba_games_final`)** |
| 03.1 | `ML/preprocessing/calculate_advanced_stats.py` | Calcula Net Efficiency, Tendencies y métricas avanzadas | SQLite raw → **SQLite (`nba_advanced_stats`)** |
| 03.2 | `ML/preprocessing/create_master_dataset.py` | Consolida todo en un dataset maestro | SQLite advanced → **SQLite (`nba_master_dataset`)** |

### 🔹 Modelos de ML

| # | Script | Descripción | Entrada → Salida |
|---|---|---|---|
| 04 | `ML/models/04_train.py` | Entrena RandomForest para predecir PTS_HOME y PTS_AWAY | SQLite → **modelos `.pkl` + métricas** |
| 05 | `ML/models/05_train_model_advanced.py` | Entrena RandomForest + XGBoost con split temporal | SQLite (o `--input`) → **modelos `.pkl` + métricas** |
| 06 | `ML/models/06_predict_upcoming_games.py` | Obtiene calendario, genera predicciones y **las guarda en BD** (PREDICTION_OU_*) | SQLite + modelos → **pronósticos guardados en BD + CSV + HTML** |

### 🔹 Predicciones y Recomendaciones

| # | Script | Descripción | Entrada → Salida |
|---|---|---|---|
| predict | `ML/models/predict.py` | Inferencia simple: predice puntos para un dataset de entrada | SQLite (o archivo) → **CSV de predicciones** |
| predict_season | `ML/models/predict_season.py` | Genera predicciones para una temporada específica | SQLite → **pronósticos por temporada** |
| 07 | `ML/models/07_nba_api_recommendations.py` | Genera recomendaciones de apuestas usando la NBA API | API → **recomendaciones HTML/Markdown** |
| 08 | `ML/models/08_the_odds_api_recommendations.py` | Consulta The Odds API, genera recomendaciones y **guarda líneas en BD** (LINE_OU_*) | The Odds API → **recomendaciones HTML/MD + BD** |
| 08.5 | `ML/scripts/08_5_populate_betting_data.py` | Calcula pronósticos OU, líneas, resultados y confianza | SQLite → **actualiza columnas betting en BD** |

### 🔹 Tracking y Análisis

| # | Script | Descripción | Entrada → Salida |
|---|---|---|---|
| 09 | `ML/models/09_track_predictions.py` | Lee recomendaciones HTML y coteja con resultados reales | HTML + BD → **CSV de tracking** |
| 09.5 | `ML/analysis/09_5_threshold_analysis.py` | Analiza en qué umbrales el modelo es más preciso | SQLite → **reporte por consola** |
| 10 | `ML/models/10_betting_dashboard.py` | Dashboard interactivo con Streamlit para explorar umbrales | SQLite → **app web interactiva** |
| backtest | `ML/models/backtest.py` | Validación retrospectiva completa del modelo | SQLite → **métricas de rendimiento** |
| run_backtest | `ML/models/run_backtest.py` | Ejecuta backtest rápido con datos de una temporada | SQLite → **métricas rápidas** |

### 🔹 Scripts Auxiliares

| # | Script | Descripción |
|---|---|---|
| scripts/migrate | `ML/scripts/migrate_parquet_to_sqlite.py` | **Una sola vez**: migra archivos `.parquet` a SQLite |
| scripts/add_cols | `ML/scripts/add_betting_columns.py` | **Una sola vez**: agrega 12 columnas de betting a la BD |
| eval | `ML/scripts/eval_full_season.py` | Auditoría histórica de Smart Picks (lee HTML y datos reales) |
| verify | `ML/scripts/verify_environment.py` | Verifica que modelos, datos y BD existan y sean válidos |
| inspect | `ML/scripts/inspect_recent_games.py` | Muestra los últimos 9 partidos en la BD |
| check_local | `ML/check_local_games.py` | Muestra próximos partidos (próximos 7-14 días) |
| search_date | `ML/search_games_by_date.py` | Búsqueda interactiva de partidos por fecha |
| analyze | `ML/preprocessing/analyze_data.py` | Análisis exploratorio de calidad de datos |
| pipeline | `ML/run_pipeline.py` | Ejecuta todo el flujo (descarga + procesamiento) |
| build_hist | `ML/preprocessing/build_historical_dataset.py` | Construye dataset histórico completo |
| data/update | `ML/data/update_historical.py` | Actualización incremental de datos históricos (alternativa a 02) |

### 🔹 Utilidades

| # | Script | Descripción |
|---|---|---|
| test_api | `ML/test_api_connection.py` | Prueba conexión a la API de la NBA |
| check_games_today | `ML/models/check_games_today.py` | Verifica si hay partidos hoy |
| check_upcoming | `ML/models/check_upcoming_games.py` | Verifica partidos próximos |
| get_todays | `ML/models/get_todays_games.py` | Obtiene partidos de hoy |
| check_local_games | `ML/check_local_games.py` | Consulta partidos locales próximos |
| search_games | `ML/search_games_by_date.py` | Busca partidos por fecha |

---

## 🔄 Secuencia de Ejecución Diaria

```
02_update_nba_data.py              → descarga partidos nuevos + calcula RESULT_OU_*
03_process_final_dataset.py        → procesa con features diferenciales
04_train.py                        → entrena modelos
06_predict_upcoming_games.py       → predice próximos partidos + guarda PREDICTION_OU_* en BD
08_the_odds_api_recommendations.py → obtiene líneas reales + guarda LINE_OU_* en BD
08_5_populate_betting_data.py      → guarda tracking OU en BD (histórico sin líneas)
09_5_threshold_analysis.py         → analiza precisión por umbral
10_betting_dashboard.py            → dashboard interactivo (Streamlit)
```

## 📦 Una Sola Vez (Setup)

```
migrate_parquet_to_sqlite.py  → migrar parquet existentes a SQLite
add_betting_columns.py        → agregar 12 columnas de betting
```
