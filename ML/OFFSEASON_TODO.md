# TODO: Mejoras del Modelo — Offseason 2026

> Generado: 2026-06-03 tras la corrección crítica de data leakage (commit `e7e2f7f`)
> Actualizado: 2026-10-20 (noche inaugural 2026-27) tras sesión de revisión completa del pipeline

## Decisiones de la sesión 2026-10-20

- Pipeline 02→03→05→06→08→09→10 revisado, corregido y validado (upserts, SEASON deducido,
  tracking con SQLite fresco, guardrails de re-training en 05)
- Models re-trained (holdout 2025): RF para PTS_HOME (MAE 10.10), XGB para PTS_AWAY (MAE 9.32);
  backup previo en `ML/model_backups/20261020_135438/`
- Umbrales del filtro 3.1 calibrados con líneas históricas (n=144, in-sample):
  `THR_TOTAL=6.0`, `THR_AWAY=6.0` (verde, jugable), `THR_HOME=6.0` (**🔴 NO jugar**,
  ninguna banda histórica supera 53%)
- Hallazgo clave: picks Under históricamente 38-46% de acierto (sesgo a la baja del modelo,
  −2.8 pts local / −1.0 visitante en holdout); Over 48-54%

## Calendario de re-training acordado

- [ ] **8 juegos por equipo** (~10% de temporada, mediados de noviembre 2026)
- [ ] **Tras el Juego de Estrellas** (~febrero 2027; evaluar modo in-season, ver abajo)
- [ ] **Cierre de temporada regular** (abril 2027, el flujo clásico con holdout completo)

Nota: el `05` ya es seguro de correr cualquier día (guardrail NaN/placeholder + fallback
de holdout si `latest` tiene <100 juegos RS). El re-train de noviembre con holdout 2025
NO incluirá los partidos 2026-27 jugados (design actual: holdout excluye temporada actual).

## 🔴 Prioridad Alta (agregado 2026-10-20)

### 0. Puertaa direccional para picks Under + calibración de sesgo

**Problema:** El modelo subestima totales (−2.8 local / −1.0 visitante vs real en holdout).
Los picks Under rinden 38-46% y arrastran el accuracy global del O/U.

**Qué hacer:**
- [ ] Calibración de sesgo en `06` (post-procesado de predicciones: sumar offset estimado)
- [ ] O alternativamente puerta direccional en `08`: Under solo si |Δ| ≥ 10
- [ ] Re-calibrar umbrales de Filtro 3.1 cuando haya ≥400 juegos con línea

**Impacto objetivo:** pasar del ~50% global a >53% sostenido (break-even con vig)

### 0.1 Validar umbral de diferencia → victorias (idea del usuario, pendiente)

Analizar con qué margen de diferencia predicha los equipos realmente ganan por ese margen
o más (relación |pred_dif| vs resultado real POINT_DIFF). Complementa la calibración
direccional: definir cuándo el pick "ganador" es rentable y no solo el O/U.

**Qué hacer:**
- [ ] Usar `POINT_DIFF_TARGET` histórico vs predicciones guardadas de 06/09
- [ ] Tabla de margen predicho (5/10/15 pts) vs % de veces que el favorito cubrió

## Contexto

Se corrigió un problema de **data leakage** causado por la migración de parquet a SQLite.
Columnas como `PREDICTION_OU_*`, `LINE_OU_*`, `CONFIDENCE_OU_*` se colaron como features.
El modelo actual funciona con 60 features limpias, R²=0.18-0.25, MAE≈9.5 pts.

---

## 🔴 Prioridad Alta

### 1. Agregar `IS_PLAYOFFS` como feature

**Problema:** El modelo NO distingue entre Regular Season y Playoffs. El scoring en playoffs
es ~10 puntos menor (216.5 vs 226.0 total), lo que causa sobreestimación en playoffs.

**Qué hacer:**
- [ ] Agregar `IS_PLAYOFFS` (0/1) a la lista de features en `03_process_final_dataset.py`
- [ ] Excluir `IS_PLAYOFFS` del `exclude_cols` en `05_train_model_advanced.py`
- [ ] Actualizar `06_predict_upcoming_games.py` para generar `IS_PLAYOFFS` en features
- [ ] Re-entrenar y comparar métricas (esperamos R² suba ~0.05-0.10)

**Impacto:** ~10 pts de corrección en predicciones de playoffs

### 2. Evaluar columnas de la BD como features legítimas

**Problema:** La BD tiene columnas que se excluyeron por ser leakage (predicciones del modelo
mismo), pero `LINE_OU_*` (líneas de apuestas) son datos externos válidos.

**Qué hacer:**
- [ ] Separar `LINE_OU_*` (líneas de casas de apuestas → **feature válida**) de
  `PREDICTION_OU_*` y `CONFIDENCE_OU_*` (predicciones propias → **leakage, excluir**)
- [ ] Verificar cobertura de `LINE_OU_*` en la BD (actualmente ~10,036 nulls de 10,193)
- [ ] Si la cobertura es baja, integrar API de odds (The Odds API, DraftKings, etc.)
- [ ] Si la cobertura es buena, agregar como features y re-entrenar

**Impacto:** Las líneas de apuestas son el predictor más fuerte disponible (MAE típico ~5 pts)

---

## 🟡 Prioridad Media

### 3. Implementar snapshots diarios para columnas TREND

**Problema:** 12 columnas `*_TREND` son todas ceros. El API de la NBA devuelve
OFF_RATING, DEF_RATING, NET_RATING y PACE como **promedio de toda la temporada**
(un solo valor por equipo), que se asigna a los 82 juegos. No hay evolución temporal.

**La idea original (correcta):** Cada día que corre 02, el API da un valor ligeramente
distinto de PACE/OFF_RATING/etc. Si guardamos ese valor día a día, podemos calcular
`TREND = promedio(últimos 10 snapshots) - snapshot_actual` para ver si un equipo
mejora o empeora respecto a su nivel estacional.

**Qué hacer:**
- [ ] Crear tabla `daily_team_snapshots` en la BD:
  ```
  snapshot_date | team_abbr | pace | off_rating | def_rating | net_rating
  2025-11-15    | SAS       | 99.5 | 113.0      | 110.8      | 2.2
  2025-11-20    | SAS       | 99.8 | 112.8      | 111.0      | 1.8
  ```
- [ ] Modificar `02_update_nba_data.py` para guardar un snapshot diario al llamar al API
- [ ] Re-calcular las 12 columnas TREND usando esta tabla:
  - `OFF_RATING_HOME_TREND = avg(últimos 10 snapshots de OFF_RATING_HOME) - snapshot_actual`
  - Idem para DEF_RATING, NET_RATING, PACE (HOME y AWAY)
  - `_TREND_DIFF = TREND_HOME - TREND_AWAY`
- [ ] Si las TREND dan señal, re-incluirlas como features del modelo (actualmente excluidas)

**Nota:** En temporadas anteriores los snapshots son un solo valor por temporada
(carga masiva histórica). La señal de TREND solo será útil desde la temporada
donde se implementen los snapshots diarios.

### 4. Corregir `POINT_DIFF_L10_PCT_DIFF`

**Problema:** Media de -667 millones por división por cero cuando ambos equipos tienen
`POINT_DIFF_L10` cercano a 0.

**Qué hacer:**
- [ ] Usar `(home - away) / (std + 1e-10)` en lugar de división por la media
- [ ] O usar diferencia absoluta directamente (ya está en `POINT_DIFF_L10_DIFF`)

### 5. Mejorar el cálculo de `DAYS_REST`

**Problema:** `DAYS_REST` tiene valores negativos (-1) y extremos (154 días). Los 22 valores
>30 días son off-season y distorsionan el modelo.

**Qué hacer:**
- [ ] Cap `DAYS_REST` en un máximo razonable (ej: 30 días)
- [ ] Investigar por qué hay valores negativos (posible bug en ordenamiento de fechas)

---

## 🟢 Prioridad Baja / Nice to have

### 6. Agregar features de lesiones

- [ ] Integrar API de lesiones (NBA Injury Report, ESPN)
- [ ] Features: `HOME_INJURY_IMPACT`, `AWAY_INJURY_IMPACT` (basado en WAR/VORP del jugador)

### 7. Features de momentum

- [ ] Racha actual (W/L streak): `HOME_STREAK`, `AWAY_STREAK` (con signo: +3 = 3W, -2 = 2L)
- [ ] Rendimiento últimos 5 partidos: `HOME_L5_RECORD`, `AWAY_L5_RECORD`

### 8. Head-to-head histórico

- [ ] Historial directo entre los dos equipos (últimos 5 enfrentamientos)
- [ ] Ventaja psicológica del equipo

### 9. Modelo separado para playoffs

- [ ] Entrenar un modelo exclusivamente con datos de playoffs (~666 partidos × 8 temporadas = ~5,300)
- [ ] O usar transfer learning: fine-tuning del modelo de RS con datos de playoffs

### 10. Backtesting automático

- [ ] Script que compare predicciones vs resultados reales cada día
- [ ] Tracking de accuracy por tipo de apuesta (moneyline, spread, O/U)
- [ ] Dashboard de rendimiento del modelo

---

## 📋 Plan de Acción para Inicio de Temporada 2026-27

1. **Actualizar datos:** Ejecutar `01_fetch_nba_data.py` con datos de temporada 2025-26 completa
2. **Regenerar dataset:** Ejecutar `03_process_final_dataset.py` con correcciones de TREND y DAYS_REST
3. **Agregar IS_PLAYOFFS:** Implementar items 1 y 2 de prioridad alta
4. **Re-entrenar:** Ejecutar `05_train_model_advanced.py --model xgb,rf`
5. **Validar:** Comparar métricas con modelo anterior (R², MAE, dirección correcta)
6. **Deploy:** Reemplazar modelos en producción y verificar predicciones
