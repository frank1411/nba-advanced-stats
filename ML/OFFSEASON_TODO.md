# TODO: Mejoras del Modelo — Offseason 2026

> Generado: 2026-06-03 tras la corrección crítica de data leakage (commit `e7e2f7f`)

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

### 3. Calcular columnas TREND correctamente

**Problema:** 12 columnas `*_TREND` son todas ceros (std=0). Se generan en
`03_process_final_dataset.py` pero el cálculo nunca se ejecuta en el pipeline actual.

**Qué hacer:**
- [ ] Corregir `calculate_differential_features()` en `03_process_final_dataset.py` para
  calcular `L10 - season_avg` correctamente (actualmente compara `L10_HOME` contra `*_HOME`
  que es por-partido, no promedio de temporada)
- [ ] Alternativa: usar `L5 - L10` como tendencia (últimos 5 vs últimos 10)
- [ ] Si las TREND dan señal, re-incluirlas en el modelo

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
