# 📊 Bitácora de Precisión: Smart Picks 3.1
**Última Actualización:** 15 de Abril de 2026 (Muestra: Temporada 2025-26 + Play-In Day 1)

Este documento audita el desempeño de los "Filtros Asimétricos (3.1)" y la métrica de "Estabilidad" del modelo.

## 📈 Resumen Ejecutivo de Desempeño

| Métrica | Valor | Estado |
| :--- | :--- | :---: |
| **Total Picks Evaluados** | 71 | 🟢 Operativo |
| **Hit Rate Global** | **46.48%** | ⚠️ Ajuste Requerido |
| **Hit Rate 'Stable' (⭐)** | **29.63%** | ❌ Crítico |
| **Mejor Mercado** | **Visita (Away)** | 🏆 61.54% |
| **Peor Mercado** | **Local (Home)** | 📉 18.18% |

---

## 🔍 Desglose por Mercado (Eficiencia)

El sistema muestra una disparidad significativa entre el análisis de equipos locales y visitantes.

| Mercado | Aciertos | Total | Eficiencia | Tendencia |
| :--- | :---: | :---: | :---: | :--- |
| 🏘️ **Puntos Local** | 4 | 22 | **18.18%** | 📉 Muy Baja |
| ✈️ **Puntos Visita** | 16 | 26 | **61.54%** | 🥇 Excelente |
| 🏀 **Total Partido** | 13 | 23 | **56.52%** | ✅ Sólida |

> [!WARNING]
> **Sesgo de Localía Detectado**: El modelo está fallando sistemáticamente en los picks de "Local Under". La mayoría de los fallos ocurren porque el modelo predice una defensa local fuerte que no se materializa en los resultados reales.

---

## ⭐ Análisis de Filtros de Estabilidad (Stable Picks)

Los picks marcados con ⭐ (Alta Estabilidad) están rindiendo por debajo de los picks normales, lo cual es contra-intuitivo.

| Categoría | Aciertos | Total | Eficiencia |
| :--- | :---: | :---: | :---: |
| **Stable Local** | 1 | 14 | 7.14% |
| **Stable Visita** | 2 | 2 | 100.00% |
| **Stable Partido** | 5 | 11 | 45.45% |

**Observación Crítica**: El filtro de estabilidad para "Locales" es actualmente un indicador inverso (92.8% de error).

---

## 🕒 Auditoría de Ayer (14/04/2026)

| Partido | Pick | Línea | Real | Resultado |
| :--- | :--- | :---: | :---: | :---: |
| MIA @ CHA | 🟢 Under Local ⭐ | 118.0 | 127 | ❌ Fallo |
| POR @ PHX | 🟢 Over Local | 110.0 | 110 | ❌ Fallo (Push) |
| POR @ PHX | 🟢 Over Partido ⭐ | 217.5 | 224 | ✅ **HIT** |

---

## 🛠️ Próximas Acciones Sugeridas

1.  **Ajuste de Umbrales**: Incrementar el umbral `THR_HOME` de 4.0 a 6.0 para reducir el ruido en picks locales.
2.  **Inversión de Lógica Stable (Local)**: Revisar la condición `(pred_h < 113 and diff_h > 0)` en el script de recomendaciones, ya que parece estar penalizando el rendimiento.
3.  **Priorización**: Centrar las apuestas en **Visita** y **Total Partido** hasta que se recalibre el sesgo local.

---
*Generado automáticamente por el Sistema de Auditoría Smart Picks.*
