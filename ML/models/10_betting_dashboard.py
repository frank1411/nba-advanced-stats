"""
Dashboard interactivo para evaluar recomendaciones de apuestas NBA.

Lee datos de betting tracking desde SQLite y permite ajustar umbrales
dinámicamente para ver cómo afecta el % de acierto.

Uso:
    streamlit run ML/models/10_betting_dashboard.py
"""
import sys
from pathlib import Path

# Asegurar que el directorio raíz esté en el path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from ML.db import NBADatabase

# Configuración de la página
st.set_page_config(
    page_title="NBA Betting Dashboard",
    page_icon="🏀",
    layout="wide"
)

# Título
st.title("🏀 NBA Betting Recommendations Dashboard")
st.caption("Datos desde SQLite — Over/Under Tracking")
st.markdown("---")


@st.cache_data(ttl=300)
def load_betting_data():
    """Carga datos de betting desde SQLite y los transforma para el dashboard."""
    with NBADatabase() as db:
        df = db.get_final_games()

    # Filtrar solo partidos con datos de betting
    betting_cols = ["LINE_OU_TOTAL", "PREDICTION_OU_TOTAL", "RESULT_OU_TOTAL"]
    available = [c for c in betting_cols if c in df.columns]
    if not available or df["LINE_OU_TOTAL"].isna().all():
        return pd.DataFrame()

    # Transformar a formato long (unexpanded records como el CSV anterior)
    records = []

    for target in ["TOTAL", "HOME", "AWAY"]:
        line_col = f"LINE_OU_{target}"
        pred_col = f"PREDICTION_OU_{target}"
        result_col = f"RESULT_OU_{target}"
        conf_col = f"CONFIDENCE_OU_{target}"

        cols_needed = [line_col, pred_col, result_col, conf_col]
        # Línea y predicción son obligatorios; el resultado y la confianza son
        # opcionales (no deben excluir filas del análisis)
        subset = df.dropna(subset=[c for c in [line_col, pred_col] if c in df.columns])

        if subset.empty:
            continue

        for _, row in subset.iterrows():
            line = row[line_col] if line_col in row else None
            pred = row[pred_col] if pred_col in row else None
            result = row[result_col] if result_col in row else None
            conf = row[conf_col] if conf_col in row else None

            if line is None or pred is None:
                continue

            threshold = pred - line
            bet_type = "Over" if pred > line else "Under"

            # Puntos reales
            if target == "TOTAL":
                actual_value = row["PTS_HOME"] + row["PTS_AWAY"]
                actual_line = line
            elif target == "HOME":
                actual_value = row["PTS_HOME"]
                actual_line = line
            else:
                actual_value = row["PTS_AWAY"]
                actual_line = line

            difference = actual_value - line

            market_map = {"TOTAL": "Partido", "HOME": "Local", "AWAY": "Visitante"}

            # Sin resultado evaluado → NaN (no cuenta ni como acierto ni como fallo)
            was_correct = np.nan if pd.isna(result) else (1 if result == "Hit" else 0)

            records.append({
                "game_date": row["GAME_DATE"],
                "away_team": row.get("TEAM_ABBREVIATION_AWAY", ""),
                "home_team": row.get("TEAM_ABBREVIATION_HOME", ""),
                "market": market_map[target],
                "bet_type": bet_type,
                "threshold": threshold,
                "has_star": (conf is not None and not pd.isna(conf) and conf >= 0.65),
                "away_line": row.get("LINE_OU_AWAY"),
                "home_line": row.get("LINE_OU_HOME"),
                "total_line": row.get("LINE_OU_TOTAL"),
                "actual_away": row.get("PTS_AWAY"),
                "actual_home": row.get("PTS_HOME"),
                "actual_total": row.get("PTS_HOME", 0) + row.get("PTS_AWAY", 0),
                "was_correct": was_correct,
                "difference": difference,
                "confidence": conf,
                "prediction": pred,
                "line": line,
            })

    return pd.DataFrame(records)


df = load_betting_data()

if df.empty:
    st.warning("⚠️ No hay datos de betting en la BD. Ejecuta primero:")
    st.code("python3 ML/scripts/add_betting_columns.py\npython3 ML/scripts/08_5_populate_betting_data.py")
    st.stop()

# Filtrar solo registros con resultados reales
df_with_results = df[df["was_correct"].notna()].copy()

if df_with_results.empty:
    st.warning("⚠️ No hay registros con resultados evaluados aún.")
    st.stop()

# Sidebar con controles
st.sidebar.header("⚙️ Configuración")

# Slider para umbral mínimo (valor absoluto)
threshold_min = st.sidebar.slider(
    "Umbral Mínimo (valor absoluto)",
    min_value=0.0,
    max_value=20.0,
    value=4.0,
    step=0.5,
    help="Filtra recomendaciones por umbral mínimo (valor absoluto de la diferencia con la línea)"
)

# Selector de mercado
market_filter = st.sidebar.selectbox(
    "Mercado",
    ["Todos", "Visitante", "Local", "Partido"],
    help="Selecciona el mercado a analizar"
)

# Selector de tipo de apuesta
bet_type_filter = st.sidebar.selectbox(
    "Tipo de Apuesta",
    ["Todos", "Over", "Under"],
    help="Selecciona el tipo de apuesta"
)

# Filtrar solo con estrella
star_only = st.sidebar.checkbox(
    "Solo con estrella ⭐",
    value=False,
    help="Filtra solo recomendaciones con confianza >= 0.65"
)

# Aplicar filtros
filtered_df = df_with_results.copy()

if market_filter != "Todos":
    filtered_df = filtered_df[filtered_df["market"] == market_filter]

if bet_type_filter != "Todos":
    filtered_df = filtered_df[filtered_df["bet_type"] == bet_type_filter]

if star_only:
    filtered_df = filtered_df[filtered_df["has_star"] == True]

# Filtro 3.1 (Umbrales Asimétricos)
use_filter_31 = st.sidebar.checkbox(
    "Aplicar Filtro 3.1 🚀",
    value=True,
    help="Aplica umbrales asimétricos por mercado"
)

if use_filter_31:
    local_mask = (filtered_df["market"] == "Local") & (abs(filtered_df["threshold"]) >= 14.0)
    partido_mask = (filtered_df["market"] == "Partido") & (abs(filtered_df["threshold"]) >= 5.0)
    visitante_mask = (filtered_df["market"] == "Visitante") & (abs(filtered_df["threshold"]) >= 5.0)

    filtered_df = filtered_df[local_mask | partido_mask | visitante_mask]
else:
    # Filtrar por umbral general (slider)
    filtered_df = filtered_df[abs(filtered_df["threshold"]) >= threshold_min]

# Validar si hay registros después de filtrar
if len(filtered_df) == 0:
    st.warning("⚠️ No se encontraron registros con los filtros seleccionados. Intenta ajustar los filtros.")
    st.stop()

# Métricas principales
st.header("📊 Métricas Principales")

col1, col2, col3, col4 = st.columns(4)

total_recs = len(filtered_df)
correct_recs = int(filtered_df["was_correct"].sum())
accuracy = (correct_recs / total_recs * 100) if total_recs > 0 else 0
avg_threshold = filtered_df["threshold"].mean() if len(filtered_df) > 0 else 0

col1.metric("Total Recomendaciones", total_recs)
col2.metric("Aciertos", correct_recs)
col3.metric("Accuracy", f"{accuracy:.1f}%")
col4.metric("Umbral Promedio", f"{avg_threshold:.1f}")

st.markdown("---")

# Análisis por mercado
st.header("📈 Análisis por Mercado")

if market_filter == "Todos":
    col1, col2, col3 = st.columns(3)

    for i, market in enumerate(["Visitante", "Local", "Partido"]):
        market_df = filtered_df[filtered_df["market"] == market]
        total = len(market_df)
        correct = int(market_df["was_correct"].sum()) if total > 0 else 0
        acc = (correct / total * 100) if total > 0 else 0

        if i == 0:
            with col1:
                st.subheader(f"🏃 {market}")
                st.metric("Total", total)
                st.metric("Aciertos", correct)
                st.metric("Accuracy", f"{acc:.1f}%")
        elif i == 1:
            with col2:
                st.subheader(f"🏠 {market}")
                st.metric("Total", total)
                st.metric("Aciertos", correct)
                st.metric("Accuracy", f"{acc:.1f}%")
        else:
            with col3:
                st.subheader(f"🎯 {market}")
                st.metric("Total", total)
                st.metric("Aciertos", correct)
                st.metric("Accuracy", f"{acc:.1f}%")
else:
    market_df = filtered_df[filtered_df["market"] == market_filter]
    total = len(market_df)
    correct = int(market_df["was_correct"].sum()) if total > 0 else 0
    acc = (correct / total * 100) if total > 0 else 0

    st.subheader(f"📊 {market_filter}")
    st.metric("Total", total)
    st.metric("Aciertos", correct)
    st.metric("Accuracy", f"{acc:.1f}%")

st.markdown("---")

# Gráfico de accuracy por umbral
st.header("🎯 Accuracy por Umbral")

if market_filter == "Todos":
    markets = ["Visitante", "Local", "Partido"]
    fig = go.Figure()

    colors = {"Visitante": "#1f77b4", "Local": "#ff7f0e", "Partido": "#2ca02c"}

    for market in markets:
        market_df = filtered_df[filtered_df["market"] == market]
        if len(market_df) > 0:
            # Agrupar por threshold (binning para suavizar)
            market_df = market_df.copy()
            market_df["threshold_bin"] = market_df["threshold"].round(1)
            threshold_stats = market_df.groupby("threshold_bin").agg({
                "was_correct": ["sum", "count"]
            }).reset_index()
            threshold_stats.columns = ["threshold", "correct", "total"]
            threshold_stats["accuracy"] = (threshold_stats["correct"] / threshold_stats["total"] * 100)
            threshold_stats = threshold_stats.sort_values("threshold")

            fig.add_trace(go.Scatter(
                x=threshold_stats["threshold"],
                y=threshold_stats["accuracy"],
                mode="lines+markers",
                name=market,
                line=dict(color=colors.get(market), width=2),
                marker=dict(size=8)
            ))
else:
    market_df = filtered_df[filtered_df["market"] == market_filter]
    if len(market_df) > 0:
        market_df = market_df.copy()
        market_df["threshold_bin"] = market_df["threshold"].round(1)
        threshold_stats = market_df.groupby("threshold_bin").agg({
            "was_correct": ["sum", "count"]
        }).reset_index()
        threshold_stats.columns = ["threshold", "correct", "total"]
        threshold_stats["accuracy"] = (threshold_stats["correct"] / threshold_stats["total"] * 100)
        threshold_stats = threshold_stats.sort_values("threshold")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=threshold_stats["threshold"],
            y=threshold_stats["accuracy"],
            mode="lines+markers",
            name=market_filter,
            line=dict(width=3, color="#1f77b4"),
            marker=dict(size=10)
        ))

fig.update_layout(
    title="Accuracy de Recomendaciones por Umbral",
    xaxis_title="Umbral (puntos)",
    yaxis_title="Accuracy (%)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, width="stretch")

st.markdown("---")

# Gráfico de confianza vs accuracy
st.header("🎲 Confianza vs Accuracy")

if filtered_df["confidence"].notna().any():
    conf_df = filtered_df.dropna(subset=["confidence"])
    conf_df = conf_df.copy()
    conf_df["conf_bin"] = pd.cut(conf_df["confidence"], bins=[0, 0.4, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0])

    conf_stats = conf_df.groupby("conf_bin", observed=True).agg({
        "was_correct": ["sum", "count"]
    }).reset_index()
    conf_stats.columns = ["conf_range", "correct", "total"]
    conf_stats["accuracy"] = (conf_stats["correct"] / conf_stats["total"] * 100)
    conf_stats["conf_label"] = conf_stats["conf_range"].astype(str)

    fig_conf = go.Figure()
    fig_conf.add_trace(go.Bar(
        x=conf_stats["conf_label"],
        y=conf_stats["accuracy"],
        text=conf_stats["total"],
        textposition="outside",
        marker_color="#1f77b4"
    ))
    fig_conf.update_layout(
        title="Accuracy por Rango de Confianza",
        xaxis_title="Rango de Confianza",
        yaxis_title="Accuracy (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig_conf, width="stretch")
    st.markdown("---")

# Tabla de datos detallados
st.header("📋 Datos Detallados")

with st.expander("Ver tabla de datos"):
    display_cols = [
        "game_date", "away_team", "home_team", "market", "bet_type",
        "threshold", "confidence", "has_star", "total_line",
        "actual_home", "actual_away", "actual_total", "was_correct", "difference"
    ]
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(
        filtered_df[available_cols].sort_values("game_date", ascending=False),
        width="stretch",
        hide_index=True
    )

st.markdown("---")

# Instrucciones
st.header("📖 Instrucciones")
st.markdown("""
### Cómo usar este dashboard:

1. **Umbral Mínimo**: Ajusta el slider para filtrar recomendaciones por umbral mínimo.
   - Más alto = Menos recomendaciones pero más confiables
   - Más bajo = Más recomendaciones pero menos confiables

2. **Mercado**: Selecciona el mercado a analizar (Visitante, Local, Partido o Todos)

3. **Tipo de Apuesta**: Filtra por Over o Under

4. **Solo con estrella ⭐**: Filtra solo recomendaciones con confianza >= 0.65

5. **Filtro 3.1 🚀**: Aplica umbrales asimétricos por mercado (Local≥14, Partido≥5, Visitante≥5)

### Métricas:

- **Accuracy**: Porcentaje de recomendaciones acertadas
- **Umbral Promedio**: Promedio de umbrales de las recomendaciones filtradas

### Gráfico de Accuracy por Umbral:

Este gráfico muestra cómo varía el accuracy según el umbral de la recomendación.
El eje X representa el umbral y el eje Y el % de acierto.
""")
