"""Page 3 — Impact Apache Spark : benchmark et scalabilite"""

import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import load_json, get_image_path

st.set_page_config(page_title="Impact Spark — MCV", page_icon="⚡", layout="wide")

st.markdown("# ⚡ Impact de Apache Spark")
st.markdown("Benchmark de performance : traitement distribue vs standalone")
st.divider()

# ── Chargement ────────────────────────────────────────────────────────────────
bench   = load_json("spark_benchmark.json")
dnn     = load_json("dnn_metrics.json")
xgb     = load_json("xgboost_metrics.json")
spark   = load_json("spark_gbt_metrics.json")

if bench is None:
    st.error("spark_benchmark.json manquant dans results/")
    st.stop()

# ── 1. KPIs timing ────────────────────────────────────────────────────────────
st.markdown("## 1. Temps d'entrainement par modele")

def _fmt_time(val):
    return f"{val:.1f} s" if isinstance(val, (int, float)) else "N/A"

c1, c2, c3 = st.columns(3)
if dnn:
    c1.metric("🧠 DNN",         _fmt_time(dnn.get('training_time_seconds')))
if xgb:
    c2.metric("🌲 XGBoost",     _fmt_time(xgb.get('training_time_seconds')))
if spark:
    c3.metric("⚡ Spark GBT",   _fmt_time(spark.get('training_time_seconds')))

# Bar chart entrainement
models_t, times_t, colors_t = [], [], []
if dnn:   models_t.append("DNN");       times_t.append(dnn["training_time_seconds"]);   colors_t.append("#3498db")
if xgb:   models_t.append("XGBoost");  times_t.append(xgb["training_time_seconds"]);   colors_t.append("#e74c3c")
if spark: models_t.append("Spark GBT");times_t.append(spark["training_time_seconds"]); colors_t.append("#2ecc71")

fig_train = go.Figure(go.Bar(
    x=models_t, y=times_t,
    marker_color=colors_t,
    text=[f"{t:.1f}s" for t in times_t], textposition="outside",
))
fig_train.update_layout(
    template="plotly_dark", height=380,
    title="Temps d'entrainement (secondes)",
    yaxis_title="Temps (s)",
)
st.plotly_chart(fig_train, use_container_width=True)

# Inference
st.markdown("### Temps d'inference (test set ~264 K lignes)")
models_i, times_i = [], []
if dnn:   models_i.append("DNN");       times_i.append(dnn.get("inference_time_seconds", 0))
if xgb:   models_i.append("XGBoost");  times_i.append(xgb.get("inference_time_seconds", 0))
if spark: models_i.append("Spark GBT");times_i.append(spark.get("inference_time_seconds", 0))

fig_inf = go.Figure(go.Bar(
    x=models_i, y=times_i,
    marker_color=colors_t[:len(models_i)],
    text=[f"{t:.3f}s" for t in times_i], textposition="outside",
))
fig_inf.update_layout(
    template="plotly_dark", height=360,
    title="Temps d'inference",
    yaxis_title="Temps (s)",
)
st.plotly_chart(fig_inf, use_container_width=True)

# ── 2. Benchmark scalabilite ──────────────────────────────────────────────────
st.markdown("## 2. Benchmark : Spark vs XGBoost Standalone")

bench_data = bench.get("results", [])
if not bench_data:
    st.warning("Pas de donnees de benchmark.")
else:
    df_b = pd.DataFrame(bench_data)
    # Calculer l'avantage XGBoost (Spark_time / Standalone_time) > 1 = XGB plus rapide
    df_b["xgb_advantage"] = df_b["spark_time"] / df_b["standalone_time"].clip(lower=1e-6)
    df_b["pct_label"] = (df_b["fraction"] * 100).astype(int).astype(str) + "%"

    c1, c2 = st.columns(2)

    with c1:
        fig_scale = go.Figure()
        fig_scale.add_trace(go.Scatter(
            x=df_b["n_rows"], y=df_b["standalone_time"],
            mode="lines+markers", name="XGBoost Standalone",
            line=dict(color="#e74c3c", width=3), marker=dict(size=9),
        ))
        fig_scale.add_trace(go.Scatter(
            x=df_b["n_rows"], y=df_b["spark_time"],
            mode="lines+markers", name="Spark GBTClassifier",
            line=dict(color="#3498db", width=3), marker=dict(size=9),
        ))
        fig_scale.update_layout(
            template="plotly_dark", height=400,
            title="Temps d'entrainement vs Volume",
            xaxis_title="Nombre de lignes (train)",
            yaxis_title="Temps (s)", yaxis_type="log",
        )
        st.plotly_chart(fig_scale, use_container_width=True)

    with c2:
        fig_adv = go.Figure()
        fig_adv.add_trace(go.Scatter(
            x=df_b["n_rows"], y=df_b["xgb_advantage"],
            mode="lines+markers", name="Avantage XGBoost",
            line=dict(color="#f39c12", width=3), marker=dict(size=9),
            fill="tozeroy", fillcolor="rgba(243,156,18,0.15)",
        ))
        fig_adv.add_hline(y=1, line_dash="dash", line_color="red",
                          annotation_text="Egalite (1x)")
        fig_adv.update_layout(
            template="plotly_dark", height=400,
            title="XGBoost est Nx plus rapide que Spark (mode local)",
            xaxis_title="Nombre de lignes (train)",
            yaxis_title="Facteur d'acceleration XGBoost / Spark",
        )
        st.plotly_chart(fig_adv, use_container_width=True)

    # Tableau detaille
    st.markdown("### 📋 Tableau de resultats du benchmark")
    disp = df_b[["pct_label", "n_rows", "spark_time", "standalone_time", "xgb_advantage"]].copy()
    disp.columns = ["Fraction", "N lignes", "Spark (s)", "XGBoost (s)", "XGBoost Nx plus rapide"]
    disp["N lignes"] = disp["N lignes"].apply(lambda x: f"{int(x):,}")
    disp["Spark (s)"] = disp["Spark (s)"].apply(lambda x: f"{x:.2f}")
    disp["XGBoost (s)"] = disp["XGBoost (s)"].apply(lambda x: f"{x:.2f}")
    disp["XGBoost Nx plus rapide"] = disp["XGBoost Nx plus rapide"].apply(lambda x: f"{x:.0f}x")
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ── 3. AUC comparaison modeles ────────────────────────────────────────────────
st.markdown("## 3. Comparaison AUC-ROC des trois modeles")

aucs, auc_names, auc_colors = [], [], []
if dnn:   aucs.append(dnn.get("metrics", {}).get("auc_roc", 0));   auc_names.append("DNN");       auc_colors.append("#3498db")
if xgb:   aucs.append(xgb.get("metrics", {}).get("auc_roc", 0));   auc_names.append("XGBoost");   auc_colors.append("#e74c3c")
if spark: aucs.append(spark.get("metrics", {}).get("auc_roc", 0)); auc_names.append("Spark GBT"); auc_colors.append("#2ecc71")

if aucs:
    fig_auc = go.Figure(go.Bar(
        x=auc_names, y=aucs,
        marker_color=auc_colors,
        text=[f"{v:.4f}" for v in aucs], textposition="outside",
    ))
    fig_auc.update_layout(
        template="plotly_dark", height=400,
        title="AUC-ROC — DNN vs XGBoost vs Spark GBT",
        yaxis=dict(range=[0.75, 0.95]),
    )
    st.plotly_chart(fig_auc, use_container_width=True)

# ── 4. Image benchmark ────────────────────────────────────────────────────────
st.markdown("## 4. Graphique de benchmark Kaggle")
img_bm = get_image_path("spark_benchmark_comparison.png")
if img_bm:
    st.image(img_bm, caption="Benchmark Spark vs Standalone — timing comparison", use_container_width=True)

img_sc = get_image_path("spark_scalability.png")
if img_sc:
    st.image(img_sc, caption="Scalabilite Spark vs volume de donnees", use_container_width=True)

# ── 5. Conclusions ────────────────────────────────────────────────────────────
st.divider()
st.markdown("## 5. Conclusions")

st.markdown("""
### 🔑 Points cles

| Aspect | Constat |
|---|---|
| **Ingestion (Spark)** | Lecture de 2M+ lignes Parquet en quelques secondes |
| **Entraînement local** | XGBoost standalone **36–97x plus rapide** que Spark GBT (mode local Kaggle) |
| **AUC-ROC** | Les 3 modeles atteignent AUC ~0.82 — plateau du aux features disponibles |
| **Scalabilite** | Sur cluster distribue (8+ noeuds), Spark devient competitif pour >10M lignes |

### ⚠️ Pourquoi Spark est plus lent en mode local ?

En mode **local** (une seule machine), Spark a un overhead important :
- Serialisation/deserialisation des partitions RDD
- Coordination du scheduler Spark (inutile sans cluster)
- GC Java vs execution native C++ de XGBoost

Le **vrai apport de Spark** dans ce projet est le **preprocessing distribue** (nettoyage, feature engineering sur 2M lignes) et la capacite a scaler sur un cluster HDFS/YARN pour des volumes plus importants.
""")
