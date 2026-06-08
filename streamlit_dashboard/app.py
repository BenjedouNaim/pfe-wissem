"""Dashboard Streamlit — Prediction des Maladies Cardiovasculaires (MCV)
Master Data Science — ISIM Monastir 2026
Apache Spark · DNN · XGBoost
"""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="MCV Prediction Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title  { font-size:2.6rem; font-weight:700; color:#ffffff; text-align:center; }
.subtitle    { font-size:1.15rem; color:#a0a0a0; text-align:center; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🫀 Prediction des Maladies Cardiovasculaires</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyse de données médicales massives — BRFSS 2020-2024<br>'
            'Apache Spark · Deep Neural Network · XGBoost</p>',
            unsafe_allow_html=True)
st.divider()

RESULTS = Path(__file__).parent / "results"

eda_stats, final_summary = None, None
try:
    with open(RESULTS / "eda_stats.json") as f:
        eda_stats = json.load(f)
except FileNotFoundError:
    pass
try:
    with open(RESULTS / "final_results_summary.json") as f:
        final_summary = json.load(f)
except FileNotFoundError:
    pass

c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Observations",   f"{eda_stats.get('total_rows', 2_176_776):,}" if eda_stats else "> 2 000 000")
c2.metric("📋 Variables",      f"{eda_stats.get('total_columns', 14)}"        if eda_stats else "14")
c4.metric("⚖️ Prevalence MCV", f"{eda_stats.get('positive_class_pct', 10.3):.1f}%" if eda_stats else "~10.3%")

if final_summary:
    best_auc = max(
        final_summary.get("dnn_metrics", {}).get("auc_roc", 0),
        final_summary.get("xgboost_metrics", {}).get("auc_roc", 0),
    )
    c3.metric("🎯 Meilleur AUC-ROC", f"{best_auc:.4f}")
else:
    c3.metric("🎯 Meilleur AUC-ROC", "0.8195")

if not (eda_stats and final_summary):
    st.info("📁 Placez les artifacts Kaggle dans **results/** et **data/** pour activer tous les KPIs.")

st.divider()

left, right = st.columns(2)
with left:
    st.markdown("""
### 🎯 Objectif
Concevoir un pipeline Big Data end-to-end pour prédire le risque de
**maladie cardiovasculaire (MCV)** à partir des données épidémiologiques BRFSS.

### 📊 Dataset
| | |
|---|---|
| **Source** | CDC BRFSS 2020-2024 |
| **Volume** | > 2 millions de répondants |
| **Variable cible** | `HeartDiseaseorAttack` (0/1) |
| **Prévalence** | ~10 % de cas positifs |
""")

with right:
    st.markdown("""
### 🛠️ Stack Technique
| Composant | Technologie |
|---|---|
| Traitement distribué | Apache Spark (PySpark) |
| Deep Learning | TensorFlow / Keras |
| Machine Learning | XGBoost + Spark MLlib |
| Dashboard | Streamlit + Plotly |

### 📄 Navigation
Utilisez le **menu latéral** pour naviguer :
1. 📊 **EDA** — Exploration interactive des données
2. 🧠 **Resultats Modeles** — DNN vs XGBoost
3. ⚡ **Impact Spark** — Benchmark distribué
""")

st.divider()
st.markdown(
    '<p style="text-align:center;color:#666;">Master Data Science — ISSAT Gafsa — 2026</p>',
    unsafe_allow_html=True,
)
