"""Page 2 — Resultats des Modeles : DNN vs XGBoost"""

import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import load_json, load_csv, get_image_path, fmt

st.set_page_config(page_title="Resultats Modeles — MCV", page_icon="🧠", layout="wide")

st.markdown("# 🧠 Resultats des Modeles")
st.markdown("Comparaison DNN vs XGBoost — métriques complètes")
st.divider()

# ── Chargement ────────────────────────────────────────────────────────────────
dnn     = load_json("dnn_metrics.json")
xgb     = load_json("xgboost_metrics.json")
cmp_df  = load_csv("metrics_comparison.csv")
summary = load_json("final_results_summary.json")

if dnn is None or xgb is None:
    st.error("Placez dnn_metrics.json et xgboost_metrics.json dans results/")
    st.stop()

dnn_m = dnn.get("metrics", {})
xgb_m = xgb.get("metrics", {})

# ── Selector seuil DNN ────────────────────────────────────────────────────────
threshold_choice = st.sidebar.radio(
    "Métriques DNN — seuil",
    ["Optimal (F1-max)", "Défaut (0.50)"],
    index=0,
)
if threshold_choice == "Optimal (F1-max)" and "metrics_optimal_threshold" in dnn:
    dnn_m = dnn["metrics_optimal_threshold"]
    opt_thresh = dnn.get("optimal_threshold", "?")
    st.sidebar.info(f"Seuil optimal DNN : **{opt_thresh}**")
elif threshold_choice == "Défaut (0.50)" and "metrics_threshold_05" in dnn:
    dnn_m = dnn["metrics_threshold_05"]

# ── 1. KPIs ───────────────────────────────────────────────────────────────────
st.markdown("## 1. Metriques cles")

kpi_keys  = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
kpi_names = ["Accuracy",  "Precision", "Recall",  "F1-Score",  "AUC-ROC"]
cols = st.columns(5)

for col, name, key in zip(cols, kpi_names, kpi_keys):
    dv = dnn_m.get(key)
    xv = xgb_m.get(key)
    if dv is not None and xv is not None:
        best_val = max(dv, xv)
        delta = dv - xv
        col.metric(
            f"🏆 {name}",
            f"{best_val:.4f}",
            delta=f"DNN {'>' if delta > 0 else '<'} XGB ({abs(delta):.4f})",
            delta_color="normal" if delta > 0 else "inverse",
        )

# ── 2. Tableau comparatif ─────────────────────────────────────────────────────
st.markdown("## 2. Tableau comparatif")

if cmp_df is not None:
    num_cols = cmp_df.select_dtypes("number").columns.tolist()
    st.dataframe(
        cmp_df.style.format(
            {c: "{:.4f}" for c in num_cols}
        ).highlight_max(subset=num_cols, axis=1, color="#0f3460"),
        use_container_width=True, hide_index=True,
    )
else:
    # Construction manuelle si CSV manquant
    rows = []
    for name, key in zip(kpi_names, kpi_keys):
        rows.append({
            "Metric": name,
            "DNN": fmt(dnn_m.get(key)),
            "XGBoost": fmt(xgb_m.get(key)),
        })
    rows.append({"Metric": "Training time (s)",
                 "DNN": dnn.get("training_time_seconds", "—"),
                 "XGBoost": xgb.get("training_time_seconds", "—")})
    rows.append({"Metric": "Inference time (s)",
                 "DNN": dnn.get("inference_time_seconds", "—"),
                 "XGBoost": xgb.get("inference_time_seconds", "—")})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── 3. Bar chart métriques ────────────────────────────────────────────────────
st.markdown("## 3. Comparaison visuelle des metriques")

dnn_vals = [dnn_m.get(k, 0) for k in kpi_keys]
xgb_vals = [xgb_m.get(k, 0) for k in kpi_keys]

x = np.arange(len(kpi_names))
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    name="DNN", x=kpi_names, y=dnn_vals,
    marker_color="#3498db",
    text=[f"{v:.3f}" for v in dnn_vals], textposition="outside",
))
fig_bar.add_trace(go.Bar(
    name="XGBoost", x=kpi_names, y=xgb_vals,
    marker_color="#e74c3c",
    text=[f"{v:.3f}" for v in xgb_vals], textposition="outside",
))
fig_bar.update_layout(
    template="plotly_dark", barmode="group", height=420,
    title="DNN vs XGBoost — Performance par metrique",
    yaxis=dict(range=[0, 1.12]),
    legend=dict(x=0.01, y=0.99),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── 4. Courbes ROC ────────────────────────────────────────────────────────────
st.markdown("## 4. Courbes ROC superposees")

fig_roc = go.Figure()
if "roc_curve" in dnn:
    fig_roc.add_trace(go.Scatter(
        x=dnn["roc_curve"]["fpr"], y=dnn["roc_curve"]["tpr"],
        mode="lines", name=f"DNN (AUC={dnn_m.get('auc_roc', 0):.4f})",
        line=dict(color="#3498db", width=3),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
    ))
if "roc_curve" in xgb:
    fig_roc.add_trace(go.Scatter(
        x=xgb["roc_curve"]["fpr"], y=xgb["roc_curve"]["tpr"],
        mode="lines", name=f"XGBoost (AUC={xgb_m.get('auc_roc', 0):.4f})",
        line=dict(color="#e74c3c", width=3),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
    ))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines", name="Aleatoire",
    line=dict(color="gray", width=1, dash="dash"),
))
fig_roc.update_layout(
    template="plotly_dark", height=550,
    title="Courbes ROC — DNN vs XGBoost",
    xaxis_title="Taux de Faux Positifs (FPR)",
    yaxis_title="Taux de Vrais Positifs (TPR)",
    legend=dict(x=0.55, y=0.1, bgcolor="rgba(0,0,0,0.5)"),
)
st.plotly_chart(fig_roc, use_container_width=True)

# ── 5. Matrices de confusion ──────────────────────────────────────────────────
st.markdown("## 5. Matrices de confusion")

c1, c2 = st.columns(2)
cls_labels = ["No CVD (0)", "CVD (1)"]

with c1:
    cm_d = np.array(dnn.get("confusion_matrix", [[0, 0], [0, 0]]))
    fig_cm = px.imshow(
        cm_d, text_auto=True,
        x=cls_labels, y=cls_labels,
        color_continuous_scale="Blues",
        title=f"Matrice de confusion — DNN (seuil={dnn.get('optimal_threshold', 0.5):.2f})",
        labels=dict(x="Prediction", y="Realite"),
    )
    fig_cm.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig_cm, use_container_width=True)

with c2:
    cm_x = np.array(xgb.get("confusion_matrix", [[0, 0], [0, 0]]))
    fig_cmx = px.imshow(
        cm_x, text_auto=True,
        x=cls_labels, y=cls_labels,
        color_continuous_scale="Oranges",
        title="Matrice de confusion — XGBoost (seuil=0.5)",
        labels=dict(x="Prediction", y="Realite"),
    )
    fig_cmx.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig_cmx, use_container_width=True)

# ── 6. Radar chart ────────────────────────────────────────────────────────────
st.markdown("## 6. Vue d'ensemble — Radar Chart")

fig_rad = go.Figure()
cats = kpi_names + [kpi_names[0]]
fig_rad.add_trace(go.Scatterpolar(
    r=dnn_vals + [dnn_vals[0]], theta=cats,
    fill="toself", name="DNN",
    line_color="#3498db", fillcolor="rgba(52,152,219,0.2)",
))
fig_rad.add_trace(go.Scatterpolar(
    r=xgb_vals + [xgb_vals[0]], theta=cats,
    fill="toself", name="XGBoost",
    line_color="#e74c3c", fillcolor="rgba(231,76,60,0.2)",
))
fig_rad.update_layout(
    template="plotly_dark", height=500,
    title="Comparaison multidimensionnelle — DNN vs XGBoost",
    polar=dict(radialaxis=dict(range=[0, 1])),
)
st.plotly_chart(fig_rad, use_container_width=True)

# ── 7. Feature Importance XGBoost ─────────────────────────────────────────────
st.markdown("## 7. Feature Importance (XGBoost)")
img = get_image_path("xgboost_feature_importance.png")
if img:
    st.image(img, caption="Importance des variables — XGBoost (weight / gain / cover)",
             use_container_width=True)
else:
    st.info("Image xgboost_feature_importance.png manquante dans results/")

# ── 8. Courbes d'apprentissage DNN ───────────────────────────────────────────
st.markdown("## 8. Courbes d'apprentissage DNN")
img_lc = get_image_path("dnn_learning_curves.png")
img_th = get_image_path("dnn_threshold_analysis.png")

c1, c2 = st.columns(2)
with c1:
    if img_lc:
        st.image(img_lc, caption="Learning curves DNN", use_container_width=True)
with c2:
    if img_th:
        st.image(img_th, caption="Analyse du seuil de classification", use_container_width=True)

# ── 9. Critères PRD ───────────────────────────────────────────────────────────
st.divider()
st.markdown("## ✅ Critères de succes (PRD)")

criteria = [
    ("AUC-ROC DNN ≥ 0.85",        dnn_m.get("auc_roc", 0) >= 0.85,       dnn_m.get("auc_roc", 0)),
    ("AUC-ROC XGBoost ≥ 0.87",    xgb_m.get("auc_roc", 0) >= 0.87,       xgb_m.get("auc_roc", 0)),
    ("F1-score classe positive ≥ 0.60",
     max(dnn_m.get("f1_score", 0), xgb_m.get("f1_score", 0)) >= 0.60,
     max(dnn_m.get("f1_score", 0), xgb_m.get("f1_score", 0))),
    ("Recall classe positive ≥ 0.65",
     max(dnn_m.get("recall", 0), xgb_m.get("recall", 0)) >= 0.65,
     max(dnn_m.get("recall", 0), xgb_m.get("recall", 0))),
]

for label, passed, value in criteria:
    if passed:
        st.success(f"✅ {label}  →  {value:.4f}")
    else:
        st.error(f"❌ {label}  →  {value:.4f}")

st.caption(
    "Note : AUC plateau à 0.82 du aux 4 features manquantes (HighBP, HighChol, Income, NoDocbcCost) "
    "dans le fichier _ml pooled BRFSS 2020-2024."
)
