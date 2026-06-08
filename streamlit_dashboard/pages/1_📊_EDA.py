"""Page 1 — Exploration des Données (EDA)"""

import sys
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import load_sample_data, load_json, VARIABLE_DESCRIPTIONS, AGE_LABELS

st.set_page_config(page_title="EDA — MCV Dashboard", page_icon="📊", layout="wide")

st.markdown("# 📊 Exploration des Données (EDA)")
st.markdown("Analyse exploratoire du dataset BRFSS 2020–2024 — échantillon 100 K lignes")
st.divider()

# ── Chargement ────────────────────────────────────────────────────────────────
df = load_sample_data()
if df is None:
    st.stop()

TARGET = "HeartDiseaseorAttack"

# ── Sidebar — Filtres ─────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔧 Filtres")

df_f = df.copy()

if "Sex" in df_f.columns:
    sex_map = {0.0: "Homme", 1.0: "Femme"}
    available_sex = sorted(df_f["Sex"].dropna().unique())
    chosen_sex = st.sidebar.multiselect(
        "Sexe", options=available_sex,
        format_func=lambda x: sex_map.get(x, str(x)),
        default=available_sex,
    )
    df_f = df_f[df_f["Sex"].isin(chosen_sex)]

if "Age" in df_f.columns:
    a_min, a_max = int(df_f["Age"].min()), int(df_f["Age"].max())
    age_range = st.sidebar.slider("Tranche d'âge", a_min, a_max, (a_min, a_max))
    df_f = df_f[(df_f["Age"] >= age_range[0]) & (df_f["Age"] <= age_range[1])]

st.sidebar.markdown(f"**{len(df_f):,}** enregistrements sélectionnés")

# ── 1. Distribution cible ─────────────────────────────────────────────────────
st.markdown("## 1. Distribution de la variable cible")

c1, c2 = st.columns(2)
counts = df_f[TARGET].value_counts().sort_index()
labels = ["Pas de MCV (0)", "MCV (1)"]
colors = ["#2ecc71", "#e74c3c"]

with c1:
    fig_pie = px.pie(
        values=counts.values, names=labels,
        color_discrete_sequence=colors,
        title="Répartition des classes",
        hole=0.35,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    fig_bar = go.Figure(go.Bar(
        x=labels, y=counts.values,
        marker_color=colors,
        text=[f"{v:,}" for v in counts.values],
        textposition="outside",
    ))
    fig_bar.update_layout(
        template="plotly_dark", height=380,
        title="Nombre de cas par classe",
        yaxis_title="Nombre",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── 2. Distribution univariée interactive ─────────────────────────────────────
st.markdown("## 2. Distribution univariée par feature")

feature_options = [c for c in df_f.columns if c != TARGET]
selected = st.selectbox(
    "Choisir une variable",
    feature_options,
    format_func=lambda x: f"{x}  —  {VARIABLE_DESCRIPTIONS.get(x, '')}",
)

c1, c2 = st.columns(2)
with c1:
    fig_hist = px.histogram(
        df_f, x=selected, color=TARGET,
        barmode="overlay",
        color_discrete_sequence=colors,
        title=f"Distribution de {selected} par classe",
        labels={TARGET: "Classe", selected: selected},
        opacity=0.75,
    )
    fig_hist.update_layout(template="plotly_dark", height=380)
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    fig_box = px.box(
        df_f, x=TARGET, y=selected,
        color=TARGET,
        color_discrete_sequence=colors,
        title=f"Boxplot de {selected} par classe",
        labels={TARGET: "Classe"},
    )
    fig_box.update_layout(template="plotly_dark", height=380, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# ── 3. Heatmap de corrélation ─────────────────────────────────────────────────
st.markdown("## 3. Matrice de corrélation (Pearson)")

num_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
corr = df_f[num_cols].corr()

fig_corr = px.imshow(
    corr, text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Heatmap de corrélation — Variables BRFSS",
    aspect="auto",
)
fig_corr.update_layout(template="plotly_dark", height=600)
st.plotly_chart(fig_corr, use_container_width=True)

# ── 4. Outliers — variables continues ────────────────────────────────────────
st.markdown("## 4. Détection des outliers")

continuous = [c for c in ["BMI", "MentHlth", "PhysHlth"] if c in df_f.columns]
if continuous:
    fig_out = make_subplots(rows=1, cols=len(continuous), subplot_titles=continuous)
    for i, col in enumerate(continuous, 1):
        for cls, color, name in [(0, "#2ecc71", "Pas MCV"), (1, "#e74c3c", "MCV")]:
            sub = df_f[df_f[TARGET] == cls][col].dropna()
            fig_out.add_trace(
                go.Box(y=sub, name=name, marker_color=color, showlegend=(i == 1)),
                row=1, col=i,
            )
    fig_out.update_layout(
        template="plotly_dark", height=420,
        title="Boxplots variables continues par classe MCV",
    )
    st.plotly_chart(fig_out, use_container_width=True)

# ── 5. Taux MCV par sous-groupe ───────────────────────────────────────────────
st.markdown("## 5. Taux de MCV par sous-groupe")

group_options = [c for c in ["Age", "Sex", "GenHlth", "Education", "Diabetes", "Smoker"]
                 if c in df_f.columns]
group_var = st.selectbox("Grouper par", group_options)

group_df = (
    df_f.groupby(group_var)[TARGET]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "TauxMCV", "count": "N"})
)
group_df["TauxMCV_pct"] = group_df["TauxMCV"] * 100

if group_var == "Age":
    group_df["label"] = group_df[group_var].map(AGE_LABELS).fillna(group_df[group_var].astype(str))
else:
    group_df["label"] = group_df[group_var].astype(str)

fig_group = px.bar(
    group_df, x="label", y="TauxMCV_pct",
    text="TauxMCV_pct",
    color="TauxMCV_pct",
    color_continuous_scale="Reds",
    title=f"Taux de MCV par {group_var}",
    labels={"label": group_var, "TauxMCV_pct": "Taux MCV (%)"},
    custom_data=["N"],
)
fig_group.update_traces(
    texttemplate="%{text:.1f}%", textposition="outside",
    hovertemplate="%{x}<br>Taux MCV: %{y:.2f}%<br>N=%{customdata[0]:,}<extra></extra>",
)
fig_group.update_layout(template="plotly_dark", height=420, coloraxis_showscale=False)
st.plotly_chart(fig_group, use_container_width=True)

# ── Stats résumé ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📋 Statistiques descriptives")
st.dataframe(df_f.describe().round(3), use_container_width=True)
