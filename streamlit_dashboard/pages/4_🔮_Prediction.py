"""Page 4 — Tester les Modeles : prediction en direct du risque de MCV.

L'utilisateur saisit un profil patient ; le DNN et XGBoost estiment en direct
la probabilite de maladie cardiovasculaire. Le preprocessing (features derivees
+ StandardScaler) est identique a celui de l'entrainement.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).parent.parent))
from utils.inference import (  # noqa: E402
    RAW_FEATURES, DNN_THRESHOLD, XGB_THRESHOLD,
    compute_derived, build_vector, scale,
    predict_xgb, predict_dnn, models_status,
)
from utils.helpers import AGE_LABELS, load_json  # noqa: E402

st.set_page_config(page_title="Prediction — MCV", page_icon="🔮", layout="wide")

st.markdown("# 🔮 Tester les Modeles — Prediction en direct")
st.markdown(
    "Renseignez un **profil patient** : les modeles **DNN** et **XGBoost** estiment "
    "instantanement la probabilite de **maladie cardiovasculaire (MCV)**."
)
st.divider()

# ── Libelles des champs categoriels ───────────────────────────────────────────
SEX_OPTS = {0: "0 — Homme", 1: "1 — Femme"}
YESNO = {0: "0 — Non", 1: "1 — Oui"}
GENHLTH = {1: "1 — Excellente", 2: "2 — Tres bonne", 3: "3 — Bonne",
           4: "4 — Moyenne", 5: "5 — Mauvaise"}
EDU = {1: "1 — Jamais scolarise", 2: "2 — Primaire", 3: "3 — College",
       4: "4 — Lycee", 5: "5 — Universitaire (1-3 ans)", 6: "6 — Diplome (4 ans+)"}
DIAB = {0: "0 — Non", 1: "1 — Pre-diabete / grossesse", 2: "2 — Oui"}

# ── Valeurs par defaut + profils d'exemple ────────────────────────────────────
DEFAULTS = {
    "Age": 8, "Sex": 1, "Education": 5, "GenHlth": 3, "BMI": 28.0,
    "PhysHlth": 3, "MentHlth": 3, "DiffWalk": 0, "Stroke": 0, "Diabetes": 0,
    "Smoker": 0, "PhysActivity": 1, "HvyAlcoholConsump": 0,
}
PRESET_LOW = {
    "Age": 3, "Sex": 0, "Education": 6, "GenHlth": 1, "BMI": 22.0,
    "PhysHlth": 0, "MentHlth": 0, "DiffWalk": 0, "Stroke": 0, "Diabetes": 0,
    "Smoker": 0, "PhysActivity": 1, "HvyAlcoholConsump": 0,
}
PRESET_HIGH = {
    "Age": 12, "Sex": 0, "Education": 2, "GenHlth": 5, "BMI": 38.0,
    "PhysHlth": 25, "MentHlth": 20, "DiffWalk": 1, "Stroke": 1, "Diabetes": 2,
    "Smoker": 1, "PhysActivity": 0, "HvyAlcoholConsump": 1,
}

# Initialisation unique du session_state (les widgets lisent ces cles).
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def _apply(preset):
    for k, v in preset.items():
        st.session_state[k] = v


# ── Etat des modeles ──────────────────────────────────────────────────────────
status = models_status()
badges = []
badges.append("✅ Scaler" if status["scaler"] else "❌ Scaler")
badges.append("✅ XGBoost" if status["xgboost"] else "❌ XGBoost")
badges.append("✅ DNN" if status["dnn"] else "❌ DNN")
st.caption("Modeles charges : " + "  ·  ".join(badges))
if not status["scaler"]:
    st.warning("`scaler.pkl` introuvable dans **models/** — les predictions seront faussees.")

# ── Profils d'exemple ─────────────────────────────────────────────────────────
b1, b2, b3, _ = st.columns([1, 1, 1, 3])
b1.button("🟢 Profil faible risque", use_container_width=True,
          on_click=_apply, args=(PRESET_LOW,))
b2.button("🔴 Profil risque eleve", use_container_width=True,
          on_click=_apply, args=(PRESET_HIGH,))
b3.button("↺ Reinitialiser", use_container_width=True,
          on_click=_apply, args=(DEFAULTS,))

st.markdown("### 🩺 Profil du patient")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographie**")
    st.selectbox("Tranche d'age", list(range(1, 14)), key="Age",
                 format_func=lambda v: f"{v} — {AGE_LABELS[v]} ans")
    st.radio("Sexe", [0, 1], key="Sex", format_func=SEX_OPTS.get, horizontal=True)
    st.selectbox("Niveau d'education", list(range(1, 7)), key="Education",
                 format_func=EDU.get)

with col2:
    st.markdown("**Etat de sante**")
    st.selectbox("Sante generale", [1, 2, 3, 4, 5], key="GenHlth",
                 format_func=GENHLTH.get)
    st.slider("IMC (BMI)", 12.0, 60.0, key="BMI", step=0.5,
              help="Indice de masse corporelle (kg/m²)")
    st.slider("Jours mauvaise sante physique (30j)", 0, 30, key="PhysHlth")
    st.slider("Jours mauvaise sante mentale (30j)", 0, 30, key="MentHlth")
    st.radio("Difficulte a marcher", [0, 1], key="DiffWalk",
             format_func=YESNO.get, horizontal=True)

with col3:
    st.markdown("**Antecedents & comportements**")
    st.radio("Antecedent d'AVC", [0, 1], key="Stroke",
             format_func=YESNO.get, horizontal=True)
    st.selectbox("Statut diabetique", [0, 1, 2], key="Diabetes",
                 format_func=DIAB.get)
    st.radio("Fumeur", [0, 1], key="Smoker", format_func=YESNO.get, horizontal=True)
    st.radio("Activite physique (30j)", [0, 1], key="PhysActivity",
             format_func=YESNO.get, horizontal=True)
    st.radio("Consommation excessive d'alcool", [0, 1], key="HvyAlcoholConsump",
             format_func=YESNO.get, horizontal=True)

# ── Features derivees (calculees automatiquement) ─────────────────────────────
values = {f: st.session_state[f] for f in RAW_FEATURES}
derived = compute_derived(values)

st.markdown("### 🧮 Features derivees (calculees automatiquement)")
d1, d2 = st.columns(2)
d1.metric("RiskScore", f"{derived['RiskScore']:.0f}",
          help="Somme : Smoker + Diabetes + Stroke + DiffWalk + HvyAlcoholConsump")
d2.metric("HealthIndex", f"{derived['HealthIndex']:.2f}",
          help="GenHlth + MentHlth/6 + PhysHlth/6")

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
st.markdown("## 🎯 Estimation du risque")

scaled = scale(build_vector(values))
try:
    p_xgb, backend_xgb = predict_xgb(scaled)
    p_dnn, backend_dnn = predict_dnn(scaled)
except Exception as exc:  # pragma: no cover - garde-fou UI
    st.error(f"Erreur lors de l'inference : {exc}")
    st.stop()


def _gauge(prob, threshold, title, color):
    pct = prob * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": " %", "font": {"size": 40}},
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, threshold * 100], "color": "rgba(46,204,113,0.20)"},
                {"range": [threshold * 100, 100], "color": "rgba(231,76,60,0.20)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3},
                          "thickness": 0.85, "value": threshold * 100},
        },
    ))
    fig.update_layout(template="plotly_dark", height=300,
                      margin=dict(t=60, b=10, l=20, r=20))
    return fig


def _verdict(prob, threshold):
    if prob >= threshold:
        return "⚠️ Risque eleve de MCV", "#e74c3c"
    return "✅ Faible risque de MCV", "#2ecc71"


cg1, cg2 = st.columns(2)

with cg1:
    if p_dnn is None:
        st.info("Modele **DNN** indisponible. Installez `h5py` et placez "
                "`dnn_best_model.h5` dans **models/**.")
    else:
        st.plotly_chart(_gauge(p_dnn, DNN_THRESHOLD, "🧠 DNN", "#3498db"),
                        use_container_width=True)
        label, color = _verdict(p_dnn, DNN_THRESHOLD)
        st.markdown(
            f"<div style='text-align:center;font-size:1.25rem;font-weight:700;"
            f"color:{color}'>{label}</div>", unsafe_allow_html=True)
        st.caption(f"Seuil de decision : {DNN_THRESHOLD:.2f} · backend : {backend_dnn}")

with cg2:
    if p_xgb is None:
        st.info("Modele **XGBoost** indisponible. Placez `xgboost_model.json` "
                "dans **models/**.")
    else:
        st.plotly_chart(_gauge(p_xgb, XGB_THRESHOLD, "🌲 XGBoost", "#e74c3c"),
                        use_container_width=True)
        label, color = _verdict(p_xgb, XGB_THRESHOLD)
        st.markdown(
            f"<div style='text-align:center;font-size:1.25rem;font-weight:700;"
            f"color:{color}'>{label}</div>", unsafe_allow_html=True)
        st.caption(f"Seuil de decision : {XGB_THRESHOLD:.2f} · backend : {backend_xgb}")

# ── Synthese / accord des modeles ─────────────────────────────────────────────
if p_dnn is not None and p_xgb is not None:
    dnn_pos = p_dnn >= DNN_THRESHOLD
    xgb_pos = p_xgb >= XGB_THRESHOLD
    avg = (p_dnn + p_xgb) / 2
    st.divider()
    s1, s2, s3 = st.columns(3)
    s1.metric("🧠 DNN", f"{p_dnn * 100:.1f} %")
    s2.metric("🌲 XGBoost", f"{p_xgb * 100:.1f} %")
    s3.metric("📊 Probabilite moyenne", f"{avg * 100:.1f} %")
    if dnn_pos == xgb_pos:
        verdict = "⚠️ risque eleve" if dnn_pos else "✅ faible risque"
        st.success(f"Les deux modeles s'accordent : **{verdict}**.")
    else:
        st.warning("Les modeles sont en **desaccord** — profil a la frontiere de decision.")

# ── Contexte : performance & details techniques ───────────────────────────────
with st.expander("ℹ️ Details techniques & performance des modeles"):
    dnn_meta = load_json("dnn_metrics.json")
    xgb_meta = load_json("xgboost_metrics.json")
    rows = []
    if dnn_meta:
        m = dnn_meta.get("metrics", {})
        rows.append({"Modele": "DNN", "AUC-ROC (test)": f"{m.get('auc_roc', 0):.4f}",
                     "Recall": f"{m.get('recall', 0):.4f}", "Seuil": f"{DNN_THRESHOLD:.2f}"})
    if xgb_meta:
        m = xgb_meta.get("metrics", {})
        rows.append({"Modele": "XGBoost", "AUC-ROC (test)": f"{m.get('auc_roc', 0):.4f}",
                     "Recall": f"{m.get('recall', 0):.4f}", "Seuil": f"{XGB_THRESHOLD:.2f}"})
    if rows:
        import pandas as pd
        st.table(pd.DataFrame(rows))
    st.markdown(
        "- **Pipeline** : 13 saisies → RiskScore & HealthIndex → `StandardScaler` "
        "→ vecteur de 15 features → modeles.\n"
        "- **DNN** : forward pass NumPy reproduit a l'identique le modele Keras "
        "(MLP 256→128→64→1, BatchNorm, sigmoid).\n"
        "- **XGBoost** : parcours des 400 arbres ; utilise la librairie `xgboost` "
        "si disponible, sinon un parcours NumPy fidele (float32).\n"
        "- Le **recall** eleve / la **precision** faible reflètent le desequilibre "
        "de classes (~10 % de cas positifs) et le choix d'un seuil oriente depistage."
    )
    st.caption("Vecteur scale envoye aux modeles (15 features) :")
    st.code(np.round(scaled, 3).tolist(), language="python")

# ── Avertissement ─────────────────────────────────────────────────────────────
st.divider()
st.warning(
    "⚠️ **Outil pedagogique** — modeles entraines sur des donnees epidemiologiques "
    "BRFSS a des fins de demonstration academique. **Ne constitue en aucun cas un "
    "diagnostic ou un avis medical.**"
)
st.markdown(
    '<p style="text-align:center;color:#666;">Master Data Science — ISSAT Gafsa — 2026</p>',
    unsafe_allow_html=True,
)
