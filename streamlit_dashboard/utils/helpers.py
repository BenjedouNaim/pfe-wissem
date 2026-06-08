"""Fonctions utilitaires partagees entre les pages du dashboard."""

import os
import json
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


@st.cache_data
def load_sample_data():
    path = DATA_DIR / "brfss_sample.csv"
    if path.exists():
        return pd.read_csv(path)
    st.warning("brfss_sample.csv non trouve dans data/. Placez-le depuis Kaggle.")
    return None


@st.cache_data
def load_json(filename):
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path, "r") as f:
            content = f.read()
        # Python's json.dumps allows NaN literals which are not valid RFC 8259 JSON.
        # Replace bare NaN with null so strict parsers don't fail.
        import re
        content = re.sub(r'\bNaN\b', 'null', content)
        return json.loads(content)
    st.warning(f"Fichier manquant : {filename}")
    return None


@st.cache_data
def load_csv(filename):
    path = RESULTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    st.warning(f"Fichier manquant : {filename}")
    return None


def get_image_path(filename):
    path = RESULTS_DIR / filename
    return str(path) if path.exists() else None


def fmt(value, pct=False):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%" if pct else f"{value:.4f}"


VARIABLE_DESCRIPTIONS = {
    "HeartDiseaseorAttack": "Maladie cardiovasculaire diagnostiquee (cible)",
    "HighBP": "Hypertension arterielle",
    "HighChol": "Hypercholesterolemie",
    "BMI": "Indice de masse corporelle",
    "Smoker": "Fumeur (0=non, 1=oui)",
    "Stroke": "Antecedents d'AVC (0=non, 1=oui)",
    "Diabetes": "Statut diabetique",
    "PhysActivity": "Activite physique dernier mois (0=non, 1=oui)",
    "GenHlth": "Sante generale (1=Excellente ... 5=Mauvaise)",
    "MentHlth": "Jours mauvaise sante mentale (0-30)",
    "PhysHlth": "Jours mauvaise sante physique (0-30)",
    "DiffWalk": "Difficulte a marcher (0=non, 1=oui)",
    "Sex": "Sexe (0=Homme, 1=Femme)",
    "Age": "Tranche d'age (1=18-24 ... 13=80+)",
    "Education": "Niveau d'education (1-6)",
    "Income": "Niveau de revenu (1-8)",
    "HvyAlcoholConsump": "Consommation excessive alcool (0=non, 1=oui)",
    "NoDocbcCost": "Renoncement aux soins pour cout (0=non, 1=oui)",
    "RiskScore": "Score de risque composite (somme de facteurs binaires)",
    "HealthIndex": "Indice de sante global (GenHlth + MentHlth/6 + PhysHlth/6)",
}

AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+",
}
