"""Inference en direct pour la page de prediction.

Charge les modeles entraines (DNN, XGBoost) et reproduit *exactement* le
pipeline de preprocessing utilise a l'entrainement :

    inputs bruts -> features derivees (RiskScore, HealthIndex) -> StandardScaler

Les deux predicteurs ont ete valides face aux artifacts d'origine :
  - XGBoost : reproduit la librairie xgboost a ~5e-7 pres (parcours d'arbres
    en float32, comme le moteur natif). Utilise la librairie si disponible,
    sinon un parcours d'arbres en pur NumPy (zero dependance lourde).
  - DNN : reproduit les metriques stockees a l'identique (forward pass NumPy
    avec BatchNormalization, eps=1e-3 — defaut Keras).

Tout est tolerant aux artifacts manquants : chaque loader renvoie None et la
page affiche un message clair plutot que de planter.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st

# ── Localisation des artifacts ────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_DASH = _HERE.parent
# On cherche d'abord dans le dashboard (deploiement autonome), puis a la racine
# du repo (models/ a cote de streamlit_dashboard/).
_MODEL_DIRS = [_DASH / "models", _DASH.parent / "models"]


def _find(filename):
    for d in _MODEL_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


# ── Schema des features ───────────────────────────────────────────────────────
# Ordre EXACT des colonnes attendu par le scaler et les modeles.
FINAL_FEATURES = [
    "Age", "GenHlth", "Stroke", "DiffWalk", "Diabetes", "PhysHlth", "Smoker",
    "Sex", "PhysActivity", "HvyAlcoholConsump", "BMI", "Education", "MentHlth",
    "RiskScore", "HealthIndex",
]
DERIVED = ["RiskScore", "HealthIndex"]
RAW_FEATURES = [f for f in FINAL_FEATURES if f not in DERIVED]  # 13 saisies
RISK_COMPONENTS = ["Smoker", "Diabetes", "Stroke", "DiffWalk", "HvyAlcoholConsump"]

# Seuils de classification (cf. notebooks / metrics JSON)
DNN_THRESHOLD = 0.6713  # seuil optimal F1 du DNN
XGB_THRESHOLD = 0.50    # XGBoost evalue au seuil par defaut


def compute_derived(values: dict) -> dict:
    """RiskScore = somme des facteurs binaires ; HealthIndex = GenHlth + (MentHlth+PhysHlth)/6."""
    risk = float(sum(values[c] for c in RISK_COMPONENTS))
    health = float(values["GenHlth"] + values["MentHlth"] / 6.0 + values["PhysHlth"] / 6.0)
    return {"RiskScore": risk, "HealthIndex": health}


def build_vector(values: dict) -> np.ndarray:
    """Construit le vecteur (1, 15) dans l'ordre FINAL_FEATURES (features derivees incluses)."""
    full = dict(values)
    full.update(compute_derived(values))
    return np.array([[full[f] for f in FINAL_FEATURES]], dtype=np.float64)


# ── Scaler ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_scaler():
    p = _find("scaler.pkl")
    if p is None:
        return None
    import joblib
    return joblib.load(p)


def scale(vec: np.ndarray) -> np.ndarray:
    sc = load_scaler()
    return vec if sc is None else sc.transform(vec)


# ── XGBoost ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _xgb_library():
    """Modele charge via la librairie xgboost (chemin le plus fidele), ou None."""
    try:
        import xgboost as xgb
    except Exception:
        return None
    p = _find("xgboost_model.json")
    if p is None:
        return None
    try:
        clf = xgb.XGBClassifier()
        clf.load_model(str(p))
        return clf
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _xgb_trees():
    """Arbres pre-parses pour le parcours en pur NumPy (fallback sans dependance)."""
    p = _find("xgboost_model.json")
    if p is None:
        return None
    model = json.loads(p.read_text())
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    pre = []
    for t in trees:
        pre.append((
            np.asarray(t["left_children"], dtype=np.int32),
            np.asarray(t["right_children"], dtype=np.int32),
            np.asarray(t["split_indices"], dtype=np.int32),
            # Le moteur natif compare en float32 -> on reproduit a l'identique.
            np.asarray(t["split_conditions"], dtype=np.float32),
        ))
    return pre


def _xgb_margin(pre, Xf):
    total = np.zeros(len(Xf), dtype=np.float64)
    for lc, rc, si, sc in pre:
        node = np.zeros(len(Xf), dtype=np.int64)
        active = lc[node] != -1
        while active.any():
            n = node[active]
            go_left = Xf[active, si[n]] < sc[n]
            node[active] = np.where(go_left, lc[n], rc[n])
            active = lc[node] != -1
        total += sc[node].astype(np.float64)
    return total


def predict_xgb(scaled_vec: np.ndarray):
    """Renvoie (proba_float, backend_str) ou (None, None) si le modele est absent."""
    clf = _xgb_library()
    if clf is not None:
        proba = float(clf.predict_proba(scaled_vec)[:, 1][0])
        return proba, "librairie xgboost"
    pre = _xgb_trees()
    if pre is None:
        return None, None
    margin = _xgb_margin(pre, scaled_vec.astype(np.float32))
    proba = float(1.0 / (1.0 + np.exp(-margin))[0])
    return proba, "NumPy (parcours d'arbres)"


# ── DNN (forward pass NumPy a partir des poids .h5) ───────────────────────────
@st.cache_resource(show_spinner=False)
def _dnn_weights():
    """Charge les poids du MLP depuis dnn_best_model.h5 via h5py, ou None."""
    p = _find("dnn_best_model.h5")
    if p is None:
        return None
    try:
        import h5py
    except Exception:
        return None

    with h5py.File(p, "r") as f:
        if "model_weights" not in f:
            return None
        wg = f["model_weights"]
        paths = []
        wg.visititems(lambda n, o: paths.append(n) if isinstance(o, h5py.Dataset) else None)

        def grab(layer, suffix):
            for pth in paths:
                if pth.startswith(layer + "/") and pth.endswith("/" + suffix):
                    return wg[pth][()]
            return None

        def layers(prefix):
            names = {p.split("/", 1)[0] for p in paths if p.startswith(prefix)}
            return sorted(names, key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0))

        dense = layers("dense")
        bn = layers("batch_normalization")
        if len(dense) < 4:
            return None

        W = [grab(d, "kernel") for d in dense]
        b = [grab(d, "bias") for d in dense]
        BN = [(grab(n, "gamma"), grab(n, "beta"),
               grab(n, "moving_mean"), grab(n, "moving_variance")) for n in bn]
        if any(w is None for w in W) or any(x is None for x in b):
            return None
    return {"W": W, "b": b, "BN": BN}


def _bn(a, params, eps=1e-3):
    gamma, beta, mean, var = params
    return gamma * (a - mean) / np.sqrt(var + eps) + beta


def predict_dnn(scaled_vec: np.ndarray):
    """Renvoie (proba_float, backend_str) ou (None, None) si le modele est absent.

    Architecture : Dense(256,relu)+BN -> Dense(128,relu)+BN -> Dense(64,relu) -> Dense(1,sigmoid)
    """
    w = _dnn_weights()
    if w is None:
        return None, None
    W, b, BN = w["W"], w["b"], w["BN"]
    relu = lambda x: np.maximum(0.0, x)

    a = relu(scaled_vec @ W[0] + b[0])
    if len(BN) >= 1:
        a = _bn(a, BN[0])
    a = relu(a @ W[1] + b[1])
    if len(BN) >= 2:
        a = _bn(a, BN[1])
    a = relu(a @ W[2] + b[2])
    z = a @ W[3] + b[3]
    proba = float(1.0 / (1.0 + np.exp(-z)).ravel()[0])
    return proba, "NumPy (poids Keras)"


# ── Etat des modeles (pour affichage) ─────────────────────────────────────────
def models_status() -> dict:
    return {
        "scaler": load_scaler() is not None,
        "xgboost": (_xgb_library() is not None) or (_xgb_trees() is not None),
        "dnn": _dnn_weights() is not None,
    }
