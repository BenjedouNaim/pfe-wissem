# PRD — Analyse de données médicales massives pour la prédiction des maladies cardiovasculaires

> **Plateforme Big Data · Apache Spark · DNN & XGBoost · Dashboard Streamlit**

---

## 1. Vue d'ensemble du projet

| Champ                            | Détail                                                                                                                       |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Titre**                        | Analyse de données médicales massives pour la prédiction des MCV à l'aide d'Apache Spark et des réseaux de neurones profonds |
| **Contexte académique**          | Master Data Science — ISIM Monastir                                                                                          |
| **Période**                      | Février → Mai 2026                                                                                                           |
| **Dataset**                      | BRFSS 2020–2024 ([Kaggle](https://www.kaggle.com/datasets/ajenks/brfss-2020-2024-cleaned-and-weighted))                      |
| **Environnement d'entraînement** | Kaggle Notebooks (GPU T4 x2)                                                                                                 |
| **Environnement de démo**        | Local — Dashboard Streamlit                                                                                                  |
| **Variable cible**               | `HeartDiseaseorAttack` (binaire : 0 / 1)                                                                                     |

---

## 2. Objectifs du projet

### 2.1 Objectif principal

Concevoir et implémenter un pipeline Big Data end-to-end capable d'ingérer, prétraiter et modéliser les données épidémiologiques BRFSS pour prédire le risque de maladie cardiovasculaire (MCV), avec une comparaison rigoureuse entre approches classiques (XGBoost) et Deep Learning (DNN).

### 2.2 Objectifs secondaires

- Démontrer l'apport de **Apache Spark** pour le traitement distribué scalable sur >2M d'enregistrements.
- Comparer quantitativement les performances des modèles ML/DL sur des données médicales déséquilibrées (~9,4% de cas positifs).
- Construire un **dashboard Streamlit interactif** pour l'exploration EDA et la présentation des résultats.
- Produire un rapport académique complet (LaTeX) documentant l'ensemble de la démarche scientifique.

---

## 3. Dataset

### 3.1 Caractéristiques générales — BRFSS 2020–2024

| Caractéristique       | Valeur                                                     |
| --------------------- | ---------------------------------------------------------- |
| Nombre d'observations | >2 millions d'enregistrements (5 années cumulées)          |
| Nombre de variables   | 18 variables sélectionnées après nettoyage                 |
| Variable cible        | `HeartDiseaseorAttack` (binaire : 0 / 1)                   |
| Prévalence MCV        | ~9,4 % de cas positifs (déséquilibre de classes)           |
| Format                | CSV, données structurées                                   |
| Source                | CDC Kaggle (`ajenks/brfss-2020-2024-cleaned-and-weighted`) |

### 3.2 Variables principales

| Variable               | Type           | Description                                      |
| ---------------------- | -------------- | ------------------------------------------------ |
| `HeartDiseaseorAttack` | Binaire        | **Cible** : 1 si maladie cardiaque diagnostiquée |
| `BMI`                  | Continue       | Indice de masse corporelle                       |
| `Smoker`               | Binaire        | Fumeur régulier (≥100 cigarettes/vie)            |
| `Stroke`               | Binaire        | Antécédents d'AVC                                |
| `Diabetes`             | Catégorielle   | Statut diabétique (oui / frontière / non)        |
| `PhysicalActivity`     | Binaire        | Activité physique au cours du dernier mois       |
| `GenHlth`              | Ordinale (1–5) | Évaluation générale de la santé                  |
| `MentHlth`             | Continue       | Jours de mauvaise santé mentale (30 jours)       |
| `PhysHlth`             | Continue       | Jours de mauvaise santé physique (30 jours)      |
| `DiffWalk`             | Binaire        | Difficulté à marcher ou monter des escaliers     |
| `Sex`                  | Binaire        | Sexe biologique                                  |
| `Age`                  | Ordinale       | Tranche d'âge (13 catégories)                    |
| `Education`            | Ordinale       | Niveau d'éducation                               |
| `Income`               | Ordinale       | Niveau de revenu                                 |
| `HighBP`               | Binaire        | Hypertension artérielle diagnostiquée            |
| `HighChol`             | Binaire        | Hypercholestérolémie diagnostiquée               |
| `HvyAlcoholConsump`    | Binaire        | Consommation excessive d'alcool                  |
| `NoDocbcCost`          | Binaire        | Renoncement aux soins pour raisons financières   |

---

## 4. Architecture technique

```
┌─────────────────────────────────────────────────────────────────┐
│                       KAGGLE NOTEBOOK                           │
│                                                                 │
│  [BRFSS CSV]                                                    │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  Spark      │───►│ Feature          │───►│  Modèles      │  │
│  │  Ingestion  │    │ Engineering      │    │  DNN / XGBoost│  │
│  │  + Cleaning │    │ + Sélection      │    │  (MLlib/TF)   │  │
│  └─────────────┘    └──────────────────┘    └───────┬───────┘  │
│                                                     │          │
│                                          ┌──────────▼───────┐  │
│                                          │  Métriques +     │  │
│                                          │  Artifacts       │  │
│                                          │  (JSON/CSV)      │  │
│                                          └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    artifacts sauvegardés
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    LOCAL — STREAMLIT DASHBOARD                  │
│                                                                 │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │  EDA      │  │  Métriques   │  │  Visualisations          │ │
│  │  Explorer │  │  Comparaison │  │  ROC / Confusion Matrix  │ │
│  └───────────┘  └──────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Stack technique

| Composant            | Technologie                                      |
| -------------------- | ------------------------------------------------ |
| Traitement distribué | Apache Spark (PySpark)                           |
| Modèle DNN           | TensorFlow / Keras                               |
| Modèle XGBoost       | `xgboost` + Spark MLlib (`GBTClassifier`)        |
| Gestion déséquilibre | SMOTE (`imbalanced-learn`) ou `scale_pos_weight` |
| Dashboard            | Streamlit                                        |
| Visualisation        | Matplotlib, Seaborn, Plotly                      |
| Environnement        | Kaggle Notebooks — dual T4 GPU                   |
| Reporting            | LaTeX (rapport académique)                       |

---

## 5. Modules fonctionnels

### Module 1 — Prétraitement et chargement des données

**Objectif :** Construire un dataset propre, encodé et normalisé prêt pour la modélisation.

**Tâches :**

- Téléchargement du dataset BRFSS depuis Kaggle (`kagglehub` ou API).
- Initialisation de la session Spark (local mode, sans HDFS).
- Lecture avec `spark.read.csv()` + inférence de schéma.
- Analyse exploratoire initiale : `printSchema()`, `describe()`, distributions des classes.
- Nettoyage :
  - Suppression des doublons.
  - Gestion des valeurs manquantes (imputation médiane pour continues, mode pour catégorielles).
  - Encodage des variables catégorielles (`StringIndexer`, `OneHotEncoder`).
- Normalisation : `StandardScaler` / `MinMaxScaler` pour les variables continues.
- Conversion Spark DataFrame ↔ Pandas selon les besoins des modèles.

**Livrables :**

- Notebook cellule 1–15 : pipeline de prétraitement documenté.
- Rapport : section 1.1.3 sur la description du dataset.

---

### Module 2 — Feature Engineering et sélection des variables

**Objectif :** Identifier les features les plus discriminantes pour la prédiction des MCV.

**Tâches :**

- Analyse de corrélation (heatmap Pearson/Spearman).
- Sélection par importance Gini (Random Forest préliminaire).
- Test χ² pour les variables catégorielles vs cible.
- Réduction de dimension : PCA optionnelle si >15 features retenues.
- Construction de variables dérivées :
  - `RiskScore` : score composite (HighBP + HighChol + Smoker + Diabetes).
  - `HealthIndex` : combinaison GenHlth + MentHlth + PhysHlth.
- Justification des features retenues (référence à la littérature médicale).

**Livrables :**

- Liste finale des N features retenues + justification dans le rapport.
- Visualisations : feature importance plot, correlation matrix.

---

### Module 3 — Entraînement des modèles

#### 3.1 DNN (Deep Neural Network)

**Architecture cible :**

```
Input (N features)
    → Dense(256, relu) + BatchNorm + Dropout(0.3)
    → Dense(128, relu) + BatchNorm + Dropout(0.3)
    → Dense(64, relu)
    → Dense(1, sigmoid)
```

**Configuration :**

- Optimizer : Adam (lr=1e-3, avec scheduler)
- Loss : `binary_crossentropy`
- Class weights pour gérer le déséquilibre (ou SMOTE en amont)
- Callbacks : EarlyStopping (patience=5), ReduceLROnPlateau
- Epochs : 50 max, batch_size : 512–2048

#### 3.2 XGBoost

**Configuration :**

- `n_estimators` : 300–500
- `max_depth` : 6
- `scale_pos_weight` : ratio négatif/positif (≈9,6)
- `learning_rate` : 0.05–0.1
- GridSearch / RandomizedSearch pour l'optimisation
- Version Spark : `GBTClassifier` (MLlib) pour benchmark distribué

**Benchmark Spark :**

- Mesure du temps d'exécution avec PySpark MLlib vs scikit-learn/xgboost standalone.
- Logging des durées pour chaque étape (chargement, fit, predict).

**Livrables :**

- Notebook cellules d'entraînement documentées.
- Fichiers sauvegardés : `dnn_model.h5`, `xgboost_model.json`, `metrics.json`.

---

### Module 4 — Évaluation et comparaison des modèles

**Métriques calculées pour chaque modèle :**

| Métrique                      | Outil                    |
| ----------------------------- | ------------------------ |
| Accuracy                      | `sklearn.metrics`        |
| Precision / Recall / F1-score | `classification_report`  |
| AUC-ROC                       | `roc_auc_score`          |
| Courbe ROC                    | `roc_curve`              |
| Matrice de confusion          | `confusion_matrix`       |
| Temps d'entraînement          | `time.time()` / `%%time` |
| Temps d'inférence             | mesuré sur le test set   |

**Analyses :**

- Tableau comparatif DNN vs XGBoost sur toutes les métriques.
- Analyse du déséquilibre de classes : impact sur recall de la classe positive.
- Analyse de l'apport de Spark : speedup ratio vs mode local.
- Courbes d'apprentissage (training vs validation loss/accuracy).

**Livrables :**

- `results/metrics_comparison.csv`
- `results/roc_curves.png`
- `results/confusion_matrices.png`

---

### Module 5 — Dashboard Streamlit

**Structure de l'application :**

```
app.py
├── 📊 Page 1 : EDA — Exploration des données
│   ├── Distribution de la variable cible (pie chart)
│   ├── Distributions univariées par feature
│   ├── Heatmap de corrélation interactive (Plotly)
│   └── Détection des outliers (boxplots BMI, MentHlth, PhysHlth)
│
├── 🧠 Page 2 : Résultats des modèles
│   ├── Tableau comparatif DNN vs XGBoost
│   ├── Courbes ROC superposées
│   ├── Matrices de confusion (DNN et XGBoost côte à côte)
│   └── Feature importance (XGBoost)
│
└── ⚡ Page 3 : Impact Spark
    ├── Graphique temps d'exécution : Spark vs sans Spark
    ├── Scalabilité : temps en fonction du volume de données
    └── Résumé des conclusions
```

**Exigences techniques :**

- Chargement des données : lecture depuis `data/brfss_sample.csv` (échantillon local de ~100K lignes).
- Chargement des modèles et métriques depuis les fichiers JSON/CSV sauvegardés depuis Kaggle.
- Sidebar : filtres par année, tranche d'âge, sexe.
- Mode dark recommandé (`st.set_page_config(layout="wide")`).
- Cache Streamlit (`@st.cache_data`) pour éviter les rechargements coûteux.

**Livrables :**

- `app.py` + `requirements.txt`
- README d'installation et lancement local

---

## 6. Structure du projet Kaggle Notebook

```
notebook/
├── 01_setup_and_data_loading.ipynb
├── 02_preprocessing_and_eda.ipynb
├── 03_feature_engineering.ipynb
├── 04_dnn_training.ipynb
├── 05_xgboost_spark_training.ipynb
├── 06_evaluation_comparison.ipynb
└── outputs/
    ├── metrics.json
    ├── metrics_comparison.csv
    ├── dnn_model.h5
    ├── xgboost_model.json
    ├── roc_curves.png
    └── confusion_matrices.png

streamlit_dashboard/
├── app.py
├── pages/
│   ├── 1_EDA.py
│   ├── 2_Model_Results.py
│   └── 3_Spark_Impact.py
├── data/
│   └── brfss_sample.csv        ← échantillon 100K lignes
├── results/
│   └── *.json / *.csv / *.png  ← artifacts depuis Kaggle
└── requirements.txt
```

---

## 7. Plan de développement

| Phase       | Tâches principales                               | Livrable                                 |
| ----------- | ------------------------------------------------ | ---------------------------------------- |
| **Phase 1** | Setup Kaggle, chargement BRFSS, analyse initiale | Notebook `01` + `02`                     |
| **Phase 2** | Feature engineering, sélection des variables     | Notebook `03` + rapport section 2        |
| **Phase 3** | Entraînement DNN                                 | Notebook `04` + `dnn_model.h5`           |
| **Phase 4** | Entraînement XGBoost + Spark benchmark           | Notebook `05` + métriques timing         |
| **Phase 5** | Évaluation comparative                           | Notebook `06` + `metrics_comparison.csv` |
| **Phase 6** | Dashboard Streamlit                              | `app.py` fonctionnel en local            |
| **Phase 7** | Rapport LaTeX + soutenance                       | PDF final                                |

---

## 8. Critères de succès

| Critère                     | Cible minimale                                      |
| --------------------------- | --------------------------------------------------- |
| AUC-ROC DNN                 | ≥ 0.85                                              |
| AUC-ROC XGBoost             | ≥ 0.87                                              |
| F1-score classe positive    | ≥ 0.60                                              |
| Recall classe positive      | ≥ 0.65                                              |
| Speedup Spark (large batch) | ≥ 2× vs pandas/sklearn                              |
| Dashboard Streamlit         | Fonctionnel en local, 3 pages opérationnelles       |
| Rapport                     | ≥ 30 pages, métriques réelles extraites du notebook |

---

## 9. Contraintes et risques

| Contrainte                        | Mitigation                                                                        |
| --------------------------------- | --------------------------------------------------------------------------------- |
| Session Kaggle limitée (~12h GPU) | Checkpointing des modèles (`model.save()`) après chaque epoch                     |
| Déséquilibre de classes (9,4%)    | SMOTE ou `scale_pos_weight` + focus sur recall/AUC                                |
| Dataset volumineux (>2M lignes)   | Échantillonnage stratifié pour itérations rapides, full dataset pour le run final |
| Streamlit en local sans Spark     | Charger les résultats pré-calculés (JSON/CSV) depuis les artifacts Kaggle         |
| PySpark MLlib vs xgboost natif    | Utiliser les deux et comparer explicitement (résultat = argument pour le rapport) |

---

## 10. Références bibliographiques clés

- **[7]** Nickels et al. — Random Forest sur BRFSS 2020, précision 91,2%, recall 67,3%
- **[8]** Mesdaghinia et al. — XGBoost sur BRFSS 2019, AUC 0,872
- **[9]** Patel et al. — DNN + SMOTE sur BRFSS 2021, F1-score 0,834
- **[10]** Kumar et al. — LSTM pour données longitudinales, AUC 0,891
- **[11]** Zhang et al. — Pipeline Spark MLlib sur BRFSS 2022, 3M enregistrements en <4 min sur 8 nœuds
- **[15]** CDC BRFSS — _Behavioral Risk Factor Surveillance System_
- **[21]** Chen & Guestrin — XGBoost: A Scalable Tree Boosting System
- **[22]** Vaswani et al. — Mécanismes d'attention

---

_Document généré le 28 mai 2026 — Version 1.0_
