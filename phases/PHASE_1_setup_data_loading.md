# PHASE 1 — Setup & Data Loading
> **Environnement : Kaggle Notebook | Runtime : GPU T4 x2**
> **Fichier à créer : `notebook/01_setup_and_data_loading.ipynb`**

---

## CE QUE TU DOIS FAIRE DANS CETTE PHASE

Tu vas créer un Jupyter Notebook Kaggle qui :
1. Installe les dépendances nécessaires
2. Initialise une session Apache Spark
3. Télécharge et charge le dataset BRFSS 2020-2024
4. Fait une analyse initiale des données

---

## ÉTAPE 1 — Première cellule : Installations

```python
# CELLULE 1 — Installations
!pip install pyspark==3.5.0 -q
!pip install xgboost==2.0.3 -q
!pip install imbalanced-learn==0.12.0 -q
!pip install kagglehub -q

print("✅ Installations terminées")
```

---

## ÉTAPE 2 — Imports globaux

```python
# CELLULE 2 — Imports
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PySpark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, StringType

print("✅ Imports OK")
```

---

## ÉTAPE 3 — Initialiser la session Spark

```python
# CELLULE 3 — Spark Session
spark = SparkSession.builder \
    .appName("BRFSS_CardiovascularPrediction") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"✅ Spark version : {spark.version}")
print(f"✅ Spark UI : {spark.sparkContext.uiWebUrl}")
```

---

## ÉTAPE 4 — Télécharger le dataset depuis Kaggle

```python
# CELLULE 4 — Téléchargement du dataset
import kagglehub

# Télécharger le dataset
path = kagglehub.dataset_download("ajenks/brfss-2020-2024-cleaned-and-weighted")
print(f"✅ Dataset téléchargé dans : {path}")

# Lister les fichiers disponibles
import os
files = os.listdir(path)
print(f"📁 Fichiers disponibles : {files}")
```

---

## ÉTAPE 5 — Trouver le bon fichier CSV et le charger

```python
# CELLULE 5 — Identifier le fichier CSV principal
import glob

# Chercher tous les CSV dans le répertoire téléchargé
csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
print(f"📄 Fichiers CSV trouvés : {csv_files}")

# Prendre le premier CSV trouvé (ou adapter si plusieurs)
CSV_PATH = csv_files[0]
print(f"✅ Fichier sélectionné : {CSV_PATH}")
```

```python
# CELLULE 6 — Charger avec Spark
t0 = time.time()

df_spark = spark.read.csv(
    CSV_PATH,
    header=True,
    inferSchema=True
)

t1 = time.time()
print(f"✅ Données chargées en {t1-t0:.2f}s")
print(f"📊 Nombre de lignes    : {df_spark.count():,}")
print(f"📊 Nombre de colonnes  : {len(df_spark.columns)}")
```

---

## ÉTAPE 6 — Analyse initiale du schéma

```python
# CELLULE 7 — Schéma et types
print("=== SCHÉMA DES DONNÉES ===")
df_spark.printSchema()
```

```python
# CELLULE 8 — Afficher les premières lignes
print("=== PREMIÈRES LIGNES ===")
df_spark.show(5, truncate=False)
```

```python
# CELLULE 9 — Statistiques descriptives
print("=== STATISTIQUES DESCRIPTIVES ===")
df_spark.describe().show()
```

---

## ÉTAPE 7 — Vérifier les colonnes attendues

```python
# CELLULE 10 — Vérification des colonnes attendues
EXPECTED_COLUMNS = [
    "HeartDiseaseorAttack", "BMI", "Smoker", "Stroke", "Diabetes",
    "PhysicalActivity", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
    "Sex", "Age", "Education", "Income", "HighBP", "HighChol",
    "HvyAlcoholConsump", "NoDocbcCost"
]

actual_cols = df_spark.columns
print("=== VÉRIFICATION DES COLONNES ===")

missing = [c for c in EXPECTED_COLUMNS if c not in actual_cols]
extra   = [c for c in actual_cols if c not in EXPECTED_COLUMNS]

if missing:
    print(f"⚠️  Colonnes MANQUANTES : {missing}")
else:
    print("✅ Toutes les colonnes attendues sont présentes")

if extra:
    print(f"ℹ️  Colonnes supplémentaires : {extra}")

print(f"\n📋 Colonnes disponibles : {actual_cols}")
```

---

## ÉTAPE 8 — Distribution de la variable cible

```python
# CELLULE 11 — Distribution de la variable cible
print("=== DISTRIBUTION DE LA VARIABLE CIBLE ===")
target_dist = df_spark.groupBy("HeartDiseaseorAttack").count().orderBy("HeartDiseaseorAttack")
target_dist.show()

# Convertir en pandas pour le graphique
target_pd = target_dist.toPandas()
total = target_pd["count"].sum()
target_pd["pct"] = (target_pd["count"] / total * 100).round(2)
print(target_pd)

# Graphique
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(
    ["Négatif (0)", "Positif (1)"],
    target_pd["count"],
    color=["#2196F3", "#F44336"],
    edgecolor="black"
)
ax.set_title("Distribution de HeartDiseaseorAttack")
ax.set_ylabel("Nombre d'observations")
for i, (v, p) in enumerate(zip(target_pd["count"], target_pd["pct"])):
    ax.text(i, v + 5000, f"{p}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("/kaggle/working/target_distribution.png", dpi=150)
plt.show()
print("✅ Graphique sauvegardé")
```

---

## ÉTAPE 9 — Vérifier les valeurs manquantes

```python
# CELLULE 12 — Valeurs manquantes
print("=== VALEURS MANQUANTES PAR COLONNE ===")

total_rows = df_spark.count()
missing_counts = []

for col_name in df_spark.columns:
    null_count = df_spark.filter(F.col(col_name).isNull()).count()
    pct = round(null_count / total_rows * 100, 2)
    missing_counts.append({"colonne": col_name, "valeurs_manquantes": null_count, "pct": pct})

missing_df = pd.DataFrame(missing_counts).sort_values("valeurs_manquantes", ascending=False)
print(missing_df.to_string(index=False))
missing_df.to_csv("/kaggle/working/missing_values_report.csv", index=False)
print("\n✅ Rapport de valeurs manquantes sauvegardé")
```

---

## ÉTAPE 10 — Sauvegarder le dataset brut en parquet (pour la suite)

```python
# CELLULE 13 — Sauvegarde Parquet
OUTPUT_PARQUET = "/kaggle/working/brfss_raw.parquet"

t0 = time.time()
df_spark.write.mode("overwrite").parquet(OUTPUT_PARQUET)
t1 = time.time()

print(f"✅ Dataset sauvegardé en Parquet : {OUTPUT_PARQUET}")
print(f"⏱️  Temps : {t1-t0:.2f}s")

# Sauvegarder aussi un échantillon CSV pour le dashboard Streamlit
sample_pd = df_spark.sample(fraction=0.05, seed=42).toPandas()
sample_pd.to_csv("/kaggle/working/brfss_sample_5pct.csv", index=False)
print(f"✅ Échantillon CSV sauvegardé : {len(sample_pd):,} lignes → /kaggle/working/brfss_sample_5pct.csv")
```

---

## ÉTAPE 11 — Résumé final de la phase

```python
# CELLULE 14 — Résumé Phase 1
summary = {
    "phase": "Phase 1 - Setup & Data Loading",
    "total_rows": df_spark.count(),
    "total_columns": len(df_spark.columns),
    "csv_path": CSV_PATH,
    "parquet_path": OUTPUT_PARQUET,
    "spark_version": spark.version
}

with open("/kaggle/working/phase1_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("=" * 50)
print("✅ PHASE 1 TERMINÉE")
print("=" * 50)
for k, v in summary.items():
    print(f"  {k}: {v}")
```

---

## FICHIERS PRODUITS PAR CETTE PHASE

| Fichier | Description |
|---|---|
| `/kaggle/working/brfss_raw.parquet` | Dataset complet en Parquet |
| `/kaggle/working/brfss_sample_5pct.csv` | Échantillon 5% pour Streamlit |
| `/kaggle/working/target_distribution.png` | Graphique distribution cible |
| `/kaggle/working/missing_values_report.csv` | Rapport valeurs manquantes |
| `/kaggle/working/phase1_summary.json` | Résumé de la phase |

---

## ⚠️ INTERVENTION HUMAINE REQUISE AVANT DE COMMENCER

> **Tu dois faire les choses suivantes AVANT de lancer ce notebook :**

1. **Créer un compte Kaggle** si ce n'est pas déjà fait.
2. **Aller sur** https://www.kaggle.com/datasets/ajenks/brfss-2020-2024-cleaned-and-weighted et cliquer sur **"+ New Notebook"** pour créer le notebook directement dans le dataset.
3. **Activer le GPU** : Settings → Accelerator → **GPU T4 x2**
4. **Activer Internet** : Settings → Internet → **On** (nécessaire pour `kagglehub`)
5. **Vérifier** que le runtime est bien Python 3 (pas R).

> **À la fin de cette phase, tu dois télécharger manuellement :**
> - `/kaggle/working/brfss_sample_5pct.csv` → le mettre dans `streamlit_dashboard/data/`
> - `/kaggle/working/phase1_summary.json` → le mettre dans `streamlit_dashboard/results/`
