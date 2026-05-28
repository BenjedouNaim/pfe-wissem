# PHASE 2 — Prétraitement & EDA
> **Environnement : Kaggle Notebook | Fichier : `02_preprocessing_eda.ipynb`**
> **PRÉREQUIS : Phase 1 terminée — `/kaggle/working/brfss_raw.parquet` doit exister**

---

## CE QUE TU DOIS FAIRE DANS CETTE PHASE

Tu vas nettoyer les données, encoder les variables, normaliser les features, et produire les visualisations EDA.

---

## ÉTAPE 1 — Setup (recharger Spark + lire le Parquet)

```python
# CELLULE 1 — Setup
import os, time, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("BRFSS_Preprocessing") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Charger le Parquet de la Phase 1
df = spark.read.parquet("/kaggle/working/brfss_raw.parquet")
print(f"✅ Données chargées : {df.count():,} lignes × {len(df.columns)} colonnes")
```

---

## ÉTAPE 2 — Définir les listes de colonnes par type

```python
# CELLULE 2 — Définition des colonnes par type

TARGET_COL = "HeartDiseaseorAttack"

# Variables binaires (0/1)
BINARY_COLS = [
    "Smoker", "Stroke", "PhysicalActivity", "DiffWalk",
    "Sex", "HighBP", "HighChol", "HvyAlcoholConsump", "NoDocbcCost"
]

# Variables catégorielles multi-valeurs
CATEGORICAL_COLS = ["Diabetes"]

# Variables ordinales (déjà numériques, pas d'encodage OHE nécessaire)
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income"]

# Variables continues
CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]

# Toutes les features (sans la target)
ALL_FEATURE_COLS = BINARY_COLS + CATEGORICAL_COLS + ORDINAL_COLS + CONTINUOUS_COLS

print(f"✅ Features totales : {len(ALL_FEATURE_COLS)}")
print(f"   Binaires    : {BINARY_COLS}")
print(f"   Catégorielles : {CATEGORICAL_COLS}")
print(f"   Ordinales   : {ORDINAL_COLS}")
print(f"   Continues   : {CONTINUOUS_COLS}")
```

---

## ÉTAPE 3 — Supprimer les doublons

```python
# CELLULE 3 — Dédoublonnage
count_before = df.count()
df = df.dropDuplicates()
count_after = df.count()

print(f"Avant dédoublonnage : {count_before:,}")
print(f"Après dédoublonnage : {count_after:,}")
print(f"Doublons supprimés  : {count_before - count_after:,}")
```

---

## ÉTAPE 4 — Gérer les valeurs manquantes

```python
# CELLULE 4 — Imputation des valeurs manquantes

# Pour les variables continues : imputer avec la médiane
print("=== IMPUTATION DES CONTINUES (médiane) ===")
for col_name in CONTINUOUS_COLS:
    median_val = df.approxQuantile(col_name, [0.5], 0.001)[0]
    null_count = df.filter(F.col(col_name).isNull()).count()
    if null_count > 0:
        df = df.fillna({col_name: median_val})
        print(f"  {col_name} : {null_count} valeurs imputées avec médiane={median_val:.2f}")
    else:
        print(f"  {col_name} : pas de manquants ✅")

# Pour les variables binaires et ordinales : imputer avec le mode
print("\n=== IMPUTATION DES BINAIRES/ORDINALES (mode) ===")
for col_name in BINARY_COLS + ORDINAL_COLS + CATEGORICAL_COLS:
    null_count = df.filter(F.col(col_name).isNull()).count()
    if null_count > 0:
        mode_val = df.groupBy(col_name).count().orderBy("count", ascending=False).first()[0]
        df = df.fillna({col_name: mode_val})
        print(f"  {col_name} : {null_count} valeurs imputées avec mode={mode_val}")
    else:
        print(f"  {col_name} : pas de manquants ✅")

print(f"\n✅ Dataset après imputation : {df.count():,} lignes")
```

---

## ÉTAPE 5 — Encoder la variable catégorielle Diabetes

```python
# CELLULE 5 — Encodage de Diabetes (StringIndexer + OHE si besoin)

# Vérifier les valeurs uniques de Diabetes
print("Valeurs uniques de Diabetes :")
df.groupBy("Diabetes").count().orderBy("Diabetes").show()

# Encoder en numérique si c'est du texte (oui/frontière/non → 0/1/2)
# Si la colonne est déjà numérique, cette étape n'est pas nécessaire
diabetes_type = dict(df.dtypes)["Diabetes"]
print(f"Type de Diabetes : {diabetes_type}")

if diabetes_type == "string":
    indexer = StringIndexer(inputCol="Diabetes", outputCol="Diabetes_encoded")
    df = indexer.fit(df).transform(df)
    df = df.drop("Diabetes").withColumnRenamed("Diabetes_encoded", "Diabetes")
    print("✅ Diabetes encodé (StringIndexer)")
else:
    print("✅ Diabetes déjà numérique, pas d'encodage nécessaire")
```

---

## ÉTAPE 6 — Convertir toutes les colonnes en Float

```python
# CELLULE 6 — Cast en Float (nécessaire pour Spark ML)

for col_name in ALL_FEATURE_COLS + [TARGET_COL]:
    df = df.withColumn(col_name, F.col(col_name).cast("float"))

print("✅ Toutes les colonnes castées en Float")
df.select(ALL_FEATURE_COLS[:5] + [TARGET_COL]).show(3)
```

---

## ÉTAPE 7 — Supprimer les lignes avec target nulle

```python
# CELLULE 7 — Nettoyage de la target
null_target = df.filter(F.col(TARGET_COL).isNull()).count()
print(f"Lignes avec target null : {null_target}")
df = df.dropna(subset=[TARGET_COL])
print(f"✅ Dataset final : {df.count():,} lignes")
```

---

## ÉTAPE 8 — EDA visualisations (sur un échantillon Pandas)

```python
# CELLULE 8 — Convertir un échantillon en Pandas pour les graphiques
# On prend 50 000 lignes pour les visualisations (plus rapide)
df_pd = df.sample(fraction=0.02, seed=42).toPandas()
print(f"✅ Échantillon pour EDA : {len(df_pd):,} lignes")
```

```python
# CELLULE 9 — Distribution de toutes les features (histogrammes)
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for i, col_name in enumerate(ALL_FEATURE_COLS):
    axes[i].hist(df_pd[col_name].dropna(), bins=20, color="#2196F3", edgecolor="black", alpha=0.7)
    axes[i].set_title(col_name, fontsize=10)
    axes[i].set_xlabel("Valeur")
    axes[i].set_ylabel("Fréquence")

# Masquer les axes en trop
for j in range(len(ALL_FEATURE_COLS), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Distribution des features BRFSS", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("/kaggle/working/eda_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Sauvegardé : eda_feature_distributions.png")
```

```python
# CELLULE 10 — Boxplots des variables continues par classe
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, col_name in enumerate(CONTINUOUS_COLS):
    df_pd.boxplot(column=col_name, by=TARGET_COL, ax=axes[i])
    axes[i].set_title(f"{col_name} par classe")
    axes[i].set_xlabel("HeartDiseaseorAttack")

plt.suptitle("Variables continues selon la variable cible")
plt.tight_layout()
plt.savefig("/kaggle/working/eda_boxplots_continuous.png", dpi=150)
plt.show()
print("✅ Sauvegardé : eda_boxplots_continuous.png")
```

```python
# CELLULE 11 — Heatmap de corrélation
plt.figure(figsize=(14, 10))
corr_matrix = df_pd[ALL_FEATURE_COLS + [TARGET_COL]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask,
    annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1,
    linewidths=0.5, annot_kws={"size": 8}
)
plt.title("Matrice de corrélation — BRFSS Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("/kaggle/working/eda_correlation_heatmap.png", dpi=150)
plt.show()
print("✅ Sauvegardé : eda_correlation_heatmap.png")

# Sauvegarder la matrice pour le dashboard
corr_matrix.to_csv("/kaggle/working/correlation_matrix.csv")
print("✅ Sauvegardé : correlation_matrix.csv")
```

```python
# CELLULE 12 — Corrélations avec la TARGET (bar chart)
target_corr = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)

plt.figure(figsize=(10, 7))
colors = ["#F44336" if v > 0 else "#2196F3" for v in target_corr.values]
plt.barh(target_corr.index, target_corr.values, color=colors, edgecolor="black")
plt.axvline(x=0, color="black", linewidth=0.8)
plt.title(f"Corrélation des features avec {TARGET_COL}", fontsize=13, fontweight="bold")
plt.xlabel("Corrélation de Pearson")
plt.tight_layout()
plt.savefig("/kaggle/working/eda_target_correlation.png", dpi=150)
plt.show()
print("✅ Sauvegardé : eda_target_correlation.png")
```

---

## ÉTAPE 9 — Sauvegarder le dataset nettoyé

```python
# CELLULE 13 — Sauvegarde du dataset prétraité
CLEAN_PARQUET = "/kaggle/working/brfss_clean.parquet"

t0 = time.time()
df.select(ALL_FEATURE_COLS + [TARGET_COL]).write.mode("overwrite").parquet(CLEAN_PARQUET)
t1 = time.time()

print(f"✅ Dataset nettoyé sauvegardé : {CLEAN_PARQUET}")
print(f"⏱️  Temps : {t1-t0:.2f}s")
print(f"📊 Lignes : {df.count():,} | Colonnes : {len(ALL_FEATURE_COLS)+1}")
```

```python
# CELLULE 14 — Sauvegarder la config des colonnes pour les phases suivantes
columns_config = {
    "target": TARGET_COL,
    "binary_cols": BINARY_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "ordinal_cols": ORDINAL_COLS,
    "continuous_cols": CONTINUOUS_COLS,
    "all_feature_cols": ALL_FEATURE_COLS,
    "total_features": len(ALL_FEATURE_COLS),
    "total_rows_clean": df.count()
}

with open("/kaggle/working/columns_config.json", "w") as f:
    json.dump(columns_config, f, indent=2)

print("✅ Configuration des colonnes sauvegardée : columns_config.json")
print(json.dumps(columns_config, indent=2))
```

---

## FICHIERS PRODUITS PAR CETTE PHASE

| Fichier | Description |
|---|---|
| `/kaggle/working/brfss_clean.parquet` | Dataset nettoyé, encodé, prêt |
| `/kaggle/working/columns_config.json` | Config colonnes par type |
| `/kaggle/working/correlation_matrix.csv` | Matrice de corrélation |
| `/kaggle/working/eda_feature_distributions.png` | Histogrammes features |
| `/kaggle/working/eda_boxplots_continuous.png` | Boxplots continues/cible |
| `/kaggle/working/eda_correlation_heatmap.png` | Heatmap corrélation |
| `/kaggle/working/eda_target_correlation.png` | Corrélation avec target |

---

## ⚠️ INTERVENTION HUMAINE REQUISE

> **Aucune action requise AVANT cette phase.**

> **À la fin de cette phase, tu dois télécharger et déplacer :**
> - `correlation_matrix.csv` → `streamlit_dashboard/results/`
> - `eda_correlation_heatmap.png` → `streamlit_dashboard/results/`
> - `eda_target_correlation.png` → `streamlit_dashboard/results/`
> - `columns_config.json` → `streamlit_dashboard/results/`
