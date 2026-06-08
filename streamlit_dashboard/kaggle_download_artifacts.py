"""
Script a coller dans une cellule Kaggle pour zipper tous les artifacts du dashboard.
Apres execution, telechargez dashboard_artifacts.zip depuis Output > Files.
"""

import os
import zipfile

OUTPUT = "/kaggle/working"

# Fichiers requis par le dashboard — organisation cible
NEEDED = {
    "data": [
        "brfss_sample.csv",
    ],
    "results": [
        # JSON metriques
        "dnn_metrics.json",
        "xgboost_metrics.json",
        "spark_gbt_metrics.json",
        "spark_benchmark.json",
        "final_results_summary.json",
        "eda_stats.json",
        "feature_selection_results.json",
        # CSV
        "metrics_comparison.csv",
        "feature_synthesis.csv",
        "dnn_training_history.csv",
        # PNG Phase 1-2 (EDA)
        "target_distribution.png",
        "univariate_distributions.png",
        "correlation_heatmap.png",
        "target_correlation.png",
        "outliers_boxplots.png",
        "mcv_by_age.png",
        "corr_pearson.png",
        "corr_spearman.png",
        "corr_comparison.png",
        "feature_importance_gini.png",
        "chi2_cramers_v.png",
        "derived_features.png",
        "pca_analysis.png",
        "pca_2d.png",
        # PNG Phase 3 (DNN)
        "dnn_learning_curves.png",
        "dnn_roc_curve.png",
        "dnn_confusion_matrix.png",
        "dnn_threshold_analysis.png",
        # PNG Phase 4 (XGBoost)
        "xgboost_feature_importance.png",
        "xgboost_roc_curve.png",
        "xgboost_confusion_matrix.png",
        "spark_benchmark_comparison.png",
        # PNG Phase 5 (Evaluation)
        "roc_curves.png",
        "confusion_matrices.png",
        "metrics_comparison_barchart.png",
        "metrics_radar_chart.png",
        "timing_comparison.png",
        "spark_scalability.png",
        "class_imbalance_analysis.png",
    ],
}

zip_path = os.path.join(OUTPUT, "dashboard_artifacts.zip")
found, missing = [], []

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for folder, files in NEEDED.items():
        for fname in files:
            src = os.path.join(OUTPUT, fname)
            if os.path.exists(src):
                z.write(src, os.path.join(folder, fname))
                found.append(fname)
            else:
                missing.append(fname)

print(f"Archive : {zip_path}")
print(f"Fichiers inclus  : {len(found)}")
if missing:
    print(f"Fichiers ABSENTS : {missing}")

# Afficher la taille
size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Taille : {size_mb:.1f} MB")
print()
print("Etapes suivantes :")
print("1. Allez dans Output > Files dans Kaggle")
print("2. Telechargez dashboard_artifacts.zip")
print("3. Dezippez dans streamlit_dashboard/ (data/ et results/ seront crees)")
