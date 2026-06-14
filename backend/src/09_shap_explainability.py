#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import mlflow
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain

# ==========================================
# 1. SETUP PATH & MLFLOW
# ==========================================
root_path = Path(__file__).resolve().parent.parent
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=root_path / ".env")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (root_path / "mlruns").as_uri())
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("09_SHAP_Explainability")

print("🔍 1. Memuat Data dan Model...")
# Load test data
test_df = pd.read_csv(root_path / "Data/split/test_data.csv")
target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']
target_labels = ['Depression', 'Anxiety', 'Stress']

# Gunakan 15 fitur spesifik dari proses seleksi fitur B-MFO
selected_features = [
    'TIPI1', 'TIPI2', 'TIPI3', 'TIPI4', 'TIPI5', 'TIPI6', 'TIPI7', 'TIPI8', 
    'TIPI9', 'TIPI10', 'education', 'urban', 'gender', 'age', 'familysize'
]
X_test = test_df[selected_features]
y_test = test_df[target_cols]
print(f"   ✓ Jumlah fitur: {len(selected_features)}")

# Load Model ClassifierChain yang asli
model_path = root_path / "models" / "multilabel_xgboost_classifier_chain.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)
print("   ✓ Model ClassifierChain berhasil dimuat")

output_dir = root_path / "outputs" / "shap_figures"
output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. SHAP ANALYSIS UNTUK CLASSIFIER CHAIN
# ==========================================
X_sample = X_test.sample(500, random_state=42) if len(X_test) > 500 else X_test
X_sample_aug = X_sample.copy().values # Konversi ke numpy agar mudah di-stack seperti saat training

with mlflow.start_run(run_name="SHAP_Analysis_ClassifierChain"):
    print("\n🧠 2. Memulai Ekstraksi SHAP Values...")
    all_shap_importance = {}
    
    for i, (target, label) in enumerate(zip(target_cols, target_labels)):
        print(f"\n   → Menganalisis alasan AI untuk kondisi: {label}...")
        estimator = model.estimators_[i]
        
        # Tentukan nama fitur untuk tahapan ini
        current_feature_names = selected_features.copy()
        if i == 1:
            current_feature_names += ['pred_depression']
        elif i == 2:
            current_feature_names += ['pred_depression', 'pred_anxiety']
            
        # Konversi X_sample_aug ke DataFrame sementara untuk SHAP dengan nama fitur yang benar
        X_shap_df = pd.DataFrame(X_sample_aug, columns=current_feature_names)
        
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_shap_df)
        
        # Generate summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap_df, show=False, max_display=15)
        plt.title(f'SHAP Feature Importance: Faktor Pemicu {label}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        fig_path = output_dir / f"shap_summary_{label.lower()}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(str(fig_path))
        print(f"      ✓ Saved: {fig_path.name}")
        
        # Generate local explanation (waterfall) for first sample
        sample_idx = 0
        explanation = explainer(X_shap_df.iloc[[sample_idx]])
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation[0], show=False)
        plt.title(f'SHAP Local Explanation: {label} (Responden #{sample_idx})', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        waterfall_path = output_dir / f"shap_waterfall_{label.lower()}.png"
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(str(waterfall_path))
        print(f"      ✓ Saved: {waterfall_path.name}")
        
        # Get Top Features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:10]
        top_features = {current_feature_names[idx]: float(mean_abs_shap[idx]) for idx in top_indices}
        all_shap_importance[label] = top_features
        
        # Update X_sample_aug untuk estimator berikutnya pada ClassifierChain
        # Prediksi threshold 0.5 lalu append ke X_sample_aug
        pred_proba = estimator.predict_proba(X_sample_aug)[:, 1]
        pred_label = (pred_proba >= 0.5).astype(int).reshape(-1, 1)
        X_sample_aug = np.hstack([X_sample_aug, pred_label])
        
    mlflow.log_dict(all_shap_importance, "shap_top_features_cc.json")
    
    print("\n" + "="*60)
    print("🎉 SHAP ANALYSIS CLASSIFIER CHAIN SELESAI!")
    print("="*60)
    for label, features in all_shap_importance.items():
        print(f"\n📊 Top 5 Faktor Pemicu {label}:")
        for rank, (feat, importance) in enumerate(list(features.items())[:5], 1):
            print(f"   {rank}. {feat}: {importance:.4f}")
