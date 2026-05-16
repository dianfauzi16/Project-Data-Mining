#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain

# ==========================================
# 1. SETUP PATH & MLFLOW
# ==========================================
root_path = Path.cwd()
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=root_path / ".env") # Load variabel dari .env
# Gunakan DagsHub URI jika ada di .env, jika tidak pakai folder lokal mlruns
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (root_path / "mlruns").as_uri())
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("09_SHAP_Explainability")


print("🔍 1. Memuat Data dan Melatih Model...")

# Load data
train_balanced = pd.read_csv(root_path / "Data/processed/train_selected_features.csv")
test_df = pd.read_csv(root_path / "Data/split/test_data.csv")

target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']
target_labels = ['Depression', 'Anxiety', 'Stress']

X_train = train_balanced.drop(columns=target_cols)
y_train = train_balanced[target_cols].astype(int)

X_test = test_df.drop(columns=target_cols)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

feature_names = X_train.columns.tolist()
print(f"   ✓ Jumlah fitur: {len(feature_names)}")
print(f"   ✓ Data test: {X_test.shape}")

# ==========================================
# 2. TRAINING MODEL FRESH (agar fitur konsisten)
# ==========================================
print("\n🔧 2. Training model untuk SHAP analysis...")

# ✅ Load parameter hasil tuning (bukan hardcode)
import json
params_path = root_path / "models" / "best_xgb_params.json"
with open(params_path, 'r') as f:
    xgb_params = json.load(f)
print(f"✓ Loaded tuned params: {xgb_params}")

# Train 3 model INDEPENDEN (bukan ClassifierChain)
# Karena SHAP TreeExplainer butuh model individual tanpa augmented features
individual_models = {}
for i, target in enumerate(target_cols):
    print(f"   → Training model untuk {target}...")
    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train[target])
    individual_models[target] = model
print("   ✓ Semua model selesai dilatih")

# Folder output
output_dir = root_path / "outputs" / "shap_figures"
output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 3. SHAP ANALYSIS
# ==========================================
X_sample = X_test.sample(500, random_state=42) if len(X_test) > 500 else X_test

with mlflow.start_run(run_name="SHAP_Analysis"):
    print("\n🧠 3. Memulai Ekstraksi SHAP Values...")
    
    all_shap_importance = {}
    
    for i, (target, label) in enumerate(zip(target_cols, target_labels)):
        print(f"\n   → Menganalisis alasan AI untuk kondisi: {label}...")
        
        model = individual_models[target]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # ==========================================
        # SHAP SUMMARY PLOT (Beeswarm)
        # ==========================================
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
        plt.title(f'SHAP Feature Importance: Faktor Pemicu {label}', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        fig_path = output_dir / f"shap_summary_{label.lower()}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(str(fig_path))
        print(f"      ✓ Saved: {fig_path.name}")
        
        # ==========================================
        # SHAP BAR PLOT (Mean Absolute)
        # ==========================================
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
        plt.title(f'Mean |SHAP|: Top 15 Fitur untuk {label}', 
                  fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        
        bar_path = output_dir / f"shap_bar_{label.lower()}.png"
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(str(bar_path))
        print(f"      ✓ Saved: {bar_path.name}")
        
        # Simpan top features untuk logging
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:10]
        top_features = {feature_names[idx]: float(mean_abs_shap[idx]) for idx in top_indices}
        all_shap_importance[label] = top_features
    
    # Log top features ke MLflow
    mlflow.log_dict(all_shap_importance, "shap_top_features.json")
    
    # ==========================================
    # 4. RANGKUMAN
    # ==========================================
    print("\n" + "="*60)
    print("🎉 SHAP ANALYSIS SELESAI!")
    print("="*60)
    
    for label, features in all_shap_importance.items():
        print(f"\n📊 Top 5 Faktor Pemicu {label}:")
        for rank, (feat, importance) in enumerate(list(features.items())[:5], 1):
            print(f"   {rank}. {feat}: {importance:.4f}")
    
    print(f"\n📁 Grafik tersimpan di: {output_dir}")
    print("="*60)

