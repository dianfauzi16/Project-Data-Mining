#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
from pathlib import Path
import mlflow

# ==========================================
# 0. MLFLOW SETUP
# ==========================================
root_path = Path.cwd()
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=root_path / ".env") # Load variabel dari .env
# Gunakan DagsHub URI jika ada di .env, jika tidak pakai folder lokal mlruns
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (root_path / "mlruns").as_uri())
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("07d_Baseline_Benchmarking")

# ==========================================
# 1. SETUP & LOAD DATA
# ==========================================
root_path = Path.cwd()
print("🔍 1. Memuat Data untuk Benchmarking...")

train_df = pd.read_csv(root_path / "Data/processed/train_selected_features.csv")
test_df = pd.read_csv(root_path / "Data/split/test_data.csv")

# Menggunakan fitur hasil seleksi B-MFO
target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']
features = [col for col in train_df.columns if col not in target_cols]

X_train, Y_train = train_df[features], train_df[target_cols]
X_test, Y_test = test_df[features], test_df[target_cols]

# SVM sangat sensitif terhadap skala data, jadi kita lakukan Scaling khusus untuk SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. DEFINISI MODEL BENCHMARK
# ==========================================
print("🥊 2. Mempersiapkan Tiga Arsitektur Berbeda...")

models = {
    # Main Model: B-MFO + XGBoost + Classifier Chains
    "B-MFO + XGBoost (Chains)": ClassifierChain(
        xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
        order='random', random_state=42
    ),
    
    # Baseline 1: Random Forest + Binary Relevance (OvR)
    "B-MFO + RF (Binary Relevance)": MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42)
    ),
    
    # Baseline 2: SVM + Binary Relevance (OvR)
    "B-MFO + SVM (Binary Relevance)": MultiOutputClassifier(
        SVC(probability=True, random_state=42)
    )
}

# ==========================================
# 3. EKSEKUSI & EVALUASI
# ==========================================
benchmarking_results = []

with mlflow.start_run(run_name="Baseline_Benchmarking_Comparison"):
    for name, model in models.items():
        print(f"⏳ Mengevaluasi {name}...")
        start_time = time.time()
        
        # Gunakan data scaled khusus untuk SVM
        X_tr = X_train_scaled if "SVM" in name else X_train
        X_te = X_test_scaled if "SVM" in name else X_test
        
        model.fit(X_tr, Y_train)
        predictions = model.predict(X_te)
        
        exec_time = time.time() - start_time
        
        # Hitung Metrik
        macro_f1 = f1_score(Y_test, predictions, average='macro')
        micro_f1 = f1_score(Y_test, predictions, average='micro')
        hamming = hamming_loss(Y_test, predictions)
        exact_match = accuracy_score(Y_test, predictions)
        
        # Hitung ROC-AUC secara aman
        # Hitung ROC-AUC secara aman menghindari error ClassifierChain predict_proba
        roc_auc_scores = []
        try:
            if isinstance(model, ClassifierChain):
                n_samples = X_te.shape[0]
                n_targets = len(model.estimators_)
                y_pred_proba = np.zeros((n_samples, n_targets))
                X_arr = X_te.values if hasattr(X_te, "values") else np.array(X_te)
                X_aug = X_arr.copy()
                for i, estimator in enumerate(model.estimators_):
                    proba = estimator.predict_proba(X_aug)[:, 1]
                    y_pred_proba[:, i] = proba
                    pred_label = (proba >= 0.5).astype(int).reshape(-1, 1)
                    X_aug = np.hstack([X_aug, pred_label])
            else:
                probas = model.predict_proba(X_te)
                y_pred_proba = np.column_stack([p[:, 1] for p in probas])

            for i, target_name in enumerate(target_cols):
                y_true = Y_test.iloc[:, i] if hasattr(Y_test, "iloc") else Y_test[:, i]
                roc_auc_scores.append(roc_auc_score(y_true, y_pred_proba[:, i]))
            macro_roc_auc = np.mean(roc_auc_scores)
        except Exception as e:
            print(f"Could not calculate ROC-AUC for {name}: {e}")
            macro_roc_auc = np.nan
        
        
        results = {
            "Model": name,
            "Macro F1": macro_f1,
            "Micro F1": micro_f1,
            "Hamming Loss": hamming,
            "Exact Match": exact_match,
            "Macro ROC-AUC": macro_roc_auc,
            "Time (s)": exec_time
        }
        benchmarking_results.append(results)
        
        # Log ke MLflow
        model_name_clean = name.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
        mlflow.log_metric(f"{model_name_clean}_Macro_F1", macro_f1)
        mlflow.log_metric(f"{model_name_clean}_Hamming_Loss", hamming)
        mlflow.log_metric(f"{model_name_clean}_Exact_Match", exact_match)
        mlflow.log_metric(f"{model_name_clean}_Macro_ROC_AUC", macro_roc_auc)
        mlflow.log_metric(f"{model_name_clean}_Execution_Time_s", exec_time)

    # ==========================================
    # 4. TABEL PERBANDINGAN FINAL
    # ==========================================
    df_results = pd.DataFrame(benchmarking_results).sort_values(by="Macro F1", ascending=False)
    print("\n" + "="*70)
    print("📊 HASIL BENCHMARKING MODEL AKHIR")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)
    
    # Log summary ke MLflow
    mlflow.log_dict(df_results.to_dict(orient='records'), "benchmark_results.json")

