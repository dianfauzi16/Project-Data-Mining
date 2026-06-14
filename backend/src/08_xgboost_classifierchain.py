#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import time
import json
from pathlib import Path
from sklearn.metrics import f1_score, classification_report, hamming_loss, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain
import mlflow
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. SETUP MLflow
# ==========================================
root_path = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=root_path / ".env")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (root_path / "mlruns").as_uri())
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("08c_XGB_ClassifierChain_Final")

# Pastikan tidak ada run MLflow yang menggantung
if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name="XGBoost_ClassifierChain_Final_DefaultThreshold"):
    models_dir = root_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    params_path = models_dir / "best_xgb_params_cc.json"
    
    if not params_path.exists():
        raise FileNotFoundError(f"❌ File {params_path} tidak ditemukan. Jalankan script tuning CC terlebih dahulu!")
        
    with open(params_path, 'r') as f:
        xgb_params = json.load(f)
    print(f"✅ Loaded best parameters from {params_path}")
    print("⚠️  Model: Classifier Chains (Label Dependence) | Threshold: Default 0.5")

    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    print("\n⏳ 1. Memuat Data...")
    target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']
    
    train_full_smote = pd.read_csv(root_path / "Data/processed/train_smote.csv")
    selected_features = [col for col in train_full_smote.columns if col not in target_cols]
    
    # Pastikan berbentuk Numpy Array
    X_train_full_smote = train_full_smote[selected_features].values 
    Y_train_full_smote = train_full_smote[target_cols].astype(int).values 
    
    test_df = pd.read_csv(root_path / "Data/split/test_data.csv")
    X_test = test_df[selected_features].values
    Y_test = test_df[target_cols].astype(int).values

    # ==========================================
    # 2. TRAINING FINAL MODEL (CLASSIFIER CHAINS)
    # ==========================================
    print("\n🔨 2. Training FINAL MODEL (Classifier Chains)...")
    final_train_start = time.time()

    # FIX 1: Bersihkan parameter sampah dari JSON
    keys_to_remove = ['objective', 'eval_metric', 'use_label_encoder', 'num_class', 'early_stopping_rounds']
    clean_params = {k: v for k, v in xgb_params.items() if k not in keys_to_remove}
    
    # Inisialisasi base estimator
    base_xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        **clean_params
    )
    
    # FIX 2: Paksa Scikit-Learn mengenali ini sebagai Classifier (Mencegah error predict_proba)
    base_xgb._estimator_type = "classifier"
    
    final_model = ClassifierChain(base_xgb, order='random', random_state=42)
    final_model.fit(X_train_full_smote, Y_train_full_smote)
    
    final_train_time = time.time() - final_train_start
    print(f"   ✓ Final model selesai dilatih ({final_train_time:.2f} detik)")

    # ==========================================
    # 3. EVALUASI DI TEST SET (Threshold Default 0.5)
    # ==========================================
    print("\n📊 3. Evaluasi di Test Set (Threshold Default 0.5)...")
    
    # Prediksi kelas (untuk F1, Hamming, Exact Match)
    # y_pred berbentuk 2D array (n_samples, n_labels)
    y_pred = final_model.predict(X_test)
    macro_f1 = f1_score(Y_test, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(Y_test, y_pred, average='micro', zero_division=0)
    hamming = hamming_loss(Y_test, y_pred)
    exact_match = accuracy_score(Y_test, y_pred)
    
    # Prediksi probabilitas (untuk ROC-AUC)
    y_test_probas = final_model.predict_proba(X_test)
    
    # FIX KRUSIAL: Deteksi format output ClassifierChain vs MultiOutputClassifier
    is_list_format = isinstance(y_test_probas, list)
    
    per_label_f1_scores = {}
    roc_auc_scores = []
    
    for i, target_name in enumerate(target_cols):
        # Hitung F1 per label (Y_test dan y_pred adalah 2D array)
        per_label_f1_scores[target_name] = f1_score(Y_test[:, i], y_pred[:, i], zero_division=0)
        
        # Ambil probabilitas kelas 1 untuk label ke-i
        if is_list_format:
            # Format List (seperti MultiOutputClassifier): list[array(n_samples, 2)]
            y_prob = y_test_probas[i][:, 1]
        else:
            # Format Single 2D Array (ClassifierChain Biner): array(n_samples, n_labels)
            y_prob = y_test_probas[:, i]
            
        roc_auc_scores.append(roc_auc_score(Y_test[:, i], y_prob))
        
    # Macro ROC-AUC (Rata-rata dari ketiga label)
    macro_roc_auc = np.mean(roc_auc_scores)

    # ==========================================
    # 4. TAMPILKAN HASIL & SAVE
    # ==========================================
    print("\n" + "="*70)
    print("📊 HASIL EVALUASI (CLASSIFIER CHAINS - Threshold 0.5)")
    print("="*70)
    print(f"{'Metrik':<25} {'Nilai':>15}")
    print("-"*70)
    print(f"{'Macro_F1':<25} {macro_f1:>15.4f}")
    print(f"{'Micro_F1':<25} {micro_f1:>15.4f}")
    print(f"{'Hamming_Loss':<25} {hamming:>15.4f}")
    print(f"{'Exact_Match':<25} {exact_match:>15.4f}")
    print(f"{'Macro_ROC-AUC':<25} {macro_roc_auc:>15.4f}")
    print("="*70)

    print("\n📈 Per-Label F1 Scores:")
    for label, f1_val in per_label_f1_scores.items():
        print(f"   → {label}: {f1_val:.4f}")
        
    print("\n📈 Per-Label ROC-AUC Scores:")
    for i, label in enumerate(target_cols):
        print(f"   → {label}: {roc_auc_scores[i]:.4f}")

    print("\n📋 Classification Report:")
    class_report = classification_report(Y_test, y_pred, target_names=['Depression', 'Anxiety', 'Stress'], zero_division=0)
    print(class_report)

    model_save_path = models_dir / "multilabel_xgboost_classifier_chain.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(final_model, f)

    # ==========================================
    # 5. MLflow LOGGING & SUMMARY
    # ==========================================
    mlflow.log_param("model_architecture", "Classifier Chains")
    mlflow.log_param("threshold_strategy", "Default 0.5")
    mlflow.log_metric("test_macro_f1", macro_f1)
    mlflow.log_metric("test_micro_f1", micro_f1)
    mlflow.log_metric("test_hamming_loss", hamming)
    mlflow.log_metric("test_exact_match", exact_match)
    mlflow.log_metric("test_macro_roc_auc", macro_roc_auc)
    mlflow.log_metric("total_eval_time_seconds", final_train_time)
    
    for i, label in enumerate(target_cols):
        mlflow.log_metric(f"test_f1_{label}", per_label_f1_scores[label])
        mlflow.log_metric(f"test_roc_auc_{label}", roc_auc_scores[i])
    
    report_path = "classification_report_xgb_cc.txt"
    with open(report_path, "w") as f: f.write(class_report)
    mlflow.log_artifact(report_path)
    mlflow.log_artifact(str(model_save_path))

    print("\n" + "="*70)
    print("🎉 CLASSIFIER CHAINS MODEL SELESAI")
    print("="*70)
    print(f"Strategy     : Classifier Chains + Default Threshold (0.5)")
    print(f"Architecture : ClassifierChain (Label Dependence / Comorbidity)")
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Micro F1     : {micro_f1:.4f}")
    print(f"Hamming Loss : {hamming:.4f}")
    print(f"Exact Match  : {exact_match:.4f}")
    print(f"Macro ROC-AUC: {macro_roc_auc:.4f}")
    print(f"Waktu Train  : {final_train_time:.2f} detik")
    print("="*70)
