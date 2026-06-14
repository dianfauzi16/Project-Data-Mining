#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score, classification_report, hamming_loss, accuracy_score, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import optuna
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 0. SETUP MLflow & PATH
# ==========================================
root_path = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=root_path / ".env")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (root_path / "mlruns").as_uri())
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("08_SVM_Baseline_Optuna_Full_Pipeline")

with mlflow.start_run(run_name="SVM_BR_Optuna_DefaultThreshold"):
    # ==========================================
    # 1. LOAD DATA (Pipeline Single-Split)
    # ==========================================
    print("⏳ 1. Memuat Data...")
    target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']
    
    # Memuat data train yang sudah di-SMOTE dan data test asli
    train_smote = pd.read_csv(root_path / "Data/processed/train_smote.csv")
    feature_names = [col for col in train_smote.columns if col not in target_cols]
    num_features = len(feature_names)
    
    X_train = train_smote[feature_names].values
    Y_train = train_smote[target_cols].astype(int).values
    
    test_df = pd.read_csv(root_path / "Data/split/test_data.csv")
    X_test = test_df[feature_names].values
    Y_test = test_df[target_cols].astype(int).values

    print(f"   ✓ Fitur: {num_features}")
    print(f"   ✓ Train (SMOTE) : {X_train.shape}")
    print(f"   ✓ Test (Asli)   : {X_test.shape}")

    # ==========================================
    # 2. HYPERPARAMETER TUNING DENGAN OPTUNA
    # ==========================================
    print("\n⏳ 2. Memulai Hyperparameter Tuning LinearSVC (Optuna)...")
    tune_start = time.time()
    
    mlkf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='macro', zero_division=0)

    def objective(trial):
        C = trial.suggest_float('C', 1e-3, 100.0, log=True)
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        max_iter = trial.suggest_int('max_iter', 2000, 5000, step=1000)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', CalibratedClassifierCV(
                LinearSVC(
                    C=C, class_weight=class_weight, loss='squared_hinge', 
                    penalty='l2', max_iter=max_iter, random_state=42, dual=False
                ),
                cv=3, method='sigmoid'
            ))
        ])
        
        model = MultiOutputClassifier(pipeline, n_jobs=-1)
        cv_results = cross_validate(model, X_train, Y_train, cv=mlkf, scoring=scorer, n_jobs=1)
        return cv_results['test_score'].mean()

    study = optuna.create_study(direction='maximize', study_name="SVM_Optuna")
    study.optimize(objective, n_trials=50)
    
    svm_params = study.best_params.copy()
    svm_params['loss'] = 'squared_hinge'
    svm_params['penalty'] = 'l2'
    
    tune_time = time.time() - tune_start
    print(f"\n   🏆 Best Macro F1 (Optuna) : {study.best_value:.4f}")
    print(f"   🔧 Best Parameters        : {svm_params}")
    print(f"   ⏱️ Waktu Tuning           : {tune_time:.2f} detik")
        
    # Ekstrak Stabilitas
    print("\n📊 Analisis Stabilitas Model (Evaluasi Ulang Best Params)...")
    def build_pipeline(params):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', CalibratedClassifierCV(
                LinearSVC(C=params['C'], class_weight=params['class_weight'], loss='squared_hinge',
                          penalty='l2', max_iter=params['max_iter'], random_state=42, dual=False),
                cv=3, method='sigmoid'
            ))
        ])
        
    best_model_cv = MultiOutputClassifier(build_pipeline(svm_params), n_jobs=-1)
    best_cv_results = cross_validate(best_model_cv, X_train, Y_train, cv=mlkf, scoring=scorer, n_jobs=1)
    std_cv_f1 = best_cv_results['test_score'].std()
    print(f"   → Standar Deviasi (Stabilitas): ± {std_cv_f1:.4f}")

    # ==========================================
    # 3. FINAL MODEL TRAINING & EVALUATION (Default Threshold 0.5)
    # ==========================================
    print("\n🔨 3. Training FINAL MODEL LinearSVC...")
    final_train_start = time.time()
    
    final_svm_model = MultiOutputClassifier(build_pipeline(svm_params), n_jobs=-1)
    final_svm_model.fit(X_train, Y_train)
    final_train_time = time.time() - final_train_start
    
    # Evaluasi menggunakan predict() yang otomatis memakai threshold default 0.5
    print("\n📊 4. Evaluasi di Test Set (Threshold Default 0.5)...")
    y_pred = final_svm_model.predict(X_test)
    
    macro_f1 = f1_score(Y_test, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(Y_test, y_pred, average='micro', zero_division=0)
    hamming = hamming_loss(Y_test, y_pred)
    exact_match = accuracy_score(Y_test, y_pred)

    # ROC-AUC tetap membutuhkan probabilitas
    y_test_probas = final_svm_model.predict_proba(X_test)
    roc_auc_scores = []
    per_label_f1 = {}
    
    for i, target_name in enumerate(target_cols):
        y_prob = y_test_probas[i][:, 1]
        roc_auc_scores.append(roc_auc_score(Y_test[:, i], y_prob))
        per_label_f1[target_name] = f1_score(Y_test[:, i], y_pred[:, i], zero_division=0)

    macro_roc_auc = np.mean(roc_auc_scores)

    # ==========================================
    # 4. TAMPILKAN HASIL
    # ==========================================
    print("\n" + "="*70)
    print("📊 HASIL EVALUASI (SVM BASELINE - Threshold 0.5)")
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
    for label, f1_val in per_label_f1.items():
        print(f"   → {label}: {f1_val:.4f}")

    print("\n📋 Classification Report:")
    class_report = classification_report(Y_test, y_pred, target_names=['Depression', 'Anxiety', 'Stress'], zero_division=0)
    print(class_report)

    # ==========================================
    # 5. MLflow LOGGING & SUMMARY (Tanpa Save File)
    # ==========================================
    total_time = tune_time + final_train_time
    
    mlflow.log_metric("tuning_best_macro_f1", study.best_value)
    mlflow.log_metric("tuning_std_dev", std_cv_f1)
    mlflow.log_metric("test_macro_f1", macro_f1)
    mlflow.log_metric("test_exact_match", exact_match)
    mlflow.log_metric("macro_roc_auc", macro_roc_auc)
    mlflow.log_param("architecture", "Binary Relevance (MultiOutputClassifier)")
    mlflow.log_param("threshold_strategy", "Default 0.5")
    
    for label, f1_val in per_label_f1.items():
        mlflow.log_metric(f"test_f1_{label}", f1_val)

    print("\n" + "="*70)
    print("🎉 SVM BASELINE SELESAI")
    print("="*70)
    print(f"Strategy     : Optuna Tuning + Default Threshold (0.5)")
    print(f"Architecture : LinearSVC + Platt Scaling (Binary Relevance)")
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Exact Match  : {exact_match:.4f}")
    print(f"ROC-AUC      : {macro_roc_auc:.4f}")
    print(f"Waktu Total  : {total_time:.2f} detik")
    print("="*70)
