#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain  # ClassifierChain untuk menangkap dependensi antar-label
import mlflow
import optuna
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 0. SETUP MLflow
# ==========================================
root_path = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=root_path / ".env")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (root_path / "mlruns").as_uri())
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("08a_XGB_ClassifierChain_Optuna_Tuning")  # Experiment khusus CC

with mlflow.start_run(run_name="XGBoost_CC_Optuna_StratifiedCV"):
    # ==========================================
    # 1. LOAD DATA TUNING
    # ==========================================
    print("⏳ 1. Memuat Data untuk Tuning (Classifier Chains)...")
    target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']
    
    train_tune_smote = pd.read_csv(root_path / "Data/processed/train_smote.csv")
    selected_features = [col for col in train_tune_smote.columns if col not in target_cols]
    
    X_train_tune_smote = train_tune_smote[selected_features]
    Y_train_tune_smote = train_tune_smote[target_cols].astype(int)
    
    print(f"✓ Fitur: {len(selected_features)} | Data Tune SMOTE: {X_train_tune_smote.shape}")
    print("⚠️  Arsitektur: Classifier Chains (Label Dependence Assumption)")
    print("⚠️  PERINGATAN: Tuning CC bersifat sekuensial, waktu komputasi lebih lama dari BR")

    # ==========================================
    # 2. HYPERPARAMETER TUNING DENGAN OPTUNA (CLASSIFIER CHAINS)
    # ==========================================
    print("\n⏳ 2. Memulai Hyperparameter Tuning XGBoost (Classifier Chains) dengan Optuna...")
    print("   💡 Tips: CC lebih lambat dari BR karena label dilatih secara berantai (sekuensial)")
    tune_start = time.time()
    
    # Definisi Cross-Validation & Scorer
    mlkf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='macro', zero_division=0)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 5.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1
        }
        
        # KUNCI: Gunakan ClassifierChain untuk menangkap dependensi antar-label
        # order='random' dengan random_state yang fixed untuk reproducibility
        model = ClassifierChain(
            XGBClassifier(**params), 
            order='random', 
            random_state=42
        )
        
        # cross_validate dengan n_jobs=1 untuk menghindari konflik multithreading
        # (CC bersifat sekuensial di dalam, jadi parallel hanya di level fold)
        cv_results = cross_validate(model, X_train_tune_smote, Y_train_tune_smote,
                                    cv=mlkf, scoring=scorer, n_jobs=1)
        
        return cv_results['test_score'].mean()

    # Buat study Optuna
    study = optuna.create_study(direction='maximize', study_name="XGBoost_CC_Optuna")
    study.optimize(objective, n_trials=30)  # 30 trials untuk efisiensi (CC lebih lambat dari BR)
    
    # Ekstrak parameter terbaik
    best_params_raw = study.best_params
    xgb_params = best_params_raw.copy()
    xgb_params['random_state'] = 42
    xgb_params['eval_metric'] = 'logloss'
    xgb_params['n_jobs'] = -1
    
    tune_time = time.time() - tune_start
    print(f"\n   🏆 Best Macro F1 (Optuna CC) : {study.best_value:.4f}")
    print(f"   ⏱️ Waktu Tuning                : {tune_time:.2f} detik")
    
    # SAVE KE FILE JSON KHUSUS CC (untuk membedakan dengan BR)
    models_dir = root_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    params_path = models_dir / "best_xgb_params_cc.json" 
    with open(params_path, 'w') as f:
        json.dump(xgb_params, f, indent=2)
    print(f"   ✅ Best params saved to        : {params_path}")
    
    # ==========================================
    # 3. EKSTRAK STABILITAS & LOGGING
    # ==========================================
    print("\n📊 Analisis Stabilitas Model (Evaluasi Ulang dengan Best Params)...")
    
    best_model = ClassifierChain(
        XGBClassifier(**xgb_params), 
        order='random', 
        random_state=42
    )
    best_cv_results = cross_validate(best_model, X_train_tune_smote, Y_train_tune_smote,
                                     cv=mlkf, scoring=scorer, n_jobs=1)
    
    mean_cv_f1 = best_cv_results['test_score'].mean()
    std_cv_f1 = best_cv_results['test_score'].std()

    print(f"   → Rata-rata Macro F1 (dari 5 Fold Stratified): {mean_cv_f1:.4f}")
    print(f"   → Standar Deviasi (Stabilitas)               : ± {std_cv_f1:.4f}")

    if std_cv_f1 < 0.02:
        print("   → ✅ Model SANGAT STABIL (Variansi rendah antar fold)")
    elif std_cv_f1 < 0.05:
        print("   → ⚠️ Model CUKUP STABIL (Variansi moderat)")
    else:
        print("   → ❌ Model KURANG STABIL (Variansi tinggi)")

    # ==========================================
    # 4. MLflow LOGGING
    # ==========================================
    mlflow.log_metric("tuning_best_macro_f1", study.best_value)
    mlflow.log_metric("tuning_std_dev", std_cv_f1)
    mlflow.log_metric("tuning_time_seconds", tune_time)
    mlflow.log_param("optimizer", "Optuna (Bayesian)")
    mlflow.log_param("architecture", "Classifier Chains (Label Dependence)")
    mlflow.log_param("n_trials", 30)
    mlflow.log_param("chain_order", "random")
    
    for k, v in xgb_params.items():
        if isinstance(v, (int, float, str)):
            mlflow.log_param(f"xgb_{k}", v)
            
    mlflow.log_artifact(str(params_path))
    
    print("\n" + "="*70)
    print("✅ OPTUNA TUNING CLASSIFIER CHAINS SELESAI!")
    print("="*70)
    print(f"Parameter tersimpan di: {params_path}")
    print(f"Best Macro F1: {study.best_value:.4f} ± {std_cv_f1:.4f}")
    print(f"📌 Langkah berikutnya: Gunakan parameter ini untuk training final & evaluasi CC")
    print("="*70)
