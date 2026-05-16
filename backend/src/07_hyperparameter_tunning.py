#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, make_scorer
import mlflow
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP PATH & LOAD DATA
# ==========================================
root_path = Path.cwd()
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=root_path / ".env") # Load variabel dari .env
# Gunakan DagsHub URI jika ada di .env, jika tidak pakai folder lokal mlruns
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (root_path / "mlruns").as_uri())
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("07b_Hyperparameter_Tuning")

print("⏳ 1. Memuat Data...")

# ✅ Load fitur MFO (dinamis)
train_df = pd.read_csv(root_path / "Data/processed/train_selected_features.csv")
test_df  = pd.read_csv(root_path / "Data/split/test_data.csv")

target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']
selected_features = [col for col in train_df.columns if col not in target_cols]

X_train = train_df[selected_features]
Y_train = train_df[target_cols].astype(int)

X_test = test_df.reindex(columns=selected_features, fill_value=0)
Y_test = test_df[target_cols].astype(int)

print(f"   ✓ Fitur MFO: {len(selected_features)} fitur")
print(f"   ✓ Train: {X_train.shape}, Test: {X_test.shape}")

# ==========================================
# 2. HYPERPARAMETER SEARCH SPACE
# ==========================================
param_distributions = {
    'n_estimators':     [100, 200, 300, 400],
    'max_depth':        [3, 4, 5, 6, 7],
    'learning_rate':    [0.01, 0.03, 0.05, 0.1],
    'gamma':            [0, 0.1, 0.2, 0.3],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
    'subsample':        [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 2, 3, 5],
}

print(f"\n🔧 2. Hyperparameter Search Space:")
for k, v in param_distributions.items():
    print(f"   {k}: {v}")

# ==========================================
# 3. TUNING PER TARGET (Multi-label aware)
# ==========================================
# Karena ClassifierChain tidak bisa langsung di-RandomizedSearchCV,
# kita tuning base XGBoost per target, lalu ambil parameter terbaik rata-rata.

print(f"\n⏳ 3. Memulai Hyperparameter Tuning (RandomizedSearchCV)...")
print(f"   n_iter=60, cv=3, scoring=f1_macro")
print("-" * 60)

all_best_params = {}
all_best_scores = {}

start_time = time.time()

for target in target_cols:
    print(f"\n   🔍 Tuning untuk: {target}...")
    
    base_xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    search = RandomizedSearchCV(
        base_xgb,
        param_distributions,
        n_iter=80,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    search.fit(X_train, Y_train[target])
    
    all_best_params[target] = search.best_params_
    all_best_scores[target] = search.best_score_
    
    print(f"      Best F1: {search.best_score_:.4f}")
    print(f"      Best Params: {search.best_params_}")

tuning_time = time.time() - start_time
print(f"\n   ✓ Tuning selesai dalam {tuning_time:.1f} detik")

# ==========================================
# 4. AGGREGATE BEST PARAMETERS
# ==========================================
# Ambil parameter yang paling sering muncul / rata-rata
from collections import Counter

print(f"\n📊 4. Aggregasi Parameter Terbaik dari 3 Target...")

# Untuk parameter integer/categorical: ambil yang paling sering (mode)
# Untuk parameter float: ambil rata-rata
aggregated_params = {}

for param_name in param_distributions.keys():
    values = [all_best_params[target][param_name] for target in target_cols]
    
    if isinstance(values[0], float):
        # Rata-rata untuk float
        avg_val = round(np.mean(values), 4)
        aggregated_params[param_name] = avg_val
    else:
        # Mode untuk integer
        counter = Counter(values)
        mode_val = counter.most_common(1)[0][0]
        aggregated_params[param_name] = mode_val
    
    print(f"   {param_name}: {values} → {aggregated_params[param_name]}")

# Tambahkan parameter tetap
aggregated_params['eval_metric'] = 'logloss'
aggregated_params['random_state'] = 42
aggregated_params['n_jobs'] = -1

print(f"\n   ✅ Final Aggregated Parameters:")
for k, v in aggregated_params.items():
    print(f"      {k}: {v}")

# ==========================================
# 5. EVALUASI DENGAN PARAMETER TUNED (ClassifierChain)
# ==========================================
print(f"\n🔨 5. Evaluasi ClassifierChain dengan Parameter Tuned...")

tuned_xgb = XGBClassifier(**aggregated_params)
tuned_chain = ClassifierChain(tuned_xgb, order='random', random_state=42)
tuned_chain.fit(X_train, Y_train)

Y_pred_tuned = tuned_chain.predict(X_test)

macro_f1 = f1_score(Y_test, Y_pred_tuned, average='macro', zero_division=0)
micro_f1 = f1_score(Y_test, Y_pred_tuned, average='micro', zero_division=0)
hamming  = hamming_loss(Y_test, Y_pred_tuned)
exact    = accuracy_score(Y_test, Y_pred_tuned)

print(f"\n" + "="*60)
print(f"🏆 HASIL HYPERPARAMETER TUNING")
print(f"="*60)
print(f"Macro F1-Score:    {macro_f1:.4f}")
print(f"Micro F1-Score:    {micro_f1:.4f}")
print(f"Hamming Loss:      {hamming:.4f}")
print(f"Exact Match:       {exact:.4f}")
print(f"="*60)

# ==========================================
# 6. SIMPAN PARAMETER TERBAIK KE FILE
# ==========================================
params_save_path = root_path / "models" / "best_xgb_params.json"
params_save_path.parent.mkdir(parents=True, exist_ok=True)

# Konversi numpy types ke Python native untuk JSON
save_params = {}
for k, v in aggregated_params.items():
    if isinstance(v, (np.integer,)):
        save_params[k] = int(v)
    elif isinstance(v, (np.floating,)):
        save_params[k] = float(v)
    else:
        save_params[k] = v

with open(params_save_path, 'w') as f:
    json.dump(save_params, f, indent=2)

print(f"\n💾 6. Parameter terbaik disimpan ke: {params_save_path}")

# ==========================================
# 7. MLFLOW LOGGING
# ==========================================
print(f"\n📈 7. Logging ke MLflow...")
with mlflow.start_run(run_name="XGBoost_HyperparameterTuning"):
    mlflow.log_param("tuning_method", "RandomizedSearchCV")
    mlflow.log_param("n_iter", 60)
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("num_features", len(selected_features))
    mlflow.log_param("tuning_strategy", "Per-target + Aggregate")
    
    for k, v in aggregated_params.items():
        mlflow.log_param(f"best_{k}", v)
    
    for target in target_cols:
        mlflow.log_metric(f"cv_f1_{target}", all_best_scores[target])
    
    mlflow.log_metric("test_macro_f1", macro_f1)
    mlflow.log_metric("test_micro_f1", micro_f1)
    mlflow.log_metric("test_hamming_loss", hamming)
    mlflow.log_metric("test_exact_match", exact)
    mlflow.log_metric("tuning_time_seconds", tuning_time)
    
    mlflow.log_artifact(str(params_save_path))
    
    print("   ✓ MLflow logging selesai!")

print(f"\n📌 CATATAN:")
print(f"   - Parameter terbaik disimpan di: models/best_xgb_params.json")
print(f"   - NB 07 (CV), NB 08, NB 11 harus load file ini untuk konsistensi")
print(f"\n✅ Hyperparameter Tuning SELESAI!")

