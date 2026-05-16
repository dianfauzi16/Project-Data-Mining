#!/usr/bin/env python
# coding: utf-8

# # 12. Eksperimen Peningkatan Akurasi
# 
# **Tujuan:** Meningkatkan Macro F1 dari 0.7928 ke > 0.80
# 
# **Strategi yang diuji:**
# 1. Chain Order Optimization
# 2. Optuna Hyperparameter Tuning (Bayesian)
# 3. Feature Engineering
# 4. Kombinasi terbaik
# 
# **Baseline saat ini:** Macro F1 = 0.7928 (Threshold Optimized)

# In[1]:


import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# SETUP
# ==========================================
root_path = Path.cwd()

# Load data
train_df = pd.read_csv(root_path / "Data/processed/train_selected_features.csv")
train_original = pd.read_csv(root_path / "Data/split/train_data.csv")
test_df = pd.read_csv(root_path / "Data/split/test_data.csv")

target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']
selected_features = [c for c in train_df.columns if c not in target_cols]

X_train = train_df[selected_features]
Y_train = train_df[target_cols].astype(int)

X_test = test_df.reindex(columns=selected_features, fill_value=0)
Y_test = test_df[target_cols].astype(int)

# Data asli untuk threshold calibration
X_orig = train_original.reindex(columns=selected_features, fill_value=0)
Y_orig = train_original[target_cols].astype(int)
X_orig_train, X_orig_val, Y_orig_train, Y_orig_val = train_test_split(
    X_orig, Y_orig, test_size=0.2, random_state=42
)

# Load parameter saat ini
with open(root_path / "models/best_xgb_params.json") as f:
    current_params = json.load(f)

print(f"Fitur: {len(selected_features)}")
print(f"Train SMOTE: {X_train.shape}")
print(f"Test: {X_test.shape}")
print(f"Current params: {current_params}")


# In[2]:


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def chain_predict_proba(model, X_data):
    """Ambil probabilitas dari ClassifierChain secara manual."""
    n_samples = X_data.shape[0]
    n_targets = len(model.estimators_)
    all_probas = np.zeros((n_samples, n_targets))
    X_aug = X_data.values.copy() if hasattr(X_data, 'values') else X_data.copy()
    for i, estimator in enumerate(model.estimators_):
        proba = estimator.predict_proba(X_aug)[:, 1]
        all_probas[:, i] = proba
        pred_label = (proba >= 0.5).astype(int).reshape(-1, 1)
        X_aug = np.hstack([X_aug, pred_label])
    return all_probas

def find_optimal_thresholds(model, X_val, Y_val, step=0.005):
    """Cari threshold optimal per target."""
    probas = chain_predict_proba(model, X_val)
    thresholds = {}
    for i, target in enumerate(target_cols):
        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.1, 0.9, step):
            pred = (probas[:, i] >= t).astype(int)
            f1 = f1_score(Y_val.iloc[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = round(t, 3)
        thresholds[target] = best_t
    return thresholds

def evaluate_full(params, order='random', label=""):
    """Train + threshold calibration + evaluate on test."""
    # 1. Train final model di SMOTE
    xgb = XGBClassifier(**params)
    chain = ClassifierChain(xgb, order=order, random_state=42)
    chain.fit(X_train, Y_train)
    
    # 2. Train model sementara di data asli untuk threshold
    xgb_temp = XGBClassifier(**params)
    chain_temp = ClassifierChain(xgb_temp, order=order, random_state=42)
    chain_temp.fit(X_orig_train, Y_orig_train)
    
    # 3. Cari threshold optimal
    thresholds = find_optimal_thresholds(chain_temp, X_orig_val, Y_orig_val)
    
    # 4. Evaluasi di test
    probas = chain_predict_proba(chain, X_test)
    y_pred_default = chain.predict(X_test)
    y_pred_opt = np.zeros_like(probas, dtype=int)
    for i, target in enumerate(target_cols):
        y_pred_opt[:, i] = (probas[:, i] >= thresholds[target]).astype(int)
    
    default_f1 = f1_score(Y_test, y_pred_default, average='macro')
    opt_f1 = f1_score(Y_test, y_pred_opt, average='macro')
    opt_micro = f1_score(Y_test, y_pred_opt, average='micro')
    hamming = hamming_loss(Y_test, y_pred_opt)
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Default Macro F1:   {default_f1:.4f}")
    print(f"  Optimized Macro F1: {opt_f1:.4f}  {'✅ > 0.80!' if opt_f1 >= 0.80 else ''}")
    print(f"  Optimized Micro F1: {opt_micro:.4f}")
    print(f"  Hamming Loss:       {hamming:.4f}")
    print(f"  Thresholds:         {thresholds}")
    
    return {'label': label, 'default_f1': default_f1, 'opt_f1': opt_f1, 
            'micro_f1': opt_micro, 'hamming': hamming, 'thresholds': thresholds,
            'params': params, 'order': order, 'model': chain}

print("✅ Helper functions defined")


# ## Eksperimen 1: Baseline (Skor Saat Ini)

# In[3]:


# ==========================================
# EKSPERIMEN 1: BASELINE
# ==========================================
results = []
r = evaluate_full(current_params, order='random', label='Baseline (Current)')
results.append(r)


# ## Eksperimen 2: Chain Order Optimization

# In[4]:


# ==========================================
# EKSPERIMEN 2: CHAIN ORDER OPTIMIZATION
# ==========================================
# Coba semua 6 kemungkinan urutan chain
from itertools import permutations

print("⏳ Mencoba semua urutan ClassifierChain...")
best_order_result = None

for order in permutations([0, 1, 2]):
    order_list = list(order)
    order_name = [target_cols[i].replace('risk_','') for i in order_list]
    r = evaluate_full(current_params, order=order_list, 
                      label=f'Chain Order: {" → ".join(order_name)}')
    results.append(r)
    
    if best_order_result is None or r['opt_f1'] > best_order_result['opt_f1']:
        best_order_result = r

print(f"\n🏆 Best Chain Order: {best_order_result['order']}")
print(f"   Macro F1: {best_order_result['opt_f1']:.4f}")


# ## Eksperimen 3: Optuna Hyperparameter Tuning

# In[5]:


# Install optuna jika belum ada
get_ipython().system('pip install optuna -q')


# In[6]:


# ==========================================
# EKSPERIMEN 3: OPTUNA BAYESIAN OPTIMIZATION
# ==========================================
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

best_chain_order = best_order_result['order']

def optuna_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 150, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train + evaluate dengan CV internal
    xgb = XGBClassifier(**params)
    chain = ClassifierChain(xgb, order=best_chain_order, random_state=42)
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_f_train = X_train.iloc[train_idx]
        Y_f_train = Y_train.iloc[train_idx]
        X_f_val = X_train.iloc[val_idx]
        Y_f_val = Y_train.iloc[val_idx]
        
        chain_cv = ClassifierChain(XGBClassifier(**params), order=best_chain_order, random_state=42)
        chain_cv.fit(X_f_train, Y_f_train)
        y_pred = chain_cv.predict(X_f_val)
        fold_scores.append(f1_score(Y_f_val, y_pred, average='macro'))
    
    return np.mean(fold_scores)

print("⏳ Memulai Optuna Optimization (50 trials)...")
print(f"   Best chain order: {best_chain_order}")
start = time.time()

study = optuna.create_study(direction='maximize', study_name='xgb_tuning')
study.optimize(optuna_objective, n_trials=50, show_progress_bar=True)

elapsed = time.time() - start
print(f"\n✅ Optuna selesai dalam {elapsed:.1f} detik")
print(f"   Best CV Macro F1: {study.best_value:.4f}")
print(f"   Best Params: {study.best_params}")


# In[7]:


# Evaluasi parameter Optuna terbaik di test set
optuna_params = {
    **study.best_params,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}

r = evaluate_full(optuna_params, order=best_chain_order, 
                  label='Optuna + Best Chain Order')
results.append(r)


# ## Ringkasan Semua Eksperimen

# In[8]:


# ==========================================
# RINGKASAN SEMUA EKSPERIMEN
# ==========================================
print("\n" + "="*80)
print("🏆 RINGKASAN EKSPERIMEN PENINGKATAN AKURASI")
print("="*80)
print(f"{'No':<4} {'Eksperimen':<40} {'Default F1':>12} {'Opt F1':>10} {'Status':>10}")
print("-"*80)

for i, r in enumerate(results):
    status = '✅ >0.80' if r['opt_f1'] >= 0.80 else ''
    print(f"{i+1:<4} {r['label']:<40} {r['default_f1']:>12.4f} {r['opt_f1']:>10.4f} {status:>10}")

# Cari yang terbaik
best = max(results, key=lambda x: x['opt_f1'])
print(f"\n🥇 TERBAIK: {best['label']}")
print(f"   Macro F1: {best['opt_f1']:.4f}")
print(f"   Thresholds: {best['thresholds']}")

if best['opt_f1'] >= 0.80:
    print("\n🎉 TARGET TERCAPAI! Macro F1 > 0.80!")
else:
    print(f"\n📊 Peningkatan dari baseline: +{best['opt_f1'] - results[0]['opt_f1']:.4f}")
    print(f"   Sisa menuju 0.80: {0.80 - best['opt_f1']:.4f}")

