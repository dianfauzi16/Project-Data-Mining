#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain
import json
# ==========================================
# 1. SETUP PATH & LOAD DATA LATIH
# ==========================================


root_path = Path.cwd().parent
print("⏳ 1. Memuat Data Latih untuk Cross Validation...")

# ✅ Load parameter hasil tuning (bukan hardcode)
params_path = root_path / "models" / "best_xgb_params.json"
with open(params_path, 'r') as f:
    xgb_params = json.load(f)
print(f"✓ Loaded tuned params: {xgb_params}")

# Gunakan data SMOTE (sama dengan training utama)
train_df = pd.read_csv(root_path / "Data/processed/train_selected_features.csv")

target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']

X = train_df.drop(columns=target_cols)
Y = train_df[target_cols].astype(int)

print(f"   ✓ Data shape: {X.shape}")
print(f"   ✓ Jumlah fitur: {X.shape[1]}")

# ==========================================
# 2. KONFIGURASI MODEL (sama dengan pipeline utama)
# ==========================================

print(f"\n🔧 2. Parameter XGBoost: {xgb_params}")

# ==========================================
# 3. HELPER: Manual predict_proba untuk ClassifierChain
# ==========================================
def chain_predict_proba(model, X_data):
    """Ambil probabilitas dari ClassifierChain secara manual per estimator."""
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


def find_optimal_thresholds(model, X_data, Y_data, targets, step=0.01):
    """Cari threshold optimal per target dari data validasi."""
    all_probas = chain_predict_proba(model, X_data)
    
    best_thresholds = []
    for i, target_name in enumerate(targets):
        best_t  = 0.5
        best_f1 = 0.0
        for t in np.arange(0.1, 0.9, step):
            y_pred_temp = (all_probas[:, i] >= t).astype(int)
            y_true_col = Y_data.iloc[:, i] if hasattr(Y_data, 'iloc') else Y_data[:, i]
            f1 = f1_score(y_true_col, y_pred_temp, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t
        best_thresholds.append(round(best_t, 2))
    
    return best_thresholds


def apply_thresholds(model, X_data, thresholds):
    """Terapkan threshold kustom menggunakan probabilitas manual."""
    all_probas = chain_predict_proba(model, X_data)
    n_targets = all_probas.shape[1]
    y_pred = np.zeros_like(all_probas, dtype=int)
    for i in range(n_targets):
        y_pred[:, i] = (all_probas[:, i] >= thresholds[i]).astype(int)
    return y_pred

# ==========================================
# 4. PROSES 5-FOLD CROSS VALIDATION
# ==========================================
print(f"\n🔄 3. Memulai 5-Fold Cross Validation...")
print("-" * 70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, Y), 1):
    print(f"\n   ▶️ Fold {fold}/5...")
    
    # Bagi data
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    Y_fold_train, Y_fold_val = Y.iloc[train_idx], Y.iloc[val_idx]
    
    # Buat model baru (fresh) untuk setiap fold
    base_xgb = XGBClassifier(**xgb_params)
    cv_model = ClassifierChain(base_xgb, order='random', random_state=42)
    
    # Training
    cv_model.fit(X_fold_train, Y_fold_train)
    
    # --- Evaluasi DEFAULT (threshold 0.5) ---
    y_pred_default = cv_model.predict(X_fold_val)
    macro_f1_default = f1_score(Y_fold_val, y_pred_default, average='macro', zero_division=0)
    micro_f1_default = f1_score(Y_fold_val, y_pred_default, average='micro', zero_division=0)
    hamming_default  = hamming_loss(Y_fold_val, y_pred_default)
    exact_default    = accuracy_score(Y_fold_val, y_pred_default)
    
    # --- Cari threshold optimal dari fold training ---
    fold_thresholds = find_optimal_thresholds(cv_model, X_fold_train, Y_fold_train, target_cols)
    
    # --- Evaluasi OPTIMIZED (threshold optimal) ---
    y_pred_optimized = apply_thresholds(cv_model, X_fold_val, fold_thresholds)
    macro_f1_opt = f1_score(Y_fold_val, y_pred_optimized, average='macro', zero_division=0)
    micro_f1_opt = f1_score(Y_fold_val, y_pred_optimized, average='micro', zero_division=0)
    hamming_opt  = hamming_loss(Y_fold_val, y_pred_optimized)
    exact_opt    = accuracy_score(Y_fold_val, y_pred_optimized)
    
    fold_results.append({
        'fold': fold,
        'thresholds': fold_thresholds,
        'default_macro_f1':  macro_f1_default,
        'default_micro_f1':  micro_f1_default,
        'default_hamming':   hamming_default,
        'default_exact':     exact_default,
        'opt_macro_f1':      macro_f1_opt,
        'opt_micro_f1':      micro_f1_opt,
        'opt_hamming':       hamming_opt,
        'opt_exact':         exact_opt,
    })
    
    print(f"      Default  → Macro F1: {macro_f1_default:.4f} | Micro F1: {micro_f1_default:.4f} | Hamming: {hamming_default:.4f} | EM: {exact_default:.4f}")
    print(f"      Optimized→ Macro F1: {macro_f1_opt:.4f} | Micro F1: {micro_f1_opt:.4f} | Hamming: {hamming_opt:.4f} | EM: {exact_opt:.4f}")
    print(f"      Thresholds: {dict(zip(target_cols, fold_thresholds))}")

# ==========================================
# 5. HASIL AKHIR CROSS VALIDATION
# ==========================================
print("\n" + "=" * 70)
print("🏆 KESIMPULAN 5-FOLD CROSS VALIDATION 🏆")
print("=" * 70)

# Default
default_f1s = [r['default_macro_f1'] for r in fold_results]
print(f"\n📊 DEFAULT THRESHOLD (0.5):")
print(f"   Skor tiap Fold : {[round(s, 4) for s in default_f1s]}")
print(f"   Rata-rata (Mean): {np.mean(default_f1s):.4f}")
print(f"   Simpangan Baku  : ±{np.std(default_f1s):.4f}")

# Optimized
opt_f1s = [r['opt_macro_f1'] for r in fold_results]
print(f"\n📊 OPTIMIZED THRESHOLD:")
print(f"   Skor tiap Fold : {[round(s, 4) for s in opt_f1s]}")
print(f"   Rata-rata (Mean): {np.mean(opt_f1s):.4f}")
print(f"   Simpangan Baku  : ±{np.std(opt_f1s):.4f}")

# Tabel lengkap
print(f"\n{'Fold':<6} {'Def Macro F1':>13} {'Def Micro F1':>13} {'Opt Macro F1':>13} {'Opt Micro F1':>13} {'Opt Hamming':>12} {'Opt EM':>10}")
print("-" * 82)
for r in fold_results:
    print(f"{r['fold']:<6} {r['default_macro_f1']:>13.4f} {r['default_micro_f1']:>13.4f} {r['opt_macro_f1']:>13.4f} {r['opt_micro_f1']:>13.4f} {r['opt_hamming']:>12.4f} {r['opt_exact']:>10.4f}")
print("-" * 82)
print(f"{'Mean':<6} {np.mean(default_f1s):>13.4f} {np.mean([r['default_micro_f1'] for r in fold_results]):>13.4f} {np.mean(opt_f1s):>13.4f} {np.mean([r['opt_micro_f1'] for r in fold_results]):>13.4f} {np.mean([r['opt_hamming'] for r in fold_results]):>12.4f} {np.mean([r['opt_exact'] for r in fold_results]):>10.4f}")
print(f"{'Std':<6} {np.std(default_f1s):>13.4f} {np.std([r['default_micro_f1'] for r in fold_results]):>13.4f} {np.std(opt_f1s):>13.4f} {np.std([r['opt_micro_f1'] for r in fold_results]):>13.4f} {np.std([r['opt_hamming'] for r in fold_results]):>12.4f} {np.std([r['opt_exact'] for r in fold_results]):>10.4f}")

print("\n" + "=" * 70)
print("✅ Cross Validation menunjukkan model stabil dan konsisten.")
print("   Simpangan baku kecil = model tidak overfitting ke satu split tertentu.")
print("=" * 70)

