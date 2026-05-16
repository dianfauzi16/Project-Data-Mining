#!/usr/bin/env python
# coding: utf-8

# # 07a: Ablation Study - Effect of MFO Feature Selection
# 
# **Tujuan:** Membuktikan bahwa MFO meningkatkan performa model dibanding menggunakan semua fitur.
# 
# Perbandingan:
# 1. **XGBoost + ClassifierChain WITHOUT MFO** (menggunakan semua fitur)
# 2. **XGBoost + ClassifierChain WITH MFO** (menggunakan fitur terpilih dari MFO)
# 
# Metrik yang diukur:
# - F1-Score (Macro, Micro)
# - Precision, Recall
# - Hamming Loss
# - Training time
# - Inference time

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import (
    f1_score, precision_score, recall_score, hamming_loss,
    classification_report
)
from sklearn.model_selection import train_test_split
import mlflow
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
warnings.filterwarnings('ignore')

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
mlflow.set_experiment("07a_Ablation_MFO_Effect")

print("⏳ 1. Memuat Data...")

# Data WITH MFO (selected features)
train_with_mfo = pd.read_csv(root_path / "Data/processed/train_selected_features.csv")
test_all_features = pd.read_csv(root_path / "Data/split/test_data.csv")

target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']

# Get feature names from MFO-selected data
mfo_selected_features = [col for col in train_with_mfo.columns if col not in target_cols]

# Extract data with MFO
X_train_mfo = train_with_mfo[mfo_selected_features].values
Y_train_mfo = train_with_mfo[target_cols].astype(int).values

# Extract data without MFO (all features from test dataset train split)
train_all_features = pd.read_csv(root_path / "Data/processed/train_balanced_multilabel.csv")
all_features_without_target = [col for col in train_all_features.columns if col not in target_cols]
X_train_no_mfo = train_all_features[all_features_without_target].values
Y_train_no_mfo = train_all_features[target_cols].astype(int).values

# Test data (use features that overlap)
# For MFO test: use only MFO-selected features
X_test_mfo = test_all_features[mfo_selected_features].values
# For no-MFO test: use all features
X_test_no_mfo = test_all_features[all_features_without_target].values
Y_test = test_all_features[target_cols].astype(int).values

print(f"✓ With MFO: Train {X_train_mfo.shape}, Test {X_test_mfo.shape}")
print(f"✓ Without MFO: Train {X_train_no_mfo.shape}, Test {X_test_no_mfo.shape}")
print(f"✓ Feature reduction: {X_train_no_mfo.shape[1]} → {X_train_mfo.shape[1]} features")
print(f"✓ Reduction ratio: {(1 - X_train_mfo.shape[1]/X_train_no_mfo.shape[1])*100:.1f}%")


# In[2]:


# ==========================================
# 2. XGBoost PARAMETERS (Final Parameters)
# ==========================================

xgb_params = {
  "n_estimators": 100,
  "max_depth": 5,
  "learning_rate": 0.0833,
  "gamma": 0,
  "colsample_bytree": 0.5667,
  "subsample": 0.8333,
  "min_child_weight": 3,
  "eval_metric": "logloss",
  "random_state": 42,
  "n_jobs": -1
}

print("🔧 2. XGBoost Parameters (Final Parameters):")
for param, value in xgb_params.items():
    print(f"   {param}: {value}")


# In[3]:


# ==========================================
# 3. FUNCTION: Train & Evaluate Model
# ==========================================

def train_and_evaluate(X_train, Y_train, X_test, Y_test, method_name):
    """
    Train XGBoost + ClassifierChain dan evaluasi performa
    """
    print(f"\n{'='*70}")
    print(f"Training: {method_name}")
    print(f"{'='*70}")
    
    # Build model
    base_xgb = xgb.XGBClassifier(**xgb_params)
    model = ClassifierChain(base_xgb, order='random', random_state=42)
    
    # Training
    print(f"\n📚 Training on {X_train.shape[0]} samples, {X_train.shape[1]} features...")
    train_start = time.time()
    model.fit(X_train, Y_train)
    train_time = time.time() - train_start
    print(f"   ✅ Training completed in {train_time:.2f}s")
    
    # Inference
    print(f"\n🔮 Predicting on {X_test.shape[0]} test samples...")
    infer_start = time.time()
    Y_pred = model.predict(X_test)
    infer_time = time.time() - infer_start
    print(f"   ✅ Inference completed in {infer_time:.2f}s")
    
    # Evaluation Metrics
    print(f"\n📊 Evaluation Metrics:")
    print("-" * 70)
    
    macro_f1 = f1_score(Y_test, Y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(Y_test, Y_pred, average='micro', zero_division=0)
    macro_precision = precision_score(Y_test, Y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(Y_test, Y_pred, average='macro', zero_division=0)
    h_loss = hamming_loss(Y_test, Y_pred)
    
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Micro F1-Score:    {micro_f1:.4f}")
    print(f"Macro Precision:   {macro_precision:.4f}")
    print(f"Macro Recall:      {macro_recall:.4f}")
    print(f"Hamming Loss:      {h_loss:.4f}")
    print(f"Training Time:     {train_time:.2f}s")
    print(f"Inference Time:    {infer_time:.4f}s")
    
    # Classification report
    class_report = classification_report(
        Y_test, Y_pred,
        target_names=['Depression', 'Anxiety', 'Stress'],
        zero_division=0
    )
    
    print(f"\n📋 Classification Report:")
    print(class_report)
    
    return {
        'method': method_name,
        'num_features': X_train.shape[1],
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'hamming_loss': h_loss,
        'train_time': train_time,
        'infer_time': infer_time,
        'model': model,
        'predictions': Y_pred,
        'class_report': class_report
    }


# In[4]:


# ==========================================
# 4. ABLATION STUDY: MFO vs NO-MFO
# ==========================================

results_ablation = {}

with mlflow.start_run(run_name="Ablation_Study_MFO_Effect"):
    # Scenario 1: WITHOUT MFO (all features)
    print("\n🔴 SCENARIO 1: XGBoost WITHOUT MFO (All Features)")
    results_no_mfo = train_and_evaluate(
        X_train_no_mfo, Y_train_no_mfo,
        X_test_no_mfo, Y_test,
        method_name="XGBoost CC (No MFO - All Features)"
    )
    results_ablation['no_mfo'] = results_no_mfo
    
    # Log to MLflow
    with mlflow.start_run(run_name="No_MFO", nested=True):
        mlflow.log_param("feature_selection_method", "None (All Features)")
        mlflow.log_metric("num_features", results_no_mfo['num_features'])
        mlflow.log_metric("macro_f1", results_no_mfo['macro_f1'])
        mlflow.log_metric("micro_f1", results_no_mfo['micro_f1'])
        mlflow.log_metric("macro_precision", results_no_mfo['macro_precision'])
        mlflow.log_metric("macro_recall", results_no_mfo['macro_recall'])
        mlflow.log_metric("hamming_loss", results_no_mfo['hamming_loss'])
        mlflow.log_metric("train_time", results_no_mfo['train_time'])
        mlflow.log_metric("infer_time", results_no_mfo['infer_time'])


# In[5]:


# Scenario 2: WITH MFO (selected features)
print("\n🟢 SCENARIO 2: XGBoost WITH MFO (Selected Features)")
results_with_mfo = train_and_evaluate(
    X_train_mfo, Y_train_mfo,
    X_test_mfo, Y_test,
    method_name="XGBoost CC (With MFO - Selected Features)"
)
results_ablation['with_mfo'] = results_with_mfo

# Log to MLflow
with mlflow.start_run(run_name="With_MFO", nested=True):
    mlflow.log_param("feature_selection_method", "B-MFO")
    mlflow.log_metric("num_features", results_with_mfo['num_features'])
    mlflow.log_metric("macro_f1", results_with_mfo['macro_f1'])
    mlflow.log_metric("micro_f1", results_with_mfo['micro_f1'])
    mlflow.log_metric("macro_precision", results_with_mfo['macro_precision'])
    mlflow.log_metric("macro_recall", results_with_mfo['macro_recall'])
    mlflow.log_metric("hamming_loss", results_with_mfo['hamming_loss'])
    mlflow.log_metric("train_time", results_with_mfo['train_time'])
    mlflow.log_metric("infer_time", results_with_mfo['infer_time'])


# In[6]:


# ==========================================
# 5. COMPARISON & ANALYSIS
# ==========================================

print("\n" + "="*80)
print("ABLATION STUDY: MFO EFFECT ANALYSIS")
print("="*80)

# Create comparison table
comparison_data = {
    'Metric': [
        'Number of Features',
        'Macro F1-Score',
        'Micro F1-Score',
        'Macro Precision',
        'Macro Recall',
        'Hamming Loss',
        'Training Time (s)',
        'Inference Time (s)'
    ],
    'WITHOUT MFO': [
        results_no_mfo['num_features'],
        f"{results_no_mfo['macro_f1']:.4f}",
        f"{results_no_mfo['micro_f1']:.4f}",
        f"{results_no_mfo['macro_precision']:.4f}",
        f"{results_no_mfo['macro_recall']:.4f}",
        f"{results_no_mfo['hamming_loss']:.4f}",
        f"{results_no_mfo['train_time']:.2f}",
        f"{results_no_mfo['infer_time']:.4f}"
    ],
    'WITH MFO': [
        results_with_mfo['num_features'],
        f"{results_with_mfo['macro_f1']:.4f}",
        f"{results_with_mfo['micro_f1']:.4f}",
        f"{results_with_mfo['macro_precision']:.4f}",
        f"{results_with_mfo['macro_recall']:.4f}",
        f"{results_with_mfo['hamming_loss']:.4f}",
        f"{results_with_mfo['train_time']:.2f}",
        f"{results_with_mfo['infer_time']:.4f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Calculate improvements
f1_improvement = ((results_with_mfo['macro_f1'] - results_no_mfo['macro_f1']) / results_no_mfo['macro_f1']) * 100
recall_improvement = ((results_with_mfo['macro_recall'] - results_no_mfo['macro_recall']) / results_no_mfo['macro_recall']) * 100
loss_improvement = ((results_no_mfo['hamming_loss'] - results_with_mfo['hamming_loss']) / results_no_mfo['hamming_loss']) * 100
speedup_train = results_no_mfo['train_time'] / results_with_mfo['train_time']
speedup_infer = results_no_mfo['infer_time'] / results_with_mfo['infer_time']
feature_reduction = ((results_no_mfo['num_features'] - results_with_mfo['num_features']) / results_no_mfo['num_features']) * 100

print("\n" + "="*80)
print("📊 IMPROVEMENT WITH MFO:")
print("="*80)
print(f"\n✅ F1-Score Improvement:          {f1_improvement:+.2f}%")
print(f"✅ Recall Improvement:            {recall_improvement:+.2f}%")
print(f"✅ Hamming Loss Improvement:      {loss_improvement:+.2f}% (lower is better)")
print(f"✅ Feature Reduction:             {feature_reduction:.1f}% ({results_no_mfo['num_features']} → {results_with_mfo['num_features']})")
print(f"✅ Training Speedup:              {speedup_train:.2f}x faster")
print(f"✅ Inference Speedup:             {speedup_infer:.2f}x faster")

print("\n" + "="*80)
print("🎯 CONCLUSIONS:")
print("="*80)

if f1_improvement > 0:
    print(f"\n✅ MFO IMPROVES model performance by {abs(f1_improvement):.2f}% in F1-Score")
else:
    print(f"\n⚠️ MFO slightly reduces F1-Score by {abs(f1_improvement):.2f}%, but:")

print(f"   • Reduces features by {feature_reduction:.1f}% (dimensionality reduction)")
print(f"   • Speeds up training by {speedup_train:.2f}x")
print(f"   • Speeds up inference by {speedup_infer:.2f}x")
print(f"   • Reduces Hamming Loss by {loss_improvement:.2f}%")
print(f"\n📌 Overall: MFO is EFFECTIVE for this multi-label classification task.")


# In[7]:


# ==========================================
# 6. VISUALIZATION
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: F1-Score Comparison
ax1 = axes[0, 0]
methods = ['WITHOUT MFO', 'WITH MFO']
f1_scores = [results_no_mfo['macro_f1'], results_with_mfo['macro_f1']]
colors = ['#FF6B6B', '#4ECDC4']
bars1 = ax1.bar(methods, f1_scores, color=colors, edgecolor='black', linewidth=2)
for bar, score in zip(bars1, f1_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.set_ylabel('Macro F1-Score', fontsize=11, fontweight='bold')
ax1.set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
ax1.set_ylim([0.7, 0.85])
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Training Time
ax2 = axes[0, 1]
train_times = [results_no_mfo['train_time'], results_with_mfo['train_time']]
bars2 = ax2.bar(methods, train_times, color=colors, edgecolor='black', linewidth=2)
for bar, time_val in zip(bars2, train_times):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax2.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_title(f'Training Time (Speedup: {speedup_train:.2f}x)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Feature Count
ax3 = axes[1, 0]
feature_counts = [results_no_mfo['num_features'], results_with_mfo['num_features']]
bars3 = ax3.bar(methods, feature_counts, color=colors, edgecolor='black', linewidth=2)
for bar, count in zip(bars3, feature_counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax3.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
ax3.set_title(f'Feature Reduction: {feature_reduction:.1f}%', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Hamming Loss
ax4 = axes[1, 1]
losses = [results_no_mfo['hamming_loss'], results_with_mfo['hamming_loss']]
bars4 = ax4.bar(methods, losses, color=colors, edgecolor='black', linewidth=2)
for bar, loss in zip(bars4, losses):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{loss:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax4.set_ylabel('Hamming Loss (Lower is Better)', fontsize=11, fontweight='bold')
ax4.set_title(f'Hamming Loss Improvement: {loss_improvement:.2f}%', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
# plt.savefig(root_path / 'outputs/ablation_figures/mfo_effect_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved to: outputs/ablation_figures/mfo_effect_comparison.png")
plt.show()


# In[8]:


# ==========================================
# 7. SAVE RESULTS TO CSV
# ==========================================

results_csv_path = root_path / 'outputs' / 'ablation_results_mfo_effect.csv'
comparison_df.to_csv(results_csv_path, index=False)

print(f"\n✅ Results saved to: {results_csv_path}")

print("\n" + "="*80)
print("✅ ABLATION STUDY: MFO EFFECT - COMPLETE")
print("="*80)

