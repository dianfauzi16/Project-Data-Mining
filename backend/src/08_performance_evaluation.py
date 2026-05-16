#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, hamming_loss
from xgboost import XGBClassifier
from sklearn.multioutput import ClassifierChain
import mlflow
from pathlib import Path
from IPython.display import display, Markdown
import json

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
mlflow.set_experiment("08_Performance_Evaluation")

# ✅ Load parameter hasil tuning (bukan hardcode)
params_path = root_path / "models" / "best_xgb_params.json"
with open(params_path, 'r') as f:
    xgb_params = json.load(f)
print(f"✓ Loaded tuned params: {xgb_params}")

print("⏳ 1. Memuat Data dan Model...")

# Load data training (fitur sudah diseleksi MFO)
train_df = pd.read_csv(root_path / "Data/processed/train_selected_features.csv")
test_df  = pd.read_csv(root_path / "Data/split/test_data.csv")

target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']

# ✅ DINAMIS: Fitur otomatis dari file MFO output
selected_features = [col for col in train_df.columns if col not in target_cols]

X_train = train_df[selected_features]
Y_train = train_df[target_cols].astype(int)

# ✅ Samakan kolom test dengan train (reindex)
X_test = test_df.reindex(columns=selected_features, fill_value=0)
Y_test = test_df[target_cols].astype(int)

print(f"   ✓ Fitur MFO: {len(selected_features)} fitur")
print(f"   ✓ Train: {X_train.shape}, Test: {X_test.shape}")


base_xgb = XGBClassifier(**xgb_params)
model = ClassifierChain(base_xgb, order='random', random_state=42)
model.fit(X_train, Y_train)
print("   ✓ Model fresh dilatih")

with mlflow.start_run(run_name="Evaluate_Unseen_Test_Data"):
    # ==========================================
    # 2. PREDIKSI DATA UJIAN (Default threshold 0.5)
    # ==========================================
    print("⏳ 2. Memprediksi 3 Kondisi Mental secara bersamaan...")
    Y_pred = model.predict(X_test)
    
    # ==========================================
    # 3. MENGHITUNG METRIK MULTI-LABEL
    # ==========================================
    print("⏳ 3. Menghitung Metrik Evaluasi Khusus Multi-label...")
    
    exact_match = accuracy_score(Y_test, Y_pred)
    h_loss      = hamming_loss(Y_test, Y_pred)
    macro_f1    = f1_score(Y_test, Y_pred, average='macro')
    micro_f1    = f1_score(Y_test, Y_pred, average='micro')
    
    class_report_str = classification_report(
        Y_test, Y_pred, 
        target_names=['Depression', 'Anxiety', 'Stress']
    )
    
    # ==========================================
    # 4. LOGGING MLFLOW & OUTPUT
    # ==========================================
    mlflow.log_metric("test_exact_match",  exact_match)
    mlflow.log_metric("test_hamming_loss", h_loss)
    mlflow.log_metric("test_macro_f1",     macro_f1)
    mlflow.log_metric("test_micro_f1",     micro_f1)
    mlflow.log_param("num_features",       len(selected_features))
    mlflow.log_param("model_source",       "Fresh Training")
    mlflow.log_text(class_report_str, "classification_report.txt")
    
    summary = f"""
    ### 🎯 Tahap 08: Hasil Evaluasi Akhir (Unseen Test Data, Default Threshold 0.5)
    ---
    Model telah diuji menggunakan data testing murni sebanyak **{len(X_test):,} responden remaja** yang belum pernah dilihat sebelumnya.
    
    **Fitur yang digunakan:** {len(selected_features)} fitur (hasil seleksi MFO)
    
    **Metrik Evaluasi Multi-label Utama:**
    * **Macro F1-Score:** **{macro_f1:.4f}** (Rata-rata performa model pada ketiga kondisi mental).
    * **Micro F1-Score:** **{micro_f1:.4f}** (Performa keseluruhan berdasarkan agregat total prediksi benar).
    * **Hamming Loss:** **{h_loss:.4f}** (Tingkat kesalahan label hanya {h_loss*100:.2f}%).
    * **Exact Match Ratio:** **{exact_match:.4f}** (Persentase AI menebak kombinasi 3 label 100% sempurna).
    
    **Catatan:** Ini adalah evaluasi dengan threshold default (0.5).  
    Lihat NB 11 untuk evaluasi dengan threshold yang dioptimasi.
    
    **Laporan Klasifikasi Detail per Kondisi Mental:**
    ```text
    {class_report_str}
    ```
    ✅ *Evaluasi selesai. Metrik telah tersimpan di MLflow.*
    """
    display(Markdown(summary))
    print(class_report_str)

