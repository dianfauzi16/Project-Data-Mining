#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from pathlib import Path
# from IPython.display import display, Markdown

# ==========================================
# 1. SETUP PATH & MLFLOW
# ==========================================
root_path = Path(__file__).resolve().parent.parent
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=root_path / ".env") # Load variabel dari .env
# Gunakan DagsHub URI jika ada di .env, jika tidak pakai folder lokal mlruns
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (root_path / "mlruns").as_uri())
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("04_Train_Test_Split")

with mlflow.start_run(run_name="Standard_80_20_Split"):
    print("⏳ 1. Memuat Dataset Bersih...")
    df = pd.read_csv(root_path / "Data/processed/cleaned_data.csv")
    
    # ==========================================
    # 2. PEMISAHAN FITUR (X) DAN TARGET (Y)
    # ==========================================
    # Fitur adalah semua kolom kecuali 3 target risiko
    target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']
    X = df.drop(columns=target_cols)
    Y = df[target_cols]
    
    # ==========================================
    # 3. TRAIN-TEST SPLIT (80/20)
    # ==========================================
    print("⏳ 2. Membagi Data (80% Train, 20% Test)...")
    # TAMBAHAN: Buat stratify key dari kombinasi ketiga label
    stratify_key = Y.astype(str).apply(lambda row: '_'.join(row), axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=stratify_key  # <-- tambah stratify
    )
    
    # ==========================================
    # 4. PENYIMPANAN DATA TERPISAH
    # ==========================================
    # Menggabungkan kembali X dan Y untuk disimpan dalam file terpisah
    train_df = pd.concat([X_train, Y_train], axis=1)
    test_df = pd.concat([X_test, Y_test], axis=1)
    
    output_dir = root_path / "Data/split"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train_data.csv", index=False)
    test_df.to_csv(output_dir / "test_data.csv", index=False)
    
    # ==========================================
    # 5. LOGGING KE MLFLOW
    # ==========================================
    mlflow.log_param("split_ratio", "80:20")
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("train_samples", len(X_train))
    mlflow.log_metric("test_samples", len(X_test))
    
    summary = f"""
    ### ✂️ Tahap 04: Train-Test Split Selesai
    ---
    Pemisahan data dilakukan **sebelum** penanganan *imbalance* untuk menjamin validitas evaluasi.
    
    **Statistik Hasil Pemisahan:**
    * **Data Training (80%):** **{len(X_train):,}** baris (Akan diproses SMOTE).
    * **Data Testing (20%):** **{len(X_test):,}** baris (Akan digunakan murni).
    
    ✅ *Data tersimpan di folder `Data/split/`. Siap dilanjutkan ke penanganan imbalance.*
    """
    print(summary)

