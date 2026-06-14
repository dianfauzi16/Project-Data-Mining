#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import mlflow
from pathlib import Path
# 1. SETUP PATH & MLFLOW
root_path = Path(__file__).resolve().parent.parent
# Menentukan lokasi penyimpanan log mlflow
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=root_path / ".env") # Load variabel dari .env
# Gunakan DagsHub URI jika ada di .env, jika tidak pakai folder lokal mlruns
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (root_path / "mlruns").as_uri())
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("01_Data_Acquisition")

# Load Data Mentah
df_raw = pd.read_csv(root_path / "Data/raw/Dataset.csv")

# 2. MULAI TRACKING DENGAN MLFLOW
with mlflow.start_run(run_name="Initial_Data_Profiling"):
    print("🔍 Memulai Inspeksi Data Awal (Data Profiling)...")

    # --- PERHITUNGAN METRIK ---
    rows, cols = df_raw.shape
    missing_values = df_raw.isnull().sum()
    total_missing = int(missing_values.sum())
    kolom_bermasalah = int(missing_values[missing_values > 0].count())
    total_duplicates = int(df_raw.duplicated().sum())
    umur_min = int(df_raw['age'].min())
    umur_max = int(df_raw['age'].max())

    # --- LOGGING KE MLFLOW ---
    # Log Parameter (Data statis/konfigurasi)
    mlflow.log_param("dataset_name", "DASS_Dataset_Raw")
    mlflow.log_param("total_columns", cols)
    
    # Log Metrik (Angka hasil perhitungan)
    mlflow.log_metric("total_rows", rows)
    mlflow.log_metric("missing_values_count", total_missing)
    mlflow.log_metric("duplicate_rows_count", total_duplicates)
    mlflow.log_metric("age_min", umur_min)
    mlflow.log_metric("age_max", umur_max)

    # Log Artifact 
    sample_path = "raw_data_sample.csv"
    df_raw.head(10).to_csv(sample_path, index=False)
    mlflow.log_artifact(sample_path)

    # --- TAMPILAN UI  ---
    profiling_msg = f"""
### 🩺 Hasil Inspeksi Data Awal (Data Profiling)
---

| Jenis Inspeksi | Hasil Temuan | 
| :--- | :--- |
| **Bentuk Data (Overview)** | {rows:,} Baris, {cols} Kolom | 
| **Data Kosong (Missing Values)** | Ditemukan **{total_missing:,}** data pada {kolom_bermasalah} kolom |
| **Data Ganda (Duplikat)** | Ditemukan **{total_duplicates}** baris duplikat | 
| **Anomali Umur (Outliers)** | Umur terendah: **{umur_min}**, tertinggi: **{umur_max}** |
| **Tipe Data** | Mayoritas *Integer* (Numerik) | 

"""
    print(profiling_msg)

    print("\n👁️ Cuplikan 5 Baris Pertama Data Mentah:")
    print(df_raw.head())

print(f"\n✅ Tracking selesai.")
