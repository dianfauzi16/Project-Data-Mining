#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
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
mlflow.set_experiment("03_Data_Preprocessing")

with mlflow.start_run(run_name="Data_Cleaning_and_Engineering"):
    print("⏳ 1. Memuat Data Mentah...")
    df = pd.read_csv(root_path / "Data/raw/Dataset.csv")
    
    # Simpan jumlah awal untuk tracking
    initial_count = len(df)

    # ==========================================
    # 2. DATA CLEANING & PENANGANAN OUTLIER
    # ==========================================
    print("⏳ 2. Cleaning & Penanganan Outlier Usia...")
    # Sesuai fokus penelitian: Remaja (15-24 tahun)
    # Ini secara otomatis membuang outlier umur 1998 tahun
    df = df[(df['age'] >= 15) & (df['age'] <= 24)].copy()
    
    # Menghapus kolom teknis yang tidak memiliki nilai psikososial
    # Kolom berakhiran E (waktu) dan I (posisi soal)
    tech_cols = [c for c in df.columns if c.endswith('E') or c.endswith('I')] + \
                ['introelapse', 'testelapse', 'surveyelapse', 'screensize', 
                 'uniquenetworklocation', 'source', 'hand']
    df = df.drop(columns=tech_cols, errors='ignore')
    
    # Menghapus kolom VCL1-16 (Vocabulary Checklist)
    # Kolom ini hanya digunakan untuk validasi atensi responden, bukan fitur psikososial
    vcl_cols = [f"VCL{i}" for i in range(1, 17)]
    df = df.drop(columns=vcl_cols, errors="ignore")
    
    # Menghapus fitur demografis yang tidak relevan untuk konteks penelitian di Indonesia
    # - engnat: English as native language, tidak relevan untuk responden Indonesia
    # - orientation: orientasi seksual, di luar cakupan penelitian psikososial remaja
    demo_drop_cols = ["engnat", "orientation"]
    df = df.drop(columns=demo_drop_cols, errors="ignore")
    print("   ✔ Kolom VCL1-16, engnat, orientation berhasil di-drop.")
    
    # ==========================================
    # 3. PENANGANAN MISSING VALUE
    # ==========================================
    print("⏳ 3. Penanganan Missing Values...")
    # Menghapus kolom 'major' karena memiliki >25% data kosong (berdasarkan EDA)
    df = df.drop(columns=['major'], errors='ignore')
    # Menghapus baris yang masih memiliki missing value di kolom lain
    df = df.dropna()

    # ==========================================
    # 4. FEATURE ENGINEERING (DASS-42 SCORING)
    # ==========================================
    print("⏳ 4. Feature Engineering: Menghitung Label Target...")
    
    # Kunci Soal DASS-42
    dep_items = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
    anx_items = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
    str_items = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]

    # Langkah Penting: Ubah skor 1-4 menjadi 0-3 agar sesuai rumus DASS
    q_cols = [f'Q{i}A' for i in range(1, 43)]
    for col in q_cols:
        df[col] = df[col] - 1

    # Menghitung Total Skor per kategori
    df['score_depression'] = df[[f'Q{i}A' for i in dep_items]].sum(axis=1)
    df['score_anxiety'] = df[[f'Q{i}A' for i in anx_items]].sum(axis=1)
    df['score_stress'] = df[[f'Q{i}A' for i in str_items]].sum(axis=1)

    # Threshold: Moderate ke atas (Binary Label)
    df['risk_depression'] = np.where(df['score_depression'] > 13, 1, 0)
    df['risk_anxiety'] = np.where(df['score_anxiety'] > 9, 1, 0)
    df['risk_stress'] = np.where(df['score_stress'] > 18, 1, 0)

    # ==========================================
    # 5. PENCEGAHAN DATA LEAKAGE & FINAL CLEANING
    # ==========================================
    # Menghapus 42 pertanyaan asli dan skor total agar model tidak "mencontek"
    df_final = df.drop(columns=q_cols + ['score_depression', 'score_anxiety', 'score_stress', 'country'])
    
    # Menghapus baris duplikat alami (responden berbeda dengan jawaban identik)
    # Mencegah data overlap saat train-test split
    # n_before_dedup = len(df_final)
    # df_final = df_final.drop_duplicates()
    # n_dupes = n_before_dedup - len(df_final)
    # print(f"   \u2714 {n_dupes} baris duplikat dihapus.")

    # ==========================================
    # 6. SIMPAN HASIL & LOGGING
    # ==========================================
    output_path = root_path / "Data/processed/cleaned_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False)

    mlflow.log_metric("rows_after_cleaning", len(df_final))
    mlflow.log_param("age_range", "15-24")
    mlflow.log_param("encoding_status", "Already Numeric Labels")

    summary = f"""
    ### 🛠️ Tahap 03: Data Preprocessing Selesai
    ---
    Data telah dibersihkan dan ditransformasi sesuai standar medis DASS-42.

    **Detail Perubahan:**
    * **Data Cleaning:** Menghapus variabel teknis, VCL1-16, engnat, orientation & memfilter usia (15-24 thn).
    * **Outlier:** Berhasil ditangani melalui pemfilteran usia.
    * **Missing Value:** Menghapus kolom 'major' dan baris kosong.
    * **Feature Engineering:** Menghasilkan 3 label target (*risk_depression, risk_anxiety, risk_stress*).
    * **Drop Fitur:** VCL1-16 (validasi),engnat, orientation (tidak relevan untuk konteks Indonesia).

    **Status Data:**
    * Jumlah awal: {initial_count:,} baris.
    * Jumlah setelah dibersihkan: **{len(df_final):,}** baris.
    * Total Fitur Psikososial: **{df_final.shape[1] - 3}** kolom.

    ✅ *Dataset bersih siap dilanjutkan ke Tahap 04: Train-Test Split.*
    """
    print(summary)

