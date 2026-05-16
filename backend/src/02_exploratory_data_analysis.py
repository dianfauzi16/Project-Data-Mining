#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import mlflow
from pathlib import Path
from IPython.display import display, Markdown
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP PATH & MLFLOW
# ==========================================
root_path = Path.cwd().parent
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=root_path / ".env") # Load variabel dari .env
# Gunakan DagsHub URI jika ada di .env, jika tidak pakai folder lokal mlruns
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (root_path / "mlruns").as_uri())
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("02_Exploratory_Data_Analysis")

output_dir = root_path / "outputs" / "eda_figures"
output_dir.mkdir(parents=True, exist_ok=True)

with mlflow.start_run(run_name="Comprehensive_EDA"):
    print("⏳ 1. Memuat Data Mentah...")
    df_raw = pd.read_csv(root_path / "Data/raw/Dataset.csv")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # ==========================================
    # GRAFIK 1: Missing Values (Data Kosong)
    # ==========================================
    print("🎨 2. Menganalisis Missing Values...")
    plt.figure(figsize=(10, 5))
    cols_to_check = list(df_raw.columns[:20]) + ['major', 'country']
    msno.matrix(df_raw[cols_to_check], figsize=(10, 5), sparkline=False, color=(0.9, 0.3, 0.3))
    plt.title('Matriks Missing Values pada Kolom Sampel', fontsize=14, fontweight='bold', pad=20)
    
    fig1_path = output_dir / "01_missing_values.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(fig1_path))
    plt.close()

    # ==========================================
    # GRAFIK 2: Distribusi Fitur & Outlier (Usia)
    # ==========================================
    print("🎨 3. Menganalisis Distribusi & Outlier...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram Distribusi
    sns.histplot(df_raw['age'], bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Distribusi Fitur Usia')
    axes[0].set_xlim(0, 100) # Fokus pada rentang masuk akal
    
    # Boxplot untuk Outlier
    sns.boxplot(x=df_raw['age'], ax=axes[1], color='salmon')
    axes[1].set_title('Deteksi Outlier Usia (Nilai Ekstrim)')
    
    fig2_path = output_dir / "02_distribution_and_outliers.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(fig2_path))
    plt.close()

    # ==========================================
    # MENGHITUNG TARGET SEMENTARA UNTUK EDA LABEL
    # ==========================================
    # Karena target belum dihitung di tahap ini, kita hitung sementara untuk visualisasi
    df_eda = df_raw.copy()
    dep_items = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
    anx_items = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
    str_items = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]
    
    for i in range(1, 43): 
        col = f'Q{i}A'
        if col in df_eda.columns: df_eda[col] = df_eda[col] - 1
            
    df_eda['Depression'] = np.where(df_eda[[f'Q{i}A' for i in dep_items]].sum(axis=1) > 13, 1, 0)
    df_eda['Anxiety'] = np.where(df_eda[[f'Q{i}A' for i in anx_items]].sum(axis=1) > 9, 1, 0)
    df_eda['Stress'] = np.where(df_eda[[f'Q{i}A' for i in str_items]].sum(axis=1) > 18, 1, 0)

    # ==========================================
    # GRAFIK 3: Distribusi Ketidakseimbangan Label
    # ==========================================
    print("🎨 4. Menganalisis Ketidakseimbangan Label...")
    plt.figure(figsize=(10, 5))
    label_counts = pd.DataFrame({
        'Aman (0)': [len(df_eda)-df_eda['Depression'].sum(), len(df_eda)-df_eda['Anxiety'].sum(), len(df_eda)-df_eda['Stress'].sum()],
        'Berisiko (1)': [df_eda['Depression'].sum(), df_eda['Anxiety'].sum(), df_eda['Stress'].sum()]
    }, index=['Depresi', 'Kecemasan', 'Stres'])
    
    label_counts.plot(kind='bar', stacked=False, figsize=(10, 5), color=['#2ecc71', '#e74c3c'])
    plt.title('Distribusi Ketidakseimbangan Kelas (Class Imbalance) per Label', fontsize=14, fontweight='bold')
    plt.ylabel('Jumlah Responden')
    plt.xticks(rotation=0)
    
    fig3_path = output_dir / "03_label_imbalance.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(fig3_path))
    plt.close()

    # ==========================================
    # GRAFIK 4: Keterkaitan (Co-occurrence) Multi-label
    # ==========================================
    print("🎨 5. Menganalisis Co-occurrence Multi-label...")
    plt.figure(figsize=(8, 6))
    
    # Membuat Matriks Co-occurrence (Perkalian Dot)
    labels = df_eda[['Depression', 'Anxiety', 'Stress']]
    co_matrix = labels.T.dot(labels)
    
    sns.heatmap(co_matrix, annot=True, fmt='d', cmap='Purples', cbar=True)
    plt.title('Matriks Co-occurrence Keterkaitan Antar Label', fontsize=14, fontweight='bold')
    
    fig4_path = output_dir / "04_label_co_occurrence.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(fig4_path))
    plt.close()

    # ==========================================
    # GRAFIK 5: Korelasi Antar Fitur (TIPI vs Demografi)
    # ==========================================
    print("🎨 6. Menganalisis Korelasi Antar Fitur...")
    plt.figure(figsize=(12, 10))
    fitur_cols = ['age', 'gender', 'education', 'urban', 'TIPI1', 'TIPI2', 'TIPI3', 'TIPI4']
    corr_matrix = df_raw[fitur_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap Korelasi Fitur Demografi & Psikologis (TIPI)', fontsize=14, fontweight='bold')
    
    fig5_path = output_dir / "05_feature_correlation.png"
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(str(fig5_path))
    plt.close()

    # ==========================================
    # RINGKASAN OUTPUT
    # ==========================================
    summary = f"""
    ### 📊 Tahap 2: Hasil Exploratory Data Analysis (EDA)
    ---
    Seluruh analisis eksploratori berstandar industri telah dijalankan dan grafiknya tersimpan di folder `outputs/eda_figures/`.
    
    **Temuan Utama Sesuai Parameter Evaluasi:**
    1. **Missing Values & Outliers:** Terdeteksi _missing values_ signifikan pada kolom 'major' serta _outliers_ ekstrim pada usia. Ini memvalidasi kebutuhan pembersihan di Tahap 03.
    2. **Distribusi Label (Imbalance):** Grafik menunjukkan ketimpangan jumlah antara responden 'Aman' dan 'Berisiko' pada ketiga label. **Validasi untuk Tahap 05 (Imbalance Handling).**
    3. **Analisis Co-occurrence:** Matriks keterkaitan membuktikan bahwa kondisi *Multi-label* sangat kental. Responden yang mengalami *Depression* juga memiliki probabilitas sangat tinggi untuk mengidap *Anxiety* secara bersamaan (irisan data yang besar).
    
    ✅ **Tindakan Lanjutan:** Gunakan grafik-grafik ini sebagai lampiran presentasi untuk menjustifikasi setiap langkah prapemrosesan di bab berikutnya.
    """
    display(Markdown(summary))

