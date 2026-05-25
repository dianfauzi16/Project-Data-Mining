#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import mlflow
from pathlib import Path
import time

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
mlflow.set_experiment("06_MFO_Feature_Selection")

print("⏳ 1. Memuat Data Training yang Sudah Seimbang (Balanced)...")
# Membaca data hasil tahap 05
train_df = pd.read_csv(root_path / "Data/processed/train_balanced_multilabel.csv")

target_cols = ['risk_depression', 'risk_anxiety', 'risk_stress']
X_train = train_df.drop(columns=target_cols).values  # ← Convert ke numpy array
Y_train = train_df[target_cols].astype(int).values   # ← Pastikan integer

feature_names = train_df.drop(columns=target_cols).columns
num_features = X_train.shape[1]

print(f"✓ Jumlah fitur awal: {num_features} fitur")
print(f"✓ Data Train shape: {X_train.shape}")

# ==========================================
# 2. INTERNAL TRAIN-VAL SPLIT UNTUK FITNESS
#    (agar evaluasi fitness jujur tanpa overfitting)
# ==========================================
print("\n🔀 Split data internal untuk fitness evaluation (80-20)...")
X_fit_train, X_fit_val, Y_fit_train, Y_fit_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42
)
print(f"   ✓ Fitness Train : {X_fit_train.shape}")
print(f"   ✓ Fitness Val   : {X_fit_val.shape}")

# ==========================================
# 3. STANDARDIZED XGBOOST PARAMETERS 
# ==========================================
xgb_params_fitness = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'random_state': 42,
    'eval_metric': 'logloss',
    'n_jobs': -1
}

xgb_params_final = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 3,
    'gamma': 0.2,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'eval_metric': 'logloss',
    'n_jobs': -1
}

print("\n🔧 2. XGBoost Parameters untuk Standardized Benchmarking:")
print(f"   Fitness Phase: n_est={xgb_params_fitness['n_estimators']}, lr={xgb_params_fitness['learning_rate']}")
print(f"   Final Training: n_est={xgb_params_final['n_estimators']}, lr={xgb_params_final['learning_rate']}")

# ==========================================
# 4. DEFINISI ALGORITMA MFO BINER
# ==========================================
class BinaryMFO:
    def __init__(self, num_moths, max_iterations, num_features, alpha=0.01, transfer_type='V-shaped', fitness_mode='full'):
        self.num_moths = num_moths
        self.max_iterations = max_iterations
        self.num_features = num_features
        self.alpha = alpha  # Penalty untuk feature count
        self.transfer_type = transfer_type
        self.fitness_mode = fitness_mode  # 'val' (80-20 split) or 'full' (seluruh SMOTE train)
        
        # Inisialisasi posisi ngengat secara acak (0 atau 1)
        self.moths = np.random.randint(2, size=(num_moths, num_features))
        self.flames = np.zeros((num_moths, num_features))
        self.moth_fitness = np.zeros(num_moths)
        self.flame_fitness = np.full(num_moths, np.inf)  # Infinity karena minimize cost
        
        # Base model XGBoost untuk Multi-label dengan parameter standardized
        base_xgb = xgb.XGBClassifier(**xgb_params_fitness)
        self.model = ClassifierChain(base_xgb, order='random', random_state=42)

    def evaluate_fitness(self, moth_position):
        """
        Hitung fitness.
        - Jika fitness_mode == 'val': Train di X_fit_train (80%) & Evaluate di X_fit_val (20%)
        - Jika fitness_mode == 'full': Train di seluruh X_train & Evaluate di X_train (seperti GA)
        
        Cost = (1 - f1_score) + alpha * (feature_count / total_features)
        Lebih rendah = lebih baik (minimize)
        """
        selected_features_mask = moth_position == 1
        
        # Jika tidak ada fitur yang dipilih, berikan penalti tinggi
        if np.count_nonzero(selected_features_mask) == 0:
            return 1.0
        
        if self.fitness_mode == 'val':
            X_sub_train = X_fit_train[:, selected_features_mask]
            X_sub_eval   = X_fit_val[:, selected_features_mask]
            Y_sub_train = Y_fit_train
            Y_sub_eval   = Y_fit_val
        else: # 'full'
            X_sub_train = X_train[:, selected_features_mask]
            X_sub_eval   = X_train[:, selected_features_mask]
            Y_sub_train = Y_train
            Y_sub_eval   = Y_train
        
        try:
            self.model.fit(X_sub_train, Y_sub_train)
            preds = self.model.predict(X_sub_eval)
            f1 = f1_score(Y_sub_eval, preds, average='macro', zero_division=0)
        except Exception as e:
            print(f"   ⚠️ Error dalam fitness evaluation: {e}")
            return 1.0
        
        # Cost = (1 - F1) + penalty untuk jumlah fitur
        num_selected = np.count_nonzero(selected_features_mask)
        cost = (1.0 - f1) + (self.alpha * (num_selected / self.num_features))
        
        return cost

    def optimize(self):
        print(f"\n🚀 Memulai MFO dengan {self.num_moths} Ngengat selama {self.max_iterations} Iterasi...")
        print(f"   Alpha (feature penalty): {self.alpha}")
        print(f"   Evaluasi: Internal Train-Val Split (80-20)")
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            print(f"--- Iterasi {iteration + 1}/{self.max_iterations} ---")
            
            # Hitung fitness untuk setiap ngengat
            for i in range(self.num_moths):
                self.moth_fitness[i] = self.evaluate_fitness(self.moths[i])
            
            # Mengurutkan ngengat berdasarkan fitness terbaik (ascending karena minimize)
            sorted_indices = np.argsort(self.moth_fitness)
            self.moths = self.moths[sorted_indices]
            self.moth_fitness = self.moth_fitness[sorted_indices]
            
            # Update Flames (Api) - Pada iterasi pertama, ngengat terbaik menjadi api
            if iteration == 0:
                self.flames = np.copy(self.moths)
                self.flame_fitness = np.copy(self.moth_fitness)
            else:
                # Menggabungkan populasi api lama dan ngengat baru, lalu ambil yang terbaik
                combined_population = np.vstack((self.flames, self.moths))
                combined_fitness = np.concatenate((self.flame_fitness, self.moth_fitness))
                
                best_indices = np.argsort(combined_fitness)[:self.num_moths]
                self.flames = combined_population[best_indices]
                self.flame_fitness = combined_fitness[best_indices]
            
            # Adaptive Flame Number: Jumlah api (flames) berkurang secara linear
            num_flames = int(np.round(self.num_moths - iteration * ((self.num_moths - 1) / self.max_iterations)))
            num_flames = max(1, num_flames)
            
            # Update Posisi Ngengat berdasarkan persamaan Logarithmic Spiral
            a = -1 + iteration * ((-1) / self.max_iterations)
            
            for i in range(self.num_moths):
                # Ngengat hanya tertarik pada active flames terbaik
                flame_no = min(i, num_flames - 1)
                
                for j in range(self.num_features):
                    distance_to_flame = abs(self.flames[flame_no, j] - self.moths[i, j])
                    b = 1
                    t = (a - 1) * np.random.rand() + 1
                    
                    new_position_continuous = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + self.flames[flame_no, j]
                    
                    if self.transfer_type == 'V-shaped':
                        # V-shaped Transfer Function (V4)
                        v_val = abs(new_position_continuous) / np.sqrt(1 + new_position_continuous**2)
                        # Binarization rule: Bit-flip based on V-value probability
                        if np.random.rand() < v_val:
                            self.moths[i, j] = 1 - self.moths[i, j]
                    else:
                        # S-shaped (Sigmoid)
                        sigmoid_val = 1 / (1 + np.exp(-new_position_continuous))
                        self.moths[i, j] = 1 if sigmoid_val > 0.5 else 0
            
            # Display progress
            best_cost = self.flame_fitness[0]
            # Hitung F1 yang benar: cost = (1 - f1) + alpha*(features/total)
            # Jadi f1 ≈ 1 - cost + alpha*(features/total)
            best_features_count = np.count_nonzero(self.flames[0])
            penalty = self.alpha * (best_features_count / self.num_features)
            approx_f1 = 1.0 - best_cost + penalty
            print(f"✅ Cost: {best_cost:.4f} | F1-Score: {approx_f1:.4f} | Fitur: {best_features_count}")
            
        execution_time = time.time() - start_time
        print(f"\n🎉 Optimasi Selesai dalam {execution_time/60:.2f} menit!")
        
        # Kembalikan mask dari fitur terbaik
        best_feature_mask = self.flames[0] == 1
        best_cost = self.flame_fitness[0]
        best_features_count = np.count_nonzero(best_feature_mask)
        penalty = self.alpha * (best_features_count / self.num_features)
        best_f1 = 1.0 - best_cost + penalty  # F1 yang benar (tanpa penalty)
        
        return best_feature_mask, best_f1, best_cost

# ==========================================
# 5. EKSEKUSI & MLFLOW LOGGING
# ==========================================
with mlflow.start_run(run_name="MFO_Feature_Selection_Standardized"):
    # Parameter MFO yang dioptimalkan
    NUM_MOTHS = 30
    MAX_ITER = 30  # Naikkan iterasi sedikit agar eksplorasi maksimal
    ALPHA_PENALTY = 0.0  # Set ke 0.0 agar setara dengan GA yang tidak menggunakan pinalti jumlah fitur
    TRANSFER_FUNC = 'V-shaped'
    FITNESS_MODE = 'full'  # Gunakan 'full' agar setara dengan evaluasi data GA
    
    print("\n" + "="*70)
    print("MFO FEATURE SELECTION (UPGRADED: V-SHAPED + ADAPTIVE FLAMES)")
    print("="*70)
    print(f"   Moths: {NUM_MOTHS} | Iterations: {MAX_ITER}")
    print(f"   Transfer Function: {TRANSFER_FUNC} | Fitness Mode: {FITNESS_MODE}")
    print(f"   Alpha penalty: {ALPHA_PENALTY}")
    
    mfo = BinaryMFO(
        num_moths=NUM_MOTHS, 
        max_iterations=MAX_ITER, 
        num_features=num_features, 
        alpha=ALPHA_PENALTY,
        transfer_type=TRANSFER_FUNC,
        fitness_mode=FITNESS_MODE
    )
    best_mask, best_f1, best_cost = mfo.optimize()
    
    # Menerapkan seleksi pada dataset
    selected_features_indices = np.where(best_mask == 1)[0]
    selected_features_names = feature_names[selected_features_indices].tolist()
    num_selected_features = len(selected_features_names)
    
    X_train_selected = X_train[:, selected_features_indices]
    
    print("\n" + "="*50)
    print(f"🔥 HASIL SELEKSI FITUR MFO 🔥")
    print(f"Dari {num_features} fitur, MFO mempertahankan {num_selected_features} fitur inti.")
    print(f"Feature Reduction: {(1 - num_selected_features/num_features)*100:.1f}%")
    print(f"Fitur yang terpilih: {selected_features_names}")
    print(f"Best F1-Score (Val): {best_f1:.4f}")
    print(f"Best Cost: {best_cost:.4f}")
    print("="*50)
    
    # Simpan dataset hasil seleksi
    output_path = root_path / "Data/processed/train_selected_features.csv"
    X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features_names)
    Y_train_df = pd.DataFrame(Y_train, columns=target_cols)
    X_train_selected_full = pd.concat([X_train_selected_df, Y_train_df], axis=1)
    X_train_selected_full.to_csv(output_path, index=False)
    print(f"\n✓ Dataset hasil seleksi disimpan ke: {output_path}")
    
    # Logging ke MLflow 
    mlflow.log_param("feature_selection_method", "Moth-Flame Optimization (MFO)")
    mlflow.log_param("mfo_num_moths", NUM_MOTHS)
    mlflow.log_param("mfo_iterations", MAX_ITER)
    mlflow.log_param("mfo_alpha_penalty", ALPHA_PENALTY)
    mlflow.log_param("fitness_evaluation", "Internal Train-Val Split (80-20)")
    
    # Log XGBoost parameters (fitness phase)
    for param_name, param_value in xgb_params_fitness.items():
        mlflow.log_param(f"fitness_xgb_{param_name}", param_value)
    
    # Log feature info
    mlflow.log_param("multilabel_strategy", "ClassifierChain")
    mlflow.log_param("num_original_features", num_features)
    mlflow.log_param("num_selected_features", num_selected_features)
    mlflow.log_param("feature_reduction_percentage", f"{(1 - num_selected_features/num_features)*100:.1f}")
    
    # Log metrics
    mlflow.log_metric("mfo_best_f1_val", best_f1)
    mlflow.log_metric("mfo_best_cost", best_cost)
    mlflow.log_metric("mfo_feature_reduction_ratio", 1 - num_selected_features/num_features)
    
    # Log selected features
    mlflow.log_text(", ".join(selected_features_names), "selected_features.txt")
    
    print("\n✓ Metadata tracking selesai di MLflow")

