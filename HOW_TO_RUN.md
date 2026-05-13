# 🚀 HOW TO RUN PROJECT - LENGKAP & PRAKTIS

## 📋 Prerequisites (Persiapan Awal)

Pastikan sudah install:

- ✅ Python 3.8+ (recommend 3.9 atau 3.10)
- ✅ Anaconda atau Miniconda (atau virtualenv)
- ✅ Git (opsional, tapi bagus untuk version control)

Cek versi Python:

```bash
python --version
```

---

## 🔧 SETUP 1: Buat Virtual Environment

### Opsi A: Pake Anaconda (Recommended)

```bash
# Buka Anaconda Prompt / Terminal

# Navigate ke project folder
cd "d:\Amikom\Semester 6\Project Data Mining\Project"

# Buat virtual environment baru
conda create -n mental_health python=3.10

# Activate environment
conda activate mental_health

# Verify
python --version  # Harus 3.10.x
```

### Opsi B: Pake virtualenv

```bash
cd "d:\Amikom\Semester 6\Project Data Mining\Project"

# Buat venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

---

## 📦 SETUP 2: Install Dependencies

```bash
# Pastikan sudah activate environment
# Cek: Di terminal harus ada (mental_health) atau (venv)

# Install semua packages
pip install -r requirements.txt

# Tunggu sampai selesai (~5-10 menit tergantung internet)
```

**Jika ada error saat install:**

```bash
# Try upgrade pip
python -m pip install --upgrade pip

# Then retry
pip install -r requirements.txt
```

**Optional - Install Jupyter untuk keamanan:**

```bash
pip install jupyter notebook
```

---

## ✅ SETUP 3: Verifikasi Instalasi

```bash
# Test import semua library
python -c "import pandas; import xgboost; import sklearn; import shap; print('✓ All good!')"

# Jika berhasil akan print: ✓ All good!
# Jika gagal akan ada error message
```

---

## 🎬 CARA 1: Run Notebook Individual (Recommended untuk learning)

### Step 1: Buka Jupyter Notebook

```bash
# Pastikan sudah activate environment
cd "d:\Amikom\Semester 6\Project Data Mining\Project"
jupyter notebook
```

Ini akan buka browser ke `http://localhost:8888/tree`

### Step 2: Navigate ke folder notebooks

```
Klik: notebooks/
```

### Step 3: Buka notebook yang mau dijalankan

Contoh mulai dari awal:

```
01_data_acquisition.ipynb
```

### Step 4: Run cells

```
Tekan: Shift + Enter (untuk run cell by cell)
Atau: Ctrl + Shift + Enter (untuk run all cells)
```

**Running Progress:**

```
Sel akan execute secara sequential
Lihat output di bawah setiap cell
Jika ada error, fix dan re-run
```

---

## 🎬 CARA 2: Run Notebook Lengkap (Batch Mode)

Jika mau run notebook without UI:

```bash
# Option A: Run specific notebook
jupyter nbconvert --to notebook --execute "notebooks/01_data_acquisition.ipynb"

# Option B: Run all notebooks in sequence
for file in notebooks/0*.ipynb; do
  jupyter nbconvert --to notebook --execute "$file"
done
```

---

## 🎬 CARA 3: Run dari Command Line (Python Script)

Jika ada script Python:

```bash
# Dari project root folder
python src/preprocessing.py
python src/mfo_optimizer.py
python src/model_trainer.py
```

---

## 📊 CARA 4: Run MLflow UI (Tracking Experiments)

```bash
# Dari project root
cd "d:\Amikom\Semester 6\Project Data Mining\Project"

# Start MLflow UI
mlflow ui

# Buka browser
http://localhost:5000
```

Disini bisa lihat:

- Semua eksperimen yang pernah dijalankan
- Parameter & metrics per run
- Comparison antar runs
- Artifacts (models, plots, etc)

---

## 📈 CARA 5: Deploy Model (Streamlit Web App)

Jika ada `app.py`:

```bash
streamlit run app.py

# Buka: http://localhost:8501
```

---

## 🎯 RECOMMENDED EXECUTION PATH (Jalankan dalam urutan ini)

### STEP 1️⃣: Phase 1 - Data Exploration

```bash
jupyter notebook
# Buka: 01_data_acquisition.ipynb
# Run semua cells (Ctrl + Shift + Enter)
# Output: Dataset statistics

# Lanjut: 02_exploratory_data_analysis.ipynb
# Run semua cells
# Output: EDA plots di eda_figures/
```

**Waktu:** ~15-20 menit

---

### STEP 2️⃣: Phase 2 - Preprocessing

```bash
jupyter notebook
# Buka: 03_data_preprocessing.ipynb
# Run all → Output: cleaned_data.csv

# Lanjut: 04_train_test_split.ipynb
# Run all → Output: train_data.csv, test_data.csv

# Lanjut: 05_multilabel_imbalance_handling.ipynb
# Run all → Output: train_balanced_multilabel.csv
```

**Waktu:** ~30-45 menit

⚠️ **CRITICAL**: Jangan skip 05! Itu balancing imbalance data

---

### STEP 3️⃣: Phase 3 - Feature Selection

```bash
jupyter notebook
# Buka: 06_mfo_feature_selection.ipynb
# Run all → Output: train_selected_features.csv
# (Progress akan di-log setiap iterasi)
```

**Waktu:** ~40-60 menit (tergantung CPU)

💡 **Tips:** Jika terlalu lama, reduce NUM_MOTHS dari 30 jadi 20

---

### STEP 4️⃣: Phase 4 - Model Training

```bash
jupyter notebook
# Buka: 07_multilabel_xgboost_modeling.ipynb
# Run all → Tuned model + parameters

# ATAU jika mau compare algorithms:
# 07c_epic_algorithm_comparison.ipynb → XGBoost vs RF vs CatBoost
```

**Waktu:** ~90-120 menit (RandomizedSearchCV intensive)

⚠️ **INTENSIVE**: CPU akan high usage, be patient

---

### STEP 5️⃣: Phase 6 - Evaluation

```bash
jupyter notebook
# Buka: 08_performance_evaluation.ipynb
# Run all → Test metrics

# Lanjut: 08b_threshold_optimization.ipynb
# Run all → Optimal thresholds per label

# Lanjut: 09_kfold_cross_validation.ipynb
# Run all → Mean F1 ± confidence interval
```

**Waktu:** ~60-90 menit

---

### STEP 6️⃣: Phase 7 - Explainability

```bash
jupyter notebook
# Buka: 10_shap_explainability.ipynb
# Run all → SHAP plots (lihat di shap_figures/)

# Lanjut: 11_final_model.ipynb
# Run all → multilabel_xgboost_mfo.pkl
```

**Waktu:** ~30-45 menit

---

## 🔥 QUICK START (Hanya 20 menit)

Jika cuma mau test-run tanpa tunggu lama:

```bash
jupyter notebook

# Run 3 notebook ini aja:
1. 01_data_acquisition.ipynb       (5 min)
2. 02_exploratory_data_analysis.ipynb  (10 min)
3. 07_multilabel_xgboost_modeling.ipynb (5 min, cuma load model)
   # Disable tuning, load pre-trained model aja
```

---

## ⚡ PRO TIPS untuk Faster Execution

### 1. Reduce Data Size (Testing)

```python
# Di cell pertama, add:
train_df = train_df.sample(frac=0.5, random_state=42)  # Ambil 50% aja
```

### 2. Reduce Iterations (MFO)

```python
NUM_MOTHS = 15  # Default 30
MAX_ITER = 10   # Default 20
```

### 3. Reduce CV Folds

```python
cv = 3  # Default 5
```

### 4. Parallel Processing

```python
n_jobs = -1  # Use all cores
```

---

## 🐛 TROUBLESHOOTING

### Error: "ModuleNotFoundError: No module named 'xgboost'"

```bash
# Solution: Install lagi
pip install --upgrade xgboost
```

### Error: "CUDA not found" (XGBoost warning)

```bash
# Ignore saja, XGBoost fallback ke CPU
# Or install CUDA (advanced, tidak perlu)
```

### Error: "Memory Error" atau hang

```bash
# Solution 1: Reduce data size
# Solution 2: Reduce model parameters
# Solution 3: Close other programs
# Solution 4: Restart kernel (Kernel → Restart & Clear Output)
```

### Error: "Test set not found"

```bash
# Solution: Run 04_train_test_split.ipynb dulu
# Jangan skip phase manapun
```

### Notebook terlalu lama / CPU 100%

```bash
# Normal! XGBoost tuning intensive
# Bisa continue browsing, jangan close terminal
# Tunggu sampai cell selesai (ada ✓ di sebelah kiri)
```

---

## 📊 MONITORING PROGRESS

### Cek apakah execution berjalan:

```bash
# Di terminal, harusnya lihat output seperti:
# [LightGBM: training started]
# [XGBoost CV Round 1/50]
# [Iteration 1/20: Cost = 0.3245]
# etc
```

### Cek folder output sudah kebuat:

```
outputs/
├── eda_figures/          # Dari 02 ✓
├── imbalance_figures/    # Dari 05 ✓
├── best_parameter/       # Dari 07 ✓
└── shap_figures/         # Dari 10 ✓
```

### Cek MLflow tracking:

```bash
mlflow ui
# Buka http://localhost:5000
# Lihat experiments & runs
```

---

## 🎯 EXPECTED OUTPUTS Per Notebook

| Notebook | Expected Output                                |
| -------- | ---------------------------------------------- |
| 01       | Terminal prints (statistics)                   |
| 02       | `outputs/eda_figures/`                         |
| 03       | `Data/processed/cleaned_data.csv`              |
| 04       | `Data/split/train_data.csv`, `test_data.csv`   |
| 05       | `Data/processed/train_balanced_multilabel.csv` |
| 06       | `Data/processed/train_selected_features.csv`   |
| 06b      | Comparison plots, terminal output              |
| 07       | Best hyperparameters, tuned model (in memory)  |
| 08       | `notebooks/classification_report.txt`          |
| 08b      | `notebooks/optimal_thresholds.txt`             |
| 09       | CV metrics printed                             |
| 10       | `outputs/shap_figures/`                        |
| 11       | `models/multilabel_xgboost_mfo.pkl`            |

---

## ✅ VERIFICATION CHECKLIST

Setelah run semua, cek:

```
✓ Data/processed/cleaned_data.csv exists
✓ Data/processed/train_balanced_multilabel.csv exists
✓ Data/processed/train_selected_features.csv exists
✓ Data/split/train_data.csv exists
✓ Data/split/test_data.csv exists
✓ outputs/eda_figures/ punya banyak plot
✓ outputs/shap_figures/ punya SHAP plots
✓ models/multilabel_xgboost_mfo.pkl exists
✓ notebooks/classification_report.txt exists
✓ notebooks/optimal_thresholds.txt exists
✓ mlruns/ folder punya banyak experiment folders
```

Jika semua ✓, berarti execution berhasil 🎉

---

## 📞 NEED HELP?

### Jika error atau stuck:

1. **Check error message** - baca dengan teliti
2. **Google error message** - sering punya solution
3. **Restart kernel** - Kernel → Restart & Clear Output
4. **Check input data** - verify file exists di Data/
5. **Check dependencies** - `pip list | grep xgboost`
6. **Check phase** - apakah sudah run phase sebelumnya?

### Common issues & solutions:

```
"No such file or directory"
→ Check path, file sudah ada?

"MemoryError"
→ Reduce data size atau batch size

"Module not found"
→ pip install [module]

"Shape mismatch"
→ Check data preprocessing, feature mismatch

"RandomState error"
→ Set random_state=42 consistently
```

---

## 🎬 FULL EXECUTION TIME ESTIMATE

```
Phase 1 (01-02):        15 min    ░░░░░░░░░░
Phase 2 (03-05):        40 min    ░░░░░░░░░░░░░░
Phase 3 (06-06b):       60 min    ░░░░░░░░░░░░░░░░░░
Phase 4 (07):          120 min    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Phase 5 (07a-07g):     300 min    (OPTIONAL - skip if just testing)
Phase 6 (08-09):        80 min    ░░░░░░░░░░░░░░░░░░░░
Phase 7 (10-11):        50 min    ░░░░░░░░░░░░░

TOTAL (Phase 1-4 + 6-7): ~365 min = 6 hours ⏱️

Jika run Phase 5 (benchmarking): +300 min = 11 hours total
```

---

## 🚀 FASTEST PATH (Testing Only)

Jika cuma pengen test tanpa full run:

```bash
jupyter notebook

# Run these 5 notebooks only:
1. 01_data_acquisition.ipynb (5 min)
2. 02_exploratory_data_analysis.ipynb (10 min)
3. 03_data_preprocessing.ipynb (10 min)
4. 04_train_test_split.ipynb (5 min)
5. 05_multilabel_imbalance_handling.ipynb (15 min)

# Total: ~45 min untuk full data prep
# Skip 06-11 jika just testing
```

---

## 🎯 SUMMARY: TL;DR

### Setup (1x):

```bash
conda create -n mental_health python=3.10
conda activate mental_health
pip install -r requirements.txt
```

### Run (every time):

```bash
jupyter notebook
# Buka notebooks/ dan run sequential dari 01-11
```

### Monitor:

```bash
mlflow ui  # Di terminal lain
```

### Done:

```
Models di: models/multilabel_xgboost_mfo.pkl
Metrics di: notebooks/classification_report.txt
SHAP plots di: outputs/shap_figures/
```

---

**Now go run it! 🚀**
