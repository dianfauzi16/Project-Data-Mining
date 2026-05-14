"""
Mental Health Risk Assessment System
Streamlit Application for Multi-Label Depression, Anxiety & Stress Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Sistem Penilaian Risiko Kesehatan Mental",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-left: 4px solid #ff0000;
        border-radius: 0.25rem;
    }
    .risk-medium {
        background-color: #ffffcc;
        padding: 1rem;
        border-left: 4px solid #ffaa00;
        border-radius: 0.25rem;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 1rem;
        border-left: 4px solid #00aa00;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL & CONFIG (DINAMIS dari JSON)
# ==========================================
BASE_DIR = Path(__file__).parent.parent

@st.cache_resource
def load_model():
    """Load trained XGBoost classifier chain model"""
    model_path = BASE_DIR / "models" / "multilabel_xgboost_classifier_chain.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_config():
    """Load configuration from JSON files saved by NB 11 (dinamis)"""
    # ✅ Load fitur & threshold dari JSON (bukan hardcode)
    threshold_path = BASE_DIR / "models" / "optimal_thresholds.json"
    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)

    selected_features = threshold_data["selected_features"]
    optimal_thresholds = threshold_data["thresholds"]

    # Mapping deskripsi fitur
    personality_features = {
        'TIPI1': 'Extraverted (vs. Reserved)',
        'TIPI2': 'Critical (vs. Lenient)',
        'TIPI3': 'Dependable (vs. Disorganized)',
        'TIPI4': 'Anxious (vs. Calm)',
        'TIPI5': 'Open to new experiences (vs. Conventional)',
        'TIPI6': 'Reserved (vs. Extraverted)',
        'TIPI7': 'Sympathetic (vs. Critical)',
        'TIPI8': 'Disorganized (vs. Dependable)',
        'TIPI9': 'Calm (vs. Anxious)',
        'TIPI10': 'Conventional (vs. Open to new experiences)'
    }

    lifestyle_features = {
        'VCL1': 'Alcohol use',
        'VCL2': 'Benzodiazepine use',
        'VCL3': 'Caffeine use',
        'VCL4': 'Cannabis use',
        'VCL5': 'Cocaine use',
        'VCL6': 'Heroin use',
        'VCL7': 'Amphetamine use',
        'VCL8': 'Ecstasy use',
        'VCL9': 'LSD use',
        'VCL10': 'Mushrooms use',
        'VCL11': 'Legal Highs',
        'VCL12': 'Nicotine use',
        'VCL13': 'Methadone use',
        'VCL14': 'VSA use',
        'VCL15': 'PCP use',
        'VCL16': 'Crack use'
    }

    config = {
        "selected_features": selected_features,
        "target_cols": ['risk_depression', 'risk_anxiety', 'risk_stress'],
        "optimal_thresholds": optimal_thresholds,
        "personality_features": {k: v for k, v in personality_features.items() if k in selected_features},
        "lifestyle_features": {k: v for k, v in lifestyle_features.items() if k in selected_features},
    }
    return config

# Load model and config
try:
    model = load_model()
    config = load_config()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def chain_predict_proba(model, X):
    """Reconstruct predict_proba for ClassifierChain with XGBoost"""
    X_arr = X.values if hasattr(X, 'values') else X.copy()
    n_samples = X_arr.shape[0]
    n_targets = len(model.estimators_)
    all_probas = np.zeros((n_samples, n_targets))
    
    X_aug = X_arr.copy()
    for i, estimator in enumerate(model.estimators_):
        proba = estimator.predict_proba(X_aug)[:, 1]
        all_probas[:, i] = proba
        pred_label = (proba >= 0.5).astype(int).reshape(-1, 1)
        X_aug = np.hstack([X_aug, pred_label])
    
    return [np.column_stack([1 - all_probas[:, i], all_probas[:, i]])
            for i in range(n_targets)]

def predict_with_thresholds(model, X, thresholds):
    """Make predictions with optimal thresholds"""
    probas = chain_predict_proba(model, X)
    predictions = []
    for i, (proba, threshold) in enumerate(zip(probas, thresholds)):
        pred = (proba[:, 1] >= threshold).astype(int)
        predictions.append(pred)
    return np.column_stack(predictions), probas

def get_risk_level(probability):
    """Classify risk level based on probability"""
    if probability >= 0.7:
        return "🔴 HIGH RISK"
    elif probability >= 0.4:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

def get_risk_color(probability):
    """Get CSS class for risk level"""
    if probability >= 0.7:
        return "risk-high"
    elif probability >= 0.4:
        return "risk-medium"
    else:
        return "risk-low"

def get_risk_explanation(condition, probability):
    """Get detailed explanation based on risk level and condition"""
    level = "high" if probability >= 0.7 else ("medium" if probability >= 0.4 else "low")
    explanations = {
        'risk_depression': {
            'high': '⚠️ **Risiko Tinggi Depression.** Skor menunjukkan kemungkinan besar mengalami gejala depresi seperti perasaan sedih berkepanjangan, kehilangan minat, atau gangguan tidur. **Sangat disarankan** untuk segera berkonsultasi dengan profesional kesehatan mental.',
            'medium': '⚡ **Risiko Sedang Depression.** Ada beberapa indikator yang menunjukkan potensi gejala depresi ringan hingga sedang. Pertimbangkan untuk berbicara dengan konselor atau psikolog untuk evaluasi lebih lanjut.',
            'low': '✅ **Risiko Rendah Depression.** Skor menunjukkan kemungkinan rendah mengalami gejala depresi saat ini. Tetap jaga kesehatan mental dengan pola hidup sehat dan dukungan sosial yang baik.'
        },
        'risk_anxiety': {
            'high': '⚠️ **Risiko Tinggi Anxiety.** Skor menunjukkan kemungkinan besar mengalami gejala kecemasan berlebihan, seperti rasa khawatir yang tidak terkendali, jantung berdebar, atau kesulitan berkonsentrasi. **Segera konsultasikan** dengan profesional.',
            'medium': '⚡ **Risiko Sedang Anxiety.** Ada indikasi potensi gangguan kecemasan ringan. Teknik relaksasi, mindfulness, dan konsultasi dengan profesional dapat membantu.',
            'low': '✅ **Risiko Rendah Anxiety.** Skor menunjukkan tingkat kecemasan yang normal. Pertahankan kebiasaan baik seperti olahraga teratur dan istirahat cukup.'
        },
        'risk_stress': {
            'high': '⚠️ **Risiko Tinggi Stress.** Skor menunjukkan kemungkinan besar mengalami tingkat stres yang tinggi. Ini dapat memengaruhi kesehatan fisik dan mental. **Sangat disarankan** untuk mencari bantuan profesional dan menerapkan strategi manajemen stres.',
            'medium': '⚡ **Risiko Sedang Stress.** Ada indikasi tingkat stres yang perlu diperhatikan. Cobalah teknik manajemen stres seperti olahraga, meditasi, atau berbicara dengan orang terdekat.',
            'low': '✅ **Risiko Rendah Stress.** Tingkat stres Anda tampak terkendali. Terus jaga keseimbangan antara pekerjaan, istirahat, dan aktivitas sosial.'
        }
    }
    return explanations.get(condition, {}).get(level, '')

# ==========================================
# 4. MAIN APP LAYOUT
# ==========================================
st.markdown('<div class="main-header">🧠 Sistem Penilaian Risiko Kesehatan Mental</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Prediksi Risiko Depresi, Kecemasan & Stres Berdasarkan Kepribadian & Gaya Hidup</div>', unsafe_allow_html=True)

# Identify which feature groups are active (dari MFO selection)
active_tipi = sorted([f for f in config['selected_features'] if f.startswith('TIPI')])
active_vcl = sorted([f for f in config['selected_features'] if f.startswith('VCL')])
active_demo = [f for f in config['selected_features'] if not f.startswith(('TIPI', 'VCL'))]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediksi", "🧠 Analisis SHAP", "📊 Info Model", "❓ Bantuan"])

with tab1:
    st.header("Masukkan Informasi Anda")
    
    # ==========================================
    # DEMOGRAPHIC INPUTS (dinamis berdasarkan fitur MFO)
    # ==========================================
    st.subheader("📋 Informasi Demografis")
    
    demo_inputs = {}
    demo_cols = st.columns(3)
    
    # Definisi semua input demografis yang mungkin
    demo_definitions = {
        'age': {'label': 'Usia', 'type': 'slider', 'min': 15, 'max': 24, 'default': 20},
        'gender': {'label': 'Jenis Kelamin', 'type': 'select', 'options': [1, 2], 
                   'format': lambda x: "Laki-laki" if x == 1 else "Perempuan"},
        'engnat': {'label': 'Bahasa Inggris sebagai Bahasa Utama', 'type': 'select', 'options': [0, 1],
                   'format': lambda x: "Tidak" if x == 0 else "Ya"},
        'education': {'label': 'Tingkat Pendidikan', 'type': 'select', 'options': [1, 2, 3, 4],
                      'format': lambda x: ["Di Bawah SMA", "SMA/Sederajat", "Sarjana (S1)", "Pascasarjana (S2/S3)"][x-1]},
        'urban': {'label': 'Wilayah Tempat Tinggal', 'type': 'select', 'options': [1, 2, 3],
                  'format': lambda x: ["Pedesaan", "Pinggiran Kota", "Perkotaan"][x-1]},
        'religion': {'label': 'Agama', 'type': 'select', 'options': [0, 1, 2, 3, 4, 5, 6],
                     'format': lambda x: ["Agnostik", "Ateis", "Buddha", "Kristen", "Hindu", "Yahudi", "Islam"][x]},
        'orientation': {'label': 'Orientasi Seksual', 'type': 'select', 'options': [0, 1, 2],
                        'format': lambda x: ["Heteroseksual", "Biseksual", "Homoseksual"][x]},
        'race': {'label': 'Ras/Etnis', 'type': 'select', 'options': [0, 1, 2, 3, 4, 5],
                 'format': lambda x: ["Kulit Putih", "Kulit Hitam", "Asia", "Campuran", "Lainnya", "Latin"][x]},
        'voted': {'label': 'Suara Politik', 'type': 'select', 'options': [0, 1, 2, 3],
                  'format': lambda x: ["Tidak", "Ya", "Abstain", "Tidak Berlaku"][x]},
        'married': {'label': 'Status Pernikahan', 'type': 'select', 'options': [0, 1, 2, 3, 4],
                    'format': lambda x: ["Belum Menikah", "Menikah", "Cerai", "Janda/Duda", "Pisah"][x]},
        'familysize': {'label': 'Jumlah Anggota Keluarga', 'type': 'number', 'min': 0, 'max': 10, 'default': 3},
    }
    
    col_idx = 0
    for feat in active_demo:
        if feat in demo_definitions:
            defn = demo_definitions[feat]
            with demo_cols[col_idx % 3]:
                if defn['type'] == 'slider':
                    demo_inputs[feat] = st.slider(defn['label'], min_value=defn['min'], max_value=defn['max'], value=defn['default'])
                elif defn['type'] == 'select':
                    demo_inputs[feat] = st.selectbox(defn['label'], defn['options'], format_func=defn['format'])
                elif defn['type'] == 'number':
                    demo_inputs[feat] = st.number_input(defn['label'], min_value=defn['min'], max_value=defn['max'], value=defn['default'])
            col_idx += 1
    
    st.divider()
    
    # ==========================================
    # PERSONALITY TRAITS (TIPI) — hanya yang dipilih MFO
    # ==========================================
    if active_tipi:
        st.subheader("🎭 Sifat Kepribadian (TIPI-10)")
        st.caption("Skala 1-7 dimana 1 = Sangat Tidak Setuju, 2 = Tidak Setuju, 3 = Agak Tidak Setuju, 4 = Netral, 5 = Agak Setuju, 6 = Setuju, 7 = Sangat Setuju")
        
        tipi_descriptions = {
            'TIPI1': '📊 Saya adalah orang yang riang, ekstrovert\n💡 (Sebaliknya: Reserved/Tertutup)',
            'TIPI2': '⚖️ Saya adalah orang yang kritis, suka berdebat\n💡 (Sebaliknya: Pemaaf/Lembut)',
            'TIPI3': '📋 Saya adalah orang yang dapat diandalkan, teratur\n💡 (Sebaliknya: Ceroboh)',
            'TIPI4': '😟 Saya mudah cemas, terguncang emosionalnya\n💡 (Sebaliknya: Tenang/Stabil)',
            'TIPI5': '🎨 Saya terbuka terhadap pengalaman baru, kreatif\n💡 (Sebaliknya: Konvensional)',
            'TIPI6': '🤐 Saya adalah orang yang pendiam, tertutup\n💡 (Sebaliknya: Ekstrovert/Sosial)',
            'TIPI7': '❤️ Saya adalah orang yang simpatik, hangat\n💡 (Sebaliknya: Kritis)',
            'TIPI8': '🗑️ Saya adalah orang yang tidak terorganisir, ceroboh\n💡 (Sebaliknya: Rapi)',
            'TIPI9': '😊 Saya adalah orang yang tenang, stabil emosional\n💡 (Sebaliknya: Mudah cemas)',
            'TIPI10': '🔄 Saya adalah orang yang konvensional\n💡 (Sebaliknya: Terbuka/Inovatif)'
        }
        
        tipi_inputs = {}
        n_tipi_cols = min(5, len(active_tipi))
        tipi_cols = st.columns(n_tipi_cols)
        
        for idx, key in enumerate(active_tipi):
            col = tipi_cols[idx % n_tipi_cols]
            with col:
                st.write(f"**{key}**")
                if key in tipi_descriptions:
                    st.caption(tipi_descriptions[key])
                tipi_inputs[key] = st.slider(
                    f"Nilai {key}",
                    min_value=1,
                    max_value=7,
                    value=4,
                    label_visibility="collapsed"
                )
        
        st.divider()
    else:
        tipi_inputs = {}
    
    # ==========================================
    # LIFESTYLE/SUBSTANCE USE (VCL) — hanya yang dipilih MFO
    # ==========================================
    if active_vcl:
        st.subheader("💊 Gaya Hidup & Penggunaan Zat (VCL)")
        st.caption("Skala 0-6: 0=Tidak Pernah, 1=Lebih dari 10 Tahun, 2=Kurang dari 10 Tahun, 3=Satu Tahun Terakhir, 4=Satu Bulan Terakhir, 5=Satu Minggu Terakhir, 6=Satu Hari Terakhir")
        
        vcl_descriptions = {
            'VCL1': '🍺 Konsumsi Alkohol',
            'VCL2': '💊 Penggunaan Benzodiazepine',
            'VCL3': '☕ Konsumsi Kafein',
            'VCL4': '🌿 Penggunaan Cannabis',
            'VCL5': '⚪ Penggunaan Kokain',
            'VCL6': '💉 Penggunaan Heroin',
            'VCL7': '⚡ Penggunaan Amfetamin',
            'VCL8': '💊 Penggunaan Ecstasy',
            'VCL9': '🌈 Penggunaan LSD',
            'VCL10': '🍄 Penggunaan Psilocybin (Jamur)',
            'VCL11': '⚗️ Legal Highs',
            'VCL12': '🚬 Penggunaan Tembakau/Nikotin',
            'VCL13': '⚕️ Penggunaan Metadon',
            'VCL14': '⚠️ Penggunaan VSA',
            'VCL15': '🔷 Penggunaan PCP',
            'VCL16': '🪨 Penggunaan Crack'
        }
        
        vcl_inputs = {}
        n_vcl_cols = min(3, len(active_vcl))
        vcl_cols = st.columns(n_vcl_cols)
        
        for idx, key in enumerate(active_vcl):
            col = vcl_cols[idx % n_vcl_cols]
            with col:
                st.write(f"**{key}**")
                if key in vcl_descriptions:
                    st.caption(vcl_descriptions[key])
                vcl_inputs[key] = st.slider(
                    f"Frekuensi {key}",
                    min_value=0,
                    max_value=6,
                    value=0,
                    label_visibility="collapsed"
                )
    else:
        vcl_inputs = {}
    
    # ==========================================
    # PREDICTION BUTTON
    # ==========================================
    st.divider()
    
    if st.button("🚀 Dapatkan Prediksi", use_container_width=True, type="primary"):
        # Prepare input data — gabungkan semua input
        input_dict = {**demo_inputs, **tipi_inputs, **vcl_inputs}
        
        # Create DataFrame dengan urutan fitur yang benar
        input_df = pd.DataFrame([input_dict])[config['selected_features']]
        
        # Make prediction
        try:
            thresholds_list = [config['optimal_thresholds'][t] for t in config['target_cols']]
            predictions, probabilities = predict_with_thresholds(model, input_df, thresholds_list)
            
            # Display results
            st.success("✅ Prediksi Selesai!")
            
            st.markdown("## 📈 Hasil Prediksi")
            
            target_labels = {
                'risk_depression': ('😔 Depression', '#ff6b6b'),
                'risk_anxiety': ('😰 Anxiety', '#ffa94d'),
                'risk_stress': ('😣 Stress', '#69db7c')
            }
            
            result_cols = st.columns(3)
            for idx, (target, (label, color)) in enumerate(target_labels.items()):
                with result_cols[idx]:
                    prob = probabilities[idx][0, 1]
                    pred = predictions[0, idx]
                    risk_level = get_risk_level(prob)
                    threshold = config['optimal_thresholds'][target]
                    
                    st.markdown(f"### {label}")
                    st.metric("Probabilitas", f"{prob:.1%}", delta=f"Ambang Batas: {threshold:.2f}")
                    st.progress(min(prob, 1.0))
                    
                    if pred == 1:
                        st.error(f"{risk_level} — 🔴 Risiko Terdeteksi")
                    else:
                        st.success(f"{risk_level} — 🟢 Tidak Ada Risiko")
            
            # Detailed Explanations
            st.markdown("---")
            st.markdown("## 📝 Penjelasan Detail per Kondisi")
            
            for target, (label, _) in target_labels.items():
                prob = probabilities[list(target_labels.keys()).index(target)][0, 1]
                explanation = get_risk_explanation(target, prob)
                with st.expander(f"{label} — {get_risk_level(prob)}", expanded=True):
                    st.markdown(explanation)
                    st.caption(f"Probabilitas: {prob:.1%} | Threshold model: {config['optimal_thresholds'][target]:.2f}")
            
            # Summary
            st.markdown("---")
            risk_count = sum(predictions[0])
            st.subheader("📋 Ringkasan")
            st.info(f"**Penilaian Keseluruhan:** {risk_count} dari 3 faktor risiko terdeteksi")
            
            if risk_count == 0:
                st.success("✅ Risiko rendah secara keseluruhan. Tetap jaga pola hidup sehat dan dukungan sosial yang baik.")
            elif risk_count == 1:
                st.warning("⚠️ Satu faktor risiko terdeteksi. Disarankan untuk berkonsultasi dengan profesional kesehatan mental untuk evaluasi lebih lanjut.")
            elif risk_count == 2:
                st.error("🔴 Dua faktor risiko terdeteksi. Sangat disarankan untuk segera berkonsultasi dengan psikolog atau psikiater.")
            else:
                st.error("🔴 Tiga faktor risiko terdeteksi. Sangat disarankan untuk segera mencari bantuan profesional kesehatan mental.")
            
            # Save to session
            st.session_state.last_prediction = {
                'input': input_dict,
                'probabilities': probabilities,
                'predictions': predictions,
                'risk_count': risk_count
            }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

with tab2:
    st.header("🧠 SHAP Analysis — Interpretasi Model")
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** menunjukkan kontribusi setiap fitur terhadap prediksi model.
    Fitur dengan nilai SHAP tinggi memiliki pengaruh besar terhadap keputusan model.
    """)
    
    shap_dir = BASE_DIR / "outputs" / "shap_figures"
    
    if shap_dir.exists():
        shap_tab1, shap_tab2, shap_tab3 = st.tabs(["😔 Depression", "😰 Anxiety", "😣 Stress"])
        
        conditions = [
            (shap_tab1, 'depression', 'Depression'),
            (shap_tab2, 'anxiety', 'Anxiety'),
            (shap_tab3, 'stress', 'Stress')
        ]
        
        for shap_tab, condition, title in conditions:
            with shap_tab:
                st.subheader(f"Faktor Pemicu {title}")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### 📊 Feature Importance")
                    bar_path = shap_dir / f"shap_bar_{condition}.png"
                    if bar_path.exists():
                        st.image(str(bar_path), use_container_width=True)
                    else:
                        st.warning("Grafik belum tersedia.")
                
                with col_b:
                    st.markdown("#### 🔍 SHAP Summary Plot")
                    summary_path = shap_dir / f"shap_summary_{condition}.png"
                    if summary_path.exists():
                        st.image(str(summary_path), use_container_width=True)
                    else:
                        st.warning("Grafik belum tersedia.")
                
                st.markdown(f"""
                **Cara Membaca:**
                - **Bar Plot** (kiri): Rata-rata kontribusi setiap fitur. Semakin tinggi = semakin penting.
                - **Summary Plot** (kanan): Merah = nilai fitur tinggi, Biru = rendah. Kanan = meningkatkan risiko, Kiri = menurunkan.
                """)
    else:
        st.warning("⚠️ Folder SHAP figures tidak ditemukan. Jalankan NB 10 terlebih dahulu.")
    
    st.info("💡 SHAP values dihitung berdasarkan keseluruhan data test, bukan per individu.")

with tab3:
    st.header("📊 Informasi Model")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Algoritma", "XGBoost + ClassifierChain")
        st.metric("Fitur Digunakan", f"{len(config['selected_features'])} (seleksi MFO)")
        st.metric("Target", len(config['target_cols']))
    
    with col2:
        st.metric("Jenis Model", "Klasifikasi Multi-Label")
        st.metric("Metode Validasi", "5-Fold Cross-Validation")
        st.metric("Seleksi Fitur", "Moth-Flame Optimization (MFO)")
    
    st.divider()
    
    # Performance Metrics
    st.subheader("🏆 Performa Model")
    perf_cols = st.columns(4)
    perf_cols[0].metric("Macro F1-Score", "0.7928")
    perf_cols[1].metric("Micro F1-Score", "0.7927")
    perf_cols[2].metric("Hamming Loss", "0.2647")
    perf_cols[3].metric("Exact Match", "0.5655")
    
    st.divider()
    
    st.subheader("📌 Fitur Terpilih (oleh MFO)")
    
    if active_tipi:
        st.markdown("#### 🎭 Sifat Kepribadian (TIPI)")
        st.text(", ".join(active_tipi))
    
    if active_vcl:
        st.markdown("#### 💊 Penggunaan Zat (VCL)")
        st.text(", ".join(active_vcl))
    
    if active_demo:
        st.markdown("#### 👥 Demografis")
        st.text(", ".join(sorted(active_demo)))
    
    st.divider()
    
    st.subheader("🎯 Ambang Batas Optimal")
    threshold_data = pd.DataFrame([
        {
            'Faktor Risiko': target.replace('risk_', '').title(),
            'Ambang Batas': f"{threshold:.2f}"
        }
        for target, threshold in config['optimal_thresholds'].items()
    ])
    st.table(threshold_data)
    
    st.info("""
    **Catatan:** Ambang batas dikalibrasi menggunakan data validasi asli (tanpa SMOTE) untuk memastikan
    performa yang realistis pada distribusi data dunia nyata.
    """)

with tab4:
    st.header("❓ Bantuan & FAQ")
    
    with st.expander("📊 Apa arti tingkat risiko?", expanded=True):
        st.markdown("""
        | Level | Probabilitas | Arti | Rekomendasi |
        |:---:|:---:|:---|:---|
        | 🟢 **RENDAH** | < 40% | Risiko rendah | Jaga pola hidup sehat |
        | 🟡 **SEDANG** | 40-70% | Ada indikasi awal | Konsultasi dengan konselor |
        | 🔴 **TINGGI** | > 70% | Risiko tinggi | Segera konsultasi profesional |
        
        **Catatan:** Ambang batas model: Depresi={:.2f}, Kecemasan={:.2f}, Stres={:.2f}.
        """.format(
            config['optimal_thresholds']['risk_depression'],
            config['optimal_thresholds']['risk_anxiety'],
            config['optimal_thresholds']['risk_stress']
        ))
    
    with st.expander("🔍 Apa itu aplikasi ini?"):
        st.write("""
        Ini adalah Sistem Penilaian Risiko Kesehatan Mental yang memprediksi kemungkinan Depresi, Kecemasan,
        dan Stres berdasarkan sifat kepribadian (TIPI-10) dan informasi gaya hidup/penggunaan zat (VCL).
        
        Sistem ini menggunakan machine learning untuk mengidentifikasi individu berisiko dan dapat membantu:
        - Deteksi dini risiko kesehatan mental
        - Screening untuk intervensi
        - Memahami hubungan kepribadian dengan kesehatan mental
        """)
    
    with st.expander("📈 Seberapa akurat model ini?"):
        st.write(f"""
        Model mencapai performa pada **data uji yang belum pernah dilihat (5.436 sampel)**:
        - **Macro F1-Score**: 0,7928 (dengan optimasi ambang batas)
        - **F1 per-label**: Depresi=0,79, Kecemasan=0,79, Stres=0,80
        - **Rata-rata 5-Fold CV**: 0,7560 ± 0,0068 (stabil)
        - **Fitur**: {len(config['selected_features'])} fitur dipilih oleh MFO (dari 37 fitur awal)
        
        Namun, model ini **TIDAK** menggantikan penilaian profesional kesehatan mental.
        Selalu konsultasikan dengan profesional kesehatan mental yang berkualifikasi.
        """)
    
    with st.expander("🎭 Apa yang diukur oleh pertanyaan TIPI?"):
        st.write("""
        TIPI-10 mengukur Lima Besar sifat kepribadian (Big Five):
        - **Ekstraversi** (terbuka vs. tertutup)
        - **Keramahan** (simpatik vs. kritis)
        - **Kesadaran** (dapat diandalkan vs. ceroboh)
        - **Neurotisisme** (mudah cemas vs. tenang)
        - **Keterbukaan** (terbuka terhadap pengalaman baru vs. konvensional)
        
        Sifat-sifat kepribadian ini diketahui berkorelasi dengan kondisi kesehatan mental.
        """)
    
    with st.expander("💊 Apa arti skor VCL?"):
        st.write("""
        Skor Penggunaan Zat (VCL):
        - **0** = Tidak Pernah Menggunakan
        - **1** = Digunakan > 10 tahun lalu
        - **2** = Digunakan dalam dekade terakhir
        - **3** = Digunakan dalam satu tahun terakhir
        - **4** = Digunakan dalam satu bulan terakhir
        - **5** = Digunakan dalam satu minggu terakhir
        - **6** = Digunakan dalam satu hari terakhir
        
        Pola penggunaan zat dapat menjadi indikator status kesehatan mental dan mekanisme koping.
        """)
    
    with st.expander("🤔 Apa yang harus saya lakukan dengan hasilnya?"):
        st.write("""
        - **Risiko Rendah**: Terus jaga pola hidup sehat
        - **Risiko Sedang**: Pertimbangkan untuk berkonsultasi dengan profesional kesehatan mental
        - **Risiko Tinggi**: Sangat disarankan untuk segera mencari bantuan profesional
        
        **Penting:** Ini hanya alat skrining. Konsultasikan dengan profesional berkualifikasi untuk diagnosis dan pengobatan.
        """)
    
    with st.expander("🔒 Bagaimana data saya ditangani?"):
        st.write("""
        - Semua prediksi dilakukan secara lokal di aplikasi ini
        - Tidak ada data yang dikirim ke server eksternal
        - Hasil tidak disimpan secara permanen
        - Aplikasi ini untuk tujuan pendidikan dan penelitian
        """)
    
    st.divider()
    st.markdown(f"""
    ### 📚 Pengembangan Model
    - **Seleksi Fitur**: Moth-Flame Optimization (MFO) — {len(config['selected_features'])} fitur terpilih dari 37
    - **Tuning Hyperparameter**: RandomizedSearchCV (agregasi per-target)
    - **Algoritma**: XGBoost dengan ClassifierChain untuk klasifikasi multi-label
    - **Kalibrasi Ambang Batas**: Dioptimasi pada data validasi asli (tanpa SMOTE)
    - **Validasi**: 5-Fold Cross-Validation pada data SMOTE
    - **Data Latih**: Diseimbangkan dengan MLSMOTE
    
    ### ⚠️ Peringatan
    Aplikasi ini hanya untuk tujuan pendidikan dan penelitian.
    Tidak boleh digunakan untuk diagnosis klinis atau keputusan pengobatan.
    Selalu konsultasikan dengan profesional kesehatan mental yang berkualifikasi.
    """)

# Footer
st.divider()
st.markdown("""
---
**Sistem Penilaian Risiko Kesehatan Mental** | Universitas Amikom | Proyek Data Mining 2026  
*Dibuat dengan Streamlit, XGBoost, dan Scikit-Learn*
""", unsafe_allow_html=True)
