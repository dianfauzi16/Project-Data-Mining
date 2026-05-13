"""
Mental Health Risk Assessment System
Streamlit Application for Multi-Label Depression, Anxiety & Stress Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Mental Health Risk Assessment",
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
# 2. LOAD MODEL & CONFIG
# ==========================================
@st.cache_resource
def load_model():
    """Load trained XGBoost classifier chain model"""
    model_path = Path(__file__).parent.parent / "models" / "multilabel_xgboost_classifier_chain.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_config():
    """Load configuration and feature information"""
    config = {
        "selected_features": [
            'TIPI1', 'TIPI2', 'TIPI3', 'TIPI4', 'TIPI5', 'TIPI6', 'TIPI7', 'TIPI8', 'TIPI9', 'TIPI10',
            'VCL1', 'VCL4', 'VCL5', 'VCL6', 'VCL7', 'VCL8', 'VCL9', 'VCL10', 'VCL12', 'VCL13',
            'VCL14', 'VCL15', 'gender', 'engnat', 'age', 'religion', 'orientation',
            'race', 'voted', 'married', 'familysize'
        ],
        "target_cols": ['risk_depression', 'risk_anxiety', 'risk_stress'],
        "optimal_thresholds": {
            'risk_depression': 0.41,
            'risk_anxiety': 0.37,
            'risk_stress': 0.38
        },
        "personality_features": {
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
        },
        "lifestyle_features": {
            'VCL1': 'Alcohol use',
            'VCL4': 'Cannabis use',
            'VCL5': 'Cocaine use',
            'VCL6': 'Heroin use',
            'VCL7': 'Amphetamine use',
            'VCL8': 'Ecstasy use',
            'VCL9': 'LSD use',
            'VCL10': 'Mushrooms use',
            'VCL12': 'Nicotine use',
            'VCL13': 'Methadone use',
            'VCL14': 'VSA use',
            'VCL15': 'PCP use'
        }
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
    """Reconstruct predict_proba for ClassifierChain with XGBoost 2.1.1"""
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

# ==========================================
# 4. MAIN APP LAYOUT
# ==========================================
st.markdown('<div class="main-header">🧠 Mental Health Risk Assessment System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict Depression, Anxiety & Stress Risk Based on Personality & Lifestyle</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Info", "❓ Help"])

with tab1:
    st.header("Input Your Information")
    
    # Create input sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Demographic Information")
        age = st.slider("Age", min_value=15, max_value=24, value=20)
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        engnat = st.selectbox("English is Native Language", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        religion = st.selectbox("Religion", [0, 1, 2, 3, 4, 5, 6], 
                               format_func=lambda x: ["Agnostic", "Atheist", "Buddhist", "Christian", "Hindu", "Jewish", "Muslim"][x])
        orientation = st.selectbox("Sexual Orientation", [0, 1, 2], 
                                  format_func=lambda x: ["Heterosexual", "Bisexual", "Homosexual"][x])
    
    with col2:
        st.subheader("👥 Social Information")
        race = st.selectbox("Race/Ethnicity", [0, 1, 2, 3, 4, 5],
                           format_func=lambda x: ["White", "Black", "Asian", "Mixed", "Other", "Latino"][x])
        voted = st.selectbox("Political Vote", [0, 1, 2, 3],
                            format_func=lambda x: ["No", "Yes", "Spoilt", "N/A"][x])
        married = st.selectbox("Marital Status", [0, 1, 2, 3, 4],
                              format_func=lambda x: ["Single", "Married", "Divorced", "Widowed", "Separated"][x])
        familysize = st.number_input("Family Size", min_value=0, max_value=10, value=3)
    
    st.divider()
    
    # Personality traits (TIPI)
    st.subheader("🎭 Personality Traits (TIPI-10)")
    st.caption("Skala 1-7 dimana 1=Sangat Tidak Setuju, 7=Sangat Setuju")
    
    tipi_descriptions = {
        'TIPI1': '📊 Saya adalah orang yang riang, ekstrovert\n💡 (Sebaliknya: Reserved/Tertutup)',
        'TIPI2': '⚖️ Saya adalah orang yang kritis, suka berdebat\n💡 (Sebaliknya: Pemaaf/Lembut)',
        'TIPI3': '📋 Saya adalah orang yang dapat diandalkan, teratur\n💡 (Sebaliknya: Ceroboh/Tidak terorganisir)',
        'TIPI4': '😟 Saya mudah cemas, terguncang emosionalnya\n💡 (Sebaliknya: Tenang/Stabil)',
        'TIPI5': '🎨 Saya terbuka terhadap pengalaman baru, kreatif\n💡 (Sebaliknya: Konvensional/Tradisional)',
        'TIPI6': '🤐 Saya adalah orang yang pendiam, tertutup\n💡 (Sebaliknya: Ekstrovert/Sosial)',
        'TIPI7': '❤️ Saya adalah orang yang simpatik, hangat\n💡 (Sebaliknya: Kritis/Tidak peduli)',
        'TIPI8': '🗑️ Saya adalah orang yang tidak terorganisir, ceroboh\n💡 (Sebaliknya: Rapi/Teratur)',
        'TIPI9': '😊 Saya adalah orang yang tenang, stabil emosional\n💡 (Sebaliknya: Mudah cemas)',
        'TIPI10': '🔄 Saya adalah orang yang konvensional, tidak kreatif\n💡 (Sebaliknya: Terbuka/Inovatif)'
    }
    
    tipi_inputs = {}
    tipi_cols = st.columns(5)
    tipi_keys = ['TIPI1', 'TIPI2', 'TIPI3', 'TIPI4', 'TIPI5', 'TIPI6', 'TIPI7', 'TIPI8', 'TIPI9', 'TIPI10']
    
    for idx, key in enumerate(tipi_keys):
        col = tipi_cols[idx % 5]
        with col:
            st.write(f"**{key}**")
            st.caption(tipi_descriptions[key])
            tipi_inputs[key] = st.slider(
                f"Nilai {key}",
                min_value=1,
                max_value=7,
                value=4,
                label_visibility="collapsed"
            )
    
    st.divider()
    
    # Lifestyle/Substance use (VCL)
    st.subheader("💊 Lifestyle & Substance Use (VCL)")
    st.caption("Skala 0-6: 0=Tidak Pernah, 1=>10 Tahun Lalu, 2=Dekade Terakhir, 3=Tahun Terakhir, 4=Bulan Terakhir, 5=Minggu Terakhir, 6=Hari Terakhir")
    
    vcl_descriptions = {
        'VCL1': '🍺 Konsumsi Alkohol\n💡 Estimasi frekuensi penggunaan alkohol dalam hidup Anda',
        'VCL4': '🌿 Penggunaan Cannabis\n💡 Estimasi frekuensi penggunaan ganja/marijuana dalam hidup Anda',
        'VCL5': '⚪ Penggunaan Kokain\n💡 Estimasi frekuensi penggunaan kokain dalam hidup Anda',
        'VCL6': '💉 Penggunaan Heroin\n💡 Estimasi frekuensi penggunaan heroin dalam hidup Anda',
        'VCL7': '⚡ Penggunaan Amfetamin\n💡 Estimasi frekuensi penggunaan amfetamin/speed dalam hidup Anda',
        'VCL8': '💊 Penggunaan Ecstasy\n💡 Estimasi frekuensi penggunaan ecstasy/MDMA dalam hidup Anda',
        'VCL9': '🌈 Penggunaan LSD\n💡 Estimasi frekuensi penggunaan LSD/asam dalam hidup Anda',
        'VCL10': '🍄 Penggunaan Psilocybin (Jamur)\n💡 Estimasi frekuensi penggunaan jamur psilocybin dalam hidup Anda',
        'VCL12': '🚬 Penggunaan Tembakau/Nikotin\n💡 Estimasi frekuensi penggunaan rokok/nikotin dalam hidup Anda',
        'VCL13': '⚕️ Penggunaan Metadon\n💡 Estimasi frekuensi penggunaan metadon dalam hidup Anda',
        'VCL14': '⚠️ Penggunaan VSA (Volatile Substance Abuse)\n💡 Inhalasi zat volatil seperti solvent, glue, atau gas',
        'VCL15': '🔷 Penggunaan PCP\n💡 Estimasi frekuensi penggunaan PCP dalam hidup Anda'
    }
    
    vcl_inputs = {}
    vcl_cols = st.columns(3)
    vcl_keys = ['VCL1', 'VCL4', 'VCL5', 'VCL6', 'VCL7', 'VCL8', 'VCL9', 'VCL10', 'VCL12', 'VCL13', 'VCL14', 'VCL15']
    
    for idx, key in enumerate(vcl_keys):
        col = vcl_cols[idx % 3]
        with col:
            st.write(f"**{key}**")
            st.caption(vcl_descriptions[key])
            vcl_inputs[key] = st.slider(
                f"Frekuensi {key}",
                min_value=0,
                max_value=6,
                value=0,
                label_visibility="collapsed"
            )
    
    # Prediction button
    st.divider()
    
    if st.button("🚀 Get Prediction", use_container_width=True, type="primary"):
        # Prepare input data
        input_dict = {
            'age': age,
            'gender': gender,
            'engnat': engnat,
            'religion': religion,
            'orientation': orientation,
            'race': race,
            'voted': voted,
            'married': married,
            'familysize': familysize,
            **tipi_inputs,
            **vcl_inputs
        }
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_dict])[config['selected_features']]
        
        # Make prediction
        try:
            predictions, probabilities = predict_with_thresholds(
                model, 
                input_df, 
                [config['optimal_thresholds'][target] for target in config['target_cols']]
            )
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            st.markdown("## 📈 Results")
            result_cols = st.columns(3)
            
            target_labels = {
                'risk_depression': '😔 Depression',
                'risk_anxiety': '😰 Anxiety',
                'risk_stress': '😣 Stress'
            }
            
            for idx, (target, label) in enumerate(target_labels.items()):
                with result_cols[idx]:
                    prob = probabilities[idx][0, 1]
                    pred = predictions[0, idx]
                    risk_level = get_risk_level(prob)
                    risk_class = get_risk_color(prob)
                    
                    st.markdown(f"""
                    <div class="{risk_class}">
                    <h3>{label}</h3>
                    <p><b>Probability:</b> {prob:.1%}</p>
                    <p><b>Risk Level:</b> {risk_level}</p>
                    <p><b>Prediction:</b> {'🔴 Risk Present' if pred == 1 else '🟢 No Risk'}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Summary
            st.markdown("---")
            risk_count = sum(predictions[0])
            st.subheader("📋 Summary")
            st.info(f"**Overall Assessment:** {risk_count} out of 3 risk factors detected")
            
            if risk_count == 0:
                st.success("✅ Low overall risk. Continue maintaining healthy lifestyle.")
            elif risk_count == 1:
                st.warning("⚠️ One risk factor detected. Consider professional consultation.")
            else:
                st.error("🔴 Multiple risk factors detected. Strongly recommend consulting with a mental health professional.")
            
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
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Algorithm", "XGBoost + ClassifierChain")
        st.metric("Features Used", len(config['selected_features']))
        st.metric("Targets", len(config['target_cols']))
    
    with col2:
        st.metric("Model Type", "Multi-Label Classification")
        st.metric("Validation Method", "5-Fold Cross-Validation")
        st.metric("Feature Selection", "Moth-Flame Optimization (MFO)")
    
    st.divider()
    
    st.subheader("📌 Selected Features")
    
    # Personality features
    st.markdown("#### 🎭 Personality Traits (TIPI-10)")
    personality_text = ", ".join([f"{k}" for k in sorted(config['selected_features']) if k.startswith('TIPI')])
    st.text(personality_text)
    
    # Lifestyle features
    st.markdown("#### 💊 Substance Use (VCL)")
    vcl_text = ", ".join([f"{k}" for k in sorted(config['selected_features']) if k.startswith('VCL')])
    st.text(vcl_text)
    
    # Demographic features
    st.markdown("#### 👥 Demographics")
    demo_features = [f for f in config['selected_features'] if not f.startswith(('TIPI', 'VCL'))]
    demo_text = ", ".join(sorted(demo_features))
    st.text(demo_text)
    
    st.divider()
    
    st.subheader("🎯 Optimal Thresholds")
    threshold_data = pd.DataFrame([
        {
            'Risk Factor': target.replace('risk_', '').title(),
            'Threshold': threshold
        }
        for target, threshold in config['optimal_thresholds'].items()
    ])
    st.table(threshold_data)
    
    st.info("""
    **Note:** Thresholds were optimized using 5-fold cross-validation to maximize F1-Score on validation set.
    Different thresholds apply to each target due to their different class distributions.
    """)

with tab3:
    st.header("❓ Help & FAQ")
    
    with st.expander("What is this application?"):
        st.write("""
        This is a Mental Health Risk Assessment System that predicts the likelihood of Depression, Anxiety, 
        and Stress based on personality traits (TIPI-10) and lifestyle/substance use (VCL) information.
        
        The system uses machine learning to identify individuals at risk and can help with:
        - Early mental health risk detection
        - Screening for interventions
        - Understanding personality-mental health relationships
        """)
    
    with st.expander("How accurate is this model?"):
        st.write("""
        The model achieved:
        - **F1-Score**: ~0.75-0.80 per target
        - **5-Fold Cross-Validation**: Robust performance across different data splits
        - **Validation Method**: Stratified train-test split with 80-20 ratio
        
        However, this model should NOT replace professional mental health assessment.
        Always consult with a qualified mental health professional for diagnosis.
        """)
    
    with st.expander("What do the TIPI questions measure?"):
        st.write("""
        TIPI-10 measures the Big Five personality traits:
        - **Extraversion** (outgoing vs. reserved)
        - **Agreeableness** (sympathetic vs. critical)
        - **Conscientiousness** (dependable vs. disorganized)
        - **Neuroticism** (anxious vs. calm)
        - **Openness** (open to new experiences vs. conventional)
        
        These personality traits are known to correlate with mental health conditions.
        """)
    
    with st.expander("What do the VCL scores mean?"):
        st.write("""
        VCL (Substance Use) scoring:
        - **0** = Never Used
        - **1** = Used > 10 years ago
        - **2** = Used in last decade
        - **3** = Used in last year
        - **4** = Used in last month
        - **5** = Used in last week
        - **6** = Used in last day
        
        Substance use patterns can be indicators of mental health status and coping mechanisms.
        """)
    
    with st.expander("What should I do with the results?"):
        st.write("""
        - **Low Risk**: Continue maintaining healthy lifestyle practices
        - **Medium Risk**: Consider consulting with a mental health professional
        - **High Risk**: Strongly recommend seeking professional help immediately
        
        **Important:** This is a screening tool only. Please consult with qualified professionals for diagnosis and treatment.
        """)
    
    with st.expander("How is my data handled?"):
        st.write("""
        - All predictions are made locally in this application
        - No data is sent to external servers
        - Results are not stored permanently
        - This application is for educational and research purposes
        """)
    
    st.divider()
    st.markdown("""
    ### 📚 Model Development
    - **Feature Selection**: Moth-Flame Optimization (MFO) to select 35 most important features
    - **Algorithm**: XGBoost with ClassifierChain for multi-label classification
    - **Validation**: 5-Fold Cross-Validation, Stratified train-test split
    - **Training Data**: 15,016 participants (age 15-24), balanced with MLSMOTE
    
    ### ⚠️ Disclaimer
    This application is for educational and research purposes only. 
    It should not be used for clinical diagnosis or treatment decisions.
    Always consult with qualified mental health professionals.
    """)

# Footer
st.divider()
st.markdown("""
---
**Mental Health Risk Assessment System** | Amikom University | Data Mining Project 2026  
*Powered by Streamlit, XGBoost, and Scikit-Learn*
""", unsafe_allow_html=True)
