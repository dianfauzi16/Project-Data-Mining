from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import pickle
import json
import yaml
from pathlib import Path
import warnings
import shap
warnings.filterwarnings('ignore')

app = FastAPI(title="Sistem Penilaian Risiko Kesehatan Mental API")

# Add CORS middleware to allow Next.js frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent

# 1. Load Model
model_path = BASE_DIR / "models" / "multilabel_xgboost_classifier_chain.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# 2. Load Config (Selected Features)
# ClassifierChain training strips feature_names_in_, so we define them explicitly based on the 15 features used
selected_features = [
    'TIPI1', 'TIPI2', 'TIPI3', 'TIPI4', 'TIPI5', 'TIPI6', 'TIPI7', 'TIPI8', 
    'TIPI9', 'TIPI10', 'education', 'urban', 'gender', 'age', 'familysize'
]

config = {
    "selected_features": selected_features,
    "target_cols": ['risk_depression', 'risk_anxiety', 'risk_stress'],
    "optimal_thresholds": {'risk_depression': 0.5, 'risk_anxiety': 0.5, 'risk_stress': 0.5},
}

# 3. Helper functions for Prediction
def chain_predict_proba(model, X):
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

def get_shap_explanations(model, input_df, target_cols):
    X_sample_aug = input_df.values.copy()
    
    explanations = {}
    base_features = list(input_df.columns)
    
    feature_translations = {
    "TIPI1": "Ekstrovert / Antusias",
    "TIPI2": "Kritis / Suka Berdebat",
    "TIPI3": "Dapat Diandalkan / Disiplin",
    "TIPI4": "Mudah Cemas / Mudah Terganggu",
    "TIPI5": "Terbuka pada Pengalaman Baru",
    "TIPI6": "Pendiam / Tertutup",
    "TIPI7": "Simpatik / Hangat",
    "TIPI8": "Ceroboh / Tidak Terorganisir",
    "TIPI9": "Tenang / Stabil Emosional",
    "TIPI10": "Konvensional / Kurang Kreatif",

    "education": "Tingkat Pendidikan",
    "urban": "Tingkat Urbanisasi",
    "gender": "Jenis Kelamin",
    "age": "Usia",
    "familysize": "Jumlah Anggota Keluarga",

    "pred_depression": "Risiko Depresi",
    "pred_anxiety": "Risiko Kecemasan",
}
    
    for i, target in enumerate(target_cols):
        estimator = model.estimators_[i]
        
        current_feature_names = base_features.copy()
        if i == 1:
            current_feature_names += ['pred_depression']
        elif i == 2:
            current_feature_names += ['pred_depression', 'pred_anxiety']
            
        X_shap_df = pd.DataFrame(X_sample_aug, columns=current_feature_names)
        
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_shap_df)
        
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        if len(sv.shape) > 1:
            sv = sv[0]
            
        top_indices = np.argsort(np.abs(sv))[::-1][:5]
        
        shap_explanation = []
        for idx in top_indices:
            raw_feat_name = current_feature_names[idx]
            impact_val = float(sv[idx])
            
            human_readable_name = feature_translations.get(raw_feat_name, raw_feat_name)
            direction = "meningkatkan_risiko" if impact_val > 0 else "menurunkan_risiko"
            
            shap_explanation.append({
                "feature": human_readable_name,
                "impact_value": impact_val,
                "type": direction
            })
            
        key_name = target.replace("risk_", "")
        explanations[key_name] = shap_explanation
        
        pred_proba = estimator.predict_proba(X_sample_aug)[:, 1]
        pred_label = (pred_proba >= 0.5).astype(int).reshape(-1, 1)
        X_sample_aug = np.hstack([X_sample_aug, pred_label])
        
    return explanations
def get_risk_level(prob):
    if prob >= 0.7: return "high"
    elif prob >= 0.4: return "medium"
    return "low"

@app.get("/api/config")
def get_config():
    """Return the selected features and configuration for dynamic frontend generation"""
    if config is None:
        raise HTTPException(status_code=500, detail="Config not loaded properly on the server.")
    return config

@app.post("/api/predict")
def predict_mental_health(data: Dict[str, Any]):
    if model is None or config is None:
        raise HTTPException(status_code=500, detail="Model or config not loaded properly on the server.")
        
    try:
        # data is a dict containing dynamic features
        input_dict = data
        
        # Build dataframe with exactly the selected features in the right order
        missing_features = [f for f in config['selected_features'] if f not in input_dict]
        if missing_features:
            raise ValueError(f"Missing features in request: {missing_features}")

        input_df = pd.DataFrame([input_dict])[config['selected_features']]
        
        # Predict (using custom chain_predict_proba for ClassifierChain)
        probabilities = chain_predict_proba(model, input_df)
        
        # Calculate SHAP local explanations
        shap_explanations = get_shap_explanations(model, input_df, config['target_cols'])
        
        dep_prob = float(probabilities[0][0, 1])
        anx_prob = float(probabilities[1][0, 1])
        str_prob = float(probabilities[2][0, 1])

        return {
            "depression": {
                "prob": dep_prob,
                "risk": get_risk_level(dep_prob),
                "shap_explanation": shap_explanations["depression"]
            },
            "anxiety": {
                "prob": anx_prob,
                "risk": get_risk_level(anx_prob),
                "shap_explanation": shap_explanations["anxiety"]
            },
            "stress": {
                "prob": str_prob,
                "risk": get_risk_level(str_prob),
                "shap_explanation": shap_explanations["stress"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Mental Health Risk Assessment API is running. Send POST request to /api/predict"}
