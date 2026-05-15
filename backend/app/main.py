from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import warnings
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

# 2. Load Config (Optimal Thresholds and Selected Features)
threshold_path = BASE_DIR / "models" / "optimal_thresholds.json"
try:
    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)
        
    config = {
        "selected_features": threshold_data["selected_features"],
        "target_cols": ['risk_depression', 'risk_anxiety', 'risk_stress'],
        "optimal_thresholds": threshold_data["thresholds"],
    }
except Exception as e:
    print(f"Error loading config: {e}")
    config = None

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

def predict_with_thresholds(model, X, thresholds):
    probas = chain_predict_proba(model, X)
    predictions = []
    for i, (proba, threshold) in enumerate(zip(probas, thresholds)):
        pred = (proba[:, 1] >= threshold).astype(int)
        predictions.append(pred)
    return np.column_stack(predictions), probas

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
        
        # Get threshold values in order
        thresholds_list = [config['optimal_thresholds'][t] for t in config['target_cols']]
        
        # Predict
        predictions, probabilities = predict_with_thresholds(model, input_df, thresholds_list)
        
        dep_prob = float(probabilities[0][0, 1])
        anx_prob = float(probabilities[1][0, 1])
        str_prob = float(probabilities[2][0, 1])

        return {
            "depression": {
                "prob": dep_prob,
                "risk": get_risk_level(dep_prob)
            },
            "anxiety": {
                "prob": anx_prob,
                "risk": get_risk_level(anx_prob)
            },
            "stress": {
                "prob": str_prob,
                "risk": get_risk_level(str_prob)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Mental Health Risk Assessment API is running. Send POST request to /api/predict"}
