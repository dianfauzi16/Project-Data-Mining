"""
Utility functions for Streamlit deployment
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union


class StreamlitModelWrapper:
    """Wrapper for model inference with helper methods"""
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize model wrapper
        
        Args:
            model_path: Path to pickled model
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
    
    def _load_model(self):
        """Load model from pickle"""
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)
    
    def chain_predict_proba(self, X: pd.DataFrame) -> List[np.ndarray]:
        """
        Reconstruct predict_proba for ClassifierChain with XGBoost 2.1.1
        
        Args:
            X: Input features DataFrame
            
        Returns:
            List of probability arrays for each target
        """
        X_arr = X.values if hasattr(X, 'values') else X.copy()
        n_samples = X_arr.shape[0]
        n_targets = len(self.model.estimators_)
        all_probas = np.zeros((n_samples, n_targets))
        
        X_aug = X_arr.copy()
        for i, estimator in enumerate(self.model.estimators_):
            proba = estimator.predict_proba(X_aug)[:, 1]
            all_probas[:, i] = proba
            pred_label = (proba >= 0.5).astype(int).reshape(-1, 1)
            X_aug = np.hstack([X_aug, pred_label])
        
        return [np.column_stack([1 - all_probas[:, i], all_probas[:, i]])
                for i in range(n_targets)]
    
    def predict_with_thresholds(
        self, 
        X: pd.DataFrame, 
        thresholds: List[float]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Make predictions with custom thresholds
        
        Args:
            X: Input features DataFrame
            thresholds: List of threshold values for each target
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        probas = self.chain_predict_proba(X)
        predictions = []
        for i, (proba, threshold) in enumerate(zip(probas, thresholds)):
            pred = (proba[:, 1] >= threshold).astype(int)
            predictions.append(pred)
        return np.column_stack(predictions), probas
    
    def predict_proba(self, X: pd.DataFrame) -> List[np.ndarray]:
        """Get probability predictions"""
        return self.chain_predict_proba(X)
    
    def predict(self, X: pd.DataFrame, thresholds: List[float] = None) -> np.ndarray:
        """Get hard predictions"""
        if thresholds is None:
            thresholds = [0.5] * len(self.model.estimators_)
        predictions, _ = self.predict_with_thresholds(X, thresholds)
        return predictions


class DataPreprocessor:
    """Data preprocessing utilities for inference"""
    
    def __init__(self, feature_order: List[str]):
        """
        Initialize preprocessor
        
        Args:
            feature_order: Expected order of features
        """
        self.feature_order = feature_order
    
    def prepare_input(self, input_dict: Dict) -> pd.DataFrame:
        """
        Prepare input dictionary as DataFrame with correct feature order
        
        Args:
            input_dict: Dictionary of input values
            
        Returns:
            DataFrame with features in correct order
        """
        input_df = pd.DataFrame([input_dict])
        return input_df[self.feature_order]
    
    def validate_input(self, input_dict: Dict) -> Tuple[bool, str]:
        """
        Validate input data
        
        Args:
            input_dict: Dictionary of input values
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check all required features present
        missing_features = set(self.feature_order) - set(input_dict.keys())
        if missing_features:
            return False, f"Missing features: {missing_features}"
        
        # Check age range
        if not (15 <= input_dict.get('age', 0) <= 24):
            return False, "Age must be between 15 and 24"
        
        # Check TIPI values (1-7)
        tipi_keys = [k for k in input_dict.keys() if k.startswith('TIPI')]
        for key in tipi_keys:
            if not (1 <= input_dict[key] <= 7):
                return False, f"{key} must be between 1 and 7"
        
        # Check VCL values (0-6)
        vcl_keys = [k for k in input_dict.keys() if k.startswith('VCL')]
        for key in vcl_keys:
            if not (0 <= input_dict[key] <= 6):
                return False, f"{key} must be between 0 and 6"
        
        return True, ""


class RiskAssessment:
    """Risk assessment and interpretation utilities"""
    
    @staticmethod
    def get_risk_level(probability: float) -> str:
        """
        Classify risk level based on probability
        
        Args:
            probability: Probability value (0-1)
            
        Returns:
            Risk level string
        """
        if probability >= 0.7:
            return "🔴 HIGH RISK"
        elif probability >= 0.4:
            return "🟡 MEDIUM RISK"
        else:
            return "🟢 LOW RISK"
    
    @staticmethod
    def get_risk_emoji(probability: float) -> str:
        """Get emoji for risk level"""
        if probability >= 0.7:
            return "🔴"
        elif probability >= 0.4:
            return "🟡"
        else:
            return "🟢"
    
    @staticmethod
    def get_recommendation(probabilities: Dict[str, float]) -> str:
        """
        Get recommendation based on risk probabilities
        
        Args:
            probabilities: Dict of target -> probability
            
        Returns:
            Recommendation string
        """
        avg_risk = np.mean(list(probabilities.values()))
        
        if avg_risk >= 0.7:
            return """
            🔴 **URGENT**: Multiple high-risk factors detected. 
            Strongly recommend immediate consultation with mental health professional.
            """
        elif avg_risk >= 0.4:
            return """
            🟡 **MODERATE**: Some risk factors detected.
            Consider professional consultation for proper assessment.
            """
        else:
            return """
            🟢 **LOW**: Minimal risk factors detected.
            Continue maintaining healthy lifestyle practices.
            """
    
    @staticmethod
    def generate_report(
        predictions: np.ndarray,
        probabilities: List[np.ndarray],
        target_names: List[str],
        optimal_thresholds: Dict[str, float]
    ) -> Dict:
        """
        Generate comprehensive assessment report
        
        Args:
            predictions: Binary predictions
            probabilities: Probability predictions
            target_names: Names of targets
            optimal_thresholds: Optimal threshold values
            
        Returns:
            Report dictionary
        """
        report = {
            'predictions': predictions[0].tolist(),
            'probabilities': {},
            'risk_levels': {},
            'recommendations': {}
        }
        
        for i, target in enumerate(target_names):
            prob = probabilities[i][0, 1]
            report['probabilities'][target] = float(prob)
            report['risk_levels'][target] = RiskAssessment.get_risk_level(prob)
            report['recommendations'][target] = (
                "At-risk" if predictions[0, i] == 1 else "Low-risk"
            )
        
        report['total_risk_factors'] = int(predictions[0].sum())
        report['average_risk_probability'] = float(np.mean(list(report['probabilities'].values())))
        
        return report


# Convenience functions
def load_model_wrapper(model_path: Union[str, Path]) -> StreamlitModelWrapper:
    """Load model wrapper"""
    return StreamlitModelWrapper(model_path)


def create_preprocessor(feature_order: List[str]) -> DataPreprocessor:
    """Create data preprocessor"""
    return DataPreprocessor(feature_order)


def assess_risk(probability: float) -> str:
    """Quick risk assessment"""
    return RiskAssessment.get_risk_level(probability)
