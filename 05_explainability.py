# --- ADD THIS CLASS TO THE TOP OF app.py ---
from sklearn.base import BaseEstimator, RegressorMixin

class TweedieEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, xgb_model, lgb_model, cat_model, weights=(0.34, 0.33, 0.33)):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.weights = weights

    def predict(self, X):
        pred_xgb = self.xgb_model.predict(X)
        pred_lgb = self.lgb_model.predict(X)
        pred_cat = self.cat_model.predict(X)
        return (self.weights[0] * pred_xgb) + (self.weights[1] * pred_lgb) + (self.weights[2] * pred_cat)
# -------------------------------------------


import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_model(model_path, data_path):
    # 1. Load Model and Data
    model = joblib.load(model_path)
    data = joblib.load(data_path)
    
    X_test, _, _ = data['test']
    preprocessor = data['preprocessor']
    
    # 2. Transform Data
    print("Transforming features for SHAP...")
    X_test_transformed = preprocessor.transform(X_test)
    
    # 3. Dynamic Feature Name Reconstruction
    # MUST match the order in Step 02: num -> cat -> bool
    num_names = ['VehPower', 'VehAge', 'DrivAge', 'LogDensity', 'Power_Age_Ratio']
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
    bool_names = ['Young_Urban', 'Is_New_Car']
    all_feature_names = num_names + cat_names + bool_names

    # Verify length match
    if X_test_transformed.shape[1] != len(all_feature_names):
        print(f"Warning: Name mismatch! Data has {X_test_transformed.shape[1]} cols, but we have {len(all_feature_names)} names.")

    # 4. The Explainer
    print("Initializing SHAP Explainer...")
    background_data = shap.sample(X_test_transformed, 100)
    # Using the .predict method ensures we stay in 'Euro' units for the output
    explainer = shap.Explainer(model.predict, background_data)
    
    # Calculate SHAP values for a sample
    print("Calculating SHAP values (subset for speed)...")
    test_samples = X_test_transformed[:200]
    shap_values = explainer(test_samples)
    
    # Assign the reconstructed names
    shap_values.feature_names = all_feature_names

    # 5. Visualization 1: Global Importance
    print("Generating Beeswarm Plot...")
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("Top Risk Drivers (Global SHAP Impact)")
    plt.tight_layout()
    plt.show()

    # 6. Visualization 2: Individual Risk (Waterfall)
    print("\nVisualizing Risk for High-Risk Case (Sample 0)...")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0])

if __name__ == "__main__":
    explain_model("tweedie_model.joblib", "processed_step_2.pkl")