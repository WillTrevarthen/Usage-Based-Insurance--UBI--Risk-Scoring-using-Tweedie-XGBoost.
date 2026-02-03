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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gini_coefficient(actual, pred, weight):
    # Sort by predicted risk (ascending)
    idx = np.argsort(pred)
    actual_sorted = actual.iloc[idx]
    weight_sorted = weight.iloc[idx]

    # Cumulative sums
    cum_actual = np.cumsum(actual_sorted) / np.sum(actual_sorted)
    cum_weight = np.cumsum(weight_sorted) / np.sum(weight_sorted)

    # Gini is 2 * Area between Lorenz curve and 45-degree line
    gini = 1 - 2 * np.trapz(cum_actual, cum_weight)
    return gini, cum_actual, cum_weight

def plot_lift_chart(df_results, bins=10):
    # Create deciles based on predicted risk
    df_results['decile'] = pd.qcut(df_results['pred'], bins, labels=False)
    
    # Aggregate actual vs predicted by decile
    lift = df_results.groupby('decile').agg({
        'actual': 'sum',
        'pred': 'sum',
        'exposure': 'sum'
    })
    
    lift['actual_pp'] = lift['actual'] / lift['exposure']
    lift['pred_pp'] = lift['pred'] / lift['exposure']

    plt.figure(figsize=(10, 6))
    plt.plot(lift.index, lift['actual_pp'], marker='o', label='Actual Pure Premium', color='navy')
    plt.plot(lift.index, lift['pred_pp'], marker='x', linestyle='--', label='Predicted Pure Premium', color='red')
    plt.title("Lift Chart: Predicted vs. Actual by Risk Decile")
    plt.xlabel("Risk Decile (0=Low, 9=High)")
    plt.ylabel("Pure Premium")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # 1. Load data and model
    data = joblib.load("processed_step_2.pkl")
    model = joblib.load("tweedie_model.joblib")
    
    X_test, y_test, w_test = data['test']
    preprocessor = data['preprocessor']
    
    # 2. Generate Predictions
    X_test_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_transformed)
    
    # 3. Calculate Gini
    gini_score, cum_actual, cum_weight = gini_coefficient(y_test, y_pred, w_test)
    print(f"\n--- Actuarial Performance ---")
    print(f"Model Gini Coefficient: {gini_score:.4f}")

    # 4. Risk Segmentation (Business Impact)
    results_df = pd.DataFrame({'actual': y_test, 'pred': y_pred, 'exposure': w_test})
    results_df['risk_group'] = pd.qcut(results_df['pred'].rank(method='first'), 3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    summary = results_df.groupby('risk_group').agg({'actual': 'sum', 'exposure': 'sum'})
    summary['Loss_Ratio_Per_Year'] = summary['actual'] / summary['exposure']
    
    print("\n--- Business Impact: Risk Segmentation ---")
    print(summary[['Loss_Ratio_Per_Year']])

    # 5. Visualizations
    plt.figure(figsize=(8, 8))
    plt.plot(cum_weight, cum_actual, label=f'Model (Gini: {gini_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random (Pure Luck)')
    plt.title("Ordered Lorenz Curve")
    plt.xlabel("Cumulative Share of Exposure")
    plt.ylabel("Cumulative Share of Claims")
    plt.legend()
    plt.show()

    plot_lift_chart(results_df)