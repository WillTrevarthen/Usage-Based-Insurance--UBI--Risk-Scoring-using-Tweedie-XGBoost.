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
from sklearn.cluster import KMeans

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
    model = joblib.load("tuned_tweedie_ensemble_model.joblib")
    
    X_test, y_test, w_test = data['test']
    preprocessor = data['preprocessor']
    
    # 2. Generate Predictions
    X_test_transformed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_transformed)
    
    # 3. Calculate Gini
    # Convert Rate back to Absolute Loss for validation (Rate * Exposure)
    actual_loss = y_test * w_test
    pred_loss = y_pred * w_test
    
    gini_score, cum_actual, cum_weight = gini_coefficient(actual_loss, pred_loss, w_test)
    print(f"\n--- Actuarial Performance ---")
    print(f"Model Gini Coefficient: {gini_score:.4f}")

    # 4. Risk Segmentation (Business Impact)
    # We add 'pred_rate' (y_pred) to use for segmentation sorting (Riskiness independent of duration)
    results_df = pd.DataFrame({'actual': actual_loss, 'pred': pred_loss, 'exposure': w_test, 'pred_rate': y_pred})
    
    # Method A: Quantiles (Equal Volume) - Standard for Statistical Validation
    results_df['risk_group_quantile'] = pd.qcut(results_df['pred_rate'].rank(method='first'), 3, labels=['Low', 'Medium', 'High'])
    
    # Method B: K-Means (Natural Clusters) - Better for Pricing/Business Logic
    # Finds natural separation points (e.g., a small group of very risky drivers)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    results_df['cluster'] = kmeans.fit_predict(results_df[['pred_rate']])
    
    # Sort clusters by average risk so 0 is Low and 2 is High
    cluster_map = results_df.groupby('cluster')['pred_rate'].mean().sort_values().index
    mapping = {cluster_map[0]: 'Low', cluster_map[1]: 'Medium', cluster_map[2]: 'High'}
    results_df['risk_group_kmeans'] = results_df['cluster'].map(mapping)

    print("\n--- Segmentation Comparison: Loss Ratios ---")
    print("\nMethod A: Equal-Sized Groups (Quantiles)")
    summary_q = results_df.groupby('risk_group_quantile').agg({'actual': 'sum', 'exposure': 'sum', 'pred_rate': 'mean'})
    summary_q['Loss_Ratio'] = summary_q['actual'] / summary_q['exposure']
    print(summary_q[['pred_rate', 'Loss_Ratio']])
    
    print("\nMethod B: Natural Clustering (K-Means)")
    summary_k = results_df.groupby('risk_group_kmeans').agg({'actual': 'sum', 'exposure': 'sum', 'pred_rate': 'mean'})
    summary_k['Loss_Ratio'] = summary_k['actual'] / summary_k['exposure']
    print(summary_k.loc[['Low', 'Medium', 'High']][['pred_rate', 'Loss_Ratio']])

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