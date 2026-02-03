import joblib
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.base import BaseEstimator, RegressorMixin

# --- 1. Define the 3-Model Ensemble Wrapper ---
class TweedieEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, xgb_model, lgb_model, cat_model, weights=(0.34, 0.33, 0.33)):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.cat_model = cat_model
        self.weights = weights

    def predict(self, X):
        # Get predictions from all three
        pred_xgb = self.xgb_model.predict(X)
        pred_lgb = self.lgb_model.predict(X)
        pred_cat = self.cat_model.predict(X)
        
        # Weighted Average
        final_pred = (self.weights[0] * pred_xgb) + \
                     (self.weights[1] * pred_lgb) + \
                     (self.weights[2] * pred_cat)
        return final_pred

def train_ensemble_model(input_file):
    # 2. Load Data
    data = joblib.load(input_file)
    X_train, y_train, w_train = data['train']
    X_test, y_test, w_test = data['test']
    preprocessor = data['preprocessor']

    print("Transforming features...")
    # Standardize data for all models
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # --- 3. Train Model A: XGBoost ---
    print("\nTraining Model 1: XGBoost...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:tweedie',
        tweedie_variance_power=1.5,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        n_jobs=-1,
        random_state=42
    )
    xgb_model.fit(
        X_train_transformed, y_train,
        sample_weight=w_train,
        eval_set=[(X_test_transformed, y_test)],
        sample_weight_eval_set=[w_test],
        verbose=0  # Silent training to keep terminal clean
    )

    # --- 4. Train Model B: LightGBM ---
    print("Training Model 2: LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        objective='tweedie',
        tweedie_variance_power=1.5,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train_transformed, y_train,
        sample_weight=w_train,
        eval_set=[(X_test_transformed, y_test)],
        eval_metric='tweedie',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # --- 5. Train Model C: CatBoost ---
    print("Training Model 3: CatBoost...")
    # CatBoost expects the objective in a specific format
    cat_model = CatBoostRegressor(
        loss_function='Tweedie:variance_power=1.5',
        n_estimators=500,
        learning_rate=0.05,
        depth=6,
        silent=True,
        random_state=42,
        allow_writing_files=False
    )
    # CatBoost handles sparse matrices (from OneHotEncoder) automatically
    cat_model.fit(
        X_train_transformed, y_train,
        sample_weight=w_train,
        eval_set=(X_test_transformed, y_test),
        early_stopping_rounds=50
    )

    # --- 6. Ensemble Evaluation ---
    print("\nEvaluating Ensemble Strategy...")
    # Equal weighting to start (0.33, 0.33, 0.33)
    ensemble = TweedieEnsemble(xgb_model, lgb_model, cat_model, weights=(1/3, 1/3, 1/3))
    
    y_pred_ens = ensemble.predict(X_test_transformed)
    
    # Calculate Deviance for comparison
    dev_xgb = mean_tweedie_deviance(y_test, xgb_model.predict(X_test_transformed), power=1.5, sample_weight=w_test)
    dev_lgb = mean_tweedie_deviance(y_test, lgb_model.predict(X_test_transformed), power=1.5, sample_weight=w_test)
    dev_cat = mean_tweedie_deviance(y_test, cat_model.predict(X_test_transformed), power=1.5, sample_weight=w_test)
    dev_ens = mean_tweedie_deviance(y_test, y_pred_ens, power=1.5, sample_weight=w_test)

    print(f"XGBoost Deviance:  {dev_xgb:.5f}")
    print(f"LightGBM Deviance: {dev_lgb:.5f}")
    print(f"CatBoost Deviance: {dev_cat:.5f}")
    print(f"-----------------------------")
    print(f"Ensemble Deviance: {dev_ens:.5f} (Lower is better)")
    
    # Save Logic
    data['preprocessor'] = preprocessor
    return ensemble, data

if __name__ == "__main__":
    model, updated_data = train_ensemble_model("processed_step_2.pkl")
    
    joblib.dump(updated_data, "processed_step_2.pkl")
    joblib.dump(model, "tweedie_model.joblib")
    print("\nSuccess! Three-model ensemble saved to 'tweedie_model.joblib'")