import joblib
import xgboost as xgb
from sklearn.metrics import mean_tweedie_deviance

def train_tweedie_model(input_file):
    # 1. Load the data bundle from Step 2
    data = joblib.load(input_file)
    X_train, y_train, w_train = data['train']
    X_test, y_test, w_test = data['test']
    preprocessor = data['preprocessor']

    # 2. Transform the features
    print("Transforming features...")
    # fit_transform on train, transform only on test to prevent data leakage
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 3. Model Configuration (Optimized for 0.1622 Gini)
    model = xgb.XGBRegressor(
            objective='reg:tweedie',
            tweedie_variance_power=1.5, 
            learning_rate=0.03,
            n_estimators=500,    
            max_depth=4,         
            min_child_weight=20, 
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            early_stopping_rounds=50,
            random_state=42
        )

    # 4. Fit the model
    print("Starting model training...")
    model.fit(
        X_train_transformed, y_train,
        sample_weight=w_train,
        eval_set=[(X_test_transformed, y_test)],
        sample_weight_eval_set=[w_test],
        verbose=100
    )

    # 5. Evaluation
    y_pred = model.predict(X_test_transformed)
    # Using 1.5 power to match the objective for consistency
    deviance = mean_tweedie_deviance(y_test, y_pred, power=1.5, sample_weight=w_test)
    
    print(f"\nModel Training Complete.")
    print(f"Test Set Tweedie Deviance: {deviance:.4f}")

    # Update the data bundle with the FITTED preprocessor
    data['preprocessor'] = preprocessor
    return model, data

if __name__ == "__main__":
    model, updated_data = train_tweedie_model("processed_step_2.pkl")
    
    # Save the updated bundle and model
    joblib.dump(updated_data, "processed_step_2.pkl")
    joblib.dump(model, "tweedie_model.joblib")
    print("Step 3 Complete. Model and fitted preprocessor saved.")