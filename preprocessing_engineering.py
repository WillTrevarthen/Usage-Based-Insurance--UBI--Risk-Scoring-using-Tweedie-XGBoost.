import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

def preprocess_data(input_file):
    df = pd.read_pickle(input_file)
    
    # 1. Feature Engineering
    # Calculate this BEFORE defining the feature lists
    df['Power_Age_Ratio'] = df['VehPower'] / (df['DrivAge'] + 1)
    df['LogDensity'] = np.log(df['Density'])
    df['Is_New_Car'] = (df['VehAge'] <= 1).astype(int)
    df['Young_Urban'] = ((df['DrivAge'] < 25) & (df['Density'] > 5000)).astype(int)

    # 2. Define EXCLUSIVE lists (No column should appear in two lists)
    cat_features = ['Area', 'VehBrand', 'VehGas', 'Region']
    num_features = ['VehPower', 'VehAge', 'DrivAge', 'LogDensity', 'Power_Age_Ratio']
    bool_features = ['Young_Urban', 'Is_New_Car']
    
    # 3. Build the Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
            ('bool', 'passthrough', bool_features)
        ]
    )

    # 4. Prepare the Final DataFrame
    # Only include the specific columns we need to avoid "not unique" errors
    all_needed_columns = cat_features + num_features + bool_features
    X = df[all_needed_columns].copy() # .copy() ensures it's a fresh dataframe
    y = df['ClaimAmount_Capped']
    w = df['Exposure']

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, w_train, w_test, preprocessor

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, w_train, w_test, preprocessor = preprocess_data("processed_step_1.pkl")
    
    data_bundle = {
        'train': (X_train, y_train, w_train),
        'test': (X_test, y_test, w_test),
        'preprocessor': preprocessor
    }
    joblib.dump(data_bundle, "processed_step_2.pkl")
    print("Step 2 Reset: Columns are now unique and mapping is clean.")