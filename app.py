import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="UBI Risk Engine", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("tweedie_model.joblib")
    # We need the bundle to get the fitted preprocessor
    data_bundle = joblib.load("processed_step_2.pkl")
    return model, data_bundle['preprocessor']

model, preprocessor = load_assets()

# 2. UI Header
st.title("🚗 Actuarial Risk Engine")
st.markdown("""
This tool uses a **Tweedie-XGBoost** model to calculate the predicted annual claim cost (Pure Premium) 
based on driver and vehicle profiles.
""")

# 3. Input Sidebar
st.sidebar.header("Policyholder Profile")

# Categorical Inputs
area = st.sidebar.selectbox("Area Code", ['A', 'B', 'C', 'D', 'E', 'F'])
veh_brand = st.sidebar.selectbox("Vehicle Brand", ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B10', 'B11', 'B12', 'B13', 'B14'])
veh_gas = st.sidebar.selectbox("Fuel Type", ['Regular', 'Diesel'])
region = st.sidebar.selectbox("Region (R-code)", ['R11', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R31', 'R41', 'R42', 'R43', 'R52', 'R53', 'R54', 'R72', 'R73', 'R74', 'R82', 'R83', 'R91', 'R93', 'R94'])

# Numerical Inputs
veh_power = st.sidebar.slider("Vehicle Power", 4, 15, 6)
veh_age = st.sidebar.slider("Vehicle Age (Years)", 0, 25, 5)
driv_age = st.sidebar.slider("Driver Age", 18, 90, 35)
density = st.sidebar.number_input("Population Density (pop/km2)", value=1000)

# 4. Feature Engineering Logic (The "Brain")
def predict_risk():
    # Construct raw dataframe
    input_dict = {
        'Area': [area],
        'VehBrand': [veh_brand],
        'VehGas': [veh_gas],
        'Region': [region],
        'VehPower': [veh_power],
        'VehAge': [veh_age],
        'DrivAge': [driv_age],
        'Density': [density]
    }
    df_input = pd.DataFrame(input_dict)

    # Apply same engineering as Script 02
    df_input['Power_Age_Ratio'] = df_input['VehPower'] / (df_input['DrivAge'] + 1)
    df_input['LogDensity'] = np.log(df_input['Density'] + 1)
    df_input['Is_New_Car'] = (df_input['VehAge'] <= 1).astype(int)
    df_input['Young_Urban'] = ((df_input['DrivAge'] < 25) & (df_input['Density'] > 5000)).astype(int)

    # Ensure column order matches Step 02 lists: cat -> num -> bool
    cat_features = ['Area', 'VehBrand', 'VehGas', 'Region']
    num_features = ['VehPower', 'VehAge', 'DrivAge', 'LogDensity', 'Power_Age_Ratio']
    bool_features = ['Young_Urban', 'Is_New_Car']
    
    X_final = df_input[cat_features + num_features + bool_features]
    
    # Transform and Predict
    X_transformed = preprocessor.transform(X_final)
    prediction = model.predict(X_transformed)[0]
    
    return prediction

# 5. Output Display
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Predicted Risk Profile")
    premium = predict_risk()
    
    # Visualizing the output
    st.metric(label="Calculated Pure Premium", value=f"€{premium:.2f}/year")
    
    # Risk Rating
    if premium < 60:
        st.success("Risk Tier: LOW")
    elif premium < 120:
        st.warning("Risk Tier: MEDIUM")
    else:
        st.error("Risk Tier: HIGH")

with col2:
    st.subheader("Risk Drivers")
    st.write("Based on your inputs, the model identifies the following factors:")
    
    # Show engineered features for transparency
    st.write(f"- **Power/Age Intensity:** {(veh_power/(driv_age+1)):.3f}")
    if veh_age <= 1:
        st.write("- **New Vehicle Discount:** Applied")
    if (driv_age < 25) and (density > 5000):
        st.write("- **Young Urban Penalty:** Applied (High Risk Zone)")

# Add a little actuarial context at the bottom
st.info("💡 **Note:** Pure Premium represents the expected cost of claims. This does not include taxes, commissions, or profit margins.")