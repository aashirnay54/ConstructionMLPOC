import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Construction Cost AI", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('construction_model.joblib')

try:
    artifacts = load_model()
    model = artifacts['model']
    scaler = artifacts['scaler']
    defaults = artifacts['default_values']
    feature_names = artifacts['feature_names']
except FileNotFoundError:
    st.error("⚠️ Model file not found. Please run 'train_model.py' first.")
    st.stop()

st.title("🏗️ AI Construction Cost Estimator")
st.markdown("Predict final project costs using **Ridge Regression**.")

# Sidebar for Currency
with st.sidebar:
    st.header("⚙️ Settings")
    currency_rate = st.number_input("Currency Conversion Rate (Units to USD)", value=30.0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Project Details")
    floor_area = st.number_input("Total Floor Area (sq m)", value=float(defaults['Total_Floor_Area']))
    lot_area = st.number_input("Lot Area (sq m)", value=float(defaults['Lot_Area']))
    duration = st.slider("Estimated Duration (Months)", 1, 36, int(defaults['Duration']))
    locality = st.selectbox("Project Neighborhood ID", range(1, 21), index=0)

with col2:
    st.subheader("2. Financials")
    prelim_est_unit = st.number_input("Prelim Est. Cost (per sq m)", value=float(defaults['Prelim_Est_Unit_Cost']))
    unit_price_start = st.number_input("Unit Price at Start", value=float(defaults['Unit_Price_Start']))

    st.divider()
    st.subheader("3. Economic Climate")
    econ_scenario = st.select_slider(
        "Market Condition Multiplier",
        options=["Recession (Low Costs)", "Normal", "Inflationary (High Costs)"],
        value="Normal"
    )
    multipliers = {"Recession (Low Costs)": 0.9, "Normal": 1.0, "Inflationary (High Costs)": 1.1}
    econ_mult = multipliers[econ_scenario]

if st.button("Calculate Final Cost", type="primary"):

    input_data = defaults.copy()

    input_data['Total_Floor_Area'] = floor_area
    input_data['Lot_Area'] = lot_area
    input_data['Duration'] = duration
    input_data['Prelim_Est_Unit_Cost'] = prelim_est_unit
    input_data['Unit_Price_Start'] = unit_price_start
    input_data['Project_Locality'] = locality

    # NOTE: Total_Prelim_Est removed to prevent data leakage

    manual_vars = ['Total_Floor_Area', 'Lot_Area', 'Duration', 'Prelim_Est_Unit_Cost', 
                   'Unit_Price_Start', 'Project_Locality']

    for col in feature_names:
        if col not in manual_vars:
            input_data[col] = input_data[col] * econ_mult

    df_input = pd.DataFrame([input_data], columns=feature_names)
    X_scaled = scaler.transform(df_input)

    prediction = model.predict(X_scaled)[0]
    prediction_usd = prediction * currency_rate

    st.success(f"### Predicted Cost: ${prediction_usd:,.2f} (approx.)")
    st.caption(f"Model Raw Output: {prediction:,.0f} Units")

    coefficients = model.coef_
    contributions = X_scaled[0] * coefficients

    contrib_df = pd.DataFrame({'Feature': feature_names, 'Impact': contributions})
    contrib_df['Abs_Impact'] = contrib_df['Impact'].abs()
    top_drivers = contrib_df.sort_values(by='Abs_Impact', ascending=False).head(8)

    st.markdown("#### Top Cost Drivers for this Project")
    st.bar_chart(top_drivers.set_index('Feature')['Impact'])
