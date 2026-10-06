"""
Heart Disease Prediction - True Top 4 Features
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

DATA_PATH = Path(__file__).parent / "heart_disease_combined.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.replace('?', np.nan, inplace=True)
    df.drop(columns=["source"], inplace=True, errors="ignore")
    for col in df.columns:
        df[col] = df[col].astype(float)
        df[col] = df[col].fillna(df[col].median())
    return df

df = load_data()

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("❤️ Heart Disease Predictor")
st.markdown(
    "**Optimized Version:** This app now relies on the **4 most statistically powerful "
    "features** discovered by the AI for predicting heart disease."
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    # Feature 1: Chest Pain
    cp_labels = {1.0: "1 - Typical Angina", 2.0: "2 - Atypical Angina", 3.0: "3 - Non-anginal", 4.0: "4 - Asymptomatic"}
    cp = st.selectbox("Chest Pain Type (cp)", options=[1.0, 2.0, 3.0, 4.0], format_func=lambda x: cp_labels[x])
    
    # Feature 2: Max Heart Rate
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
    
with col2:
    # Feature 3: Number of Major Vessels
    ca = st.selectbox("Number of Major Vessels Blocked (ca)", options=[0.0, 1.0, 2.0, 3.0])
    
    # Feature 4: Thallium Stress Test
    thal_labels = {3.0: "3 - Normal", 6.0: "6 - Fixed Defect", 7.0: "7 - Reversible Defect"}
    thal = st.selectbox("Thallium Stress Test (thal)", options=[3.0, 6.0, 7.0], format_func=lambda x: thal_labels[x])

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    
    # The true top 4 features are mapped from the UI.
    # The other 9 features are defaulted to the dataset median/mode.
    user_input = pd.DataFrame({
        "age": [df['age'].median()],
        "sex": [df['sex'].mode()[0]],            
        "cp": [float(cp)],             
        "trestbps": [df['trestbps'].median()],
        "chol": [df['chol'].median()],         
        "fbs": [df['fbs'].mode()[0]],
        "restecg": [df['restecg'].mode()[0]],        
        "thalach": [float(thalach)],      
        "exang": [df['exang'].mode()[0]],          
        "oldpeak": [df['oldpeak'].median()],        
        "slope": [df['slope'].mode()[0]],          
        "ca": [float(ca)],             
        "thal": [float(thal)]            
    })

    prediction = model.predict(user_input)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ DISEASE PRESENT")
    else:
        st.success("### ✅ NO DISEASE")

    with st.expander("How this works (Under the hood)"):
        st.write(
            "The model processes 13 inputs total. Your 4 chosen inputs heavily guide the decision, "
            "while the remaining 9 features are locked at the "
            "statistical averages of the clinical dataset."
        )
        st.dataframe(user_input, hide_index=True)

st.divider()
with st.expander("View Source CSV Dataset"):
    st.dataframe(df)
