
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import pyodbc

def get_db_connection():
    conn = pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=r"RUDRA\SQLEXPRESS";'
        'DATABASE="sugardb";'
        'Trusted_Connection=yes;'
    )
    return conn


# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stAlert {padding: 1rem; border-radius: 0.5rem;}
    h1 {color: #1f77b4; padding-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)


# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('diabetes_m.pkl')
        scaler = joblib.load('diabetes_sc.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None


# Header
st.title("🏥 Diabetes Prediction System")
st.markdown("### AI-Powered Risk Assessment Tool")

# Load model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("❌ **Model files not found!**")
    st.info("""
    Please run the following command first:
    ```
    python diabetes_prediction.py
    ```
    This will train and save the model files.
    """)
    st.stop()

# Sidebar inputs
st.sidebar.title("⚙️ Patient Information")

st.sidebar.subheader("Demographics")
age = st.sidebar.slider('Age', 21, 100, 30)
pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 0)

st.sidebar.subheader("Medical Measurements")
glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 120)
bp = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 130, 70)
skin = st.sidebar.slider('Skin Thickness (mm)', 0, 100, 20)
insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 900, 80)
bmi = st.sidebar.number_input('BMI', 10.0, 70.0, 25.0, 0.1)
dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5, 0.01)

# Predict button
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 Predict", type="primary", use_container_width=True)

# Main content
if predict_btn:
    # Prepare input
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Standardize
    input_std = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_std)[0]
    
    # Get probability if available
    try:
        probability = model.predict_proba(input_std)[0]
        prob_negative = probability[0] * 100
        prob_positive = probability[1] * 100
    except:
        prob_positive = 100 if prediction == 1 else 0
        prob_negative = 100 - prob_positive
    
    # Display results
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction box
        if prediction == 0:
            if prob_positive < 30:
                st.success("### ✅ LOW RISK - Not Diabetic")
            else:
                st.warning("### ⚠️ MODERATE RISK - Not Diabetic")
        else:
            if prob_positive > 70:
                st.error("### 🔴 HIGH RISK - Diabetic")
            else:
                st.warning("### ⚠️ MODERATE RISK - Diabetic")
        
        # Probabilities
        st.subheader("Probability Breakdown")
        pcol1, pcol2 = st.columns(2)
        pcol1.metric("Non-Diabetic", f"{prob_negative:.1f}%")
        pcol2.metric("Diabetic", f"{prob_positive:.1f}%")
    
    with col2:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_positive,
            title={'text': "Risk Level"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors
    st.markdown("---")
    st.subheader("⚠️ Risk Factor Analysis")
    
    risk_factors = []
    positive_factors = []
    
    if glucose > 125:
        risk_factors.append("🔴 High Glucose Level (>125 mg/dL)")
    elif glucose < 100:
        positive_factors.append("🟢 Normal Glucose Level")
    
    if bmi > 30:
        risk_factors.append("🔴 High BMI - Obesity (>30)")
    elif 18.5 <= bmi <= 24.9:
        positive_factors.append("🟢 Healthy BMI (18.5-24.9)")
    
    if age > 45:
        risk_factors.append("🟡 Age Factor (>45)")
    
    if bp > 80:
        risk_factors.append("🔴 High Blood Pressure (>80 mm Hg)")
    elif 60 <= bp <= 80:
        positive_factors.append("🟢 Normal Blood Pressure")
    
    if dpf > 0.5:
        risk_factors.append("🟡 Higher Genetic Predisposition")
    
    if risk_factors:
        st.warning("**Identified Risk Factors:**")
        for factor in risk_factors:
            st.markdown(f"- {factor}")
    
    if positive_factors:
        st.success("**Positive Health Indicators:**")
        for factor in positive_factors:
            st.markdown(f"- {factor}")
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    if prediction == 1:
        st.error("""
        **Important Actions:**
        - Consult a healthcare professional immediately
        - Get comprehensive diabetes screening
        - Monitor blood glucose regularly
        - Consider lifestyle modifications
        """)
    else:
        st.success("""
        **Maintain Healthy Practices:**
        - Regular health check-ups
        - Balanced diet
        - Exercise regularly (30+ min daily)
        - Monitor weight and BMI
        """)
    
    # Disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ MEDICAL DISCLAIMER**
    
    This prediction is for educational purposes only. It should NOT replace 
    professional medical advice. Always consult qualified healthcare professionals 
    for medical concerns.
    """)
    
else:
    # Initial page
    st.markdown("---")
    st.info("👈 Enter patient information in the sidebar and click **Predict**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "Random Forest")
    col2.metric("Accuracy", "~75%")

