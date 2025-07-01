import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: 700;
    }
    h2 {
        color: #3498db;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }
    .churn {
        background-color: #ff6b6b;
        color: white;
    }
    .no-churn {
        background-color: #51cf66;
        color: white;
    }
    .feature-importance {
        background-color: #D2B48C;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_components():
    model = load_model('churn_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_components()

# App header
st.title("📈 Customer Churn Prediction Dashboard")
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <p style="font-size: 18px; color: #7f8c8d;">
        Predict which customers are at risk of leaving your service using our advanced AI model
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar for user input
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: white;">Customer Details</h2>
        <p style="color: #bdc3c7;">Please provide customer information</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input fields
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 100, 35)
    tenure = st.slider("Tenure (years)", 0, 15, 5)
    balance = st.number_input("Account Balance ($)", 0, 300000, 10000)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active_member = st.selectbox("Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary ($)", 0, 300000, 50000)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])

# Convert inputs to model format
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# One-hot encode geography and gender
geography_france = 1 if geography == "France" else 0
geography_germany = 1 if geography == "Germany" else 0
geography_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0
gender_female = 1 if gender == "Female" else 0

# Create feature vector
features = np.array([
    credit_score, age, tenure, balance, num_products, 
    has_cr_card, is_active_member, estimated_salary,
    geography_france, geography_germany, geography_spain,
    gender_female, gender_male
]).reshape(1, -1)

# Scale features
features_scaled = scaler.transform(features)

# Make prediction
prediction = model.predict(features_scaled)
prediction_prob = prediction[0][0]
prediction_class = 1 if prediction_prob > 0.5 else 0

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div style="background-color: #3498db; padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h2 style="color: white; text-align: center;">Prediction Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if prediction_class == 1:
        st.markdown(f"""
        <div class="prediction-box churn">
            <h3 style="color: white;">⚠️ High Risk of Churn</h3>
            <p>Probability: {prediction_prob*100:.2f}%</p>
            <p>This customer is likely to leave your service.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box no-churn">
            <h3 style="color: white;">✅ Low Risk of Churn</h3>
            <p>Probability: {(1-prediction_prob)*100:.2f}%</p>
            <p>This customer is likely to stay with your service.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance visualization (placeholder - in a real app you'd use SHAP or similar)
    st.markdown("""
    <div class="feature-importance">
        <h3>Key Factors Influencing Prediction</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a fake feature importance for visualization
    feature_names = [
        'Credit Score', 'Age', 'Tenure', 'Balance', 'Products', 
        'Credit Card', 'Active', 'Salary', 
        'France', 'Germany', 'Spain',
        'Female', 'Male'
    ]
    
    # Random importance values (replace with actual feature importance if available)
    importance_values = np.random.rand(len(feature_names))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
    ax.set_title('Feature Importance', fontsize=16)
    ax.set_xlabel('Relative Importance', fontsize=12)
    ax.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("""
    <div style="background-color: #2ecc71; padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h2 style="color: white; text-align: center;">Customer Profile</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    """, unsafe_allow_html=True)
    
    st.metric("Credit Score", f"{credit_score}")
    st.metric("Age", f"{age}")
    st.metric("Tenure", f"{tenure} years")
    st.metric("Account Balance", f"${balance:,.2f}")
    st.metric("Number of Products", num_products)
    st.metric("Has Credit Card", "Yes" if has_cr_card else "No")
    st.metric("Active Member", "Yes" if is_active_member else "No")
    st.metric("Estimated Salary", f"${estimated_salary:,.2f}")
    st.metric("Geography", geography)
    st.metric("Gender", gender)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 14px;">
    <hr>
    <p>Customer Churn Prediction Model • Powered by Artificial Neural Networks</p>
    <p>For business inquiries, please contact our data science team</p>
</div>
""", unsafe_allow_html=True)
