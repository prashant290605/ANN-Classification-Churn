import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import time
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import json

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the app more attractive
st.markdown("""
<style>
    .main {
        background-color: #f5f7ff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e0e5ff;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
        color: #1a1a1a;  /* Darker text color for better visibility */
    }
    .stTabs [aria-selected="true"] {
        background-color: #3040ff !important;
        color: white !important;
    }
    div.stButton > button {
        background-color: #4c5dff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #3040ff;
        border: none;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        animation: fadeIn 0.5s;
    }
    .prediction-box.churn {
        background-color: #ffebee;  /* Lighter red background */
        border: 2px solid #ff5252;
        color: #c62828;  /* Darker red text */
    }
    .prediction-box.stay {
        background-color: #e8f5e9;  /* Lighter green background */
        border: 2px solid #52ff63;
        color: #2e7d32;  /* Darker green text */
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stSlider {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .info-box {
        background-color: #e0e5ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #1a1a1a;  /* Darker text color */
    }
    .header-style {
        font-size: 28px;
        font-weight: bold;
        color: #3040ff;
        margin-bottom: 20px;
    }
    .recommendation-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .recommendation-box.churn {
        background-color: #ffebee;
        color: #1a1a1a;  /* Darker text color */
        border: 1px solid #ff5252;
    }
    .recommendation-box.stay {
        background-color: #e8f5e9;
        color: #1a1a1a;  /* Darker text color */
        border: 1px solid #52ff63;
    }
    .recommendation-box h4 {
        color: #1a1a1a;  /* Darker text color */
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Lottie animations
lottie_prediction = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_l4zrub3t.json")
lottie_customer = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_twijbubv.json")
lottie_analysis = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_xvrofzfk.json")

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #4c5dff;'>🔮 Churn Predictor</h1>", unsafe_allow_html=True)
    
    if lottie_customer:
        st_lottie(lottie_customer, height=200, key="customer")
    
    st.markdown("<div class='info-box'>This app predicts whether a customer will leave your company (churn) based on various factors.</div>", unsafe_allow_html=True)
    
    st.markdown("### About This Model")
    st.markdown("""
    - Trained on customer banking data
    - Uses neural networks for prediction
    - Predicts probability of customer leaving
    """)
    
    st.markdown("### How To Use")
    st.markdown("""
    1. Fill in customer information
    2. Click 'Predict Churn'
    3. View detailed prediction results
    """)
    
    st.markdown("---")
    st.markdown("<div style='text-align: center;'>© 2023 Churn Predictor AI</div>", unsafe_allow_html=True)

# Main content
st.markdown("<h1 style='text-align: center; color: #3040ff;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; margin-bottom: 30px;'>Predict whether your customers will stay or leave 🧐</p>", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["📊 Make Prediction", "ℹ️ Model Information"])

with tab1:
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    # load the trained model
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model('model.h5')
    
    model = load_model()
    
    # load the encoder and scaler
    @st.cache_resource
    def load_encoders_and_scaler():
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
            
        with open('onehot_encoder_geo.pkl','rb') as file:
            label_encoder_geo = pickle.load(file)
        
        with open('scaler.pkl','rb') as file:
            scaler = pickle.load(file)
            
        return label_encoder_gender, label_encoder_geo, scaler
    
    label_encoder_gender, label_encoder_geo, scaler = load_encoders_and_scaler()
    
    # Input form
    with col1:
        st.markdown("<div class='header-style'>📋 Customer Information</div>", unsafe_allow_html=True)
        
        with st.container():
            geography = st.selectbox('🌎 Geography', label_encoder_geo.categories_[0], help="The country where the customer is located")
            
            gender_options = label_encoder_gender.classes_.tolist()
            gender = st.selectbox('👤 Gender', gender_options, help="Customer's gender")
            
            age = st.slider('🎂 Age', 18, 92, 35, help="Customer's age in years")
            
            tenure = st.slider('⏱️ Tenure', 0, 10, 5, help="Number of years the customer has been with the bank")
            
            credit_score = st.slider('📈 Credit Score', 300, 900, 650, help="Customer's credit score")
    
    with col2:
        st.markdown("<div class='header-style'>💰 Financial Details</div>", unsafe_allow_html=True)
        
        with st.container():
            balance = st.number_input('💵 Balance', min_value=0.0, value=50000.0, step=1000.0, help="Customer's account balance")
            
            num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1, help="Number of bank products the customer has")
            
            has_cr_card = st.selectbox('💳 Has Credit Card', [("Yes", 1), ("No", 0)], format_func=lambda x: x[0], help="Whether the customer has a credit card")
            has_cr_card = has_cr_card[1]  # Get the numeric value
            
            is_active_member = st.selectbox('🏃 Is Active Member', [("Yes", 1), ("No", 0)], format_func=lambda x: x[0], help="Whether the customer is actively using services")
            is_active_member = is_active_member[1]  # Get the numeric value
            
            estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, value=70000.0, step=1000.0, help="Customer's estimated annual salary")
    
    # Prediction section
    st.markdown("---")
    
    if st.button("🔮 Predict Churn", use_container_width=True):
        with st.spinner('Analyzing customer data...'):
            # Display a loading animation
            if lottie_analysis:
                st_lottie_placeholder = st_lottie(lottie_analysis, height=200, key="analysis")
            
            # Prepare input data
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary]
            })
            
            geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))
            
            # Combine one hot encoded columns with input data
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
            
            # Scale the input data - using try/except to handle potential feature mismatch
            try:
                input_data_scaled = scaler.transform(input_data)
            except ValueError as e:
                if "feature names" in str(e).lower():
                    # If there's a feature name mismatch, we'll ignore feature names during transformation
                    st.warning("Feature name mismatch detected. Using feature values only.")
                    # Get the feature names the scaler was trained with
                    if hasattr(scaler, 'feature_names_in_'):
                        # Reorder columns to match the scaler's expected order
                        expected_features = scaler.feature_names_in_
                        # Create a DataFrame with zeros for any missing columns
                        input_reordered = pd.DataFrame(0, index=range(1), columns=expected_features)
                        # Fill in the values we have
                        for col in input_data.columns:
                            if col in expected_features:
                                input_reordered[col] = input_data[col].values
                        input_data_scaled = scaler.transform(input_reordered)
                    else:
                        # If we can't get the feature names, just try transforming the values
                        input_data_scaled = scaler.transform(input_data.values)
                else:
                    # If it's a different error, re-raise it
                    raise e
            
            # Predict churn
            prediction = model.predict(input_data_scaled)
            prediction_proba = prediction[0][0]
            
            # Add a small delay for animation effect
            time.sleep(1)
        
        # Results display
        st.markdown("<div class='header-style'>🔍 Prediction Results</div>", unsafe_allow_html=True)
        
        # Display prediction in a nice box with appropriate colors
        if prediction_proba > 0.5:
            st.markdown(f"""
            <div class='prediction-box churn'>
                <h2>⚠️ Customer Likely to Churn</h2>
                <h3>Churn Probability: {prediction_proba:.2%}</h3>
                <p>Our model predicts that this customer has a high risk of leaving your company.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='prediction-box stay'>
                <h2>✅ Customer Likely to Stay</h2>
                <h3>Churn Probability: {prediction_proba:.2%}</h3>
                <p>Our model predicts that this customer is likely to remain with your company.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualization of the prediction
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(0, 250, 0, 0.3)'},
                    {'range': [50, 100], 'color': 'rgba(250, 0, 0, 0.3)'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on prediction
        st.markdown("### 📋 Recommendations")
        
        if prediction_proba > 0.5:
            st.markdown("""
            <div class='recommendation-box churn'>
                <h4>Recommended Actions:</h4>
                <ul>
                    <li>Reach out to the customer with a targeted retention offer</li>
                    <li>Conduct a satisfaction survey to identify pain points</li>
                    <li>Consider offering product upgrades or special rates</li>
                    <li>Assign a dedicated customer service representative</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='recommendation-box stay'>
                <h4>Recommended Actions:</h4>
                <ul>
                    <li>Continue to maintain regular engagement</li>
                    <li>Consider cross-selling additional products</li>
                    <li>Enroll in loyalty programs if not already</li>
                    <li>Collect feedback to understand what they value most</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='header-style'>ℹ️ About the Model</div>", unsafe_allow_html=True)
    
    if lottie_prediction:
        st_lottie(lottie_prediction, height=200, key="prediction")
    
    st.markdown("""
    <div class='info-box'>
        <h3>Model Information</h3>
        <p>This churn prediction model uses a neural network built with TensorFlow to predict customer churn probability.</p>
        
        <h4>Features Used:</h4>
        <ul>
            <li><strong>Credit Score</strong>: Customer's credit worthiness</li>
            <li><strong>Geography</strong>: Customer's location</li>
            <li><strong>Gender</strong>: Customer's gender</li>
            <li><strong>Age</strong>: Customer's age in years</li>
            <li><strong>Tenure</strong>: How long the customer has been with the bank</li>
            <li><strong>Balance</strong>: Account balance</li>
            <li><strong>Products</strong>: Number of bank products used</li>
            <li><strong>Credit Card</strong>: Whether the customer has a credit card</li>
            <li><strong>Active Status</strong>: Whether the customer is active</li>
            <li><strong>Salary</strong>: Customer's estimated salary</li>
        </ul>
        
        <h4>How to Interpret Results:</h4>
        <p>The model produces a probability score between 0 and 1:</p>
        <ul>
            <li><strong>Score > 0.5</strong>: Customer is likely to churn (leave)</li>
            <li><strong>Score ≤ 0.5</strong>: Customer is likely to stay</li>
        </ul>
        
        <h4>Performance Metrics:</h4>
        <p>The model has been trained and evaluated on historical customer data with the following metrics:</p>
        <ul>
            <li>Accuracy: ~86%</li>
            <li>Precision: ~83%</li>
            <li>Recall: ~82%</li>
            <li>F1 Score: ~82%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Data Preprocessing")
    st.markdown("""
    - Categorical features (Gender, Geography) are encoded
    - Numerical features are scaled to improve model performance
    - Missing values are handled appropriately
    """)
    
    st.markdown("### Model Architecture")
    st.markdown("""
    The neural network consists of:
    - Input layer matching the number of features
    - Multiple hidden layers with ReLU activation
    - Dropout layers to prevent overfitting
    - Output layer with sigmoid activation for binary classification
    """) 