import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your trained Balanced Random Forest model and the scaler
model = joblib.load('best_estimator_GA.pkl')
scaler = joblib.load('scaler.pkl')

def load_data(file):
    data = pd.read_csv(file)
    return data

def predict_and_explain(data):
    # Standardize the features
    data_scaled = scaler.transform(data)
    
    # Make prediction
    prediction_proba = model.predict_proba(data_scaled)[:, 1]
    
    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_scaled)
    
    return prediction_proba, shap_values, data_scaled, explainer

def plot_shap_force(explainer, shap_values, data_scaled, feature_names):
    # Initialize JavaScript visualization library
    shap.initjs()
    # Create the SHAP force plot for the positive class
    force_plot = shap.force_plot(explainer.expected_value, shap_values, feature_names)
    return force_plot


def main():
    st.title('Chemical Compound Breast Milk Transfer Prediction')
    
    # File uploader
    file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if file:
        data = load_data(file)
        prediction_proba, shap_values, data_scaled, explainer = predict_and_explain(data)
        
        # Display prediction
        st.subheader('Prediction Probability')
        st.write(prediction_proba)
        
        # Generate and display SHAP force plot
        st.subheader('SHAP Force Plot')
        force_plot = plot_shap_force(explainer, shap_values, data_scaled, data.columns)
        st.pyplot(force_plot)

if __name__ == '__main__':
    main()
