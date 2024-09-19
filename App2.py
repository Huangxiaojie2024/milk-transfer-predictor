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
    
    return prediction_proba, shap_values, data_scaled

def plot_shap_force(shap_values, data_scaled, feature_names):
    # Take the SHAP values for the positive class (index 1)
    shap.initjs()  # Initialize JavaScript visualization library
    return shap.force_plot(explainer.expected_value[1], shap_values[1], feature_names)

def main():
    st.title('Chemical Compound Breast Milk Transfer Prediction')
    
    # File uploader
    file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if file:
        data = load_data(file)
        prediction_proba, shap_values, data_scaled = predict_and_explain(data)
        
        # Display prediction
        st.subheader('Prediction Probability')
        st.write(prediction_proba)
        
        # Plot SHAP force plot
        st.subheader('SHAP Force Plot')
        force_plot_html = plot_shap_force(shap_values, data_scaled, data.columns)
        st.components.v1.html(force_plot_html.html(), height=300)  # Render the force plot as HTML

if __name__ == '__main__':
    main()
