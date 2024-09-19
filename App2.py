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
    
    return prediction_proba, shap_values

def plot_shap(shap_values, feature_names):
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar")
    
def main():
    st.title('Chemical Compound Breast Milk Transfer Prediction')
    
    # File uploader
    file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if file:
        data = load_data(file)
        prediction_proba, shap_values = predict_and_explain(data)
        
        # Display prediction
        st.subheader('Prediction Probability')
        st.write(prediction_proba)
        
        # Plot SHAP values
        st.subheader('SHAP Force Plot')
        plt.figure(figsize=(10, 5))
        plot_shap(shap_values, data.columns)
        st.pyplot(plt)
        
if __name__ == '__main__':
    main()
