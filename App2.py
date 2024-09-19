import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the saved model and scaler
model = joblib.load('best_estimator_GA.pkl')
scaler = joblib.load('scaler.pkl')

# Set up the app title
st.title("BRF Prediction and SHAP Visualization")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file with 84 features", type="csv")

if uploaded_file:
    # Read the CSV file
    input_data = pd.read_csv(uploaded_file)

    # Check if the CSV has 84 features
    if input_data.shape[1] == 84:
        # Standardize the input features
        scaled_data = scaler.transform(input_data)

        # Generate predictions
        prediction_probs = model.predict_proba(scaled_data)

        # Display prediction probabilities
        st.write("Prediction Probabilities:", prediction_probs)

        # Explain model predictions using SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_data)

        # SHAP force plot for class 1
        st.write("SHAP Force Plot for the first instance (Class 1):")
        shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], scaled_data[0, :], matplotlib=True)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.error("The uploaded CSV must contain exactly 84 features.")
