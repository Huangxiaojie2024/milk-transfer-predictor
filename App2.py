import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import streamlit.components.v1 as components

# Set page title and layout
st.set_page_config(page_title="Prediction of Chemical Transfer to Breast Milk", layout="wide")

st.title("Chemical Transfer Predictor for Breast Milk")

# Load model and scaler
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_resource
def load_scaler(scaler_path):
    return joblib.load(scaler_path)

model = load_model("best_estimator_GA.pkl")
scaler = load_scaler("scaler.pkl")

# Load the training feature names for comparison
training_feature_names = pd.read_csv("training_feature_names.csv")  # Ensure this file contains the correct feature names
expected_features = training_feature_names.columns.tolist()

# File upload
st.sidebar.header("Upload New Drug Compound Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with 84 features", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Check if it contains 84 features
        if data.shape[1] != 84:
            st.error(f"The uploaded file contains {data.shape[1]} features, but 84 features are required. Please check the file.")
        else:
            # Check if the feature names match
            if list(data.columns) != expected_features:
                st.error("The feature names in the uploaded file do not match the expected feature names. Please check the file.")
            else:
                st.subheader("Uploaded Data Preview")
                st.dataframe(data.head())

                # Feature standardization
                scaled_data = scaler.transform(data)
                scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
                st.subheader("Standardized Feature Preview")
                st.dataframe(scaled_df.head())

                # Prediction probabilities
                probabilities = model.predict_proba(scaled_data)
                prob_df = pd.DataFrame(probabilities, columns=["Probability_0 (low-risk)", "Probability_1 (high-risk)"])
                st.subheader("Prediction Probabilities")
                st.dataframe(prob_df.head())

                # Select sample for SHAP force plot
                st.sidebar.header("SHAP Force Plot Options")
                sample_index = st.sidebar.number_input(
                    "Select Sample Index (starting from 1)",
                    min_value=1,
                    max_value=len(data),
                    value=1,
                    step=1
                ) - 1  # Adjust for zero-based indexing
                class_index = st.sidebar.selectbox(
                    "Select Class to Explain",
                    options=[0, 1],
                    index=1,  # Default to class 1
                    format_func=lambda x: f"Class {x}"
                )

                if st.sidebar.button("Show SHAP Force Plot"):
                    # Select a single sample
                    single_sample = scaled_data[sample_index].reshape(1, -1)

                    # Use SHAP to explain the model
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(single_sample)

                    # Extract SHAP values for the specified class and sample
                    shap_value = shap_values[0][:, class_index]  # Extract SHAP values corresponding to the class
                    base_value = float(explainer.expected_value[class_index])

                    # Create SHAP force plot
                    st.subheader(f"SHAP Force Plot - Sample Index {sample_index + 1} (Class {class_index})")
                    shap.initjs()  # Initialize JavaScript library

                    # Generate force plot and save as HTML
                    force_plot = shap.force_plot(
                        base_value,
                        shap_value,
                        single_sample[0],
                        feature_names=data.columns,
                        matplotlib=False
                    )

                    # Save as HTML file
                    html_file = f"force_plot_{sample_index + 1}.html"
                    shap.save_html(html_file, force_plot)

                    # Display HTML in Streamlit
                    with open(html_file) as f:
                        components.html(f.read(), height=500, scrolling=True)

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a CSV file with 84 features on the left.")
