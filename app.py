import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import streamlit.components.v1 as components

# Set page title and layout
st.set_page_config(page_title="Predictor of Chemical Transfer to Breast Milk", layout="wide")

st.title("Predictor of Chemical Transfer to Breast Milk")

# Load model and scaler
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_resource
def load_scaler(scaler_path):
    return joblib.load(scaler_path)

model = load_model("best_estimator_GA.pkl")
scaler = load_scaler("scaler.pkl")

# File upload
st.sidebar.header("Upload New Drug Compound Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file containing 84 features", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Check if the data has 84 features
        if data.shape[1] != 84:
            st.error(f"The uploaded file contains {data.shape[1]} features, but 84 are required. Please check the file.")
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
            prob_df = pd.DataFrame(probabilities, columns=["Probability_0 (low-risk chemical)", "Probability_1 (high-risk chemical)"])
            st.subheader("Prediction Probabilities")
            st.dataframe(prob_df.head())

            # Select sample index for SHAP force plot
            st.sidebar.header("SHAP Force Plot Options")
            sample_index = st.sidebar.number_input(
                "Select Sample Index (starting from 1)",
                min_value=1,
                max_value=len(data),
                value=1,
                step=1
            )
            class_index = st.sidebar.selectbox(
                "Select Class to Explain",
                options=[0, 1],
                index=1,  # Default select class 1
                format_func=lambda x: f"Class {x}"
            )

            if st.sidebar.button("Show SHAP Force Plot"):
                # Adjust sample index for 0-based array
                adjusted_index = sample_index - 1
                # Select a single sample
                single_sample = scaled_data[adjusted_index].reshape(1, -1)

                # Use SHAP to explain the model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(single_sample)

                # Ensure the sample index is within bounds for SHAP values
                if adjusted_index < len(shap_values[class_index]):
                    shap_value = shap_values[adjusted_index-1,:,class_index]  # 提取类别对应的 SHAP 值  # Correctly extract the SHAP values for the selected class
                    base_value = float(explainer.expected_value[class_index])

                    # Create SHAP force plot
                    st.subheader(f"SHAP Force Plot - Sample Index {sample_index} (Class {class_index})")
                    shap.initjs()  # Initialize JavaScript library

                    # Generate force plot
                    force_plot = shap.force_plot(
                        base_value,
                        shap_value,
                        single_sample[0],
                        feature_names=data.columns,
                        matplotlib=False
                    )

                    # Save as HTML file
                    html_file = f"force_plot_{sample_index}.html"
                    shap.save_html(html_file, force_plot)

                    # Display HTML in Streamlit
                    with open(html_file) as f:
                        components.html(f.read(), height=500, scrolling=True)
                else:
                    st.error(f"Cannot retrieve SHAP values for sample index {sample_index}, out of bounds.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV file containing 84 features on the left.")
