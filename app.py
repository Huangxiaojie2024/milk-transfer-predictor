import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from imblearn.ensemble import BalancedRandomForestClassifier

# Load model and scaler
@st.cache_resource
def load_model():
    with open('best_estimator_GA.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Function to generate SHAP plots
def plot_shap_values(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 输出 shap_values 和 expected_value 的形状
    st.write(f"SHAP values shape: {len(shap_values)}")
    st.write(f"Expected value shape: {len(explainer.expected_value)}")

    st.header("SHAP Force Plot for the First Sample")

    # 确保 X 中有至少一个样本
    if X.shape[0] > 0:
        # 检查 shap_values 的长度，以防止超出索引
        if len(shap_values) > 1:
            # 生成类别 1 的 SHAP 力图
            shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0, :], feature_names=feature_names, matplotlib=True)
        else:
            # 如果只有一个类别，生成类别 0 的 SHAP 力图
            st.write("Only one class found, generating SHAP plot for class 0.")
            shap.force_plot(explainer.expected_value[0], shap_values[0][0], X.iloc[0, :], feature_names=feature_names, matplotlib=True)

        plt.savefig("shap_force_plot.png")
        st.image("shap_force_plot.png")
    else:
        st.error("Input data is empty or does not have enough rows.")

# Streamlit app
st.title("Assessing Chemical Exposure Risk in Breastfeeding Infants: An Explainable Machine Learning Model for Human Milk Transfer Prediction")

# Load the model and scaler
model, scaler = load_model()

# File uploader for CSV input
uploaded_file = st.file_uploader("Please upload a CSV file with 84 molecular descriptors", type="csv")

if uploaded_file is not None:
    # Read CSV file
    input_data = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the uploaded file
    st.write("Uploaded data preview:")
    st.write(input_data.head())
    
    # Ensure the input data has 84 descriptors
    if input_data.shape[1] == 84:
        # Normalize the input data
        normalized_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
        
        # Make predictions
        predictions = model.predict(normalized_data)
        prediction_probabilities = model.predict_proba(normalized_data)[:, 1]  # Probabilities
        
        # Show prediction probabilities
        st.write("Breast milk transfer probabilities for the uploaded compounds:")
        st.write(prediction_probabilities)
        
        # Generate and display SHAP force plot for the first sample
        plot_shap_values(model, normalized_data, feature_names=input_data.columns)
    else:
        st.error("The uploaded file does not have 84 molecular descriptors. Please check your file.")
