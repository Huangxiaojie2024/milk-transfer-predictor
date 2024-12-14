import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
from typing import Tuple, Dict, Any
import base64

# Set page config
st.set_page_config(
    page_title="Chemical Transfer into Human Milk Predictor",
    page_icon="🧬",
    layout="wide"
)

# Load models and scalers
@st.cache_resource
def load_resources() -> Tuple[Any, Any, list]:
    """Load model, scaler and selected features"""
    with open('GA_chemopy_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('GA_chemopy_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('GA_chemopy_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
    return model, scaler, selected_features

def process_features(df: pd.DataFrame, scaler: Any, selected_features: list) -> pd.DataFrame:
    """Process input features using scaler and feature selection"""
    # Select features
    df_selected = df[selected_features]
    
    # Scale features
    df_scaled = pd.DataFrame(
        scaler.transform(df_selected),
        columns=selected_features,
        index=df_selected.index
    )
    return df_scaled

def calculate_shap_values(model: Any, features_scaled: pd.DataFrame) -> np.ndarray:
    """Calculate SHAP values for all instances"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled)
    return shap_values

def create_shap_force_plot(explainer: Any, shap_values: np.ndarray, 
                          features_scaled: pd.DataFrame, sample_idx: int) -> plt.Figure:
    """Create SHAP force plot for a single prediction"""
    plt.figure(figsize=(10, 2))
    
    # Use class 1 (high-risk) SHAP values
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[:, :, 1][sample_idx],  # Get SHAP values for class 1
        features_scaled.iloc[sample_idx,:],
        matplotlib=True,
        show=False
    )
    
    return plt.gcf()

def get_feature_importance(shap_values: np.ndarray, features: list) -> pd.DataFrame:
    """Calculate feature importance based on mean absolute SHAP values"""
    # Use class 1 (high-risk) SHAP values
    mean_shap = np.abs(shap_values[:, :, 1]).mean(0)
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    return importance_df

def main():
    # Load resources
    model, scaler, selected_features = load_resources()
    
    # Page header
    st.title("🧬 Chemical Transfer into Human Milk Predictor")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Calculate molecular descriptors using MOE (v2022.02) and DS2019
        2. Prepare a CSV file with the required 84 descriptors
        3. Upload your file below
        4. View prediction results and SHAP interpretation
        """)
        
        st.markdown("---")
        st.markdown("**Developer**: Xiaojie Huang")
        st.markdown("**Version**: 1.0")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Molecular Descriptors (CSV format)",
        type=['csv'],
        help="Upload a CSV file containing the 84 molecular descriptors"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process data
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())
            
            # Process features
            df_scaled = process_features(df, scaler, selected_features)
            st.subheader("Standardized Feature Preview")
            st.dataframe(df_scaled.head())
            
            # Make predictions and calculate SHAP values
            predictions = model.predict_proba(df_scaled)
            explainer = shap.TreeExplainer(model)
            shap_values = calculate_shap_values(model, df_scaled)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Prediction Results")
                results_df = pd.DataFrame({
                    'Molecule ID': df.index,
                    'Prediction': ['High Risk' if p > 0.5 else 'Low Risk' for p in predictions[:, 1]],
                    'Risk Probability': [f"{p:.3f}" for p in predictions[:, 1]]
                })
                st.dataframe(results_df)
                
                # Feature importance
                importance_df = get_feature_importance(shap_values, selected_features)
                st.subheader("Top 10 Important Features")
                st.dataframe(importance_df.head(10))
                
                # Download results
                csv = results_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">Download Results</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                st.subheader("SHAP Analysis")
                molecule_idx = st.selectbox(
                    "Select molecule for SHAP analysis",
                    options=range(len(df)),
                    format_func=lambda x: f"Molecule {x+1}"
                )
                
                st.markdown(f"Risk Probability: **{predictions[molecule_idx, 1]:.3f}**")
                
                # Create SHAP force plot
                fig = create_shap_force_plot(explainer, shap_values, df_scaled, molecule_idx)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.error("Detailed error: ")
            st.exception(e)
            
if __name__ == "__main__":
    main()
