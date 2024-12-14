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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .plot-container {
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

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
    try:
        # Select features
        df_selected = df[selected_features]
        
        # Scale features
        df_scaled = pd.DataFrame(
            scaler.transform(df_selected),
            columns=selected_features,
            index=df_selected.index
        )
        return df_scaled
    except KeyError as e:
        missing_features = [feat for feat in selected_features if feat not in df.columns]
        raise KeyError(f"Missing required features: {missing_features}")

def calculate_shap_values(model: Any, features_scaled: pd.DataFrame) -> np.ndarray:
    """Calculate SHAP values for all instances"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled)
    return shap_values

def create_shap_force_plot(explainer: Any, shap_values: np.ndarray, 
                          features_scaled: pd.DataFrame, sample_idx: int) -> plt.Figure:
    """Create SHAP force plot for a single prediction"""
    plt.figure(figsize=(20, 4))  # Increased figure size
    
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[:, :, 1][sample_idx],
        features_scaled.iloc[sample_idx,:],
        matplotlib=True,
        show=False,
        text_rotation=45  # Better text readability
    )
    plt.tight_layout()
    return plt.gcf()

def main():
    model, scaler, selected_features = load_resources()
    
    # Page header
    st.title("🧬 Chemical Transfer into Human Milk Predictor")
    st.markdown("---")

    # Detailed instructions in sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        ### Step 1: Get Chemopy Descriptors
        1. Visit [ChemDes](http://www.scbdd.com/chemdes/)
        2. Input your molecule's SMILES structure
        3. Select "Chemopy Descriptors" in the options
        4. Calculate and download the descriptors

        ### Step 2: Prepare Your Data
        - Ensure your CSV file contains all required Chemopy descriptors
        - The model uses 84 selected descriptors for prediction
        - Data should be properly formatted (see example template)

        ### Step 3: Upload and Predict
        - Upload your prepared CSV file
        - View predictions and SHAP analysis
        - Download results for further analysis
        """)
        
        # Add download template option
        st.markdown("### Download Template")
        if st.button("📥 Download Feature Template"):
            template_df = pd.DataFrame(columns=selected_features)
            csv = template_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="descriptor_template.csv">Download Template CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Developer**: Xiaojie Huang")
        st.markdown("**Version**: 1.0")
    
    # Main content
    st.header("📊 Prediction Interface")
    
    uploaded_file = st.file_uploader(
        "Upload Chemopy Descriptors (CSV format)",
        type=['csv'],
        help="Upload the CSV file containing your Chemopy descriptors"
    )
    
    if uploaded_file is not None:
        try:
            # Data processing and prediction
            df = pd.read_csv(uploaded_file)
            
            with st.expander("View Raw Data", expanded=False):
                st.dataframe(df)
            
            df_scaled = process_features(df, scaler, selected_features)
            
            with st.expander("View Processed Data", expanded=False):
                st.dataframe(df_scaled)
            
            # Predictions and SHAP values
            predictions = model.predict_proba(df_scaled)
            explainer = shap.TreeExplainer(model)
            shap_values = calculate_shap_values(model, df_scaled)
            
            # Results section
            st.header("🎯 Results")
            
            # Prediction results
            results_df = pd.DataFrame({
                'Molecule ID': range(1, len(df) + 1),
                'Prediction': ['High Risk' if p > 0.5 else 'Low Risk' for p in predictions[:, 1]],
                'Risk Probability': [f"{p:.3f}" for p in predictions[:, 1]]
            })
            st.subheader("Prediction Results")
            st.dataframe(results_df)
            
            # Download results
            csv = results_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )
            
            # SHAP Analysis
            st.header("🔍 SHAP Analysis")
            
            molecule_idx = st.selectbox(
                "Select molecule for detailed analysis",
                options=range(len(df)),
                format_func=lambda x: f"Molecule {x+1}"
            )
            
            # Prediction info
            risk_prob = predictions[molecule_idx, 1]
            risk_color = "red" if risk_prob > 0.5 else "green"
            st.markdown(f"""
                ### Molecule {molecule_idx + 1}
                Risk Probability: <span style='color:{risk_color};font-weight:bold'>{risk_prob:.3f}</span>
                """, unsafe_allow_html=True)
            
            # SHAP force plot
            st.subheader("Feature Contribution Analysis")
            fig = create_shap_force_plot(explainer, shap_values, df_scaled, molecule_idx)
            st.pyplot(fig, bbox_inches='tight', pad_inches=1)
            
            # Feature importance
            with st.expander("View Feature Importance", expanded=False):
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': np.abs(shap_values[:, :, 1]).mean(0)
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(importance_df)
                
        except Exception as e:
            st.error("❌ Error Processing Data")
            st.error(f"Details: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
