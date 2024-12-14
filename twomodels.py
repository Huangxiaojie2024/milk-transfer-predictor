import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import streamlit.components.v1 as components
import base64

# Set page config
st.set_page_config(
    page_title="Prediction of Chemical Transfer to Breast Milk",
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
    }
    .pred-text {
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Expected molecular descriptors for Model 1 (MOE+DS)
DESCRIPTORS_MODEL1 = [
    # 您的84个MOE+DS特征列表
]

@st.cache_resource
def load_resources():
    """Load all models and scalers"""
    # Load BRF_MOE+DS_GA_84 model
    model1 = joblib.load("BRF_MOE+DS_GA_84.pkl")
    scaler1 = joblib.load("MOE+DS_scaler.pkl")
    with open("MOE+DS_features.pkl", 'rb') as f:
        features1 = joblib.load(f)
    
    # Load BRF_Chemopy_GA_101 model
    model2 = joblib.load("BRF_Chemopy_GA_101.pkl")
    scaler2 = joblib.load("Chemopy_scaler.pkl")
    with open("Chemopy_features.pkl", 'rb') as f:
        features2 = joblib.load(f)
        
    return model1, scaler1, features1, model2, scaler2, features2

def extract_features(data, selected_features):
    """Extract required features from input data"""
    try:
        return data[selected_features]
    except KeyError as e:
        missing_features = [f for f in selected_features if f not in data.columns]
        raise KeyError(f"Missing required features: {missing_features}")

def create_shap_force_plot(model, data_scaled, data_original, sample_idx):
    """Create SHAP force plot using original feature values"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_scaled)
    
    force_plot = shap.force_plot(
        explainer.expected_value[1],
        shap_values[:,:,1][sample_idx],
        data_original.iloc[sample_idx,:],
        matplotlib=False
    )
    
    shap.save_html(f"force_plot_{sample_idx}.html", force_plot)
    
    with open(f"force_plot_{sample_idx}.html", 'r', encoding='utf-8') as f:
        components.html(f.read(), height=500, scrolling=True)

def main():
    # Load resources
    model1, scaler1, features1, model2, scaler2, features2 = load_resources()
    
    # Page title
    st.title("🧬 Chemical Transfer Predictor for Breast Milk")
    
    # Introduction
    st.markdown("""
    ## About This Tool
    This application provides two advanced models for predicting chemical transfer into human breast milk:
    
    ### 1. BRF_MOE+DS_GA_84 Model
    - Uses 84 selected features from MOE and Discovery Studio
    - Automatically extracts required features from your input data
    - Upload any MOE/DS descriptor file, and the model will select needed features
    
    ### 2. BRF_Chemopy_GA_101 Model
    - Uses 101 Chemopy molecular descriptors
    - Requires pre-calculation of Chemopy descriptors from ChemDes
    - Follow these steps:
        1. Visit [ChemDes](http://www.scbdd.com/chemdes/)
        2. Input your molecule's SMILES structure
        3. Select "Chemopy Descriptors" in the options
        4. Calculate and download the descriptors
        5. Upload the downloaded file here
    """)
    
    # Sidebar - Model Selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.radio(
        "Choose Prediction Model:",
        ["BRF_MOE+DS_GA_84", "BRF_Chemopy_GA_101"],
        help="Select which model to use for prediction"
    )
    
    # Model-specific instructions
    if model_choice == "BRF_MOE+DS_GA_84":
        st.info("""
        **Using BRF_MOE+DS_GA_84 Model**
        - Upload your MOE/DS descriptor file
        - The model will automatically extract the required 84 features
        - Features are selected using genetic algorithm optimization
        """)
    else:
        st.info("""
        **Using BRF_Chemopy_GA_101 Model**
        1. Calculate Chemopy descriptors:
           - Go to [ChemDes](http://www.scbdd.com/chemdes/)
           - Enter your SMILES structure
           - Select Chemopy descriptors
           - Download results
        2. Upload the Chemopy descriptor file here
        """)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload descriptor file (CSV format)",
        type=["csv"],
        help="Upload your molecular descriptor file"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            
            # Select appropriate model and process data
            if model_choice == "BRF_MOE+DS_GA_84":
                # Extract required features
                data_selected = extract_features(data, features1)
                current_model = model1
                current_scaler = scaler1
            else:
                # Use all features for Chemopy model
                data_selected = data
                current_model = model2
                current_scaler = scaler2
            
            # Show data preview
            with st.expander("View Selected Features", expanded=False):
                st.dataframe(data_selected)
            
            # Scale features
            data_scaled = current_scaler.transform(data_selected)
            
            # Make predictions
            probabilities = current_model.predict_proba(data_scaled)
            
            # Results section
            st.header("Prediction Results")
            results_df = pd.DataFrame({
                'Sample': range(1, len(data) + 1),
                'Risk Class': ['High Risk' if p >= 0.5 else 'Low Risk' for p in probabilities[:, 1]],
                'Transfer Probability': [f"{p:.3f}" for p in probabilities[:, 1]]
            })
            st.dataframe(results_df)
            
            # SHAP Analysis
            st.header("SHAP Analysis")
            sample_idx = st.number_input(
                "Select sample to analyze:",
                min_value=0,
                max_value=len(data)-1,
                value=0
            )
            
            # Show prediction details
            prob = float(probabilities[sample_idx, 1])
            risk_class = "High Risk" if prob >= 0.5 else "Low Risk"
            st.markdown(f"""
                ### Sample {sample_idx + 1} Analysis
                - **Transfer Probability**: <span style='color:{"red" if prob >= 0.5 else "green"}'>{prob:.3f}</span>
                - **Risk Classification**: <span style='color:{"red" if prob >= 0.5 else "green"}'>{risk_class}</span>
                """, unsafe_allow_html=True)
            
            # SHAP force plot
            st.subheader("Feature Contribution Analysis")
            create_shap_force_plot(current_model, data_scaled, data_selected, sample_idx)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error("Error processing file")
            st.exception(e)
    
    else:
        # Show feature templates
        if model_choice == "BRF_MOE+DS_GA_84":
            with st.expander("View Required MOE+DS Features"):
                st.dataframe(pd.DataFrame(features1, columns=["Required Features"]))
        else:
            st.write("Please calculate and upload Chemopy descriptors from ChemDes website.")

if __name__ == "__main__":
    main()
