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

# Expected molecular descriptors for Model 1
DESCRIPTORS_MODEL1 = [
    "apol", "ast_fraglike", "a_acc", "a_nCl", "a_nI", "a_nS",
    "BCUT_SLOGP_0", "BCUT_SLOGP_2", "b_1rotR", "b_max1len", "chiral_u",
    "GCUT_PEOE_0", "GCUT_SLOGP_2", "GCUT_SMR_0", "h_logD", "h_log_pbo",
    "h_pKa", "h_pstates", "h_pstrain", "lip_druglike", "lip_violation",
    "opr_leadlike", "opr_nring", "opr_violation", "PEOE_RPC-", "PEOE_VSA+1",
    "PEOE_VSA+2", "PEOE_VSA+4", "PEOE_VSA+5", "PEOE_VSA+6", "PEOE_VSA-0",
    "PEOE_VSA-4", "PEOE_VSA_FHYD", "Q_VSA_PNEG", "reactive", "rsynth",
    "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5",
    "SlogP_VSA7", "SlogP_VSA9", "SMR_VSA1", "SMR_VSA4", "SMR_VSA5",
    "SMR_VSA6", "vsa_other", "ALogP98_Unknown", "ES_Count_aaaC",
    "ES_Count_aaCH", "ES_Count_aaO", "ES_Count_dS", "ES_Count_dsCH",
    "ES_Count_dsN", "ES_Count_sCH3", "ES_Count_sNH2", "ES_Count_sOH",
    "ES_Count_ssCH2", "ES_Count_sSH", "ES_Count_ssNH", "ES_Count_ssssN",
    "ES_Count_tN", "ES_Sum_sssCH", "ES_Sum_ssssC", "QED", "QED_HBD",
    "QED_MW", "QED_PSA", "Num_BridgeBonds", "Num_BridgeHeadAtoms",
    "Num_MesoStereoAtomsCIP", "Num_NegativeAtoms", "Num_RingFusionBonds",
    "Num_Rings3", "Num_Rings4", "Num_Rings6", "Num_Rings7",
    "Num_Rings9Plus", "Num_SpiroAtoms", "Num_TerminalRotomers",
    "Num_TrueAtropisomerCenters", "Molecular_FractionalPolarSASA", "IC"
]

@st.cache_resource
def load_resources():
    """Load all models and scalers"""
    # Load Model 1 resources
    model1 = joblib.load("best_estimator_GA.pkl")
    scaler1 = joblib.load("scaler.pkl")
    
    # Load Model 2 resources
    model2 = joblib.load("GA_chemopy_model.pkl")
    scaler2 = joblib.load("GA_chemopy_scaler.pkl")
    with open("GA_chemopy_features.pkl", 'rb') as f:
        features2 = joblib.load(f)
        
    return model1, scaler1, model2, scaler2, features2

def create_shap_force_plot(model, data_scaled, data_original, sample_idx):
    """Create SHAP force plot using original feature values"""
    # Initialize explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(data_scaled)
    
    # Create force plot using original values
    force_plot = shap.force_plot(
        explainer.expected_value[1],  # Use expected value for positive class
        shap_values[:,:,1][sample_idx],  # Use SHAP values for positive class
        data_original.iloc[sample_idx,:],  # Use original feature values
        matplotlib=False
    )
    
    # Save and display plot
    shap.save_html(f"force_plot_{sample_idx}.html", force_plot)
    
    with open(f"force_plot_{sample_idx}.html", 'r', encoding='utf-8') as f:
        components.html(f.read(), height=500, scrolling=True)

def main():
    # Load resources
    model1, scaler1, model2, scaler2, features2 = load_resources()
    
    # Page title
    st.title("🧬 Chemical Transfer Predictor for Breast Milk")
    
    # Sidebar - Model Selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.radio(
        "Choose Prediction Model:",
        ["Model 1 (Original)", "Model 2 (Enhanced)"],
        help="Select which model to use for prediction"
    )
    
    # Display model descriptions
    st.sidebar.markdown("""
    **Model Details:**
    - **Model 1**: Original balanced random forest model
    - **Model 2**: Enhanced model with genetic algorithm feature selection
    """)
    
    # File upload section
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file with descriptors",
        type=["csv"],
        help="Upload a CSV file containing the required molecular descriptors"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            
            # Select appropriate model and resources
            if model_choice == "Model 1 (Original)":
                current_model = model1
                current_scaler = scaler1
                expected_features = DESCRIPTORS_MODEL1
            else:
                current_model = model2
                current_scaler = scaler2
                expected_features = features2
            
            # Validate features
            if data.shape[1] != len(expected_features):
                st.error(f"Error: Expected {len(expected_features)} features, but got {data.shape[1]}")
                return
                
            # Data preview
            with st.expander("View Input Data", expanded=False):
                st.dataframe(data)
            
            # Process features
            data_scaled = current_scaler.transform(data)
            
            # Make predictions
            probabilities = current_model.predict_proba(data_scaled)
            
            # Display results
            st.header("Prediction Results")
            results_df = pd.DataFrame({
                'Sample': range(1, len(data) + 1),
                'Risk Class': ['High Risk' if p >= 0.5 else 'Low Risk' for p in probabilities[:, 1]],
                'Transfer Probability': [f"{p:.3f}" for p in probabilities[:, 1]]
            })
            st.dataframe(results_df)
            
            # SHAP Analysis section
            st.header("SHAP Analysis")
            sample_idx = st.number_input(
                "Select sample to analyze:",
                min_value=0,
                max_value=len(data)-1,
                value=0
            )
            
            # Display prediction for selected sample
            prob = float(probabilities[sample_idx, 1])
            risk_class = "High Risk" if prob >= 0.5 else "Low Risk"
            st.markdown(f"""
                ### Sample {sample_idx + 1} Analysis
                - **Transfer Probability**: <span style='color:{"red" if prob >= 0.5 else "green"}'>{prob:.3f}</span>
                - **Risk Classification**: <span style='color:{"red" if prob >= 0.5 else "green"}'>{risk_class}</span>
                """, unsafe_allow_html=True)
            
            # Generate SHAP plot
            st.subheader("Feature Contribution Analysis")
            create_shap_force_plot(current_model, data_scaled, data, sample_idx)
            
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
        # Display template information
        st.info("Please upload a CSV file containing the required molecular descriptors.")
        if st.button("Show Required Descriptors"):
            st.write("Required descriptors for the selected model:")
            if model_choice == "Model 1 (Original)":
                st.dataframe(pd.DataFrame(DESCRIPTORS_MODEL1, columns=["Descriptor"]))
            else:
                st.dataframe(pd.DataFrame(features2, columns=["Descriptor"]))

if __name__ == "__main__":
    main()
