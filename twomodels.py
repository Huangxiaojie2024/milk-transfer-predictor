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

@st.cache_resource
def load_resources():
    """Load all models and scalers"""
    # Load Model 1 (MOE+DS)
    model1 = joblib.load("best_estimator_GA.pkl")
    scaler1 = joblib.load("scaler.pkl")
    features1 = [
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
    
    # Load Model 2 (Chemopy)
    model2 = joblib.load("GA_chemopy_model.pkl")
    scaler2 = joblib.load("GA_chemopy_scaler.pkl")
    with open("GA_chemopy_features.pkl", 'rb') as f:
        features2 = joblib.load(f)
        
    return model1, scaler1, features1, model2, scaler2, features2

def process_features(data, model_choice, features_list):
    """Extract and validate required features"""
    try:
        # Check which features are available in the input data
        available_features = list(set(features_list) & set(data.columns))
        missing_features = list(set(features_list) - set(data.columns))
        
        if missing_features:
            st.warning(f"Missing {len(missing_features)} required features. Please ensure all required features are present.")
            with st.expander("View missing features"):
                st.write(missing_features)
            return None
            
        # Extract only the required features in the correct order
        return data[features_list]
    
    except Exception as e:
        st.error(f"Error processing features: {str(e)}")
        return None

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
    ## Welcome to the Chemical Transfer Predictor
    This tool provides two advanced models for predicting chemical transfer into human breast milk:

    ### 1. BRF_MOE+DS_GA_84 Model
    - Utilizes 84 optimally selected features from MOE and Discovery Studio descriptors
    - Features selected through genetic algorithm optimization
    - **Input Requirements**: MOE and Discovery Studio molecular descriptors
    
    ### 2. BRF_Chemopy_GA_101 Model
    - Uses 101 Chemopy molecular descriptors
    - **Important**: Requires pre-calculation of descriptors from ChemDes
    
    ### How to Use
    1. **For BRF_MOE+DS_GA_84**:
       - Prepare your MOE and DS descriptors
       - Upload your descriptor file directly
       - The model will automatically extract required features
    
    2. **For BRF_Chemopy_GA_101**:
       - Visit [ChemDes](http://www.scbdd.com/chemdes/)
       - Input your molecule's SMILES structure
       - Select "Chemopy Descriptors"
       - Calculate and download descriptors
       - Upload the Chemopy file here
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
        - Optimized for accurate prediction of breast milk transfer
        """)
    else:
        st.info("""
        **Using BRF_Chemopy_GA_101 Model**
        1. First obtain Chemopy descriptors:
           - Visit [ChemDes](http://www.scbdd.com/chemdes/)
           - Enter your SMILES structure
           - Select Chemopy descriptors
           - Calculate and download
        2. Upload the Chemopy descriptor file here
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload descriptor file (CSV format)",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            
            # Select model and features based on choice
            if model_choice == "BRF_MOE+DS_GA_84":
                current_model = model1
                current_scaler = scaler1
                features_list = features1
            else:
                current_model = model2
                current_scaler = scaler2
                features_list = features2
            
            # Process and validate features
            processed_data = process_features(data, model_choice, features_list)
            
            if processed_data is None:
                return
            
            # Display processed data preview
            st.subheader("Selected Features Preview")
            st.dataframe(processed_data.head())
            
            # Scale features
            scaled_data = current_scaler.transform(processed_data)
            
            # Make predictions
            probabilities = current_model.predict_proba(scaled_data)
            
            # Display results
            st.header("🎯 Prediction Results")
            results_df = pd.DataFrame({
                'Sample': range(1, len(processed_data) + 1),
                'Risk Class': ['High Risk' if p >= 0.5 else 'Low Risk' for p in probabilities[:, 1]],
                'Transfer Probability': [f"{p:.3f}" for p in probabilities[:, 1]]
            })
            st.dataframe(results_df)
            
            # SHAP Analysis
            st.header("🔍 SHAP Analysis")
            sample_idx = st.number_input(
                "Select sample to analyze:",
                min_value=0,
                max_value=len(processed_data)-1,
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
            
            # SHAP visualization
            st.subheader("Feature Impact Analysis")
            create_shap_force_plot(current_model, scaled_data, processed_data, sample_idx)
            
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
        # Show feature requirements
        if model_choice == "BRF_MOE+DS_GA_84":
            st.info("Please upload a file containing MOE and DS descriptors.")
            with st.expander("View required features"):
                st.write(features1)
        else:
            st.info("Please upload a file containing Chemopy descriptors from ChemDes.")
            with st.expander("View required features"):
                st.write(features2)

if __name__ == "__main__":
    main()
