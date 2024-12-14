import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import streamlit.components.v1 as components
import base64

# Set page config
st.set_page_config(
    page_title="Chemical Transfer Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

# Enhanced CSS with all styles
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .title-text {
        color: #1e3d59;
        text-align: center;
        padding: 1.5rem;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle-text {
        color: #2b6777;
        text-align: center;
        padding: 0.8rem;
        font-size: 1.5rem;
        font-weight: 500;
    }
    .info-box {
        background-color: #e7f3fe;
        border-left: 6px solid #2196F3;
        padding: 1.2rem;
        margin: 1.2rem 0;
        border-radius: 0 5px 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 1.2rem;
        margin: 1.2rem 0;
        border-radius: 0 5px 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-box {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        background-color: white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .feature-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 1.2rem;
        margin: 1.2rem 0;
        border-radius: 0 5px 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .model-header {
        color: #2b6777;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .feature-list {
        list-style-type: none;
        padding-left: 0;
    }
    .feature-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
    .feature-list li:last-child {
        border-bottom: none;
    }
    .baby-icon {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .feature-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .feature-table th, .feature-table td {
        padding: 0.5rem;
        border: 1px solid #ddd;
        text-align: left;
    }
    .feature-table th {
        background-color: #f5f5f5;
    }
    .feature-expander {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# 加载模型和资源
@st.cache_resource
def load_resources():
    """Load all models and scalers"""
    # Model 1 (MOE+DS)
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
    
    # Model 2 (Chemopy)
    model2 = joblib.load("GA_chemopy_model.pkl")
    scaler2 = joblib.load("GA_chemopy_scaler.pkl")
    with open("GA_chemopy_features.pkl", 'rb') as f:
        features2 = joblib.load(f)
        
    return model1, scaler1, features1, model2, scaler2, features2

def process_features(data, features_list):
    """Extract and validate features"""
    available_features = list(set(features_list) & set(data.columns))
    missing_features = list(set(features_list) - set(data.columns))
    
    return data[features_list] if not missing_features else None, missing_features

def create_shap_force_plot(model, data_scaled, data_original, sample_idx):
    """Generate SHAP force plot with enhanced styling"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_scaled)
    
    force_plot = shap.force_plot(
        explainer.expected_value[1],
        shap_values[:,:,1][sample_idx],
        data_original.iloc[sample_idx,:],
        matplotlib=False,
        plot_cmap=["#ff0d57", "#1e88e5"]
    )
    
    shap.save_html(f"force_plot_{sample_idx}.html", force_plot)
    with open(f"force_plot_{sample_idx}.html", 'r', encoding='utf-8') as f:
        components.html(f.read(), height=500, scrolling=True)

def display_features_table(features_list, model_name):
    """Display features in a paginated table format"""
    features_df = pd.DataFrame(features_list, columns=['Feature Name'])
    features_df.index = range(1, len(features_df) + 1)
    
    st.markdown(f"### 📊 Optimal Feature Subset for {model_name}")
    st.markdown("""
    <div class='info-box'>
    These features were selected using genetic algorithm optimization to provide the best predictive performance.
    </div>
    """, unsafe_allow_html=True)
    
    # 添加搜索功能
    search_term = st.text_input("🔍 Search features", "")
    
    if search_term:
        filtered_df = features_df[features_df['Feature Name'].str.contains(search_term, case=False)]
    else:
        filtered_df = features_df
    
    # 分页显示
    page_size = st.selectbox("Features per page", [10, 20, 50, 100])
    total_pages = len(filtered_df) // page_size + (1 if len(filtered_df) % page_size != 0 else 0)
    page = st.number_input("Page", 1, total_pages, 1)
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered_df))
    
    st.dataframe(filtered_df.iloc[start_idx:end_idx], height=400)
    st.markdown(f"Showing features {start_idx + 1} to {end_idx} of {len(filtered_df)}")

def main():
    # Load resources
    model1, scaler1, features1, model2, scaler2, features2 = load_resources()
    
    # Enhanced Header with baby icon
    st.markdown("""
    <div class='baby-icon'>
        👶
    </div>
    <h1 class='title-text'>Chemical Transfer Risk Predictor for Human Milk</h1>
    """, unsafe_allow_html=True)
    st.markdown("<h3 class='subtitle-text'>Advanced Machine Learning Models for Assessing Chemical Exposure Risk in Breastfeeding Infants</h3>", unsafe_allow_html=True)
    
    # Enhanced Introduction with SHAP explanation
    st.markdown("""
    <div class='info-box'>
    <h4>Welcome to the Chemical Transfer Risk Predictor!</h4>
    This advanced tool leverages state-of-the-art Balanced Random Forest (BRF) models to assess the risk 
    of chemical transfer into human breast milk. Choose between two specialized models, each optimized 
    for different molecular descriptor sets. The tool provides SHAP (SHapley Additive exPlanations) force plots 
    for interpretable visualization analysis, helping you understand which molecular features contribute most 
    significantly to the prediction results.
    </div>
    """, unsafe_allow_html=True)
    
    # Main Layout
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("### 🎯 Model Selection")
        
        model_choice = st.radio(
            "Select Prediction Model",
            ["BRF_MOE+DS_GA_84", "BRF_Chemopy_GA_101"]
        )
        
        # Enhanced Model Information Boxes
        if model_choice == "BRF_MOE+DS_GA_84":
            st.markdown("""
            <div class='feature-box'>
            <h4 class='model-header'>BRF_MOE+DS_GA_84 Model</h4>
            <ul class='feature-list'>
            <li>🔬 Based on Balanced Random Forest algorithm</li>
            <li>📊 Requires molecular descriptors from MOE and Discovery Studio software</li>
            <li>🎯 Automatically selects optimal 84 descriptors using genetic algorithm</li>
            <li>⚖️ Balanced approach for reliable predictions</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='feature-box'>
            <h4 class='model-header'>BRF_Chemopy_GA_101 Model</h4>
            <ul class='feature-list'>
            <li>🔬 Based on Balanced Random Forest algorithm</li>
            <li>🧪 Uses comprehensive Chemopy molecular descriptors</li>
            <li>🎯 Automatically selects best 101 descriptors via genetic algorithm</li>
            <li>🔗 Calculate descriptors at <a href='http://www.scbdd.com/chemdes/'>ChemDes</a></li>
            <li>⚖️ Balanced approach for reliable predictions</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature Subset Viewer
        with st.expander("📋 View Optimal Feature Subset"):
            if model_choice == "BRF_MOE+DS_GA_84":
                display_features_table(features1, "BRF_MOE+DS_GA_84")
            else:
                display_features_table(features2, "BRF_Chemopy_GA_101")
        
        # File Upload Section
        st.markdown("### 📤 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file with molecular descriptors",
            type=["csv"],
            help="Ensure your file contains all required molecular descriptors"
        )
        
        if uploaded_file:
            st.markdown("<div class='success-box'>✅ File successfully uploaded!</div>", unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            try:
                # Load and process data
                data = pd.read_csv(uploaded_file)
                
                # Data Preview
                st.markdown("### 📊 Data Preview")
                st.dataframe(data.head(), height=200)
                
                # Process features
                features_list = features1 if model_choice == "BRF_MOE+DS_GA_84" else features2
                processed_data, missing_features = process_features(data, features_list)
                
                if missing_features:
                    st.markdown(f"""
                    <div class='warning-box'>
                    ⚠️ Warning: {len(missing_features)} required features are missing. 
                    Please expand below to view details.
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("View Missing Features"):
                        st.write(missing_features)
                else:
                    # Model predictions
                    current_model = model1 if model_choice == "BRF_MOE+DS_GA_84" else model2
                    current_scaler = scaler1 if model_choice == "BRF_MOE+DS_GA_84" else scaler2
                    
                    scaled_data = current_scaler.transform(processed_data)
                    probabilities = current_model.predict_proba(scaled_data)
                    
                    # Enhanced Results Display
                    st.markdown("### 📈 Prediction Results")
                    results_df = pd.DataFrame({
                        'Sample': range(1, len(processed_data) + 1),
                        'Risk Classification': ['High Risk' if p >= 0.5 else 'Low Risk' for p in probabilities[:, 1]],
                        'Transfer Probability': [f"{p:.3f}" for p in probabilities[:, 1]]
                    })
                    
                    # Style the dataframe
                    def color_risk(val):
                        color = 'red' if val == 'High Risk' else 'green'
                        return f'color: {color}; font-weight: bold'
                    
                    styled_results = results_df.style.applymap(
                        color_risk, subset=['Risk Classification']
                    )
                    
                    st.dataframe(styled_results, height=300)
                    
                    # Results Summary
                    high_risk_count = sum(1 for p in probabilities[:, 1] if p >= 0.5)
                    low_risk_count = len(probabilities) - high_risk_count
                    
                    st.markdown("""
                    <div class='info-box'>
                    <h4>Results Summary</h4>
                    """, unsafe_allow_html=True)
                    
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    with col_metric1:
                        st.metric("Total Samples", len(probabilities))
                    with col_metric2:
                        st.metric("High Risk Compounds", high_risk_count)
                    with col_metric3:
                        st.metric("Low Risk Compounds", low_risk_count)
                    
                    # Download Results
                    st.download_button(
                        "📥 Download Prediction Results",
                        results_df.to_csv(index=False),
                        "chemical_transfer_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # SHAP Analysis Section
                    st.markdown("### 🔍 Detailed SHAP Analysis")
                    sample_idx = st.number_input(
                        "Select sample number for detailed analysis",
                        1, len(processed_data), 1
                    ) - 1  # Convert to 0-based index internally
                    
                    # Enhanced Prediction Info
                    prob = probabilities[sample_idx, 1]
                    risk_color = "red" if prob >= 0.5 else "green"
                    st.markdown(f"""
                    <div class='info-box'>
                    <h4>Analysis for Sample {sample_idx + 1}</h4>
                    <p style='font-size: 1.1em;'>
                        Transfer Probability: <span style='color:{risk_color};
                        font-weight:bold'>{prob:.3f}</span><br>
                        Classification: <span style='color:{risk_color};
                        font-weight:bold'>{"High Risk" if prob >= 0.5 else "Low Risk"}</span>
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Feature Importance Analysis
                    st.markdown("""
                    <div class='info-box'>
                    <h4>Feature Importance Visualization</h4>
                    <p>The SHAP force plot below shows how each feature contributes to the prediction:
                    <ul>
                    <li>Red push: Features increasing the risk prediction</li>
                    <li>Blue push: Features decreasing the risk prediction</li>
                    <li>Width: Magnitude of the feature's impact</li>
                    </ul>
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # SHAP Force Plot
                    st.markdown("#### 🎯 Feature Contribution Analysis")
                    create_shap_force_plot(current_model, scaled_data, processed_data, sample_idx)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.markdown("""
                <div class='warning-box'>
                ⚠️ Please check your input file format and try again. Ensure all required features are present.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='info-box'>
                <h4>Troubleshooting Tips</h4>
                <ul>
                <li>Verify that your CSV file contains all required features</li>
                <li>Check for any missing or incorrect values</li>
                <li>Ensure feature names match exactly with the required format</li>
                <li>Verify that all numerical values are properly formatted</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box'>
            <h4>Getting Started</h4>
            <p>1. Select your preferred prediction model</p>
            <p>2. View the required features in the feature subset viewer</p>
            <p>3. Prepare your CSV file with required molecular descriptors</p>
            <p>4. Upload your file to begin the analysis</p>
            <p>5. Explore SHAP visualization to understand prediction results</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
        
