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

# 自定义CSS，提升界面美观度
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
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .title-text {
        color: #1e3d59;
        text-align: center;
        padding: 1rem;
    }
    .subtitle-text {
        color: #2b6777;
        text-align: center;
        padding: 0.5rem;
    }
    .info-box {
        background-color: #e7f3fe;
        border-left: 6px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .feature-box {
        border: 1px solid #ddd;
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
        # 您的84个特征列表
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
    """Generate SHAP force plot"""
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
    
    # Page Header
    st.markdown("<h1 class='title-text'>🧬 Chemical Transfer Risk Predictor for Human Milk</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='subtitle-text'>Advanced Machine Learning Models for Predicting Chemical Exposure Risk</h3>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class='info-box'>
    This tool utilizes state-of-the-art machine learning models to predict the risk of chemical transfer 
    into human breast milk. It offers two complementary prediction models, each optimized for specific 
    types of molecular descriptors.
    </div>
    """, unsafe_allow_html=True)
    
    # Main Content Area - Using columns for layout
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("### Model Selection and Data Input")
        
        # Model Selection with Detailed Info
        model_choice = st.radio(
            "Select Prediction Model",
            ["BRF_MOE+DS_GA_84", "BRF_Chemopy_GA_101"]
        )
        
        # Model Information Box
        if model_choice == "BRF_MOE+DS_GA_84":
            st.markdown("""
            <div class='feature-box'>
            <h4>BRF_MOE+DS_GA_84 Model</h4>
            <ul>
            <li>84 optimized MOE and DS descriptors</li>
            <li>Genetic algorithm feature selection</li>
            <li>High accuracy on diverse chemical structures</li>
            <li>Balanced ensemble learning approach</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='feature-box'>
            <h4>BRF_Chemopy_GA_101 Model</h4>
            <ul>
            <li>101 Chemopy molecular descriptors</li>
            <li>Requires ChemDes calculation</li>
            <li>Visit <a href='http://www.scbdd.com/chemdes/'>ChemDes</a> for descriptors</li>
            <li>Optimized for comprehensive chemical space</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # File Upload Section
        st.markdown("### Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file with molecular descriptors",
            type=["csv"],
            help="Please ensure your file contains all required descriptors"
        )
        
        if uploaded_file:
            st.markdown("<div class='info-box'>✅ File uploaded successfully</div>", unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            try:
                # Load and process data
                data = pd.read_csv(uploaded_file)
                
                # Preview uploaded data
                st.markdown("### Data Preview")
                st.dataframe(data.head(), height=200)
                
                # Select features based on model
                features_list = features1 if model_choice == "BRF_MOE+DS_GA_84" else features2
                processed_data, missing_features = process_features(data, features_list)
                
                if missing_features:
                    st.markdown(f"""
                    <div class='warning-box'>
                    ⚠️ Missing {len(missing_features)} required features. 
                    Click to view missing features.
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("Missing Features"):
                        st.write(missing_features)
                else:
                    # Scale features and make predictions
                    current_model = model1 if model_choice == "BRF_MOE+DS_GA_84" else model2
                    current_scaler = scaler1 if model_choice == "BRF_MOE+DS_GA_84" else scaler2
                    
                    scaled_data = current_scaler.transform(processed_data)
                    probabilities = current_model.predict_proba(scaled_data)
                    
                    # Results Display
                    st.markdown("### 🎯 Prediction Results")
                    results_df = pd.DataFrame({
                        'Sample': range(1, len(processed_data) + 1),
                        'Risk Class': ['High Risk' if p >= 0.5 else 'Low Risk' for p in probabilities[:, 1]],
                        'Transfer Probability': [f"{p:.3f}" for p in probabilities[:, 1]]
                    })
                    st.dataframe(results_df)
                    
                    # Download Results
                    st.download_button(
                        "📥 Download Results",
                        results_df.to_csv(index=False),
                        "prediction_results.csv",
                        "text/csv"
                    )
                    
                    # SHAP Analysis
                    st.markdown("### 🔍 SHAP Analysis")
                    sample_idx = st.number_input(
                        "Select sample for detailed analysis",
                        0, len(processed_data)-1, 0
                    )
                    
                    # Prediction Info
                    prob = probabilities[sample_idx, 1]
                    st.markdown(f"""
                    <div class='info-box'>
                    <h4>Sample {sample_idx + 1} Analysis</h4>
                    <p>Transfer Probability: <span style='color:{"red" if prob >= 0.5 else "green"};
                    font-weight:bold'>{prob:.3f}</span></p>
                    <p>Classification: <span style='color:{"red" if prob >= 0.5 else "green"};
                    font-weight:bold'>{"High Risk" if prob >= 0.5 else "Low Risk"}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # SHAP Force Plot
                    st.markdown("#### Feature Contribution Analysis")
                    create_shap_force_plot(current_model, scaled_data, processed_data, sample_idx)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.exception(e)
        else:
            st.markdown("<div class='info-box'>⚙️ Upload your data to begin analysis</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
