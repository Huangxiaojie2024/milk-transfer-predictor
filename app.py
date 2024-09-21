import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import streamlit.components.v1 as components

# Set page title and layout
st.set_page_config(page_title="Prediction of Chemical Transfer to Breast Milk", layout="wide")

st.title("Chemical Transfer Predictor for Breast Milk")

# Expected molecular descriptors
expected_descriptors = [
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

# Display expected molecular descriptors as a DataFrame
st.subheader("Expected Molecular Descriptors")
expected_df = pd.DataFrame(expected_descriptors, columns=["Expected Descriptors"])
st.dataframe(expected_df)

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
st.sidebar.header("Upload New Chemical Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with 84 features", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Check if it contains 84 features
        if data.shape[1] != 84:
            st.error(f"The uploaded file contains {data.shape[1]} features, but 84 features are required. Please check the file.")
        else:
            # Check if the columns match the expected descriptors
            if not (list(data.columns) == expected_descriptors):
                st.error("The uploaded file does not contain the correct descriptors. Please ensure the columns match the expected list.")
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
                prob_df = pd.DataFrame(probabilities, columns=["Probability_0(low-risk)", "Probability_1(high-risk)"])
                st.subheader("Prediction Probabilities")
                st.dataframe(prob_df.head())

                # Select sample for SHAP force plot
                st.sidebar.header("SHAP Force Plot Options")
                sample_index = st.sidebar.number_input(
                    "Select Sample Index (starting from 0)",
                    min_value=0,
                    max_value=len(data)-1,
                    value=0,
                    step=1
                )
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
                    shap_value = shap_values[class_index]  # Extract SHAP values corresponding to the class
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

                    # Display the force plot in Streamlit
                    components.html(force_plot, height=500)

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a CSV file with 84 features on the left.")
