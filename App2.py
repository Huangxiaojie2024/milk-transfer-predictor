def plot_shap_values(model, X, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 检查shap_values是否有足够的元素
    if len(shap_values) > 1:
        st.header("SHAP Force Plot for the First Sample")
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0, :], feature_names=feature_names, matplotlib=True)
        plt.savefig("shap_force_plot.png")
        st.image("shap_force_plot.png")
    else:
        st.error("SHAP values do not have enough elements for the force plot.")

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
