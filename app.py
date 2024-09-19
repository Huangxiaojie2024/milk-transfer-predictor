{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6eb7d21f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-09-19 11:02:24.073 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:29.805 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run D:\\ProgramData\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2024-09-19 11:02:29.808 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:29.809 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:29.811 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:29.813 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:30.330 Thread 'Thread-5': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:30.344 Thread 'Thread-5': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:30.413 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:30.415 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:30.417 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:30.418 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:30.420 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2024-09-19 11:02:30.423 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import pickle\n",
    "import shap\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Load model and scaler\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    with open('best_estimator_GA.pkl', 'rb') as f:\n",
    "        model = pickle.load(f)\n",
    "    with open('scaler.pkl', 'rb') as f:\n",
    "        scaler = pickle.load(f)\n",
    "    return model, scaler\n",
    "\n",
    "# Function to generate SHAP plots\n",
    "def plot_shap_values(model, X, feature_names):\n",
    "    explainer = shap.TreeExplainer(model)\n",
    "    shap_values = explainer.shap_values(X)\n",
    "    \n",
    "    st.header(\"SHAP Force Plot for the First Sample\")\n",
    "    shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0, :], feature_names=feature_names, matplotlib=True)\n",
    "    plt.savefig(\"shap_force_plot.png\")\n",
    "    st.image(\"shap_force_plot.png\")\n",
    "\n",
    "# Streamlit app\n",
    "st.title(\"Assessing Chemical Exposure Risk in Breastfeeding Infants: An Explainable Machine Learning Model for Human Milk Transfer Prediction\")\n",
    "\n",
    "# Load the model and scaler\n",
    "model, scaler = load_model()\n",
    "\n",
    "# File uploader for CSV input\n",
    "uploaded_file = st.file_uploader(\"Please upload a CSV file with 84 molecular descriptors\", type=\"csv\")\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    # Read CSV file\n",
    "    input_data = pd.read_csv(uploaded_file)\n",
    "    \n",
    "    # Display the first few rows of the uploaded file\n",
    "    st.write(\"Uploaded data preview:\")\n",
    "    st.write(input_data.head())\n",
    "    \n",
    "    # Ensure the input data has 84 descriptors\n",
    "    if input_data.shape[1] == 84:\n",
    "        # Normalize the input data\n",
    "        normalized_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)\n",
    "        \n",
    "        # Make predictions\n",
    "        predictions = model.predict(normalized_data)\n",
    "        prediction_probabilities = model.predict_proba(normalized_data)[:, 1]  # Probabilities\n",
    "        \n",
    "        # Show prediction probabilities\n",
    "        st.write(\"Breast milk transfer probabilities for the uploaded compounds:\")\n",
    "        st.write(prediction_probabilities)\n",
    "        \n",
    "        # Generate and display SHAP force plot for the first sample\n",
    "        plot_shap_values(model, normalized_data, feature_names=input_data.columns)\n",
    "    else:\n",
    "        st.error(\"The uploaded file does not have 84 molecular descriptors. Please check your file.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76d91774",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
