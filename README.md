# Assessing Chemical Exposure Risk in Breastfeeding Infants: An Explainable Machine Learning Model for Human Milk Transfer Prediction

This application predicts the probability of compounds transferring to breast milk based on molecular descriptors using a pre-trained model. It also provides SHAP force plots to interpret the model's predictions.

## Files
- `app.py`: Main Streamlit application
- `best_estimator_GA.pkl`: Trained machine learning model
- `scaler.pkl`: Scaler for normalizing input data
- `requirements.txt`: Dependencies

## Usage
Upload a CSV file with 84 molecular descriptors, and the app will output the breast milk transfer probability and SHAP force plots for the first compound.
