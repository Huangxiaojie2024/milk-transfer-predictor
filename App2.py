import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np

# 加载模型和标准化工具
with open('best_estimator_GA.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# 创建Streamlit界面
st.title('Compound Transfer Prediction in Breast Milk')

# 上传CSV文件
uploaded_file = st.file_uploader("Upload your CSV file with 84 features", type=['csv'])
if uploaded_file is not None:
    # 读取CSV文件
    data = pd.read_csv(uploaded_file)
    
    # 特征标准化
    scaled_data = scaler.transform(data)
    
    # 进行预测
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)
    
    # 显示预测结果
    st.write(f'Prediction: {prediction[0]}')
    st.write(f'Probability: {probability[0][1]}')
    
    # 计算SHAP值
    explainer = shap.Explainer(model, scaled_data)
    shap_values = explainer(scaled_data)
    
    # 显示SHAP force plot
    st.write(shap.plots.force(explainer.expected_value, shap_values.values, scaled_data, feature_names=data.columns))
