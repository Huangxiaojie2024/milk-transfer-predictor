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
    probabilities = model.predict_proba(scaled_data)
    
    # 创建SHAP解释器
    explainer = shap.Explainer(model.predict_proba, scaled_data)
    shap_values = explainer.shap_values(scaled_data)
    
    # 选择类别1的SHAP值
    shap_values_class_1 = shap_values[:, 1]  # 假设类别1的概率是第二列
    
    # 显示SHAP force plot
    for i in range(len(shap_values_class_1)):
        st.write(shap.plots.force(shap_values_class_1[i], scaled_data[i], feature_names=data.columns, show=False))
        st.pyplot()
