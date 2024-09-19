import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BalancedRandomForestClassifier
from io import StringIO

# 设置页面标题
st.title("化合物母乳转移预测")

# 文件上传
uploaded_file = st.file_uploader("上传包含84个特征的CSV文件", type=["csv"])

if uploaded_file is not None:
    try:
        # 读取CSV文件
        data = pd.read_csv(uploaded_file)
        
        # 检查特征数量
        if data.shape[1] != 84:
            st.error(f"CSV文件应包含84个特征，但当前有{data.shape[1]}个特征。")
        else:
            st.success("文件上传成功！预览数据如下：")
            st.dataframe(data.head())
            
            # 加载Scaler和模型
            scaler = joblib.load("scaler.pkl")
            model = joblib.load("best_estimator_GA.pkl")
            
            # 特征标准化
            data_scaled = scaler.transform(data)
            
            # 进行预测
            probabilities = model.predict_proba(data_scaled)[:, 1]  # 假设正类概率
            predictions = model.predict(data_scaled)
            
            # 添加预测结果到DataFrame
            data['预测概率'] = probabilities
            data['预测结果'] = predictions
            
            st.subheader("预测结果")
            st.dataframe(data[['预测概率', '预测结果']].head())
            
            # 选择样本进行SHAP分析
            sample_index = st.number_input(f"选择要查看SHAP力图的样本索引 (0 - {data.shape[0]-1})", min_value=0, max_value=data.shape[0]-1, step=1)
            
            if st.button("生成SHAP力图"):
                # 使用SHAP解释模型
                explainer = shap.Explainer(model, scaler.transform(data))
                shap_values = explainer(data_scaled)
                
                # 绘制力图
                st.subheader(f"样本 {sample_index} 的SHAP力图")
                shap.initjs()
                # 使用matplotlib生成力图
                fig, ax = plt.subplots()
                shap.force_plot(explainer.expected_value[1], shap_values[sample_index][1], data.iloc[sample_index], matplotlib=True, show=False, ax=ax)
                st.pyplot(fig, bbox_inches='tight', dpi=300)
                
    except Exception as e:
        st.error(f"发生错误: {e}")
