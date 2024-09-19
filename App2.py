# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息

# 设置页面配置
st.set_page_config(
    page_title='母乳转移预测应用',
    layout='wide'
)

# 定义一个函数来加载模型和标准化器，并使用缓存以提高性能
@st.cache(allow_output_mutation=True)
def load_model_and_scaler():
    # 加载预训练的平衡随机森林模型
    with open('best_estimator_GA.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    # 加载标准化器
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# 调用函数加载模型和标准化器
model, scaler = load_model_and_scaler()

# 定义一个辅助函数来在 Streamlit 中显示 SHAP 图
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# 主函数
def main():
    st.title('母乳转移预测应用')

    st.markdown("""
    请上传包含新药物84个特征描述符的CSV文件，应用程序将预测其在母乳中的转移概率，并生成SHAP力量图以解释模型的预测。
    """)

    # 文件上传
    uploaded_file = st.file_uploader('选择一个CSV文件', type='csv')

    if uploaded_file is not None:
        # 读取上传的CSV文件
        data = pd.read_csv(uploaded_file)

        # 检查是否包含84个特征
        if data.shape[1] != 84:
            st.error(f'上传的文件包含 {data.shape[1]} 个特征，必须包含84个特征。')
            return

        # 显示原始数据
        st.subheader('原始数据')
        st.write(data.head())

        # 特征标准化
        X_scaled = scaler.transform(data)

        # 预测概率
        probabilities = model.predict_proba(X_scaled)[:, 1]  # 获取类别1的概率

        # 显示预测结果
        st.subheader('预测结果')
        for idx, prob in enumerate(probabilities):
            st.write(f'样本 {idx + 1} 的转移概率: {prob:.4f}')

        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        # 显示SHAP力量图
        st.subheader('SHAP 力量图')
        st.write('以下是第一个样本的SHAP力量图：')

        # 初始化JavaScript可视化
        shap.initjs()

        # 生成SHAP力量图
        force_plot = shap.force_plot(
            explainer.expected_value[1],  # 类别1的期望值
            shap_values[1][0, :],         # 第一个样本的SHAP值
            data.iloc[0, :]               # 第一个样本的原始特征值
        )

        # 在Streamlit中显示SHAP力量图
        st_shap(force_plot)

    else:
        st.info('请上传一个CSV文件以继续。')

if __name__ == '__main__':
    main()
