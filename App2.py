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
@st.cache_resource
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
    shap_html = f"<head>{shap.getjs()}</head><body>{plot}</body>"
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
        try:
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

            # 检查shap_values的结构
            if isinstance(shap_values, list):
                # 多类别情况，选择第二个类别（索引1）
                if len(shap_values) > 1:
                    expected_value = explainer.expected_value[1]
                    shap_value = shap_values[1][0, :]
                else:
                    # 只有一个类别
                    expected_value = explainer.expected_value[0]
                    shap_value = shap_values[0][0, :]
            else:
                # 单数组情况
                expected_value = explainer.expected_value
                shap_value = shap_values[0, :]

            # 添加样本选择
            if data.shape[0] > 1:
                sample_idx = st.slider('选择要查看的样本', 1, data.shape[0], 1) - 1  # 从0开始索引
            else:
                sample_idx = 0

            # 获取选择的样本
            selected_data = data.iloc[sample_idx, :]
            if isinstance(shap_values, list) and len(shap_values) > 1:
                selected_shap_value = shap_values[1][sample_idx, :]
                selected_expected_value = explainer.expected_value[1]
            else:
                selected_shap_value = shap_values[0][sample_idx, :]
                selected_expected_value = explainer.expected_value

            # 显示SHAP力量图
            st.subheader('SHAP 力量图')
            st.write(f'以下是样本 {sample_idx + 1} 的SHAP力量图：')

            # 生成SHAP力量图
            force_plot = shap.plots.force(
                base_value=selected_expected_value, 
                shap_values=selected_shap_value, 
                features=selected_data,
                matplotlib=False  # 使用JS绘图
            )

            # 在Streamlit中显示SHAP力量图
            st_shap(force_plot, height=300)

        except Exception as e:
            st.error(f"发生错误: {e}")
    else:
        st.info('请上传一个CSV文件以继续。')

if __name__ == '__main__':
    main()
