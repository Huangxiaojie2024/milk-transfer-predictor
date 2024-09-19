import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 加载模型和标准化器
model = joblib.load('best_estimator_GA.pkl')
scaler = joblib.load('scaler.pkl')

# 设置应用程序标题
st.title("BRF 预测与 SHAP 瀑布图可视化")

# 上传 CSV 文件
uploaded_file = st.file_uploader("上传一个包含 84 个特征的 CSV 文件", type="csv")

if uploaded_file:
    # 读取 CSV 文件
    input_data = pd.read_csv(uploaded_file)

    # 确保 CSV 文件有 84 个特征
    if input_data.shape[1] == 84:
        # 对输入特征进行标准化
        scaled_data = scaler.transform(input_data)

        # 生成预测
        prediction_probs = model.predict_proba(scaled_data)

        # 显示预测概率
        st.write("预测概率:", prediction_probs)

        # 使用 SHAP 解释模型预测
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_data)

        # 打印 SHAP 值的形状，确保获取到正确的值
        st.write("SHAP Values Shape:", [sv.shape for sv in shap_values])

        # 选择类别1的 SHAP 值
        st.write("类别 1 的 SHAP 瀑布图（第一条样本）:")

        # 绘制瀑布图，选择类别1
        shap_value_instance = shap_values[1][0]  # 选择第一条样本的 SHAP 值
        shap.initjs()
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_value_instance)
        st.pyplot(fig)

    else:
        st.error("上传的 CSV 必须包含 84 个特征。")
