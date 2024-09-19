import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 加载模型和标准化器
model = joblib.load('best_estimator_GA.pkl')
scaler = joblib.load('scaler.pkl')

# 设置应用标题
st.title("BRF 预测与 SHAP 可视化")

# 上传CSV文件
uploaded_file = st.file_uploader("上传一个包含 84 个特征的 CSV 文件", type="csv")

if uploaded_file:
    # 读取CSV文件
    input_data = pd.read_csv(uploaded_file)

    # 确保CSV有84个特征
    if input_data.shape[1] == 84:
        # 对输入数据进行标准化
        scaled_data = scaler.transform(input_data)

        # 生成预测
        prediction_probs = model.predict_proba(scaled_data)

        # 显示预测概率
        st.write("预测概率:", prediction_probs)

        # 使用SHAP解释模型预测
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_data)

        # 打印SHAP值的形状，确保正确获取值
        st.write("SHAP Values Shape:", [sv.shape for sv in shap_values])

        # 选择类别1的 SHAP 值
        st.write("类别 1 的 SHAP 力图（第一条样本）:")
        # 只选择类别1的SHAP值进行绘图
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], scaled_data[0, :], matplotlib=True)

        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.error("上传的 CSV 必须包含 84 个特征。")
