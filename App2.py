import streamlit as st
import pandas as pd
import joblib
import shap

# 加载模型和标准化工具
model = joblib.load('best_estimator_GA.pkl')
scaler = joblib.load('scaler.pkl')

# 设置应用标题
st.title("BRF预测和SHAP可视化")

# 上传CSV文件
uploaded_file = st.file_uploader("上传一个包含84个特征的CSV文件", type="csv")

if uploaded_file:
    # 读取CSV文件
    input_data = pd.read_csv(uploaded_file)

    # 检查CSV是否包含84个特征
    if input_data.shape[1] == 84:
        # 标准化输入特征
        scaled_data = scaler.transform(input_data)

        # 生成预测
        prediction_probs = model.predict_proba(scaled_data)

        # 显示预测概率
        st.write("预测概率：", prediction_probs)

        # 使用SHAP解释模型预测
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_data)

        # SHAP力图展示第一个实例（类别1）
        st.write("第一个实例的SHAP力图（类别1）：")
        force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], scaled_data[0, :])

        # 显示SHAP力图
        st_shap(force_plot)
    else:
        st.error("上传的CSV必须包含恰好84个特征。")
