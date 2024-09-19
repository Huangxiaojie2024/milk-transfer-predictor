import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import BalancedRandomForestClassifier
import shap
import matplotlib.pyplot as plt
import base64
import io

st.set_page_config(page_title="化合物母乳转移预测", layout="wide")

st.title("化合物母乳转移预测")

st.write("""
    上传一个包含84个特征的CSV文件，系统将进行预测并展示结果和解释。
    - **文件要求**：
        - 包含84个特征，列名与训练时一致。
        - 每行代表一个化合物。
""")

@st.cache_resource
def load_model():
    model = joblib.load('best_estimator_GA.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

with st.spinner('加载模型...'):
    model, scaler = load_model()

uploaded_file = st.file_uploader("选择CSV文件", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("上传的数据预览：")
        st.dataframe(data.head())

        # 检查特征数量
        if data.shape[1] != 84:
            st.error(f"CSV文件应包含84个特征，但当前有{data.shape[1]}个特征。")
        else:
            # 特征标准化
            scaled_data = scaler.transform(data)
            st.success("特征标准化完成。")

            # 进行预测
            predictions = model.predict(scaled_data)
            prediction_proba = model.predict_proba(scaled_data)[:, 1]  # 假设1为正类

            # 显示预测结果
            result_df = data.copy()
            result_df['预测结果'] = predictions
            result_df['预测概率'] = prediction_proba

            st.write("预测结果：")
            st.dataframe(result_df)

            # 提供下载预测结果的功能
            csv = result_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="预测结果.csv">下载预测结果 CSV 文件</a>'
            st.markdown(href, unsafe_allow_html=True)

            # 计算和展示SHAP值
            if st.button("展示SHAP解释"):
                st.write("计算SHAP值，请稍候...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(scaled_data)

                st.write("展示前10个样本的SHAP力图：")
                shap.initjs()
                for i in range(min(10, len(data))):
                    st.write(f"样本 {i+1} 的SHAP解释：")
                    # 使用matplotlib绘图
                    plt.figure()
                    shap.force_plot(explainer.expected_value[1], shap_values[1][i], data.iloc[i], matplotlib=True, show=False)
                    st.pyplot(bbox_inches='tight', dpi=300)
                    plt.clf()
    except Exception as e:
        st.error(f"文件处理失败：{e}")
