import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import BalancedRandomForestClassifier

# 设置页面标题和布局
st.set_page_config(page_title="化合物母乳转移预测", layout="wide")

st.title("化合物母乳转移预测应用")

# 加载模型和标准化器
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_resource
def load_scaler(scaler_path):
    return joblib.load(scaler_path)

model = load_model("best_estimator_GA.pkl")
scaler = load_scaler("scaler.pkl")

# 文件上传
st.sidebar.header("上传新药化合物数据")
uploaded_file = st.sidebar.file_uploader("上传包含84个特征的CSV文件", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        # 检查是否包含84个特征
        if data.shape[1] != 84:
            st.error(f"上传的文件包含 {data.shape[1]} 个特征，预期需要84个特征。请检查文件。")
        else:
            st.subheader("上传的数据预览")
            st.dataframe(data.head())

            # 特征标准化
            scaled_data = scaler.transform(data)
            st.subheader("标准化后的特征预览")
            st.dataframe(pd.DataFrame(scaled_data, columns=data.columns).head())

            # 预测概率
            probabilities = model.predict_proba(scaled_data)
            data["Probability_0"] = probabilities[:, 0]
            data["Probability_1"] = probabilities[:, 1]
            st.subheader("预测概率")
            st.dataframe(data[["Probability_0", "Probability_1"]].head())

            # 选择要查看 SHAP 力图的样本
            st.sidebar.header("SHAP 力图选项")
            sample_index = st.sidebar.number_input(
                "选择样本索引（从0开始）",
                min_value=0,
                max_value=len(data)-1,
                value=0,
                step=1
            )

            if st.sidebar.button("显示 SHAP 力图"):
                # 使用 SHAP 解释模型
                explainer = shap.Explainer(model, scaler.transform(data))
                shap_values = explainer(scaled_data)

                # 绘制 SHAP 力图
                st.subheader(f"SHAP 力图 - 样本索引 {sample_index}")
                shap.initjs()
                fig, ax = plt.subplots()
                shap.plots._waterfall.waterfall_legacy(shap_values[sample_index], max_display=10, show=False)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"文件处理出现错误: {e}")
else:
    st.info("请在左侧上传一个包含84个特征的CSV文件。")
