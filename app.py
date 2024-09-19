import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import streamlit.components.v1 as components

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
            scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
            st.subheader("标准化后的特征预览")
            st.dataframe(scaled_df.head())

            # 预测概率
            probabilities = model.predict_proba(scaled_data)
            prob_df = pd.DataFrame(probabilities, columns=["Probability_0", "Probability_1"])
            st.subheader("预测概率")
            st.dataframe(prob_df.head())

            # 选择要查看 SHAP 力图的样本
            st.sidebar.header("SHAP 力图选项")
            sample_index = st.sidebar.number_input(
                "选择样本索引（从0开始）",
                min_value=0,
                max_value=len(data)-1,
                value=0,
                step=1
            )
            class_index = st.sidebar.selectbox(
                "选择要解释的类别",
                options=[0, 1],
                index=1,  # 默认选择类别1
                format_func=lambda x: f"类别 {x}"
            )

            if st.sidebar.button("显示 SHAP 力图"):
                # 选择单个样本
                single_sample = scaled_data[sample_index].reshape(1, -1)

                # 使用 SHAP 解释模型
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(single_sample)

                # 提取指定类别和样本的 SHAP 值
                shap_value = shap_values[0][:, class_index] # 提取类别对应的 SHAP 值
                base_value = float(explainer.expected_value[class_index])

                # 创建 SHAP force plot
                st.subheader(f"SHAP Force Plot - 样本索引 {sample_index}（类别 {class_index}）")
                shap.initjs()  # 初始化 JavaScript 库

                # 生成 force plot 并保存为 HTML
                force_plot = shap.force_plot(
                    base_value,
                    shap_value,
                    single_sample[0],
                    feature_names=data.columns,
                    matplotlib=False
                )

                # 保存为 HTML 文件
                html_file = f"force_plot_{sample_index}.html"
                shap.save_html(html_file, force_plot)

                # 在 Streamlit 中显示 HTML
                with open(html_file) as f:
                    components.html(f.read(), height=500, scrolling=True)

    except Exception as e:
        st.error(f"文件处理出现错误: {e}")
else:
    st.info("请在左侧上传一个包含84个特征的CSV文件。")
