import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import streamlit.components.v1 as components

# Set page title and layout
st.set_page_config(page_title="化合物母乳转移预测", layout="wide")

st.title("化合物母乳转移预测应用")

# Load model and scaler
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_resource
def load_scaler(scaler_path):
    return joblib.load(scaler_path)

model = load_model("best_estimator_GA.pkl")
scaler = load_scaler("scaler.pkl")

# File upload
st.sidebar.header("上传新药化合物数据")
uploaded_file = st.sidebar.file_uploader("上传包含84个特征的CSV文件", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Check if the data has 84 features
        if data.shape[1] != 84:
            st.error(f"上传的文件包含 {data.shape[1]} 个特征，预期需要84个特征。请检查文件。")
        else:
            st.subheader("上传的数据预览")
            st.dataframe(data.head())

            # Feature standardization
            scaled_data = scaler.transform(data)
            scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
            st.subheader("标准化后的特征预览")
            st.dataframe(scaled_df.head())

            # Prediction probabilities
            probabilities = model.predict_proba(scaled_data)
            prob_df = pd.DataFrame(probabilities, columns=["Probability_0", "Probability_1"])
            st.subheader("预测概率")
            st.dataframe(prob_df.head())

            # Select sample index for SHAP force plot
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
                # Select a single sample
                single_sample = scaled_data[sample_index].reshape(1, -1)

                # Use SHAP to explain the model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(single_sample)

                # Ensure the sample index is within bounds for SHAP values
                if sample_index < len(shap_values[class_index]):
                    shap_value = shap_values[class_index][0]  # Correctly extract the SHAP values for the selected class
                    base_value = float(explainer.expected_value[class_index])

                    # Create SHAP force plot
                    st.subheader(f"SHAP Force Plot - 样本索引 {sample_index}（类别 {class_index}）")
                    shap.initjs()  # Initialize JavaScript library

                    # Generate force plot
                    force_plot = shap.force_plot(
                        base_value,
                        shap_value,
                        single_sample[0],
                        feature_names=data.columns,
                        matplotlib=False
                    )

                    # Save as HTML file
                    html_file = f"force_plot_{sample_index}.html"
                    shap.save_html(html_file, force_plot)

                    # Display HTML in Streamlit
                    with open(html_file) as f:
                        components.html(f.read(), height=500, scrolling=True)
                else:
                    st.error(f"无法获取样本索引 {sample_index} 的 SHAP 值，超出了范围。")

    except Exception as e:
        st.error(f"文件处理出现错误: {e}")
else:
    st.info("请在左侧上传一个包含84个特征的CSV文件。")
