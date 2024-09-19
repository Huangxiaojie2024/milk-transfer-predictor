import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 加载模型和标准化器
@st.cache_resource
def load_model():
    with open('best_estimator_GA.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# 读取CSV文件并进行标准化
def preprocess_input(csv_file, scaler):
    input_data = pd.read_csv(csv_file)
    if input_data.shape[1] != 84:
        st.error("CSV文件必须包含84个特征")
        return None
    scaled_data = scaler.transform(input_data)
    return scaled_data

# 生成SHAP force plot
def plot_shap_force(model, data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap.force_plot(explainer.expected_value[1], shap_values[1], data, matplotlib=True)
    plt.tight_layout()
    st.pyplot()

# Streamlit 应用程序的布局
st.title('化合物母乳转移预测')
st.write('请上传一个包含84个特征的CSV文件')

uploaded_file = st.file_uploader("上传CSV文件", type="csv")

if uploaded_file is not None:
    model, scaler = load_model()
    input_data = preprocess_input(uploaded_file, scaler)
    
    if input_data is not None:
        # 预测概率
        prediction_proba = model.predict_proba(input_data)
        st.write(f"预测的母乳转移概率为：{prediction_proba[:, 1]}")

        # 生成SHAP force plot
        st.write("生成SHAP Force Plot")
        plot_shap_force(model, input_data)
