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
    return scaled_data, input_data.columns

# 生成 SHAP force plot 并保存为图片
def plot_shap_force(model, feature_values, feature_names):
    explainer = shap.TreeExplainer(model)
    
    # 计算 SHAP 值并生成 force plot
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    
    # 保存图片
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    
    # 展示图片
    st.image("shap_force_plot.png")

# Streamlit 应用程序的布局
st.title('化合物母乳转移预测')
st.write('请上传一个包含84个特征的CSV文件')

uploaded_file = st.file_uploader("上传CSV文件", type="csv")

if uploaded_file is not None:
    model, scaler = load_model()
    input_data, feature_names = preprocess_input(uploaded_file, scaler)
    
    if input_data is not None:
        # 预测概率
        prediction_proba = model.predict_proba(input_data)
        st.write(f"预测的母乳转移概率为：{prediction_proba[:, 1]}")

        # 生成 SHAP force plot 对每个样本单独生成
        for i in range(min(len(input_data), 5)):  # 只显示前5个样本的force plot
            plot_shap_force(model, input_data[i], feature_names)
