import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 加载模型和标准化器
model = joblib.load('best_estimator_GA.pkl')
scaler = joblib.load('scaler.pkl')

st.title('化合物母乳转移预测')

# 上传CSV文件
uploaded_file = st.file_uploader("选择一个包含84个特征的CSV文件", type="csv")

if uploaded_file is not None:
    # 读取CSV文件
    data = pd.read_csv(uploaded_file)
    
    # 显示上传的数据
    st.write("上传的数据：")
    st.write(data)

    # 检查数据的维度
    if data.shape[1] == 84:
        # 特征标准化
        scaled_data = scaler.transform(data)

        # 进行预测
        predictions = model.predict_proba(scaled_data)[:, 1]  # 获取阳性类别的概率

        # 显示预测结果
        st.write("预测的阳性转移概率：")
        st.write(predictions)

        # 生成SHAP值
        explainer = shap.Explainer(model)
        shap_values = explainer(scaled_data)

        # 绘制SHAP force plot
        st.write("SHAP值可视化：")
        shap.initjs()

        for i in range(len(data)):
            fig = plt.figure()
            shap.plots.force(explainer.expected_value[0], shap_values, matplotlib=True, ax=fig)
            plt.savefig(f'shap_force_plot_{i}.png')  # 保存每个实例的SHAP force plot
            st.image(f'shap_force_plot_{i}.png')  # 显示SHAP force plot
            plt.clf()  # 清理当前图像

    else:
        st.error("上传的CSV文件特征数应为84个！")
