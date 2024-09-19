import streamlit as st
import pandas as pd
import joblib
import shap

# 加载模型和标准化器
model = joblib.load('best_estimator_GA.pkl')
scaler = joblib.load('scaler.pkl')

# 创建Streamlit应用
def main():
    st.title('Compound Breast Milk Transfer Prediction')

    # 创建文件上传器
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        # 读取CSV文件
        data = pd.read_csv(uploaded_file)

        # 确保数据有84个特征
        if data.shape[1] == 84:
            # 特征标准化
            scaled_data = scaler.transform(data)

            # 进行预测
            prediction = model.predict_proba(scaled_data)[:, 1]  # 假设第二列是正类的概率

            # 生成SHAP值
            shap_values = shap.TreeExplainer(model).shap_values(scaled_data)

            # 显示预测概率
            st.write('Predicted Probability:', prediction)

            # 显示SHAP force plot
            st.subheader('SHAP Force Plot')
            shap.summary_plot(shap_values, scaled_data, feature_names=data.columns)

        else:
            st.error('The CSV file must contain exactly 84 features.')

if __name__ == '__main__':
    main()
