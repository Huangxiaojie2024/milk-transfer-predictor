def calculate_shap_values(model: Any, features_scaled: pd.DataFrame) -> Tuple[Any, np.ndarray]:
    """Calculate SHAP values for all instances"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_scaled)
    return explainer, shap_values

def main():
    model, scaler, selected_features = load_resources()
    
    # Page header
    st.title("🧬 Chemical Transfer into Human Milk Predictor")
    st.markdown("---")

    # ... [其他代码保持不变]

    if uploaded_file is not None:
        try:
            # 加载和处理数据
            df = pd.read_csv(uploaded_file)
            
            with st.expander("View Raw Data", expanded=False):
                st.dataframe(df)
            
            # 保存原始特征值用于SHAP可视化
            df_original = df[selected_features].copy()
            
            # 标准化特征用于预测
            df_scaled = process_features(df, scaler, selected_features)
            
            with st.expander("View Processed Data", expanded=False):
                st.dataframe(df_scaled)
            
            # 预测和计算SHAP值
            predictions = model.predict_proba(df_scaled)
            explainer, shap_values = calculate_shap_values(model, df_scaled)
            
            # Results section
            st.header("🎯 Results")
            
            # Prediction results
            results_df = pd.DataFrame({
                'Molecule ID': range(1, len(df) + 1),
                'Prediction': ['High Risk' if p > 0.5 else 'Low Risk' for p in predictions[:, 1]],
                'Risk Probability': [f"{p:.3f}" for p in predictions[:, 1]]
            })
            st.subheader("Prediction Results")
            st.dataframe(results_df)
            
            # Download results button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )
            
            # SHAP Analysis
            st.header("🔍 SHAP Analysis")
            
            molecule_idx = st.selectbox(
                "Select molecule for detailed analysis",
                options=range(len(df)),
                format_func=lambda x: f"Molecule {x+1}"
            )
            
            # Prediction info
            risk_prob = predictions[molecule_idx, 1]
            risk_color = "red" if risk_prob > 0.5 else "green"
            st.markdown(f"""
                ### Molecule {molecule_idx + 1}
                Risk Probability: <span style='color:{risk_color};font-weight:bold'>{risk_prob:.3f}</span>
                """, unsafe_allow_html=True)
            
            # SHAP force plot using original values
            st.subheader("Feature Contribution Analysis")
            
            # 使用shap原生的force_plot
            st_shap = st.container()
            with st_shap:
                shap.initjs()  # 初始化JavaScript
                force_plot = shap.force_plot(
                    explainer.expected_value[1],
                    shap_values[1][molecule_idx, :],  # 使用类别1的SHAP值
                    df_original.iloc[molecule_idx, :],  # 使用原始特征值
                    link='logit',  # 使用logit链接函数
                    matplotlib=False  # 不使用matplotlib版本
                )
                shap.save_html(f"shap_plot_{molecule_idx}.html", force_plot)
                
                # 读取并显示HTML
                with open(f"shap_plot_{molecule_idx}.html", 'r', encoding='utf-8') as f:
                    html = f.read()
                st.components.v1.html(html, height=200)
            
            # Feature importance table
            with st.expander("View Feature Importance", expanded=True):
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Original_Value': df_original.iloc[molecule_idx],
                    'SHAP_Value': shap_values[1][molecule_idx, :],
                    'Absolute_SHAP_Value': np.abs(shap_values[1][molecule_idx, :])
                })
                importance_df = importance_df.sort_values('Absolute_SHAP_Value', ascending=False)
                st.dataframe(importance_df)
                
        except Exception as e:
            st.error("❌ Error Processing Data")
            st.error(f"Details: {str(e)}")
            st.exception(e)
