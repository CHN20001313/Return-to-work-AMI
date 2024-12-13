import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import time
# 加载 JSON 格式模型
model = xgb.Booster()
model.load_model("xgboost_model.json")

# 定义分类变量选项
occupation_options = {
    "Blue-collar workers": 0,
    "White-collar workers": 1,
    "Self-employed": 2,
    "Others": 3
}
income_options = {
    "<2500": 0,
    "2500-5000": 1,
    "5000-10000": 2,
    ">10000": 3
}
cpx_options = {
    "No": 0,
    "Yes": 1
}

# 设置页面宽度
st.set_page_config(layout="wide")

# 页面标题和简介
st.title("AMI Return to Work Probability Predictor")
st.markdown("""
This tool predicts the likelihood of returning to work after acute myocardial infarction (AMI) based on patient characteristics.

**Instructions:**
- Fill in your details on the left.
- Click **Predict** to see your return-to-work probability and recommendations.
""")

# 创建两列布局
col1, col2 = st.columns(2)

# 左侧输入区域
with col1:
    st.header("Input Features")
    occupation = st.selectbox("Occupation", options=list(occupation_options.keys()))
    income = st.selectbox("Income Level", options=list(income_options.keys()))
    cpx = st.selectbox("Phase II Cardiac Rehabilitation", options=list(cpx_options.keys()))
    FPG = st.number_input("Fasting Plasma Glucose (FPG, mmol/L)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=30, step=1)
    crea = st.number_input("Creatinine Level (umol/L)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)

    if st.button("Predict"):
        with st.spinner("Calculating..."):
            time.sleep(3)  # 模拟计算时间
        st.success("Calculation complete!")
        # 特征编码
        feature_names = ["Occupation", "Income Level", "Phase II CR", "FPG", "Age", "Creatinine Level"]
        encoded_features = [
            occupation_options[occupation],
            income_options[income],
            cpx_options[cpx],
            FPG, age, crea
        ]
        input_features = np.array(encoded_features).reshape(1, -1)
        dmatrix = xgb.DMatrix(input_features)

        # 预测概率
        probabilities = model.predict(dmatrix)
        predicted_probability = probabilities[0]

        # 风险分组逻辑
        if predicted_probability < 0.237482:
            risk_group = "Low Return to Work Probability"
            risk_color = "red"
            advice = (
                "You have a low probability of returning to work. Please consult a healthcare professional as soon as possible "
                "for detailed evaluation and treatment guidance."
            )
        elif 0.237482 <= predicted_probability <= 0.435475:
            risk_group = "Medium Return to Work Probability"
            risk_color = "yellow"
            advice = (
                "Your probability of returning to work is moderate. It is recommended to monitor your health closely "
                "and consider consulting a healthcare professional for further evaluation."
            )
        else:
            risk_group = "High Return to Work Probability"
            risk_color = "green"
            advice = "You have a high probability of returning to work."



        # 显示结果在右侧
        with col2:
            st.header("Prediction Results")
            #st.markdown(
             #   f"<h3 style='font-size:24px;'>Prediction Probability: {predicted_probability * 100:.2f}%</h3>",
              #  unsafe_allow_html=True
            #)
            st.markdown(
                f"<h3 style='font-size:24px; color:{risk_color};'>Risk Group: {risk_group}</h3>",
                unsafe_allow_html=True
            )
            st.write(advice)

            # SHAP 力图
            st.header(
                f"Based on feature values, predicted probability of Return to Work is {predicted_probability * 100:.2f}%")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pd.DataFrame(input_features, columns=feature_names))
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                pd.DataFrame(input_features, columns=feature_names),
                matplotlib=True
            )
            plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=1200)
            st.image("shap_force_plot.png", caption="Feature Contribution (SHAP Force Plot)")