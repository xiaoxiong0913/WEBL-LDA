import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings

# 处理版本警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 定义特征名称
feature_names = [
    'Lym（10^9/L）',
    'Hb(g/L)',
    'Alb(g/L)',
    'reperfusiontherapy(yes1，no0)',
    'ECMO(yes1,no0)',
    'ACEI/ARB(yes1,no0)'
]

# 加载模型和标准化器
model_path = 'lda_model.pkl'
scaler_path = 'scaler.pkl'

with open(model_path, 'rb') as model_file, open(scaler_path, 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# 尝试从标准化器中获取特征名称
try:
    feature_names = scaler.feature_names_in_
except AttributeError:
    pass

# 定义风险阈值
risk_cutoff = 0.479

# 创建 Streamlit Web 应用程序标题
st.title(
    'A machine learning-based model to predict 1-year risk among patients with acute myocardial infarction and cardiogenic shock '
)

# 介绍部分
st.markdown("""
## Introduction
This web-based calculator was developed based on the Linear Discriminant Analysis (LDA) model. Users can obtain the one-year risk of death for a particular case by selecting parameters and clicking the "Calculate" button.
""")

# 创建输入表单
st.markdown("## Selection Panel")
st.markdown("Picking up parameters")

with st.form("prediction_form"):
    lym = st.slider('Lym (10^9/L)', min_value=0.0, max_value=8.0, value=1.0, step=0.1, key='Lym（10^9/L）')
    hb = st.slider('Hb (g/L)', min_value=0.0, max_value=200.0, value=100.0, step=1.0, key='Hb(g/L)')
    alb = st.slider('Alb (g/L)', min_value=0.0, max_value=50.0, value=25.0, step=0.1, key='Alb(g/L)')
    reperfusion_therapy = st.selectbox('Reperfusion Therapy', options=[0, 1],
                                       format_func=lambda x: 'Yes' if x == 1 else 'No',
                                       key='reperfusiontherapy(yes1，no0)')
    ecmo = st.selectbox('ECMO', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='ECMO(yes1,no0)')
    acei_arb = st.selectbox('ACEI/ARB', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No',
                            key='ACEI/ARB(yes1,no0)')

    submit_button = st.form_submit_button("Predict")

# 定义正常值范围
normal_ranges = {
    "Lym（10^9/L）": (0.8, 4.0),
    "Hb(g/L)": (120, 170),
    "Alb(g/L)": (35, 50)
}

# 处理表单提交
if submit_button:
    data = {
        "Lym（10^9/L）": lym,
        "Hb(g/L)": hb,
        "Alb(g/L)": alb,
        "reperfusiontherapy(yes1，no0)": reperfusion_therapy,
        "ECMO(yes1,no0)": ecmo,
        "ACEI/ARB(yes1,no0)": acei_arb,
    }

    try:
        # 将输入数据转换为 DataFrame，使用精确的特征名称和顺序
        data_df = pd.DataFrame([data], columns=feature_names)

        # 使用加载的标准化器对数据进行标准化
        data_scaled = scaler.transform(data_df)

        # 进行预测
        prediction = model.predict_proba(data_scaled)[:, 1][0]  # 获取类别 1 的概率

        # 显示预测结果
        st.subheader("Prediction Result:")
        st.write(f'Prediction: {prediction * 100:.2f}%')

        # 风险分层和个性化建议
        if prediction >= risk_cutoff:
            st.markdown("<span style='color:red'>High risk: This patient is classified as a high-risk patient.</span>",
                        unsafe_allow_html=True)
            st.subheader("Personalized Recommendations:")

            # 对数值进行建议
            for feature, (normal_min, normal_max) in normal_ranges.items():
                value = data[feature]
                if value < normal_min:
                    st.markdown(
                        f"<span style='color:red'>{feature}: Your value is {value}. It is lower than the normal range ({normal_min} - {normal_max}). Consider increasing it towards {normal_min}.</span>",
                        unsafe_allow_html=True)
                elif value > normal_max:
                    st.markdown(
                        f"<span style='color:red'>{feature}: Your value is {value}. It is higher than the normal range ({normal_min} - {normal_max}). Consider decreasing it towards {normal_max}.</span>",
                        unsafe_allow_html=True)
                else:
                    st.write(f"{feature}: Your value is within the normal range ({normal_min} - {normal_max}).")

            # 对治疗和疗法的建议
            if reperfusion_therapy == 0:
                st.write("Consider undergoing reperfusion therapy.")
            if ecmo == 0:
                st.write("Consider ECMO therapy for better management.")
            if acei_arb == 0:
                st.write("Consider using ACEI/ARB medication.")

        else:
            st.markdown("<span style='color:green'>Low risk: This patient is classified as a low-risk patient.</span>",
                        unsafe_allow_html=True)

    except Exception as e:
        st.error(f'Error: {str(e)}')
