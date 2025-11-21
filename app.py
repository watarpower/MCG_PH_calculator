import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面配置 (Nature 风格布局)
# ==========================================
st.set_page_config(
    page_title="ML Risk Calculator",
    page_icon="⚕️",
    layout="wide"
)

# 自定义CSS，模仿学术期刊的字体和简洁感
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
        font-weight: bold;
    }
    h3 {
        font-family: 'Arial', sans-serif;
        color: #34495e;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    .report-box {
        border: 1px solid #e6e6e6;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 2. 加载资源 (使用缓存加速)
# ==========================================
@st.cache_resource
def load_model_and_features():
    # 加载模型
    try:
        model = joblib.load('final_model_RF.pkl')
    except FileNotFoundError:
        st.error("错误：未找到 'final_model_RF.pkl' 文件。")
        return None, None

    # 加载特征名称
    try:
        with open('selected_features_1SE_建模数据.txt', 'r', encoding='utf-8') as f:
            # 假设特征是用逗号分隔 或者 换行符分隔
            content = f.read().strip()
            if ',' in content:
                features = [x.strip() for x in content.split(',')]
            else:
                features = [x.strip() for x in content.split('\n')]
    except FileNotFoundError:
        st.error("错误：未找到 'selected_features_1SE_建模数据.txt' 文件。")
        return None, None

    return model, features


model, feature_names = load_model_and_features()

# ==========================================
# 3. 侧边栏：患者参数输入
# ==========================================
if model and feature_names:
    st.sidebar.header("📋 Patient Data Input")
    st.sidebar.markdown("Please enter the clinical parameters:")

    input_data = {}

    # 自动生成输入框
    # 注意：由于txt只包含名字，无法自动判断是数字还是分类。
    # 这里默认使用数字输入框 (number_input)。
    # 如果你有分类变量（如0/1），建议手动修改下面的代码逻辑。

    for feature in feature_names:
        # 尝试根据特征名猜测是否为分类变量（这是一个简单的启发式规则，可修改）
        feature_lower = feature.lower()
        if 'sex' in feature_lower or 'gender' in feature_lower or 'history' in feature_lower or 'code' in feature_lower:
            # 认为是二分类变量 (示例)
            input_data[feature] = st.sidebar.selectbox(f"{feature}", options=[0, 1], index=0)
        else:
            # 认为是连续变量，默认值为0.0
            input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0, format="%.2f")

    # 转换为DataFrame，确保列顺序与训练时完全一致
    input_df = pd.DataFrame([input_data], columns=feature_names)

# ==========================================
# 4. 主界面：预测与解释
# ==========================================
st.title("🏥 PH Detection Calculator")
st.markdown("Based on MCG")
st.markdown("---")

if st.sidebar.button("Calculate Prediction"):
    if model and feature_names:
        with st.spinner('Calculating probabilities and SHAP values...'):

            # A. 预测概率
            # 注意：假设是二分类问题，取索引1为阳性概率
            try:
                probability = model.predict_proba(input_df)[0, 1]
            except:
                # 如果模型不支持predict_proba (很少见)，尝试predict
                prediction = model.predict(input_df)[0]
                probability = 1.0 if prediction == 1 else 0.0

            # B. SHAP 值计算
            # 针对随机森林使用 TreeExplainer
            try:
                # 注意：如果你的模型是 VotingClassifier，直接用TreeExplainer可能报错
                # 如果报错，尝试提取内部的 RF 模型： explainer = shap.TreeExplainer(model.estimators_[0])
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)

                # 针对二分类，TreeExplainer有时返回(n_samples, n_features, 2)，我们需要取正类[..., 1]
                if len(shap_values.values.shape) == 3:
                    shap_values_to_plot = shap_values[0, :, 1]
                else:
                    shap_values_to_plot = shap_values[0]

            except Exception as e:
                st.error(f"SHAP计算出错: {e}")
                shap_values_to_plot = None

            # ====================
            # 结果展示区域
            # ====================
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 📊 Risk Score")
                # 使用仪表盘样式或大号字体显示概率
                risk_percent = probability * 100

                color = "green"
                if risk_percent > 30: color = "orange"
                if risk_percent > 70: color = "red"

                st.markdown(
                    f"""
                    <div class="report-box" style="text-align: center;">
                        <h2 style="color: {color}; font-size: 48px; margin: 0;">{risk_percent:.1f}%</h2>
                        <p style="color: gray;">Predicted Probability</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### Interpretation")
                if risk_percent > 50:
                    st.warning("⚠️ High Risk Detected")
                else:
                    st.success("✅ Low Risk Detected")

            with col2:
                st.markdown("### 🔍 SHAP Interpretation")
                st.markdown("Contributing factors pushing the risk **higher (Red)** or **lower (Blue)**:")

                if shap_values_to_plot is not None:
                    # 绘制瀑布图
                    fig, ax = plt.subplots(figsize=(8, 6))
                    shap.plots.waterfall(shap_values_to_plot, show=False, max_display=10)
                    st.pyplot(fig)
                else:
                    st.warning("无法生成SHAP图，请检查模型兼容性。")

            # 添加免责声明
            st.markdown("---")
            st.caption(
                "**Disclaimer:** This tool is for research purposes only and should not be used as a substitute for professional medical advice.")
    else:
        st.error("Model not loaded properly.")
else:
    st.info("👈 Please input clinical parameters in the sidebar and click 'Calculate'.")