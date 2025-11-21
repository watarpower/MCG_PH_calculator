import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameConverter(TransformerMixin, BaseEstimator):
    def __init__(self, feature_names): self.feature_names = feature_names

    def fit(self, X, y=None): return self

    def transform(self, X): return pd.DataFrame(X, columns=self.feature_names)
    pass

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
            shap_values_to_plot = None
            try:
                # --- 关键修改：处理 Pipeline 结构 ---
                # 1. 判断模型是否为 Pipeline
                if hasattr(model, 'steps') or hasattr(model, 'named_steps'):
                    # 提取流水线中最后一个步骤，即真正的机器学习模型
                    final_estimator = model._final_estimator
                    
                    # 提取预处理器（除了最后一步之外的所有步骤）
                    preprocessor = model[:-1]
                    
                    # 使用预处理器将用户的原始输入 (input_df) 转换为模型能看懂的格式
                    processed_data = preprocessor.transform(input_df)
                    
                    # 如果预处理结果是稀疏矩阵，转换为普通数组
                    if hasattr(processed_data, "toarray"):
                        processed_data = processed_data.toarray()
                    
                    # 尝试获取处理后的特征名称（为了让 SHAP 图的标签更准确）
                    try:
                        processed_feature_names = preprocessor.get_feature_names_out()
                        data_for_shap = pd.DataFrame(processed_data, columns=processed_feature_names)
                    except:
                        # 如果获取失败，就用没有列名的 DataFrame，SHAP图照样能出
                        data_for_shap = pd.DataFrame(processed_data)
                        st.warning("提示：未能自动获取预处理后的特征名，SHAP图可能显示通用名称。")

                else:
                    # 如果不是 Pipeline，说明加载的就是单纯的模型
                    final_estimator = model
                    data_for_shap = input_df
                # --- 修改结束 ---

                # 2. 定义解释器 (针对最终模型)
                # 使用 TreeExplainer 来解释这个提取出来的 final_estimator
                try:
                    # 多数情况直接用这个
                    explainer = shap.TreeExplainer(final_estimator)
                    shap_values = explainer(data_for_shap)
                except Exception:
                    # 如果是复杂的 VotingClassifier，有时需要提供数据背景才能工作
                    explainer = shap.TreeExplainer(final_estimator, data=data_for_shap, model_output="probability")
                    shap_values = explainer(data_for_shap)

                # 3. 提取用于绘图的 SHAP 值
                # SHAP 的结果形状可能是 (样本数, 特征数, 类别数)，例如 (1, 15, 2)
                # 我们需要提取正类（风险升高那一类，索引为1）的贡献值
                if len(shap_values.values.shape) == 3:
                    # 取出第0个样本，所有特征，第1个类别的值
                    shap_values_to_plot = shap_values[0, :, 1]
                else:
                    # 如果结果已经是二维的，直接取第0个样本
                    shap_values_to_plot = shap_values[0]

            except Exception as e:
                # 打印详细的错误信息以便调试
                st.error(f"SHAP计算出错详细信息: {str(e)}")
                st.info("💡 分析：您的模型是一个复杂的 Pipeline。如果上面的错误提示与 'VotingClassifier' 或 'data' 有关，说明 SHAP 难以直接解释该投票模型。")
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

