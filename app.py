import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. 核心配置与中文字体设置
# ==========================================
st.set_page_config(
    page_title="肺动脉高压风险预测系统",
    page_icon="🏥",
    layout="wide"
)

# 解决Matplotlib中文显示问题
# 涵盖 Windows(SimHei, Microsoft YaHei) 和 Linux/Streamlit Cloud(WenQuanYi, Noto Sans)
font_list = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
matplotlib.rc('font', family=font_list[0]) # 强制 SHAP 使用该字体

# 自定义CSS样式 (美化界面，模仿学术期刊风格)
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1 { color: #2c3e50; font-weight: bold; font-family: "Microsoft YaHei", sans-serif; }
    h3 { color: #34495e; font-family: "Microsoft YaHei", sans-serif; }
    .stButton>button {
        background-color: #007bff; color: white; border-radius: 5px; height: 3em; width: 100%; font-size: 16px;
    }
    .report-box {
        border: 1px solid #e6e6e6; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .threshold-info {
        font-size: 12px; color: #666; text-align: center; margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 定义必要的类 (防止 AttributeError)
# ==========================================
class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X)

# ==========================================
# 3. 加载模型与特征
# ==========================================
@st.cache_resource
def load_model_and_features():
    # 加载模型
    try:
        model = joblib.load('final_model_RF.pkl')
    except FileNotFoundError:
        st.error("❌ 错误：未找到模型文件 'final_model_RF.pkl'。")
        return None, None

    # 加载特征名称
    try:
        with open('selected_features_1SE_建模数据.txt', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if ',' in content:
                features = [x.strip() for x in content.split(',')]
            else:
                features = [x.strip() for x in content.split('\n')]
    except FileNotFoundError:
        st.error("❌ 错误：未找到特征文件 'selected_features_1SE_建模数据.txt'。")
        return None, None
        
    return model, features

model, feature_names = load_model_and_features()

# ==========================================
# 4. 侧边栏：中文输入界面
# ==========================================
if model and feature_names:
    st.sidebar.header("📋 患者参数录入")
    st.sidebar.markdown("请在下方输入临床特征值：")
    
    input_data = {}
    
    for feature in feature_names:
        # 简单的逻辑判断：如果是性别等分类变量，显示下拉框，否则显示数字框
        feature_lower = feature.lower()
        if 'sex' in feature_lower or 'gender' in feature_lower or 'code' in feature_lower:
            input_data[feature] = st.sidebar.selectbox(f"{feature} (分类变量)", options=[0, 1], index=0)
        else:
            input_data[feature] = st.sidebar.number_input(f"{feature} (数值)", value=0.0, format="%.2f")

    # 转换为 DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)

# ==========================================
# 5. 主界面：预测逻辑
# ==========================================
st.title("🏥 肺动脉高压风险智能预测系统")
st.markdown("基于机器学习随机森林算法构建 | 仅供科研参考")
st.markdown("---")

if st.sidebar.button("🔍 开始预测风险"):
    if model and feature_names:
        with st.spinner('正在计算模型预测概率与 SHAP 解释值，请稍候...'):
            
            # ---------------------------
            # A. 计算预测概率
            # ---------------------------
            try:
                # 尝试获取阳性概率 (索引1)
                probability = model.predict_proba(input_df)[0, 1]
            except:
                # 如果不支持概率，直接取预测类别
                prediction = model.predict(input_df)[0]
                probability = 1.0 if prediction == 1 else 0.0

            # ---------------------------
            # B. 计算 SHAP 值 (Pipeline 兼容版)
            # ---------------------------
            shap_values_to_plot = None
            try:
                # 1. 处理 Pipeline 结构：拆解出预处理器和最终模型
                if hasattr(model, 'steps') or hasattr(model, 'named_steps'):
                    final_estimator = model._final_estimator
                    preprocessor = model[:-1]
                    
                    # 预处理数据
                    processed_data = preprocessor.transform(input_df)
                    if hasattr(processed_data, "toarray"):
                        processed_data = processed_data.toarray()
                    
                    # 尝试恢复特征名
                    try:
                        processed_feature_names = preprocessor.get_feature_names_out()
                        data_for_shap = pd.DataFrame(processed_data, columns=processed_feature_names)
                    except:
                        data_for_shap = pd.DataFrame(processed_data)
                else:
                    # 非 Pipeline 模型
                    final_estimator = model
                    data_for_shap = input_df

                # 2. 使用 TreeExplainer 解释
                try:
                    explainer = shap.TreeExplainer(final_estimator)
                    shap_values = explainer(data_for_shap)
                except Exception:
                    # 备用方案：如果直接解释失败，尝试传入数据背景
                    explainer = shap.TreeExplainer(final_estimator, data=data_for_shap, model_output="probability")
                    shap_values = explainer(data_for_shap)

                # 3. 提取绘图数据 (针对二分类模型取正类)
                if len(shap_values.values.shape) == 3:
                    shap_values_to_plot = shap_values[0, :, 1] # 取正类
                else:
                    shap_values_to_plot = shap_values[0]

            except Exception as e:
                st.error(f"SHAP 计算模块出错: {str(e)}")
                shap_values_to_plot = None

            # ---------------------------
            # C. 结果展示区域 (基于约登指数更新)
            # ---------------------------
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 📊 预测风险评分")
                risk_percent = probability * 100
                
                # ==============================================
                # 🔴 最佳截断值设定 (基于约登指数)
                # ==============================================
                # 你的截图显示最佳阈值为 0.35703
                optimal_threshold = 35.703 
                youden_index = 0.771
                # ==============================================

                if risk_percent > optimal_threshold:
                    # === 高风险设置 ===
                    color = "#dc3545"  # 红色
                    risk_label = "高风险 (High Risk)"
                    icon = "⚠️"
                    advice_box = "warning" # Streamlit 组件类型
                    advice_text = f"模型预测概率 ({risk_percent:.1f}%) 已超过最佳截断值 ({optimal_threshold:.1f}%)。\n\n**建议：** 考虑进行超声心动图或右心导管检查以进一步确诊。"
                else:
                    # === 低风险设置 ===
                    color = "#28a745"  # 绿色
                    risk_label = "低风险 (Low Risk)"
                    icon = "✅"
                    advice_box = "success"
                    advice_text = f"模型预测概率 ({risk_percent:.1f}%) 低于最佳截断值 ({optimal_threshold:.1f}%)。\n\n**建议：** 目前风险较低，建议按常规流程进行随访。"
                
                # 展示大数字仪表盘
                st.markdown(
                    f"""
                    <div class="report-box" style="text-align: center; border-left: 5px solid {color};">
                        <h2 style="color: {color}; font-size: 50px; margin: 0;">{risk_percent:.1f}%</h2>
                        <p style="color: gray; font-size: 14px; margin-bottom: 5px;">患病概率 (Probability)</p>
                        <div class="threshold-info">
                            Optimal Cut-off: {optimal_threshold:.3f}%<br>
                            (Youden Index: {youden_index})
                        </div>
                        <hr style="margin: 15px 0;">
                        <h3 style="color: {color}; margin: 0;">{icon} {risk_label}</h3>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # 展示临床建议
                st.markdown("### 🩺 决策建议")
                if advice_box == "warning":
                    st.warning(advice_text)
                else:
                    st.success(advice_text)

            with col2:
                st.markdown("### 🔍 SHAP 可解释性分析 (力图)")
                st.markdown("下图展示了各特征如何推动风险值 **升高 (红色)** 或 **降低 (蓝色)**：")
                
                if shap_values_to_plot is not None:
                    try:
                        # 绘制力图 (Force Plot)
                        fig, ax = plt.subplots(figsize=(12, 4))
                        shap.plots.force(shap_values_to_plot, matplotlib=True, show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as plot_err:
                         st.error(f"绘图失败，可能是字体配置问题: {plot_err}")
                else:
                    st.warning("无法生成 SHAP 图，请检查输入数据或模型结构。")
            
            st.markdown("---")
            st.caption(f"**说明：** 本工具采用约登指数 (Youden Index = {youden_index}) 确定的最佳截断值 {optimal_threshold/100:.5f} 进行风险分层。结果仅供科研参考。")
    else:
        st.error("系统错误：模型未加载。")
else:
    st.info("👈 请在左侧侧边栏输入患者的临床参数，然后点击“开始预测风险”按钮。")
