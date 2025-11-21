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
font_list = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False 
matplotlib.rc('font', family=font_list[0])

# 自定义CSS样式
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
# 2. 定义必要的类
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
    try:
        model = joblib.load('final_model_RF.pkl')
    except FileNotFoundError:
        st.error("❌ 错误：未找到模型文件 'final_model_RF.pkl'。")
        return None, None

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
# 4. 侧边栏：输入界面
# ==========================================
if model and feature_names:
    st.sidebar.header("📋 患者参数录入")
    st.sidebar.markdown("请在下方输入临床特征值：")
    
    input_data = {}
    for feature in feature_names:
        feature_lower = feature.lower()
        if 'sex' in feature_lower or 'gender' in feature_lower or 'code' in feature_lower:
            input_data[feature] = st.sidebar.selectbox(f"{feature} (分类变量)", options=[0, 1], index=0)
        else:
            input_data[feature] = st.sidebar.number_input(f"{feature} (数值)", value=0.0, format="%.2f")

    input_df = pd.DataFrame([input_data], columns=feature_names)

# ==========================================
# 5. 主界面：预测逻辑
# ==========================================
st.title("🏥 基于心磁成像装置的肺动脉高压检测计算器")
st.markdown("基于机器学习随机森林算法构建")
st.markdown("---")

if st.sidebar.button("🔍 开始预测风险"):
    if model and feature_names:
        with st.spinner('正在计算模型预测概率与 SHAP 解释值，请稍候...'):
            
            # A. 计算预测概率
            try:
                probability = model.predict_proba(input_df)[0, 1]
            except:
                prediction = model.predict(input_df)[0]
                probability = 1.0 if prediction == 1 else 0.0

            # B. 计算 SHAP 值 (含修复)
            shap_values_to_plot = None
            try:
                # 1. 处理 Pipeline
                if hasattr(model, 'steps') or hasattr(model, 'named_steps'):
                    final_estimator = model._final_estimator
                    preprocessor = model[:-1]
                    processed_data = preprocessor.transform(input_df)
                    if hasattr(processed_data, "toarray"):
                        processed_data = processed_data.toarray()
                    try:
                        processed_feature_names = preprocessor.get_feature_names_out()
                        data_for_shap = pd.DataFrame(processed_data, columns=processed_feature_names)
                    except:
                        data_for_shap = pd.DataFrame(processed_data)
                else:
                    final_estimator = model
                    data_for_shap = input_df

                # ==================================================
                # 【关键修复】强制数据类型转换，解决 ufunc 'add' 报错
                # ==================================================
                data_for_shap = data_for_shap.apply(pd.to_numeric, errors='coerce')
                data_for_shap = data_for_shap.fillna(0)
                data_for_shap = data_for_shap.astype('float64')
                # ==================================================

                # 2. 使用 TreeExplainer
                try:
                    explainer = shap.TreeExplainer(final_estimator)
                    shap_values = explainer(data_for_shap)
                except Exception:
                    explainer = shap.TreeExplainer(final_estimator, data=data_for_shap, model_output="probability")
                    shap_values = explainer(data_for_shap)

                # 3. 提取绘图数据
                if len(shap_values.values.shape) == 3:
                    shap_values_to_plot = shap_values[0, :, 1]
                else:
                    shap_values_to_plot = shap_values[0]

            except Exception as e:
                st.error(f"SHAP 计算模块出错: {str(e)}")
                shap_values_to_plot = None

            # C. 结果展示
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 📊 预测风险评分")
                risk_percent = probability * 100
                
                optimal_threshold = 35.703 
                youden_index = 0.771

                if risk_percent > optimal_threshold:
                    color = "#dc3545"
                    risk_label = "高风险 (High Risk)"
                    icon = "⚠️"
                    advice_box = "warning"
                    advice_text = f"模型预测概率 ({risk_percent:.1f}%) 已超过最佳截断值 ({optimal_threshold:.1f}%)。\n\n**建议：** 考虑进行超声心动图或右心导管检查以进一步确诊。"
                else:
                    color = "#28a745"
                    risk_label = "低风险 (Low Risk)"
                    icon = "✅"
                    advice_box = "success"
                    advice_text = f"模型预测概率 ({risk_percent:.1f}%) 低于最佳截断值 ({optimal_threshold:.1f}%)。\n\n**建议：** 目前风险较低，建议按常规流程进行随访。"
                
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
                        # =================================================
                        # 🛠️ 终极修复：手动拆解数据，绕过对象类型检查
                        # =================================================
                        
                        # 1. 提取基准值 (Base Value)
                        # 确保它是纯粹的浮点数 (float)，而不是数组
                        base_value = shap_values_to_plot.base_values
                        if isinstance(base_value, np.ndarray):
                            base_value = base_value.item()
                            
                        # 2. 提取 SHAP 贡献值 (SHAP values)
                        # 确保它是 numpy 数组
                        shap_vals = shap_values_to_plot.values
                        
                        # 3. 提取特征原始值 (Feature values)
                        # 强制转换为 numpy 纯数组，丢掉 Pandas 索引和表头
                        features_display = shap_values_to_plot.data
                        if hasattr(features_display, 'values'):
                            features_display = features_display.values
                        # 如果是多维的，展平成一维
                        if isinstance(features_display, np.ndarray):
                            features_display = features_display.flatten()
                            
                        # 4. 提取特征名称
                        feature_names_display = shap_values_to_plot.feature_names
                        
                        # 5. 绘制力图 (使用 force_plot 而不是 plots.force)
                        # 这种方式对数据类型最宽容
                        fig = shap.force_plot(
                            base_value, 
                            shap_vals, 
                            features_display, 
                            feature_names=feature_names_display, 
                            matplotlib=True, # 必须为 True 才能在 Streamlit 显示
                            show=False
                        )
                        
                        # 调整布局并展示
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception as plot_err:
                         st.error(f"绘图失败。这通常是由于数据类型不兼容(非数值)或字体配置引起的。\n\n错误详情: {plot_err}")
                else:
                    st.warning("无法生成 SHAP 图，请检查输入数据或模型结构。")
            
            st.markdown("---")
            st.caption(f"说明： 本工具采用约登指数 (Youden Index = {youden_index}) 确定的最佳截断值 {optimal_threshold/100:.5f} 进行风险分层。结果仅供科研参考。")
    else:
        st.error("系统错误：模型未加载。")
else:
    st.info("👈 请在左侧侧边栏输入患者的临床参数，然后点击“开始预测风险”按钮。")

