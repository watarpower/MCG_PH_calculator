import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import os
import requests
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. 核心配置与字体准备
# ==========================================
st.set_page_config(
    page_title="肺动脉高压风险预测系统",
    page_icon="🏥",
    layout="wide"
)

def configure_font_environment():
    """
    下载 SimHei 字体并注册。
    """
    font_filename = "SimHei.ttf"
    font_url = "https://cdn.jsdelivr.net/gh/StellarCN/scp_zh@master/fonts/SimHei.ttf"

    if not os.path.exists(font_filename):
        with st.spinner("正在初始化中文字体环境 (SimHei)..."):
            try:
                response = requests.get(font_url, timeout=10)
                if response.status_code == 200:
                    with open(font_filename, "wb") as f:
                        f.write(response.content)
            except Exception as e:
                st.warning(f"字体下载异常: {e}")

    if os.path.exists(font_filename):
        fm.fontManager.addfont(font_filename)
        return True
    return False

is_font_ready = configure_font_environment()

# --- 自定义 CSS (保持原样) ---
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1 { color: #2c3e50; font-weight: bold; font-family: sans-serif; }
    h3 { color: #34495e; font-family: sans-serif; }
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
# 2. 类定义与加载
# ==========================================
class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None): return self
    def transform(self, X): return pd.DataFrame(X)

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
            features = [x.strip() for x in (content.split(',') if ',' in content else content.split('\n'))]
    except FileNotFoundError:
        st.error("❌ 错误：未找到特征文件 'selected_features_1SE_建模数据.txt'。")
        return None, None
        
    return model, features

model, feature_names = load_model_and_features()

# ==========================================
# 3. 侧边栏
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
    
    st.sidebar.markdown("---")
    if is_font_ready:
        st.sidebar.caption("✅ 字体状态：SimHei (已加载)")

# ==========================================
# 4. 主逻辑
# ==========================================
st.title("🏥 基于心磁成像装置的肺动脉高压检测计算器")
st.markdown("基于机器学习随机森林算法构建 | 仅供科研参考")
st.markdown("---")

if st.sidebar.button("🔍 开始预测风险"):
    if model and feature_names:
        with st.spinner('正在计算模型预测概率与 SHAP 解释值，请稍候...'):
            
            # A. 概率计算
            try:
                probability = model.predict_proba(input_df)[0, 1]
            except:
                prediction = model.predict(input_df)[0]
                probability = 1.0 if prediction == 1 else 0.0

            # B. SHAP 计算
            final_explanation = None
            try:
                if hasattr(model, 'steps') or hasattr(model, 'named_steps'):
                    final_estimator = model._final_estimator
                    preprocessor = model[:-1]
                    processed_data = preprocessor.transform(input_df)
                    if hasattr(processed_data, "toarray"):
                        processed_data = processed_data.toarray()
                    processed_data_df = pd.DataFrame(processed_data)
                else:
                    final_estimator = model
                    processed_data_df = input_df

                try:
                    explainer = shap.TreeExplainer(final_estimator)
                    shap_values_obj = explainer(processed_data_df)
                except:
                    explainer = shap.TreeExplainer(final_estimator, data=processed_data_df, model_output="probability")
                    shap_values_obj = explainer(processed_data_df)

                if shap_values_obj is not None:
                    if len(shap_values_obj.values.shape) == 3:
                        shap_contribution = shap_values_obj.values[0, :, 1]
                        base_val = shap_values_obj.base_values[0, 1]
                    else:
                        shap_contribution = shap_values_obj.values[0]
                        base_val = shap_values_obj.base_values[0]

                    final_explanation = shap.Explanation(
                        values=shap_contribution,
                        base_values=base_val,
                        data=input_df.iloc[0].values,
                        feature_names=feature_names
                    )
            except Exception as e:
                st.error(f"SHAP 计算出错: {str(e)}")

            # C. 结果展示 (UI 保持原始设计)
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
                st.markdown("### 🔍 SHAP 可解释性分析 (瀑布图)")
                st.markdown("下图展示了各特征对预测结果的贡献：**红色**条表示增加风险，**蓝色**条表示降低风险。")
                
                if final_explanation is not None:
                    try:
                        # =================================================
                        # 🛠️ 策略：优先保证中文，然后暴力修复数字
                        # =================================================
                        
                        # 1. 全局设置 SimHei，确保【中文一定显示】
                        plt.rcParams['font.sans-serif'] = ['SimHei']
                        plt.rcParams['axes.unicode_minus'] = False 
                        
                        # 2. 创建画布
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # 3. 绘图 (此时中文正常，负号可能是方框)
                        shap.plots.waterfall(final_explanation, show=False, max_display=14)
                        
                        # 4. 获取当前轴
                        ax = plt.gca()
                        
                        # 5. 准备【英文字体】对象，用于修复数字
                        #    DejaVu Sans 是 Matplotlib 自带的，对符号支持最好
                        en_font = fm.FontProperties(family='DejaVu Sans', size=12)
                        
                        # --- 修复 A: 拦截 X 轴数值，强制转为英文格式 ---
                        # 定义一个格式化器，它不管系统怎么想，直接返回带 ASCII 短横线的字符串
                        def force_ascii_minus(x, pos):
                            return '{:.1f}'.format(x).replace('−', '-') # 替换 Unicode 减号
                        
                        ax.xaxis.set_major_formatter(ticker.FuncFormatter(force_ascii_minus))
                        
                        # 强制 X 轴刻度使用英文字体
                        for label in ax.get_xticklabels():
                            label.set_fontproperties(en_font)

                        # --- 修复 B: 图内数值 (柱子上的数字) ---
                        for txt in ax.texts:
                            # 1. 设置为英文字体
                            txt.set_fontproperties(en_font)
                            # 2. 获取现有文本
                            original_text = txt.get_text()
                            # 3. 暴力替换：把所有可能的“方框”或“Unicode减号”替换为“-”
                            #    SimHei 的方框在内部可能就是无法渲染的 Unicode 字符
                            new_text = original_text.replace('−', '-')
                            txt.set_text(new_text)

                        # --- 修复 C: X 轴标题 (如果有) ---
                        ax.set_xlabel(ax.get_xlabel(), fontproperties=en_font)

                        # 注意：Y 轴标签我们完全不动，因为第一步全局设置了 SimHei，所以它们是好的。
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception as plot_err:
                         st.error(f"绘图失败: {plot_err}")
                else:
                    st.warning("无法生成 SHAP 图，请检查输入数据或模型结构。")
            
            st.markdown("---")
            st.caption(f"**说明：** 本工具采用约登指数 (Youden Index = {youden_index}) 确定的最佳截断值 {optimal_threshold/100:.5f} 进行风险分层。结果仅供科研参考。")
    else:
        st.error("系统错误：模型未加载。")
else:
    st.info("👈 请在左侧侧边栏输入患者的临床参数，然后点击“开始预测风险”按钮。")
