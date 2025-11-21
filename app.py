import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker  # 引入 ticker 用于强制格式化
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
    下载 SimHei 字体文件备用。
    注意：我们不再全局设置 font.sans-serif = SimHei，
    而是只在绘图时，针对中文字段手动应用字体。
    这样可以防止 SimHei 污染数字显示的负号。
    """
    font_filename = "SimHei.ttf"
    font_url = "https://cdn.jsdelivr.net/gh/StellarCN/scp_zh@master/fonts/SimHei.ttf"

    if not os.path.exists(font_filename):
        with st.spinner("正在初始化中文字体环境..."):
            try:
                response = requests.get(font_url, timeout=10)
                if response.status_code == 200:
                    with open(font_filename, "wb") as f:
                        f.write(response.content)
            except Exception as e:
                st.warning(f"字体下载异常: {e}")

    return os.path.exists(font_filename)

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
        with st.spinner('正在计算...'):
            
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

            # C. 结果展示 (恢复原始代码)
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
                        # 🚀 100% 解决负号问题的重构方案
                        # =================================================
                        
                        # 1. 强制重置样式，清除任何全局字体设置，回归默认英文环境
                        plt.clf()
                        plt.style.use('default')
                        plt.rcParams['axes.unicode_minus'] = False # 全局关闭 Unicode 减号，使用 ASCII 连字符

                        # 2. 创建画布
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # 3. 绘制瀑布图
                        shap.plots.waterfall(final_explanation, show=False, max_display=14)
                        
                        # 4. 获取当前轴
                        ax = plt.gca()
                        
                        # 5. 准备字体 (精确控制)
                        # SimHei 用于中文 (Y轴)
                        zh_font = fm.FontProperties(fname="SimHei.ttf", size=12) if os.path.exists("SimHei.ttf") else fm.FontProperties(family='sans-serif', size=12)
                        # Arial/DejaVu 用于数字 (X轴和Bar上的文字) -> 确保负号显示
                        en_font = fm.FontProperties(family=['Arial', 'DejaVu Sans', 'sans-serif'], size=12)
                        
                        # --- 关键修复 A: X 轴刻度 ---
                        # 强制使用 FuncFormatter 重新格式化数字，确保输出的是普通 ASCII 字符串
                        def ascii_format(x, pos):
                            return '{:.1f}'.format(x) # Python 默认格式化就是用短横线，不是 Unicode 减号
                        
                        ax.xaxis.set_major_formatter(ticker.FuncFormatter(ascii_format))
                        
                        for label in ax.get_xticklabels():
                            label.set_fontproperties(en_font)

                        # --- 关键修复 B: 图内文字 (柱子旁的数值) ---
                        for txt in ax.texts:
                            txt.set_fontproperties(en_font)
                            # 双重保险：手动替换文本中的减号
                            original = txt.get_text()
                            # 替换 \u2212 (Unicode减号) 为 - (ASCII连字符)
                            fixed = original.replace('−', '-').replace('\u2212', '-')
                            txt.set_text(fixed)

                        # --- 关键修复 C: Y 轴特征名 (仅此处使用中文) ---
                        # 只有这里我们需要 SimHei
                        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font)
                        
                        # --- 关键修复 D: X 轴标签 ---
                        ax.set_xlabel(ax.get_xlabel(), fontproperties=en_font)

                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception as plot_err:
                         st.error(f"绘图失败: {plot_err}")
                         import traceback
                         st.text(traceback.format_exc())
                else:
                    st.warning("无法生成 SHAP 图，请检查输入数据或模型结构。")
            
            st.markdown("---")
            st.caption(f"**说明：** 本工具采用约登指数 (Youden Index = {youden_index}) 确定的最佳截断值 {optimal_threshold/100:.5f} 进行风险分层。结果仅供科研参考。")
    else:
        st.error("系统错误：模型未加载。")
else:
    st.info("👈 请在左侧侧边栏输入患者的临床参数，然后点击“开始预测风险”按钮。")
