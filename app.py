import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import os
import requests
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. 核心配置与字体加载
# ==========================================
st.set_page_config(
    page_title="肺动脉高压风险预测系统",
    page_icon="🏥",
    layout="wide"
)

def configure_font():
    """
    下载并注册 SimHei 字体，确保全局可以使用中文。
    """
    font_filename = "SimHei.ttf"
    font_url = "https://cdn.jsdelivr.net/gh/StellarCN/scp_zh@master/fonts/SimHei.ttf"

    # 1. 下载字体
    if not os.path.exists(font_filename):
        with st.spinner("正在初始化中文字体资源 (SimHei)..."):
            try:
                response = requests.get(font_url, timeout=10)
                if response.status_code == 200:
                    with open(font_filename, "wb") as f:
                        f.write(response.content)
                else:
                    st.warning(f"字体下载失败 (Code {response.status_code})")
            except Exception as e:
                st.warning(f"网络异常，字体下载失败: {e}")

    # 2. 注册字体 (关键步骤)
    if os.path.exists(font_filename):
        try:
            # 这一步让 Matplotlib 知道 SimHei 的存在
            fm.fontManager.addfont(font_filename)
            return True
        except Exception as e:
            st.error(f"字体注册失败: {e}")
            return False
    return False

is_font_ready = configure_font()

# --- 自定义 CSS ---
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
# 2. 类与模型加载
# ==========================================
class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X)

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
# 3. 输入界面
# ==========================================
if model and feature_names:
    st.sidebar.header("📋 患者参数录入")
    
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
# 4. 核心逻辑：预测与 SHAP
# ==========================================
st.title("🏥 基于心磁成像装置的肺动脉高压检测计算器")
st.markdown("基于机器学习随机森林算法构建 | 仅供科研参考")
st.markdown("---")

if st.sidebar.button("🔍 开始预测风险"):
    if model and feature_names:
        with st.spinner('正在计算...'):
            # A. 概率
            try:
                probability = model.predict_proba(input_df)[0, 1]
            except:
                prediction = model.predict(input_df)[0]
                probability = 1.0 if prediction == 1 else 0.0

            # B. SHAP
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
                st.error(f"SHAP 计算出错: {e}")

            # C. 显示
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 📊 预测风险评分")
                risk_percent = probability * 100
                optimal_threshold = 35.703 
                
                if risk_percent > optimal_threshold:
                    color, risk_label, icon = "#dc3545", "高风险 (High Risk)", "⚠️"
                    advice = f"风险值 ({risk_percent:.1f}%) > 阈值 ({optimal_threshold:.1f}%)。"
                else:
                    color, risk_label, icon = "#28a745", "低风险 (Low Risk)", "✅"
                    advice = f"风险值 ({risk_percent:.1f}%) < 阈值 ({optimal_threshold:.1f}%)。"
                
                st.markdown(f"""
                    <div class="report-box" style="text-align: center; border-left: 5px solid {color};">
                        <h2 style="color: {color}; font-size: 50px; margin: 0;">{risk_percent:.1f}%</h2>
                        <p style="color: gray;">患病概率</p>
                        <hr>
                        <h3 style="color: {color};">{icon} {risk_label}</h3>
                    </div>""", unsafe_allow_html=True)
                if risk_percent > optimal_threshold: st.warning(advice)
                else: st.success(advice)

            with col2:
                st.markdown("### 🔍 SHAP 可解释性分析")
                st.markdown("下图展示了各特征对预测结果的贡献。")
                
                if final_explanation is not None:
                    try:
                        # ==================================================
                        # 🔴 修正方案：全局中文 + 局部强制英文数字 🔴
                        # ==================================================
                        
                        # 1. 设置全局字体为 SimHei (解决中文不显示问题)
                        plt.rcParams['font.sans-serif'] = ['SimHei'] 
                        plt.rcParams['axes.unicode_minus'] = False # 尝试软性修复负号
                        
                        # 2. 创建画布
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # 3. 绘图
                        shap.plots.waterfall(final_explanation, show=False, max_display=14)
                        
                        # 4. 【核心后处理】
                        # 获取当前坐标轴
                        ax = plt.gca()
                        
                        # 定义一个绝对安全的英文字体 (用于替换数字)
                        english_font = fm.FontProperties(family='DejaVu Sans')
                        
                        # A. 强制 X 轴刻度 (数字) 使用英文字体
                        for label in ax.get_xticklabels():
                            label.set_fontproperties(english_font)
                            # 双重保险：手动替换减号
                            text = label.get_text()
                            if '−' in text: 
                                label.set_text(text.replace('−', '-'))
                                
                        # B. 强制图内数字 (柱子旁边的标注) 使用英文字体
                        for txt in ax.texts:
                            txt.set_fontproperties(english_font)
                            text = txt.get_text()
                            if '−' in text:
                                txt.set_text(text.replace('−', '-'))
                                
                        # 注意：Y 轴标签 (ax.get_yticklabels) 我们不动它
                        # 因为全局已经设为 SimHei，所以中文特征名会自动正确显示
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"绘图错误: {e}")
                        import traceback
                        st.text(traceback.format_exc())
                else:
                    st.warning("无 SHAP 数据。")
else:
    st.info("👈 请输入参数并点击开始预测。")
