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
# 1. 核心配置与字体加载 (SimHei)
# ==========================================
st.set_page_config(
    page_title="肺动脉高压风险预测系统",
    page_icon="🏥",
    layout="wide"
)

def configure_font():
    """
    下载 SimHei 字体并注册到 Matplotlib。
    """
    font_filename = "SimHei.ttf"
    font_url = "https://cdn.jsdelivr.net/gh/StellarCN/scp_zh@master/fonts/SimHei.ttf"

    # 1. 下载
    if not os.path.exists(font_filename):
        with st.spinner("正在初始化字体资源..."):
            try:
                response = requests.get(font_url, timeout=10)
                if response.status_code == 200:
                    with open(font_filename, "wb") as f:
                        f.write(response.content)
            except Exception as e:
                st.warning(f"字体下载失败: {e}")

    # 2. 注册
    if os.path.exists(font_filename):
        try:
            fm.fontManager.addfont(font_filename)
            # 全局设置为 SimHei，确保中文一定能显示
            plt.rcParams['font.sans-serif'] = ['SimHei']
            # 尝试软性关闭 Unicode 减号 (如果失效，下面有暴力修复兜底)
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except Exception:
            return False
    return False

is_font_ready = configure_font()

# --- 自定义 CSS ---
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .report-box { border: 1px solid #e6e6e6; padding: 20px; background-color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 模型加载
# ==========================================
class DataFrameConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return pd.DataFrame(X)

@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load('final_model_RF.pkl')
        with open('selected_features_1SE_建模数据.txt', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            features = [x.strip() for x in (content.split(',') if ',' in content else content.split('\n'))]
        return model, features
    except:
        return None, None

model, feature_names = load_model_and_features()

# ==========================================
# 3. 侧边栏输入
# ==========================================
if model and feature_names:
    st.sidebar.header("📋 患者参数录入")
    input_data = {}
    for feature in feature_names:
        if any(x in feature.lower() for x in ['sex', 'gender', 'code']):
            input_data[feature] = st.sidebar.selectbox(f"{feature}", [0, 1])
        else:
            input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0, format="%.2f")
    input_df = pd.DataFrame([input_data], columns=feature_names)
    st.sidebar.markdown("---")
    st.sidebar.caption("✅ 字体状态：SimHei" if is_font_ready else "⚠️ 字体未加载")

# ==========================================
# 4. 核心逻辑
# ==========================================
st.title("🏥 肺动脉高压风险预测系统")
st.markdown("---")

if st.sidebar.button("🔍 开始预测"):
    if model:
        with st.spinner('正在分析...'):
            # --- 预测 ---
            try:
                prob = model.predict_proba(input_df)[0, 1]
            except:
                prob = 1.0 if model.predict(input_df)[0] == 1 else 0.0
            
            # --- SHAP ---
            explanation = None
            try:
                # 兼容 Pipeline 和 Model
                estimator = model._final_estimator if hasattr(model, '_final_estimator') else model
                data = model[:-1].transform(input_df) if hasattr(model, '_final_estimator') else input_df
                if hasattr(data, "toarray"): data = data.toarray()
                data_df = pd.DataFrame(data, columns=feature_names) # 确保列名对齐

                try:
                    shap_values = shap.TreeExplainer(estimator)(data_df)
                except:
                    shap_values = shap.TreeExplainer(estimator, data=data_df, model_output="probability")(data_df)

                # 提取数据
                vals = shap_values.values[0, :, 1] if len(shap_values.values.shape) == 3 else shap_values.values[0]
                base = shap_values.base_values[0, 1] if len(shap_values.values.shape) == 3 else shap_values.base_values[0]
                
                explanation = shap.Explanation(values=vals, base_values=base, data=input_df.iloc[0].values, feature_names=feature_names)
            except Exception as e:
                st.error(f"SHAP error: {e}")

            # --- 展示 ---
            c1, c2 = st.columns([1, 2])
            with c1:
                risk = prob * 100
                color = "#dc3545" if risk > 35.703 else "#28a745"
                st.markdown(f"""
                    <div class="report-box" style="border-left: 5px solid {color}; text-align: center;">
                        <h2 style="color: {color}; font-size: 40px;">{risk:.1f}%</h2>
                        <p>患病风险</p>
                    </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown("### SHAP 因子分析")
                if explanation is not None:
                    try:
                        # 1. 基础绘图 (SimHei 环境)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(explanation, show=False, max_display=14)
                        
                        # ========================================================
                        # 🛡️ 终极修复：暴力遍历并替换字符 🛡️
                        # 这段代码会检查图上的每一个字，把那个不显示的减号替换掉
                        # ========================================================
                        
                        # 准备英文字体 (用于数字)
                        english_font = fm.FontProperties(family='DejaVu Sans')
                        
                        ax = plt.gca()
                        
                        # 1. 修复 X 轴刻度 (底部数字)
                        for label in ax.get_xticklabels():
                            text = label.get_text()
                            # 只要包含 Unicode 减号，就替换为 ASCII 连字符
                            if '−' in text or '-' in text:
                                # 强制设为英文字体
                                label.set_fontproperties(english_font) 
                                # 替换字符
                                new_text = text.replace('−', '-') 
                                label.set_text(new_text)

                        # 2. 修复图内的数值标注 (柱子旁边的数字)
                        for txt in ax.texts:
                            text = txt.get_text()
                            if '−' in text or '-' in text:
                                txt.set_fontproperties(english_font)
                                new_text = text.replace('−', '-')
                                txt.set_text(new_text)
                                
                        # 3. (可选) 修复 X 轴标签
                        xlabel = ax.xaxis.get_label()
                        if '−' in xlabel.get_text():
                             xlabel.set_fontproperties(english_font)
                             xlabel.set_text(xlabel.get_text().replace('−', '-'))

                        # 注意：Y 轴标签 (特征名) 我们不动它，让它保持 SimHei 显示中文
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"绘图失败: {e}")
else:
    st.info("等待输入...")
