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
# 1. 核心配置
# ==========================================
st.set_page_config(
    page_title="肺动脉高压风险预测系统",
    page_icon="🏥",
    layout="wide"
)

def prepare_font_resource():
    """
    仅下载 SimHei 字体文件备用，绝不进行全局注册。
    """
    font_filename = "SimHei.ttf"
    font_url = "https://cdn.jsdelivr.net/gh/StellarCN/scp_zh@master/fonts/SimHei.ttf"

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
    return os.path.exists(font_filename)

is_font_ready = prepare_font_resource()

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
# 2. 定义类与加载模型
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
        st.sidebar.caption("✅ 字体资源：SimHei (准备就绪)")

# ==========================================
# 4. 主逻辑：预测与 SHAP
# ==========================================
st.title("🏥 基于心磁成像装置的肺动脉高压检测计算器")
st.markdown("基于机器学习随机森林算法构建 | 仅供科研参考")
st.markdown("---")

if st.sidebar.button("🔍 开始预测风险"):
    if model and feature_names:
        with st.spinner('正在计算模型预测概率与 SHAP 解释值...'):
            
            # A. 概率计算
            try:
                probability = model.predict_proba(input_df)[0, 1]
            except:
                prediction = model.predict(input_df)[0]
                probability = 1.0 if prediction == 1 else 0.0

            # B. SHAP 值计算
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

            # C. 结果显示
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 📊 预测风险评分")
                risk_percent = probability * 100
                optimal_threshold = 35.703 
                
                if risk_percent > optimal_threshold:
                    color, risk_label, icon = "#dc3545", "高风险 (High Risk)", "⚠️"
                    advice = f"风险值 ({risk_percent:.1f}%) > 阈值 ({optimal_threshold:.1f}%)。建议进一步检查。"
                else:
                    color, risk_label, icon = "#28a745", "低风险 (Low Risk)", "✅"
                    advice = f"风险值 ({risk_percent:.1f}%) < 阈值 ({optimal_threshold:.1f}%)。建议常规随访。"
                
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
                        # =====================================================
                        # 🔴 终极修复方案：逆向策略 🔴
                        # 1. 先重置所有 Matplotlib 设置为默认（英文环境）
                        #    这样保证了负号、数字绝对正常。
                        # =====================================================
                        plt.clf()
                        plt.rcParams.update(plt.rcParamsDefault)
                        
                        # 2. 强制把减号设为 ASCII 连字符 (双重保险)
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        # 3. 创建画布并绘图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.plots.waterfall(final_explanation, show=False, max_display=14)
                        
                        # =====================================================
                        # 4. 后处理：只对“中文特征名”进行字体修补
                        #    此时图上的负号已经是完美的了。
                        # =====================================================
                        ax = plt.gca()
                        
                        # 准备中文字体对象
                        if os.path.exists("SimHei.ttf"):
                            chinese_font = fm.FontProperties(fname="SimHei.ttf")
                            chinese_font.set_size(12)
                            
                            # 遍历 Y 轴标签（特征名），手动赋予中文字体
                            for label in ax.get_yticklabels():
                                label.set_fontproperties(chinese_font)
                        else:
                            st.warning("SimHei 字体未加载，中文可能显示为方框。")

                        # 5. 再次暴力检查所有的文本对象
                        #    如果有漏网之鱼使用了 Unicode 减号，强制替换为连字符
                        for txt in ax.texts + ax.get_xticklabels():
                            original = txt.get_text()
                            if '−' in original: # 这里的 '−' 是 Unicode 减号
                                txt.set_text(original.replace('−', '-'))
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"绘图错误: {e}")
                else:
                    st.warning("无 SHAP 数据。")

else:
    st.info("👈 请输入参数并点击开始预测。")
