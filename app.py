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
# 1. 核心配置与字体修复
# ==========================================
st.set_page_config(
    page_title="基于心磁成像装置的肺动脉高压检测计算器",
    page_icon="🏥",
    layout="wide"
)

def configure_font_environment():
    """
    下载 SimHei 字体并强制 Matplotlib 使用它。
    目标：同时正确显示中文和负号。
    """
    font_filename = "SimHei.ttf"
    # 纯 URL 字符串，避免 Markdown 格式导致下载失败
    font_url = "https://cdn.jsdelivr.net/gh/StellarCN/scp_zh@master/fonts/SimHei.ttf"

    # 如果本地没有字体，尝试下载
    if not os.path.exists(font_filename):
        with st.spinner("正在初始化中文字体环境 (SimHei)..."):
            try:
                response = requests.get(font_url, timeout=10)
                if response.status_code == 200:
                    with open(font_filename, "wb") as f:
                        f.write(response.content)
                else:
                    st.warning(f"字体下载失败 (Code {response.status_code})，将尝试使用系统默认字体。")
            except Exception as e:
                st.warning(f"网络异常，字体下载失败: {e}")

    # 注册字体并配置 Matplotlib
    if os.path.exists(font_filename):
        try:
            fm.fontManager.addfont(font_filename)

            # 全局字体设置：优先使用 SimHei
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]

            # 禁用 Unicode minus，让坐标轴等地方使用普通 '-'
            plt.rcParams["axes.unicode_minus"] = False

            # 再同步一份到 matplotlib.rcParams，避免个别库内部单独读取
            matplotlib.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["SimHei", "DejaVu Sans"],
                "axes.unicode_minus": False,
            })

            return True
        except Exception as e:
            # 这里用 warning 而不是 error，避免出现红色大错误提示
            st.warning(f"字体配置出错，已退回系统默认字体: {e}")
            return False

    return False

def fix_shap_minus_signs(ax=None):
    """
    将图中所有文本里的 Unicode 减号 U+2212 替换为普通 ASCII '-'，
    解决部分中文字体（如 SimHei）不包含 U+2212 导致负号显示为方框的问题。
    """
    if ax is None:
        ax = plt.gca()

    def _replace_minus(text: str) -> str:
        return text.replace("\u2212", "-") if text else text

    # 坐标轴刻度标签
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        s = label.get_text()
        new_s = _replace_minus(s)
        if new_s != s:
            label.set_text(new_s)

    # 图中的所有文本对象（包括 SHAP 的数值标签）
    for text_obj in ax.texts:
        s = text_obj.get_text()
        new_s = _replace_minus(s)
        if new_s != s:
            text_obj.set_text(new_s)

# 执行字体配置
is_font_ready = configure_font_environment()

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
# 2. 定义必要的类 (防止模型加载报错)
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
if model is not None and feature_names is not None:
    st.sidebar.header("📋 受试者参数录入")
    st.sidebar.markdown("请在下方输入心磁和临床特征参数值：")
    
    input_data = {}
    
    for feature in feature_names:
        feature_lower = feature.lower()
        if 'sex' in feature_lower or 'gender' in feature_lower or 'code' in feature_lower:
            input_data[feature] = st.sidebar.selectbox(f"{feature} (分类变量)", options=[0, 1], index=0)
        else:
            input_data[feature] = st.sidebar.number_input(f"{feature} (数值)", value=0.0, format="%.2f")

    input_df = pd.DataFrame([input_data], columns=feature_names)
    


# ==========================================
# 5. 主界面：预测与解释逻辑
# ==========================================
st.title("🏥 基于心磁成像装置的肺动脉高压检测计算器")
st.markdown("基于随机森林算法构建")
st.markdown("---")

if st.sidebar.button("🔍 预测"):
    if model is not None and feature_names is not None:
        with st.spinner('正在计算模型预测风险与 SHAP 解释值，请稍候...'):
            
            # A. 计算概率（只用于内部风险判断，不展示给用户）
            try:
                probability = model.predict_proba(input_df)[0, 1]
            except Exception:
                prediction = model.predict(input_df)[0]
                probability = 1.0 if prediction == 1 else 0.0

            # B. 计算 SHAP
            final_explanation = None
            try:
                # 1. 准备模型输入
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

                # 2. 计算 SHAP 值
                shap_values_obj = None 
                try:
                    explainer = shap.TreeExplainer(final_estimator)
                    shap_values_obj = explainer(processed_data_df)
                except Exception:
                    explainer = shap.TreeExplainer(
                        final_estimator, 
                        data=processed_data_df, 
                        model_output="probability"
                    )
                    shap_values_obj = explainer(processed_data_df)

                # 3. 提取数据
                if shap_values_obj is not None:
                    if len(shap_values_obj.values.shape) == 3:
                        shap_contribution = shap_values_obj.values[0, :, 1]
                        base_val = shap_values_obj.base_values[0, 1]
                    else:
                        shap_contribution = shap_values_obj.values[0]
                        base_val = shap_values_obj.base_values[0]

                    original_input_values = input_df.iloc[0].values

                    # 4. 构建解释对象
                    final_explanation = shap.Explanation(
                        values=shap_contribution,
                        base_values=base_val,
                        data=original_input_values,
                        feature_names=feature_names
                    )
                else:
                    st.error("SHAP 计算未返回有效结果")

            except Exception as e:
                st.error(f"SHAP 计算模块出错: {str(e)}")
                final_explanation = None

            # C. 结果展示
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### 📊 检测结果")

                risk_percent = probability * 100
                optimal_threshold = 35.703   # 仍用于内部划分，不展示数值
                youden_index = 0.771

                if risk_percent > optimal_threshold:
                    color = "#dc3545"
                    risk_label = "高风险 (High Risk)"
                    icon = "⚠️"
                    advice_box = "warning"
                    advice_text = (
                        "模型评估结果为 **高风险**，提示患者当前患肺动脉高压的可能性较高。\n\n"
                        "**建议：** 建议进一步完善右心导管检查，"
                        "并结合临床情况进行综合评估。"
                    )
                else:
                    color = "#28a745"
                    risk_label = "低风险 (Low Risk)"
                    icon = "✅"
                    advice_box = "success"
                    advice_text = (
                        "模型评估结果为 **低风险**，提示患者当前患肺动脉高压的可能性较低。\n\n"
                        "**建议：** 可继续观察，根据临床症状和体征决定是否进一步检查。"
                    )
                
                # 卡片中只展示“高/低风险”，不展示具体概率
                st.markdown(
                    f"""
                    <div class="report-box" style="text-align: center; border-left: 5px solid {color};">
                        <h2 style="color: {color}; font-size: 40px; margin: 0;">{icon} {risk_label}</h2>
                        <p style="color: gray; font-size: 14px; margin-top: 10px;">
                            本结果仅供科研与辅助决策参考。
                        </p>
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
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # 再次确认当前绘图环境的字体配置
                        plt.rcParams["font.family"] = "sans-serif"
                        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
                        plt.rcParams["axes.unicode_minus"] = False

                        # 绘制 SHAP 瀑布图（不 show，让我们有机会修改文本）
                        shap.plots.waterfall(final_explanation, show=False, max_display=14)

                        # 把 SHAP 文本中的 Unicode 减号替换成普通 '-'
                        fix_shap_minus_signs(ax)

                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as plot_err:
                        st.error(f"绘图失败。调试信息: {plot_err}")
                else:
                    st.warning("无法生成 SHAP 图，请检查输入数据或模型结构。")
            
            st.markdown("---")
            # 这里保留方法学说明，可以视需要保留或删除
            st.caption(
                f"**说明：** 本工具采用约登指数 (Youden Index = {youden_index}) "
                f"确定的最佳截断值进行风险分层，结果仅供科研参考。"
            )
    else:
        st.error("系统错误：模型未加载。")
else:
    st.info("👈 请在左侧侧边栏输入患者的临床参数，然后点击“预测”按钮。")


