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
    font_url = "https://cdn.jsdelivr.net/gh/StellarCN/scp_zh@master/fonts/SimHei.ttf"

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

    if os.path.exists(font_filename):
        try:
            fm.fontManager.addfont(font_filename)
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            matplotlib.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["SimHei", "DejaVu Sans"],
                "axes.unicode_minus": False,
            })
            return True
        except Exception as e:
            st.warning(f"字体配置出错，已退回系统默认字体: {e}")
            return False
    return False

def fix_shap_minus_signs(ax=None):
    """
    把 SHAP 图中的 U+2212 负号替换成普通 '-'。
    """
    if ax is None:
        ax = plt.gca()

    def _replace_minus(text: str) -> str:
        return text.replace("\u2212", "-") if text else text

    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        s = label.get_text()
        new_s = _replace_minus(s)
        if new_s != s:
            label.set_text(new_s)

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
# 3. 预后评估 Cox / 联合模型参数与函数
# ==========================================
# Step 1：图 1 中的 Cox 模型
# 6MWT + WHO 功能分级(1–4) + NT-proBNP -> xbeta
COEF_6MWT = -0.006
COEF_FC1 = -0.868
COEF_FC2 = -3.200
COEF_FC3 = -1.646
COEF_BNP = 0.000  # SPSS 表中为 .000，如有更精确系数可在此处替换

# Step 2：图 2 中的联合模型
# xbeta（COMPERA 原始）+ R/T 比值 -> 总线性预测值
COEF_XBETA = 0.764           # “COMPERA原始” 的 B
COEF_RT_RATIO = 0.189        # R 波和 T 波峰值时刻两极磁感应强度差值比值 的 B
COX_INTERCEPT = 0.0          # 这里用作把线性预测值映射到 0–1 概率的逻辑变换常数
PROGNOSIS_THRESHOLD = 0.50359  # 联合模型的截断值（概率尺度）

def compute_xbeta(six_mwt: float, who_fc: int, ntprobnp: float) -> float:
    """
    根据 6MWT、WHO 功能分级 (1-4)、NT-proBNP 计算 Cox 回归线性预测值 xbeta。
    FC=4 作为基线组，对应 FC1=FC2=FC3=0。
    """
    fc1 = 1 if who_fc == 1 else 0
    fc2 = 1 if who_fc == 2 else 0
    fc3 = 1 if who_fc == 3 else 0
    xbeta = (
        COEF_6MWT * six_mwt +
        COEF_FC1 * fc1 +
        COEF_FC2 * fc2 +
        COEF_FC3 * fc3 +
        COEF_BNP * ntprobnp
    )
    return xbeta

def compute_prognosis_probability(six_mwt: float, who_fc: int, ntprobnp: float, rt_ratio: float):
    """
    两步 Cox 联合模型：
      1）6MWT + WHO-FC + NT-proBNP -> xbeta
      2）xbeta 与 R/T 比值 -> 线性预测值，再用逻辑函数变为 0–1 概率

    说明：由于目前没有基线生存/常数项，这里用 logistic 函数把线性预测值转成“相对概率”，
          然后仍采用你给出的 0.50359 作为高危/低危的分界。
    """
    xbeta = compute_xbeta(six_mwt, who_fc, ntprobnp)
    linear_pred = COX_INTERCEPT + COEF_XBETA * xbeta + COEF_RT_RATIO * rt_ratio
    prob = 1.0 / (1.0 + np.exp(-linear_pred))
    return prob, xbeta, linear_pred

# ==========================================
# 4. 加载模型与特征（第一步：PH 是否高风险）
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
# 5. 侧边栏：输入界面
# ==========================================
six_mwt = None
who_fc = None
ntprobnp = None
rt_ratio = None

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

    # ---- 新增：预后评估所需 4 项参数 ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 预后评估参数（仅在高风险时使用）")
    six_mwt = st.sidebar.number_input("6分钟步行距离 6MWT (m)", min_value=0.0, value=0.0, step=1.0)
    who_fc = st.sidebar.selectbox("WHO 心功能分级 (1-4)", options=[1, 2, 3, 4], index=0)
    ntprobnp = st.sidebar.number_input("NT-proBNP", min_value=0.0, value=0.0, step=1.0)
    rt_ratio = st.sidebar.number_input("R波和T波峰值时刻两极磁感应强度差值比值", value=0.0, format="%.3f")

# ==========================================
# 6. 主界面：预测与解释逻辑
# ==========================================
st.title("🏥 基于心磁成像装置的肺动脉高压检测计算器")
st.markdown("基于随机森林算法构建")
st.markdown("---")

if st.sidebar.button("🔍 预测"):
    if model is not None and feature_names is not None:
        with st.spinner('正在计算模型预测风险与 SHAP 解释值，请稍候...'):
            # A. 随机森林概率（仅用于内部高/低风险判断）
            try:
                probability = model.predict_proba(input_df)[0, 1]
            except Exception:
                prediction = model.predict(input_df)[0]
                probability = 1.0 if prediction == 1 else 0.0

            # B. 计算 SHAP
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

                if shap_values_obj is not None:
                    if len(shap_values_obj.values.shape) == 3:
                        shap_contribution = shap_values_obj.values[0, :, 1]
                        base_val = shap_values_obj.base_values[0, 1]
                    else:
                        shap_contribution = shap_values_obj.values[0]
                        base_val = shap_values_obj.base_values[0]

                    original_input_values = input_df.iloc[0].values

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
                st.markdown("### 📊 肺动脉高压检测结果")

                risk_percent = probability * 100
                optimal_threshold = 35.703   # 用于内部高/低风险划分（%）
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
                
                # 只展示高/低风险，不展示具体概率
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

                # ---- 新增：第二步 —— 预后评估（仅在高风险时执行）----
                if risk_percent > optimal_threshold:
                    st.markdown("---")
                    st.markdown("### 📈 预后评估（临床恶化风险）")

                    try:
                        prog_prob, prog_xbeta, prog_lp = compute_prognosis_probability(
                            six_mwt or 0.0,
                            int(who_fc) if who_fc is not None else 4,
                            ntprobnp or 0.0,
                            rt_ratio or 0.0
                        )

                        if prog_prob >= PROGNOSIS_THRESHOLD:
                            prog_label = "高危"
                            prog_color = "#dc3545"
                            prog_icon = "⚠️"
                        else:
                            prog_label = "低危"
                            prog_color = "#28a745"
                            prog_icon = "✅"

                        # 按你的要求输出“临床恶化：高危/低危”
                        st.markdown(
                            f"""
                            <div class="report-box" style="border-left: 5px solid {prog_color};">
                                <h3 style="color:{prog_color}; margin:0;">{prog_icon} 临床恶化：{prog_label}</h3>
                                <p style="color: gray; font-size: 13px; margin-top:8px;">
                                    预后评估基于两步 Cox 联合模型（6MWT、WHO 心功能分级、NT-proBNP 及 R/T 比值），
                                    截断值为 {PROGNOSIS_THRESHOLD:.5f}。
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error(f"预后评估计算失败，请检查输入参数：{e}")
                else:
                    # 低风险患者不做预后评估
                    st.markdown("---")
                    st.info("当前为 **低风险**，暂不进行临床恶化预后评估。")

            with col2:
                st.markdown("### 🔍 SHAP 可解释性分析 (瀑布图)")
                st.markdown("下图展示了各特征对预测结果的贡献：**红色**条表示增加风险，**蓝色**条表示降低风险。")
                
                if final_explanation is not None:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))

                        plt.rcParams["font.family"] = "sans-serif"
                        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
                        plt.rcParams["axes.unicode_minus"] = False

                        shap.plots.waterfall(final_explanation, show=False, max_display=14)
                        fix_shap_minus_signs(ax)

                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as plot_err:
                        st.error(f"绘图失败。调试信息: {plot_err}")
                else:
                    st.warning("无法生成 SHAP 图，请检查输入数据或模型结构。")
            
            st.markdown("---")
            st.caption(
                f"**说明：** 本工具采用约登指数 (Youden Index = {youden_index}) "
                f"确定的最佳截断值进行风险分层，结果仅供科研参考。"
            )
    else:
        st.error("系统错误：模型未加载。")
else:
    st.info("👈 请在左侧侧边栏输入患者的临床参数，然后点击“预测”按钮。")
