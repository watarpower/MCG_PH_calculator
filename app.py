import streamlit as st
import pandas as pd
import numpy as np
import math
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import os
import requests
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. 页面配置 & 字体修复
# ==========================================
st.set_page_config(
    page_title="基于心磁成像装置的肺动脉高压风险计算器",
    page_icon="🏥",
    layout="wide"
)

def configure_font_environment():
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

is_font_ready = configure_font_environment()

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
# 2. 防止 joblib 加载报错的类
# ==========================================
class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X)

# ==========================================
# 3. 预后评估：两步 Cox + 常数（与 SPSS 完全一致）
# ==========================================
# Step 1：6MWT + WHO-FC + NT-proBNP -> xbeta_step1（SPSS 标尺）
COEF_6MWT = -0.0060487159
FC_COEF_MAP = {
    1: 0.0,              # FC(1) 参考组
    2: -0.8677105258,    # FC(2)
    3: -3.20036354,      # FC(3)
    4: -1.64640119,      # FC(4)
}
COEF_BNP = 0.0004712203

COX_XBETA_OFFSET = 3.7917941943  # 使 xbeta_step1 与 SPSS 完全一致的常数

def compute_xbeta_step1(six_mwt: float, who_fc: int, ntprobnp: float) -> float:
    b_fc = FC_COEF_MAP.get(int(who_fc), 0.0)
    xbeta_raw = COEF_6MWT * six_mwt + b_fc + COEF_BNP * ntprobnp
    xbeta_spss = xbeta_raw + COX_XBETA_OFFSET
    return xbeta_spss

# Step 2：联合 Cox：xbeta_step1 + R/T 比值 -> 最终 xbeta
COEF_XBETA = 0.7641513097
COEF_RT_RATIO = 0.1894249156
COX_COMBINED_OFFSET = -0.8246894986
PROGNOSIS_THRESHOLD = 0.50359  # 联合模型截断值

def compute_combined_xbeta(six_mwt: float, who_fc: int, ntprobnp: float, rt_ratio: float):
    xbeta_step1 = compute_xbeta_step1(six_mwt, who_fc, ntprobnp)
    combined_xbeta = (
        COEF_XBETA * xbeta_step1 +
        COEF_RT_RATIO * rt_ratio +
        COX_COMBINED_OFFSET
    )
    return combined_xbeta, xbeta_step1

# ==========================================
# 4. 加载随机森林模型与特征（PH 检测）
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
# 5. 标题 & 参数输入（主区域，3 行 × 5 列）
# ==========================================
st.title("🏥 基于心磁成像装置的肺动脉高压风险计算器")
st.markdown("---")

six_mwt = None
who_fc = None
ntprobnp = None
rt_ratio = None
input_df = None

if model is not None and feature_names is not None:
    st.subheader("📋 受试者参数录入")
    st.markdown("请在下方输入心磁和临床特征参数值，然后点击下方的“预测”按钮。")

    input_data = {}

    # ------- 诊断模型特征：按 5 列排版，最多 3 行 -------
    cols_per_row = 5
    n_features = len(feature_names)
    n_rows = math.ceil(n_features / cols_per_row)

    for r in range(n_rows):
        row_cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx >= n_features:
                break
            feature = feature_names[idx]
            feature_lower = feature.lower()

            with row_cols[c]:
                if 'sex' in feature_lower or 'gender' in feature_lower or 'code' in feature_lower:
                    input_data[feature] = st.selectbox(
                        f"{feature}",
                        options=[0, 1],
                        index=0
                    )
                else:
                    input_data[feature] = st.number_input(
                        f"{feature}",
                        value=0.0,
                        format="%.2f"
                    )

    input_df = pd.DataFrame([input_data], columns=feature_names)

    # ------- 预后评估参数：单独一行（4 列） -------
    st.markdown("#### ✨ 预后评估参数")
    prog_cols = st.columns(4)

    with prog_cols[0]:
        six_mwt = st.number_input("6分钟步行距离 (6MWT, m)", min_value=0.0, value=0.0, step=1.0)
    with prog_cols[1]:
        who_fc = st.selectbox("WHO 心功能分级 (1-4)", options=[1, 2, 3, 4], index=0)
    with prog_cols[2]:
        ntprobnp = st.number_input("NT-proBNP", min_value=0.0, value=0.0, step=1.0)
    with prog_cols[3]:
        rt_ratio = st.number_input("R波和T波峰值时刻两极磁感应强度差值比值", value=0.0, format="%.2f")

    # 预测按钮：居中铺满
    predict_clicked = st.button("🔍 预测", use_container_width=True)
else:
    predict_clicked = False
# ==========================================
# 6. 主界面：PH 检测 + SHAP + 预后评估
# ==========================================
if predict_clicked and (model is not None) and (input_df is not None):
    with st.spinner('正在计算模型预测风险与 SHAP 解释值，请稍候...'):
        # A. 随机森林预测（PH 是否高风险）
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

        # ========= 左列：PH 检测 + 预后 =========
        with col1:
            st.markdown("### 📊 肺动脉高压检测结果")

            risk_percent = probability * 100
            optimal_threshold = 35.703   # 内部高/低风险划分阈值（百分比）
            youden_index = 0.771

            if risk_percent > optimal_threshold:
                color = "#dc3545"
                risk_label = "高风险"
                icon = "⚠️"
                advice_box = "warning"
                advice_text = (
                    "模型评估结果为 **高风险**，提示患者当前患肺动脉高压的可能性较高。\n\n"
                    "**建议：** 建议进一步完善右心导管检查，"
                    "并结合临床情况进行综合评估。"
                )
            else:
                color = "#28a745"
                risk_label = "低风险"
                icon = "✅"
                advice_box = "success"
                advice_text = (
                    "模型评估结果为 **低风险**，提示患者当前患肺动脉高压的可能性较低。\n\n"
                    "**建议：** 可继续观察、密切随访，根据临床症状和体征决定是否进一步检查。"
                )
            
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

            # ---- 第二步：仅在 PH 高风险时进行预后评估 ----
            if risk_percent > optimal_threshold:
                st.markdown("---")
                st.markdown("### 📈 预后评估（临床恶化风险）")

                try:
                    combined_xbeta, xbeta_step1 = compute_combined_xbeta(
                        six_mwt or 0.0,
                        int(who_fc) if who_fc is not None else 1,
                        ntprobnp or 0.0,
                        rt_ratio or 0.0
                    )

                    # 判定高危 / 低危
                    if combined_xbeta >= PROGNOSIS_THRESHOLD:
                        prog_label = "高危"
                        prog_color = "#dc3545"
                        prog_icon = "⚠️"
                        prog_box_type = "warning"
                    else:
                        prog_label = "低危"
                        prog_color = "#28a745"
                        prog_icon = "✅"
                        prog_box_type = "success"

                    st.markdown(
                        f"""
                        <div class="report-box" style="border-left: 5px solid {prog_color};">
                            <h3 style="color:{prog_color}; margin:0;">{prog_icon} 临床恶化：{prog_label}</h3>
                            <p style="color: gray; font-size: 13px; margin-top:8px;">
                                预后评估模型为基于心磁特征参数的联合模型。
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("#### 📌 预后决策建议")
                    if prog_box_type == "warning":
                        st.warning(
                            "模型提示患者存在 **临床恶化高危**。\n\n"
                            "建议在肺动脉高压专科医生评估下：\n"
                            "- 密切随访临床症状、体征和 WHO 心功能分级；\n"
                            "- 考虑强化或调整治疗方案；\n"
                            "- 缩短随访间隔，定期监测 6MWT、NT-proBNP 及超声心动图和心磁成像。\n\n"
                            "以上内容仅供科研与辅助决策参考，不能替代临床医生的个体化判断。"
                        )
                    else:
                        st.success(
                            "模型提示患者目前为 **临床恶化低危**。\n\n"
                            "建议：\n"
                            "- 继续现有治疗和管理方案；\n"
                            "- 按既定计划定期随访，复查 6MWT、NT-proBNP 和 WHO 心功能分级；\n"
                            "- 如出现气促加重、晕厥等症状，应及时就诊并重新评估。\n\n"
                            "以上内容仅供科研与辅助决策参考，不能替代临床医生的个体化判断。"
                        )

                except Exception as e:
                    st.error(f"预后评估计算失败，请检查输入参数：{e}")
            else:
                st.markdown("---")
                st.info("当前为 **低风险**，暂不进行临床恶化预后评估。")

        # ========= 右列：SHAP 瀑布图 =========
        with col2:
            st.markdown("### 🔍 SHAP 可解释性分析")
            st.markdown("下图展示了各特征对预测结果的贡献：**红色**条表示增加风险，**蓝色**条表示降低风险。")
            
            if final_explanation is not None:
                try:
                    fig, ax = plt.subplots(figsize=(8, 6))

                    plt.rcParams["font.family"] = "sans-serif"
                    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
                    plt.rcParams["axes.unicode_minus"] = False

                    shap.plots.waterfall(final_explanation, show=False, max_display=10)
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
    st.info("👉 请在上方输入患者的参数后，点击“预测”按钮。")
