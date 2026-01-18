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
from typing import Tuple, Optional, Any


# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="基于心磁成像装置的肺动脉高压风险计算器",
    page_icon="🏥",
    layout="wide"
)


# ==========================================
# 2. 字体环境（缓存初始化，避免每次 rerun 反复配置）
# ==========================================
def configure_font_environment() -> bool:
    """
    优先使用系统字体（你 packages.txt 已装 fonts-wqy-microhei），
    其次尝试下载 SimHei.ttf（如果你仍需要）。
    """
    # 尽量优先用系统已安装字体（更稳定）
    preferred_fonts = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"]

    # 如果你仍希望保留下载 SimHei 的兜底
    font_filename = "SimHei.ttf"
    font_url = "https://cdn.jsdelivr.net/gh/StellarCN/scp_zh@master/fonts/SimHei.ttf"

    # 若本地没有 SimHei.ttf，则尝试下载（失败也不致命）
    if not os.path.exists(font_filename):
        try:
            response = requests.get(font_url, timeout=10)
            if response.status_code == 200:
                with open(font_filename, "wb") as f:
                    f.write(response.content)
        except Exception:
            # 网络失败就算了，走系统字体
            pass

    # 如果下载成功，把字体注册进 matplotlib
    if os.path.exists(font_filename):
        try:
            fm.fontManager.addfont(font_filename)
        except Exception:
            pass

    # 统一设置 matplotlib 字体和负号
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = preferred_fonts
    plt.rcParams["axes.unicode_minus"] = False

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": preferred_fonts,
        "axes.unicode_minus": False,
    })
    return True


@st.cache_resource
def init_fonts_once() -> bool:
    return configure_font_environment()


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


_ = init_fonts_once()


# ==========================================
# 3. CSS
# ==========================================
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

    /* 调大 st.warning / st.success 内文字和列表字号 */
    .stAlert p,
    .stAlert ul,
    .stAlert li {
        font-size: 23px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 4. 防止 joblib 加载报错的类（保持不变）
# ==========================================
class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X)


# ==========================================
# 5. 预后评估：两步 Cox + 常数（与 SPSS 完全一致）
# ==========================================
COEF_6MWT = -0.0060487159
FC_COEF_MAP = {
    1: 0.0,
    2: -0.8677105258,
    3: -3.20036354,
    4: -1.64640119,
}
COEF_BNP = 0.0004712203
COX_XBETA_OFFSET = 3.7917941943

def compute_xbeta_step1(six_mwt: float, who_fc: int, ntprobnp: float) -> float:
    b_fc = FC_COEF_MAP.get(int(who_fc), 0.0)
    xbeta_raw = COEF_6MWT * six_mwt + b_fc + COEF_BNP * ntprobnp
    return xbeta_raw + COX_XBETA_OFFSET

COEF_XBETA = 0.7641513097
COEF_RT_RATIO = 0.1894249156
COX_COMBINED_OFFSET = -0.8246894986
PROGNOSIS_THRESHOLD = 0.50359

def compute_combined_xbeta(six_mwt: float, who_fc: int, ntprobnp: float, rt_ratio: float):
    xbeta_step1 = compute_xbeta_step1(six_mwt, who_fc, ntprobnp)
    combined_xbeta = (
        COEF_XBETA * xbeta_step1 +
        COEF_RT_RATIO * rt_ratio +
        COX_COMBINED_OFFSET
    )
    return combined_xbeta, xbeta_step1


# ==========================================
# 6. 加载随机森林模型与特征（PH 检测）
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


def is_sklearn_pipeline(obj: Any) -> bool:
    return hasattr(obj, "steps") and isinstance(getattr(obj, "steps", None), list)


def transform_through_pipeline(pipeline: Any, X: pd.DataFrame):
    """
    不用 pipeline[:-1]（会触发“未fit”warning），
    直接按 steps[:-1] 逐步调用 transform。
    """
    Xt = X
    for _, step in pipeline.steps[:-1]:
        if step is None or step == "passthrough":
            continue
        # 有些 step 可能是 ColumnTransformer / Transformer
        Xt = step.transform(Xt)
    return Xt


model, feature_names = load_model_and_features()


# ==========================================
# 7. 标题 & 输入区域
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

    # ✅ SHAP 开关：默认不算，避免点一下就高负载
    do_shap = st.checkbox("显示 SHAP 解释（较慢，可能占用更多资源）", value=False)

    input_data = {}

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

    predict_clicked = st.button("🔍 预测", use_container_width=True)
else:
    do_shap = False
    predict_clicked = False


# ==========================================
# 8. 主界面：PH 检测 +（可选）SHAP + 预后评估
# ==========================================
if predict_clicked and (model is not None) and (input_df is not None):
    # A. 预测（先做轻量预测）
    with st.spinner('正在计算模型预测风险，请稍候...'):
        try:
            probability = model.predict_proba(input_df)[0, 1]
        except Exception:
            prediction = model.predict(input_df)[0]
            probability = 1.0 if prediction == 1 else 0.0

    # B. 可选：SHAP 计算
    final_explanation = None
    shap_ready = False

    if do_shap:
        with st.spinner('正在计算 SHAP 解释值（可能较慢）...'):
            try:
                # pipeline / 非 pipeline 两种
                if is_sklearn_pipeline(model):
                    final_estimator = model.steps[-1][1]
                    processed = transform_through_pipeline(model, input_df)

                    # sparse -> dense
                    if hasattr(processed, "toarray"):
                        processed = processed.toarray()

                    processed_df = pd.DataFrame(processed)
                else:
                    final_estimator = model
                    processed_df = input_df.copy()

                # explainer 缓存在 session_state（每个用户会话内复用）
                est_id = id(final_estimator)
                cache_key = "shap_explainer"
                cache_id_key = "shap_explainer_est_id"

                explainer = None
                if st.session_state.get(cache_id_key) == est_id and cache_key in st.session_state:
                    explainer = st.session_state[cache_key]
                else:
                    # 先尝试普通 TreeExplainer
                    try:
                        explainer = shap.TreeExplainer(final_estimator)
                    except Exception:
                        explainer = shap.TreeExplainer(
                            final_estimator,
                            data=processed_df,
                            model_output="probability"
                        )
                    st.session_state[cache_key] = explainer
                    st.session_state[cache_id_key] = est_id

                shap_values_obj = explainer(processed_df)

                # 取二分类的正类（如果是 3D）
                if hasattr(shap_values_obj, "values"):
                    if len(shap_values_obj.values.shape) == 3:
                        shap_contribution = shap_values_obj.values[0, :, 1]
                        base_val = shap_values_obj.base_values[0, 1]
                    else:
                        shap_contribution = shap_values_obj.values[0]
                        base_val = shap_values_obj.base_values[0]

                    # 如果维度对不上，就用处理后的特征名/数据
                    if processed_df.shape[1] == len(feature_names):
                        data_vals = input_df.iloc[0].values
                        used_feature_names = feature_names
                    else:
                        data_vals = processed_df.iloc[0].values
                        used_feature_names = [f"f{i}" for i in range(processed_df.shape[1])]

                    final_explanation = shap.Explanation(
                        values=shap_contribution,
                        base_values=base_val,
                        data=data_vals,
                        feature_names=used_feature_names
                    )
                    shap_ready = True
                else:
                    st.warning("SHAP 计算未返回有效结果。")

            except Exception as e:
                st.error(f"SHAP 计算模块出错: {str(e)}")
                final_explanation = None
                shap_ready = False

    # C. 结果展示（两列）
    col1, col2 = st.columns([1, 2])

    # ========= 左列：PH 检测 + 预后 =========
    with col1:
        st.markdown("### 📊 预测结果")

        risk_percent = probability * 100
        optimal_threshold = 35.703   # 内部高/低风险阈值（百分比）
        youden_index = 0.771

        if risk_percent > optimal_threshold:
            color = "#dc3545"
            risk_label = "肺动脉高压高风险"
            icon = "⚠️"
            advice_box = "warning"
            advice_text = (
                "模型评估结果为 **高风险**，提示患者当前患肺动脉高压的可能性较高。\n\n"
                "**建议：** 建议进一步完善右心导管检查，"
                "并结合临床情况进行综合评估。"
            )
        else:
            color = "#28a745"
            risk_label = "肺动脉高压低风险"
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
            st.markdown("### 📈 预后评估")

            try:
                combined_xbeta, _ = compute_combined_xbeta(
                    six_mwt or 0.0,
                    int(who_fc) if who_fc is not None else 1,
                    ntprobnp or 0.0,
                    rt_ratio or 0.0
                )

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
                            本结果仅供科研与辅助决策参考，不能替代临床医生的个体化判断。
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

        if not do_shap:
            st.info("你已关闭 SHAP 解释（默认关闭以提升稳定性）。如需查看解释，请勾选上方“显示 SHAP 解释”。")
        else:
            if shap_ready and (final_explanation is not None):
                try:
                    fig, ax = plt.subplots(figsize=(8, 6))

                    # 让 shap 画到我们创建的 ax 上（更稳）
                    plt.sca(ax)
                    shap.plots.waterfall(final_explanation, show=False, max_display=13)

                    fix_shap_minus_signs(ax)
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close(fig)  # ✅ 关键：释放内存，防止越点越占导致空白
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
