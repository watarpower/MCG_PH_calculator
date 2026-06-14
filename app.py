# -*- coding: utf-8 -*-

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
from typing import Any, List, Tuple, Optional


# ==========================================
# 0. 全局参数
# ==========================================

MODEL_FILE = "XGB_final_model_package.pkl"
FEATURE_FILE = "final_selected_features.txt"

# 心磁成像肺动脉高压检测模型固定截断值
PH_DETECTION_THRESHOLD = 0.420199602842331

# 如果特征文件意外缺失，使用该列表兜底
FALLBACK_FEATURES = [
    "QT间期",
    "R峰-T峰最大电流矢量强度不匹配指数",
    "R峰-T峰总电流矢量角度差值",
    "R峰两极角度",
    "T峰两极角度",
    "T峰两极距离",
    "T峰两极不对称指数",
    "R峰-T峰两极角度差值",
    "QRS间期最大阴极磁感应强度",
]


# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="基于心磁成像技术的肺动脉高压风险计算器",
    page_icon="🏥",
    layout="wide"
)


# ==========================================
# 2. 字体环境
# ==========================================
def configure_font_environment() -> bool:
    """
    优先使用系统字体。
    Streamlit Cloud 可在 packages.txt 中安装 fonts-wqy-microhei。
    """
    preferred_fonts = [
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]

    font_filename = "SimHei.ttf"
    font_url = "https://cdn.jsdelivr.net/gh/StellarCN/scp_zh@master/fonts/SimHei.ttf"

    if not os.path.exists(font_filename):
        try:
            response = requests.get(font_url, timeout=10)
            if response.status_code == 200:
                with open(font_filename, "wb") as f:
                    f.write(response.content)
        except Exception:
            pass

    if os.path.exists(font_filename):
        try:
            fm.fontManager.addfont(font_filename)
        except Exception:
            pass

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
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .report-box {
        border: 1px solid #e6e6e6;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .threshold-info {
        font-size: 12px;
        color: #666;
        text-align: center;
        margin-top: 5px;
    }
    .probability-box {
        text-align: center;
        font-size: 22px;
        margin-top: 12px;
        color: #2c3e50;
    }
    .stAlert p,
    .stAlert ul,
    .stAlert li {
        font-size: 21px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 4. 防止 joblib 加载自定义转换器时报错
# ==========================================
class DataFrameConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X)


# ==========================================
# 5. 预后评估：两步 Cox + 常数
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

def compute_combined_xbeta(six_mwt: float, who_fc: int, ntprobnp: float, rt_ratio: float) -> Tuple[float, float]:
    xbeta_step1 = compute_xbeta_step1(six_mwt, who_fc, ntprobnp)
    combined_xbeta = (
        COEF_XBETA * xbeta_step1 +
        COEF_RT_RATIO * rt_ratio +
        COX_COMBINED_OFFSET
    )
    return combined_xbeta, xbeta_step1


# ==========================================
# 6. 加载 XGB 模型与最终特征
# ==========================================
def parse_feature_content(content: str) -> List[str]:
    """
    支持按换行、英文逗号或中文逗号分隔的特征文件。
    """
    content = content.replace("\ufeff", "").strip()
    if not content:
        return []

    if "," in content or "，" in content:
        content = content.replace("，", ",")
        features = [x.strip() for x in content.split(",") if x.strip()]
    else:
        features = [x.strip() for x in content.splitlines() if x.strip()]

    # 去重但保留原始顺序
    seen = set()
    unique_features = []
    for f in features:
        if f not in seen:
            unique_features.append(f)
            seen.add(f)
    return unique_features


def extract_model_from_package(package: Any) -> Any:
    """
    兼容以下保存格式：
    1. 直接保存模型对象；
    2. dict包，例如 {"model": model, "features": [...], ...}；
    3. tuple/list，例如 (model, features)。
    """
    if isinstance(package, dict):
        candidate_keys = [
            "model",
            "final_model",
            "best_model",
            "estimator",
            "pipeline",
            "clf",
            "classifier",
            "xgb_model",
        ]
        for key in candidate_keys:
            if key in package and package[key] is not None:
                return package[key]

        raise ValueError(
            "模型文件是dict，但未找到可识别的模型键。"
            f"当前键包括：{list(package.keys())}"
        )

    if isinstance(package, (list, tuple)):
        for item in package:
            if hasattr(item, "predict") or hasattr(item, "predict_proba"):
                return item

    return package


def extract_features_from_package(package: Any) -> Optional[List[str]]:
    """
    如果模型包中也保存了特征名，可以作为兜底。
    但本程序优先使用 final_selected_features.txt。
    """
    if isinstance(package, dict):
        candidate_keys = [
            "features",
            "feature_names",
            "selected_features",
            "final_features",
            "feature_names_in_",
        ]
        for key in candidate_keys:
            if key in package and package[key] is not None:
                vals = package[key]
                if isinstance(vals, (list, tuple, np.ndarray, pd.Index)):
                    return [str(x) for x in list(vals)]

    obj = extract_model_from_package(package)
    if hasattr(obj, "feature_names_in_"):
        return [str(x) for x in obj.feature_names_in_]

    return None


@st.cache_resource
def load_model_and_features() -> Tuple[Optional[Any], Optional[List[str]]]:
    try:
        package = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        st.error(f"❌ 错误：未找到模型文件 `{MODEL_FILE}`。请确认该文件已上传至 GitHub 仓库根目录。")
        return None, None
    except Exception as e:
        st.error(f"❌ 模型文件 `{MODEL_FILE}` 加载失败：{e}")
        return None, None

    try:
        model = extract_model_from_package(package)
    except Exception as e:
        st.error(f"❌ 无法从 `{MODEL_FILE}` 中解析模型对象：{e}")
        return None, None

    features = None
    try:
        with open(FEATURE_FILE, "r", encoding="utf-8") as f:
            features = parse_feature_content(f.read())
    except FileNotFoundError:
        package_features = extract_features_from_package(package)
        if package_features:
            features = package_features
            st.warning(f"未找到 `{FEATURE_FILE}`，已使用模型包内保存的特征名。")
        else:
            features = FALLBACK_FEATURES
            st.warning(f"未找到 `{FEATURE_FILE}`，已使用代码内置的最终特征列表。")
    except Exception as e:
        st.error(f"❌ 特征文件 `{FEATURE_FILE}` 读取失败：{e}")
        return None, None

    if not features:
        st.error(f"❌ 特征文件 `{FEATURE_FILE}` 为空，无法构建输入表单。")
        return None, None

    return model, features


def is_sklearn_pipeline(obj: Any) -> bool:
    return hasattr(obj, "steps") and isinstance(getattr(obj, "steps", None), list)


def transform_through_pipeline(pipeline: Any, X: pd.DataFrame):
    """
    对 sklearn Pipeline 执行除最后一个估计器以外的 transform。
    """
    Xt = X
    for _, step in pipeline.steps[:-1]:
        if step is None or step == "passthrough":
            continue
        Xt = step.transform(Xt)
    return Xt


def get_final_estimator(model: Any) -> Any:
    if is_sklearn_pipeline(model):
        return model.steps[-1][1]
    return model


def get_processed_data_for_shap(model: Any, X: pd.DataFrame) -> Tuple[Any, pd.DataFrame, List[str]]:
    """
    返回 final_estimator、processed_df 和解释图中使用的特征名。
    """
    if is_sklearn_pipeline(model):
        final_estimator = model.steps[-1][1]
        processed = transform_through_pipeline(model, X)
        if hasattr(processed, "toarray"):
            processed = processed.toarray()

        processed_df = pd.DataFrame(processed)

        if processed_df.shape[1] == len(X.columns):
            used_feature_names = list(X.columns)
            processed_df.columns = used_feature_names
        elif hasattr(final_estimator, "feature_names_in_") and len(final_estimator.feature_names_in_) == processed_df.shape[1]:
            used_feature_names = [str(x) for x in final_estimator.feature_names_in_]
            processed_df.columns = used_feature_names
        else:
            used_feature_names = [f"f{i}" for i in range(processed_df.shape[1])]
            processed_df.columns = used_feature_names
    else:
        final_estimator = model
        processed_df = X.copy()
        used_feature_names = list(X.columns)

    return final_estimator, processed_df, used_feature_names


def predict_positive_probability(model: Any, X: pd.DataFrame) -> float:
    """
    尽量提取阳性类别概率。
    兼容 sklearn / XGBoost / 部分只返回概率的模型。
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        if isinstance(proba, list):
            proba = np.asarray(proba)

        proba = np.asarray(proba)

        if proba.ndim == 2:
            if proba.shape[1] == 1:
                return float(proba[0, 0])

            classes = getattr(model, "classes_", None)
            if classes is not None and 1 in list(classes):
                pos_idx = list(classes).index(1)
            else:
                pos_idx = 1
            return float(proba[0, pos_idx])

        return float(proba.ravel()[0])

    # 如果没有 predict_proba，尝试 predict。
    pred = model.predict(X)
    pred = np.asarray(pred).ravel()[0]

    # 如果 predict 返回0-1连续概率，则直接使用；如果返回类别，则映射为0或1。
    try:
        pred_float = float(pred)
        if 0.0 <= pred_float <= 1.0:
            return pred_float
        return 1.0 if pred_float >= 1.0 else 0.0
    except Exception:
        return 1.0 if pred == 1 else 0.0


model, feature_names = load_model_and_features()


# ==========================================
# 7. 标题与输入区域
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
    st.markdown("请在下方输入心磁特征参数值，然后点击“预测”按钮。")

    with st.expander("当前模型配置", expanded=False):
        st.write(f"模型文件：`{MODEL_FILE}`")
        st.write(f"特征文件：`{FEATURE_FILE}`")
        st.write(f"肺动脉高压检测截断值：`{PH_DETECTION_THRESHOLD:.15f}`")
        st.write("模型输入特征：")
        st.write(feature_names)

    do_shap = st.checkbox("显示 SHAP 解释（较慢，可能占用更多资源）", value=False)

    input_data = {}

    cols_per_row = 3
    n_features = len(feature_names)
    n_rows = math.ceil(n_features / cols_per_row)

    for r in range(n_rows):
        row_cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            idx = r * cols_per_row + c
            if idx >= n_features:
                break

            feature = feature_names[idx]

            with row_cols[c]:
                input_data[feature] = st.number_input(
                    label=f"{feature}",
                    value=0.0,
                    format="%.6f",
                    key=f"input_{idx}_{feature}"
                )

    input_df = pd.DataFrame([input_data], columns=feature_names).astype(float)

    st.markdown("---")
    st.markdown("#### ✨ 预后评估参数")
    st.caption("预后评估仅在肺动脉高压检测结果为高风险时显示。")

    prog_cols = st.columns(4)

    with prog_cols[0]:
        six_mwt = st.number_input("6分钟步行距离 (6MWT, m)", min_value=0.0, value=0.0, step=1.0)
    with prog_cols[1]:
        who_fc = st.selectbox("WHO 心功能分级 (1-4)", options=[1, 2, 3, 4], index=0)
    with prog_cols[2]:
        ntprobnp = st.number_input("NT-proBNP", min_value=0.0, value=0.0, step=1.0)
    with prog_cols[3]:
        rt_ratio = st.number_input("R波和T波峰值时刻两极磁感应强度差值比值", value=0.0, format="%.6f")

    predict_clicked = st.button("🔍 预测", use_container_width=True)
else:
    do_shap = False
    predict_clicked = False


# ==========================================
# 8. 主界面：PH检测 + SHAP + 预后评估
# ==========================================
if predict_clicked and (model is not None) and (input_df is not None):

    # A. 预测
    with st.spinner("正在计算模型预测风险，请稍候..."):
        try:
            probability = predict_positive_probability(model, input_df)
            probability = float(np.clip(probability, 0.0, 1.0))
        except Exception as e:
            st.error(f"模型预测失败，请检查输入特征名称、顺序和模型文件是否一致：{e}")
            st.stop()

    is_high_risk = probability >= PH_DETECTION_THRESHOLD

    # B. 可选：SHAP
    final_explanation = None
    shap_ready = False

    if do_shap:
        with st.spinner("正在计算 SHAP 解释值（可能较慢）..."):
            try:
                final_estimator, processed_df, used_feature_names = get_processed_data_for_shap(model, input_df)

                est_id = id(final_estimator)
                cache_key = "shap_explainer"
                cache_id_key = "shap_explainer_est_id"

                if st.session_state.get(cache_id_key) == est_id and cache_key in st.session_state:
                    explainer = st.session_state[cache_key]
                else:
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

                if hasattr(shap_values_obj, "values"):
                    values = shap_values_obj.values
                    base_values = shap_values_obj.base_values

                    if len(values.shape) == 3:
                        shap_contribution = values[0, :, 1]
                        base_val = base_values[0, 1]
                    else:
                        shap_contribution = values[0]
                        if np.asarray(base_values).ndim == 0:
                            base_val = base_values
                        else:
                            base_val = np.asarray(base_values).ravel()[0]

                    final_explanation = shap.Explanation(
                        values=shap_contribution,
                        base_values=base_val,
                        data=processed_df.iloc[0].values,
                        feature_names=used_feature_names
                    )
                    shap_ready = True
                else:
                    st.warning("SHAP 计算未返回有效结果。")

            except Exception as e:
                st.error(f"SHAP 计算模块出错：{e}")
                final_explanation = None
                shap_ready = False

    # C. 结果展示
    col1, col2 = st.columns([1, 2])

    # ========= 左列：PH检测 + 预后 =========
    with col1:
        st.markdown("### 📊 预测结果")

        risk_percent = probability * 100
        threshold_percent = PH_DETECTION_THRESHOLD * 100

        if is_high_risk:
            color = "#dc3545"
            risk_label = "肺动脉高压高风险"
            icon = "⚠️"
            advice_box = "warning"
            advice_text = (
                "模型评估结果为 **高风险**，提示患者当前患肺动脉高压的可能性较高。\n\n"
                "**建议：** 建议进一步完善超声心动图、右心导管检查及相关病因学评估，"
                "并结合临床症状、体征和其他检查结果进行综合判断。"
            )
        else:
            color = "#28a745"
            risk_label = "肺动脉高压低风险"
            icon = "✅"
            advice_box = "success"
            advice_text = (
                "模型评估结果为 **低风险**，提示患者当前患肺动脉高压的可能性较低。\n\n"
                "**建议：** 可结合临床情况继续观察和随访；如存在持续胸闷、气促、晕厥、右心负荷增高等表现，"
                "仍建议进一步专科评估。"
            )

        st.markdown(
            f"""
            <div class="report-box" style="text-align: center; border-left: 5px solid {color};">
                <h2 style="color: {color}; font-size: 36px; margin: 0;">{icon} {risk_label}</h2>
                <div class="probability-box">
                    预测概率：<b>{risk_percent:.2f}%</b><br>
                    分类截断值：<b>{threshold_percent:.2f}%</b>
                </div>
                <p style="color: gray; font-size: 14px; margin-top: 10px;">
                    本结果仅供科研与辅助决策参考，不能替代临床医生的诊断。
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

        # ---- 预后评估：仅在PH高风险时显示 ----
        if is_high_risk:
            st.markdown("---")
            st.markdown("### 📈 预后评估")

            try:
                combined_xbeta, xbeta_step1 = compute_combined_xbeta(
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
                        <p style="color: #2c3e50; font-size: 18px; margin-top:10px;">
                            联合预后评分：<b>{combined_xbeta:.4f}</b><br>
                            预后截断值：<b>{PROGNOSIS_THRESHOLD:.5f}</b>
                        </p>
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
                        "- 结合指南和临床情况考虑强化或调整治疗方案；\n"
                        "- 缩短随访间隔，定期监测 6MWT、NT-proBNP、超声心动图和心磁成像。\n\n"
                        "以上内容仅供科研与辅助决策参考，不能替代临床医生的个体化判断。"
                    )
                else:
                    st.success(
                        "模型提示患者目前为 **临床恶化低危**。\n\n"
                        "建议：\n"
                        "- 继续现有治疗和管理方案；\n"
                        "- 按既定计划定期随访，复查 6MWT、NT-proBNP 和 WHO 心功能分级；\n"
                        "- 如出现气促加重、晕厥、下肢水肿加重等表现，应及时就诊并重新评估。\n\n"
                        "以上内容仅供科研与辅助决策参考，不能替代临床医生的个体化判断。"
                    )

            except Exception as e:
                st.error(f"预后评估计算失败，请检查输入参数：{e}")
        else:
            st.markdown("---")
            st.info("当前为 **肺动脉高压低风险**，暂不进行临床恶化预后评估。")

    # ========= 右列：SHAP解释 =========
    with col2:
        st.markdown("### 🔍 SHAP 可解释性分析")
        st.markdown("下图展示各特征对预测结果的贡献：红色通常表示增加模型输出，蓝色通常表示降低模型输出。")

        if not do_shap:
            st.info("SHAP 解释默认关闭，以提升网页稳定性。如需查看解释，请勾选上方“显示 SHAP 解释”。")
        else:
            if shap_ready and (final_explanation is not None):
                try:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.sca(ax)
                    shap.plots.waterfall(final_explanation, show=False, max_display=min(13, len(feature_names)))
                    fix_shap_minus_signs(ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as plot_err:
                    st.error(f"绘图失败。调试信息：{plot_err}")
            else:
                st.warning("无法生成 SHAP 图，请检查输入数据或模型结构。")

    st.markdown("---")
    st.caption(
        f"**说明：** 本工具使用 XGB 心磁成像检测模型；"
        f"肺动脉高压风险分类截断值为 {PH_DETECTION_THRESHOLD:.15f}。"
        "结果仅供科研与辅助决策参考。"
    )

else:
    st.info("👉 请在上方输入患者参数后，点击“预测”按钮。")
