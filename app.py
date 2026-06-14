# -*- coding: utf-8 -*-

import math
from typing import Any, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from sklearn.base import BaseEstimator, TransformerMixin


# ==========================================
# 1. 页面配置
# ==========================================

st.set_page_config(
    page_title="基于心磁成像技术的肺动脉高压患病及临床恶化风险计算器",
    page_icon="🏥",
    layout="wide"
)


# ==========================================
# 2. 全局参数
# ==========================================

MODEL_FILE = "XGB_final_model_package.pkl"
FEATURE_FILE = "final_selected_features.txt"

PH_DETECTION_THRESHOLD = 0.420199602842331
PROGNOSIS_THRESHOLD = 0.3202

# 页面显示名 -> 模型内部真实特征名
DISPLAY_TO_MODEL_NAME = {
    "QT间期": "QTc",
}

# 模型内部真实特征名 -> 页面显示名
MODEL_TO_DISPLAY_NAME = {
    "QTc": "QT间期",
}

# 兜底特征列表：这里必须使用模型训练时的内部特征名，即 QTc
DEFAULT_MODEL_FEATURES = [
    "QTc",
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
# 3. 预后评估：两步 Cox 模型
# ==========================================

# 第一步：基线 Cox 模型
# 变量：6MWT、WHO-FC、NT-proBNP
# WHO-FC 以 I 级为参照

COEF_6MWT = -0.00211921
COEF_BNP = 0.00043149

FC_COEF_MAP = {
    1: 0.0,
    2: 0.41155628,
    3: -0.82986348,
    4: -2.1000644,
}


def compute_xbeta_step1(six_mwt: float, who_fc: int, ntprobnp: float) -> float:
    """
    基线 Cox 模型 Xβ：

    Xβ_baseline =
    -0.00211921 × 6MWT
    + WHO-FC对应系数
    + 0.00043149 × NT-proBNP
    """
    b_fc = FC_COEF_MAP.get(int(who_fc), 0.0)

    xbeta_baseline = (
        COEF_6MWT * float(six_mwt)
        + b_fc
        + COEF_BNP * float(ntprobnp)
    )

    return float(xbeta_baseline)


# 第二步：联合 Cox 模型
# 变量：基线模型 Xβ、R峰-T峰两极角度差值、R峰-T峰磁感应强度梯度比值

COEF_BASELINE_XBETA = 0.75531729
COEF_RT_ANGLE_DIFF = 0.01020034
COEF_RT_GRADIENT_RATIO = 0.13618361


def compute_combined_xbeta(
    six_mwt: float,
    who_fc: int,
    ntprobnp: float,
    rt_angle_diff: float,
    rt_gradient_ratio: float
) -> Tuple[float, float]:
    """
    联合 Cox 模型 Xβ：

    Xβ_combined =
    0.75531729 × Xβ_baseline
    + 0.01020034 × R峰-T峰两极角度差值
    + 0.13618361 × R峰-T峰磁感应强度梯度比值
    """

    xbeta_baseline = compute_xbeta_step1(
        six_mwt=six_mwt,
        who_fc=who_fc,
        ntprobnp=ntprobnp
    )

    xbeta_combined = (
        COEF_BASELINE_XBETA * xbeta_baseline
        + COEF_RT_ANGLE_DIFF * float(rt_angle_diff)
        + COEF_RT_GRADIENT_RATIO * float(rt_gradient_ratio)
    )

    return float(xbeta_combined), float(xbeta_baseline)


# ==========================================
# 4. 字体设置
# ==========================================

def configure_font_environment() -> bool:
    preferred_fonts = [
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans"
    ]

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


_ = init_fonts_once()


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


# ==========================================
# 5. CSS
# ==========================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }

    h1 {
        color: #2c3e50;
        font-weight: bold;
        font-family: sans-serif;
    }

    h3 {
        color: #34495e;
        font-family: sans-serif;
    }

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

    /* 输入参数名称字体大小 */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
    }

    /* 输入框内数字字体大小 */
    div[data-testid="stNumberInput"] input {
        font-size: 18px !important;
        font-weight: 500 !important;
    }

    /* 下拉框字体大小，例如 WHO 心功能分级 */
    div[data-testid="stSelectbox"] div {
        font-size: 18px !important;
    }

    /* 普通说明文字 */
    .stMarkdown p {
        font-size: 17px;
    }

    .stAlert p,
    .stAlert ul,
    .stAlert li {
        font-size: 20px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 6. 兼容旧模型 pipeline 的类
# ==========================================

class DataFrameConverter(BaseEstimator, TransformerMixin):
    """
    用于兼容旧模型 pipeline 中可能存在的 DataFrameConverter。

    关键点：
    如果输入已经是 DataFrame，必须保留原始列名。
    否则 XGBoost 可能看到 0,1,2...8 这种列名，并报 feature_names mismatch。
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()

        arr = np.asarray(X)
        cols = getattr(self, "feature_names_in_", None)

        if cols is not None and len(cols) == arr.shape[1]:
            return pd.DataFrame(arr, columns=cols)

        return pd.DataFrame(arr)


# ==========================================
# 7. 工具函数：特征名、模型加载、预测
# ==========================================

def to_model_feature_name(feature_name: str) -> str:
    """
    页面或 txt 中的显示名 -> 模型内部真实特征名。
    """
    feature_name = str(feature_name).strip()
    return DISPLAY_TO_MODEL_NAME.get(feature_name, feature_name)


def to_display_feature_name(feature_name: str) -> str:
    """
    模型内部真实特征名 -> 页面显示名。
    """
    feature_name = str(feature_name).strip()
    return MODEL_TO_DISPLAY_NAME.get(feature_name, feature_name)


def parse_feature_file(feature_file: str) -> List[str]:
    """
    读取特征文件。
    支持逗号分隔或换行分隔。
    """
    with open(feature_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    if "," in content:
        raw_features = [x.strip() for x in content.split(",") if x.strip()]
    else:
        raw_features = [x.strip() for x in content.splitlines() if x.strip()]

    model_features = [to_model_feature_name(f) for f in raw_features]

    cleaned = []
    for f in model_features:
        if f not in cleaned:
            cleaned.append(f)

    return cleaned


def extract_estimator_from_package(obj: Any) -> Any:
    """
    兼容模型文件为单个 estimator 或 dict package 的情况。
    """
    if isinstance(obj, dict):
        candidate_keys = [
            "model",
            "final_model",
            "best_model",
            "estimator",
            "pipeline",
            "clf",
            "classifier"
        ]

        for key in candidate_keys:
            if key in obj:
                return obj[key]

    return obj


def extract_features_from_package(obj: Any) -> Optional[List[str]]:
    """
    如果 pkl package 中保存了特征名，则尝试读取。
    """
    if isinstance(obj, dict):
        candidate_keys = [
            "feature_names",
            "features",
            "selected_features",
            "final_selected_features",
            "model_features"
        ]

        for key in candidate_keys:
            if key in obj:
                feats = obj[key]
                if isinstance(feats, (list, tuple, np.ndarray, pd.Index)):
                    return [to_model_feature_name(str(f)) for f in list(feats)]

    return None


def is_sklearn_pipeline(obj: Any) -> bool:
    return hasattr(obj, "steps") and isinstance(getattr(obj, "steps", None), list)


def get_final_estimator(model_obj: Any) -> Any:
    if is_sklearn_pipeline(model_obj):
        return model_obj.steps[-1][1]
    return model_obj


def get_estimator_feature_names(model_obj: Any) -> Optional[List[str]]:
    """
    从模型本身读取训练时的特征名。
    优先读取 XGBoost booster 的 feature_names。
    """

    # 如果是 pipeline，优先看最后一步 XGBoost booster
    if is_sklearn_pipeline(model_obj):
        try:
            final_estimator = model_obj.steps[-1][1]

            if hasattr(final_estimator, "get_booster"):
                booster = final_estimator.get_booster()
                if getattr(booster, "feature_names", None) is not None:
                    return [str(x) for x in booster.feature_names]

            if hasattr(final_estimator, "feature_names_in_"):
                return [str(x) for x in list(final_estimator.feature_names_in_)]
        except Exception:
            pass

    # XGBoost sklearn wrapper
    try:
        if hasattr(model_obj, "get_booster"):
            booster = model_obj.get_booster()
            if getattr(booster, "feature_names", None) is not None:
                return [str(x) for x in booster.feature_names]
    except Exception:
        pass

    # sklearn estimator
    if hasattr(model_obj, "feature_names_in_"):
        return [str(x) for x in list(model_obj.feature_names_in_)]

    return None


@st.cache_resource
def load_model_and_features():
    """
    返回：
    model_obj: 模型或 pipeline
    model_features: 传入模型的内部特征名
    display_features: 页面显示的特征名
    """

    try:
        raw_package = joblib.load(MODEL_FILE)
        model_obj = extract_estimator_from_package(raw_package)
    except FileNotFoundError:
        st.error(f"❌ 模型文件 `{MODEL_FILE}` 未找到。请确认该文件已放在 GitHub 仓库根目录。")
        return None, None, None
    except ModuleNotFoundError as e:
        st.error(
            f"❌ 模型文件 `{MODEL_FILE}` 加载失败：缺少 Python 模块 `{e.name}`。\n\n"
            f"如果这是 XGBoost 模型，请在 requirements.txt 中加入：\n\n"
            f"`xgboost`"
        )
        return None, None, None
    except Exception as e:
        st.error(f"❌ 模型文件 `{MODEL_FILE}` 加载失败：{e}")
        return None, None, None

    # 1. 优先使用模型训练时保存的特征名
    model_features = get_estimator_feature_names(model_obj)

    # 2. 如果模型自身没有，则尝试从 package 读取
    if model_features is None:
        model_features = extract_features_from_package(raw_package)

    # 3. 如果 package 也没有，则读取 txt
    if model_features is None:
        try:
            model_features = parse_feature_file(FEATURE_FILE)
        except FileNotFoundError:
            st.warning(f"⚠️ 未找到 `{FEATURE_FILE}`，将使用代码内置的默认特征列表。")
            model_features = DEFAULT_MODEL_FEATURES.copy()
        except Exception as e:
            st.warning(f"⚠️ `{FEATURE_FILE}` 读取失败：{e}。将使用代码内置的默认特征列表。")
            model_features = DEFAULT_MODEL_FEATURES.copy()

    # 强制 QT间期 -> QTc
    model_features = [to_model_feature_name(f) for f in model_features]

    # 去重保序
    cleaned = []
    for f in model_features:
        if f not in cleaned:
            cleaned.append(f)

    model_features = cleaned
    display_features = [to_display_feature_name(f) for f in model_features]

    return model_obj, model_features, display_features


def prepare_input_for_model(input_df: pd.DataFrame, model_features: List[str]) -> pd.DataFrame:
    """
    预测前最后一次校正列名和顺序。
    """

    X = input_df.copy()

    if "QT间期" in X.columns:
        X = X.rename(columns={"QT间期": "QTc"})

    corrected_features = [to_model_feature_name(f) for f in model_features]

    missing_cols = [f for f in corrected_features if f not in X.columns]
    extra_cols = [f for f in X.columns if f not in corrected_features]

    if missing_cols:
        raise ValueError(
            "输入数据缺少模型所需特征："
            + "，".join(missing_cols)
            + "\n当前输入特征为："
            + "，".join(list(X.columns))
        )

    if extra_cols:
        X = X.drop(columns=extra_cols)

    X = X[corrected_features].astype(float)

    return X


def predict_probability_safely(model_obj: Any, X_named: pd.DataFrame) -> float:
    """
    安全预测概率。

    优先尝试原模型或 pipeline；
    如果 pipeline 内部导致列名错乱，则尝试直接使用最后的 XGBoost estimator。
    """

    first_error = None

    try:
        if hasattr(model_obj, "predict_proba"):
            return float(model_obj.predict_proba(X_named)[0, 1])
    except Exception as e:
        first_error = e

    try:
        final_estimator = get_final_estimator(model_obj)

        if hasattr(final_estimator, "predict_proba"):
            return float(final_estimator.predict_proba(X_named)[0, 1])

        if hasattr(final_estimator, "predict"):
            pred = final_estimator.predict(X_named)
            pred = np.asarray(pred).ravel()[0]
            return float(pred)

    except Exception as e2:
        if first_error is not None:
            raise RuntimeError(
                f"原模型预测失败：{first_error}\n"
                f"最终 estimator 预测也失败：{e2}"
            )
        raise e2

    try:
        pred = model_obj.predict(X_named)
        pred = np.asarray(pred).ravel()[0]
        return float(pred)
    except Exception as e3:
        if first_error is not None:
            raise RuntimeError(
                f"模型预测失败：{first_error}\n"
                f"备用预测也失败：{e3}"
            )
        raise e3


# ==========================================
# 8. XGBoost 原生 SHAP
# ==========================================

def compute_xgb_native_shap_explanation(
    model_obj: Any,
    X_named: pd.DataFrame,
    model_features: List[str]
):
    """
    使用 XGBoost 原生 pred_contribs=True 计算 SHAP 贡献值。

    这样可以绕过 shap.TreeExplainer 对某些 XGBoost 保存格式解析失败的问题，
    例如：could not convert string to float: '[5E-1]'
    """

    if not XGB_AVAILABLE:
        raise RuntimeError("当前环境未安装 xgboost，无法计算 XGBoost 原生 SHAP。")

    if not SHAP_AVAILABLE:
        raise RuntimeError("当前环境未安装 shap，无法绘制 SHAP 图。")

    X_for_model = prepare_input_for_model(X_named, model_features)

    final_estimator = get_final_estimator(model_obj)

    if not hasattr(final_estimator, "get_booster"):
        raise RuntimeError("当前最终模型不是 XGBoost 模型，无法使用 XGBoost 原生 SHAP。")

    booster = final_estimator.get_booster()

    booster_features = getattr(booster, "feature_names", None)

    if booster_features is not None and len(booster_features) == X_for_model.shape[1]:
        used_features = [to_model_feature_name(f) for f in booster_features]
        X_for_model = X_for_model[used_features]
    else:
        used_features = [to_model_feature_name(f) for f in model_features]
        X_for_model = X_for_model[used_features]

    dmatrix = xgb.DMatrix(
        X_for_model.astype(float),
        feature_names=used_features
    )

    contribs = booster.predict(dmatrix, pred_contribs=True)

    # contribs 最后一列为 base value
    shap_values = contribs[0, :-1]
    base_value = contribs[0, -1]

    display_names = [to_display_feature_name(f) for f in used_features]
    data_values = X_for_model.iloc[0].values.astype(float)

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=data_values,
        feature_names=display_names
    )

    return explanation


# ==========================================
# 9. 加载模型
# ==========================================

model, feature_names, display_feature_names = load_model_and_features()


# ==========================================
# 10. 页面标题
# ==========================================

st.title("🏥 基于心磁成像技术的肺动脉高压患病及临床恶化风险计算器")
st.markdown("---")


# ==========================================
# 11. 输入区域
# ==========================================

six_mwt = None
who_fc = None
ntprobnp = None
rt_gradient_ratio = None
input_df = None
rt_angle_diff_from_ph_inputs = 0.0

if model is not None and feature_names is not None and display_feature_names is not None:
    st.subheader("📋 心磁特征参数录入")
    st.markdown("请在下方输入心磁特征参数值，然后点击下方的“预测”按钮。")

    if SHAP_AVAILABLE:
        do_shap = st.checkbox(
            "显示 SHAP 解释",
            value=False
        )
    else:
        do_shap = False
        st.info("当前环境未安装 shap，已自动关闭 SHAP 解释功能。")

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

            model_feature = to_model_feature_name(feature_names[idx])
            display_feature = to_display_feature_name(model_feature)

            with row_cols[c]:
                input_data[model_feature] = st.number_input(
                    label=display_feature,
                    value=0.0,
                    step=1.0,
                    format="%g",
                    key=f"ph_input_{idx}_{model_feature}"
                )

    feature_names_for_model = [to_model_feature_name(f) for f in feature_names]
    input_df = pd.DataFrame([input_data], columns=feature_names_for_model).astype(float)

    # 预后模型中的 R峰-T峰两极角度差值直接引用上方心磁特征输入值
    rt_angle_diff_from_ph_inputs = float(input_data.get("R峰-T峰两极角度差值", 0.0))

    st.markdown("---")
    st.markdown("#### ✨ 肺动脉高压患者临床参数")
    st.caption(
        "预后评估仅在肺动脉高压检测结果为高风险时展示。"
            )

    prog_cols = st.columns(4)

    with prog_cols[0]:
        six_mwt = st.number_input(
            "6分钟步行距离 (m)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%g"
        )

    with prog_cols[1]:
        who_fc = st.selectbox(
            "WHO 心功能分级",
            options=[1, 2, 3, 4],
            index=0
        )

    with prog_cols[2]:
        ntprobnp = st.number_input(
            "NT-proBNP (pg/mL)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%g"
        )

    with prog_cols[3]:
        rt_gradient_ratio = st.number_input(
            "R峰-T峰磁感应强度梯度比值",
            value=0.0,
            step=0.1,
            format="%g"
        )
   
    predict_clicked = st.button("🔍 预测", use_container_width=True)

else:
    do_shap = False
    predict_clicked = False


# ==========================================
# 12. 预测与结果展示
# ==========================================

if predict_clicked and (model is not None) and (input_df is not None):

    # A. PH 检测预测
    with st.spinner("正在计算肺动脉高压预测风险，请稍候..."):
        try:
            X_for_model = prepare_input_for_model(input_df, feature_names)
            probability = predict_probability_safely(model, X_for_model)

        except Exception as e:
            st.error(
                "模型预测失败，请检查输入特征名称、顺序和模型文件是否一致：\n\n"
                f"{e}"
            )
            st.stop()

    # B. SHAP 计算
    final_explanation = None
    shap_ready = False

    if do_shap and SHAP_AVAILABLE:
        with st.spinner("正在计算 SHAP 解释值（可能较慢）..."):
            try:
                final_explanation = compute_xgb_native_shap_explanation(
                    model_obj=model,
                    X_named=input_df,
                    model_features=feature_names
                )
                shap_ready = True

            except Exception as e:
                st.error(f"SHAP 计算模块出错：{str(e)}")
                final_explanation = None
                shap_ready = False

    # C. 结果展示
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📊 肺动脉高压检测结果")

        if probability >= PH_DETECTION_THRESHOLD:
            color = "#dc3545"
            risk_label = "肺动脉高压高风险"
            icon = "⚠️"
            advice_box = "warning"
            advice_text = (
                "模型评估结果为 **高风险**，提示患者当前患肺动脉高压的可能性较高。\n\n"
                "**建议：** 建议进一步完善超声心动图、右心导管检查及相关病因学评估，"
                "并结合临床情况进行综合判断。"
            )
        else:
            color = "#28a745"
            risk_label = "肺动脉高压低风险"
            icon = "✅"
            advice_box = "success"
            advice_text = (
                "模型评估结果为 **低风险**，提示患者当前患肺动脉高压的可能性较低。\n\n"
                "**建议：** 可结合临床症状、体征及其他检查结果继续观察和随访。"
            )

        st.markdown(
            f"""
            <div class="report-box" style="text-align: center; border-left: 5px solid {color};">
                <h2 style="color: {color}; font-size: 36px; margin: 0;">
                    {icon} {risk_label}
                </h2>
                <p style="color: gray; font-size: 13px;">
                    本结果仅供科研与辅助决策参考，不能替代临床医生判断。
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

        # 仅在 PH 高风险时展示预后评估
        if probability >= PH_DETECTION_THRESHOLD:
            st.markdown("---")
            st.markdown("### 📈 临床恶化预后评估")

            try:
                combined_xbeta, baseline_xbeta = compute_combined_xbeta(
                    six_mwt=six_mwt or 0.0,
                    who_fc=int(who_fc) if who_fc is not None else 1,
                    ntprobnp=ntprobnp or 0.0,
                    rt_angle_diff=rt_angle_diff_from_ph_inputs,
                    rt_gradient_ratio=rt_gradient_ratio or 0.0
                )

                if combined_xbeta >= PROGNOSIS_THRESHOLD:
                    prog_label = "临床恶化高风险"
                    prog_color = "#dc3545"
                    prog_icon = "⚠️"
                    prog_box_type = "warning"
                else:
                    prog_label = "临床恶化低风险"
                    prog_color = "#28a745"
                    prog_icon = "✅"
                    prog_box_type = "success"

                st.markdown(
                    f"""
                    <div class="report-box" style="border-left: 5px solid {prog_color};">
                        <h3 style="color:{prog_color}; margin:0;">
                            {prog_icon} {prog_label}
                        </h3>                       
                        </p>
                        <p style="color: gray; font-size: 13px;">
                            本结果仅供科研与辅助决策参考，不能替代临床医生判断。
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("#### 📌 预后决策建议")

                if prog_box_type == "warning":
                    st.warning(
                        "模型提示患者存在 **临床恶化高风险**。\n\n"
                        "建议在肺动脉高压专科医生评估下：\n"
                        "- 密切随访临床症状、体征及 WHO 心功能分级；\n"
                        "- 定期监测 6MWT、NT-proBNP、超声心动图及心磁成像指标；\n"
                        "- 根据病情考虑优化或调整治疗方案；\n"
                        "- 必要时缩短随访间隔。\n\n"
                        "以上内容仅供科研与辅助决策参考。"
                    )
                else:
                    st.success(
                        "模型提示患者目前为 **临床恶化低风险**。\n\n"
                        "建议：\n"
                        "- 继续现有治疗和管理方案；\n"
                        "- 按计划定期随访，复查 6MWT、NT-proBNP 和 WHO 心功能分级；\n"
                        "- 如出现气促加重、晕厥、活动耐量下降等情况，应及时就诊并重新评估。\n\n"
                        "以上内容仅供科研与辅助决策参考。"
                    )

            except Exception as e:
                st.error(f"预后评估计算失败，请检查输入参数：{e}")

        else:
            st.markdown("---")
            st.info("当前肺动脉高压检测结果为 **低风险**，暂不进行临床恶化预后评估。")

    with col2:
        st.markdown("### 🔍 SHAP 可解释性分析")
        st.markdown("下图展示各特征对预测结果的贡献。红色通常表示增加预测风险，蓝色通常表示降低预测风险。")

        if not SHAP_AVAILABLE:
            st.info("当前环境未安装 shap，无法显示 SHAP 解释。")
        elif not do_shap:
            st.info("你已关闭 SHAP 解释。需要查看时，请勾选上方“显示 SHAP 解释”。")
        else:
            if shap_ready and (final_explanation is not None):
                try:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.sca(ax)

                    shap.plots.waterfall(
                        final_explanation,
                        show=False,
                        max_display=13
                    )

                    fix_shap_minus_signs(ax)
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close(fig)

                except Exception as plot_err:
                    st.error(f"SHAP 绘图失败：{plot_err}")
            else:
                st.warning("无法生成 SHAP 图，请检查输入数据或模型结构。")

else:
    st.info("👉 请在上方输入患者参数后，点击“预测”按钮。")
