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
# 1. 核心配置与字体终极修复 (容灾版)
# ==========================================
st.set_page_config(
    page_title="肺动脉高压风险预测系统",
    page_icon="🏥",
    layout="wide"
)

def configure_font_robust():
    """
    多线路下载 SimHei 字体，并强制配置 Matplotlib。
    解决策略：SimHei (主字体) + unicode_minus=False (解决负号)。
    """
    font_filename = "SimHei.ttf"
    
    # 定义多个下载源 (容灾备份)
    # 1. jsDelivr CDN (高速，推荐)
    # 2. GitHub Raw (备用)
    # 3. Google Fonts Noto Sans SC (最后防线，虽不是SimHei但也能用)
    download_urls = [
        "https://cdn.jsdelivr.net/gh/StellarCN/scp_zh@master/fonts/SimHei.ttf",
        "https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf",
        "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
    ]

    # 1. 下载字体文件 (如果本地没有)
    if not os.path.exists(font_filename):
        with st.spinner("正在初始化中文字体环境 (SimHei)..."):
            download_success = False
            for url in download_urls:
                try:
                    # 设置较长的超时时间
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        with open(font_filename, "wb") as f:
                            f.write(response.content)
                        download_success = True
                        break # 下载成功，跳出循环
                except Exception as e:
                    continue # 试下一个链接
            
            if not download_success:
                st.error("所有字体下载源均连接失败，中文可能无法正常显示。")

    # 2. 强制向 Matplotlib 注册该字体文件
    if os.path.exists(font_filename):
        try:
            # 清除旧缓存并添加新字体
            fm.fontManager.addfont(font_filename)
            
            # 获取字体文件的真实名称
            prop = fm.FontProperties(fname=font_filename)
            font_name = prop.get_name() # 通常是 'SimHei'
            
            # 3. 全局样式设置
            # 强制 SimHei 为第一优先级
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name] 
            
            # 【关键】SimHei 不支持数学减号，必须设为 False 用短横线代替
            plt.rcParams['axes.unicode_minus'] = False 
            
            # 强制 SHAP 内部也更新
            matplotlib.rc('font', family=font_name)
            matplotlib.rc('axes', unicode_minus=False)
            
            return True
        except Exception as e:
            st.error(f"字体配置出错: {e}")
            return False
    return False

# 执行字体配置
is_font_ready = configure_font_robust()

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
    else:
        st.sidebar.caption("⚠️ 字体状态：未加载 (可能乱码)")

# ==========================================
# 5. 主界面：预测与解释逻辑
# ==========================================
st.title("🏥 基于心磁成像装置的肺动脉高压检测计算器")
st.markdown("基于机器学习随机森林算法构建 | 仅供科研参考")
st.markdown("---")

if st.sidebar.button("🔍 开始预测风险"):
    if model and feature_names:
        with st.spinner('正在计算模型预测概率与 SHAP 解释值，请稍候...'):
            
            # A. 计算概率
            try:
                probability = model.predict_proba(input_df)[0, 1]
            except:
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
                    explainer = shap.TreeExplainer(final_estimator, data=processed_data_df, model_output="probability")
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

                    # 4. 构建解释对象 (移花接木：数学用归一化值，显示用原始值)
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
                        # 绘制瀑布图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # 【最后一道防线】绘图前再次强制确认参数
                        plt.rcParams['axes.unicode_minus'] = False
                        plt.rcParams['font.sans-serif'] = ['SimHei']
                        
                        shap.plots.waterfall(final_explanation, show=False, max_display=14)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as plot_err:
                         st.error(f"绘图失败。调试信息: {plot_err}")
                else:
                    st.warning("无法生成 SHAP 图，请检查输入数据或模型结构。")
            
            st.markdown("---")
            st.caption(f"**说明：** 本工具采用约登指数 (Youden Index = {youden_index}) 确定的最佳截断值 {optimal_threshold/100:.5f} 进行风险分层。结果仅供科研参考。")
    else:
        st.error("系统错误：模型未加载。")
else:
    st.info("👈 请在左侧侧边栏输入患者的临床参数，然后点击“开始预测风险”按钮。")
