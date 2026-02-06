import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="EC保育治疗综合疗效预测系统 (AI-Driven)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 样式美化 (CSS) ---
st.markdown("""
    <style>
    .main-title {font-size: 2.5rem; color: #2c3e50; text-align: center; font-weight: 700; margin-bottom: 10px;}
    .sub-title {text-align: center; color: #7f8c8d; margin-bottom: 30px;}
    .feature-card {background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px; border-top: 4px solid #3498db;}
    .card-header {font-size: 1.2rem; font-weight: 600; color: #2980b9; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;}
    .result-box {background-color: white; border-radius: 12px; padding: 15px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 10px;}
    .result-value {font-size: 2.2rem; font-weight: 800; color: #2c3e50;}
    .result-label {font-size: 0.9rem; color: #7f8c8d;}
    </style>
""", unsafe_allow_html=True)

# --- 3. 模型加载逻辑 ---

MODEL_CONFIG = {
    "6-Month": {
        "model": "best_lr_model.pkl",
        "scaler": "scaler.pkl",
        "features": "feature_names.pkl",
        "color": "#f1c40f", # 黄色
        "title": "📅 6个月缓解率"
    },
    "12-Month": {
        "model": "model_12m.pkl",
        "scaler": "scaler_12m.pkl",
        "features": "features_12m.pkl",
        "color": "#3498db", # 蓝色
        "title": "📅 12个月缓解率"
    },
    "Total": {
        "model": "model_total.pkl",
        "scaler": "scaler_total.pkl",
        "features": "features_total.pkl",
        "color": "#2ecc71", # 绿色
        "title": "📈 总缓解率"
    }
}

@st.cache_resource
def load_all_models():
    """一次性加载所有模型"""
    loaded_data = {}
    status_log = []
    
    for key, config in MODEL_CONFIG.items():
        try:
            m = joblib.load(config["model"])
            s = joblib.load(config["scaler"])
            f = joblib.load(config["features"])
            loaded_data[key] = {"model": m, "scaler": s, "features": f}
            status_log.append(f"✅ {key} 模型加载成功")
        except FileNotFoundError:
            status_log.append(f"❌ {key} 文件缺失 (请检查目录下是否有 {config['model']})")
        except Exception as e:
            status_log.append(f"❌ {key} 加载出错: {e}")
            
    return loaded_data, status_log

models_data, load_logs = load_all_models()

with st.sidebar:
    st.title("⚙️ 系统状态")
    for log in load_logs:
        if "✅" in log:
            st.success(log)
        else:
            st.error(log)
    st.markdown("---")
    st.info("本系统基于 Logistic Regression 算法构建。")

if not models_data:
    st.error("未检测到任何模型文件，请先运行训练脚本！")
    st.stop()

# --- 4. 特征整合 ---

all_needed_features = set()
for key in models_data:
    for feat in models_data[key]["features"]:
        all_needed_features.add(feat)

sorted_features = sorted(list(all_needed_features))

# 定义临床特征 (包含 G2)
CLINICAL_LIST = [
    'BMI', 'PCOS', 'IR', 'HE4', 'G2', 
    'Myometrialinvasion', 'Myometria', 
    'maxtumorsize', 'maxtumor'
]

clinical_feats_found = []
radiomics_feats_found = []

for feat in sorted_features:
    if feat in CLINICAL_LIST:
        clinical_feats_found.append(feat)
    else:
        radiomics_feats_found.append(feat)

# 保持顺序
clinical_feats_sorted = [f for f in CLINICAL_LIST if f in clinical_feats_found]

# --- 5. 主界面构建 ---

st.markdown('<div class="main-title">EC 保育治疗疗效预测系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">集成多模态数据的真实AI预测模型</div>', unsafe_allow_html=True)

user_inputs = {}

col_clin, col_rad = st.columns([1, 2]) 

# === 左侧：临床特征 ===
with col_clin:
    st.markdown('<div class="feature-card"><div class="card-header">📋 临床特征 (Clinical)</div></div>', unsafe_allow_html=True)
    
    with st.container():
        if not clinical_feats_sorted:
            st.warning("加载的模型中未包含指定的临床特征。")
            
        for feat in clinical_feats_sorted:
            
            # --- 1. 分化程度 (G2) 特殊处理 ---
            if feat == 'G2':
                val = st.selectbox(
                    "分化程度 (Histological Grade)",
                    options=[0, 1],
                    # 0 对应 G1, 1 对应 G2
                    format_func=lambda x: "G1 " if x == 0 else "G2 ", 
                    key=f"in_{feat}"
                )

            # --- 2. 其他二分类变量 ---
            elif feat in ['PCOS', 'IR', 'Myometrialinvasion', 'Myometria']:
                # 标签映射
                label_map = {
                    'PCOS': "多囊卵巢 (PCOS)",
                    'IR': "胰岛素抵抗 (IR)",
                    'Myometrialinvasion': "肌层浸润 (Myometrial Invasion)",
                    'Myometria': "肌层浸润 (Myometria)"
                }
                display_label = label_map.get(feat, feat)
                
                val = st.selectbox(
                    f"{display_label}", 
                    options=[0, 1], 
                    format_func=lambda x: "有/Yes (1)" if x==1 else "无/No (0)",
                    key=f"in_{feat}"
                )
                
            # --- 3. 连续数值变量 ---
            elif feat == 'BMI':
                val = st.number_input(f"{feat}", value=22.0, min_value=10.0, max_value=50.0, step=0.1, key=f"in_{feat}")
            elif feat == 'HE4':
                val = st.number_input(f"{feat} (pmol/L)", value=50.0, min_value=0.0, step=1.0, key=f"in_{feat}")
            elif feat in ['maxtumorsize', 'maxtumor']:
                val = st.number_input(f"最大肿瘤直径 ({feat}, cm)", value=2.0, min_value=0.0, step=0.1, key=f"in_{feat}")
            else:
                val = st.number_input(f"{feat}", value=0.0, key=f"in_{feat}")
            
            user_inputs[feat] = val

# === 右侧：影像组学特征 ===
with col_rad:
    st.markdown('<div class="feature-card" style="border-top: 4px solid #9b59b6;"><div class="card-header">☢️ 影像组学特征 (Radiomics)</div></div>', unsafe_allow_html=True)
    
    if not radiomics_feats_found:
        st.info("模型似乎仅依赖临床特征，未检测到影像组学特征。")
    else:
        st.info(f"系统自动检测到 {len(radiomics_feats_found)} 个影像组学特征。")
        with st.expander("展开/折叠 影像特征录入面板", expanded=True):
            r_cols = st.columns(3)
            for i, feat in enumerate(radiomics_feats_found):
                short_name = feat.split('_')[-1]
                with r_cols[i % 3]:
                    user_inputs[feat] = st.number_input(
                        label=short_name,
                        value=0.0000, 
                        step=0.0001,
                        format="%.4f",
                        help=f"完整特征名: {feat}",
                        key=f"in_{feat}"
                    )

# --- 6. 预测逻辑 ---
st.markdown("---")

if st.button("🚀 开始综合预测 (Run Prediction)", type="primary", use_container_width=True):
    
    results_cols = st.columns(3)
    
    for idx, (model_key, config) in enumerate(MODEL_CONFIG.items()):
        
        if model_key not in models_data:
            continue
            
        model_info = models_data[model_key]
        model = model_info["model"]
        scaler = model_info["scaler"]
        required_features = model_info["features"]
        
        try:
            input_vector = [user_inputs[f] for f in required_features]
            input_df = pd.DataFrame([input_vector], columns=required_features)
            
            input_scaled = scaler.transform(input_df)
            prob = model.predict_proba(input_scaled)[0][1]
            
            with results_cols[idx]:
                title = config["title"]
                color = config["color"]
                prob_pct = prob * 100
                res_color = "#27ae60" if prob > 0.5 else "#e67e22"
                res_text = "预测: 缓解 (Response)" if prob > 0.5 else "预测: 未缓解 (No Response)"
                
                st.markdown(f"""
                <div class="result-box" style="border-top: 5px solid {color};">
                    <div style="font-weight:bold; color:{color}; margin-bottom:5px;">{title}</div>
                    <div class="result-value" style="color: {res_color}">{prob_pct:.2f}%</div>
                    <div class="result-label">{res_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
        except KeyError as e:
            st.error(f"{model_key} 预测失败: 缺少特征 {e}")
        except Exception as e:
            st.error(f"{model_key} 运行出错: {e}")

    st.success("✅ 计算完成！结果基于您本地训练的真实 Logistic Regression 模型。")