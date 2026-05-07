# ==============================================================================
# app.py — Professional Streamlit Dashboard
# Plant Disease Detection & Smart Agricultural Decision Support
# ==============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import glob

from src.config.settings import Config

# ── Page Config ──
st.set_page_config(page_title="AGRI-X AI", page_icon="🌿", layout="wide")

# ── Initialize Config ──
@st.cache_resource
def get_config():
    return Config()

config = get_config()

# ── Load Predictor ──
@st.cache_resource
def get_predictor():
    from src.api.predictor import PlantDiseasePredictor
    return PlantDiseasePredictor(config)

predictor = get_predictor()

# ══════════════════════════════════════════════════════════════════════════════
# CSS STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, section.main {
    background-color: #f4f9f1 !important; color: #1a2e1c !important;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #eaf6e4 0%, #f4f9f1 45%, #e8f4fe 100%) !important;
}
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important; visibility: hidden !important;
}
p, span, div, label, li, [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    color: #1a2e1c !important; font-family: 'Inter', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e3018 0%, #1a5c30 100%) !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] div,
[data-testid="stSidebar"] li {
    color: #d4f0db !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }

/* Cards */
.card {
    background: #ffffff; border: 1px solid #cce8d5; border-radius: 20px;
    padding: 24px 28px; box-shadow: 0 2px 18px rgba(46,158,86,0.09);
    margin-bottom: 16px; animation: fadeUp 0.5s ease both;
}
.lbl {
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: 0.2em !important; text-transform: uppercase !important;
    color: #1f8c47 !important; margin-bottom: 7px !important;
}
.big-title {
    font-family: 'Lora', serif !important; font-size: 1.45rem !important;
    font-weight: 700 !important; color: #0e3018 !important;
}
.conf-num {
    font-size: 2.6rem !important; font-weight: 700 !important;
    color: #1f8c47 !important; line-height: 1 !important; margin: 6px 0 !important;
}
.bar-bg { background: #dff0e6; border-radius: 100px; height: 10px; overflow: hidden; margin: 8px 0 5px; }
.bar-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #56c97a, #1f8c47); }

/* Recommendations */
.rec { border-radius: 14px; padding: 18px 20px; margin-top: 12px; }
.rec.healthy { background: #edfaf3; border-left: 4px solid #1f8c47; }
.rec.warning { background: #fff8ec; border-left: 4px solid #f0a500; }
.rec.danger  { background: #fff3f3; border-left: 4px solid #e05454; }
.rec.info    { background: #eff6ff; border-left: 4px solid #3d86d8; }
.rec-head { font-family:'Lora',serif!important; font-size:1.05rem!important; font-weight:700!important; color:#0e3018!important; }
.rec-body { font-size:13.5px!important; color:#2d4e35!important; line-height:1.65!important; }

/* Severity badges */
.sev-mild { background:#fff3cd; color:#856404; padding:4px 12px; border-radius:8px; font-weight:600; font-size:13px; }
.sev-moderate { background:#ffe0cc; color:#c45100; padding:4px 12px; border-radius:8px; font-weight:600; font-size:13px; }
.sev-severe { background:#f8d7da; color:#721c24; padding:4px 12px; border-radius:8px; font-weight:600; font-size:13px; }
.sev-healthy { background:#d4edda; color:#155724; padding:4px 12px; border-radius:8px; font-weight:600; font-size:13px; }

/* Image */
[data-testid="stImage"] img { border-radius: 16px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.10) !important; }

/* Hero */
.hero-wrap { text-align: center; padding: 1.5rem 1rem 0.8rem; }
.hero-title {
    font-family: 'Lora', serif !important; font-size: clamp(2rem, 5vw, 3rem) !important;
    font-weight: 700 !important; color: #0e3018 !important; line-height: 1.18 !important;
}
.hero-title .acc { color: #1f8c47 !important; }
.hero-sub { font-size: 14px !important; color: #2e5c38 !important; max-width: 550px; margin: 0 auto; line-height: 1.6 !important; }
.divider { display:flex; align-items:center; gap:10px; margin:1.2rem 0; }
.divider::before, .divider::after { content:''; flex:1; height:1px; background:linear-gradient(to right,transparent,#6dbb88,transparent); opacity:0.5; }

@keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 AGRI-X AI")
    st.markdown("*Plant Disease Detection Framework*")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🔍 Prediction",
        "📊 Model Comparison",
        "📈 Training History",
        "🎯 Evaluation",
        "🔬 Research Analysis",
        "ℹ️ About",
    ], label_visibility="collapsed")

    st.markdown("---")

    # Model selection
    available_models = predictor.get_available_models()
    if not available_models:
        available_models = config.model.model_names
        st.warning("⚠️ No trained models found. Run `python train.py` first.")

    selected_model = st.selectbox("🤖 Select Model", available_models)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;font-size:11px;color:#8fbb9e;margin-top:20px;'>"
        "Powered by TensorFlow<br>© AGRI-X AI Research</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Prediction":
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">AGRI-X <span class="acc">AI</span></div>
        <div class="hero-sub">
            Upload a leaf photograph to detect plant diseases instantly,
            view Grad-CAM explanations, and receive treatment guidance.
        </div>
    </div>
    <div class="divider">🍃</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        try:
            result = predictor.predict(image, selected_model, top_k=3)

            # ── Image + Info ──
            col_img, col_info = st.columns([1, 1], gap="medium")

            with col_img:
                st.markdown('<div class="card" style="padding:14px;">', unsafe_allow_html=True)
                st.image(image, use_container_width=True, caption="Uploaded Image")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_info:
                conf = result["confidence"] * 100
                sev_class = "sev-healthy" if result["is_healthy"] else (
                    "sev-mild" if result.get("severity", {}).get("severity") == "Mild" else
                    "sev-moderate" if result.get("severity", {}).get("severity") == "Moderate" else
                    "sev-severe"
                )
                sev_text = "Healthy" if result["is_healthy"] else result.get("severity", {}).get("severity", "N/A")

                st.markdown(f"""
                <div class="card" style="min-height:220px;">
                    <div class="lbl">Detection Result</div>
                    <div class="big-title">{result['display_name']}</div>
                    <span class="{sev_class}">{sev_text}</span>
                    <div style="margin-top:18px;">
                        <div class="lbl">Confidence Score</div>
                        <div class="conf-num">{conf:.1f}%</div>
                        <div class="bar-bg"><div class="bar-fill" style="width:{conf:.1f}%;"></div></div>
                    </div>
                    <div style="margin-top:12px;font-size:12px;color:#5a8468;">
                        Model: {result['model_name']} | Active Area: {result['activation_percentage']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Top 3 Predictions ──
            st.markdown('<div class="card"><div class="lbl">🎯 Top 3 Predictions</div>', unsafe_allow_html=True)
            for i, pred in enumerate(result["top_k_predictions"]):
                pct = pred["confidence"] * 100
                st.markdown(
                    f"**{i+1}. {pred['display_name']}** — `{pct:.2f}%`"
                )
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Grad-CAM ──
            st.markdown('<div class="card"><div class="lbl">🔥 Grad-CAM Explainability</div>', unsafe_allow_html=True)
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                st.image(image.resize((224, 224)), caption="Original", use_container_width=True)
            with gc2:
                st.image(result["gradcam_heatmap"], caption="Heatmap", use_container_width=True, clamp=True)
            with gc3:
                st.image(result["gradcam_overlay"], caption="Overlay", use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Recommendations ──
            rec = result["recommendation"]
            urgency_map = {"low": "healthy", "medium": "info", "high": "warning", "critical": "danger"}
            rec_cls = urgency_map.get(rec.urgency, "info")
            icon_map = {"low": "✅", "medium": "ℹ️", "high": "⚠️", "critical": "🚨"}
            rec_icon = icon_map.get(rec.urgency, "🔬")

            st.markdown(f"""
            <div class="card">
                <div class="lbl">💡 Smart Recommendations</div>
                <div class="rec {rec_cls}">
                    <div style="font-size:22px;margin-bottom:6px;">{rec_icon}</div>
                    <div class="rec-head">{rec.disease_name} — {rec.severity}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""
                <div class="card">
                    <div class="lbl">🧪 Pesticide</div>
                    <div class="rec-body">{rec.pesticide}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card">
                    <div class="lbl">💧 Irrigation</div>
                    <div class="rec-body">{rec.irrigation}</div>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="card">
                    <div class="lbl">🍄 Fungicide</div>
                    <div class="rec-body">{rec.fungicide}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card">
                    <div class="lbl">🌱 Fertilizer</div>
                    <div class="rec-body">{rec.fertilizer}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card">
                <div class="lbl">📝 Additional Notes</div>
                <div class="rec-body">{rec.additional_notes}</div>
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError:
            st.error(f"⚠️ Model '{selected_model}' not found. Please run `python train.py` first.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;">
            <div style="font-size:56px;margin-bottom:1rem;">🌾</div>
            <div class="big-title">Awaiting your leaf image</div>
            <p style="font-size:13px;color:#5a8468;">Supports JPG, JPEG & PNG · Max 200 MB</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown('<div class="hero-wrap"><div class="hero-title">📊 Model Comparison</div></div>', unsafe_allow_html=True)

    csv_path = os.path.join(config.paths.metrics_dir, "model_comparison.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Show comparison chart
        chart_path = os.path.join(config.paths.graphs_dir, "model_comparison_metrics.png")
        if os.path.exists(chart_path):
            st.image(chart_path, caption="Performance Comparison", use_container_width=True)

        inf_path = os.path.join(config.paths.graphs_dir, "inference_time_comparison.png")
        size_path = os.path.join(config.paths.graphs_dir, "model_size_comparison.png")
        c1, c2 = st.columns(2)
        if os.path.exists(inf_path):
            c1.image(inf_path, caption="Inference Time", use_container_width=True)
        if os.path.exists(size_path):
            c2.image(size_path, caption="Model Size", use_container_width=True)
    else:
        st.info("No comparison data yet. Run `python train.py` to generate.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAINING HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Training History":
    st.markdown('<div class="hero-wrap"><div class="hero-title">📈 Training History</div></div>', unsafe_allow_html=True)

    all_comp = os.path.join(config.paths.graphs_dir, "all_models_training_comparison.png")
    if os.path.exists(all_comp):
        st.image(all_comp, caption="All Models — Training Comparison", use_container_width=True)

    history_plots = glob.glob(os.path.join(config.paths.graphs_dir, "*_training_history.png"))
    if history_plots:
        for plot_path in sorted(history_plots):
            name = os.path.basename(plot_path).replace("_training_history.png", "")
            st.image(plot_path, caption=f"{name} — Training History", use_container_width=True)
    elif not os.path.exists(all_comp):
        st.info("No training history yet. Run `python train.py` to generate.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Evaluation":
    st.markdown('<div class="hero-wrap"><div class="hero-title">🎯 Evaluation Results</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Confusion Matrices", "ROC Curves", "Classification Reports"])

    with tab1:
        cm_plots = glob.glob(os.path.join(config.paths.confusion_matrix_dir, "*.png"))
        if cm_plots:
            for p in sorted(cm_plots):
                name = os.path.basename(p).replace("_confusion_matrix.png", "")
                st.image(p, caption=f"{name}", use_container_width=True)
        else:
            st.info("No confusion matrices yet.")

    with tab2:
        roc_plots = glob.glob(os.path.join(config.paths.roc_curves_dir, "*.png"))
        if roc_plots:
            for p in sorted(roc_plots):
                st.image(p, use_container_width=True)
        else:
            st.info("No ROC curves yet.")

    with tab3:
        reports = glob.glob(os.path.join(config.paths.metrics_dir, "*_classification_report.txt"))
        if reports:
            for r in sorted(reports):
                name = os.path.basename(r).replace("_classification_report.txt", "")
                with st.expander(f"📋 {name}"):
                    st.code(open(r).read())
        else:
            st.info("No classification reports yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESEARCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Research Analysis":
    st.markdown('<div class="hero-wrap"><div class="hero-title">🔬 Research Analysis</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Cross Validation", "Statistical Analysis", "Ablation Study",
        "Compression Study", "Grad-CAM Gallery",
    ])

    with tab1:
        cv_path = os.path.join(config.paths.metrics_dir, "cross_validation_results.csv")
        if os.path.exists(cv_path):
            st.dataframe(pd.read_csv(cv_path), use_container_width=True, hide_index=True)
        else:
            st.info("Run `python evaluate.py` for cross-validation results.")

    with tab2:
        for fname in ["statistical_summary.csv", "paired_comparisons.csv", "model_rankings.csv"]:
            fpath = os.path.join(config.paths.summary_dir, fname)
            if os.path.exists(fpath):
                st.markdown(f"**{fname.replace('.csv','').replace('_',' ').title()}**")
                st.dataframe(pd.read_csv(fpath), use_container_width=True, hide_index=True)

    with tab3:
        abl_path = os.path.join(config.paths.summary_dir, "ablation_study.csv")
        if os.path.exists(abl_path):
            st.dataframe(pd.read_csv(abl_path), use_container_width=True, hide_index=True)
        else:
            st.info("Run `python evaluate.py` for ablation results.")

    with tab4:
        comp_path = os.path.join(config.paths.summary_dir, "compression_study.csv")
        if os.path.exists(comp_path):
            st.dataframe(pd.read_csv(comp_path), use_container_width=True, hide_index=True)
        else:
            st.info("Run `python evaluate.py` for compression analysis.")

    with tab5:
        gc_images = glob.glob(os.path.join(config.paths.gradcam_dir, "*.png"))
        if gc_images:
            for img_path in sorted(gc_images)[:12]:
                st.image(img_path, use_container_width=True)
        else:
            st.info("No Grad-CAM images yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">About AGRI-X <span class="acc">AI</span></div>
        <div class="hero-sub">An Explainable Multi-Model Deep Learning Framework for
        Plant Disease Detection and Smart Agricultural Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🏗️ Architecture
    - **Models**: MobileNetV2, ResNet50, EfficientNetB0, DenseNet121, ViT
    - **Transfer Learning**: ImageNet pretrained weights with two-phase fine-tuning
    - **Explainability**: Grad-CAM heatmap visualization
    - **Dataset**: PlantVillage (Tomato, Potato, Bell Pepper)

    ### 📊 Research Features
    - K-Fold Cross Validation
    - Statistical Analysis with Confidence Intervals
    - Ablation Study
    - Model Compression Analysis
    - Disease Severity Estimation
    - Intelligent Recommendation Engine

    ### 🚀 Quick Start
    ```bash
    # Train all models
    python train.py

    # Run advanced evaluation
    python evaluate.py

    # Launch dashboard
    streamlit run app.py
    ```

    ### 📚 Tech Stack
    TensorFlow · Keras · Scikit-learn · Streamlit · Matplotlib · Seaborn
    """)

# ── Footer ──
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1rem;border-top:1px solid #bde0cb;font-size:12px;color:#4a7a58;">
    Powered by <b style="color:#1f8c47;">TensorFlow</b> &amp; <b style="color:#1f8c47;">Streamlit</b>
    &nbsp;·&nbsp; <b style="color:#1f8c47;">AGRI-X AI</b> Research Framework
</div>
""", unsafe_allow_html=True)