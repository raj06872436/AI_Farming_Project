"""
AGRI-X AI — Agricultural Intelligence Platform
Explainable Multi-Model Deep Learning Framework for Plant Disease Detection
and Smart Agricultural Decision Support — Streamlit Dashboard (Phase 2)
"""
import os, sys, json, time
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import theme  # registers the light "agrix" Plotly template
import glob

from app_utils import (
    CUSTOM_CSS, BASE_DIR, SAVED_MODELS_DIR, GRAPHS_DIR, GRADCAM_DIR,
    RESEARCH_FIGURES, CONFUSION_DIR, ROC_DIR, METRICS_DIR, METRICS_CSV,
    load_class_names, load_registry, load_dataset_summary, load_metrics_csv,
    find_model_path, get_available_models, find_training_graph,
    find_gradcam_images, get_all_graph_images,
    confidence_bar_html, severity_badge_html, metric_card_html,
    recommendation_card_html, feature_card_html, stat_card_html,
    weather_strip_html, how_it_works_html, badge_pill_html,
    chip_html, page_header_html, section_head_html, kv_html, meter_html, context_bar_html,
)

# Phase-2 service modules
from location_service import detect_location, render_location_card, render_location_override_form, render_location_details
from weather_service import (
    fetch_weather, get_current_weather, get_hourly_forecast,
    get_daily_forecast, render_current_weather_card,
    render_hourly_forecast, render_daily_forecast,
)
from risk_engine import calculate_risk, render_risk_card
from treatment_engine import get_treatment, render_treatment_cards
from crop_advisor import render_crop_advisor_page, _get_current_season, calculate_soil_quality
from yield_predictor import render_yield_page
from insights_engine import render_insights_dashboard
from farmer_chatbot import render_chatbot
from report_generator import render_report_section

# ─── Page Config ───
st.set_page_config(
    page_title="AGRI-X AI — Agricultural Intelligence Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Load Global Data ───
CLASS_NAMES = load_class_names()
REGISTRY = load_registry()
DATASET_SUMMARY = load_dataset_summary()
METRICS_DF = load_metrics_csv()

GRADCAM_LAYERS = {
    "MobileNetV2": "out_relu",
    "ResNet50": "conv5_block3_out",
    "EfficientNetB0": "top_activation",
    "DenseNet121": "relu",
}

# Import the new modular disease heatmap pipeline
from disease_heatmap import run_disease_heatmap_pipeline
from lesion_segmentation import analyze_leaf
import lesionnet_inference as lni
import research_views as rv

# ─── Location & Weather (session-cached) ───
if "location" not in st.session_state:
    st.session_state["location"] = None

# Track previous location to detect changes and clear stale results
_prev_location_key = None
if st.session_state.get("location"):
    _prev = st.session_state["location"]
    _prev_location_key = f"{_prev.get('latitude',0):.4f},{_prev.get('longitude',0):.4f}"

location = detect_location()

# Clear stale crop/yield results if location changed
_new_location_key = f"{location['latitude']:.4f},{location['longitude']:.4f}" if location else None
if _prev_location_key and _new_location_key and _prev_location_key != _new_location_key:
    for key in ["crop_results", "crop_params", "yield_result", "selected_yield_crop"]:
        st.session_state.pop(key, None)

if "weather_raw" not in st.session_state:
    st.session_state["weather_raw"] = None
    st.session_state["weather_current"] = {}

# Re-fetch weather whenever the farm location changes (GPS, city search, coordinates),
# not only on first load — otherwise a new location keeps showing the old weather.
if location and (
    st.session_state["weather_raw"] is None
    or st.session_state.get("weather_location_key") != _new_location_key
):
    raw = fetch_weather(location["latitude"], location["longitude"])
    st.session_state["weather_raw"] = raw
    st.session_state["weather_current"] = get_current_weather(raw) if raw else {}
    st.session_state["weather_location_key"] = _new_location_key

weather_raw = st.session_state["weather_raw"]
weather_current = st.session_state["weather_current"]

# ─── Model Loading (cached) ───
@st.cache_resource
def cached_load_model(path):
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def cached_load_lesionnet(path):
    return lni.load(path)

def predict_image(model, image, class_names):
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    conf = float(np.max(preds[0]))
    top5 = np.argsort(preds[0])[::-1][:5]
    top5_list = [{"class": class_names[i], "confidence": float(preds[0][i])} for i in top5 if i < len(class_names)]
    return idx, conf, top5_list, arr, preds

def estimate_severity(confidence, act_pct):
    """
    Severity from the share of the leaf that is diseased (a common field scale):
    under 10 % mild, 10–25 % moderate, over 25 % severe.
    """
    score = min(act_pct / 100.0, 1.0)
    if act_pct < 10:
        return "Mild", score, "#B45309", "Early-stage infection on a small part of the leaf."
    elif act_pct < 25:
        return "Moderate", score, "#C2410C", "A good part of the leaf is affected. Treat soon."
    else:
        return "Severe", score, "#B91C1C", "A large part of the leaf is damaged. Act immediately."

# ─── Knowledge Base ───
DISEASE_KB = {
    "Pepper__bell___Bacterial_spot": {"disease": "Bacterial Spot", "plant": "Bell Pepper", "pesticide": "Copper-based bactericides at 7-day intervals. Streptomycin sulfate for severe cases.", "fungicide": "Copper oxychloride 50% WP at 2.5g/L. Mancozeb 75% WP preventive.", "irrigation": "Switch to drip irrigation. Avoid overhead watering. Water early morning.", "fertilizer": "Balanced NPK (10-10-10). Increase potassium for disease resistance.", "notes": "Remove infected debris. Practice 2-3 year crop rotation."},
    "Pepper__bell___healthy": {"disease": "Healthy", "plant": "Bell Pepper", "pesticide": "No pesticide needed.", "fungicide": "Optional preventive neem oil biweekly.", "irrigation": "Maintain consistent moisture via drip.", "fertilizer": "Balanced NPK every 2-3 weeks.", "notes": "Plant is healthy. Continue monitoring."},
    "Potato___Early_blight": {"disease": "Early Blight", "plant": "Potato", "pesticide": "Chlorothalonil or mancozeb at first symptoms. Spray every 7-10 days.", "fungicide": "Azoxystrobin 23% SC at 1ml/L.", "irrigation": "Use furrow or drip. Avoid wetting foliage.", "fertilizer": "Adequate nitrogen, apply potash to strengthen cell walls.", "notes": "Remove lower infected leaves. Mulch to prevent soil splash."},
    "Potato___Late_blight": {"disease": "Late Blight", "plant": "Potato", "pesticide": "URGENT: Metalaxyl + mancozeb immediately. Spray every 5-7 days.", "fungicide": "Cymoxanil + mancozeb curative. Dimethomorph for resistant strains.", "irrigation": "STOP overhead irrigation. Drip only. Improve drainage.", "fertilizer": "Reduce nitrogen. Increase phosphorus and potassium.", "notes": "HIGHLY DESTRUCTIVE — act immediately! Destroy infected plants."},
    "Potato___healthy": {"disease": "Healthy", "plant": "Potato", "pesticide": "No treatment needed.", "fungicide": "Optional preventive mancozeb before monsoon.", "irrigation": "1-2 inches/week. Mulch to retain moisture.", "fertilizer": "NPK 12-12-17 at planting.", "notes": "Healthy crop. Continue weekly monitoring."},
    "Tomato_Bacterial_spot": {"disease": "Bacterial Spot", "plant": "Tomato", "pesticide": "Copper hydroxide 77% WP at 2g/L every 7 days.", "fungicide": "Bordeaux mixture 1%. Streptocycline 100ppm for severe cases.", "irrigation": "Drip only. Avoid leaf wetting.", "fertilizer": "Balanced NPK with extra calcium.", "notes": "Highly contagious — isolate infected plants."},
    "Tomato_Early_blight": {"disease": "Early Blight", "plant": "Tomato", "pesticide": "Chlorothalonil 75% WP at 2g/L.", "fungicide": "Mancozeb 75% WP preventive. Azoxystrobin curative.", "irrigation": "Mulch soil. Drip preferred.", "fertilizer": "Avoid excess nitrogen. Apply potash.", "notes": "Prune lower branches for air circulation."},
    "Tomato_Late_blight": {"disease": "Late Blight", "plant": "Tomato", "pesticide": "EMERGENCY: Metalaxyl-mancozeb immediately. Repeat every 5 days.", "fungicide": "Cymoxanil + mancozeb. Mandipropamid for resistance management.", "irrigation": "STOP overhead watering. Drip only.", "fertilizer": "Reduce nitrogen. Maximize potassium.", "notes": "CRITICAL — can destroy entire crop in days!"},
    "Tomato_Leaf_Mold": {"disease": "Leaf Mold", "plant": "Tomato", "pesticide": "Chlorothalonil preventively in humid conditions.", "fungicide": "Mancozeb 2.5g/L or copper oxychloride 3g/L.", "irrigation": "Reduce humidity. Improve ventilation.", "fertilizer": "Balanced NPK. Avoid excess nitrogen.", "notes": "Common in greenhouses. Increase air flow."},
    "Tomato_Septoria_leaf_spot": {"disease": "Septoria Leaf Spot", "plant": "Tomato", "pesticide": "Chlorothalonil or copper-based spray every 7-10 days.", "fungicide": "Mancozeb preventive + azoxystrobin curative.", "irrigation": "Mulch heavily. Drip irrigation.", "fertilizer": "Standard NPK. Extra potassium.", "notes": "Spreads rapidly in wet conditions."},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"disease": "Spider Mites", "plant": "Tomato", "pesticide": "Abamectin 1.8% EC at 0.5ml/L. Biocontrol with predatory mites.", "fungicide": "Not applicable (pest, not fungal).", "irrigation": "Increase humidity — mites thrive in dry conditions.", "fertilizer": "Standard nutrition. Avoid plant stress.", "notes": "Check undersides of leaves. Neem oil as organic alternative."},
    "Tomato__Target_Spot": {"disease": "Target Spot", "plant": "Tomato", "pesticide": "Chlorothalonil 75% WP every 7-10 days.", "fungicide": "Azoxystrobin + difenoconazole combination.", "irrigation": "Improve air circulation. Drip only.", "fertilizer": "Balanced NPK. Extra potassium + calcium.", "notes": "Concentric ring pattern on leaves. Remove infected leaves."},
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {"disease": "Yellow Leaf Curl Virus", "plant": "Tomato", "pesticide": "Control whitefly: Imidacloprid 17.8% SL. Yellow sticky traps.", "fungicide": "Not applicable (viral disease).", "irrigation": "Normal. Use reflective mulch to repel whiteflies.", "fertilizer": "Boost immunity with balanced NPK + micronutrients.", "notes": "VIRAL — no cure! Remove infected plants. Use resistant varieties."},
    "Tomato__Tomato_mosaic_virus": {"disease": "Tomato Mosaic Virus", "plant": "Tomato", "pesticide": "No chemical treatment. Control aphid vectors.", "fungicide": "Not applicable (viral).", "irrigation": "Normal. Avoid handling wet plants.", "fertilizer": "Balanced nutrition for plant vigor.", "notes": "VIRAL — extremely contagious via contact. Disinfect tools."},
    "Tomato_healthy": {"disease": "Healthy", "plant": "Tomato", "pesticide": "No treatment needed.", "fungicide": "Optional preventive copper before rainy season.", "irrigation": "1-1.5 inches/week. Mulch to retain moisture.", "fertilizer": "NPK 10-10-10 every 2 weeks. Calcium during fruiting.", "notes": "Healthy! Continue monitoring. Rotate crops annually."},
}

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
PAGES = [
    "🏠 Home",
    "🔬 Disease Detection",
    "🌾 Crop Advisor",
    "🤖 Farmer Assistant",
    "📊 Research Dashboard",
    "📈 Model Comparison",
    "📋 Reports & About",
]


def go_to(page_name):
    """Button callback: switch the sidebar menu to another page."""
    st.session_state["nav_page"] = page_name


available = get_available_models(REGISTRY)
all_models = list(REGISTRY.keys())

# LesionNet (our model, research/lesion_net) — listed first when its weights are present
LESIONNET_PATH = lni.find_lesionnet_path()
LN_MODELS = lni.available_models()            # LesionNet, LesionNet-Mobile (whichever weights exist)
DETECT_MODELS = LN_MODELS + available


def model_meta(m):
    if m in lni.METAS:
        return lni.METAS[m]
    return REGISTRY.get(m, {})

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-logo">🌿</div>
        <div><div class="sb-name">AGRI-X AI</div><div class="sb-tag">Smart farming assistant</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Menu</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")

    if location or weather_current:
        st.markdown('<div class="sb-label">Your farm</div>', unsafe_allow_html=True)
        place = f"📍 {location.get('city', '')}, {location.get('state', '')}" if location else ""
        wx = ""
        if weather_current:
            wx = (f"<div style='margin-top:4px;'>{weather_current.get('icon', '🌡️')} "
                  f"<b style='color:var(--text);'>{weather_current.get('temperature', 0):.0f}°C</b> · "
                  f"{weather_current.get('description', '')}</div>")
        st.markdown(f'<div class="sb-card">{place}{wx}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sb-label">AI models · {len(DETECT_MODELS)}/{len(all_models) + 2} ready</div>',
                unsafe_allow_html=True)
    rows = []
    for m in [lni.NAME, lni.MOBILE] + all_models:
        acc = model_meta(m).get("accuracy", 0)
        color = theme.MODEL_COLORS.get(m, theme.GREEN)
        dot = "🟢" if m in DETECT_MODELS else "⚪"
        rows.append(f"""
        <div class="sb-model">
            <span style="font-size:0.6rem;">{dot}</span>
            <span style="flex:1;">{m}</span>
            <div class="bar"><div style="width:{acc * 100:.0f}%;background:{color};"></div></div>
            <span style="font-weight:700;color:{color};min-width:38px;text-align:right;">{acc:.1%}</span>
        </div>""")
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption("v2.1 · Fresh Field UI")

# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    best_acc = max((r.get('accuracy', 0) for r in REGISTRY.values()), default=0)
    best_model = max(REGISTRY.items(), key=lambda x: x[1].get("accuracy", 0))[0] if REGISTRY else "—"
    season = _get_current_season()

    hero_col, snap_col = st.columns([1.65, 1], gap="medium")
    with hero_col:
        st.markdown(f"""
        <div class="home-hero fade-in">
            {chip_html('🌿 AI for Indian farms', 'green')}
            <h1>Spot crop disease early.<br><span class="accent">Grow with confidence.</span></h1>
            <p class="lead">Photograph a leaf and AGRI-X AI tells you what is wrong, how serious it is,
            and what to do — along with weather alerts, the best crops for your soil and a yield forecast.</p>
            <div class="hero-chips">
                {chip_html(f'🧠 {len(REGISTRY)} AI models', 'gray')}
                {chip_html(f'🎯 {len(CLASS_NAMES)} leaf conditions', 'gray')}
                {chip_html(f'🏆 {best_acc:.1%} best accuracy', 'amber')}
                {chip_html('🔥 Shows where the disease is', 'gray')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with snap_col:
        if weather_current:
            temp = weather_current.get("temperature", 0)
            snap_body = f"""
                <div class="snapshot-place">📍 {location.get('city', 'Your farm') if location else 'Your farm'}</div>
                <div style="color:var(--text-3);font-size:0.85rem;margin-bottom:0.8rem;">
                    {location.get('state', '') if location else ''}</div>
                <div style="display:flex;align-items:center;gap:12px;">
                    <span style="font-size:2.4rem;">{weather_current.get('icon', '🌡️')}</span>
                    <div><div class="snapshot-temp">{temp:.0f}°C</div>
                    <div style="color:var(--text-2);font-size:0.9rem;">{weather_current.get('description', '')}</div></div>
                </div>
                <div class="snapshot-grid">
                    {kv_html('Humidity', f"{weather_current.get('humidity', 0):.0f}%")}
                    {kv_html('Rain', f"{weather_current.get('precipitation', 0):.1f} mm")}
                    {kv_html('Wind', f"{weather_current.get('wind_speed', 0):.0f} km/h")}
                </div>"""
        else:
            snap_body = """<div style="color:var(--text-3);padding:1.5rem 0;">Weather is unavailable right now.
                Set your location in <b>Crop Advisor → Location</b>.</div>"""
        st.markdown(f"""
        <div class="snapshot fade-in">
            <div class="snapshot-head">
                <span class="snapshot-title">Today on your farm</span>
                {chip_html(f'🗓️ {season} season', 'sky')}
            </div>
            {snap_body.strip()}
        </div>
        """, unsafe_allow_html=True)

    # Quick actions
    st.markdown(section_head_html("What would you like to do?"), unsafe_allow_html=True)
    actions = [
        ("🔬 Diagnose a leaf", "Upload a photo and get a diagnosis", "🔬 Disease Detection"),
        ("🌾 Choose crops", "Best crops, profit and yield for your farm", "🌾 Crop Advisor"),
        ("🤖 Ask the assistant", "Farming questions in plain words", "🤖 Farmer Assistant"),
        ("📄 Download a report", "Save today's results as CSV or HTML", "📋 Reports & About"),
    ]
    with st.container(key="quick_actions"):
        qcols = st.columns(4)
        for col, (label, hint, target) in zip(qcols, actions):
            with col:
                st.button(label, key=f"qa_{target}", on_click=go_to, args=(target,), use_container_width=True)
                st.caption(hint)

    # Key numbers
    total_imgs = DATASET_SUMMARY.get("total_images", 20638) if DATASET_SUMMARY else 20638
    st.markdown(section_head_html("At a glance", f"Best model: {best_model}"), unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(stat_card_html(str(len(REGISTRY)), "AI models trained", "🧠", theme.GREEN), unsafe_allow_html=True)
    with sc2:
        st.markdown(stat_card_html(str(len(CLASS_NAMES)), "Leaf conditions", "🎯", theme.SKY), unsafe_allow_html=True)
    with sc3:
        st.markdown(stat_card_html(f"{total_imgs:,}", "Training images", "🖼️", theme.PURPLE), unsafe_allow_html=True)
    with sc4:
        st.markdown(stat_card_html(f"{best_acc:.1%}", "Best accuracy", "🏆", theme.AMBER), unsafe_allow_html=True)

    # Features
    st.markdown(section_head_html("Everything in one place"), unsafe_allow_html=True)
    features = [
        ("🔬", "Disease Detection", "Identifies 15 leaf conditions and highlights the affected area."),
        ("🌦️", "Weather Intelligence", "Live weather, 7-day forecast and farm weather alerts."),
        ("🌾", "Crop Advisor", "Crops ranked for your season, soil and climate, with profit estimates."),
        ("📦", "Yield Forecast", "Expected harvest, what limits it, and what-if improvements."),
        ("⚠️", "Risk Analysis", "How likely the disease is to spread in today's weather."),
        ("💊", "Treatment Guide", "Organic and chemical treatment steps for each disease."),
        ("🤖", "Farmer Assistant", "Quick answers on diseases, crops and weather."),
        ("📊", "Research Analytics", "Accuracy, training curves and model comparison."),
    ]
    fcols = st.columns(4)
    for i, (icon, title, desc) in enumerate(features):
        with fcols[i % 4]:
            st.markdown(feature_card_html(icon, title, desc, index=i), unsafe_allow_html=True)

    st.markdown(section_head_html("How it works"), unsafe_allow_html=True)
    st.markdown(how_it_works_html(), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: DISEASE DETECTION
# ══════════════════════════════════════════════════════════════════
elif page == "🔬 Disease Detection":
    st.markdown(page_header_html(
        "🔬 Disease Detection", "Diagnose a leaf in seconds",
        "Upload a clear photo of a single leaf. The AI names the problem, shows where it is on the leaf "
        "and tells you what to do next.",
    ), unsafe_allow_html=True)

    if not DETECT_MODELS:
        st.error("No trained models found in `saved_models/`. Please train models first.")
        st.stop()

    col_upload, col_result = st.columns([1, 1.25], gap="large")

    with col_upload:
        with st.container(border=True, key="panel_upload"):
            st.markdown("#### 1 · Upload a leaf photo")
            uploaded = st.file_uploader(
                "Upload a leaf image", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
                help="JPG, PNG or WEBP. One leaf, in focus, on a plain background works best.",
                label_visibility="collapsed",
            )
            if uploaded:
                st.image(Image.open(uploaded).convert("RGB"), width="stretch")

            st.markdown("#### 2 · Choose a model")
            default_model = lni.NAME if lni.NAME in DETECT_MODELS else max(
                available, key=lambda m: REGISTRY.get(m, {}).get("accuracy", 0))
            model_choice = st.selectbox(
                "Model", DETECT_MODELS, index=DETECT_MODELS.index(default_model),
                format_func=lambda m: (f"{m} ⭐ recommended · {model_meta(m).get('accuracy', 0):.1%} accurate"
                                       if m == lni.NAME else
                                       f"{m} · lighter · {model_meta(m).get('accuracy', 0):.1%} accurate" if m == lni.MOBILE
                                       else f"{m} · {model_meta(m).get('accuracy', 0):.1%} accurate"),
                help="LesionNet finds the diseased spots first and names the disease from them, so a messy "
                     "background fools it far less. LesionNet-Mobile is a lighter version for slower computers. "
                     "ResNet50 is the most accurate classic model; MobileNetV2 is the fastest.",
                label_visibility="collapsed",
            )
            meta = model_meta(model_choice)
            st.markdown(
                chip_html(f"🎯 {meta.get('accuracy', 0):.1%} accuracy", "green") + " "
                + chip_html(f"⚡ {meta.get('inference_ms', '?')} ms", "sky") + " "
                + chip_html(f"💾 {meta.get('size_mb', '?')} MB", "gray")
                + (" " + chip_html(f"🛡️ {meta['grey_bg_accuracy']:.0%} on a new background", "green")
                   if model_choice in lni.METAS else ""),
                unsafe_allow_html=True,
            )
            if model_choice in lni.METAS:
                st.caption("🔎 " + meta["description"] + " The red map you'll see is exactly what it used.")
            st.write("")
            analyze = st.button("🔍 Analyze leaf", type="primary", use_container_width=True,
                                disabled=not uploaded)

    # Run the analysis once and keep the result, so opening tabs/expanders doesn't lose it
    upload_id = f"{uploaded.name}:{uploaded.size}" if uploaded else None
    if analyze and uploaded:
        image = Image.open(uploaded).convert("RGB")
        use_ln = model_choice in lni.METAS
        with st.spinner(f"Loading {model_choice}..."):
            model = (cached_load_lesionnet(lni.model_path(model_choice)) if use_ln
                     else cached_load_model(find_model_path(model_choice)))
        with st.spinner("Analyzing the leaf..."):
            t0 = time.time()
            if use_ln:
                ln_out = lni.predict(model, image, CLASS_NAMES)
                idx, conf, top5, img_arr = ln_out["idx"], ln_out["confidence"], ln_out["top5"], ln_out["arr"]
            else:
                idx, conf, top5, img_arr, preds = predict_image(model, image, CLASS_NAMES)
            inf_time = (time.time() - t0) * 1000
        pred_class = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
        is_healthy = "healthy" in pred_class.lower()
        kb = DISEASE_KB.get(pred_class, {})
        diag = {
            "upload_id": upload_id, "model": model_choice, "class": pred_class,
            "plant": kb.get("plant", pred_class.split("_")[0]),
            "disease": kb.get("disease", pred_class.replace("_", " ")),
            "display_name": pred_class.replace("_", " ").replace("  ", " — "),
            "confidence": conf, "inference_ms": inf_time, "is_healthy": is_healthy,
            "top5": top5, "image": image, "heatmap": None, "overlay": None,
            "act_pct": 0.0, "severity": None, "risk": None,
        }
        if not is_healthy:
            with st.spinner("Measuring the diseased area..."):
                if use_ln:
                    # LesionNet's own lesion map: the evidence its decision was computed from
                    hm = None
                    lesion = lni.lesion_result(image, ln_out["seg"])
                    overlay = lni.make_overlay(lesion)
                    analysed = lesion["view"]
                    diag["lesion_heat"] = lni.heat_image(lesion)
                else:
                    # Grad-CAM = where the model looked; lesion segmentation = the actual sick tissue
                    hm = run_disease_heatmap_pipeline(model, img_arr, model_choice)
                    attention = hm["masked_heatmap"] if hm is not None else None
                    lesion, overlay, analysed = analyze_leaf(image, attention=attention, disease_class=pred_class)
            diag["heatmap"] = hm
            diag["lesion"] = lesion
            diag["analysed"] = analysed
            diag["overlay"] = overlay
            diag["act_pct"] = lesion["affected_pct"]
            sev_name, sev_score, sev_color, sev_desc = estimate_severity(conf, diag["act_pct"])
            diag["severity"] = {"level": sev_name, "score": sev_score, "color": sev_color, "desc": sev_desc}
            if weather_current:
                diag["risk"] = calculate_risk(
                    pred_class,
                    weather_current.get("temperature", 25),
                    weather_current.get("humidity", 60),
                    weather_current.get("precipitation", 0),
                )
        diag["treatment"] = get_treatment(pred_class)
        st.session_state["diagnosis"] = diag

        # Shared with the Farmer Assistant and Field Reports pages
        st.session_state["last_prediction"] = {
            "class": pred_class, "display_name": diag["display_name"], "confidence": conf,
            "model": model_choice, "inference_ms": inf_time, "is_healthy": is_healthy,
        }
        st.session_state["last_severity"] = (
            {"level": diag["severity"]["level"], "score": diag["severity"]["score"]} if diag["severity"] else None)
        st.session_state["last_risk"] = diag["risk"]
        if diag["treatment"]:
            st.session_state["last_treatment"] = diag["treatment"]
        if is_healthy:
            st.balloons()

    diag = st.session_state.get("diagnosis")
    if diag and diag.get("upload_id") != upload_id:
        diag = None  # a different (or no) photo is selected — don't show an old result

    with col_result:
        if not diag:
            st.markdown("""
            <div class="empty-state">
                <div class="big">🍃</div>
                <b>No leaf analysed yet</b><br>
                Upload a photo on the left and press <b>Analyze leaf</b>.<br>
                <span style="font-size:0.85rem;">Supported crops: tomato, potato and bell pepper.</span>
            </div>""", unsafe_allow_html=True)
        else:
            healthy = diag["is_healthy"]
            conf = diag["confidence"]
            conf_color = theme.GREEN if conf >= 0.8 else theme.AMBER if conf >= 0.5 else theme.RED
            chips = [chip_html("✅ Healthy leaf", "green") if healthy else chip_html("⚠️ Disease detected", "red")]
            sev = diag.get("severity")
            if sev:
                tone = {"Severe": "red", "Moderate": "amber"}.get(sev["level"], "sky")
                chips.append(chip_html(f"Severity: {sev['level']}", tone))
            risk = diag.get("risk")
            if risk and risk.get("risk_level") not in (None, "Unknown"):
                tone = {"Critical": "red", "High": "red", "Moderate": "amber"}.get(risk["risk_level"], "green")
                chips.append(chip_html(f"Spread risk today: {risk['risk_level']}", tone))
            chips.append(chip_html(f"🧠 {diag['model']}", "gray"))
            low_conf = conf < 0.6
            if low_conf:
                chips.append(chip_html("🤔 Low confidence", "amber"))

            if healthy:
                advice = ("No disease found. Keep watering at the roots, keep checking leaves every week, "
                          "and rotate crops each season.")
                advice_style = "background:var(--green-soft);color:var(--green-strong);"
            else:
                tr = diag.get("treatment") or {}
                first = (tr.get("chemical_treatment") or tr.get("organic_treatment") or [None])[0]
                advice = sev["desc"] if sev else ""
                if first:
                    advice += f" <b>First step:</b> {first}."
                advice_style = "background:var(--red-soft);color:var(--red);"
            if low_conf:
                advice += (" <br><b>The AI is not very sure.</b> Try a sharper photo of a single leaf in daylight, "
                           "or pick LesionNet / ResNet50.")

            reliable = (diag.get("lesion") or {}).get("reliable", True)
            area_txt = f"{'≈ ' if not reliable else ''}{diag['act_pct']:.1f}% of leaf"
            third_kv = (kv_html("Diseased area", area_txt) if not healthy
                        else kv_html("Analysis time", f"{diag['inference_ms']:.0f} ms"))
            if not healthy and not reliable:
                notes = "; ".join(diag["lesion"].get("quality_notes", []))
                advice += (f" <br><b>Area is approximate</b> — {notes}. "
                           "A photo of one whole leaf on a plain background gives the best measurement.")
            st.markdown(f"""
            <div class="diag-card {'is-disease' if not healthy else ''} fade-in">
                <div class="diag-plant">{diag['plant']}</div>
                <div class="diag-name">{diag['disease'] if not healthy else 'Healthy'}</div>
                <div class="diag-chips">{''.join(chips)}</div>
                <div class="kv-label">How sure the AI is · <b style="color:{conf_color};">{conf:.1%}</b></div>
                {meter_html(conf, conf_color)}
                <div class="diag-grid">
                    {kv_html('Confidence', f'{conf:.1%}', conf_color)}
                    {kv_html('Severity', sev['level'] if sev else 'None', sev['color'] if sev else theme.GREEN)}
                    {third_kv}
                </div>
                <div class="diag-advice" style="{advice_style}">{advice}</div>
            </div>
            """, unsafe_allow_html=True)

            if diag["overlay"] is not None:
                gc1, gc2 = st.columns(2)
                gc1.image(diag["image"], caption="Your photo", width="stretch")
                gc2.image(diag["overlay"], width="stretch",
                          caption=f"Diseased tissue: {diag['act_pct']:.1f}% of the leaf")
                if (diag.get("lesion") or {}).get("source") == lni.NAME:
                    st.markdown(chip_html("🟥 diseased tissue found by LesionNet (darker red = more certain)", "red"),
                                unsafe_allow_html=True)
                else:
                    st.markdown(
                        chip_html("🟥 dead / spotted", "red") + " " + chip_html("🟨 yellowing", "amber")
                        + (" " + chip_html("🟦 mould", "sky") if (diag.get('lesion') or {}).get('breakdown', {}).get('pale / mould', 0) > 0 else ""),
                        unsafe_allow_html=True)

    # Details below the summary
    if diag:
        st.markdown(section_head_html("Details & next steps"), unsafe_allow_html=True)
        tab_treat, tab_risk, tab_heat, tab_top = st.tabs(
            ["💊 Treatment", "⚠️ Weather risk", "🔥 Disease map", "📊 Other possibilities"])

        with tab_treat:
            if diag.get("treatment"):
                render_treatment_cards(diag["treatment"])
            else:
                kb = DISEASE_KB.get(diag["class"])
                if kb:
                    for icon, label, key in [("🧪", "Pesticide", "pesticide"), ("🧫", "Fungicide", "fungicide"),
                                             ("💧", "Irrigation", "irrigation"), ("🌱", "Fertilizer", "fertilizer"),
                                             ("📝", "Notes", "notes")]:
                        st.markdown(recommendation_card_html(label, icon, kb[key]), unsafe_allow_html=True)
                else:
                    st.info("No treatment guidance available for this result.")

        with tab_risk:
            if diag.get("risk"):
                render_risk_card(diag["risk"])
            elif diag["is_healthy"]:
                st.success("The leaf is healthy, so there is no disease to spread.")
            else:
                st.info("Weather data is unavailable, so spread risk can't be estimated. "
                        "Set your location in Crop Advisor → Location.")

        with tab_heat:
            lesion = diag.get("lesion")
            hm = diag.get("heatmap")
            if lesion is None:
                st.info("Disease maps are only made for diseased leaves.")
            elif lesion.get("source") == lni.NAME:
                regions = lesion.get("regions", [])
                st.markdown(
                    chip_html(f"🍃 Diseased: {lesion['affected_pct']:.1f}% of leaf", "red") + " "
                    + chip_html(f"🎯 {len(regions)} spot{'s' if len(regions) != 1 else ''}", "gray") + " "
                    + chip_html("🧠 map = the evidence LesionNet used", "green"),
                    unsafe_allow_html=True)
                st.caption("LesionNet first marks every pixel as background, healthy leaf or diseased tissue. "
                           "It then names the disease using only the leaf and diseased pixels, so the background "
                           "cannot influence the answer. This map is not a guess about where it looked — it is "
                           "the input to its decision.")
                d1, d2, d3 = st.columns(3)
                d1.image(lesion["leaf_mask"], caption="Leaf found by LesionNet", width="stretch", clamp=True)
                d2.image(diag["lesion_heat"], caption="How sure it is each pixel is diseased", width="stretch")
                d3.image(diag["overlay"], caption="Diseased tissue", width="stretch")
                if regions:
                    st.markdown("**Largest spots**")
                    for rs in regions[:8]:
                        emoji = "🔴" if rs["severity"] == "severe" else "🟠" if rs["severity"] == "moderate" else "🟡"
                        st.markdown(f"{emoji} **Spot {rs['id']}** — {rs['area_pct']:.1f}% of the leaf · "
                                    f"{rs['confidence']:.0%} sure")
            else:
                bd = lesion.get("breakdown", {})
                st.markdown(
                    chip_html(f"🍃 Diseased: {lesion['affected_pct']:.1f}% of leaf", "red") + " "
                    + chip_html(f"Dead / spotted {bd.get('dead / spotted', 0):.1f}%", "gray") + " "
                    + chip_html(f"Yellowing {bd.get('yellowing', 0):.1f}%", "gray") + " "
                    + (chip_html(f"Mould {bd.get('pale / mould', 0):.1f}%", "gray") + " " if bd.get('pale / mould', 0) else "")
                    + chip_html(f"🎯 {len(lesion['regions'])} spot{'s' if len(lesion['regions']) != 1 else ''}", "gray"),
                    unsafe_allow_html=True)
                st.caption("The diseased area is measured pixel by pixel from colour: tissue that is browner, "
                           "darker or yellower than this leaf's own healthy green. The Grad-CAM map shows where "
                           "the AI model looked when it named the disease.")
                import matplotlib.cm as cm_detail
                d1, d2, d3 = st.columns(3)
                d1.image(lesion["leaf_mask"], caption="Leaf found in the photo", width="stretch", clamp=True)
                if hm is not None:
                    d2.image(cm_detail.jet(hm["masked_heatmap"])[:, :, :3], caption="Where the AI looked (Grad-CAM++)",
                             width="stretch", clamp=True)
                d3.image(diag["overlay"], caption="Diseased tissue", width="stretch")
                regions = lesion.get("regions", [])
                if regions:
                    st.markdown("**Largest spots**")
                    for rs in regions[:8]:
                        emoji = "🔴" if rs["severity"] == "severe" else "🟠" if rs["severity"] == "moderate" else "🟡"
                        kind = {"severe": "mostly dead tissue", "moderate": "mixed", "mild": "mostly yellowing"}[rs["severity"]]
                        st.markdown(f"{emoji} **Spot {rs['id']}** — {rs['area_pct']:.1f}% of the leaf · {kind}")

        with tab_top:
            for item in diag["top5"]:
                st.progress(item["confidence"], text=f"{item['class'].replace('_', ' ')} — {item['confidence']:.1%}")


# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# PAGE: CROP ADVISOR (unified: location + weather + terrain + crops + insights)
# ══════════════════════════════════════════════════════════════════
elif page == "🌾 Crop Advisor":
    st.markdown(page_header_html(
        "🌾 Crop Advisor", "Plan your next crop",
        "Your location, weather and soil in one place — with the best crops to grow, expected profit and a yield forecast.",
    ), unsafe_allow_html=True)

    # Farm context at a glance (same data every tab uses)
    _soil_q, _ = calculate_soil_quality(location.get("soil_data") if location else None)
    st.markdown(context_bar_html([
        ("📍", "Location", f"{location.get('city', '—')}, {location.get('state', '')}" if location else "—"),
        (weather_current.get("icon", "🌡️") if weather_current else "🌡️", "Weather",
         f"{weather_current.get('temperature', 0):.0f}°C · {weather_current.get('humidity', 0):.0f}% humidity" if weather_current else "Unavailable"),
        ("🪨", "Soil", (f"{location.get('soil_group_label', '')} · {location.get('soil_type', '')} · "
                       f"{_soil_q:.0%} fertile") if location else "—"),
        ("🧭", "Farming region", location.get("agro_region", "—") if location else "—"),
        ("🗓️", "Season", _get_current_season()),
    ]), unsafe_allow_html=True)

    tab_crop, tab_weather, tab_forecast, tab_terrain, tab_insights, tab_loc = st.tabs([
        "🌾 Crops & Yield", "☀️ Weather", "📅 7-Day Forecast", "⛰️ Soil & Terrain", "🧠 Insights", "📍 Change Location"
    ])

    # ── TAB: Location ──
    with tab_loc:
        render_location_card(location)
        render_location_override_form()

        if st.button("🔄 Refresh All Data", key="refresh_weather", use_container_width=True):
            loc = detect_location()
            raw = fetch_weather(loc["latitude"], loc["longitude"])
            st.session_state["weather_raw"] = raw
            st.session_state["weather_current"] = get_current_weather(raw) if raw else {}
            st.rerun()

    # ── TAB: Current Weather ──
    with tab_weather:
        if weather_current:
            render_current_weather_card(weather_current)

            # Agricultural weather alerts
            temp = weather_current.get("temperature", 0)
            humidity = weather_current.get("humidity", 0)
            wind = weather_current.get("wind_speed", 0)
            uv = weather_current.get("uv_index", 0)
            precip = weather_current.get("precipitation", 0)

            alerts = []
            if temp >= 40:
                alerts.append(("🔴", "Extreme Heat", f"Temperature at {temp:.1f}°C — crop heat stress likely. Increase irrigation.", "#B91C1C"))
            elif temp >= 35:
                alerts.append(("🟠", "High Temperature", f"Temperature at {temp:.1f}°C — monitor for wilting and water stress.", "#B45309"))
            elif temp <= 5:
                alerts.append(("🔵", "Frost Risk", f"Temperature at {temp:.1f}°C — protect frost-sensitive crops.", "#2563A8"))
            if humidity >= 85:
                alerts.append(("🟡", "High Humidity", f"Humidity at {humidity}% — increased risk of fungal diseases.", "#A16207"))
            if wind >= 40:
                alerts.append(("🟠", "Strong Winds", f"Wind at {wind:.0f} km/h — risk of crop lodging and physical damage.", "#C2410C"))
            if uv >= 8:
                alerts.append(("🔴", "High UV Index", f"UV index at {uv:.1f} — sunscald risk on fruits and vegetables.", "#6D28D9"))
            if precip >= 20:
                alerts.append(("🟡", "Heavy Rainfall", f"Rainfall at {precip:.1f}mm — check drainage, risk of waterlogging.", "#2563A8"))

            if alerts:
                st.markdown("### ⚠️ Agricultural Weather Alerts")
                for dot, title, desc, color in alerts:
                    st.markdown(f"""
                    <div class="glass-card" style="border-left:4px solid {color};padding:0.8rem 1.2rem;margin-bottom:0.5rem;">
                        <div style="font-weight:700;color:{color};font-size:0.95rem;">{dot} {title}</div>
                        <div style="color:#47554B;font-size:0.85rem;margin-top:0.2rem;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="glass-card" style="border-left:4px solid #2E7D32;padding:0.8rem 1.2rem;">
                    <div style="font-weight:700;color:#2E7D32;font-size:0.95rem;">✅ No Weather Alerts</div>
                    <div style="color:#47554B;font-size:0.85rem;margin-top:0.2rem;">Current conditions are favorable for agricultural activities.</div>
                </div>
                """, unsafe_allow_html=True)

            # Hourly
            if weather_raw:
                st.divider()
                hourly = get_hourly_forecast(weather_raw, hours=12)
                render_hourly_forecast(hourly)
        else:
            st.warning("Unable to fetch weather data. Go to the Location tab and click Refresh All Data.")

    # ── TAB: 7-Day Forecast ──
    with tab_forecast:
        if weather_raw:
            daily = get_daily_forecast(weather_raw)
            render_daily_forecast(daily)

            st.divider()
            st.markdown("### 📈 Temperature Trend")
            if daily:
                fig = go.Figure()
                dates = [d["day_name"] for d in daily]
                fig.add_trace(go.Scatter(x=dates, y=[d["temp_max"] for d in daily], mode="lines+markers", name="Max Temp", line=dict(color="#B91C1C", width=3), marker=dict(size=8, symbol="circle")))
                fig.add_trace(go.Scatter(x=dates, y=[d["temp_min"] for d in daily], mode="lines+markers", name="Min Temp", line=dict(color="#2563A8", width=3), marker=dict(size=8, symbol="circle"), fill="tonexty", fillcolor="rgba(37,99,168,0.08)"))
                fig.update_layout(template="agrix", height=350, margin=dict(t=30, b=30, l=40, r=20), yaxis_title="°C", showlegend=True, legend=dict(orientation="h", y=1.12), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🌧️ Precipitation Forecast")
            if daily:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=dates, y=[d["precipitation"] for d in daily], name="Rainfall (mm)", marker_color="#2563A8", marker_line_color="#1E4E8C", marker_line_width=1))
                fig2.update_layout(template="agrix", height=280, margin=dict(t=20, b=30, l=40, r=20), yaxis_title="mm", showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)

            if daily:
                st.markdown("### 💨 Wind & UV Summary")
                wind_cols = st.columns(len(daily))
                for i, d in enumerate(daily):
                    wc = "#B91C1C" if d["wind_speed"] >= 40 else "#B45309" if d["wind_speed"] >= 20 else "#2E7D32"
                    uc = "#B91C1C" if d["uv_index"] >= 8 else "#B45309" if d["uv_index"] >= 5 else "#2E7D32"
                    with wind_cols[i]:
                        st.markdown(f"""
                        <div style="text-align:center;padding:0.5rem;background:rgba(30,42,34,0.03);border-radius:10px;">
                            <div style="font-size:0.7rem;color:#6B776E;">{d['day_name'][:3]}</div>
                            <div style="font-size:0.9rem;font-weight:700;color:{wc};margin:0.2rem 0;">💨{d['wind_speed']:.0f}</div>
                            <div style="font-size:0.9rem;font-weight:700;color:{uc};">☀️{d['uv_index']:.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("No forecast data available. Go to the Location tab and click Refresh All Data.")

    # ── TAB: Terrain & Soil ──
    with tab_terrain:
        st.markdown("### ⛰️ Elevation & Soil Analysis")
        st.markdown('<p style="color:#6B776E;font-size:0.9rem;margin-bottom:1rem;">Terrain and soil data for your location, powered by Open-Meteo Elevation API and ISRIC SoilGrids.</p>', unsafe_allow_html=True)
        render_location_details(location)

    # ── TAB: Crops & Yield ──
    with tab_crop:
        render_crop_advisor_page(location, weather_current, weather_raw)
        st.divider()
        render_yield_page(location, weather_current, weather_raw)

    # ── TAB: Insights ──
    with tab_insights:
        daily = get_daily_forecast(weather_raw) if weather_raw else []
        render_insights_dashboard(weather_current, daily, location)


# ══════════════════════════════════════════════════════════════════
# PAGE: FARMER ASSISTANT
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Farmer Assistant":
    prediction_result = st.session_state.get("last_prediction")
    render_chatbot(weather_current, prediction_result, location)


# ══════════════════════════════════════════════════════════════════
# PAGE: RESEARCH DASHBOARD (PRESERVED)
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Research Dashboard":
    st.markdown(page_header_html(
        "📊 Research", "Research dashboard",
        "Our LesionNet study, training history, evaluation metrics and explanation maps for every model.",
    ), unsafe_allow_html=True)

    RESEARCH_MODELS = rv.research_models(REGISTRY)
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Models", len(RESEARCH_MODELS))
    rc2.metric("Trained", len(available) + len(LN_MODELS))
    rc3.metric("Classes", len(CLASS_NAMES))
    rc4.metric("Images", f"{DATASET_SUMMARY['total_images']:,}" if DATASET_SUMMARY else "—")

    tab_study, tab_train, tab_cm, tab_roc, tab_gradcam, tab_metrics = st.tabs([
        "🛡️ LesionNet Study", "📈 Training History", "🔢 Confusion Matrices", "📉 ROC & PR Curves",
        "🔥 Explanation Maps", "📋 Metrics & Reports"
    ])

    with tab_study:
        rv.render_study_tab()

    with tab_train:
        comp_path = os.path.join(GRAPHS_DIR, "all_models_training_comparison.png")
        if os.path.exists(comp_path):
            st.markdown("**All Models — Training Comparison**")
            st.image(comp_path, width="stretch")
            st.divider()
        for mname in RESEARCH_MODELS:
            gpath = find_training_graph(mname)
            if gpath:
                st.markdown(f"**{mname} — Training History**")
                st.image(gpath, width="stretch")
                st.divider()
            else:
                st.caption(f"⏳ {mname} — training graph not available")

    with tab_cm:
        combined_cm_path = os.path.join(CONFUSION_DIR, "all_models_confusion_matrix.png")
        if os.path.exists(combined_cm_path):
            st.markdown("**📊 Overall Confusion Matrix — All Models**")
            st.image(combined_cm_path, width="stretch")
            st.divider()
        st.markdown("**Individual Confusion Matrices**")
        cm_cols = st.columns(2)
        col_idx = 0
        for mname in RESEARCH_MODELS:
            cm_path = os.path.join(CONFUSION_DIR, f"{mname}_confusion_matrix.png")
            if os.path.exists(cm_path):
                with cm_cols[col_idx % 2]:
                    st.markdown(f"**{mname}**")
                    st.image(cm_path, width="stretch")
                col_idx += 1
        if col_idx == 0:
            st.info("No confusion matrices found. Run evaluation to generate them.")

    with tab_roc:
        combined_roc_path = os.path.join(ROC_DIR, "all_models_roc_curves.png")
        if os.path.exists(combined_roc_path):
            st.markdown("**📈 ROC Curves — All Models (Macro-Average)**")
            st.image(combined_roc_path, width="stretch")
            st.divider()
        st.markdown("**Individual ROC & Precision-Recall Curves**")
        for mname in RESEARCH_MODELS:
            roc_path = os.path.join(ROC_DIR, f"{mname}_roc_curves.png")
            pr_path = os.path.join(ROC_DIR, f"{mname}_pr_curves.png")
            if os.path.exists(roc_path) or os.path.exists(pr_path):
                st.markdown(f"#### {mname}")
                rc1, rc2 = st.columns(2)
                if os.path.exists(roc_path):
                    rc1.image(roc_path, caption="ROC Curves", width="stretch")
                if os.path.exists(pr_path):
                    rc2.image(pr_path, caption="PR Curves", width="stretch")
                st.divider()

    with tab_gradcam:
        _q = os.path.join(rv.LN_FIGS, "fig_qualitative.png")
        if os.path.exists(_q):
            st.markdown("**LesionNet vs. Grad-CAM++ — test photos**")
            st.image(_q, width=820, caption="Grad-CAM++ (classic models) shows roughly where a model looked; "
                     "LesionNet's map is the evidence its decision is computed from.")
            st.divider()
        for mname in REGISTRY:
            gcimgs = find_gradcam_images(mname)
            if gcimgs:
                st.markdown(f"**{mname} — Grad-CAM Samples**")
                cols = st.columns(min(len(gcimgs), 3))
                for i, gp in enumerate(gcimgs[:6]):
                    cols[i % 3].image(gp, width="stretch", caption=os.path.basename(gp).replace(".png","").replace("_"," "))
                st.divider()

    with tab_metrics:
        if METRICS_DF is not None:
            st.markdown("**📊 Model Performance Summary**")
            display_df = rv.metrics_table(METRICS_DF).copy()
            for col in ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
            if "Inference Time (ms)" in display_df.columns:
                display_df["Inference Time (ms)"] = display_df["Inference Time (ms)"].apply(lambda x: f"{x:.1f} ms" if pd.notna(x) else "—")
            if "Model Size (MB)" in display_df.columns:
                display_df["Model Size (MB)"] = display_df["Model Size (MB)"].apply(lambda x: f"{x:.1f} MB")
            st.dataframe(display_df.astype(object).where(display_df.notna(), "—"), use_container_width=True, hide_index=True)
            st.divider()

        st.markdown("**📝 Classification Reports**")
        for mname in RESEARCH_MODELS:
            report_path = os.path.join(METRICS_DIR, f"{mname}_classification_report.txt")
            if os.path.exists(report_path):
                with st.expander(f"📄 {mname} — Classification Report"):
                    with open(report_path, "rb") as f:
                        raw_report = f.read()
                    # Reports written on Windows are cp1252; decode either way
                    try:
                        report_text = raw_report.decode("utf-8")
                    except UnicodeDecodeError:
                        report_text = raw_report.decode("cp1252", errors="replace")
                    st.code(report_text, language="text")

        st.divider()
        st.markdown("**📈 All Generated Charts**")
        all_imgs = get_all_graph_images()
        if all_imgs:
            for fname, fpath in all_imgs.items():
                with st.expander(fname.replace("_", " ").replace(".png", "")):
                    st.image(fpath, width="stretch")
        else:
            st.info("No report images found. Run the training pipeline to generate evaluation reports.")


# ══════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON (PRESERVED)
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown(page_header_html(
        "📈 Models", "Compare the AI models",
        "Accuracy, speed and size of each model — and where each one fits best.",
    ), unsafe_allow_html=True)

    import robustness_view
    robustness_view.render()

    CMP = rv.compare_registry(REGISTRY)
    METRICS_ALL = rv.metrics_table(METRICS_DF)
    if METRICS_ALL is not None:
        st.markdown("### 📊 Performance Metrics")
        display_df = METRICS_ALL.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
        if "Inference Time (ms)" in display_df.columns:
            display_df["Inference Time (ms)"] = display_df["Inference Time (ms)"].apply(lambda x: f"{x:.1f} ms" if pd.notna(x) else "—")
        if "Model Size (MB)" in display_df.columns:
            display_df["Model Size (MB)"] = display_df["Model Size (MB)"].apply(lambda x: f"{x:.1f} MB")
        st.dataframe(display_df.astype(object).where(display_df.notna(), "—"), use_container_width=True, hide_index=True)
        st.caption("Classic models: their original test split. LesionNet models: the 3,109-photo held-out split. "
                   "LesionNet speeds are estimated on the same scale from a same-machine relative measurement.")
    else:
        rows = []
        for m, d in CMP.items():
            rows.append({"Model": m, "Accuracy": f"{d.get('accuracy',0):.0%}", "Params": d.get("params","—"), "Size": f"{d.get('size_mb','—')} MB", "Inference": f"{d.get('inference_ms','—')} ms", "Status": d.get("status","—").upper()})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Performance Comparison")
    c1, c2 = st.columns(2)
    names = list(CMP.keys())
    color_list = [theme.MODEL_COLORS.get(m, theme.GREEN) for m in CMP]

    with c1:
        accs = [CMP[m].get("accuracy", 0) for m in names]
        fig = go.Figure(go.Bar(x=names, y=accs, marker_color=color_list[:len(names)], text=[f"{a:.1%}" for a in accs], textposition="outside"))
        fig.update_layout(title="Accuracy Comparison", yaxis_range=[0,1.1], template="agrix", height=400, margin=dict(t=50,b=30))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sizes = [CMP[m].get("size_mb", 0) for m in names]
        fig2 = go.Figure(go.Bar(x=names, y=sizes, marker_color=color_list[:len(names)], text=[f"{s}MB" for s in sizes], textposition="outside"))
        fig2.update_layout(title="Model Size Comparison", template="agrix", height=400, margin=dict(t=50,b=30))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        inf_times = [CMP[m].get("inference_ms", 0) for m in names]
        fig_inf = go.Figure(go.Bar(x=names, y=inf_times, marker_color=color_list[:len(names)], text=[f"{t:.0f}ms" for t in inf_times], textposition="outside"))
        fig_inf.update_layout(title="Inference Speed (ms)", template="agrix", height=400, margin=dict(t=50,b=30))
        st.plotly_chart(fig_inf, use_container_width=True)

    with c4:
        if METRICS_ALL is not None and "F1 Score" in METRICS_ALL.columns:
            f1s = METRICS_ALL["F1 Score"].tolist()
            m_names = METRICS_ALL["Model"].tolist()
            fig_f1 = go.Figure(go.Bar(x=m_names, y=f1s, marker_color=[theme.MODEL_COLORS.get(m, theme.GREEN) for m in m_names], text=[f"{f:.2%}" for f in f1s], textposition="outside"))
            fig_f1.update_layout(title="F1 Score Comparison", yaxis_range=[0,1.1], template="agrix", height=400, margin=dict(t=50,b=30))
            st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown("### Architecture Analysis")
    for mname, mdata in CMP.items():
        with st.expander(f"🏗️ {mname}", expanded=False):
            dc1, dc2 = st.columns(2)
            dc1.markdown(f"""
**Architecture:** {mdata.get('architecture', mname)}
**Parameters:** {mdata.get('params', '—')}
**Model Size:** {mdata.get('size_mb', '—')} MB
**Inference Speed:** {mdata.get('inference_ms', '—')} ms
""")
            dc2.markdown(f"""
**✅ Strengths:** {mdata.get('strengths', '—')}
**⚠️ Weaknesses:** {mdata.get('weaknesses', '—')}
**🚀 Best For:** {mdata.get('deployment', '—')}
""")

    st.markdown("### Multi-Metric Radar")
    categories = ["Accuracy", "Speed", "Size Efficiency", "Deployability"]
    fig3 = go.Figure()
    color_map = theme.MODEL_COLORS
    for m, d in CMP.items():
        acc = d.get("accuracy", 0)
        speed = max(0, 1 - d.get("inference_ms", 50) / 200)
        size_eff = max(0, 1 - d.get("size_mb", 50) / 120)
        deploy = 0.9 if d.get("status") == "verified" else 0.3
        fig3.add_trace(go.Scatterpolar(r=[acc, speed, size_eff, deploy], theta=categories, fill='toself', name=m, line=dict(color=color_map.get(m, "#6B776E"))))
    fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), template="agrix", height=450, margin=dict(t=30,b=30))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: REPORTS & ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "📋 Reports & About":
    st.markdown(page_header_html(
        "📋 Reports & About", "Reports, data and project info",
        "Download today's field report, explore the training dataset and read how AGRI-X AI works.",
    ), unsafe_allow_html=True)
    tab_reports, tab_paper, tab_dataset, tab_about = st.tabs(
        ["📄 Field Reports", "🔬 Research Paper", "📋 Dataset Explorer", "ℹ️ About"])

    with tab_paper:
        rv.render_paper_tab()

    with tab_reports:
        st.markdown(section_head_html("📄 Field report", "CSV or HTML"), unsafe_allow_html=True)

        render_report_section(
            prediction=st.session_state.get("last_prediction"),
            weather=weather_current if weather_current else None,
            location=location,
            risk=st.session_state.get("last_risk"),
            treatment=st.session_state.get("last_treatment"),
            severity=st.session_state.get("last_severity"),
            crop_results=st.session_state.get("crop_results"),
            yield_result=st.session_state.get("yield_result"),
        )

    with tab_dataset:
        st.markdown(section_head_html("📋 Dataset explorer", "PlantVillage"), unsafe_allow_html=True)

        if DATASET_SUMMARY:
            ds = DATASET_SUMMARY
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Total Images", f"{ds['total_images']:,}")
            dc2.metric("Classes", ds['num_classes'])
            dc3.metric("Min Class", ds['min_class_size'])
            dc4.metric("Max Class", ds['max_class_size'])

            dist = ds.get("class_distribution", {})
            if dist:
                st.markdown("### Class Distribution")
                df = pd.DataFrame({"Class": list(dist.keys()), "Count": list(dist.values())})
                df["Display"] = df["Class"].str.replace("_", " ")
                fig = px.bar(df, x="Count", y="Display", orientation="h", color="Count",
                             color_continuous_scale=["#E5F1E3", "#2E7D32"], height=500)
                fig.update_layout(template="agrix", yaxis=dict(autorange="reversed"),
                                  margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Dataset Statistics")
                stats_df = pd.DataFrame({
                    "Metric": ["Total Images", "Number of Classes", "Mean Class Size", "Std Class Size", "Min Class Size", "Max Class Size", "Image Resolution", "Class Imbalance Ratio"],
                    "Value": [f"{ds['total_images']:,}", str(ds['num_classes']), str(ds['mean_class_size']), str(ds['std_class_size']), str(ds['min_class_size']), str(ds['max_class_size']), ds['image_size'], f"{ds['max_class_size']/ds['min_class_size']:.1f}x"]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

                with st.expander("📊 Per-Class Breakdown"):
                    class_df = pd.DataFrame({"Class": list(dist.keys()), "Images": list(dist.values())})
                    class_df["Percentage"] = (class_df["Images"] / class_df["Images"].sum() * 100).round(1).astype(str) + "%"
                    class_df["Plant"] = class_df["Class"].apply(lambda x: x.split("_")[0].replace("Pepper","Pepper Bell"))
                    st.dataframe(class_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Dataset summary not found. Run the training pipeline to generate dataset statistics.")
        st.divider()
        rv.render_plantdoc_dataset()

    with tab_about:
        st.markdown(section_head_html("ℹ️ About AGRI-X AI", "Methodology, architecture and references"), unsafe_allow_html=True)

        st.markdown("""
        ### 🎯 Project Overview

        **AGRI-X AI** is an end-to-end, research-grade framework that combines transfer learning,
        explainable AI, and smart agricultural decision support for automated
        plant disease identification and farming intelligence.

        Its default diagnostic model is **LesionNet**, our own network: it outlines the diseased spots
        first and names the disease from those spots only, so a changed background barely affects it
        (98.7% on a grey background vs. 45–81% for classic CNNs) and its lesion map doubles as a
        severity measure. See the **Research Paper** tab.

        ### 🏗️ Architecture
        """)

        st.code("""
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (app.py)                  │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2 Services                                                │
│  Location · Weather · Risk · Treatment · Crop · Yield · Chat    │
├─────────────────────────────────────────────────────────────────┤
│                     API Layer (src/api/)                         │
│                 PlantDiseasePredictor                            │
├──────────────────────┬──────────────────────────────────────────┤
│  Pipelines Layer     │  Visualization Layer                     │
│  • Training          │  • Training Plots                        │
│  • Evaluation        │  • Evaluation Plots                      │
│  • Hyperparameter    │  • Grad-CAM Heatmaps                    │
├──────────────────────┼──────────────────────────────────────────┤
│  Models Layer        │  Evaluation Layer                        │
│  • MobileNetV2       │  • MetricsCalculator                    │
│  • ResNet50          │  • CrossValidator                       │
│  • EfficientNetB0    │  • StatisticalAnalyzer                  │
│  • DenseNet121       │  • AblationStudy                        │
│  • LesionNet (ours)  │  • Background-robustness tests           │
├──────────────────────┴──────────────────────────────────────────┤
│                     Data Layer (src/data/)                       │
│              DatasetManager — Loading, Splitting, Augmentation   │
├─────────────────────────────────────────────────────────────────┤
│  Config · Entity · Utils — Settings, Dataclasses, Logging       │
└─────────────────────────────────────────────────────────────────┘
        """, language="text")

        st.markdown("### 🔬 Methodology")
        methods = [
            ("1", "Two-Phase Transfer Learning", "Phase 1: Frozen backbone (5 epochs) → Phase 2: Fine-tune top layers (15 epochs)"),
            ("2", "Data Augmentation", "Rotation ±30°, horizontal flip, zoom ±20%, brightness 0.8-1.2×, shifts ±15%"),
            ("3", "Multi-Model Comparison", "MobileNetV2, ResNet50, EfficientNetB0, DenseNet121 trained & evaluated"),
            ("4", "Grad-CAM Explainability", "Gradient-weighted Class Activation Mapping for visual disease region identification"),
            ("5", "K-Fold Cross-Validation", "5-fold stratified CV for statistically robust performance estimates"),
            ("6", "Disease Risk Scoring", "Weather-aware risk calculation combining temperature, humidity, and rainfall"),
            ("7", "LesionNet (our model)", "Segmentation head + lesion-gated pooling: the classifier sees only leaf and lesion pixels; trained from automatic colour-based lesion masks with background-swap augmentation"),
            ("8", "Robustness & Faithfulness Tests", "Grey / swapped / background-only test photos, evidence-on-lesion, pointing game, severity correlation, and a PlantDoc field-photo test"),
        ]
        for num, title, desc in methods:
            st.markdown(f"""
            <div class="method-card">
                <span class="method-number">{num}</span>
                <span class="method-title">{title}</span>
                <div class="method-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Technology Stack")
        techs = ["Python 3.11", "TensorFlow 2.18", "Keras 3", "Streamlit 1.57",
                  "Plotly", "Scikit-learn", "NumPy", "Pandas", "OpenCV", "Matplotlib",
                  "Open-Meteo API", "Seaborn"]
        st.markdown("".join(f'<span class="tech-badge">{t}</span>' for t in techs), unsafe_allow_html=True)

        st.markdown("### 📚 References")
        refs = [
            ("1", "Sandler, M., et al. (2018). \"MobileNetV2: Inverted Residuals and Linear Bottlenecks.\" CVPR 2018."),
            ("2", "He, K., et al. (2016). \"Deep Residual Learning for Image Recognition.\" CVPR 2016."),
            ("3", "Tan, M. & Le, Q. (2019). \"EfficientNet: Rethinking Model Scaling.\" ICML 2019."),
            ("4", "Huang, G., et al. (2017). \"Densely Connected Convolutional Networks.\" CVPR 2017."),
            ("5", "Selvaraju, R. R., et al. (2017). \"Grad-CAM: Visual Explanations from Deep Networks.\" ICCV 2017."),
            ("6", "Hughes, D. P. & Salathe, M. (2015). \"PlantVillage: An open access repository of plant health images.\""),
            ("7", "Chattopadhay, A., et al. (2018). \"Grad-CAM++: Generalized Gradient-based Visual Explanations.\" WACV 2018."),
            ("8", "Singh, D., et al. (2020). \"PlantDoc: A Dataset for Visual Plant Disease Detection.\" CoDS-COMAD 2020."),
            ("9", "Geirhos, R., et al. (2020). \"Shortcut Learning in Deep Neural Networks.\" Nature Machine Intelligence."),
            ("10", "Noyan, M. A. (2022). \"Uncovering Bias in the PlantVillage Dataset.\" arXiv:2206.04374."),
            ("11", "Naik, R., Patel, D., Mehta, S., Mathukia, P. & Tawde, P. (2026). \"LesionNet: Lesion-Gated Classification for Background-Robust, Faithfully Explainable and Severity-Aware Plant Disease Diagnosis.\" Manuscript in preparation."),
        ]
        for num, text in refs:
            st.markdown(f"""
            <div class="ref-card">
                <span class="ref-number">{num}</span> {text}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center;padding:1rem;color:var(--text-3);">
            <strong>🌿 AGRI-X AI — Empowering Agriculture with Explainable AI</strong><br>
            <em>Built with TensorFlow, Keras, and Streamlit</em>
        </div>
        """, unsafe_allow_html=True)
