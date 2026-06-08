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
import glob

from app_utils import (
    CUSTOM_CSS, BASE_DIR, SAVED_MODELS_DIR, GRAPHS_DIR, GRADCAM_DIR,
    RESEARCH_FIGURES, CONFUSION_DIR, ROC_DIR, METRICS_DIR, METRICS_CSV,
    load_class_names, load_registry, load_dataset_summary, load_metrics_csv,
    find_model_path, get_available_models, find_training_graph,
    find_gradcam_images, get_all_graph_images,
    confidence_bar_html, severity_badge_html, metric_card_html,
    recommendation_card_html, feature_card_html, stat_card_html,
)

# Phase-2 service modules
from location_service import detect_location, render_location_card, render_location_override_form
from weather_service import (
    fetch_weather, get_current_weather, get_hourly_forecast,
    get_daily_forecast, render_current_weather_card,
    render_hourly_forecast, render_daily_forecast,
)
from risk_engine import calculate_risk, render_risk_card
from treatment_engine import get_treatment, render_treatment_cards
from crop_advisor import render_crop_advisor_page
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
    "EfficientNetB0": "top_conv",
    "DenseNet121": "relu",
}

# ─── Location & Weather (session-cached) ───
if "location" not in st.session_state:
    st.session_state["location"] = None

location = detect_location()

if "weather_raw" not in st.session_state:
    st.session_state["weather_raw"] = None
    st.session_state["weather_current"] = {}

if location and st.session_state["weather_raw"] is None:
    raw = fetch_weather(location["latitude"], location["longitude"])
    st.session_state["weather_raw"] = raw
    st.session_state["weather_current"] = get_current_weather(raw) if raw else {}

weather_raw = st.session_state["weather_raw"]
weather_current = st.session_state["weather_current"]

# ─── Model Loading (cached) ───
@st.cache_resource
def cached_load_model(path):
    return tf.keras.models.load_model(path, compile=False)

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

def generate_gradcam(model, img_array, layer_name):
    try:
        target = None
        backbone = None
        for layer in model.layers:
            if layer.name == layer_name:
                target = layer; break
            if hasattr(layer, 'layers'):
                try:
                    target = layer.get_layer(layer_name); backbone = layer; break
                except (ValueError, KeyError):
                    continue
        if target is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    target = layer; break
                if hasattr(layer, 'layers'):
                    for sl in reversed(layer.layers):
                        if isinstance(sl, tf.keras.layers.Conv2D):
                            target = sl; backbone = layer; break
                    if target: break
        if target is None:
            return None, None, 0.0
        if backbone is None:
            grad_model = tf.keras.models.Model(inputs=model.input, outputs=[target.output, model.output])
        else:
            backbone_dual = tf.keras.models.Model(inputs=backbone.input, outputs=[target.output, backbone.output])
            inp = model.input; x = inp
            for l in model.layers:
                if isinstance(l, tf.keras.layers.InputLayer): continue
                if l is backbone: break
                x = l(x)
            conv_out, bb_out = backbone_dual(x)
            y = bb_out; past_backbone = False
            for l in model.layers:
                if l is backbone: past_backbone = True; continue
                if past_backbone and not isinstance(l, tf.keras.layers.InputLayer):
                    y = l(y)
            grad_model = tf.keras.models.Model(inputs=inp, outputs=[conv_out, y])
        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(img_array)
            pred_idx = tf.argmax(predictions[0])
            class_out = predictions[:, pred_idx]
        grads = tape.gradient(class_out, conv_out)
        if grads is None: return None, None, 0.0
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        hm_uint8 = np.uint8(255 * heatmap)
        hm_img = Image.fromarray(hm_uint8).resize((224, 224), Image.BILINEAR)
        heatmap_full = np.array(hm_img).astype(np.float32) / 255.0
        import matplotlib.cm as cm
        colormap = cm.jet(heatmap_full)[:, :, :3]
        original = img_array[0]
        overlay = np.clip(0.6 * original + 0.4 * colormap, 0, 1)
        act_pct = float(np.sum(heatmap_full > 0.3) / heatmap_full.size * 100)
        return heatmap_full, overlay, act_pct
    except Exception as e:
        st.warning(f"Grad-CAM error: {e}")
        return None, None, 0.0

def estimate_severity(confidence, act_pct):
    score = 0.4 * confidence + 0.6 * min(act_pct / 100, 1.0)
    if score < 0.35:
        return "Mild", score, "#FFA500", "Early-stage infection. Minimal spread."
    elif score < 0.65:
        return "Moderate", score, "#FF6347", "Moderate infection. Treatment recommended."
    else:
        return "Severe", score, "#DC143C", "Severe infection. Immediate action required."

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
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.4rem;">🌿 AGRI-X AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Agricultural Intelligence Platform</p>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔬 Disease Detection",
            "🌦️ Weather & Location",
            "🌾 Crop Advisor",
            "🤖 Farmer Assistant",
            "📊 Research Dashboard",
            "📈 Model Comparison",
            "📋 Reports & About",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    available = get_available_models(REGISTRY)
    all_models = list(REGISTRY.keys())
    st.caption(f"**Models Available:** {len(available)}/{len(all_models)}")
    for m in all_models:
        status = "✅" if m in available else "⏳"
        acc = REGISTRY[m].get("accuracy", 0)
        color = "#2ecc71" if acc >= 0.90 else "#3498db" if acc >= 0.80 else "#f39c12" if acc >= 0.60 else "#e74c3c"
        st.markdown(f'{status} {m} — <span style="color:{color};font-weight:600">{acc:.0%}</span>', unsafe_allow_html=True)
    st.divider()

    if REGISTRY:
        best = max(REGISTRY.items(), key=lambda x: x[1].get("accuracy", 0))
        st.success(f"🏆 Best: {best[0]} ({best[1].get('accuracy',0):.0%})")

    # Location mini-badge
    if location:
        st.markdown(f"📍 {location.get('city','')}, {location.get('state','')}")
    if weather_current:
        t = weather_current.get("temperature", 0)
        st.markdown(f"{weather_current.get('icon','🌡️')} {t:.0f}°C · {weather_current.get('description','')}")

    st.caption("v2.0 • Phase 2 — AI Platform")

# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    # Hero
    st.markdown("""
    <div class="hero-banner fade-in">
        <h1 class="hero-title" style="font-size:2.6rem;margin-bottom:0.3rem;">🌿 AGRI-X AI</h1>
        <p style="color:#8899aa;font-size:1.1rem;margin:0 0 0.5rem 0;">
            Explainable Multi-Model Deep Learning Framework for<br>
            <span class="shimmer-text" style="font-size:1.3rem;font-weight:700;">Plant Disease Detection & Agricultural Intelligence</span>
        </p>
        <p style="color:#667;font-size:0.9rem;margin-top:0.8rem;">
            Powered by MobileNetV2 · ResNet50 · EfficientNetB0 · DenseNet121 — with Grad-CAM Explainability
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    total_imgs = DATASET_SUMMARY.get("total_images", 20638) if DATASET_SUMMARY else 20638
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("🧠 Models", len(REGISTRY))
    sc2.metric("📊 Classes", len(CLASS_NAMES))
    sc3.metric("🖼️ Images", f"{total_imgs:,}")
    sc4.metric("🏆 Best Acc", f"{max((r.get('accuracy',0) for r in REGISTRY.values()), default=0):.0%}")
    sc5.metric("📍 Location", location.get("city", "—") if location else "—")

    # Feature cards
    st.markdown("### ✨ Platform Features")
    features = [
        ("🔬", "Disease Detection", "AI-powered plant disease identification with Grad-CAM explainability and severity analysis"),
        ("🌦️", "Weather Intelligence", "Real-time weather data, forecasts, and agricultural weather alerts"),
        ("🌾", "Crop Advisor", "Smart crop recommendations with yield prediction and revenue forecasting"),
        ("⚠️", "Risk Analysis", "Weather-aware disease risk scoring and spread probability estimation"),
        ("💊", "Treatment Guide", "Comprehensive organic & chemical treatment recommendations per disease"),
        ("🤖", "Farmer Assistant", "AI chatbot for instant farming advice, disease help, and crop guidance"),
        ("📊", "Research Analytics", "Training curves, ROC/PR curves, confusion matrices, and model comparison"),
        ("📄", "Field Reports", "Downloadable CSV & HTML reports with complete analysis results"),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(feature_card_html(icon, title, desc), unsafe_allow_html=True)

    # Quick weather
    if weather_current:
        st.markdown("### 🌦️ Current Conditions")
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("🌡️ Temperature", f"{weather_current.get('temperature',0):.1f}°C")
        wc2.metric("💧 Humidity", f"{weather_current.get('humidity',0)}%")
        wc3.metric("🌧️ Rainfall", f"{weather_current.get('precipitation',0):.1f}mm")
        wc4.metric("💨 Wind", f"{weather_current.get('wind_speed',0):.1f}km/h")


# ══════════════════════════════════════════════════════════════════
# PAGE: DISEASE DETECTION (PRESERVED — extended with risk/treatment)
# ══════════════════════════════════════════════════════════════════
elif page == "🔬 Disease Detection":
    st.markdown('<h1 class="hero-title fade-in">🔬 Disease Detection & Diagnosis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub fade-in">Upload a leaf image for AI-powered disease identification with explainable results</p>', unsafe_allow_html=True)

    if not available:
        st.error("No trained models found in `saved_models/`. Please train models first.")
        st.stop()

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### 📤 Upload & Configure")
        model_choice = st.selectbox("Select Model Architecture", available, help="Choose a trained model for inference")
        meta = REGISTRY.get(model_choice, {})
        mcols = st.columns(3)
        mcols[0].metric("Accuracy", f"{meta.get('accuracy',0):.0%}")
        mcols[1].metric("Size", f"{meta.get('size_mb','?')} MB")
        mcols[2].metric("Speed", f"{meta.get('inference_ms','?')} ms")
        uploaded = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"], help="Supported: JPG, JPEG, PNG")
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col_result:
        st.markdown("#### 🧠 Analysis Results")
        if uploaded and model_choice:
            if st.button("🔍 Analyze Disease", use_container_width=True, type="primary"):
                image = Image.open(uploaded)
                model_path = find_model_path(model_choice)
                with st.spinner(f"Loading {model_choice}..."):
                    model = cached_load_model(model_path)
                with st.spinner("Running inference..."):
                    t0 = time.time()
                    idx, conf, top5, img_arr, preds = predict_image(model, image, CLASS_NAMES)
                    inf_time = (time.time() - t0) * 1000
                pred_class = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
                display_name = pred_class.replace("_", " ").replace("  ", " — ")
                is_healthy = "healthy" in pred_class.lower()

                # Store prediction in session for reports/chatbot
                st.session_state["last_prediction"] = {
                    "class": pred_class, "display_name": display_name,
                    "confidence": conf, "model": model_choice,
                    "inference_ms": inf_time, "is_healthy": is_healthy,
                }

                # Result header
                if is_healthy:
                    st.success(f"✅ **{display_name}**")
                else:
                    st.error(f"⚠️ **{display_name}**")

                st.markdown("**Prediction Confidence**")
                st.progress(conf, text=f"{conf:.1%}")

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Confidence", f"{conf:.1%}")
                mc2.metric("Inference", f"{inf_time:.0f}ms")
                mc3.metric("Model", model_choice)

                # Severity + Grad-CAM for diseased plants
                if not is_healthy:
                    heatmap, overlay, act_pct = generate_gradcam(model, img_arr, GRADCAM_LAYERS.get(model_choice, ""))
                    sev_name, sev_score, sev_color, sev_desc = estimate_severity(conf, act_pct)

                    # Store severity
                    st.session_state["last_severity"] = {"level": sev_name, "score": sev_score}

                    if sev_name == "Severe":
                        st.error(f"🔴 **Severity: {sev_name}** — {sev_desc}")
                    elif sev_name == "Moderate":
                        st.warning(f"🟠 **Severity: {sev_name}** — {sev_desc}")
                    else:
                        st.info(f"🟡 **Severity: {sev_name}** — {sev_desc}")

                    if overlay is not None:
                        st.markdown("**🔥 Grad-CAM Explainability**")
                        gc1, gc2 = st.columns(2)
                        gc1.image(img_arr[0], caption="Original", use_container_width=True)
                        gc2.image(overlay, caption=f"Disease Activation ({act_pct:.1f}%)", use_container_width=True)

                    # Disease Risk (weather-aware)
                    if weather_current:
                        risk = calculate_risk(
                            pred_class,
                            weather_current.get("temperature", 25),
                            weather_current.get("humidity", 60),
                            weather_current.get("precipitation", 0),
                        )
                        st.session_state["last_risk"] = risk
                        render_risk_card(risk)
                else:
                    st.balloons()
                    st.session_state["last_severity"] = None
                    st.session_state["last_risk"] = None

                # Top-5
                with st.expander("📊 Top-5 Predictions", expanded=False):
                    for item in top5:
                        dn = item["class"].replace("_", " ")
                        st.progress(item["confidence"], text=f"{dn}: {item['confidence']:.1%}")

                # Treatment recommendations
                treatment = get_treatment(pred_class)
                if treatment:
                    st.session_state["last_treatment"] = treatment
                    st.markdown("---")
                    st.markdown("### 💊 Treatment Recommendations")
                    render_treatment_cards(treatment)

                # Fallback KB recommendations
                kb = DISEASE_KB.get(pred_class)
                if kb and not treatment:
                    st.markdown("---")
                    st.markdown("### 💊 Agricultural Recommendations")
                    with st.expander("🧪 Pesticide", expanded=True):
                        st.write(kb["pesticide"])
                    with st.expander("🧫 Fungicide"):
                        st.write(kb["fungicide"])
                    with st.expander("💧 Irrigation"):
                        st.write(kb["irrigation"])
                    with st.expander("🌱 Fertilizer"):
                        st.write(kb["fertilizer"])
                    with st.expander("📝 Additional Notes"):
                        st.info(kb["notes"])
        else:
            st.info("👈 Upload a leaf image and click **Analyze Disease** to begin.")


# ══════════════════════════════════════════════════════════════════
# PAGE: WEATHER & LOCATION
# ══════════════════════════════════════════════════════════════════
elif page == "🌦️ Weather & Location":
    st.markdown('<h1 class="hero-title fade-in">🌦️ Weather & Location Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub fade-in">Real-time weather data and location detection for precision agriculture</p>', unsafe_allow_html=True)

    # Location section
    render_location_card(location)
    render_location_override_form()

    # Refresh weather button
    if st.button("🔄 Refresh Weather Data", key="refresh_weather"):
        loc = detect_location()
        raw = fetch_weather(loc["latitude"], loc["longitude"])
        st.session_state["weather_raw"] = raw
        st.session_state["weather_current"] = get_current_weather(raw) if raw else {}
        st.rerun()

    st.divider()

    # Current weather
    if weather_current:
        st.markdown("### ☀️ Current Weather")
        render_current_weather_card(weather_current)

        # Hourly
        if weather_raw:
            hourly = get_hourly_forecast(weather_raw, hours=12)
            render_hourly_forecast(hourly)

            st.divider()

            # Daily
            daily = get_daily_forecast(weather_raw)
            render_daily_forecast(daily)

            # Temperature chart
            st.divider()
            st.markdown("### 📈 Temperature Trend")
            if daily:
                fig = go.Figure()
                dates = [d["day_name"] for d in daily]
                fig.add_trace(go.Scatter(x=dates, y=[d["temp_max"] for d in daily], mode="lines+markers", name="Max", line=dict(color="#e74c3c", width=2), marker=dict(size=8)))
                fig.add_trace(go.Scatter(x=dates, y=[d["temp_min"] for d in daily], mode="lines+markers", name="Min", line=dict(color="#3498db", width=2), marker=dict(size=8), fill="tonexty", fillcolor="rgba(52,152,219,0.1)"))
                fig.update_layout(template="plotly_dark", height=350, margin=dict(t=30, b=30), yaxis_title="°C", showlegend=True, legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Unable to fetch weather data. Check your internet connection or try refreshing.")


# ══════════════════════════════════════════════════════════════════
# PAGE: CROP ADVISOR (includes yield prediction & insights)
# ══════════════════════════════════════════════════════════════════
elif page == "🌾 Crop Advisor":
    tab_crop, tab_yield, tab_insights = st.tabs(["🌾 Crop Recommendations", "📊 Yield Prediction", "🧠 Insights"])

    with tab_crop:
        render_crop_advisor_page(location, weather_current)

    with tab_yield:
        render_yield_page(weather_current)

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
    st.markdown('<h1 class="hero-title fade-in">📊 Research Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub fade-in">Training history, evaluation metrics, and Grad-CAM visualizations</p>', unsafe_allow_html=True)

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Models", len(REGISTRY))
    rc2.metric("Trained", len(available))
    rc3.metric("Classes", len(CLASS_NAMES))
    rc4.metric("Images", f"{DATASET_SUMMARY['total_images']:,}" if DATASET_SUMMARY else "—")

    tab_train, tab_cm, tab_roc, tab_gradcam, tab_metrics = st.tabs([
        "📈 Training History", "🔢 Confusion Matrices", "📉 ROC & PR Curves",
        "🔥 Grad-CAM Gallery", "📋 Metrics & Reports"
    ])

    with tab_train:
        comp_path = os.path.join(GRAPHS_DIR, "all_models_training_comparison.png")
        if os.path.exists(comp_path):
            st.markdown("**All Models — Training Comparison**")
            st.image(comp_path, use_container_width=True)
            st.divider()
        for mname in REGISTRY:
            gpath = find_training_graph(mname)
            if gpath:
                st.markdown(f"**{mname} — Training History**")
                st.image(gpath, use_container_width=True)
                st.divider()
            else:
                st.caption(f"⏳ {mname} — training graph not available")

    with tab_cm:
        combined_cm_path = os.path.join(CONFUSION_DIR, "all_models_confusion_matrix.png")
        if os.path.exists(combined_cm_path):
            st.markdown("**📊 Overall Confusion Matrix — All Models**")
            st.image(combined_cm_path, use_container_width=True)
            st.divider()
        st.markdown("**Individual Confusion Matrices**")
        cm_cols = st.columns(2)
        col_idx = 0
        for mname in REGISTRY:
            cm_path = os.path.join(CONFUSION_DIR, f"{mname}_confusion_matrix.png")
            if os.path.exists(cm_path):
                with cm_cols[col_idx % 2]:
                    st.markdown(f"**{mname}**")
                    st.image(cm_path, use_container_width=True)
                col_idx += 1
        if col_idx == 0:
            st.info("No confusion matrices found. Run evaluation to generate them.")

    with tab_roc:
        combined_roc_path = os.path.join(ROC_DIR, "all_models_roc_curves.png")
        if os.path.exists(combined_roc_path):
            st.markdown("**📈 ROC Curves — All Models (Macro-Average)**")
            st.image(combined_roc_path, use_container_width=True)
            st.divider()
        st.markdown("**Individual ROC & Precision-Recall Curves**")
        for mname in REGISTRY:
            roc_path = os.path.join(ROC_DIR, f"{mname}_roc_curves.png")
            pr_path = os.path.join(ROC_DIR, f"{mname}_pr_curves.png")
            if os.path.exists(roc_path) or os.path.exists(pr_path):
                st.markdown(f"#### {mname}")
                rc1, rc2 = st.columns(2)
                if os.path.exists(roc_path):
                    rc1.image(roc_path, caption="ROC Curves", use_container_width=True)
                if os.path.exists(pr_path):
                    rc2.image(pr_path, caption="PR Curves", use_container_width=True)
                st.divider()

    with tab_gradcam:
        for mname in REGISTRY:
            gcimgs = find_gradcam_images(mname)
            if gcimgs:
                st.markdown(f"**{mname} — Grad-CAM Samples**")
                cols = st.columns(min(len(gcimgs), 3))
                for i, gp in enumerate(gcimgs[:6]):
                    cols[i % 3].image(gp, use_container_width=True, caption=os.path.basename(gp).replace(".png","").replace("_"," "))
                st.divider()

    with tab_metrics:
        if METRICS_DF is not None:
            st.markdown("**📊 Model Performance Summary**")
            display_df = METRICS_DF.copy()
            for col in ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
            if "Inference Time (ms)" in display_df.columns:
                display_df["Inference Time (ms)"] = display_df["Inference Time (ms)"].apply(lambda x: f"{x:.1f} ms")
            if "Model Size (MB)" in display_df.columns:
                display_df["Model Size (MB)"] = display_df["Model Size (MB)"].apply(lambda x: f"{x:.1f} MB")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.divider()

        st.markdown("**📝 Classification Reports**")
        for mname in REGISTRY:
            report_path = os.path.join(METRICS_DIR, f"{mname}_classification_report.txt")
            if os.path.exists(report_path):
                with st.expander(f"📄 {mname} — Classification Report"):
                    with open(report_path, "r") as f:
                        st.code(f.read(), language="text")

        st.divider()
        st.markdown("**📈 All Generated Charts**")
        all_imgs = get_all_graph_images()
        if all_imgs:
            for fname, fpath in all_imgs.items():
                with st.expander(fname.replace("_", " ").replace(".png", "")):
                    st.image(fpath, use_container_width=True)
        else:
            st.info("No report images found. Run the training pipeline to generate evaluation reports.")


# ══════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON (PRESERVED)
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.markdown('<h1 class="hero-title fade-in">📈 Multi-Model Comparison</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub fade-in">Architecture analysis, performance metrics, and deployment suitability</p>', unsafe_allow_html=True)

    if METRICS_DF is not None:
        st.markdown("### 📊 Performance Metrics")
        display_df = METRICS_DF.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
        if "Inference Time (ms)" in display_df.columns:
            display_df["Inference Time (ms)"] = display_df["Inference Time (ms)"].apply(lambda x: f"{x:.1f} ms")
        if "Model Size (MB)" in display_df.columns:
            display_df["Model Size (MB)"] = display_df["Model Size (MB)"].apply(lambda x: f"{x:.1f} MB")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        rows = []
        for m, d in REGISTRY.items():
            rows.append({"Model": m, "Accuracy": f"{d.get('accuracy',0):.0%}", "Params": d.get("params","—"), "Size": f"{d.get('size_mb','—')} MB", "Inference": f"{d.get('inference_ms','—')} ms", "Status": d.get("status","—").upper()})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Performance Comparison")
    c1, c2 = st.columns(2)
    names = list(REGISTRY.keys())
    color_list = ["#2ecc71","#3498db","#e74c3c","#f39c12"]

    with c1:
        accs = [REGISTRY[m].get("accuracy", 0) for m in names]
        fig = go.Figure(go.Bar(x=names, y=accs, marker_color=color_list[:len(names)], text=[f"{a:.0%}" for a in accs], textposition="outside"))
        fig.update_layout(title="Accuracy Comparison", yaxis_range=[0,1.1], template="plotly_dark", height=400, margin=dict(t=50,b=30))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sizes = [REGISTRY[m].get("size_mb", 0) for m in names]
        fig2 = go.Figure(go.Bar(x=names, y=sizes, marker_color=color_list[:len(names)], text=[f"{s}MB" for s in sizes], textposition="outside"))
        fig2.update_layout(title="Model Size Comparison", template="plotly_dark", height=400, margin=dict(t=50,b=30))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        inf_times = [REGISTRY[m].get("inference_ms", 0) for m in names]
        fig_inf = go.Figure(go.Bar(x=names, y=inf_times, marker_color=color_list[:len(names)], text=[f"{t:.0f}ms" for t in inf_times], textposition="outside"))
        fig_inf.update_layout(title="Inference Speed (ms)", template="plotly_dark", height=400, margin=dict(t=50,b=30))
        st.plotly_chart(fig_inf, use_container_width=True)

    with c4:
        if METRICS_DF is not None and "F1 Score" in METRICS_DF.columns:
            f1s = METRICS_DF["F1 Score"].tolist()
            m_names = METRICS_DF["Model"].tolist()
            fig_f1 = go.Figure(go.Bar(x=m_names, y=f1s, marker_color=color_list[:len(m_names)], text=[f"{f:.2%}" for f in f1s], textposition="outside"))
            fig_f1.update_layout(title="F1 Score Comparison", yaxis_range=[0,1.1], template="plotly_dark", height=400, margin=dict(t=50,b=30))
            st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown("### Architecture Analysis")
    for mname, mdata in REGISTRY.items():
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
    color_map = {"MobileNetV2": "#2ecc71", "ResNet50": "#3498db", "EfficientNetB0": "#e74c3c", "DenseNet121": "#f39c12"}
    for m, d in REGISTRY.items():
        acc = d.get("accuracy", 0)
        speed = max(0, 1 - d.get("inference_ms", 50) / 200)
        size_eff = max(0, 1 - d.get("size_mb", 50) / 120)
        deploy = 0.9 if d.get("status") == "verified" else 0.3
        fig3.add_trace(go.Scatterpolar(r=[acc, speed, size_eff, deploy], theta=categories, fill='toself', name=m, line=dict(color=color_map.get(m, "#888"))))
    fig3.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), template="plotly_dark", height=450, margin=dict(t=30,b=30))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# PAGE: REPORTS & ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "📋 Reports & About":
    tab_reports, tab_dataset, tab_about = st.tabs(["📄 Field Reports", "📋 Dataset Explorer", "ℹ️ About"])

    with tab_reports:
        st.markdown('<h1 class="hero-title fade-in">📄 Field Report Generator</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub fade-in">Download comprehensive analysis reports in CSV or HTML format</p>', unsafe_allow_html=True)

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
        st.markdown('<h1 class="hero-title fade-in">📋 Dataset Explorer</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub fade-in">PlantVillage dataset analysis — class distribution and statistics</p>', unsafe_allow_html=True)

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
                             color_continuous_scale=["#1a1f2e", "#2ecc71"], height=500)
                fig.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"),
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

    with tab_about:
        st.markdown('<h1 class="hero-title fade-in">ℹ️ About AGRI-X AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub fade-in">Project methodology, architecture, and references</p>', unsafe_allow_html=True)

        st.markdown("""
        ### 🎯 Project Overview

        **AGRI-X AI** is an end-to-end, research-grade framework that combines transfer learning,
        explainable AI (Grad-CAM), and smart agricultural decision support for automated
        plant disease identification and farming intelligence.

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
        techs = ["Python 3.13", "TensorFlow 2.21", "Keras 3.13", "Streamlit 1.55",
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
        ]
        for num, text in refs:
            st.markdown(f"""
            <div class="ref-card">
                <span class="ref-number">{num}</span> {text}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="text-align:center;padding:1rem;color:#667;">
            <strong>🌿 AGRI-X AI — Empowering Agriculture with Explainable AI</strong><br>
            <em>Built with TensorFlow, Keras, and Streamlit</em>
        </div>
        """, unsafe_allow_html=True)
