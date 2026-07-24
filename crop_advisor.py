"""
AGRI-X AI — Crop Recommendation & Revenue Forecaster (Module 12)
Land area + location + weather → crop suitability → yield → revenue → profit.
"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ── Crop Database ──
CROP_DB = {
    "Rice": {
        "season": ["Kharif"], "temp_min": 20, "temp_max": 35, "humidity_min": 60, "humidity_max": 80,
        "rainfall_min": 1000, "rainfall_max": 2000, "soil": ["Alluvial", "Clay", "Loamy"],
        "yield_min": 1.0, "yield_max": 1.5, "price_min": 2000, "price_max": 2400,
        "msp": 2183, "cost_per_acre": 18000, "duration_days": 120,
        "water_need": "High", "icon": "🌾",
    },
    "Wheat": {
        "season": ["Rabi"], "temp_min": 10, "temp_max": 25, "humidity_min": 50, "humidity_max": 70,
        "rainfall_min": 400, "rainfall_max": 700, "soil": ["Alluvial", "Loamy", "Clay"],
        "yield_min": 1.2, "yield_max": 1.8, "price_min": 2100, "price_max": 2500,
        "msp": 2275, "cost_per_acre": 16000, "duration_days": 135,
        "water_need": "Moderate", "icon": "🌾",
    },
    "Tomato": {
        "season": ["Kharif", "Rabi"], "temp_min": 18, "temp_max": 30, "humidity_min": 50, "humidity_max": 80,
        "rainfall_min": 400, "rainfall_max": 600, "soil": ["Loamy", "Sandy", "Red"],
        "yield_min": 8, "yield_max": 15, "price_min": 800, "price_max": 2500,
        "msp": 0, "cost_per_acre": 45000, "duration_days": 100,
        "water_need": "Moderate", "icon": "🍅",
    },
    "Potato": {
        "season": ["Rabi"], "temp_min": 15, "temp_max": 25, "humidity_min": 60, "humidity_max": 80,
        "rainfall_min": 500, "rainfall_max": 800, "soil": ["Loamy", "Sandy", "Alluvial"],
        "yield_min": 6, "yield_max": 10, "price_min": 600, "price_max": 1500,
        "msp": 0, "cost_per_acre": 35000, "duration_days": 100,
        "water_need": "Moderate", "icon": "🥔",
    },
    "Bell Pepper": {
        "season": ["Kharif", "Rabi"], "temp_min": 18, "temp_max": 28, "humidity_min": 60, "humidity_max": 70,
        "rainfall_min": 600, "rainfall_max": 1200, "soil": ["Loamy", "Sandy"],
        "yield_min": 4, "yield_max": 8, "price_min": 1500, "price_max": 4000,
        "msp": 0, "cost_per_acre": 55000, "duration_days": 90,
        "water_need": "Moderate", "icon": "🫑",
    },
    "Cotton": {
        "season": ["Kharif"], "temp_min": 25, "temp_max": 35, "humidity_min": 50, "humidity_max": 65,
        "rainfall_min": 500, "rainfall_max": 1000, "soil": ["Black", "Alluvial"],
        "yield_min": 0.6, "yield_max": 0.8, "price_min": 6000, "price_max": 7500,
        "msp": 6620, "cost_per_acre": 22000, "duration_days": 165,
        "water_need": "Moderate", "icon": "🏵️",
    },
    "Sugarcane": {
        "season": ["Kharif", "Rabi"], "temp_min": 20, "temp_max": 35, "humidity_min": 70, "humidity_max": 85,
        "rainfall_min": 1500, "rainfall_max": 2500, "soil": ["Alluvial", "Loamy", "Clay"],
        "yield_min": 25, "yield_max": 35, "price_min": 290, "price_max": 350,
        "msp": 315, "cost_per_acre": 40000, "duration_days": 330,
        "water_need": "Very High", "icon": "🎋",
    },
    "Maize": {
        "season": ["Kharif"], "temp_min": 20, "temp_max": 30, "humidity_min": 55, "humidity_max": 75,
        "rainfall_min": 500, "rainfall_max": 1000, "soil": ["Loamy", "Alluvial", "Red"],
        "yield_min": 1.0, "yield_max": 1.5, "price_min": 1900, "price_max": 2300,
        "msp": 2090, "cost_per_acre": 14000, "duration_days": 100,
        "water_need": "Moderate", "icon": "🌽",
    },
    "Soybean": {
        "season": ["Kharif"], "temp_min": 20, "temp_max": 30, "humidity_min": 60, "humidity_max": 70,
        "rainfall_min": 600, "rainfall_max": 1000, "soil": ["Black", "Loamy"],
        "yield_min": 0.4, "yield_max": 0.6, "price_min": 4200, "price_max": 5000,
        "msp": 4600, "cost_per_acre": 15000, "duration_days": 100,
        "water_need": "Moderate", "icon": "🫘",
    },
    "Mustard": {
        "season": ["Rabi"], "temp_min": 10, "temp_max": 25, "humidity_min": 40, "humidity_max": 60,
        "rainfall_min": 300, "rainfall_max": 500, "soil": ["Loamy", "Sandy", "Alluvial"],
        "yield_min": 0.5, "yield_max": 0.8, "price_min": 5000, "price_max": 6500,
        "msp": 5650, "cost_per_acre": 12000, "duration_days": 120,
        "water_need": "Low", "icon": "🌻",
    },
    "Onion": {
        "season": ["Rabi"], "temp_min": 15, "temp_max": 25, "humidity_min": 50, "humidity_max": 70,
        "rainfall_min": 350, "rainfall_max": 600, "soil": ["Loamy", "Sandy", "Alluvial"],
        "yield_min": 6, "yield_max": 12, "price_min": 800, "price_max": 3000,
        "msp": 0, "cost_per_acre": 40000, "duration_days": 130,
        "water_need": "Moderate", "icon": "🧅",
    },
    "Chilli": {
        "season": ["Kharif"], "temp_min": 20, "temp_max": 35, "humidity_min": 60, "humidity_max": 70,
        "rainfall_min": 500, "rainfall_max": 1200, "soil": ["Loamy", "Black", "Red"],
        "yield_min": 1.5, "yield_max": 3.0, "price_min": 8000, "price_max": 15000,
        "msp": 0, "cost_per_acre": 50000, "duration_days": 150,
        "water_need": "Moderate", "icon": "🌶️",
    },
}

SOIL_TYPES = ["Alluvial", "Black", "Red", "Laterite", "Sandy", "Loamy", "Clay"]


def _get_current_season():
    """Determine Indian agricultural season from current month."""
    month = datetime.now().month
    if month in [6, 7, 8, 9, 10]:
        return "Kharif"
    elif month in [11, 12, 1, 2, 3]:
        return "Rabi"
    else:
        return "Zaid"


def _score_crop(crop_name, crop, temp, humidity, annual_rainfall, soil, season):
    """Score a crop 0-100 based on conditions."""
    score = 0
    reasons = []

    # Season match (25 pts)
    if season in crop["season"] or season == "Zaid":
        score += 25
        reasons.append(f"✅ Season match ({season})")
    else:
        score += 5
        reasons.append(f"⚠️ Off-season (best: {', '.join(crop['season'])})")

    # Temperature (25 pts)
    if crop["temp_min"] <= temp <= crop["temp_max"]:
        score += 25
        reasons.append(f"✅ Temp {temp:.0f}°C in optimal range")
    elif abs(temp - crop["temp_min"]) <= 5 or abs(temp - crop["temp_max"]) <= 5:
        score += 12
        reasons.append(f"⚠️ Temp {temp:.0f}°C near optimal ({crop['temp_min']}-{crop['temp_max']}°C)")
    else:
        reasons.append(f"❌ Temp {temp:.0f}°C outside range ({crop['temp_min']}-{crop['temp_max']}°C)")

    # Humidity (20 pts)
    if crop["humidity_min"] <= humidity <= crop["humidity_max"]:
        score += 20
        reasons.append(f"✅ Humidity {humidity:.0f}% suitable")
    elif abs(humidity - crop["humidity_min"]) <= 10 or abs(humidity - crop["humidity_max"]) <= 10:
        score += 10
        reasons.append(f"⚠️ Humidity {humidity:.0f}% marginal")
    else:
        reasons.append(f"❌ Humidity {humidity:.0f}% unsuitable")

    # Rainfall estimate (15 pts)
    if crop["rainfall_min"] <= annual_rainfall <= crop["rainfall_max"]:
        score += 15
        reasons.append(f"✅ Rainfall adequate")
    elif annual_rainfall >= crop["rainfall_min"] * 0.7:
        score += 8
        reasons.append(f"⚠️ Rainfall marginal (irrigation may be needed)")
    else:
        score += 3
        reasons.append(f"❌ Rainfall insufficient — heavy irrigation required")

    # Soil match (15 pts)
    if soil in crop["soil"]:
        score += 15
        reasons.append(f"✅ {soil} soil is ideal")
    else:
        score += 5
        reasons.append(f"⚠️ {soil} soil — not ideal but possible")

    return min(100, max(0, score)), reasons


def recommend_crops(temp, humidity, annual_rainfall, soil, season, land_area_acres):
    """
    Recommend crops ranked by suitability score.
    Returns list of dicts with scores, yields, revenue, profit.
    """
    results = []
    for name, crop in CROP_DB.items():
        score, reasons = _score_crop(name, crop, temp, humidity, annual_rainfall, soil, season)
        # Only recommend if score > 20
        if score < 20:
            continue

        yield_min = crop["yield_min"] * land_area_acres
        yield_max = crop["yield_max"] * land_area_acres
        # Revenue in ₹ (price is per quintal = 100 kg, yield is tonnes)
        rev_min = yield_min * 10 * crop["price_min"]  # tonnes → quintals × price
        rev_max = yield_max * 10 * crop["price_max"]
        total_cost = crop["cost_per_acre"] * land_area_acres
        profit_min = rev_min - total_cost
        profit_max = rev_max - total_cost

        results.append({
            "name": name, "icon": crop["icon"], "score": score, "reasons": reasons,
            "season": ", ".join(crop["season"]), "duration": crop["duration_days"],
            "water_need": crop["water_need"],
            "yield_min": yield_min, "yield_max": yield_max,
            "price_min": crop["price_min"], "price_max": crop["price_max"],
            "msp": crop["msp"],
            "revenue_min": rev_min, "revenue_max": rev_max,
            "total_cost": total_cost,
            "profit_min": profit_min, "profit_max": profit_max,
            "cost_per_acre": crop["cost_per_acre"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def render_crop_card(crop, rank):
    """Render a single crop recommendation card."""
    sc = crop["score"]
    if sc >= 75:
        border_color, badge_color = "#2ecc71", "#2ecc71"
    elif sc >= 50:
        border_color, badge_color = "#f39c12", "#f39c12"
    else:
        border_color, badge_color = "#e74c3c", "#e74c3c"

    profit_color = "#2ecc71" if crop["profit_min"] > 0 else "#e74c3c"

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {border_color};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
            <div>
                <span style="font-size:1.5rem;">{crop['icon']}</span>
                <span style="font-size:1.2rem;font-weight:700;color:#fff;margin-left:0.5rem;">#{rank} {crop['name']}</span>
            </div>
            <span style="background:rgba(255,255,255,0.08);color:{badge_color};padding:4px 14px;border-radius:12px;font-size:0.85rem;font-weight:700;">
                {sc}% Match</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;margin-bottom:0.8rem;">
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;">EST. YIELD</div>
                <div style="font-size:1rem;font-weight:700;color:#3498db;">📦 {crop['yield_min']:.1f}-{crop['yield_max']:.1f} t</div></div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;">REVENUE</div>
                <div style="font-size:1rem;font-weight:700;color:#f39c12;">💰 ₹{crop['revenue_min']/1000:.0f}K-{crop['revenue_max']/1000:.0f}K</div></div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;">EST. PROFIT</div>
                <div style="font-size:1rem;font-weight:700;color:{profit_color};">📈 ₹{crop['profit_min']/1000:.0f}K-{crop['profit_max']/1000:.0f}K</div></div>
        </div>
        <div style="display:flex;gap:0.8rem;flex-wrap:wrap;font-size:0.8rem;color:#aab;">
            <span>🗓️ {crop['season']}</span>
            <span>⏱️ {crop['duration']} days</span>
            <span>💧 {crop['water_need']}</span>
            <span>💸 Cost: ₹{crop['cost_per_acre']:,}/acre</span>
            {'<span>🏛️ MSP: ₹' + str(crop["msp"]) + '/q</span>' if crop["msp"] > 0 else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_revenue_comparison_chart(crops):
    """Render Plotly chart comparing revenue across top crops."""
    if not crops:
        return
    names = [c["name"] for c in crops[:6]]
    rev_min = [c["revenue_min"] / 1000 for c in crops[:6]]
    rev_max = [c["revenue_max"] / 1000 for c in crops[:6]]
    costs = [c["total_cost"] / 1000 for c in crops[:6]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Min Revenue", x=names, y=rev_min,
                         marker_color="rgba(52,152,219,0.7)", text=[f"₹{v:.0f}K" for v in rev_min], textposition="outside"))
    fig.add_trace(go.Bar(name="Max Revenue", x=names, y=rev_max,
                         marker_color="rgba(46,204,113,0.7)", text=[f"₹{v:.0f}K" for v in rev_max], textposition="outside"))
    fig.add_trace(go.Bar(name="Input Cost", x=names, y=costs,
                         marker_color="rgba(231,76,60,0.7)", text=[f"₹{v:.0f}K" for v in costs], textposition="outside"))
    fig.update_layout(
        title="💰 Revenue vs Cost Comparison", barmode="group",
        template="plotly_dark", height=400, margin=dict(t=50, b=30),
        yaxis_title="Amount (₹ Thousands)", legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)


def _estimate_annual_rainfall(weather_raw):
    """Estimate annual rainfall (mm) from 7-day daily forecast precipitation sums."""
    if not weather_raw or "daily" not in weather_raw:
        return None
    daily = weather_raw["daily"]
    precip_list = daily.get("precipitation_sum", [])
    if not precip_list:
        return None
    weekly_total = sum(precip_list)
    # Extrapolate 7-day total to annual (×52 weeks)
    annual_estimate = weekly_total * 52
    # Clamp to reasonable range
    return max(100, min(5000, round(annual_estimate)))


def render_crop_advisor_page(location, weather_current, weather_raw=None):
    """Full crop advisor page renderer."""
    st.markdown("### 🌾 Crop Recommendations")

    # Auto-fill weather data
    temp = weather_current.get("temperature", 25) if weather_current else 25
    humidity = weather_current.get("humidity", 60) if weather_current else 60
    city_name = location.get("city", "Unknown") if location else "Unknown"

    # Auto-estimate annual rainfall from forecast
    auto_rainfall = _estimate_annual_rainfall(weather_raw)
    rainfall_default = auto_rainfall if auto_rainfall else 800

    # Input section
    st.markdown("### 📝 Farm Details")

    # Show auto-detected weather info
    if weather_current:
        st.info(
            f"🌡️ **Temperature ({temp:.1f}°C)**, 💧 **Humidity ({humidity:.0f}%)**, "
            f"and 🌧️ **Est. Annual Rainfall (~{rainfall_default} mm)** are auto-filled "
            f"from weather data for **📍 {city_name}**. You can override rainfall below."
        )

    # Auto-fill soil type if detected, matching keys in SOIL_TYPES
    detected_soil = location.get("soil_type", "Loamy") if location else "Loamy"
    soil_index = 5  # default to Loamy (index 5)
    for idx, s_type in enumerate(SOIL_TYPES):
        if s_type.lower() in detected_soil.lower() or detected_soil.lower() in s_type.lower():
            soil_index = idx
            break

    col1, col2, col3 = st.columns(3)
    with col1:
        land_area = st.number_input("🏞️ Land Area", min_value=0.1, max_value=10000.0, value=5.0, step=0.5, key="crop_land")
        unit = st.selectbox("Unit", ["Acres", "Hectares", "Bigha"], key="crop_unit")
    with col2:
        soil = st.selectbox("🪨 Soil Type", SOIL_TYPES, index=soil_index, key="crop_soil")
        season = st.selectbox("🗓️ Season", ["Auto-detect", "Kharif", "Rabi", "Zaid"], key="crop_season")
    with col3:
        est_rainfall = st.number_input(
            "🌧️ Annual Rainfall (mm)",
            min_value=0, max_value=5000,
            value=rainfall_default, step=50,
            key="crop_rain",
            help="Auto-estimated from 7-day weather forecast. Adjust if you know your region's average."
        )

    # Convert to acres
    if unit == "Hectares":
        acres = land_area * 2.471
    elif unit == "Bigha":
        acres = land_area * 0.625
    else:
        acres = land_area

    # Determine season
    if season == "Auto-detect":
        season = _get_current_season()

    st.markdown(f"""
    <div style="display:flex;gap:1rem;flex-wrap:wrap;margin:0.5rem 0 1rem 0;">
        <span style="background:rgba(52,152,219,0.1);color:#3498db;padding:4px 12px;border-radius:8px;font-size:0.85rem;">
            📍 {city_name}</span>
        <span style="background:rgba(46,204,113,0.1);color:#2ecc71;padding:4px 12px;border-radius:8px;font-size:0.85rem;">
            📏 {acres:.1f} acres</span>
        <span style="background:rgba(52,152,219,0.1);color:#3498db;padding:4px 12px;border-radius:8px;font-size:0.85rem;">
            🌡️ {temp:.1f}°C</span>
        <span style="background:rgba(155,89,182,0.1);color:#9b59b6;padding:4px 12px;border-radius:8px;font-size:0.85rem;">
            💧 {humidity:.0f}% humidity</span>
        <span style="background:rgba(142,68,173,0.1);color:#8e44ad;padding:4px 12px;border-radius:8px;font-size:0.85rem;">
            🌧️ ~{est_rainfall} mm/year</span>
        <span style="background:rgba(243,156,18,0.1);color:#f39c12;padding:4px 12px;border-radius:8px;font-size:0.85rem;">
            🗓️ {season} season</span>
        <span style="background:rgba(231,76,60,0.1);color:#e74c3c;padding:4px 12px;border-radius:8px;font-size:0.85rem;">
            🪨 {soil} soil</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🌾 Get Crop Recommendations", type="primary", use_container_width=True, key="btn_crop"):
        with st.spinner("Analyzing conditions and scoring crops..."):
            crops = recommend_crops(temp, humidity, est_rainfall, soil, season, acres)

        if not crops:
            st.warning("No suitable crops found for current conditions. Try adjusting parameters.")
            return

        st.session_state["crop_results"] = crops
        st.markdown(f"### 🏆 Top {min(len(crops), 5)} Recommended Crops")
        for i, crop in enumerate(crops[:5], 1):
            render_crop_card(crop, i)
            with st.expander(f"📋 {crop['name']} — Suitability Details"):
                for r in crop["reasons"]:
                    st.markdown(r)

        st.markdown("### 📊 Revenue Analysis")
        render_revenue_comparison_chart(crops[:6])

    elif "crop_results" in st.session_state:
        crops = st.session_state["crop_results"]
        st.markdown(f"### 🏆 Top {min(len(crops), 5)} Recommended Crops")
        for i, crop in enumerate(crops[:5], 1):
            render_crop_card(crop, i)
            with st.expander(f"📋 {crop['name']} — Suitability Details"):
                for r in crop["reasons"]:
                    st.markdown(r)
        st.markdown("### 📊 Revenue Analysis")
        render_revenue_comparison_chart(crops[:6])
