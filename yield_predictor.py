"""
AGRI-X AI — Yield Prediction Engine (Module 7)
Estimates agricultural yield based on crop, location, weather, season, and area.
"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Yield Database (per acre, in tonnes) ──
YIELD_DB = {
    "Rice":        {"base_yield": 1.2, "optimal_temp": (22, 32), "optimal_humidity": (65, 80), "optimal_rain": (150, 250), "price_per_tonne": 22000, "icon": "🌾", "harvest_months": 4},
    "Wheat":       {"base_yield": 1.5, "optimal_temp": (12, 22), "optimal_humidity": (50, 65), "optimal_rain": (50, 90),  "price_per_tonne": 23000, "icon": "🌾", "harvest_months": 4.5},
    "Maize":       {"base_yield": 1.3, "optimal_temp": (21, 30), "optimal_humidity": (55, 75), "optimal_rain": (60, 120), "price_per_tonne": 21000, "icon": "🌽", "harvest_months": 3.5},
    "Cotton":      {"base_yield": 0.7, "optimal_temp": (25, 35), "optimal_humidity": (50, 65), "optimal_rain": (70, 120), "price_per_tonne": 65000, "icon": "🏵️", "harvest_months": 5.5},
    "Soybean":     {"base_yield": 0.5, "optimal_temp": (20, 30), "optimal_humidity": (60, 70), "optimal_rain": (60, 100), "price_per_tonne": 46000, "icon": "🫘", "harvest_months": 3.5},
    "Sugarcane":   {"base_yield": 30,  "optimal_temp": (22, 33), "optimal_humidity": (70, 85), "optimal_rain": (180, 300),"price_per_tonne": 3150,  "icon": "🎋", "harvest_months": 11},
    "Tomato":      {"base_yield": 12,  "optimal_temp": (18, 28), "optimal_humidity": (50, 70), "optimal_rain": (50, 80),  "price_per_tonne": 15000, "icon": "🍅", "harvest_months": 3.5},
    "Potato":      {"base_yield": 8,   "optimal_temp": (15, 22), "optimal_humidity": (60, 80), "optimal_rain": (50, 80),  "price_per_tonne": 12000, "icon": "🥔", "harvest_months": 3.5},
    "Groundnut":   {"base_yield": 0.6, "optimal_temp": (25, 35), "optimal_humidity": (50, 60), "optimal_rain": (50, 80),  "price_per_tonne": 56000, "icon": "🥜", "harvest_months": 4},
    "Mustard":     {"base_yield": 0.6, "optimal_temp": (10, 25), "optimal_humidity": (40, 60), "optimal_rain": (30, 50),  "price_per_tonne": 55000, "icon": "🌻", "harvest_months": 4},
    "Onion":       {"base_yield": 9,   "optimal_temp": (15, 25), "optimal_humidity": (50, 70), "optimal_rain": (40, 70),  "price_per_tonne": 18000, "icon": "🧅", "harvest_months": 4.5},
    "Chilli":      {"base_yield": 2.0, "optimal_temp": (20, 33), "optimal_humidity": (60, 70), "optimal_rain": (60, 120), "price_per_tonne": 120000,"icon": "🌶️", "harvest_months": 5},
    "Bell Pepper": {"base_yield": 6,   "optimal_temp": (18, 28), "optimal_humidity": (60, 70), "optimal_rain": (60, 100), "price_per_tonne": 30000, "icon": "🫑", "harvest_months": 3},
    "Pulses":      {"base_yield": 0.5, "optimal_temp": (18, 28), "optimal_humidity": (50, 65), "optimal_rain": (40, 70),  "price_per_tonne": 60000, "icon": "🫛", "harvest_months": 3.5},
}


def _calc_weather_factor(value, optimal_range):
    """Calculate a 0.0-1.0 factor based on how close value is to optimal range."""
    lo, hi = optimal_range
    if lo <= value <= hi:
        return 1.0
    elif value < lo:
        diff = lo - value
        return max(0.3, 1.0 - diff / (lo * 0.5 + 1))
    else:
        diff = value - hi
        return max(0.3, 1.0 - diff / (hi * 0.5 + 1))


def predict_yield(crop_name, area_acres, temperature, humidity, monthly_rainfall,
                  soil_quality=0.8, irrigation_access=True):
    """
    Predict yield for a given crop under current conditions.

    Returns dict with: expected_yield, yield_per_acre, harvest_date,
    expected_revenue, confidence, recommendations
    """
    crop = YIELD_DB.get(crop_name)
    if not crop:
        return None

    # Weather adjustment factors (0.3 - 1.0 each)
    temp_factor = _calc_weather_factor(temperature, crop["optimal_temp"])
    humid_factor = _calc_weather_factor(humidity, crop["optimal_humidity"])
    rain_factor = _calc_weather_factor(monthly_rainfall, crop["optimal_rain"])

    # Combined environmental factor
    env_factor = (temp_factor * 0.35 + humid_factor * 0.25 + rain_factor * 0.25 +
                  soil_quality * 0.15)

    # Irrigation bonus
    if irrigation_access:
        env_factor = min(1.0, env_factor * 1.1)

    # Yield calculation
    base = crop["base_yield"]
    estimated_yield_per_acre = base * env_factor
    total_yield = estimated_yield_per_acre * area_acres

    # Revenue
    price = crop["price_per_tonne"]
    revenue = total_yield * price

    # Harvest date
    harvest_date = datetime.now() + timedelta(days=int(crop["harvest_months"] * 30))

    # Confidence score
    confidence = min(0.95, (temp_factor + humid_factor + rain_factor) / 3 * 0.9 + 0.1)

    # Recommendations
    recs = []
    if temp_factor < 0.7:
        lo, hi = crop["optimal_temp"]
        if temperature < lo:
            recs.append(f"🌡️ Temperature ({temperature:.0f}°C) is below optimal ({lo}-{hi}°C). Consider greenhouse or delayed planting.")
        else:
            recs.append(f"🌡️ Temperature ({temperature:.0f}°C) exceeds optimal ({lo}-{hi}°C). Use shade nets and increase watering.")
    if humid_factor < 0.7:
        recs.append("💧 Humidity is outside optimal range. Adjust irrigation to maintain moisture.")
    if rain_factor < 0.7:
        if monthly_rainfall < crop["optimal_rain"][0]:
            recs.append("🌧️ Rainfall is insufficient. Ensure reliable irrigation infrastructure.")
        else:
            recs.append("🌧️ Excessive rainfall expected. Improve drainage to prevent waterlogging.")
    if not irrigation_access:
        recs.append("🚰 No irrigation access. Consider drip or rainwater harvesting systems.")
    if soil_quality < 0.6:
        recs.append("🪨 Soil quality is low. Apply organic matter, compost, and soil amendments.")
    if not recs:
        recs.append("✅ Conditions are favorable. Maintain current practices for optimal yield.")

    return {
        "crop_name": crop_name,
        "icon": crop["icon"],
        "area_acres": area_acres,
        "expected_yield": round(total_yield, 2),
        "yield_per_acre": round(estimated_yield_per_acre, 2),
        "expected_revenue": round(revenue),
        "harvest_date": harvest_date.strftime("%B %d, %Y"),
        "harvest_months": crop["harvest_months"],
        "confidence": round(confidence, 2),
        "price_per_tonne": price,
        "env_factor": round(env_factor, 2),
        "temp_factor": round(temp_factor, 2),
        "humid_factor": round(humid_factor, 2),
        "rain_factor": round(rain_factor, 2),
        "recommendations": recs,
    }


def render_yield_prediction(result):
    """Render yield prediction results as beautiful cards."""
    if not result:
        st.warning("Yield prediction unavailable for this crop.")
        return

    conf = result["confidence"]
    conf_color = "#2ecc71" if conf >= 0.8 else "#f39c12" if conf >= 0.6 else "#e74c3c"

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {conf_color};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
            <div>
                <span style="font-size:2rem;">{result['icon']}</span>
                <span style="font-size:1.3rem;font-weight:800;color:#fff;margin-left:0.5rem;">{result['crop_name']} Yield Forecast</span>
            </div>
            <span style="background:rgba(255,255,255,0.08);color:{conf_color};padding:5px 14px;border-radius:12px;font-size:0.85rem;font-weight:700;">
                {conf:.0%} Confidence</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;margin-bottom:1rem;">
            <div style="text-align:center;padding:0.8rem;background:rgba(255,255,255,0.04);border-radius:12px;">
                <div style="font-size:0.7rem;color:#8899aa;text-transform:uppercase;">Expected Yield</div>
                <div style="font-size:1.4rem;font-weight:800;color:#2ecc71;">📦 {result['expected_yield']:.1f} t</div>
            </div>
            <div style="text-align:center;padding:0.8rem;background:rgba(255,255,255,0.04);border-radius:12px;">
                <div style="font-size:0.7rem;color:#8899aa;text-transform:uppercase;">Yield / Acre</div>
                <div style="font-size:1.4rem;font-weight:800;color:#3498db;">📐 {result['yield_per_acre']:.2f} t</div>
            </div>
            <div style="text-align:center;padding:0.8rem;background:rgba(255,255,255,0.04);border-radius:12px;">
                <div style="font-size:0.7rem;color:#8899aa;text-transform:uppercase;">Expected Revenue</div>
                <div style="font-size:1.4rem;font-weight:800;color:#f39c12;">💰 ₹{result['expected_revenue']:,}</div>
            </div>
            <div style="text-align:center;padding:0.8rem;background:rgba(255,255,255,0.04);border-radius:12px;">
                <div style="font-size:0.7rem;color:#8899aa;text-transform:uppercase;">Harvest By</div>
                <div style="font-size:1.1rem;font-weight:700;color:#9b59b6;">🗓️ {result['harvest_date']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_yield_factors_chart(result):
    """Render a radar chart of environmental factors."""
    if not result:
        return
    categories = ["Temperature", "Humidity", "Rainfall", "Overall"]
    values = [result["temp_factor"], result["humid_factor"],
              result["rain_factor"], result["env_factor"]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill='toself', name='Suitability',
        line=dict(color='#2ecc71', width=2),
        fillcolor='rgba(46,204,113,0.2)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
        template="plotly_dark", height=350, margin=dict(t=30, b=30, l=60, r=60),
        title="🌍 Environmental Suitability Factors",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_yield_page(weather_current):
    """Full yield prediction page."""
    st.markdown("### 📊 Yield Prediction Engine")

    col1, col2, col3 = st.columns(3)
    with col1:
        crop = st.selectbox("🌱 Select Crop", list(YIELD_DB.keys()), key="yield_crop")
        area = st.number_input("🏞️ Land Area (acres)", min_value=0.1, max_value=10000.0, value=5.0, step=0.5, key="yield_area")
    with col2:
        temp = weather_current.get("temperature", 25) if weather_current else 25
        humidity = weather_current.get("humidity", 60) if weather_current else 60
        st.metric("🌡️ Current Temp", f"{temp:.1f}°C")
        st.metric("💧 Current Humidity", f"{humidity:.0f}%")
    with col3:
        monthly_rain = st.number_input("🌧️ Monthly Rainfall (mm)", min_value=0, max_value=500, value=80, step=10, key="yield_rain")
        soil_q = st.slider("🪨 Soil Quality", 0.1, 1.0, 0.8, 0.05, key="yield_soil")
        irrigation = st.checkbox("🚰 Irrigation Access", value=True, key="yield_irr")

    if st.button("📊 Predict Yield", type="primary", use_container_width=True, key="btn_yield"):
        with st.spinner("Calculating yield forecast..."):
            result = predict_yield(crop, area, temp, humidity, monthly_rain, soil_q, irrigation)

        if result:
            st.session_state["yield_result"] = result
            render_yield_prediction(result)

            c1, c2 = st.columns(2)
            with c1:
                render_yield_factors_chart(result)
            with c2:
                st.markdown("#### 📋 Recommendations to Improve Yield")
                for rec in result["recommendations"]:
                    st.markdown(f"- {rec}")

    elif "yield_result" in st.session_state:
        result = st.session_state["yield_result"]
        render_yield_prediction(result)
        c1, c2 = st.columns(2)
        with c1:
            render_yield_factors_chart(result)
        with c2:
            st.markdown("#### 📋 Recommendations to Improve Yield")
            for rec in result["recommendations"]:
                st.markdown(f"- {rec}")
