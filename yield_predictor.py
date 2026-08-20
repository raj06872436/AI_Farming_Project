"""
AGRI-X AI — Yield Prediction Engine (Module 7) v3.0
Location-aware yield prediction using unified crop database.
Auto-fills soil, rainfall, and weather data — no redundant manual inputs.
"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import unified database and utilities from crop_advisor
from crop_advisor import (
    UNIFIED_CROP_DB, calculate_soil_quality, estimate_annual_rainfall,
)


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
                  soil_quality=0.8, irrigation_access=True, elevation=None):
    """
    Predict yield for a given crop under current conditions.
    Uses the unified crop database from crop_advisor.

    Returns dict with: expected_yield, yield_per_acre, harvest_date,
    expected_revenue, confidence, recommendations
    """
    crop = UNIFIED_CROP_DB.get(crop_name)
    if not crop:
        return None

    # Weather adjustment factors (0.3 - 1.0 each)
    temp_factor = _calc_weather_factor(temperature, crop["optimal_temp"])
    humid_factor = _calc_weather_factor(humidity, crop["optimal_humidity"])
    rain_factor = _calc_weather_factor(monthly_rainfall, crop["optimal_monthly_rain"])

    # Combined environmental factor
    env_factor = (temp_factor * 0.30 + humid_factor * 0.20 + rain_factor * 0.20 +
                  soil_quality * 0.20)

    # Irrigation bonus
    if irrigation_access:
        env_factor = min(1.0, env_factor * 1.1)

    # Elevation adjustment
    if elevation is not None:
        if crop_name in ["Rice", "Sugarcane"] and elevation > 1500:
            env_factor *= 0.7  # Penalize high-altitude for lowland crops
        elif crop_name in ["Wheat", "Potato", "Mustard"] and 300 < elevation < 1800:
            env_factor = min(1.0, env_factor * 1.05)  # Slight bonus

    # Yield calculation
    base = crop["yield_per_acre"]
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
        if monthly_rainfall < crop["optimal_monthly_rain"][0]:
            recs.append("🌧️ Rainfall is insufficient. Ensure reliable irrigation infrastructure.")
        else:
            recs.append("🌧️ Excessive rainfall expected. Improve drainage to prevent waterlogging.")
    if not irrigation_access:
        recs.append("🚰 No irrigation access. Consider drip or rainwater harvesting systems.")
    if soil_quality < 0.6:
        recs.append("🪨 Soil quality is low. Apply organic matter, compost, and soil amendments.")
    if elevation is not None and elevation > 2000:
        recs.append(f"⛰️ High elevation ({elevation:.0f}m) may limit growing season. Choose cold-tolerant varieties.")
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
        "soil_quality": round(soil_quality, 2),
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
    categories = ["Temperature", "Humidity", "Rainfall", "Soil Quality", "Overall"]
    values = [result["temp_factor"], result["humid_factor"],
              result["rain_factor"], result["soil_quality"], result["env_factor"]]
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


def render_yield_page(location, weather_current, weather_raw):
    """
    Full yield prediction page — location-aware, auto-fills everything.
    Only asks for: crop, land area, irrigation.
    """
    st.markdown("### 📊 Yield Prediction Engine")

    # ── Get all parameters from location & weather (no manual input needed) ──
    temp = weather_current.get("temperature", 25) if weather_current else 25
    humidity = weather_current.get("humidity", 60) if weather_current else 60
    elevation = location.get("elevation") if location else None

    # Soil quality from real ISRIC data
    soil_data = location.get("soil_data") if location else None
    soil_quality, soil_breakdown = calculate_soil_quality(soil_data)

    # Monthly rainfall from best available estimate
    rainfall_est, rainfall_conf, rainfall_method = estimate_annual_rainfall(location, weather_raw)
    monthly_rain = rainfall_est / 12  # Convert annual to approximate monthly

    # Show auto-detected conditions as read-only metrics
    st.markdown("""
    <div class="glass-card" style="border-left:4px solid #2ecc71;margin-bottom:1rem;">
        <div style="font-size:0.85rem;font-weight:600;color:#2ecc71;margin-bottom:0.5rem;">
            ✅ All environmental data auto-detected from your location & weather
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌡️ Temperature", f"{temp:.1f}°C")
    m2.metric("💧 Humidity", f"{humidity:.0f}%")
    m3.metric("🌧️ Est. Monthly Rain", f"{monthly_rain:.0f} mm")
    m4.metric("🧪 Soil Quality", f"{soil_quality:.0%}")

    # ── Only ask what the system can't know ──
    # Pre-select crop from crop advisor if user clicked "Predict Yield"
    selected = st.session_state.get("selected_yield_crop")
    crop_names = list(UNIFIED_CROP_DB.keys())
    default_idx = 0
    if selected and selected in crop_names:
        default_idx = crop_names.index(selected)

    col1, col2, col3 = st.columns(3)
    with col1:
        crop = st.selectbox("🌱 Select Crop", crop_names, index=default_idx, key="yield_crop")
    with col2:
        # Pre-fill area from crop advisor if available
        crop_params = st.session_state.get("crop_params", {})
        default_area = crop_params.get("acres", 5.0)
        area = st.number_input(
            "🏞️ Land Area (acres)", min_value=0.1, max_value=10000.0,
            value=default_area, step=0.5, key="yield_area",
        )
    with col3:
        irrigation = st.checkbox("🚰 Irrigation Access", value=True, key="yield_irr")

    if st.button("📊 Predict Yield", type="primary", use_container_width=True, key="btn_yield"):
        with st.spinner("Calculating yield forecast..."):
            result = predict_yield(
                crop, area, temp, humidity, monthly_rain,
                soil_quality, irrigation, elevation,
            )

        if result:
            st.session_state["yield_result"] = result
            # Clear the one-shot trigger
            if "selected_yield_crop" in st.session_state:
                del st.session_state["selected_yield_crop"]

    # Persistent render
    result = st.session_state.get("yield_result")
    if result:
        render_yield_prediction(result)

        c1, c2 = st.columns(2)
        with c1:
            render_yield_factors_chart(result)
        with c2:
            st.markdown("#### 📋 Recommendations to Improve Yield")
            for rec in result["recommendations"]:
                st.markdown(f"- {rec}")

            # Show data sources
            st.markdown("---")
            st.markdown("##### 📡 Data Sources")
            soil_src = "ISRIC SoilGrids API" if soil_data else "Default estimate"
            st.caption(f"""
            🌡️ Weather: Open-Meteo API (real-time)
            🪨 Soil Quality: {soil_src} (pH={soil_breakdown.get('ph_value','—')}, OC={soil_breakdown.get('soc_value','—')} g/kg)
            🌧️ Rainfall: {rainfall_method} (~{rainfall_est} mm/year → ~{monthly_rain:.0f} mm/month)
            ⛰️ Elevation: {f'{elevation:.0f}m' if elevation else 'Unavailable'}
            """)
