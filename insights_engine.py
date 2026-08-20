"""
AGRI-X AI — Agricultural Insights Engine (Module 8)
Generates intelligent farming insights from weather, location, and season data.
"""
import streamlit as st
from datetime import datetime


def _get_season_info():
    """Determine current season and related agricultural info."""
    month = datetime.now().month
    if month in [6, 7, 8, 9, 10]:
        return {
            "season": "Kharif", "period": "June – October",
            "icon": "🌧️", "color": "#3498db",
            "description": "Monsoon season. Ideal for rain-fed crops like rice, maize, cotton, soybean.",
            "suitable_crops": ["Rice", "Maize", "Cotton", "Soybean", "Groundnut", "Chilli", "Tomato", "Bell Pepper"],
        }
    elif month in [11, 12, 1, 2, 3]:
        return {
            "season": "Rabi", "period": "November – March",
            "icon": "❄️", "color": "#2ecc71",
            "description": "Winter season. Cool and dry — ideal for wheat, mustard, potato, onion.",
            "suitable_crops": ["Wheat", "Mustard", "Potato", "Onion", "Tomato", "Bell Pepper", "Sugarcane"],
        }
    else:
        return {
            "season": "Zaid", "period": "April – May",
            "icon": "☀️", "color": "#f39c12",
            "description": "Summer season. Short-duration crops with irrigation support.",
            "suitable_crops": ["Tomato", "Bell Pepper", "Onion", "Sugarcane"],
        }


def generate_weather_alerts(weather_current, daily_forecast=None):
    """Generate weather-based farming alerts."""
    alerts = []
    if not weather_current:
        return alerts

    temp = weather_current.get("temperature", 25)
    humidity = weather_current.get("humidity", 60)
    precipitation = weather_current.get("precipitation", 0)
    wind = weather_current.get("wind_speed", 0)
    uv = weather_current.get("uv_index", 0)

    # Temperature alerts
    if temp >= 40:
        alerts.append({"level": "Critical", "icon": "🔴", "color": "#DC143C",
                        "title": "Extreme Heat Alert",
                        "message": f"Temperature is {temp:.0f}°C. Provide shade for crops, increase irrigation frequency, and avoid mid-day field work."})
    elif temp >= 35:
        alerts.append({"level": "Warning", "icon": "🟠", "color": "#e74c3c",
                        "title": "Heat Stress Warning",
                        "message": f"Temperature is {temp:.0f}°C. Crops may experience heat stress. Increase watering and apply mulch."})
    elif temp <= 5:
        alerts.append({"level": "Critical", "icon": "🔴", "color": "#3498db",
                        "title": "Frost Alert",
                        "message": f"Temperature is {temp:.0f}°C. Risk of frost damage. Cover sensitive crops and avoid irrigation at night."})

    # Humidity alerts
    if humidity >= 90:
        alerts.append({"level": "Warning", "icon": "🟡", "color": "#f39c12",
                        "title": "High Humidity — Disease Risk",
                        "message": f"Humidity at {humidity}%. Fungal diseases (blight, mold) risk is elevated. Apply preventive fungicides."})
    elif humidity <= 25:
        alerts.append({"level": "Warning", "icon": "🟡", "color": "#e67e22",
                        "title": "Very Low Humidity",
                        "message": f"Humidity at {humidity}%. Spider mite risk increases. Mist plants and increase watering."})

    # Rainfall alerts
    if precipitation >= 20:
        alerts.append({"level": "Warning", "icon": "🟠", "color": "#9b59b6",
                        "title": "Heavy Rainfall",
                        "message": f"Precipitation at {precipitation:.1f}mm. Check field drainage. Avoid pesticide application."})

    # Wind alerts
    if wind >= 40:
        alerts.append({"level": "Warning", "icon": "🟠", "color": "#e74c3c",
                        "title": "High Wind Warning",
                        "message": f"Wind speed {wind:.0f} km/h. Stake tall crops, secure greenhouse covers."})

    # UV alerts
    if uv >= 9:
        alerts.append({"level": "Info", "icon": "🟡", "color": "#f39c12",
                        "title": "High UV Index",
                        "message": f"UV index {uv:.0f}. Protect workers and consider shade nets for sensitive crops."})

    # Multi-day rainfall check
    if daily_forecast:
        rain_days = sum(1 for d in daily_forecast if d.get("precipitation", 0) > 5)
        if rain_days >= 4:
            alerts.append({"level": "Warning", "icon": "🟠", "color": "#3498db",
                            "title": "Extended Wet Period",
                            "message": f"{rain_days} rainy days in forecast. High disease risk. Apply preventive fungicides now."})

    if not alerts:
        alerts.append({"level": "Info", "icon": "🟢", "color": "#2ecc71",
                        "title": "Favorable Conditions",
                        "message": "Current weather conditions are favorable for farming. Continue regular practices."})

    return alerts


def generate_farming_suggestions(weather_current, season_info):
    """Generate actionable farming suggestions."""
    suggestions = []
    if not weather_current:
        return suggestions

    temp = weather_current.get("temperature", 25)
    humidity = weather_current.get("humidity", 60)
    season = season_info["season"]

    # Irrigation suggestions
    if temp >= 30 and humidity < 50:
        suggestions.append({
            "category": "💧 Irrigation",
            "title": "Increase Watering Frequency",
            "detail": "Hot and dry conditions detected. Water crops early morning and late evening. Consider drip irrigation for efficiency.",
        })
    elif humidity >= 80:
        suggestions.append({
            "category": "💧 Irrigation",
            "title": "Reduce Irrigation",
            "detail": "High humidity detected. Reduce watering to prevent root rot and fungal growth. Ensure proper drainage.",
        })
    else:
        suggestions.append({
            "category": "💧 Irrigation",
            "title": "Maintain Regular Schedule",
            "detail": "Conditions are moderate. Maintain 1-1.5 inches per week. Morning watering preferred.",
        })

    # Fertilizer suggestions
    if season == "Kharif":
        suggestions.append({
            "category": "🌱 Fertilizer",
            "title": "Monsoon Fertilizer Application",
            "detail": "Apply nitrogen in split doses (not all at once) to prevent leaching during rains. Use urea with neem coating.",
        })
    elif season == "Rabi":
        suggestions.append({
            "category": "🌱 Fertilizer",
            "title": "Winter Nutrient Management",
            "detail": "Apply balanced NPK (20-20-20). Increase potash for cold resistance. Add zinc and boron micro-nutrients.",
        })
    else:
        suggestions.append({
            "category": "🌱 Fertilizer",
            "title": "Summer Nutrition",
            "detail": "Apply light doses of nitrogen and potash. Foliar spray for quick nutrient uptake. Maintain organic matter.",
        })

    # Pest management
    if temp >= 28 and humidity < 45:
        suggestions.append({
            "category": "🐛 Pest Management",
            "title": "Spider Mite Watch",
            "detail": "Hot, dry conditions favor spider mites. Scout undersides of leaves. Apply neem oil or abamectin if detected.",
        })
    elif humidity >= 75 and temp >= 20:
        suggestions.append({
            "category": "🐛 Pest Management",
            "title": "Fungal Disease Prevention",
            "detail": "Warm and humid — ideal for fungal diseases. Apply preventive mancozeb or copper-based sprays. Improve air circulation.",
        })

    # General
    suggestions.append({
        "category": "📋 General",
        "title": f"{season} Season Best Practices",
        "detail": f"Current season: {season}. Suitable crops: {', '.join(season_info['suitable_crops'][:5])}. Plan sowing accordingly.",
    })

    return suggestions


def render_insights_dashboard(weather_current, daily_forecast, location):
    """Render the full insights dashboard."""
    st.markdown('<h1 class="hero-title fade-in">🧠 Agricultural Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub fade-in">Intelligent farming insights powered by weather, location, and seasonal analysis</p>', unsafe_allow_html=True)

    season_info = _get_season_info()

    # Season card
    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {season_info['color']};">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.8rem;">{season_info['icon']}</span>
                <span style="font-size:1.2rem;font-weight:700;color:#fff;margin-left:0.5rem;">{season_info['season']} Season</span>
                <span style="color:#8899aa;font-size:0.85rem;margin-left:0.5rem;">({season_info['period']})</span>
            </div>
            <span style="color:{season_info['color']};font-size:0.85rem;font-weight:600;">📍 {location.get('city', 'Unknown')}, {location.get('state', '')}</span>
        </div>
        <p style="color:#ccd;font-size:0.9rem;margin:0.8rem 0 0.5rem 0;line-height:1.5;">{season_info['description']}</p>
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
            {"".join(f'<span style="background:rgba(255,255,255,0.06);color:#2ecc71;padding:3px 10px;border-radius:8px;font-size:0.8rem;">{c}</span>' for c in season_info['suitable_crops'])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Weather alerts
    st.markdown("### ⚠️ Weather Alerts")
    alerts = generate_weather_alerts(weather_current, daily_forecast)
    for alert in alerts:
        if alert["level"] == "Critical":
            st.error(f"{alert['icon']} **{alert['title']}** — {alert['message']}")
        elif alert["level"] == "Warning":
            st.warning(f"{alert['icon']} **{alert['title']}** — {alert['message']}")
        else:
            st.info(f"{alert['icon']} **{alert['title']}** — {alert['message']}")

    # Farming suggestions
    st.markdown("### 🌾 Farming Suggestions")
    suggestions = generate_farming_suggestions(weather_current, season_info)
    for sug in suggestions:
        with st.expander(f"{sug['category']} — {sug['title']}", expanded=False):
            st.write(sug["detail"])

    # Yield potential — uses unified crop database and full location data
    st.markdown("### 📊 Crop Yield Potential")
    if weather_current:
        temp = weather_current.get("temperature", 25)
        humidity = weather_current.get("humidity", 60)
        from crop_advisor import UNIFIED_CROP_DB, calculate_soil_quality, estimate_annual_rainfall
        from yield_predictor import _calc_weather_factor

        # Get soil quality from location
        soil_data = location.get("soil_data") if location else None
        soil_quality, _ = calculate_soil_quality(soil_data)
        soil_type = location.get("soil_type", "Loamy") if location else "Loamy"
        elevation = location.get("elevation") if location else None

        potentials = []
        for name, crop in UNIFIED_CROP_DB.items():
            if name in season_info["suitable_crops"]:
                tf = _calc_weather_factor(temp, crop["optimal_temp"])
                hf = _calc_weather_factor(humidity, crop["optimal_humidity"])
                # Include soil match in suitability
                soil_match = 1.0 if any(
                    s.lower() in soil_type.lower() or soil_type.lower() in s.lower()
                    for s in crop.get("soil", [])
                ) else 0.5
                overall = (tf * 0.35 + hf * 0.25 + soil_quality * 0.20 + soil_match * 0.20)
                potentials.append((name, crop["icon"], overall))
        potentials.sort(key=lambda x: x[2], reverse=True)

        if potentials:
            cols = st.columns(min(len(potentials), 4))
            for i, (name, icon, score) in enumerate(potentials[:4]):
                sc = score * 100
                color = "#2ecc71" if sc >= 75 else "#f39c12" if sc >= 50 else "#e74c3c"
                with cols[i]:
                    st.markdown(f"""
                    <div style="text-align:center;padding:1rem;background:rgba(255,255,255,0.03);border-radius:12px;border:1px solid rgba(255,255,255,0.06);">
                        <div style="font-size:1.8rem;">{icon}</div>
                        <div style="font-size:0.9rem;font-weight:600;color:#fff;margin:0.3rem 0;">{name}</div>
                        <div style="font-size:1.2rem;font-weight:800;color:{color};">{sc:.0f}%</div>
                        <div style="font-size:0.7rem;color:#8899aa;">Suitability</div>
                    </div>
                    """, unsafe_allow_html=True)

