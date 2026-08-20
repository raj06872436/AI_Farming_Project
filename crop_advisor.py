"""
AGRI-X AI — Crop Recommendation & Revenue Forecaster (Module 12) v3.0
Unified crop database + location-aware auto-fill + integrated yield prediction.
"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Unified Crop Database (merged CROP_DB + YIELD_DB) ──
# Every crop has: season, temp/humidity/rainfall ranges, soil types,
# yield per acre (tonnes), harvest info, pricing, cost, water needs.
UNIFIED_CROP_DB = {
    "Rice": {
        "season": ["Kharif"], "temp_min": 20, "temp_max": 35,
        "humidity_min": 60, "humidity_max": 80,
        "rainfall_min": 1000, "rainfall_max": 2000,
        "soil": ["Alluvial", "Clay", "Loamy", "Silt Loam"],
        "yield_per_acre": 1.2, "yield_max_per_acre": 1.5,
        "optimal_temp": (22, 32), "optimal_humidity": (65, 80),
        "optimal_monthly_rain": (150, 250),
        "price_per_tonne": 22000, "price_min_qt": 2000, "price_max_qt": 2400,
        "msp": 2183, "cost_per_acre": 18000, "duration_days": 120,
        "harvest_months": 4, "water_need": "High", "icon": "🌾",
    },
    "Wheat": {
        "season": ["Rabi"], "temp_min": 10, "temp_max": 25,
        "humidity_min": 50, "humidity_max": 70,
        "rainfall_min": 400, "rainfall_max": 700,
        "soil": ["Alluvial", "Loamy", "Clay", "Silt Loam"],
        "yield_per_acre": 1.2, "yield_max_per_acre": 1.8,
        "optimal_temp": (12, 22), "optimal_humidity": (50, 65),
        "optimal_monthly_rain": (50, 90),
        "price_per_tonne": 23000, "price_min_qt": 2100, "price_max_qt": 2500,
        "msp": 2275, "cost_per_acre": 16000, "duration_days": 135,
        "harvest_months": 4.5, "water_need": "Moderate", "icon": "🌾",
    },
    "Tomato": {
        "season": ["Kharif", "Rabi"], "temp_min": 18, "temp_max": 30,
        "humidity_min": 50, "humidity_max": 80,
        "rainfall_min": 400, "rainfall_max": 600,
        "soil": ["Loamy", "Sandy", "Sandy Loam", "Loam"],
        "yield_per_acre": 10, "yield_max_per_acre": 15,
        "optimal_temp": (18, 28), "optimal_humidity": (50, 70),
        "optimal_monthly_rain": (50, 80),
        "price_per_tonne": 15000, "price_min_qt": 800, "price_max_qt": 2500,
        "msp": 0, "cost_per_acre": 45000, "duration_days": 100,
        "harvest_months": 3.5, "water_need": "Moderate", "icon": "🍅",
    },
    "Potato": {
        "season": ["Rabi"], "temp_min": 15, "temp_max": 25,
        "humidity_min": 60, "humidity_max": 80,
        "rainfall_min": 500, "rainfall_max": 800,
        "soil": ["Loamy", "Sandy", "Alluvial", "Sandy Loam"],
        "yield_per_acre": 6, "yield_max_per_acre": 10,
        "optimal_temp": (15, 22), "optimal_humidity": (60, 80),
        "optimal_monthly_rain": (50, 80),
        "price_per_tonne": 12000, "price_min_qt": 600, "price_max_qt": 1500,
        "msp": 0, "cost_per_acre": 35000, "duration_days": 100,
        "harvest_months": 3.5, "water_need": "Moderate", "icon": "🥔",
    },
    "Bell Pepper": {
        "season": ["Kharif", "Rabi"], "temp_min": 18, "temp_max": 28,
        "humidity_min": 60, "humidity_max": 70,
        "rainfall_min": 600, "rainfall_max": 1200,
        "soil": ["Loamy", "Sandy", "Sandy Loam", "Loam"],
        "yield_per_acre": 4, "yield_max_per_acre": 8,
        "optimal_temp": (18, 28), "optimal_humidity": (60, 70),
        "optimal_monthly_rain": (60, 100),
        "price_per_tonne": 30000, "price_min_qt": 1500, "price_max_qt": 4000,
        "msp": 0, "cost_per_acre": 55000, "duration_days": 90,
        "harvest_months": 3, "water_need": "Moderate", "icon": "🫑",
    },
    "Cotton": {
        "season": ["Kharif"], "temp_min": 25, "temp_max": 35,
        "humidity_min": 50, "humidity_max": 65,
        "rainfall_min": 500, "rainfall_max": 1000,
        "soil": ["Clay", "Alluvial", "Clay Loam"],
        "yield_per_acre": 0.6, "yield_max_per_acre": 0.8,
        "optimal_temp": (25, 35), "optimal_humidity": (50, 65),
        "optimal_monthly_rain": (70, 120),
        "price_per_tonne": 65000, "price_min_qt": 6000, "price_max_qt": 7500,
        "msp": 6620, "cost_per_acre": 22000, "duration_days": 165,
        "harvest_months": 5.5, "water_need": "Moderate", "icon": "🏵️",
    },
    "Sugarcane": {
        "season": ["Kharif", "Rabi"], "temp_min": 20, "temp_max": 35,
        "humidity_min": 70, "humidity_max": 85,
        "rainfall_min": 1500, "rainfall_max": 2500,
        "soil": ["Alluvial", "Loamy", "Clay", "Loam"],
        "yield_per_acre": 25, "yield_max_per_acre": 35,
        "optimal_temp": (22, 33), "optimal_humidity": (70, 85),
        "optimal_monthly_rain": (180, 300),
        "price_per_tonne": 3150, "price_min_qt": 290, "price_max_qt": 350,
        "msp": 315, "cost_per_acre": 40000, "duration_days": 330,
        "harvest_months": 11, "water_need": "Very High", "icon": "🎋",
    },
    "Maize": {
        "season": ["Kharif"], "temp_min": 20, "temp_max": 30,
        "humidity_min": 55, "humidity_max": 75,
        "rainfall_min": 500, "rainfall_max": 1000,
        "soil": ["Loamy", "Alluvial", "Loam", "Sandy Loam"],
        "yield_per_acre": 1.0, "yield_max_per_acre": 1.5,
        "optimal_temp": (21, 30), "optimal_humidity": (55, 75),
        "optimal_monthly_rain": (60, 120),
        "price_per_tonne": 21000, "price_min_qt": 1900, "price_max_qt": 2300,
        "msp": 2090, "cost_per_acre": 14000, "duration_days": 100,
        "harvest_months": 3.5, "water_need": "Moderate", "icon": "🌽",
    },
    "Soybean": {
        "season": ["Kharif"], "temp_min": 20, "temp_max": 30,
        "humidity_min": 60, "humidity_max": 70,
        "rainfall_min": 600, "rainfall_max": 1000,
        "soil": ["Clay", "Loamy", "Clay Loam"],
        "yield_per_acre": 0.4, "yield_max_per_acre": 0.6,
        "optimal_temp": (20, 30), "optimal_humidity": (60, 70),
        "optimal_monthly_rain": (60, 100),
        "price_per_tonne": 46000, "price_min_qt": 4200, "price_max_qt": 5000,
        "msp": 4600, "cost_per_acre": 15000, "duration_days": 100,
        "harvest_months": 3.5, "water_need": "Moderate", "icon": "🫘",
    },
    "Mustard": {
        "season": ["Rabi"], "temp_min": 10, "temp_max": 25,
        "humidity_min": 40, "humidity_max": 60,
        "rainfall_min": 300, "rainfall_max": 500,
        "soil": ["Loamy", "Sandy", "Alluvial", "Sandy Loam"],
        "yield_per_acre": 0.5, "yield_max_per_acre": 0.8,
        "optimal_temp": (10, 25), "optimal_humidity": (40, 60),
        "optimal_monthly_rain": (30, 50),
        "price_per_tonne": 55000, "price_min_qt": 5000, "price_max_qt": 6500,
        "msp": 5650, "cost_per_acre": 12000, "duration_days": 120,
        "harvest_months": 4, "water_need": "Low", "icon": "🌻",
    },
    "Onion": {
        "season": ["Rabi"], "temp_min": 15, "temp_max": 25,
        "humidity_min": 50, "humidity_max": 70,
        "rainfall_min": 350, "rainfall_max": 600,
        "soil": ["Loamy", "Sandy", "Alluvial", "Sandy Loam", "Loam"],
        "yield_per_acre": 6, "yield_max_per_acre": 12,
        "optimal_temp": (15, 25), "optimal_humidity": (50, 70),
        "optimal_monthly_rain": (40, 70),
        "price_per_tonne": 18000, "price_min_qt": 800, "price_max_qt": 3000,
        "msp": 0, "cost_per_acre": 40000, "duration_days": 130,
        "harvest_months": 4.5, "water_need": "Moderate", "icon": "🧅",
    },
    "Chilli": {
        "season": ["Kharif"], "temp_min": 20, "temp_max": 35,
        "humidity_min": 60, "humidity_max": 70,
        "rainfall_min": 500, "rainfall_max": 1200,
        "soil": ["Loamy", "Clay", "Clay Loam", "Loam"],
        "yield_per_acre": 1.5, "yield_max_per_acre": 3.0,
        "optimal_temp": (20, 33), "optimal_humidity": (60, 70),
        "optimal_monthly_rain": (60, 120),
        "price_per_tonne": 120000, "price_min_qt": 8000, "price_max_qt": 15000,
        "msp": 0, "cost_per_acre": 50000, "duration_days": 150,
        "harvest_months": 5, "water_need": "Moderate", "icon": "🌶️",
    },
    "Groundnut": {
        "season": ["Kharif"], "temp_min": 25, "temp_max": 35,
        "humidity_min": 50, "humidity_max": 60,
        "rainfall_min": 500, "rainfall_max": 800,
        "soil": ["Sandy", "Sandy Loam", "Loamy", "Loam"],
        "yield_per_acre": 0.5, "yield_max_per_acre": 0.7,
        "optimal_temp": (25, 35), "optimal_humidity": (50, 60),
        "optimal_monthly_rain": (50, 80),
        "price_per_tonne": 56000, "price_min_qt": 5000, "price_max_qt": 6500,
        "msp": 5850, "cost_per_acre": 18000, "duration_days": 120,
        "harvest_months": 4, "water_need": "Moderate", "icon": "🥜",
    },
    "Pulses": {
        "season": ["Rabi", "Kharif"], "temp_min": 18, "temp_max": 28,
        "humidity_min": 50, "humidity_max": 65,
        "rainfall_min": 400, "rainfall_max": 700,
        "soil": ["Loamy", "Clay", "Alluvial", "Loam"],
        "yield_per_acre": 0.4, "yield_max_per_acre": 0.6,
        "optimal_temp": (18, 28), "optimal_humidity": (50, 65),
        "optimal_monthly_rain": (40, 70),
        "price_per_tonne": 60000, "price_min_qt": 5500, "price_max_qt": 7000,
        "msp": 6600, "cost_per_acre": 14000, "duration_days": 105,
        "harvest_months": 3.5, "water_need": "Low", "icon": "🫛",
    },
}

# ── Average Annual Rainfall by Indian State (mm) ──
# Used as a much better fallback than extrapolating 7-day forecast × 52
STATE_RAINFALL_AVG = {
    "Andhra Pradesh": 940, "Arunachal Pradesh": 2780, "Assam": 2820,
    "Bihar": 1210, "Chhattisgarh": 1290, "Goa": 3005,
    "Gujarat": 820, "Haryana": 570, "Himachal Pradesh": 1580,
    "Jharkhand": 1300, "Karnataka": 1140, "Kerala": 3055,
    "Madhya Pradesh": 1100, "Maharashtra": 1060, "Manipur": 1480,
    "Meghalaya": 2820, "Mizoram": 2500, "Nagaland": 2000,
    "Odisha": 1490, "Punjab": 550, "Rajasthan": 530,
    "Sikkim": 2740, "Tamil Nadu": 960, "Telangana": 950,
    "Tripura": 2100, "Uttar Pradesh": 990, "Uttarakhand": 1550,
    "West Bengal": 1750, "Delhi": 790, "Jammu and Kashmir": 1100,
    "Ladakh": 100,
}

SOIL_TYPES = ["Alluvial", "Clay", "Loamy", "Sandy", "Sandy Loam",
              "Silt Loam", "Clay Loam", "Loam", "Silty", "Laterite"]


def _get_current_season():
    """Determine Indian agricultural season from current month."""
    month = datetime.now().month
    if month in [6, 7, 8, 9, 10]:
        return "Kharif"
    elif month in [11, 12, 1, 2, 3]:
        return "Rabi"
    else:
        return "Zaid"


def estimate_annual_rainfall(location, weather_raw):
    """
    Estimate annual rainfall using best available data.
    Priority: State average → forecast-supplemented state average → raw extrapolation.
    Returns (estimate_mm, confidence, method).
    """
    state = location.get("state", "") if location else ""
    state_avg = None

    # Try state-level average
    for key, val in STATE_RAINFALL_AVG.items():
        if key.lower() in state.lower() or state.lower() in key.lower():
            state_avg = val
            break

    # Get weekly precipitation from forecast
    weekly_precip = 0
    if weather_raw and "daily" in weather_raw:
        daily = weather_raw["daily"]
        precip_list = daily.get("precipitation_sum", [])
        weekly_precip = sum(precip_list) if precip_list else 0

    if state_avg:
        # Use state average, lightly adjusted by current forecast trend
        # If forecast week is much wetter/drier than average weekly, adjust ±15%
        avg_weekly = state_avg / 52
        if avg_weekly > 0 and weekly_precip > 0:
            ratio = weekly_precip / avg_weekly
            adjustment = max(0.85, min(1.15, ratio))
            estimate = int(state_avg * adjustment)
        else:
            estimate = state_avg
        return max(100, min(5000, estimate)), "high", f"State average ({state})"

    if weekly_precip > 0:
        # No state data — use seasonal extrapolation (still better than ×52)
        month = datetime.now().month
        # Monsoon months (Jun–Sep) contribute ~75% of annual rainfall
        if month in [6, 7, 8, 9]:
            annual = weekly_precip * 52 * 0.6  # discount monsoon bias
        elif month in [12, 1, 2, 3]:
            annual = weekly_precip * 52 * 2.5  # dry season, scale up
        else:
            annual = weekly_precip * 52
        return max(100, min(5000, int(annual))), "low", "Forecast extrapolation"

    return 800, "fallback", "Default estimate"


def calculate_soil_quality(soil_data):
    """
    Calculate a 0-1 soil quality score from ISRIC SoilGrids data.
    Considers pH, organic carbon, and clay/sand/silt balance.
    Returns (score, breakdown_dict).
    """
    if not soil_data:
        return 0.6, {"ph": 0.5, "organic": 0.5, "texture": 0.7, "overall": 0.6}

    # pH score (6.0-7.5 optimal for most crops)
    ph = soil_data.get("ph", {}).get("value", 6.5)
    if 6.0 <= ph <= 7.5:
        ph_score = 1.0
    elif 5.5 <= ph <= 8.0:
        ph_score = 0.7
    elif 5.0 <= ph <= 8.5:
        ph_score = 0.4
    else:
        ph_score = 0.2

    # Organic carbon score (higher is better for fertility)
    soc = soil_data.get("organic_carbon", {}).get("value", 10)
    if soc >= 25:
        oc_score = 1.0
    elif soc >= 15:
        oc_score = 0.8
    elif soc >= 8:
        oc_score = 0.6
    elif soc >= 4:
        oc_score = 0.4
    else:
        oc_score = 0.2

    # Texture balance score (loamy = balanced = best)
    clay = soil_data.get("clay", {}).get("value", 25)
    sand = soil_data.get("sand", {}).get("value", 40)
    silt = soil_data.get("silt", {}).get("value", 35)
    # Ideal loam: ~20-30% clay, ~30-50% sand, ~30-50% silt
    clay_dev = abs(clay - 25) / 25
    sand_dev = abs(sand - 40) / 40
    silt_dev = abs(silt - 35) / 35
    texture_score = max(0.2, 1.0 - (clay_dev + sand_dev + silt_dev) / 3)

    # Weighted overall
    overall = ph_score * 0.35 + oc_score * 0.30 + texture_score * 0.35
    overall = round(max(0.1, min(1.0, overall)), 2)

    return overall, {
        "ph": round(ph_score, 2),
        "organic": round(oc_score, 2),
        "texture": round(texture_score, 2),
        "overall": overall,
        "ph_value": ph,
        "soc_value": soc,
    }


def _match_soil_type(detected_soil, crop_soils):
    """Check if the detected soil type matches any of the crop's suitable soils."""
    if not detected_soil:
        return False
    detected_lower = detected_soil.lower()
    for cs in crop_soils:
        if cs.lower() in detected_lower or detected_lower in cs.lower():
            return True
    return False


def _score_crop(crop_name, crop, temp, humidity, annual_rainfall, soil, season, elevation=None):
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

    # Rainfall (15 pts)
    if crop["rainfall_min"] <= annual_rainfall <= crop["rainfall_max"]:
        score += 15
        reasons.append("✅ Rainfall adequate")
    elif annual_rainfall >= crop["rainfall_min"] * 0.7:
        score += 8
        reasons.append("⚠️ Rainfall marginal (irrigation may be needed)")
    else:
        score += 3
        reasons.append("❌ Rainfall insufficient — heavy irrigation required")

    # Soil match (15 pts)
    if _match_soil_type(soil, crop["soil"]):
        score += 15
        reasons.append(f"✅ {soil} soil is ideal")
    else:
        score += 5
        reasons.append(f"⚠️ {soil} soil — not ideal but possible")

    # Elevation bonus/penalty (if available)
    if elevation is not None:
        if crop_name in ["Rice", "Sugarcane"] and elevation > 1500:
            score -= 10
            reasons.append(f"⚠️ Elevation {elevation:.0f}m is too high for {crop_name}")
        elif crop_name in ["Wheat", "Potato", "Mustard"] and elevation > 500 and elevation < 2000:
            score += 5
            reasons.append(f"✅ Mid-altitude ({elevation:.0f}m) suits {crop_name}")

    return min(100, max(0, score)), reasons


def recommend_crops(temp, humidity, annual_rainfall, soil, season, land_area_acres, elevation=None):
    """
    Recommend crops ranked by suitability score.
    Returns list of dicts with scores, yields, revenue, profit.
    """
    results = []
    for name, crop in UNIFIED_CROP_DB.items():
        score, reasons = _score_crop(
            name, crop, temp, humidity, annual_rainfall, soil, season, elevation
        )
        # Only recommend if score > 20
        if score < 20:
            continue

        yield_min = crop["yield_per_acre"] * land_area_acres
        yield_max = crop["yield_max_per_acre"] * land_area_acres
        # Revenue in ₹ (price is per quintal = 100 kg, yield is tonnes)
        rev_min = yield_min * 10 * crop["price_min_qt"]  # tonnes → quintals × price
        rev_max = yield_max * 10 * crop["price_max_qt"]
        total_cost = crop["cost_per_acre"] * land_area_acres
        profit_min = rev_min - total_cost
        profit_max = rev_max - total_cost

        results.append({
            "name": name, "icon": crop["icon"], "score": score, "reasons": reasons,
            "season": ", ".join(crop["season"]), "duration": crop["duration_days"],
            "water_need": crop["water_need"],
            "yield_min": yield_min, "yield_max": yield_max,
            "yield_per_acre": crop["yield_per_acre"],
            "yield_max_per_acre": crop["yield_max_per_acre"],
            "price_min_qt": crop["price_min_qt"], "price_max_qt": crop["price_max_qt"],
            "price_per_tonne": crop["price_per_tonne"],
            "msp": crop["msp"],
            "revenue_min": rev_min, "revenue_max": rev_max,
            "total_cost": total_cost,
            "profit_min": profit_min, "profit_max": profit_max,
            "cost_per_acre": crop["cost_per_acre"],
            "harvest_months": crop["harvest_months"],
            "optimal_temp": crop["optimal_temp"],
            "optimal_humidity": crop["optimal_humidity"],
            "optimal_monthly_rain": crop["optimal_monthly_rain"],
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


def _render_auto_detected_summary(city, temp, humidity, soil_type, soil_quality_score,
                                   rainfall_est, rainfall_confidence, rainfall_method,
                                   season, elevation):
    """Render a summary card showing all auto-detected parameters."""
    conf_color = {"high": "#2ecc71", "low": "#f39c12", "fallback": "#e74c3c"}.get(rainfall_confidence, "#888")
    elev_text = f"{elevation:.0f}m" if elevation is not None else "—"

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid #3498db;margin-bottom:1rem;">
        <div style="font-size:1rem;font-weight:700;color:#fff;margin-bottom:0.8rem;">
            🌐 Auto-Detected Farm Conditions
            <span style="font-size:0.7rem;color:#8899aa;margin-left:0.5rem;">All data from your location & weather APIs</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.6rem;">
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.6rem;color:#8899aa;text-transform:uppercase;">📍 Location</div>
                <div style="font-size:0.95rem;font-weight:700;color:#3498db;">{city}</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.6rem;color:#8899aa;text-transform:uppercase;">🌡️ Temperature</div>
                <div style="font-size:0.95rem;font-weight:700;color:#e74c3c;">{temp:.1f}°C</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.6rem;color:#8899aa;text-transform:uppercase;">💧 Humidity</div>
                <div style="font-size:0.95rem;font-weight:700;color:#3498db;">{humidity:.0f}%</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.6rem;color:#8899aa;text-transform:uppercase;">🗓️ Season</div>
                <div style="font-size:0.95rem;font-weight:700;color:#f39c12;">{season}</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.6rem;color:#8899aa;text-transform:uppercase;">🪨 Soil Type</div>
                <div style="font-size:0.95rem;font-weight:700;color:#CD853F;">{soil_type}</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.6rem;color:#8899aa;text-transform:uppercase;">🧪 Soil Quality</div>
                <div style="font-size:0.95rem;font-weight:700;color:#2ecc71;">{soil_quality_score:.0%}</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.6rem;color:#8899aa;text-transform:uppercase;">🌧️ Annual Rain</div>
                <div style="font-size:0.95rem;font-weight:700;color:{conf_color};">~{rainfall_est} mm</div>
                <div style="font-size:0.55rem;color:#8899aa;">{rainfall_method}</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.6rem;color:#8899aa;text-transform:uppercase;">⛰️ Elevation</div>
                <div style="font-size:0.95rem;font-weight:700;color:#9b59b6;">{elev_text}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_crop_advisor_page(location, weather_current, weather_raw=None):
    """Full crop advisor page renderer — unified, location-aware."""
    st.markdown("### 🌾 Smart Crop Recommendations")

    # ── Auto-detect all parameters from location & weather ──
    temp = weather_current.get("temperature", 25) if weather_current else 25
    humidity = weather_current.get("humidity", 60) if weather_current else 60
    city_name = location.get("city", "Unknown") if location else "Unknown"
    elevation = location.get("elevation") if location else None

    # Soil from location
    detected_soil = location.get("soil_type", "Loamy") if location else "Loamy"
    soil_data = location.get("soil_data") if location else None
    soil_quality, soil_breakdown = calculate_soil_quality(soil_data)

    # Rainfall from state average + forecast
    rainfall_est, rainfall_conf, rainfall_method = estimate_annual_rainfall(location, weather_raw)

    # Season
    auto_season = _get_current_season()

    # ── Show auto-detected summary ──
    _render_auto_detected_summary(
        city_name, temp, humidity, detected_soil, soil_quality,
        rainfall_est, rainfall_conf, rainfall_method, auto_season, elevation,
    )

    # ── Only ask for what the system CANNOT know ──
    col1, col2 = st.columns(2)
    with col1:
        land_area = st.number_input(
            "🏞️ Land Area", min_value=0.1, max_value=10000.0,
            value=5.0, step=0.5, key="crop_land",
        )
        unit = st.selectbox("Unit", ["Acres", "Hectares", "Bigha"], key="crop_unit")
    with col2:
        irrigation = st.checkbox("🚰 Irrigation Access", value=True, key="crop_irrigation")

    # ── Optional overrides (collapsed by default) ──
    with st.expander("⚙️ Override Auto-Detected Parameters", expanded=False):
        st.caption("These values are auto-detected from your location. Only change if you have more accurate data.")
        ov1, ov2, ov3 = st.columns(3)
        with ov1:
            soil_index = 0
            for idx, s in enumerate(SOIL_TYPES):
                if s.lower() in detected_soil.lower() or detected_soil.lower() in s.lower():
                    soil_index = idx
                    break
            soil_override = st.selectbox("🪨 Soil Type", SOIL_TYPES, index=soil_index, key="crop_soil_ov")
            season_override = st.selectbox(
                "🗓️ Season", ["Auto-detect", "Kharif", "Rabi", "Zaid"], key="crop_season_ov",
            )
        with ov2:
            rainfall_override = st.number_input(
                "🌧️ Annual Rainfall (mm)", min_value=0, max_value=5000,
                value=rainfall_est, step=50, key="crop_rain_ov",
            )
        with ov3:
            soil_q_override = st.slider(
                "🧪 Soil Quality Override", 0.1, 1.0, soil_quality, 0.05, key="crop_sq_ov",
                help=f"Auto-calculated: {soil_quality:.0%} (pH={soil_breakdown.get('ph_value', '?')}, OC={soil_breakdown.get('soc_value', '?')}g/kg)",
            )

    # Use values from override widgets (they default to auto-detected values)
    soil_final = soil_override
    rainfall_final = rainfall_override
    soil_quality_final = soil_q_override
    season_final = season_override if season_override != "Auto-detect" else auto_season

    # Convert to acres
    if unit == "Hectares":
        acres = land_area * 2.471
    elif unit == "Bigha":
        acres = land_area * 0.625
    else:
        acres = land_area

    if st.button("🌾 Get Crop Recommendations", type="primary", use_container_width=True, key="btn_crop"):
        with st.spinner("Analyzing conditions and scoring crops..."):
            crops = recommend_crops(
                temp, humidity, rainfall_final, soil_final, season_final, acres, elevation
            )

        if not crops:
            st.warning("No suitable crops found for current conditions. Try adjusting parameters in the override section.")
            return

        st.session_state["crop_results"] = crops
        st.session_state["crop_params"] = {
            "temp": temp, "humidity": humidity, "rainfall": rainfall_final,
            "soil": soil_final, "soil_quality": soil_quality_final,
            "season": season_final, "acres": acres, "irrigation": irrigation,
            "elevation": elevation,
        }

    # Render results (persistent)
    crops = st.session_state.get("crop_results")
    if crops:
        st.markdown(f"### 🏆 Top {min(len(crops), 5)} Recommended Crops")
        st.caption("Click **📊 Predict Yield** on any crop for a detailed forecast.")
        for i, crop in enumerate(crops[:5], 1):
            render_crop_card(crop, i)
            c_left, c_right = st.columns([3, 1])
            with c_left:
                with st.expander(f"📋 {crop['name']} — Suitability Details"):
                    for r in crop["reasons"]:
                        st.markdown(r)
            with c_right:
                if st.button(f"📊 Predict Yield", key=f"yield_btn_{crop['name']}", use_container_width=True):
                    st.session_state["selected_yield_crop"] = crop["name"]

        st.markdown("### 📊 Revenue Analysis")
        render_revenue_comparison_chart(crops[:6])
