"""
AGRI-X AI — Disease Risk Engine (Module 3)
Weather-aware disease risk scoring.
"""

# Disease-specific environmental risk thresholds
DISEASE_RISK_PROFILES = {
    "Pepper__bell___Bacterial_spot": {
        "name": "Bacterial Spot", "high_humidity": 80, "temp_range": (24, 35),
        "rain_risk": True, "base_risk": 40,
        "desc_template": "Warm, wet conditions favor bacterial spread on pepper plants.",
    },
    "Pepper__bell___healthy": {
        "name": "Healthy", "high_humidity": 999, "temp_range": (0, 100),
        "rain_risk": False, "base_risk": 5,
        "desc_template": "Plant is healthy. Low risk under current conditions.",
    },
    "Potato___Early_blight": {
        "name": "Early Blight", "high_humidity": 70, "temp_range": (24, 34),
        "rain_risk": True, "base_risk": 35,
        "desc_template": "Warm temperatures and moderate humidity promote Alternaria solani.",
    },
    "Potato___Late_blight": {
        "name": "Late Blight", "high_humidity": 75, "temp_range": (10, 24),
        "rain_risk": True, "base_risk": 50,
        "desc_template": "Cool, humid conditions are ideal for Phytophthora infestans. CRITICAL pathogen.",
    },
    "Potato___healthy": {
        "name": "Healthy", "high_humidity": 999, "temp_range": (0, 100),
        "rain_risk": False, "base_risk": 5,
        "desc_template": "Healthy potato crop. Minimal risk.",
    },
    "Tomato_Bacterial_spot": {
        "name": "Bacterial Spot", "high_humidity": 80, "temp_range": (25, 35),
        "rain_risk": True, "base_risk": 40,
        "desc_template": "Warm, rainy conditions accelerate Xanthomonas spread on tomatoes.",
    },
    "Tomato_Early_blight": {
        "name": "Early Blight", "high_humidity": 70, "temp_range": (24, 34),
        "rain_risk": True, "base_risk": 35,
        "desc_template": "Warm, humid weather promotes Alternaria solani on tomato foliage.",
    },
    "Tomato_Late_blight": {
        "name": "Late Blight", "high_humidity": 75, "temp_range": (10, 24),
        "rain_risk": True, "base_risk": 55,
        "desc_template": "High humidity and cool temperatures create CRITICAL conditions for late blight.",
    },
    "Tomato_Leaf_Mold": {
        "name": "Leaf Mold", "high_humidity": 85, "temp_range": (18, 28),
        "rain_risk": False, "base_risk": 40,
        "desc_template": "Very high humidity (>85%) strongly favors Passalora fulva growth.",
    },
    "Tomato_Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot", "high_humidity": 75, "temp_range": (20, 30),
        "rain_risk": True, "base_risk": 38,
        "desc_template": "Wet, warm conditions promote Septoria lycopersici spore dispersal.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Spider Mites", "high_humidity": 40, "temp_range": (28, 42),
        "rain_risk": False, "base_risk": 30,
        "desc_template": "Hot, dry conditions favor spider mite populations. Low humidity increases risk.",
    },
    "Tomato__Target_Spot": {
        "name": "Target Spot", "high_humidity": 80, "temp_range": (20, 30),
        "rain_risk": True, "base_risk": 35,
        "desc_template": "Humid, warm conditions promote Corynespora cassiicola on tomato leaves.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Yellow Leaf Curl Virus", "high_humidity": 60, "temp_range": (25, 38),
        "rain_risk": False, "base_risk": 45,
        "desc_template": "Warm temperatures increase whitefly vector activity, spreading TYLCV.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus", "high_humidity": 60, "temp_range": (20, 35),
        "rain_risk": False, "base_risk": 35,
        "desc_template": "Virus spread is mechanical. Current weather has moderate influence.",
    },
    "Tomato_healthy": {
        "name": "Healthy", "high_humidity": 999, "temp_range": (0, 100),
        "rain_risk": False, "base_risk": 5,
        "desc_template": "Healthy tomato plant. Minimal disease risk.",
    },
}


def calculate_risk(disease_class: str, temperature: float, humidity: float, precipitation: float) -> dict:
    """
    Calculate disease risk score based on prediction + weather.
    Returns dict with: risk_level, risk_score, risk_color, reason, disease_name
    """
    profile = DISEASE_RISK_PROFILES.get(disease_class)
    if not profile:
        return {"risk_level": "Unknown", "risk_score": 0, "risk_color": "#888",
                "reason": "Disease profile not found.", "disease_name": disease_class}

    # Start from base risk
    score = profile["base_risk"]

    # Humidity factor
    if disease_class.endswith("spider_mite") or "Spider" in disease_class:
        # Spider mites: LOWER humidity = higher risk
        if humidity < 40:
            score += 25
        elif humidity < 55:
            score += 10
    else:
        if humidity >= profile["high_humidity"]:
            score += 25
        elif humidity >= profile["high_humidity"] - 10:
            score += 15

    # Temperature factor
    t_min, t_max = profile["temp_range"]
    if t_min <= temperature <= t_max:
        score += 20  # In optimal disease range
    elif abs(temperature - t_min) <= 5 or abs(temperature - t_max) <= 5:
        score += 8   # Close to range

    # Rainfall factor
    if profile["rain_risk"] and precipitation > 0:
        if precipitation > 5:
            score += 15
        elif precipitation > 1:
            score += 8

    # Clamp 0-100
    score = max(0, min(100, score))

    # Classify
    if score >= 75:
        level, color = "Critical", "#DC143C"
    elif score >= 55:
        level, color = "High", "#e74c3c"
    elif score >= 35:
        level, color = "Moderate", "#f39c12"
    else:
        level, color = "Low", "#2ecc71"

    # Build reason
    reasons = []
    if humidity >= profile.get("high_humidity", 999):
        reasons.append(f"Humidity ({humidity}%) above {profile['high_humidity']}% threshold")
    elif "Spider" in disease_class and humidity < 40:
        reasons.append(f"Low humidity ({humidity}%) favors mite reproduction")
    if t_min <= temperature <= t_max:
        reasons.append(f"Temperature ({temperature:.1f}°C) in disease-optimal range ({t_min}-{t_max}°C)")
    if profile["rain_risk"] and precipitation > 1:
        reasons.append(f"Rainfall ({precipitation:.1f}mm) promotes pathogen spread")
    if not reasons:
        reasons.append(profile["desc_template"])

    return {
        "risk_level": level,
        "risk_score": score,
        "risk_color": color,
        "reason": ". ".join(reasons) + ".",
        "disease_name": profile["name"],
    }


def render_risk_card(risk: dict):
    """Render a disease risk assessment card."""
    import streamlit as st
    level = risk["risk_level"]
    score = risk["risk_score"]
    color = risk["risk_color"]

    icon_map = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}
    icon = icon_map.get(level, "⚪")

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {color};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
            <span style="font-size:1rem;font-weight:700;color:#fff;">⚠️ Disease Risk Assessment</span>
            <span style="font-size:0.8rem;color:{color};background:rgba(255,255,255,0.08);padding:4px 12px;border-radius:12px;font-weight:600;">
                {icon} {level}</span>
        </div>
        <div style="margin-bottom:0.8rem;">
            <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                <span style="color:#8899aa;font-size:0.8rem;">Risk Score</span>
                <span style="color:{color};font-weight:700;">{score}%</span>
            </div>
            <div style="height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">
                <div style="width:{score}%;height:100%;background:{color};border-radius:4px;transition:width 0.8s;"></div>
            </div>
        </div>
        <div style="color:#ccd;font-size:0.85rem;line-height:1.5;">
            <strong>Disease:</strong> {risk['disease_name']}<br>
            <strong>Analysis:</strong> {risk['reason']}
        </div>
    </div>
    """, unsafe_allow_html=True)
