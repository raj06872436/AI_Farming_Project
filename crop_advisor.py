"""
AGRI-X AI — Crop Recommendation & Revenue Forecaster (Module 12) v3.5
Unified crop database + GDD thermal modeling + nutrient feasibility + companion/rotation intelligence.
"""
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Unified Crop Database (merged agronomic parameters) ──
UNIFIED_CROP_DB = {
    "Rice": {
        "season": ["Kharif"], "temp_min": 20, "temp_max": 38,
        "humidity_min": 60, "humidity_max": 85,
        "rainfall_min": 1000, "rainfall_max": 2000,
        "soil": ["Alluvial", "Clay", "Loamy", "Silt Loam", "Clay Loam"],
        "yield_per_acre": 1.2, "yield_max_per_acre": 1.6,
        "optimal_temp": (22, 32), "optimal_humidity": (65, 80),
        "optimal_monthly_rain": (150, 250),
        "price_per_tonne": 22000, "price_min_qt": 2000, "price_max_qt": 2400,
        "msp": 2183, "cost_per_acre": 18000, "duration_days": 120,
        "harvest_months": 4, "water_need": "High", "icon": "🌾",
        "t_base": 10.0, "gdd_target": 1800,
        "npk_demand": {"N": "High", "P": "Moderate", "K": "High"},
        "rotation_crops": ["Wheat", "Mustard", "Chickpea / Gram", "Lentils"],
        "companion_crops": ["Azolla (biofertilizer)", "Duckweed", "Sesbania"],
        "root_depth": "Shallow to Medium (0.4–0.6m)",
    },
    "Wheat": {
        "season": ["Rabi"], "temp_min": 8, "temp_max": 26,
        "humidity_min": 45, "humidity_max": 70,
        "rainfall_min": 400, "rainfall_max": 750,
        "soil": ["Alluvial", "Loamy", "Clay", "Silt Loam", "Loam"],
        "yield_per_acre": 1.2, "yield_max_per_acre": 1.8,
        "optimal_temp": (12, 22), "optimal_humidity": (50, 65),
        "optimal_monthly_rain": (50, 90),
        "price_per_tonne": 23000, "price_min_qt": 2100, "price_max_qt": 2500,
        "msp": 2275, "cost_per_acre": 16000, "duration_days": 135,
        "harvest_months": 4.5, "water_need": "Moderate", "icon": "🌾",
        "t_base": 0.0, "gdd_target": 1600,
        "npk_demand": {"N": "High", "P": "High", "K": "Moderate"},
        "rotation_crops": ["Green Gram (Mung)", "Soybean", "Maize", "Cotton"],
        "companion_crops": ["Mustard (intercrop 9:1)", "Chickpea"],
        "root_depth": "Medium (0.8–1.2m)",
    },
    "Tomato": {
        "season": ["Kharif", "Rabi", "Zaid"], "temp_min": 15, "temp_max": 33,
        "humidity_min": 45, "humidity_max": 80,
        "rainfall_min": 400, "rainfall_max": 700,
        "soil": ["Loamy", "Sandy", "Sandy Loam", "Loam", "Clay Loam"],
        "yield_per_acre": 10, "yield_max_per_acre": 15,
        "optimal_temp": (18, 28), "optimal_humidity": (50, 70),
        "optimal_monthly_rain": (50, 80),
        "price_per_tonne": 15000, "price_min_qt": 800, "price_max_qt": 2500,
        "msp": 0, "cost_per_acre": 45000, "duration_days": 100,
        "harvest_months": 3.5, "water_need": "Moderate", "icon": "🍅",
        "t_base": 10.0, "gdd_target": 1400,
        "npk_demand": {"N": "Moderate", "P": "High", "K": "Very High (Calcium)"},
        "rotation_crops": ["Maize", "Beans", "Cabbage / Cauliflower (Avoid Solanaceae)"],
        "companion_crops": ["French Marigold (nematode control)", "Basil", "Garlic"],
        "root_depth": "Medium (0.6–0.9m)",
    },
    "Potato": {
        "season": ["Rabi"], "temp_min": 12, "temp_max": 25,
        "humidity_min": 55, "humidity_max": 80,
        "rainfall_min": 450, "rainfall_max": 800,
        "soil": ["Loamy", "Sandy", "Alluvial", "Sandy Loam", "Loam"],
        "yield_per_acre": 6, "yield_max_per_acre": 10,
        "optimal_temp": (15, 22), "optimal_humidity": (60, 80),
        "optimal_monthly_rain": (50, 80),
        "price_per_tonne": 12000, "price_min_qt": 600, "price_max_qt": 1500,
        "msp": 0, "cost_per_acre": 35000, "duration_days": 100,
        "harvest_months": 3.5, "water_need": "Moderate", "icon": "🥔",
        "t_base": 7.0, "gdd_target": 1300,
        "npk_demand": {"N": "Moderate", "P": "High", "K": "Very High"},
        "rotation_crops": ["Maize", "Mustard", "Legumes / Green Manure"],
        "companion_crops": ["Beans", "Horseradish", "Marigold"],
        "root_depth": "Shallow (0.3–0.5m)",
    },
    "Bell Pepper": {
        "season": ["Kharif", "Rabi"], "temp_min": 16, "temp_max": 30,
        "humidity_min": 55, "humidity_max": 75,
        "rainfall_min": 500, "rainfall_max": 1100,
        "soil": ["Loamy", "Sandy", "Sandy Loam", "Loam", "Clay Loam"],
        "yield_per_acre": 4, "yield_max_per_acre": 8,
        "optimal_temp": (18, 28), "optimal_humidity": (60, 70),
        "optimal_monthly_rain": (60, 100),
        "price_per_tonne": 30000, "price_min_qt": 1500, "price_max_qt": 4000,
        "msp": 0, "cost_per_acre": 55000, "duration_days": 90,
        "harvest_months": 3, "water_need": "Moderate", "icon": "🫑",
        "t_base": 10.0, "gdd_target": 1350,
        "npk_demand": {"N": "Moderate", "P": "Moderate", "K": "High"},
        "rotation_crops": ["Cucurbits", "Beans", "Radish"],
        "companion_crops": ["Basil", "Onions", "Marigold"],
        "root_depth": "Medium (0.5–0.8m)",
    },
    "Cotton": {
        "season": ["Kharif"], "temp_min": 22, "temp_max": 38,
        "humidity_min": 45, "humidity_max": 70,
        "rainfall_min": 500, "rainfall_max": 1100,
        "soil": ["Clay", "Alluvial", "Clay Loam", "Black"],
        "yield_per_acre": 0.6, "yield_max_per_acre": 0.9,
        "optimal_temp": (25, 35), "optimal_humidity": (50, 65),
        "optimal_monthly_rain": (70, 120),
        "price_per_tonne": 65000, "price_min_qt": 6000, "price_max_qt": 7500,
        "msp": 6620, "cost_per_acre": 22000, "duration_days": 165,
        "harvest_months": 5.5, "water_need": "Moderate", "icon": "🏵️",
        "t_base": 15.0, "gdd_target": 2200,
        "npk_demand": {"N": "High", "P": "Moderate", "K": "High (Boron, Zinc)"},
        "rotation_crops": ["Wheat", "Sorghum", "Chickpea", "Groundnut"],
        "companion_crops": ["Cowpea (pest trap crop)", "Green Gram", "Castor (border)"],
        "root_depth": "Deep (1.2–1.8m)",
    },
    "Sugarcane": {
        "season": ["Kharif", "Rabi"], "temp_min": 18, "temp_max": 38,
        "humidity_min": 65, "humidity_max": 88,
        "rainfall_min": 1400, "rainfall_max": 2500,
        "soil": ["Alluvial", "Loamy", "Clay", "Loam", "Clay Loam"],
        "yield_per_acre": 25, "yield_max_per_acre": 35,
        "optimal_temp": (22, 33), "optimal_humidity": (70, 85),
        "optimal_monthly_rain": (180, 300),
        "price_per_tonne": 3150, "price_min_qt": 290, "price_max_qt": 350,
        "msp": 315, "cost_per_acre": 40000, "duration_days": 330,
        "harvest_months": 11, "water_need": "Very High", "icon": "🎋",
        "t_base": 12.0, "gdd_target": 3500,
        "npk_demand": {"N": "Very High", "P": "Moderate", "K": "High"},
        "rotation_crops": ["Sunnhemp (green manure)", "Rice", "Wheat"],
        "companion_crops": ["Soybean (intercrop)", "Onion", "Mustard"],
        "root_depth": "Deep (1.2–1.5m)",
    },
    "Maize": {
        "season": ["Kharif", "Rabi", "Zaid"], "temp_min": 18, "temp_max": 34,
        "humidity_min": 50, "humidity_max": 80,
        "rainfall_min": 450, "rainfall_max": 1000,
        "soil": ["Loamy", "Alluvial", "Loam", "Sandy Loam", "Clay Loam"],
        "yield_per_acre": 1.0, "yield_max_per_acre": 1.6,
        "optimal_temp": (21, 30), "optimal_humidity": (55, 75),
        "optimal_monthly_rain": (60, 120),
        "price_per_tonne": 21000, "price_min_qt": 1900, "price_max_qt": 2300,
        "msp": 2090, "cost_per_acre": 14000, "duration_days": 100,
        "harvest_months": 3.5, "water_need": "Moderate", "icon": "🌽",
        "t_base": 10.0, "gdd_target": 1450,
        "npk_demand": {"N": "Very High", "P": "Moderate", "K": "Moderate"},
        "rotation_crops": ["Soybean", "Groundnut", "Mustard", "Wheat"],
        "companion_crops": ["Cowpea", "Beans (nitrogen fixer)", "Pumpkin (ground cover)"],
        "root_depth": "Medium to Deep (0.8–1.2m)",
    },
    "Soybean": {
        "season": ["Kharif"], "temp_min": 18, "temp_max": 33,
        "humidity_min": 55, "humidity_max": 75,
        "rainfall_min": 550, "rainfall_max": 1000,
        "soil": ["Clay", "Loamy", "Clay Loam", "Black"],
        "yield_per_acre": 0.4, "yield_max_per_acre": 0.65,
        "optimal_temp": (20, 30), "optimal_humidity": (60, 70),
        "optimal_monthly_rain": (60, 100),
        "price_per_tonne": 46000, "price_min_qt": 4200, "price_max_qt": 5000,
        "msp": 4600, "cost_per_acre": 15000, "duration_days": 100,
        "harvest_months": 3.5, "water_need": "Moderate", "icon": "🫘",
        "t_base": 10.0, "gdd_target": 1300,
        "npk_demand": {"N": "Low (fixes N)", "P": "High", "K": "Moderate"},
        "rotation_crops": ["Wheat", "Mustard", "Gram"],
        "companion_crops": ["Sorghum / Jowar", "Maize"],
        "root_depth": "Medium (0.6–1.0m)",
    },
    "Mustard": {
        "season": ["Rabi"], "temp_min": 8, "temp_max": 26,
        "humidity_min": 35, "humidity_max": 65,
        "rainfall_min": 250, "rainfall_max": 500,
        "soil": ["Loamy", "Sandy", "Alluvial", "Sandy Loam", "Loam"],
        "yield_per_acre": 0.5, "yield_max_per_acre": 0.8,
        "optimal_temp": (10, 25), "optimal_humidity": (40, 60),
        "optimal_monthly_rain": (30, 50),
        "price_per_tonne": 55000, "price_min_qt": 5000, "price_max_qt": 6500,
        "msp": 5650, "cost_per_acre": 12000, "duration_days": 120,
        "harvest_months": 4, "water_need": "Low", "icon": "🌻",
        "t_base": 5.0, "gdd_target": 1200,
        "npk_demand": {"N": "Moderate", "P": "Moderate", "K": "Low (Sulfur needed)"},
        "rotation_crops": ["Kharif Rice", "Cotton", "Maize", "Fallow"],
        "companion_crops": ["Chickpea (mixed crop)", "Wheat"],
        "root_depth": "Medium (0.8–1.1m)",
    },
    "Onion": {
        "season": ["Rabi", "Kharif"], "temp_min": 12, "temp_max": 28,
        "humidity_min": 45, "humidity_max": 70,
        "rainfall_min": 300, "rainfall_max": 650,
        "soil": ["Loamy", "Sandy", "Alluvial", "Sandy Loam", "Loam"],
        "yield_per_acre": 6, "yield_max_per_acre": 12,
        "optimal_temp": (15, 25), "optimal_humidity": (50, 70),
        "optimal_monthly_rain": (40, 70),
        "price_per_tonne": 18000, "price_min_qt": 800, "price_max_qt": 3000,
        "msp": 0, "cost_per_acre": 40000, "duration_days": 130,
        "harvest_months": 4.5, "water_need": "Moderate", "icon": "🧅",
        "t_base": 6.0, "gdd_target": 1500,
        "npk_demand": {"N": "Moderate", "P": "High", "K": "High"},
        "rotation_crops": ["Tomato", "Chilli", "Maize"],
        "companion_crops": ["Carrot (pest protection)", "Cabbage", "Chamomile"],
        "root_depth": "Shallow (0.3–0.4m)",
    },
    "Chilli": {
        "season": ["Kharif", "Rabi"], "temp_min": 18, "temp_max": 35,
        "humidity_min": 50, "humidity_max": 75,
        "rainfall_min": 450, "rainfall_max": 1200,
        "soil": ["Loamy", "Clay", "Clay Loam", "Loam", "Black"],
        "yield_per_acre": 1.5, "yield_max_per_acre": 3.0,
        "optimal_temp": (20, 33), "optimal_humidity": (60, 70),
        "optimal_monthly_rain": (60, 120),
        "price_per_tonne": 120000, "price_min_qt": 8000, "price_max_qt": 15000,
        "msp": 0, "cost_per_acre": 50000, "duration_days": 150,
        "harvest_months": 5, "water_need": "Moderate", "icon": "🌶️",
        "t_base": 12.0, "gdd_target": 1700,
        "npk_demand": {"N": "High", "P": "Moderate", "K": "High"},
        "rotation_crops": ["Legumes", "Maize", "Groundnut"],
        "companion_crops": ["Marigold (aphid repellent)", "Garlic", "Coriander"],
        "root_depth": "Medium (0.6–0.9m)",
    },
    "Groundnut": {
        "season": ["Kharif"], "temp_min": 22, "temp_max": 36,
        "humidity_min": 45, "humidity_max": 65,
        "rainfall_min": 450, "rainfall_max": 800,
        "soil": ["Sandy", "Sandy Loam", "Loamy", "Loam"],
        "yield_per_acre": 0.5, "yield_max_per_acre": 0.75,
        "optimal_temp": (25, 35), "optimal_humidity": (50, 60),
        "optimal_monthly_rain": (50, 80),
        "price_per_tonne": 56000, "price_min_qt": 5000, "price_max_qt": 6500,
        "msp": 5850, "cost_per_acre": 18000, "duration_days": 120,
        "harvest_months": 4, "water_need": "Moderate", "icon": "🥜",
        "t_base": 13.0, "gdd_target": 1400,
        "npk_demand": {"N": "Low (fixes N)", "P": "High (Gypsum)", "K": "Moderate"},
        "rotation_crops": ["Cotton", "Sorghum", "Wheat", "Sunflower"],
        "companion_crops": ["Pigeon Pea (Red Gram)", "Sunflower", "Castor"],
        "root_depth": "Medium (0.5–0.9m)",
    },
    "Pulses": {
        "season": ["Rabi", "Kharif"], "temp_min": 15, "temp_max": 30,
        "humidity_min": 45, "humidity_max": 70,
        "rainfall_min": 350, "rainfall_max": 700,
        "soil": ["Loamy", "Clay", "Alluvial", "Loam", "Clay Loam"],
        "yield_per_acre": 0.4, "yield_max_per_acre": 0.65,
        "optimal_temp": (18, 28), "optimal_humidity": (50, 65),
        "optimal_monthly_rain": (40, 70),
        "price_per_tonne": 60000, "price_min_qt": 5500, "price_max_qt": 7000,
        "msp": 6600, "cost_per_acre": 14000, "duration_days": 105,
        "harvest_months": 3.5, "water_need": "Low", "icon": "🫛",
        "t_base": 8.0, "gdd_target": 1250,
        "npk_demand": {"N": "Low (Rhizobium fixes N)", "P": "High", "K": "Low"},
        "rotation_crops": ["Rice", "Wheat", "Maize", "Cotton (ideal restorative crop)"],
        "companion_crops": ["Maize", "Sorghum", "Millet"],
        "root_depth": "Medium (0.6–1.0m)",
    },
}

# ── Average Annual Rainfall by Indian State (mm) ──
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
              "Silt Loam", "Clay Loam", "Loam", "Silty", "Laterite", "Black"]


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
    Priority: State average adjusted by forecast anomaly → seasonal forecast extrapolation.
    """
    state = location.get("state", "") if location else ""
    state_avg = None

    for key, val in STATE_RAINFALL_AVG.items():
        if key.lower() in state.lower() or state.lower() in key.lower():
            state_avg = val
            break

    weekly_precip = 0
    if weather_raw and "daily" in weather_raw:
        daily = weather_raw["daily"]
        precip_list = daily.get("precipitation_sum", [])
        weekly_precip = sum(precip_list) if precip_list else 0

    if state_avg:
        avg_weekly = state_avg / 52
        if avg_weekly > 0 and weekly_precip > 0:
            ratio = weekly_precip / avg_weekly
            adjustment = max(0.85, min(1.15, ratio))
            estimate = int(state_avg * adjustment)
        else:
            estimate = state_avg
        return max(100, min(5000, estimate)), "high", f"State average ({state})"

    if weekly_precip > 0:
        month = datetime.now().month
        if month in [6, 7, 8, 9]:
            annual = weekly_precip * 52 * 0.6
        elif month in [12, 1, 2, 3]:
            annual = weekly_precip * 52 * 2.5
        else:
            annual = weekly_precip * 52
        return max(100, min(5000, int(annual))), "low", "Forecast extrapolation"

    return 800, "fallback", "Default estimate"


def calculate_nutrient_availability(ph: float):
    """
    Calculate macronutrient (N, P, K) and micronutrient availability scores (0.0 to 1.0)
    based on soil pH chemistry.
    """
    # Nitrogen (optimal 6.0 - 8.0)
    n_score = 1.0 if 6.0 <= ph <= 8.0 else max(0.3, 1.0 - abs(ph - 7.0) * 0.3)

    # Phosphorus (heavily locked below 5.5 by Al/Fe and above 8.0 by Ca)
    if 6.2 <= ph <= 7.3:
        p_score = 1.0
    elif 5.5 <= ph < 6.2 or 7.3 < ph <= 8.0:
        p_score = 0.75
    elif ph < 5.5:
        p_score = max(0.2, 0.75 - (5.5 - ph) * 0.4)
    else:
        p_score = max(0.3, 0.75 - (ph - 8.0) * 0.35)

    # Potassium (optimal 6.0 - 8.5)
    k_score = 1.0 if 6.0 <= ph <= 8.5 else max(0.4, 1.0 - abs(ph - 7.2) * 0.25)

    # Micronutrients (Fe, Zn, Mn readily available < 7.0, locked in alkaline > 7.8)
    micro_score = 1.0 if ph <= 6.8 else max(0.3, 1.0 - (ph - 6.8) * 0.4)

    return {
        "nitrogen": round(n_score, 2),
        "phosphorus": round(p_score, 2),
        "potassium": round(k_score, 2),
        "micronutrients": round(micro_score, 2),
    }


def calculate_soil_quality(soil_data):
    """
    Calculate a 0-1 soil quality score from ISRIC SoilGrids data,
    evaluating pH, organic carbon, texture balance, and Available Water Capacity (AWHC).
    """
    if not soil_data:
        return 0.6, {"ph": 0.6, "organic": 0.5, "texture": 0.7, "awhc": 0.6, "overall": 0.6, "ph_value": 6.5, "soc_value": 10}

    # Extract topsoil values
    top = soil_data.get("topsoil", soil_data)
    ph = top.get("ph", {}).get("value", 6.5) if isinstance(top.get("ph"), dict) else 6.5
    soc = top.get("organic_carbon", {}).get("value", 10) if isinstance(top.get("organic_carbon"), dict) else 10
    clay = top.get("clay", {}).get("value", 25) if isinstance(top.get("clay"), dict) else 25
    sand = top.get("sand", {}).get("value", 40) if isinstance(top.get("sand"), dict) else 40
    silt = top.get("silt", {}).get("value", 35) if isinstance(top.get("silt"), dict) else 35
    awhc = soil_data.get("awhc_topsoil", 140)

    # pH score
    ph_score = 1.0 if 6.0 <= ph <= 7.5 else (0.75 if 5.5 <= ph <= 8.0 else (0.45 if 5.0 <= ph <= 8.5 else 0.2))

    # SOC score
    oc_score = 1.0 if soc >= 20 else (0.8 if soc >= 12 else (0.6 if soc >= 7 else (0.4 if soc >= 4 else 0.2)))

    # Texture score
    clay_dev = abs(clay - 25) / 25
    sand_dev = abs(sand - 40) / 40
    silt_dev = abs(silt - 35) / 35
    texture_score = max(0.2, 1.0 - (clay_dev + sand_dev + silt_dev) / 3)

    # AWHC score (ideal 120-180 mm/m)
    awhc_score = min(1.0, max(0.3, awhc / 150.0))

    # Weighted overall
    overall = ph_score * 0.30 + oc_score * 0.25 + texture_score * 0.25 + awhc_score * 0.20
    overall = round(max(0.1, min(1.0, overall)), 2)

    nutrients = calculate_nutrient_availability(ph)

    return overall, {
        "ph": round(ph_score, 2),
        "organic": round(oc_score, 2),
        "texture": round(texture_score, 2),
        "awhc": round(awhc_score, 2),
        "overall": overall,
        "ph_value": ph,
        "soc_value": soc,
        "awhc_val": awhc,
        "nutrients": nutrients,
    }


def _match_soil_type(detected_soil, crop_soils):
    """Check if detected soil matches crop's preferred soil varieties."""
    if not detected_soil:
        return False
    detected_lower = detected_soil.lower()
    for cs in crop_soils:
        if cs.lower() in detected_lower or detected_lower in cs.lower():
            return True
    return False


def _score_crop(crop_name, crop, temp, humidity, annual_rainfall, soil, season, elevation=None, soil_data=None):
    """
    Score a crop 0-100 based on agronomic match across thermal, moisture, soil, and altitude conditions.
    """
    score = 0
    reasons = []

    # 1. Season match (25 pts)
    if season in crop["season"] or season == "Zaid":
        score += 25
        reasons.append(f"✅ Season match: Ideal for **{season}** planting.")
    else:
        score += 6
        reasons.append(f"⚠️ Off-season: Best sown in **{', '.join(crop['season'])}**.")

    # 2. Temperature & Thermal Range (25 pts)
    t_min, t_max = crop["temp_min"], crop["temp_max"]
    opt_lo, opt_hi = crop["optimal_temp"]
    if opt_lo <= temp <= opt_hi:
        score += 25
        reasons.append(f"✅ Optimal Temperature: Current {temp:.1f}°C is in prime growth zone ({opt_lo}–{opt_hi}°C).")
    elif t_min <= temp <= t_max:
        score += 18
        reasons.append(f"✅ Tolerable Temperature: {temp:.1f}°C is within acceptable bounds ({t_min}–{t_max}°C).")
    elif abs(temp - t_min) <= 4 or abs(temp - t_max) <= 4:
        score += 10
        reasons.append(f"⚠️ Temperature Marginal: {temp:.1f}°C approaches thermal tolerance limits.")
    else:
        reasons.append(f"❌ Thermal Stress: {temp:.1f}°C is outside viable range ({t_min}–{t_max}°C).")

    # 3. Humidity (15 pts)
    h_min, h_max = crop["humidity_min"], crop["humidity_max"]
    if h_min <= humidity <= h_max:
        score += 15
        reasons.append(f"✅ Ambient Humidity: {humidity:.0f}% is suitable for disease prevention and transpiration.")
    elif abs(humidity - h_min) <= 12 or abs(humidity - h_max) <= 12:
        score += 8
        reasons.append(f"⚠️ Marginal Humidity: {humidity:.0f}% may require adjusted irrigation or canopy aeration.")
    else:
        reasons.append(f"❌ Unfavorable Humidity: {humidity:.0f}% elevates stress / pathogen risks.")

    # 4. Rainfall & Moisture Support (15 pts)
    r_min, r_max = crop["rainfall_min"], crop["rainfall_max"]
    if r_min <= annual_rainfall <= r_max:
        score += 15
        reasons.append("✅ Water Balance: Regional annual rainfall is adequate.")
    elif annual_rainfall >= r_min * 0.7:
        score += 9
        reasons.append(f"⚠️ Moisture Gap: Supplementary irrigation required (need ~{r_min}mm).")
    else:
        score += 4
        reasons.append("❌ Low Precipitation: Relies heavily on perennial irrigation infrastructure.")

    # 5. Soil Texture & Chemistry (15 pts)
    if _match_soil_type(soil, crop["soil"]):
        score += 15
        reasons.append(f"✅ Soil Compatibility: **{soil}** soil offers ideal root penetration & aeration.")
    else:
        score += 6
        reasons.append(f"⚠️ Soil Texture: **{soil}** is workable with organic amendments and tillage.")

    # 6. Nutrient / pH Feasibility Bonus/Penalty (5 pts)
    if soil_data:
        ph_val = soil_data.get("ph", {}).get("value", 6.5) if isinstance(soil_data.get("ph"), dict) else 6.5
        nutrients = calculate_nutrient_availability(ph_val)
        if nutrients["phosphorus"] >= 0.8:
            score += 5
            reasons.append(f"✅ Nutrient Bioavailability: Soil pH ({ph_val:.1f}) ensures high phosphorus & potassium uptake.")
        elif nutrients["phosphorus"] < 0.5:
            score -= 5
            reasons.append(f"⚠️ Nutrient Lockup: Soil pH ({ph_val:.1f}) causes phosphorus/micronutrient fixation. Apply rock phosphate / lime.")

    # 7. Elevation Adjustment
    if elevation is not None:
        if crop_name in ["Rice", "Sugarcane"] and elevation > 1500:
            score -= 12
            reasons.append(f"⚠️ High Altitude: Elevation ({elevation:.0f}m) slows thermal development for {crop_name}.")
        elif crop_name in ["Wheat", "Potato", "Mustard"] and 400 < elevation < 2000:
            score += 5
            reasons.append(f"✅ Altitude Synergy: Elevation ({elevation:.0f}m) provides cool night temperatures.")

    return min(100, max(0, score)), reasons


def recommend_crops(temp, humidity, annual_rainfall, soil, season, land_area_acres, elevation=None, soil_data=None):
    """
    Rank crops by agronomic score and calculate detailed financial returns.
    """
    results = []
    for name, crop in UNIFIED_CROP_DB.items():
        score, reasons = _score_crop(
            name, crop, temp, humidity, annual_rainfall, soil, season, elevation, soil_data
        )
        if score < 20:
            continue

        yield_min = crop["yield_per_acre"] * land_area_acres
        yield_max = crop["yield_max_per_acre"] * land_area_acres
        rev_min = yield_min * 10 * crop["price_min_qt"]
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
            "rotation_crops": crop.get("rotation_crops", []),
            "companion_crops": crop.get("companion_crops", []),
            "npk_demand": crop.get("npk_demand", {}),
            "t_base": crop.get("t_base", 10.0),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def render_crop_card(crop, rank):
    """Render a comprehensive crop recommendation card with rotation & companion info."""
    sc = crop["score"]
    border_color = "#2ecc71" if sc >= 75 else "#f39c12" if sc >= 50 else "#e74c3c"
    profit_color = "#2ecc71" if crop["profit_min"] > 0 else "#e74c3c"
    msp_badge = f'<span style="background:rgba(46,204,113,0.15);color:#2ecc71;padding:3px 10px;border-radius:10px;font-size:0.75rem;font-weight:700;">🏛️ MSP: ₹{crop["msp"]}/q</span>' if crop["msp"] > 0 else '<span style="background:rgba(255,255,255,0.06);color:#8899aa;padding:3px 10px;border-radius:10px;font-size:0.75rem;">Open Market</span>'

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {border_color};margin-bottom:0.8rem;padding:1.2rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
            <div>
                <span style="font-size:1.6rem;">{crop['icon']}</span>
                <span style="font-size:1.25rem;font-weight:800;color:#fff;margin-left:0.5rem;">#{rank} {crop['name']}</span>
                <span style="margin-left:0.6rem;">{msp_badge}</span>
            </div>
            <span style="background:rgba(255,255,255,0.08);color:{border_color};padding:5px 14px;border-radius:12px;font-size:0.9rem;font-weight:800;">
                {sc}% Agronomic Match</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;margin-bottom:0.8rem;">
            <div style="text-align:center;padding:0.6rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">Estimated Yield</div>
                <div style="font-size:1.15rem;font-weight:800;color:#3498db;">📦 {crop['yield_min']:.1f} – {crop['yield_max']:.1f} t</div>
            </div>
            <div style="text-align:center;padding:0.6rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">Gross Revenue</div>
                <div style="font-size:1.15rem;font-weight:800;color:#f39c12;">💰 ₹{crop['revenue_min']/1000:.0f}K – ₹{crop['revenue_max']/1000:.0f}K</div>
            </div>
            <div style="text-align:center;padding:0.6rem;background:rgba(255,255,255,0.04);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">Estimated Net Profit</div>
                <div style="font-size:1.15rem;font-weight:800;color:{profit_color};">📈 ₹{crop['profit_min']/1000:.0f}K – ₹{crop['profit_max']/1000:.0f}K</div>
            </div>
        </div>
        <div style="display:flex;gap:0.8rem;flex-wrap:wrap;font-size:0.82rem;color:#aab;margin-bottom:0.5rem;">
            <span>🗓️ <b>Season:</b> {crop['season']}</span>
            <span>⏱️ <b>Cycle:</b> {crop['duration']} days</span>
            <span>💧 <b>Water Demand:</b> {crop['water_need']}</span>
            <span>💸 <b>Est. Cost:</b> ₹{crop['cost_per_acre']:,}/acre</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.6rem;padding-top:0.6rem;border-top:1px solid rgba(255,255,255,0.06);font-size:0.78rem;">
            <div>
                <span style="color:#2ecc71;font-weight:700;">🔄 Recommended Rotation:</span>
                <span style="color:#ccd;"> {", ".join(crop['rotation_crops'][:3])}</span>
            </div>
            <div>
                <span style="color:#f39c12;font-weight:700;">🌿 Companion Intercropping:</span>
                <span style="color:#ccd;"> {", ".join(crop['companion_crops'][:2])}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_revenue_comparison_chart(crops):
    """Render Plotly chart comparing revenue and profit across top crops."""
    if not crops:
        return
    names = [c["name"] for c in crops[:6]]
    rev_min = [c["revenue_min"] / 1000 for c in crops[:6]]
    rev_max = [c["revenue_max"] / 1000 for c in crops[:6]]
    costs = [c["total_cost"] / 1000 for c in crops[:6]]
    profit_avg = [((c["profit_min"] + c["profit_max"]) / 2) / 1000 for c in crops[:6]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Min Revenue", x=names, y=rev_min,
                         marker_color="rgba(52,152,219,0.7)", text=[f"₹{v:.0f}K" for v in rev_min], textposition="outside"))
    fig.add_trace(go.Bar(name="Max Revenue", x=names, y=rev_max,
                         marker_color="rgba(46,204,113,0.7)", text=[f"₹{v:.0f}K" for v in rev_max], textposition="outside"))
    fig.add_trace(go.Bar(name="Input Cost", x=names, y=costs,
                         marker_color="rgba(231,76,60,0.7)", text=[f"₹{v:.0f}K" for v in costs], textposition="outside"))
    fig.add_trace(go.Scatter(name="Avg Net Profit", x=names, y=profit_avg, mode="lines+markers",
                             line=dict(color="#f1c40f", width=3), marker=dict(size=8, symbol="diamond")))

    fig.update_layout(
        title="💰 Farm Financial Forecast: Revenue vs Input Cost vs Net Profit",
        barmode="group", template="plotly_dark", height=420, margin=dict(t=50, b=30),
        yaxis_title="Amount (₹ Thousands)", legend=dict(orientation="h", y=1.14),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_auto_detected_summary(city, temp, humidity, soil_type, soil_quality_score,
                                   rainfall_est, rainfall_confidence, rainfall_method,
                                   season, elevation, soil_breakdown):
    """Render a rich summary card showing auto-detected parameters and soil chemistry."""
    conf_color = {"high": "#2ecc71", "low": "#f39c12", "fallback": "#e74c3c"}.get(rainfall_confidence, "#888")
    elev_text = f"{elevation:.0f}m" if elevation is not None else "—"
    nutrients = soil_breakdown.get("nutrients", {})

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid #3498db;margin-bottom:1rem;padding:1.2rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
            <span style="font-size:1.05rem;font-weight:700;color:#fff;">🌐 Auto-Detected Environmental Profile</span>
            <span style="font-size:0.75rem;color:#8899aa;">Live APIs · Open-Meteo & ISRIC SoilGrids</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.6rem;margin-bottom:0.8rem;">
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">📍 Location</div>
                <div style="font-size:0.95rem;font-weight:700;color:#3498db;">{city}</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">🌡️ Temperature</div>
                <div style="font-size:0.95rem;font-weight:700;color:#e74c3c;">{temp:.1f}°C</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">💧 Humidity</div>
                <div style="font-size:0.95rem;font-weight:700;color:#3498db;">{humidity:.0f}%</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">🗓️ Season</div>
                <div style="font-size:0.95rem;font-weight:700;color:#f39c12;">{season}</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">🪨 Soil Texture</div>
                <div style="font-size:0.95rem;font-weight:700;color:#CD853F;">{soil_type}</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">🧪 Fertility Index</div>
                <div style="font-size:0.95rem;font-weight:700;color:#2ecc71;">{soil_quality_score:.0%}</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">🌧️ Est. Annual Rain</div>
                <div style="font-size:0.95rem;font-weight:700;color:{conf_color};">~{rainfall_est} mm</div>
            </div>
            <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="font-size:0.65rem;color:#8899aa;text-transform:uppercase;">⛰️ Elevation</div>
                <div style="font-size:0.95rem;font-weight:700;color:#9b59b6;">{elev_text}</div>
            </div>
        </div>
        <div style="font-size:0.75rem;color:#8899aa;background:rgba(255,255,255,0.02);padding:0.5rem 0.8rem;border-radius:8px;display:flex;justify-content:space-between;">
            <span>🧪 <b>pH Nutrient Availability:</b> N: {nutrients.get('nitrogen',0.8):.0%} · P: {nutrients.get('phosphorus',0.8):.0%} · K: {nutrients.get('potassium',0.8):.0%}</span>
            <span>💧 <b>AWHC:</b> {soil_breakdown.get('awhc_val', 140)} mm/m</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_crop_advisor_page(location, weather_current, weather_raw=None):
    """Full crop advisor page renderer — location-aware with GDD & nutrient intelligence."""
    st.markdown("### 🌾 Smart Crop Recommendations & Rotation Planner")

    temp = weather_current.get("temperature", 25) if weather_current else 25
    humidity = weather_current.get("humidity", 60) if weather_current else 60
    city_name = location.get("city", "Unknown") if location else "Unknown"
    elevation = location.get("elevation") if location else None

    detected_soil = location.get("soil_type", "Loamy") if location else "Loamy"
    soil_data = location.get("soil_data") if location else None
    soil_quality, soil_breakdown = calculate_soil_quality(soil_data)

    rainfall_est, rainfall_conf, rainfall_method = estimate_annual_rainfall(location, weather_raw)
    auto_season = _get_current_season()

    _render_auto_detected_summary(
        city_name, temp, humidity, detected_soil, soil_quality,
        rainfall_est, rainfall_conf, rainfall_method, auto_season, elevation, soil_breakdown
    )

    col1, col2 = st.columns(2)
    with col1:
        land_area = st.number_input(
            "🏞️ Farm Land Area", min_value=0.1, max_value=10000.0,
            value=5.0, step=0.5, key="crop_land",
        )
        unit = st.selectbox("Land Area Unit", ["Acres", "Hectares", "Bigha"], key="crop_unit")
    with col2:
        irrigation = st.checkbox("🚰 Drip / Canal Irrigation Available", value=True, key="crop_irrigation")

    with st.expander("⚙️ Override Farm & Soil Parameters (Optional)", expanded=False):
        ov1, ov2, ov3 = st.columns(3)
        with ov1:
            soil_index = 0
            for idx, s in enumerate(SOIL_TYPES):
                if s.lower() in detected_soil.lower() or detected_soil.lower() in s.lower():
                    soil_index = idx
                    break
            soil_override = st.selectbox("🪨 Soil Type", SOIL_TYPES, index=soil_index, key="crop_soil_ov")
            season_override = st.selectbox(
                "🗓️ Sowing Season", ["Auto-detect", "Kharif", "Rabi", "Zaid"], key="crop_season_ov",
            )
        with ov2:
            rainfall_override = st.number_input(
                "🌧️ Annual Rainfall (mm)", min_value=0, max_value=5000,
                value=rainfall_est, step=50, key="crop_rain_ov",
            )
        with ov3:
            soil_q_override = st.slider(
                "🧪 Soil Fertility Score", 0.1, 1.0, soil_quality, 0.05, key="crop_sq_ov",
                help=f"Auto-calculated: {soil_quality:.0%} (pH={soil_breakdown.get('ph_value', 6.5)}, SOC={soil_breakdown.get('soc_value', 10)}g/kg)",
            )

    soil_final = soil_override
    rainfall_final = rainfall_override
    soil_quality_final = soil_q_override
    season_final = season_override if season_override != "Auto-detect" else auto_season

    if unit == "Hectares":
        acres = land_area * 2.471
    elif unit == "Bigha":
        acres = land_area * 0.625
    else:
        acres = land_area

    if st.button("🌾 Compute Optimal Crop Rankings", type="primary", use_container_width=True, key="btn_crop"):
        with st.spinner("Analyzing multi-factor agronomic suitability..."):
            crops = recommend_crops(
                temp, humidity, rainfall_final, soil_final, season_final, acres, elevation, soil_data
            )

        if not crops:
            st.warning("No suitable crops found for current parameters. Try adjusting the override section.")
            return

        st.session_state["crop_results"] = crops
        st.session_state["crop_params"] = {
            "temp": temp, "humidity": humidity, "rainfall": rainfall_final,
            "soil": soil_final, "soil_quality": soil_quality_final,
            "season": season_final, "acres": acres, "irrigation": irrigation,
            "elevation": elevation,
        }

    crops = st.session_state.get("crop_results")
    if crops:
        st.markdown(f"### 🏆 Top {min(len(crops), 5)} Recommended Crops for Your Farm")
        st.caption("Includes crop rotation synergies and companion planting for natural pest suppression.")
        for i, crop in enumerate(crops[:5], 1):
            render_crop_card(crop, i)
            c_left, c_right = st.columns([3, 1])
            with c_left:
                with st.expander(f"📋 {crop['name']} — Detailed Agronomic Breakdown"):
                    for r in crop["reasons"]:
                        st.markdown(r)
            with c_right:
                if st.button(f"📊 Predict Yield", key=f"yield_btn_{crop['name']}", use_container_width=True):
                    st.session_state["selected_yield_crop"] = crop["name"]
                    st.session_state["yield_auto_trigger"] = True

        st.markdown("### 📊 Farm Revenue & Profit Analysis")
        render_revenue_comparison_chart(crops[:6])
