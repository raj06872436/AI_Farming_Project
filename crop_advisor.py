"""
AGRI-X AI — Crop Recommendation & Revenue Forecaster (Module 12) v4.0

What changed in v4
  • 40 crops (cereals, millets, pulses, oilseeds, vegetables, spices, fruit) with
    current MSPs — see crop_database.py
  • Crops are judged against the climate of their *growing season* at your exact
    location (5-year weather records), not against today's temperature
  • Soil: Indian soil group (black, red, alluvial, laterite…), texture and pH, from
    live SoilGrids data or, when that is unavailable, the regional soil profile
  • Regional fit: crops commonly grown in your agro-climatic region get credit —
    seed, buyers and know-how are close by
"""
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

import theme  # registers the light "agrix" Plotly template
from climate_service import SEASON_LABEL, get_climate_normals, season_window
from crop_database import CROP_DB, SOIL_TEXTURES
from india_agro_data import SOIL_GROUPS, find_profile

UNIFIED_CROP_DB = CROP_DB          # name kept for the yield predictor / insights modules
SOIL_GROUP_NAMES = list(SOIL_GROUPS.keys())
SEASONS = ["Kharif", "Rabi", "Zaid"]
_SHORT_SEASON = {"Kharif": "Jun–Oct", "Rabi": "Nov–Mar", "Zaid": "Mar–Jun"}


# ─────────────────────────────────────────────────────────────────────
# Season & climate helpers
# ─────────────────────────────────────────────────────────────────────
def _get_current_season():
    """
    The sowing season farmers are planning for right now.
    Feb–Apr → Zaid (summer), May–Aug → Kharif (monsoon), Sep–Jan → Rabi (winter).
    """
    m = datetime.now().month
    if m in (2, 3, 4):
        return "Zaid"
    if m in (5, 6, 7, 8):
        return "Kharif"
    return "Rabi"


def estimate_annual_rainfall(location, weather_raw=None):
    """
    Annual rainfall for the location.
    Returns (mm, confidence, method) — confidence is "high" for measured local records,
    "medium" for regional normals.
    """
    normals = get_climate_normals(location)
    conf = "high" if normals.get("live") else "medium"
    return int(normals["annual_rain"]), conf, normals["source"]


# ─────────────────────────────────────────────────────────────────────
# Soil
# ─────────────────────────────────────────────────────────────────────
def calculate_nutrient_availability(ph: float):
    """Share (0–1) of N, P, K and micronutrients plants can take up at this soil pH."""
    n_score = 1.0 if 6.0 <= ph <= 8.0 else max(0.3, 1.0 - abs(ph - 7.0) * 0.3)
    if 6.2 <= ph <= 7.3:
        p_score = 1.0
    elif 5.5 <= ph < 6.2 or 7.3 < ph <= 8.0:
        p_score = 0.75
    elif ph < 5.5:
        p_score = max(0.2, 0.75 - (5.5 - ph) * 0.4)
    else:
        p_score = max(0.3, 0.75 - (ph - 8.0) * 0.35)
    k_score = 1.0 if 6.0 <= ph <= 8.5 else max(0.4, 1.0 - abs(ph - 7.2) * 0.25)
    micro_score = 1.0 if ph <= 6.8 else max(0.3, 1.0 - (ph - 6.8) * 0.4)
    return {"nitrogen": round(n_score, 2), "phosphorus": round(p_score, 2),
            "potassium": round(k_score, 2), "micronutrients": round(micro_score, 2)}


def _val(d, key, default):
    v = d.get(key)
    return v.get("value", default) if isinstance(v, dict) else default


def calculate_soil_quality(soil_data):
    """
    0–1 soil fertility score from pH, organic carbon, nitrogen, texture balance and
    water-holding capacity. Works with live SoilGrids data and regional profiles.
    """
    if not soil_data:
        soil_data = {}
    top = soil_data.get("topsoil", soil_data) or {}
    ph = _val(top, "ph", 7.0)
    soc = _val(top, "organic_carbon", 6)
    clay = _val(top, "clay", 25)
    sand = _val(top, "sand", 40)
    nitrogen = _val(top, "nitrogen", None)
    awhc = soil_data.get("awhc_topsoil", 140)

    ph_score = 1.0 if 6.0 <= ph <= 7.5 else 0.8 if 5.5 <= ph <= 8.0 else 0.55 if 5.0 <= ph <= 8.5 else 0.3
    # Indian soils: organic carbon 7.5 g/kg (0.75 %) is "high", under 5 g/kg (0.5 %) is "low"
    oc_score = 1.0 if soc >= 10 else 0.85 if soc >= 7.5 else 0.65 if soc >= 5 else 0.45 if soc >= 3 else 0.3
    # Loam is ideal; heavy clay and pure sand both lose points
    texture_score = max(0.35, 1.0 - (abs(clay - 25) / 40 + abs(sand - 40) / 60) / 2)
    awhc_score = min(1.0, max(0.3, awhc / 150.0))
    if nitrogen is not None:
        n_score = 1.0 if nitrogen >= 1.5 else 0.8 if nitrogen >= 1.0 else 0.6 if nitrogen >= 0.6 else 0.4
        overall = 0.25 * ph_score + 0.22 * oc_score + 0.18 * n_score + 0.20 * texture_score + 0.15 * awhc_score
    else:
        n_score = None
        overall = 0.30 * ph_score + 0.28 * oc_score + 0.24 * texture_score + 0.18 * awhc_score
    overall = round(max(0.1, min(1.0, overall)), 2)

    return overall, {
        "ph": round(ph_score, 2), "organic": round(oc_score, 2), "texture": round(texture_score, 2),
        "awhc": round(awhc_score, 2), "nitrogen_score": n_score, "overall": overall,
        "ph_value": round(ph, 1), "soc_value": round(soc, 1), "awhc_val": awhc,
        "nitrogen_value": nitrogen, "nutrients": calculate_nutrient_availability(ph),
        "source": soil_data.get("source", "unknown"),
    }


def _texture_points(texture, crop_textures):
    """6 for a preferred texture, 4 for a neighbouring one, 1 otherwise."""
    if not texture:
        return 3
    t = texture.lower()
    if any(t == c.lower() for c in crop_textures):
        return 6
    order = [s.lower() for s in SOIL_TEXTURES]
    # Map finer classes (e.g. "Silty Clay") onto the nearest of our 7
    base = next((s for s in order if s in t), None) or ("clay" if "clay" in t else "loam")
    i = order.index(base)
    if any(abs(i - order.index(c.lower())) <= 1 for c in crop_textures if c.lower() in order):
        return 4
    return 1


# ─────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────
def score_crop(name, crop, ctx):
    """
    Score a crop 0–100 for this farm and season.
      Season fit 20 · Temperature 25 · Water 20 · Soil 15 · pH 10 · Regional fit 10
    Returns (score, reasons, season_climate).
    """
    reasons = []
    season = ctx["season"]
    perennial = "Perennial" in crop["season"]
    calc_season = "Perennial" if perennial else season
    w = season_window(ctx["normals"], calc_season, crop["duration_days"])
    score = 0.0

    # 1. Season (20)
    if perennial:
        score += 14
        reasons.append(f"🌳 Plantation crop — {crop.get('note') or 'plan for several years.'}")
    elif season in crop["season"]:
        score += 20
        reasons.append(f"✅ Sown in **{season}** ({SEASON_LABEL[season]}).")
    else:
        score += 4
        reasons.append(f"⚠️ Not usually a {season} crop — best sown in **{', '.join(crop['season'])}**.")

    # 2. Temperature over the growing period (25)
    t_min, lo, hi, t_max = crop["temp_range"]
    t = w["mean_temp"]
    if lo <= t <= hi:
        pts = 25
        reasons.append(f"✅ Growing-season temperature ~{t:.0f}°C is ideal ({lo}–{hi}°C).")
    elif t_min <= t <= t_max:
        gap = (lo - t) / max(1, lo - t_min) if t < lo else (t - hi) / max(1, t_max - hi)
        pts = 22 - 12 * gap
        reasons.append(f"🟡 Season averages ~{t:.0f}°C — workable, a bit {'cool' if t < lo else 'warm'} (best {lo}–{hi}°C).")
    else:
        pts = 2
        reasons.append(f"❌ Season averages ~{t:.0f}°C — {'too cold' if t < t_min else 'too hot'} (needs {t_min}–{t_max}°C).")
    if w["coldest"] < t_min - 4:
        pts -= 4
        reasons.append(f"🥶 Nights can drop to ~{w['coldest']:.0f}°C — frost / cold injury risk.")
    if w["hottest"] > t_max + 3:
        pts -= 4
        reasons.append(f"🔥 Days can reach ~{w['hottest']:.0f}°C — heat stress at flowering.")
    score += max(0, pts)

    # 3. Water (20): seasonal rain vs crop water need, with irrigation
    need_lo, need_hi = crop["water_mm"]
    rain = w["rain"]
    if need_lo <= rain <= need_hi * 1.5:
        score += 20
        reasons.append(f"✅ Season rain (~{rain} mm) covers the crop's needs ({need_lo}–{need_hi} mm).")
    elif rain < need_lo:
        gap = need_lo - rain
        if ctx["irrigation"]:
            pts = 18 if gap < 0.6 * need_lo or crop["water_need"] in ("Low", "Moderate") else 14
            reasons.append(f"💧 Needs ~{gap:.0f} mm more water than rain gives — irrigation available.")
        else:
            ratio = rain / need_lo
            bonus = 1.2 if crop["water_need"] == "Low" else 1.0
            pts = max(2, min(20, 20 * ratio * bonus))
            reasons.append(f"{'🟡' if ratio > 0.6 else '❌'} Rain (~{rain} mm) is short of the {need_lo} mm needed "
                           f"and there's no irrigation.")
        score += pts
    else:
        pts = 17 if crop["water_need"] in ("High", "Very High") else 10
        score += pts
        reasons.append(f"🌧️ Very wet season (~{rain} mm) — {'fine for this crop' if pts > 12 else 'needs good drainage / raised beds'}.")

    # 4. Soil group + texture (15)
    grp = ctx.get("soil_group")
    grp_label = SOIL_GROUPS.get(grp, (grp or "your",))[0]
    if grp in crop["soil_groups"]:
        score += 9
        reasons.append(f"✅ Does well on **{grp_label}** soil.")
    else:
        score += 3
        reasons.append(f"🟡 Prefers {', '.join(SOIL_GROUPS[g][0] for g in crop['soil_groups'][:2])} soil over {grp_label}.")
    score += _texture_points(ctx.get("texture"), crop["textures"])

    # 5. pH (10)
    ph = ctx.get("ph", 7.0)
    p_lo, p_hi = crop["ph_range"]
    if p_lo <= ph <= p_hi:
        score += 10
    elif p_lo - 0.5 <= ph <= p_hi + 0.5:
        score += 6
        reasons.append(f"🟡 Soil pH {ph:.1f} is slightly outside the best range ({p_lo}–{p_hi}).")
    else:
        score += 2
        fix = "add lime" if ph < p_lo else "add gypsum / organic matter"
        reasons.append(f"❌ Soil pH {ph:.1f} is outside the crop's range ({p_lo}–{p_hi}) — {fix}.")

    # 6. Regional fit (10)
    if name in ctx.get("region_crops", []):
        score += 10
        reasons.append(f"🏘️ Widely grown in {ctx.get('region_name', 'your region')} — seed, buyers and advice are nearby.")
    else:
        score += 2

    # Elevation
    elev = ctx.get("elevation")
    if elev is not None:
        if name in ("Rice", "Sugarcane", "Cotton", "Banana") and elev > 1500:
            score -= 10
            reasons.append(f"⛰️ At {elev:.0f} m the season is too short and cool for {name}.")
        elif name in ("Potato", "Cabbage", "Cauliflower", "Garlic", "Ginger") and elev > 800:
            score += 3
            reasons.append(f"⛰️ Cool nights at {elev:.0f} m suit {name}.")

    return int(round(min(100, max(0, score)))), reasons, w


def build_context(location, season, irrigation, soil_group=None, texture=None, ph=None, soil_quality=None):
    """Everything the scorer needs about this farm, filled from location data."""
    profile = find_profile(location)
    soil_data = (location or {}).get("soil_data")
    q, breakdown = calculate_soil_quality(soil_data)
    return {
        "season": season,
        "normals": get_climate_normals(location),
        "irrigation": irrigation,
        "soil_group": soil_group or (location or {}).get("soil_group") or profile["soil_group"],
        "texture": texture or (location or {}).get("soil_type") or profile["texture"],
        "ph": ph if ph is not None else breakdown["ph_value"],
        "soil_quality": soil_quality if soil_quality is not None else q,
        "elevation": (location or {}).get("elevation"),
        "region_crops": (location or {}).get("region_crops") or profile["crops"],
        "region_name": (location or {}).get("agro_region") or profile["name"],
    }


def recommend_crops(ctx, land_area_acres):
    """Rank every crop for this farm and work out yield, revenue and profit."""
    results = []
    for name, crop in UNIFIED_CROP_DB.items():
        score, reasons, w = score_crop(name, crop, ctx)
        # Poorly suited crops yield less: scale the typical range by suitability
        fit = max(0.45, min(1.0, score / 88))
        soil_factor = 0.85 + 0.15 * ctx.get("soil_quality", 0.7)
        yield_min = crop["yield_per_acre"] * land_area_acres * fit * soil_factor
        yield_max = crop["yield_max_per_acre"] * land_area_acres * fit * soil_factor
        price_lo = max(crop["price_min_qt"], 0.9 * crop["msp"]) if crop["msp"] else crop["price_min_qt"]
        rev_min = yield_min * 10 * price_lo
        rev_max = yield_max * 10 * max(crop["price_max_qt"], crop["msp"])
        total_cost = crop["cost_per_acre"] * land_area_acres
        results.append({
            "name": name, "icon": crop["icon"], "category": crop["category"], "score": score, "reasons": reasons,
            "season": ", ".join(crop["season"]), "duration": crop["duration_days"],
            "water_need": crop["water_need"],
            "yield_min": yield_min, "yield_max": yield_max,
            "yield_per_acre": crop["yield_per_acre"], "yield_max_per_acre": crop["yield_max_per_acre"],
            "price_min_qt": crop["price_min_qt"], "price_max_qt": crop["price_max_qt"],
            "price_per_tonne": crop["price_per_tonne"], "msp": crop["msp"],
            "revenue_min": rev_min, "revenue_max": rev_max, "total_cost": total_cost,
            "profit_min": rev_min - total_cost, "profit_max": rev_max - total_cost,
            "cost_per_acre": crop["cost_per_acre"], "harvest_months": crop["harvest_months"],
            "optimal_temp": crop["optimal_temp"], "optimal_humidity": crop["optimal_humidity"],
            "optimal_monthly_rain": crop["optimal_monthly_rain"],
            "rotation_crops": crop.get("rotation_crops", []), "companion_crops": crop.get("companion_crops", []),
            "npk_demand": crop.get("npk_demand", {}), "t_base": crop.get("t_base", 10.0),
            "note": crop.get("note", ""), "climate": w,
            "regional": name in ctx.get("region_crops", []),
        })
    results.sort(key=lambda x: (x["score"], (x["profit_min"] + x["profit_max"]) / 2), reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────
def _money(v):
    """₹ in thousands / lakhs, readable."""
    sign = "−" if v < 0 else ""
    v = abs(v)
    if v >= 1e7:
        return f"{sign}₹{v / 1e7:.1f} Cr"
    if v >= 1e5:
        return f"{sign}₹{v / 1e5:.1f} L"
    return f"{sign}₹{v / 1e3:.0f}K"


def render_crop_card(crop, rank):
    """Recommendation card with yield, money, climate fit, rotation & companion crops."""
    sc = crop["score"]
    border_color = theme.GREEN if sc >= 75 else theme.AMBER if sc >= 55 else theme.RED
    profit_color = theme.GREEN if crop["profit_min"] > 0 else theme.AMBER if crop["profit_max"] > 0 else theme.RED
    msp_badge = (f'<span class="chip chip-green" style="font-size:0.72rem;">🏛️ MSP ₹{crop["msp"]:,}/q</span>'
                 if crop["msp"] > 0 else '<span class="chip chip-gray" style="font-size:0.72rem;">Open market</span>')
    region_badge = ('<span class="chip chip-sky" style="font-size:0.72rem;">🏘️ Grown in your region</span>'
                    if crop.get("regional") else "")
    w = crop["climate"]
    unit = "t"
    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {border_color};margin-bottom:0.6rem;padding:1.1rem 1.2rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:0.6rem;flex-wrap:wrap;margin-bottom:0.7rem;">
            <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">
                <span style="font-size:1.6rem;">{crop['icon']}</span>
                <span style="font-size:1.2rem;font-weight:800;color:var(--text);">#{rank} {crop['name']}</span>
                <span class="chip chip-gray" style="font-size:0.72rem;">{crop['category']}</span>
                {msp_badge}{region_badge}
            </div>
            <span style="background:{border_color}14;color:{border_color};padding:5px 14px;border-radius:12px;font-size:0.9rem;font-weight:800;">
                {sc}/100 match</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;margin-bottom:0.7rem;">
            <div class="card-soft" style="text-align:center;">
                <div class="kv-label">Expected harvest</div>
                <div style="font-size:1.1rem;font-weight:800;color:{theme.SKY};">📦 {crop['yield_min']:.1f} – {crop['yield_max']:.1f} {unit}</div>
            </div>
            <div class="card-soft" style="text-align:center;">
                <div class="kv-label">Gross income</div>
                <div style="font-size:1.1rem;font-weight:800;color:{theme.AMBER};">💰 {_money(crop['revenue_min'])} – {_money(crop['revenue_max'])}</div>
            </div>
            <div class="card-soft" style="text-align:center;">
                <div class="kv-label">Profit after costs</div>
                <div style="font-size:1.1rem;font-weight:800;color:{profit_color};">📈 {_money(crop['profit_min'])} – {_money(crop['profit_max'])}</div>
            </div>
        </div>
        <div style="display:flex;gap:0.9rem;flex-wrap:wrap;font-size:0.82rem;color:var(--text-2);margin-bottom:0.4rem;">
            <span>🗓️ <b>Season:</b> {crop['season']}</span>
            <span>⏱️ <b>Duration:</b> {crop['duration']} days</span>
            <span>🌡️ <b>Season climate:</b> ~{w['mean_temp']:.0f}°C, {w['rain']} mm rain</span>
            <span>💧 <b>Water need:</b> {crop['water_need']}</span>
            <span>💸 <b>Cost:</b> ₹{crop['cost_per_acre']:,}/acre</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.5rem;padding-top:0.5rem;border-top:1px solid var(--border);font-size:0.8rem;">
            <div><span style="color:var(--green);font-weight:700;">🔄 Rotate with:</span>
                 <span style="color:var(--text-2);"> {", ".join(crop['rotation_crops'][:3])}</span></div>
            <div><span style="color:var(--amber);font-weight:700;">🌿 Grow alongside:</span>
                 <span style="color:var(--text-2);"> {", ".join(crop['companion_crops'][:2])}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_revenue_comparison_chart(crops):
    """Revenue vs cost vs profit for the top crops."""
    if not crops:
        return
    top = crops[:6]
    names = [c["name"] for c in top]
    rev_min = [c["revenue_min"] / 1000 for c in top]
    rev_max = [c["revenue_max"] / 1000 for c in top]
    costs = [c["total_cost"] / 1000 for c in top]
    profit_avg = [((c["profit_min"] + c["profit_max"]) / 2) / 1000 for c in top]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Income (low)", x=names, y=rev_min, marker_color=theme.rgba(theme.SKY, 0.75)))
    fig.add_trace(go.Bar(name="Income (high)", x=names, y=rev_max, marker_color=theme.rgba(theme.GREEN, 0.75)))
    fig.add_trace(go.Bar(name="Cost", x=names, y=costs, marker_color=theme.rgba(theme.RED, 0.7)))
    fig.add_trace(go.Scatter(name="Average profit", x=names, y=profit_avg, mode="lines+markers",
                             line=dict(color=theme.AMBER, width=3), marker=dict(size=8, symbol="diamond")))
    fig.update_layout(
        title="💰 Income vs cost vs profit (₹ thousands)",
        barmode="group", template="agrix", height=420, margin=dict(t=50, b=30),
        yaxis_title="₹ thousands", legend=dict(orientation="h", y=-0.12, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _land_to_acres(area, unit):
    return area * {"Acres": 1.0, "Hectares": 2.471, "Bigha": 0.625, "Guntha": 0.025}[unit]


def render_crop_advisor_page(location, weather_current, weather_raw=None):
    """Full crop advisor page — location-, season- and soil-aware."""
    st.markdown('<div class="section-head" style="margin-top:0;"><div class="sh-title">🌾 Best crops for your farm</div>'
                '<span>Ranked by season climate, soil, water and what grows well in your region</span></div>',
                unsafe_allow_html=True)

    profile = find_profile(location)
    soil_data = (location or {}).get("soil_data")
    soil_quality, soil_breakdown = calculate_soil_quality(soil_data)
    normals = get_climate_normals(location)
    soil_live = soil_breakdown.get("source") == "soilgrids"
    group = (location or {}).get("soil_group") or profile["soil_group"]
    texture = (location or {}).get("soil_type") or profile["texture"]

    # Where the numbers come from (so nobody mistakes a regional estimate for a soil test)
    nutrients = soil_breakdown.get("nutrients", {})
    facts = [
        f'<span class="chip chip-sky">🧭 {profile["name"]}</span>',
        f'<span class="chip chip-gray">🪨 {SOIL_GROUPS[group][0]} · {texture}</span>',
        f'<span class="chip chip-gray">🧪 pH {soil_breakdown["ph_value"]} · organic carbon {soil_breakdown["soc_value"]} g/kg</span>',
        f'<span class="chip chip-gray">N {nutrients.get("nitrogen", 0.8):.0%} · P {nutrients.get("phosphorus", 0.8):.0%} '
        f'· K {nutrients.get("potassium", 0.8):.0%} available</span>',
        f'<span class="chip chip-gray">🌧️ ~{normals["annual_rain"]} mm rain / year</span>',
        f'<span class="chip chip-{"green" if soil_live else "amber"}">'
        f'{"🛰️ Measured soil" if soil_live else "📘 Typical regional soil"}</span>',
        f'<span class="chip chip-{"green" if normals.get("live") else "amber"}">'
        f'{"🛰️ Local 5-year climate" if normals.get("live") else "📘 Regional climate normals"}</span>',
    ]
    st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:0.9rem;">{"".join(facts)}</div>',
                unsafe_allow_html=True)

    auto_season = _get_current_season()
    with st.container(border=True, key="panel_crop_inputs"):
        c1, c2, c3, c4 = st.columns([1.1, 0.9, 1.2, 1.2])
        with c1:
            land_area = st.number_input("🏞️ Farm land area", min_value=0.1, max_value=10000.0,
                                        value=5.0, step=0.5, key="crop_land")
        with c2:
            unit = st.selectbox("Unit", ["Acres", "Hectares", "Bigha", "Guntha"], key="crop_unit")
        with c3:
            season = st.selectbox(
                "🗓️ Sowing season", SEASONS, index=SEASONS.index(auto_season), key="crop_season",
                format_func=lambda s: f"{s} ({_SHORT_SEASON[s]})" + (" ★ next" if s == auto_season else ""),
            )
        with c4:
            st.write("")
            st.write("")
            irrigation = st.checkbox("🚰 Irrigation available", value=True, key="crop_irrigation")

        with st.expander("⚙️ I know my soil — adjust the details (optional)", expanded=False):
            st.caption("Have a Soil Health Card or lab report? Enter its values for a more accurate ranking.")
            o1, o2, o3, o4 = st.columns(4)
            with o1:
                group_ov = st.selectbox("Soil type", SOIL_GROUP_NAMES, index=SOIL_GROUP_NAMES.index(group),
                                        format_func=lambda g: SOIL_GROUPS[g][0], key="crop_group_ov")
            with o2:
                tex_options = SOIL_TEXTURES
                tex_idx = next((i for i, t in enumerate(tex_options) if t.lower() == texture.lower()),
                               next((i for i, t in enumerate(tex_options) if t.lower() in texture.lower()), 2))
                texture_ov = st.selectbox("Texture", tex_options, index=tex_idx, key="crop_tex_ov")
            with o3:
                ph_ov = st.number_input("Soil pH", min_value=3.5, max_value=10.0, step=0.1,
                                        value=float(soil_breakdown["ph_value"]), key="crop_ph_ov")
            with o4:
                sq_ov = st.slider("Fertility score", 0.1, 1.0, float(soil_quality), 0.05, key="crop_sq_ov",
                                  help="Auto-calculated from pH, organic carbon, nitrogen, texture and water-holding capacity.")

    acres = _land_to_acres(land_area, unit)

    if st.button("🌾 Find the best crops", type="primary", use_container_width=True, key="btn_crop"):
        with st.spinner(f"Checking {len(UNIFIED_CROP_DB)} crops against your season, soil and water..."):
            ctx = build_context(location, season, irrigation, group_ov, texture_ov, ph_ov, sq_ov)
            crops = recommend_crops(ctx, acres)
        st.session_state["crop_results"] = crops
        st.session_state["crop_params"] = {
            "temp": weather_current.get("temperature", 25) if weather_current else 25,
            "humidity": weather_current.get("humidity", 60) if weather_current else 60,
            "rainfall": normals["annual_rain"], "soil": texture_ov, "soil_group": group_ov,
            "soil_quality": sq_ov, "ph": ph_ov, "season": season, "acres": acres,
            "irrigation": irrigation, "elevation": (location or {}).get("elevation"),
        }

    crops = st.session_state.get("crop_results")
    if crops:
        params = st.session_state.get("crop_params", {})
        st.markdown(
            f'<div class="section-head"><div class="sh-title">🏆 Top crops for {params.get("season", "")} '
            f'on {params.get("acres", 0):.1f} acres</div><span>Scores out of 100 · money for your whole farm</span></div>',
            unsafe_allow_html=True)
        f1, f2 = st.columns([2.2, 1])
        with f1:
            cat_filter = st.radio("Show", ["All", "Field crops", "Vegetables & spices", "Fruit"], horizontal=True,
                                  key="crop_cat_filter", label_visibility="collapsed")
        with f2:
            sort_by = st.radio("Sort", ["Best match", "Highest profit"], horizontal=True,
                               key="crop_sort", label_visibility="collapsed")
        groups = {"Field crops": {"Cereal", "Millet", "Pulse", "Oilseed", "Cash crop"},
                  "Vegetables & spices": {"Vegetable", "Spice", "Fruit vegetable"}, "Fruit": {"Fruit"}}
        shown = [c for c in crops if cat_filter == "All" or c["category"] in groups[cat_filter]]
        if sort_by == "Highest profit":
            # only crops that actually suit the farm (score ≥ 60), then by average profit
            shown = sorted([c for c in shown if c["score"] >= 60],
                           key=lambda c: (c["profit_min"] + c["profit_max"]) / 2, reverse=True)
        for i, crop in enumerate(shown[:6], 1):
            render_crop_card(crop, i)
            c_left, c_right = st.columns([3, 1])
            with c_left:
                with st.expander(f"📋 Why {crop['name']}? — full breakdown"):
                    for r in crop["reasons"]:
                        st.markdown(r)
                    if crop.get("note"):
                        st.caption(f"ℹ️ {crop['note']}")
            with c_right:
                if st.button("📊 Predict Yield", key=f"yield_btn_{crop['name']}", use_container_width=True):
                    st.session_state["selected_yield_crop"] = crop["name"]
                    st.session_state["yield_auto_trigger"] = True
                    st.toast(f"{crop['icon']} {crop['name']} yield forecast is ready below ⬇️")

        with st.expander(f"📑 All {len(crops)} crops ranked"):
            import pandas as pd
            table = pd.DataFrame([{
                "Crop": f"{c['icon']} {c['name']}", "Type": c["category"], "Score": c["score"],
                "Season": c["season"], "Harvest (t)": f"{c['yield_min']:.1f}–{c['yield_max']:.1f}",
                "Profit": f"{_money(c['profit_min'])} – {_money(c['profit_max'])}",
                "MSP ₹/q": c["msp"] or "—", "In your region": "✅" if c["regional"] else "",
            } for c in crops])
            st.dataframe(table, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-head"><div class="sh-title">📊 Money comparison</div>'
                    '<span>Top 6 shown above</span></div>', unsafe_allow_html=True)
        render_revenue_comparison_chart(shown[:6])
        st.caption("Prices: MSP for Kharif 2026-27 and Rabi 2027-28 (Government of India), sugarcane FRP 2026-27; "
                   "other crops use typical mandi ranges, which change every season. Yields and costs are typical "
                   "figures — your results will vary.")
