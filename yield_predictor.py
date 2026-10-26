"""
AGRI-X AI — Yield Prediction Engine (Module 7) v3.5
Non-linear agronomic modeling + Limiting-factor Waterfall + Interactive Scenario Simulator ("What-If" Analysis).
"""
import streamlit as st
import plotly.graph_objects as go
import theme  # registers the light "agrix" Plotly template
import numpy as np
from datetime import datetime, timedelta

from crop_advisor import (
    UNIFIED_CROP_DB, calculate_soil_quality, estimate_annual_rainfall,
)


def _calc_weather_factor(value, optimal_range):
    """Calculate a 0.0-1.0 factor based on how close value is to optimal range (compatibility helper)."""
    lo, hi = optimal_range
    if lo <= value <= hi:
        return 1.0
    elif value < lo:
        diff = lo - value
        return max(0.3, 1.0 - diff / (lo * 0.5 + 1))
    else:
        diff = value - hi
        return max(0.3, 1.0 - diff / (hi * 0.5 + 1))


def _calc_cardinal_temp_factor(temp, t_min, t_opt_range, t_max):
    """
    Non-linear agronomic cardinal temperature response curve (Yin / Yan & Hunt model).
    Smooth plateau in optimal range with steep decline towards thermal limits.
    """
    opt_lo, opt_hi = t_opt_range
    if opt_lo <= temp <= opt_hi:
        return 1.0

    if temp <= t_min or temp >= t_max:
        return 0.25

    if temp < opt_lo:
        # Sub-optimal ascending curve
        t_opt = opt_lo
        denom = t_opt - t_min
        if denom <= 0:
            return 0.5
        ratio = (temp - t_min) / denom
        return max(0.25, min(1.0, float(ratio ** 1.35)))
    else:
        # Supra-optimal descending curve (heat stress)
        t_opt = opt_hi
        denom = t_max - t_opt
        if denom <= 0:
            return 0.5
        ratio = (t_max - temp) / denom
        return max(0.25, min(1.0, float(ratio ** 1.45)))


def _calc_humidity_vpd_factor(humidity, optimal_range):
    """Calculate humidity factor modeling vapor pressure deficit (VPD) stress."""
    lo, hi = optimal_range
    if lo <= humidity <= hi:
        return 1.0
    elif humidity < lo:
        diff = lo - humidity
        return max(0.35, 1.0 - (diff / 60.0) ** 1.2)
    else:
        diff = humidity - hi
        return max(0.40, 1.0 - (diff / 50.0) ** 1.3)


def _calc_water_balance_factor(monthly_rain, optimal_monthly_rain, irrigation=True):
    """Calculate water availability factor incorporating irrigation buffer."""
    lo, hi = optimal_monthly_rain
    effective_water = monthly_rain + (45.0 if irrigation else 0.0)

    if lo <= effective_water <= hi * 1.3:
        return 1.0
    elif effective_water < lo:
        deficit_ratio = effective_water / max(1.0, lo)
        return max(0.30, float(deficit_ratio ** 0.85))
    else:
        excess = effective_water - (hi * 1.3)
        return max(0.45, 1.0 - (excess / (hi * 1.5)))


def predict_yield(crop_name, area_acres, temperature, humidity, monthly_rainfall,
                  soil_quality=0.8, irrigation_access=True, elevation=None,
                  sim_irr_bonus=0.0, sim_soc_boost=0.0, sim_shade_bonus=0.0,
                  sim_water_bonus=0.0):
    """
    Predict yield using non-linear agronomic modeling and factor decomposition.
    Returns metrics, factor breakdowns for waterfall visualization, and recommendations.
    """
    crop = UNIFIED_CROP_DB.get(crop_name)
    if not crop:
        return None

    # Base potential yield per acre
    base_potential = crop.get("yield_max_per_acre", crop["yield_per_acre"] * 1.25)
    t_min = crop.get("temp_min", 10)
    t_max = crop.get("temp_max", 38)
    opt_temp = crop["optimal_temp"]
    opt_humid = crop["optimal_humidity"]
    opt_rain = crop["optimal_monthly_rain"]

    # Effective temperature (accounting for shade net / microclimate simulation)
    eff_temp = temperature - sim_shade_bonus if temperature > opt_temp[1] else temperature

    # Non-linear factors (0.25 to 1.0)
    temp_factor = _calc_cardinal_temp_factor(eff_temp, t_min, opt_temp, t_max)
    humid_factor = _calc_humidity_vpd_factor(humidity, opt_humid)
    rain_factor = _calc_water_balance_factor(monthly_rainfall + sim_water_bonus, opt_rain,
                                             irrigation=irrigation_access)

    # Soil fertility factor
    adj_soil_q = min(1.0, soil_quality + sim_soc_boost)
    soil_factor = max(0.35, 0.45 + 0.55 * adj_soil_q)

    # Elevation modifier
    elev_factor = 1.0
    if elevation is not None:
        if crop_name in ["Rice", "Sugarcane"] and elevation > 1500:
            elev_factor = 0.72
        elif crop_name in ["Wheat", "Potato", "Mustard"] and 400 < elevation < 2000:
            elev_factor = 1.05

    # Irrigation efficiency & Simulation modifier
    irr_factor = 1.0 + (0.12 if irrigation_access else 0.0) + sim_irr_bonus

    # Multiplicative agronomic yield index
    combined_factor = (
        (temp_factor ** 0.32) *
        (humid_factor ** 0.18) *
        (rain_factor ** 0.25) *
        (soil_factor ** 0.25) *
        elev_factor * irr_factor
    )
    combined_factor = max(0.20, min(1.05, combined_factor))

    estimated_yield_per_acre = round(base_potential * combined_factor, 2)
    total_yield = round(estimated_yield_per_acre * area_acres, 2)

    price = crop["price_per_tonne"]
    revenue = round(total_yield * price)
    harvest_date = datetime.now() + timedelta(days=int(crop["harvest_months"] * 30))

    confidence = min(0.95, (temp_factor + humid_factor + rain_factor + adj_soil_q) / 4 * 0.9 + 0.08)

    # Limiting-factor decomposition (per acre). Each factor is applied in turn to the
    # running yield, so the bars of the waterfall chart add up exactly to the final yield.
    running = base_potential
    stage_losses = []
    for factor, weight in ((temp_factor, 0.32), (humid_factor, 0.18),
                           (rain_factor, 0.25), (soil_factor, 0.25)):
        new_running = running * (factor ** weight)
        stage_losses.append(running - new_running)
        running = new_running
    loss_temp, loss_humid, loss_water, loss_soil = (round(x, 2) for x in stage_losses)
    # Whatever remains (irrigation boost, elevation, safety clamp) is the net adjustment
    gain_irr = round(estimated_yield_per_acre - (base_potential - loss_temp - loss_humid
                                                 - loss_water - loss_soil), 2)

    # Dynamic recommendations
    recs = []
    if temp_factor < 0.75:
        if eff_temp < opt_temp[0]:
            recs.append(f"🌡️ Low Temperature ({eff_temp:.1f}°C): Use polytunnel or black plastic mulch to warm root zone.")
        else:
            recs.append(f"🌡️ Heat Stress ({eff_temp:.1f}°C): Apply agro-shade net (35-50% shade) and increase morning irrigation.")
    if rain_factor < 0.75:
        if monthly_rainfall < opt_rain[0]:
            recs.append("💧 Water Deficit: Shift to micro-drip irrigation to conserve moisture and avoid stomatal closure.")
        else:
            recs.append("🌧️ Waterlogging: Construct raised planting beds and field drainage channels.")
    if adj_soil_q < 0.65:
        recs.append("🪨 Soil Organic Carbon: Apply 4–5 tonnes/acre well-decomposed FYM or vermicompost to boost water capacity.")
    if not irrigation_access:
        recs.append("🚰 Rainfed Risk: Install farm-pond rainwater harvesting or borewell drip integration.")
    if not recs:
        recs.append("✅ Favorable Agronomic Conditions: Standard crop nutrition and pest scouting will maximize yield.")

    return {
        "crop_name": crop_name,
        "icon": crop["icon"],
        "area_acres": area_acres,
        "base_potential": base_potential,
        "expected_yield": total_yield,
        "yield_per_acre": estimated_yield_per_acre,
        "expected_revenue": revenue,
        "harvest_date": harvest_date.strftime("%B %d, %Y"),
        "harvest_months": crop["harvest_months"],
        "confidence": round(confidence, 2),
        "price_per_tonne": price,
        "env_factor": round(combined_factor, 2),
        "temp_factor": round(temp_factor, 2),
        "humid_factor": round(humid_factor, 2),
        "rain_factor": round(rain_factor, 2),
        "soil_quality": round(adj_soil_q, 2),
        "loss_breakdown": {
            "potential": base_potential,
            "loss_temp": loss_temp,
            "loss_humid": loss_humid,
            "loss_water": loss_water,
            "loss_soil": loss_soil,
            "gain_irr": gain_irr,
            "final": estimated_yield_per_acre,
        },
        "recommendations": recs,
    }


def render_yield_prediction(result):
    """Render yield prediction results as high-impact glassmorphic cards."""
    if not result:
        st.warning("Yield prediction unavailable.")
        return

    conf = result["confidence"]
    conf_color = "#2E7D32" if conf >= 0.8 else "#B45309" if conf >= 0.6 else "#B91C1C"

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {conf_color};margin-bottom:1rem;padding:1.4rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
            <div>
                <span style="font-size:2.2rem;">{result['icon']}</span>
                <span style="font-size:1.4rem;font-weight:800;color:#1E2A22;margin-left:0.5rem;">{result['crop_name']} Yield Forecast</span>
            </div>
            <span style="background:rgba(30,42,34,0.08);color:{conf_color};padding:6px 16px;border-radius:14px;font-size:0.9rem;font-weight:700;">
                {conf:.0%} AI Confidence</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;">
            <div style="text-align:center;padding:0.9rem;background:rgba(30,42,34,0.04);border-radius:12px;">
                <div style="font-size:0.68rem;color:#6B776E;text-transform:uppercase;">Total Expected Yield</div>
                <div style="font-size:1.5rem;font-weight:800;color:#2E7D32;margin-top:0.2rem;">📦 {result['expected_yield']:.1f} t</div>
                <div style="font-size:0.75rem;color:#6B776E;">({result['area_acres']:.1f} acres)</div>
            </div>
            <div style="text-align:center;padding:0.9rem;background:rgba(30,42,34,0.04);border-radius:12px;">
                <div style="font-size:0.68rem;color:#6B776E;text-transform:uppercase;">Yield / Acre</div>
                <div style="font-size:1.5rem;font-weight:800;color:#2563A8;margin-top:0.2rem;">📐 {result['yield_per_acre']:.2f} t</div>
                <div style="font-size:0.75rem;color:#6B776E;">Stress-free potential: {result['base_potential']:.1f} t</div>
            </div>
            <div style="text-align:center;padding:0.9rem;background:rgba(30,42,34,0.04);border-radius:12px;">
                <div style="font-size:0.68rem;color:#6B776E;text-transform:uppercase;">Expected Revenue</div>
                <div style="font-size:1.5rem;font-weight:800;color:#B45309;margin-top:0.2rem;">💰 ₹{result['expected_revenue']:,}</div>
                <div style="font-size:0.75rem;color:#6B776E;">@ ₹{result['price_per_tonne']:,}/t</div>
            </div>
            <div style="text-align:center;padding:0.9rem;background:rgba(30,42,34,0.04);border-radius:12px;">
                <div style="font-size:0.68rem;color:#6B776E;text-transform:uppercase;">Estimated Harvest</div>
                <div style="font-size:1.2rem;font-weight:700;color:#6D28D9;margin-top:0.3rem;">🗓️ {result['harvest_date']}</div>
                <div style="font-size:0.75rem;color:#6B776E;">~{result['harvest_months']} months cycle</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_yield_waterfall_chart(result):
    """Render a Plotly Waterfall chart visualizing potential yield decomposition and limiting factors."""
    if not result or "loss_breakdown" not in result:
        return

    b = result["loss_breakdown"]
    fig = go.Figure(go.Waterfall(
        name="Yield Factors",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
        x=["Potential Yield", "Thermal Stress", "Humidity / VPD", "Water Deficit", "Soil Fertility", "Irrigation & Terrain", "Final Yield"],
        textposition="outside",
        text=[
            f"{b['potential']:.2f} t",
            f"-{b['loss_temp']:.2f} t" if b['loss_temp'] > 0 else "0",
            f"-{b['loss_humid']:.2f} t" if b['loss_humid'] > 0 else "0",
            f"-{b['loss_water']:.2f} t" if b['loss_water'] > 0 else "0",
            f"-{b['loss_soil']:.2f} t" if b['loss_soil'] > 0 else "0",
            f"{b['gain_irr']:+.2f} t" if b['gain_irr'] != 0 else "0",
            f"{b['final']:.2f} t"
        ],
        y=[
            b["potential"],
            -b["loss_temp"],
            -b["loss_humid"],
            -b["loss_water"],
            -b["loss_soil"],
            b["gain_irr"],
            b["final"]
        ],
        connector={"line": {"color": "rgba(30,42,34,0.2)"}},
        decreasing={"marker": {"color": "#B91C1C"}},
        increasing={"marker": {"color": "#2E7D32"}},
        totals={"marker": {"color": "#2563A8"}},
    ))

    fig.update_layout(
        title="📉 Yield Limiting-Factor Decomposition (Tonnes / Acre)",
        template="agrix", height=380, margin=dict(t=50, b=30, l=40, r=20),
        yaxis_title="Yield (t / acre)", showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_yield_factors_chart(result):
    """Render a radar chart of agronomic factors."""
    if not result:
        return
    categories = ["Thermal Feasibility", "Atmospheric Humidity", "Water Balance", "Soil Quality", "Overall Index"]
    values = [result["temp_factor"], result["humid_factor"],
              result["rain_factor"], result["soil_quality"], result["env_factor"]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill='toself', name='Agronomic Index',
        line=dict(color='#2E7D32', width=2),
        fillcolor='rgba(46,125,50,0.25)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
        template="agrix", height=380, margin=dict(t=40, b=30, l=60, r=60),
        title="🌍 Multi-Factor Suitability Radar",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_scenario_simulator(crop, area, temp, humidity, monthly_rain, soil_quality, irrigation, elevation):
    """
    Render an interactive "What-If" scenario simulator enabling farmers to model
    the yield and revenue impact of farm interventions.
    """
    with st.expander("💡 Interactive Scenario Simulator ('What-If' Analysis)", expanded=False):
        st.markdown('<p style="color:#6B776E;font-size:0.88rem;">Simulate agricultural interventions (e.g. drip irrigation, compost, shade nets) to see projected yield and profit improvements.</p>', unsafe_allow_html=True)

        sc_col1, sc_col2, sc_col3 = st.columns(3)
        with sc_col1:
            irr_method = st.selectbox(
                "💧 Irrigation Enhancement",
                ["Current Practice", "Precision Drip Irrigation (+18% Yield)", "Solar Micro-Sprinkler (+10% Yield)"],
                key="sim_irr_choice"
            )
        with sc_col2:
            soil_amend = st.selectbox(
                "🪱 Soil Amendments",
                ["None", "Add Vermicompost (+15% SOC)", "Add Biochar + Mycorrhiza (+30% SOC)"],
                key="sim_soil_choice"
            )
        with sc_col3:
            micro_climate = st.selectbox(
                "🛡️ Microclimate Protection",
                ["None", "Agro Shade Net (Reduces Heat Stress by 3°C)", "Reflective Plastic Mulch (+10% Moisture Retention)"],
                key="sim_shade_choice"
            )

        # Map simulator choices to parameter boosts
        sim_irr_bonus = 0.18 if "Drip" in irr_method else (0.10 if "Sprinkler" in irr_method else 0.0)
        sim_soc_boost = 0.15 if "+15%" in soil_amend else (0.30 if "+30%" in soil_amend else 0.0)
        sim_shade_bonus = 3.0 if "Shade Net" in micro_climate else 0.0
        sim_water_bonus = 15.0 if "Mulch" in micro_climate else 0.0  # mm/month retained

        # Baseline prediction
        base_res = predict_yield(
            crop, area, temp, humidity, monthly_rain, soil_quality, irrigation, elevation
        )
        # Simulated prediction
        sim_res = predict_yield(
            crop, area, temp, humidity, monthly_rain, soil_quality,
            irrigation or sim_irr_bonus > 0, elevation,
            sim_irr_bonus=sim_irr_bonus, sim_soc_boost=sim_soc_boost,
            sim_shade_bonus=sim_shade_bonus, sim_water_bonus=sim_water_bonus,
        )

        if base_res and sim_res:
            diff_yield = sim_res["expected_yield"] - base_res["expected_yield"]
            diff_rev = sim_res["expected_revenue"] - base_res["expected_revenue"]
            pct_gain = (diff_yield / max(0.1, base_res["expected_yield"])) * 100
            gain_color = "#2E7D32" if diff_yield >= 0 else "#B91C1C"
            rev_sign = "+" if diff_rev >= 0 else "−"

            st.markdown(f"""
            <div style="background:rgba(46,125,50,0.08);border:1px solid #2E7D32;border-radius:12px;padding:1rem;margin-top:0.8rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:0.8rem;color:#6B776E;text-transform:uppercase;">Simulated Total Yield</div>
                        <div style="font-size:1.4rem;font-weight:800;color:#2E7D32;">📦 {sim_res['expected_yield']:.1f} tonnes <span style="font-size:0.85rem;color:{gain_color};">({diff_yield:+.1f} t / {pct_gain:+.0f}%)</span></div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem;color:#6B776E;text-transform:uppercase;">Simulated Gross Revenue</div>
                        <div style="font-size:1.4rem;font-weight:800;color:#B45309;">💰 ₹{sim_res['expected_revenue']:,} <span style="font-size:0.85rem;color:{gain_color};">({rev_sign}₹{abs(diff_rev):,})</span></div>
                    </div>
                    <div>
                        <div style="font-size:0.8rem;color:#6B776E;text-transform:uppercase;">Efficiency Rating</div>
                        <div style="font-size:1.4rem;font-weight:800;color:#2563A8;">⚡ {sim_res['env_factor']:.0%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_yield_page(location, weather_current, weather_raw):
    """
    Full yield prediction page with non-linear models, waterfall analysis, and what-if simulation.
    """
    st.markdown('<div class="section-head"><div class="sh-title">📦 Yield forecast</div>'
                '<span>How much you can expect to harvest — and what holds it back</span></div>',
                unsafe_allow_html=True)

    temp = weather_current.get("temperature", 25) if weather_current else 25
    humidity = weather_current.get("humidity", 60) if weather_current else 60
    elevation = location.get("elevation") if location else None

    soil_data = location.get("soil_data") if location else None
    soil_quality, soil_breakdown = calculate_soil_quality(soil_data)

    rainfall_est, rainfall_conf, rainfall_method = estimate_annual_rainfall(location, weather_raw)

    # Reuse the overrides the farmer set in the Crop Advisor (soil fertility, rainfall)
    crop_params = st.session_state.get("crop_params", {})
    if crop_params.get("soil_quality") is not None:
        soil_quality = crop_params["soil_quality"]
    if crop_params.get("rainfall"):
        rainfall_est = crop_params["rainfall"]
        rainfall_method = "Crop Advisor setting"
    monthly_rain = rainfall_est / 12

    # Inputs the model uses (compact)
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:0.8rem;">'
        f'<span class="chip chip-gray">🌡️ {temp:.1f}°C</span>'
        f'<span class="chip chip-gray">💧 {humidity:.0f}% humidity</span>'
        f'<span class="chip chip-sky">🌧️ ~{monthly_rain:.0f} mm rain / month</span>'
        f'<span class="chip chip-green">🧪 Soil fertility {soil_quality:.0%}</span>'
        '</div>', unsafe_allow_html=True)

    # Crop selector — a crop picked via "Predict Yield" in the Crop Advisor is applied once.
    # (Setting the widget's session key is required: Streamlit ignores `index` once the
    # widget already exists.)
    crop_names = list(UNIFIED_CROP_DB.keys())
    selected = st.session_state.pop("selected_yield_crop", None)
    if selected in crop_names:
        st.session_state["yield_crop"] = selected

    with st.container(border=True, key="panel_yield_inputs"):
        col1, col2, col3 = st.columns([1.2, 1, 1.2])
        with col1:
            crop = st.selectbox("🌱 Crop", crop_names, key="yield_crop")
        with col2:
            default_area = float(crop_params.get("acres", 5.0))
            area = st.number_input(
                "🏞️ Land area (acres)", min_value=0.1, max_value=10000.0,
                value=default_area, step=0.5, key="yield_area",
            )
        with col3:
            st.write("")
            st.write("")
            irrigation = st.checkbox("🚰 Irrigation available", value=True, key="yield_irr")

    # Once requested (button here, or "Predict Yield" in the Crop Advisor), the forecast is
    # recomputed from the current inputs on every rerun, so it never goes stale when the
    # crop, area, irrigation or location changes. The model is pure arithmetic — cheap.
    if st.session_state.pop("yield_auto_trigger", False):
        st.session_state["yield_active"] = True
    if st.button("📊 Calculate Detailed Yield Forecast", type="primary", use_container_width=True, key="btn_yield"):
        st.session_state["yield_active"] = True

    result = None
    if st.session_state.get("yield_active"):
        result = predict_yield(
            crop, area, temp, humidity, monthly_rain,
            soil_quality, irrigation, elevation,
        )
        st.session_state["yield_result"] = result
    if result:
        render_yield_prediction(result)

        # Visualizations: Waterfall Chart & Radar Chart
        c1, c2 = st.columns([3, 2])
        with c1:
            render_yield_waterfall_chart(result)
        with c2:
            render_yield_factors_chart(result)

        # Interactive Scenario Simulator ("What-If" Analysis)
        render_scenario_simulator(crop, area, temp, humidity, monthly_rain, soil_quality, irrigation, elevation)

        # Agronomic Action Plan
        st.markdown("#### 📋 Agronomic Action Plan to Maximize Yield")
        for rec in result["recommendations"]:
            st.markdown(f"- {rec}")

        # Data Sources
        st.markdown("---")
        st.markdown("##### 📡 Live Data Diagnostics")
        soil_src = "ISRIC SoilGrids 2.0 (Dual Depth)" if soil_data else "Regional Agronomic Baseline"
        st.caption(f"""
        🌡️ Thermal Modeling: Open-Meteo API & Non-linear Cardinal Temperature Plateau
        🪨 Soil & Chemistry: {soil_src} (pH={soil_breakdown.get('ph_value', 6.5)}, SOC={soil_breakdown.get('soc_value', 10)} g/kg, AWHC={soil_breakdown.get('awhc_val', 140)} mm/m)
        🌧️ Water Balance: {rainfall_method} (~{rainfall_est} mm/yr → ~{monthly_rain:.0f} mm/mo)
        ⛰️ Topography: {f'{elevation:.0f}m Elevation' if elevation is not None else 'Standard Lowland'}
        """)
