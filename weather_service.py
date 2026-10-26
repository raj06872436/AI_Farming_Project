"""
AGRI-X AI — Weather Service (Module 2)
Real-time weather via Open-Meteo (free, no API key).
"""
import requests, streamlit as st
from datetime import datetime

WMO_CODES = {
    0: ("☀️", "Clear sky"), 1: ("🌤️", "Mainly clear"), 2: ("⛅", "Partly cloudy"),
    3: ("☁️", "Overcast"), 45: ("🌫️", "Foggy"), 48: ("🌫️", "Rime fog"),
    51: ("🌦️", "Light drizzle"), 53: ("🌦️", "Moderate drizzle"), 55: ("🌧️", "Dense drizzle"),
    61: ("🌧️", "Slight rain"), 63: ("🌧️", "Moderate rain"), 65: ("🌧️", "Heavy rain"),
    71: ("🌨️", "Slight snow"), 73: ("🌨️", "Moderate snow"), 75: ("❄️", "Heavy snow"),
    80: ("🌦️", "Slight showers"), 81: ("🌧️", "Moderate showers"), 82: ("⛈️", "Violent showers"),
    95: ("⛈️", "Thunderstorm"), 96: ("⛈️", "Thunderstorm+hail"), 99: ("⛈️", "Heavy thunderstorm"),
}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather(lat: float, lon: float):
    """Fetch current + hourly + daily weather from Open-Meteo."""
    try:
        params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,surface_pressure,cloud_cover,weather_code,uv_index",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,uv_index_max,wind_speed_10m_max",
            "timezone": "auto", "forecast_days": 7,
        }
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
        data = resp.json()
        return data if "current" in data else None
    except Exception:
        return None

def get_current_weather(weather_data):
    if not weather_data or "current" not in weather_data:
        return {}
    c = weather_data["current"]
    code = c.get("weather_code", 0)
    icon, desc = WMO_CODES.get(code, ("🌡️", "Unknown"))
    return {
        "temperature": c.get("temperature_2m", 0), "humidity": c.get("relative_humidity_2m", 0),
        "precipitation": c.get("precipitation", 0), "wind_speed": c.get("wind_speed_10m", 0),
        "pressure": c.get("surface_pressure", 0), "cloud_cover": c.get("cloud_cover", 0),
        "uv_index": c.get("uv_index", 0), "weather_code": code, "icon": icon, "description": desc,
    }

def get_hourly_forecast(weather_data, hours=24):
    if not weather_data or "hourly" not in weather_data:
        return []
    h = weather_data["hourly"]
    forecasts, now = [], datetime.now()
    for i in range(min(hours * 2, len(h.get("time", [])))):
        try:
            t = datetime.fromisoformat(h["time"][i])
            if t < now: continue
        except (ValueError, IndexError): continue
        code = h.get("weather_code", [0]*100)[i] if i < len(h.get("weather_code",[])) else 0
        icon, desc = WMO_CODES.get(code, ("🌡️", "Unknown"))
        forecasts.append({
            "time": h["time"][i], "hour": t.strftime("%I %p"),
            "temperature": h.get("temperature_2m", [0]*100)[i],
            "humidity": h.get("relative_humidity_2m", [0]*100)[i],
            "precip_prob": h.get("precipitation_probability", [0]*100)[i],
            "icon": icon, "description": desc,
        })
        if len(forecasts) >= hours: break
    return forecasts

def get_daily_forecast(weather_data):
    if not weather_data or "daily" not in weather_data:
        return []
    d = weather_data["daily"]
    days = []
    for i in range(len(d.get("time", []))):
        code = d.get("weather_code", [0]*7)[i] if i < len(d.get("weather_code",[])) else 0
        icon, desc = WMO_CODES.get(code, ("🌡️", "Unknown"))
        try:
            dt = datetime.fromisoformat(d["time"][i]); day_name = dt.strftime("%a %b %d")
        except (ValueError, IndexError):
            day_name = d["time"][i]
        days.append({
            "date": d["time"][i], "day_name": day_name,
            "temp_max": d.get("temperature_2m_max", [0]*7)[i],
            "temp_min": d.get("temperature_2m_min", [0]*7)[i],
            "precipitation": d.get("precipitation_sum", [0]*7)[i],
            "uv_index": d.get("uv_index_max", [0]*7)[i],
            "wind_speed": d.get("wind_speed_10m_max", [0]*7)[i],
            "icon": icon, "description": desc,
        })
    return days

def render_current_weather_card(current):
    if not current:
        st.warning("Weather data unavailable."); return
    temp = current["temperature"]
    tc = "#B91C1C" if temp >= 35 else "#B45309" if temp >= 25 else "#2E7D32" if temp >= 15 else "#2563A8"
    uv = current.get("uv_index", 0)
    uc = "#B91C1C" if uv >= 8 else "#B45309" if uv >= 5 else "#2E7D32"
    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {tc};padding:1.5rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
            <div><span style="font-size:2.5rem;">{current['icon']}</span>
            <span style="font-size:2rem;font-weight:800;color:{tc};margin-left:0.5rem;">{temp:.1f}°C</span></div>
            <div style="text-align:right;"><div style="color:#1E2A22;font-weight:600;">{current['description']}</div>
            <div style="color:#6B776E;font-size:0.8rem;">Current Conditions</div></div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;">
            <div style="text-align:center;padding:0.6rem;background:rgba(30,42,34,0.04);border-radius:10px;">
                <div style="font-size:0.7rem;color:#6B776E;">HUMIDITY</div>
                <div style="font-size:1.2rem;font-weight:700;color:#2563A8;">💧 {current['humidity']}%</div></div>
            <div style="text-align:center;padding:0.6rem;background:rgba(30,42,34,0.04);border-radius:10px;">
                <div style="font-size:0.7rem;color:#6B776E;">WIND</div>
                <div style="font-size:1.2rem;font-weight:700;color:#2E7D32;">💨 {current['wind_speed']:.1f} km/h</div></div>
            <div style="text-align:center;padding:0.6rem;background:rgba(30,42,34,0.04);border-radius:10px;">
                <div style="font-size:0.7rem;color:#6B776E;">RAINFALL</div>
                <div style="font-size:1.2rem;font-weight:700;color:#6D28D9;">🌧️ {current['precipitation']:.1f} mm</div></div>
            <div style="text-align:center;padding:0.6rem;background:rgba(30,42,34,0.04);border-radius:10px;">
                <div style="font-size:0.7rem;color:#6B776E;">PRESSURE</div>
                <div style="font-size:1.2rem;font-weight:700;color:#C2410C;">🔽 {current['pressure']:.0f} hPa</div></div>
            <div style="text-align:center;padding:0.6rem;background:rgba(30,42,34,0.04);border-radius:10px;">
                <div style="font-size:0.7rem;color:#6B776E;">UV INDEX</div>
                <div style="font-size:1.2rem;font-weight:700;color:{uc};">☀️ {uv:.1f}</div></div>
            <div style="text-align:center;padding:0.6rem;background:rgba(30,42,34,0.04);border-radius:10px;">
                <div style="font-size:0.7rem;color:#6B776E;">CLOUD COVER</div>
                <div style="font-size:1.2rem;font-weight:700;color:#6B776E;">☁️ {current['cloud_cover']}%</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

def render_hourly_forecast(hourly):
    if not hourly: return
    st.markdown("#### ⏰ Hourly Forecast")
    cards = ""
    for h in hourly[:12]:
        cards += f"""<div style="min-width:90px;text-align:center;padding:0.6rem;background:rgba(30,42,34,0.04);border-radius:12px;flex-shrink:0;">
            <div style="font-size:0.7rem;color:#6B776E;">{h['hour']}</div>
            <div style="font-size:1.4rem;margin:0.3rem 0;">{h['icon']}</div>
            <div style="font-size:1rem;font-weight:700;color:#1E2A22;">{h['temperature']:.0f}°</div>
            <div style="font-size:0.65rem;color:#2563A8;">💧{h['humidity']}%</div></div>"""
    st.markdown(f'<div style="display:flex;gap:0.5rem;overflow-x:auto;padding:0.5rem 0;">{cards}</div>', unsafe_allow_html=True)

def render_daily_forecast(daily):
    if not daily: return
    st.markdown("#### 📅 7-Day Forecast")
    for d in daily:
        bw = max(10, min(100, (d["temp_max"] - d["temp_min"]) * 5))
        st.markdown(f"""<div style="display:flex;align-items:center;padding:0.5rem 0.8rem;margin:0.3rem 0;background:rgba(30,42,34,0.03);border-radius:10px;">
            <div style="width:100px;font-size:0.85rem;color:#47554B;">{d['day_name']}</div>
            <div style="width:40px;font-size:1.2rem;text-align:center;">{d['icon']}</div>
            <div style="flex:1;display:flex;align-items:center;gap:0.5rem;margin:0 1rem;">
                <span style="font-size:0.85rem;color:#2563A8;min-width:35px;">{d['temp_min']:.0f}°</span>
                <div style="flex:1;height:6px;background:rgba(30,42,34,0.08);border-radius:3px;overflow:hidden;">
                    <div style="width:{bw}%;height:100%;background:linear-gradient(90deg,#2563A8,#B91C1C);border-radius:3px;"></div></div>
                <span style="font-size:0.85rem;color:#B91C1C;min-width:35px;">{d['temp_max']:.0f}°</span></div>
            <div style="width:60px;font-size:0.75rem;color:#6D28D9;text-align:right;">🌧️{d['precipitation']:.1f}mm</div></div>""", unsafe_allow_html=True)
