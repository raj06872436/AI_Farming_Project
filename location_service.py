"""
AGRI-X AI — Location Service (Module 1) v3.0
Automatic geolocation via Browser GPS / IP + dual-depth soil analysis + AWHC + PyDeck map.
"""
import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pydeck as pdk

from india_agro_data import find_profile, regional_soil_data

# ── Default fallback location (New Delhi, India) ──
DEFAULT_LOCATION = {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "city": "New Delhi",
    "state": "Delhi",
    "country": "India",
    "source": "default",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _ip_geolocation():
    """Detect location from public IP using free ip-api.com."""
    try:
        resp = requests.get(
            "http://ip-api.com/json/?fields=status,city,regionName,country,lat,lon",
            timeout=5,
        )
        data = resp.json()
        if data.get("status") == "success":
            return {
                "latitude": data["lat"],
                "longitude": data["lon"],
                "city": data.get("city", "Unknown"),
                "state": data.get("regionName", "Unknown"),
                "country": data.get("country", "Unknown"),
                "source": "ip-api",
            }
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> dict:
    """
    Convert latitude/longitude to a place name using free APIs.
    Priority: Nominatim (OpenStreetMap) → BigDataCloud fallback.
    Returns dict with city, district, state, country or None.
    """
    # --- Nominatim (OpenStreetMap) ---
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1, "zoom": 10},
            headers={"User-Agent": "AGRI-X-AI/3.0 (Agricultural Research Platform)"},
            timeout=6,
        )
        data = resp.json()
        addr = data.get("address", {})
        if addr:
            city = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("county")
                or addr.get("state_district")
                or "Unknown"
            )
            state = addr.get("state", addr.get("region", "Unknown"))
            country = addr.get("country", "Unknown")
            district = addr.get("state_district", addr.get("county", city))
            return {"city": city, "district": district, "state": state, "country": country}
    except Exception:
        pass

    # --- BigDataCloud fallback ---
    try:
        resp = requests.get(
            "https://api.bigdatacloud.net/data/reverse-geocode-client",
            params={"latitude": lat, "longitude": lon, "localityLanguage": "en"},
            timeout=6,
        )
        data = resp.json()
        if data:
            return {
                "city": data.get("city", data.get("locality", "Unknown")),
                "district": data.get("localityInfo", {}).get("administrative", [{}])[0].get("name", "Unknown"),
                "state": data.get("principalSubdivision", "Unknown"),
                "country": data.get("countryName", "Unknown"),
            }
    except Exception:
        pass

    return None


@st.cache_data(ttl=86400, show_spinner=False)
def forward_geocode(query: str) -> list:
    """
    Convert a city/place name to latitude/longitude using Nominatim.
    Returns a list of matching results (up to 5).
    """
    import math

    kind_names = {
        "city": "City", "town": "Town", "village": "Village", "hamlet": "Village",
        "suburb": "Locality", "neighbourhood": "Locality", "quarter": "Locality",
        "county": "Taluka / Tehsil", "municipality": "Town", "state_district": "District",
        "district": "District", "administrative": "Area", "state": "State",
    }
    results = []
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "addressdetails": 1, "limit": 10,
                    "accept-language": "en"},
            headers={"User-Agent": "AGRI-X-AI/3.0 (Agricultural Research Platform)"},
            timeout=8,
        )
        data = resp.json()
    except Exception:
        return results

    def _km(a, b):
        dlat = math.radians(b["latitude"] - a["latitude"])
        dlon = math.radians(b["longitude"] - a["longitude"]) * math.cos(math.radians(a["latitude"]))
        return 6371 * math.hypot(dlat, dlon)

    for item in data:
        addr = item.get("address", {})
        place = (addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet")
                 or addr.get("suburb") or addr.get("county") or addr.get("state_district")
                 or item.get("name") or query.title())
        kind_key = item.get("addresstype") or item.get("type") or ""
        entry = {
            "latitude": float(item["lat"]),
            "longitude": float(item["lon"]),
            "display_name": item.get("display_name", ""),
            "city": place,
            "district": addr.get("state_district") or addr.get("county") or "",
            "state": addr.get("state", addr.get("region", "")),
            "country": addr.get("country", ""),
            "kind": kind_names.get(kind_key, kind_key.replace("_", " ").title() or "Place"),
        }
        # Nominatim often returns the same town several times (city, its boundary,
        # its taluka...). Merge results that are the same place within ~8 km.
        if any(r["city"].lower() == entry["city"].lower() and _km(r, entry) < 8 for r in results):
            continue
        results.append(entry)
        if len(results) >= 6:
            break
    return results


@st.cache_data(ttl=86400, show_spinner=False)
def get_elevation(lat: float, lon: float) -> float:
    """Get elevation in meters from Open-Meteo Elevation API."""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": lat, "longitude": lon},
            timeout=5,
        )
        data = resp.json()
        elev = data.get("elevation", [None])
        return elev[0] if isinstance(elev, list) and elev else elev
    except Exception:
        return None


def _calc_awhc(clay: float, sand: float, silt: float, soc: float) -> float:
    """
    Estimate Available Water Holding Capacity (AWHC) in mm/m depth using Saxton-Rawls pedotransfer logic.
    AWHC = Field Capacity (FC) - Permanent Wilting Point (PWP).
    """
    fc = 0.15 * sand + 0.38 * silt + 0.42 * clay + 0.02 * soc
    pwp = 0.05 * sand + 0.12 * silt + 0.28 * clay
    awhc_pct = max(5.0, min(30.0, fc - pwp))
    return round(awhc_pct * 10, 1)


def _query_soilgrids(lat: float, lon: float):
    """
    One SoilGrids request. Returns (data, status) where status is
    "ok", "empty" (service answered but has no data here, e.g. a city centre)
    or "error" (service down / timed out).
    """
    try:
        resp = requests.get(
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
            params={
                "lon": round(lon, 4), "lat": round(lat, 4),
                "property": ["clay", "sand", "silt", "phh2o", "soc", "bdod", "nitrogen", "cec"],
                "depth": ["0-5cm", "15-30cm"],
                "value": "mean",
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None, "error"
        layers = resp.json().get("properties", {}).get("layers", [])
    except Exception:
        return None, "error"

    result = {"topsoil": {}, "subsoil": {}}
    for layer in layers:
        name = layer.get("name", "")
        for d in layer.get("depths", []):
            val = d.get("values", {}).get("mean")
            if val is None:
                continue
            target = result["topsoil"] if "0-5" in d.get("label", "") else result["subsoil"]
            if name in ("clay", "sand", "silt"):
                target[name] = {"value": round(val / 10.0, 1), "unit": "%"}                 # g/kg → %
            elif name == "phh2o":
                target["ph"] = {"value": round(val / 10.0, 2), "unit": ""}                 # pH×10 → pH
            elif name == "soc":
                target["organic_carbon"] = {"value": round(val / 10.0, 2), "unit": "g/kg"}  # dg/kg → g/kg
            elif name == "bdod":
                target["bulk_density"] = {"value": round(val / 100.0, 2), "unit": "g/cm³"}  # cg/cm³ → g/cm³
            elif name == "nitrogen":
                target["nitrogen"] = {"value": round(val / 100.0, 2), "unit": "g/kg"}       # cg/kg → g/kg
            elif name == "cec":
                target["cec"] = {"value": round(val / 10.0, 1), "unit": "cmol/kg"}          # mmol/kg → cmol/kg
    if not result["topsoil"].get("clay"):
        return None, "empty"
    return result, "ok"


def _finish_soil(result):
    """Add flat top-soil keys and water-holding capacity to a soil dict."""
    top = result.get("topsoil", {})
    for k, v in top.items():
        result.setdefault(k, v)
    c_top = top.get("clay", {}).get("value", 25)
    s_top = top.get("sand", {}).get("value", 40)
    si_top = top.get("silt", {}).get("value", 35)
    soc_top = top.get("organic_carbon", {}).get("value", 10)
    result["awhc_topsoil"] = _calc_awhc(c_top, s_top, si_top, soc_top)
    sub = result.get("subsoil", {})
    if sub:
        result["awhc_subsoil"] = _calc_awhc(
            sub.get("clay", {}).get("value", c_top), sub.get("sand", {}).get("value", s_top),
            sub.get("silt", {}).get("value", si_top), sub.get("organic_carbon", {}).get("value", max(2, soc_top * 0.5)))
    else:
        result["awhc_subsoil"] = result["awhc_topsoil"]
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def get_soil_data(lat: float, lon: float) -> dict:
    """
    Live dual-depth soil properties (0–5 cm and 15–30 cm) from ISRIC SoilGrids.
    SoilGrids has no data for built-up areas, so if the exact point is empty we
    sample farmland a few km around it. Returns None if the service is unavailable.
    """
    data, status = _query_soilgrids(lat, lon)
    if status == "empty":
        for dlat, dlon in ((0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03), (0.06, 0.06), (-0.06, -0.06)):
            data, status = _query_soilgrids(lat + dlat, lon + dlon)
            if status != "empty":
                break
    if not data:
        return None
    data = _finish_soil(data)
    data["source"] = "soilgrids"
    return data


def _attach_soil(loc: dict) -> dict:
    """
    Add soil + agro-region information to a location dict:
    live SoilGrids data when available, otherwise the region's typical soil
    (from india_agro_data) — never a blind "Loamy" guess.
    """
    profile = find_profile(loc)
    soil = get_soil_data(loc["latitude"], loc["longitude"])
    if not soil:
        soil = _finish_soil(regional_soil_data(profile))
        soil["source"] = "regional"
    loc["soil_data"] = soil
    loc["soil_type"] = _classify_soil(soil)
    loc["soil_group"] = profile["soil_group"]
    loc["soil_group_label"] = profile["soil_label"]
    loc["soil_note"] = profile["soil_note"]
    loc["agro_region"] = profile["name"]
    loc["agro_zone"] = profile["zone"]
    loc["region_crops"] = profile["crops"]
    loc["profile_match"] = profile["match"]
    return loc


def _classify_soil(soil: dict) -> str:
    """Soil texture classification based on USDA soil texture triangle."""
    if not soil:
        return "Loamy"
    clay = soil.get("clay", {}).get("value", soil.get("topsoil", {}).get("clay", {}).get("value", 25))
    sand = soil.get("sand", {}).get("value", soil.get("topsoil", {}).get("sand", {}).get("value", 40))
    silt = soil.get("silt", {}).get("value", soil.get("topsoil", {}).get("silt", {}).get("value", 35))

    if clay >= 40:
        if sand >= 45:
            return "Sandy Clay"
        elif silt >= 40:
            return "Silty Clay"
        return "Clay"
    elif clay >= 27:
        if sand >= 45:
            return "Sandy Clay Loam"
        elif sand <= 20:
            return "Silty Clay Loam"
        return "Clay Loam"
    elif sand >= 70:
        if clay >= 15:
            return "Sandy Loam"
        return "Sandy"
    elif silt >= 80:
        return "Silt"
    elif silt >= 50:
        return "Silt Loam"
    elif sand >= 52:
        return "Sandy Loam"
    else:
        return "Loam"


def set_manual_location(lat: float, lon: float, city: str, state: str, country: str, source: str = "manual",
                        district: str = ""):
    """Store a location in session state and update terrain & soil profiles."""
    loc = {
        "latitude": lat,
        "longitude": lon,
        "city": city,
        "district": district,
        "state": state,
        "country": country,
        "source": source,
        "elevation": get_elevation(lat, lon),
    }
    st.session_state["location"] = _attach_soil(loc)

    # Clear stale crop/yield cache
    for key in ["crop_results", "crop_params", "yield_result", "selected_yield_crop"]:
        st.session_state.pop(key, None)


def detect_location():
    """
    Return location dict. Priority:
    1. Session-state location
    2. URL parameters (from Browser GPS)
    3. IP-based geolocation
    4. Default fallback
    """
    # Check if URL query params contain GPS coordinates from browser button
    params = st.query_params
    if "gps_lat" in params and "gps_lon" in params:
        try:
            g_lat = float(params["gps_lat"])
            g_lon = float(params["gps_lon"])
            del st.query_params["gps_lat"]
            del st.query_params["gps_lon"]
            _apply_gps_coordinates(g_lat, g_lon)
            return st.session_state["location"]
        except Exception:
            pass

    if "location" in st.session_state and st.session_state["location"]:
        loc = st.session_state["location"]
        if "agro_region" not in loc or "soil_data" not in loc:
            if loc.get("elevation") is None:
                loc["elevation"] = get_elevation(loc["latitude"], loc["longitude"])
            _attach_soil(loc)
        return loc

    loc = _ip_geolocation()
    if not loc:
        loc = DEFAULT_LOCATION.copy()
    loc = dict(loc)
    loc["elevation"] = get_elevation(loc["latitude"], loc["longitude"])
    st.session_state["location"] = _attach_soil(loc)
    return st.session_state["location"]


def render_location_card(loc: dict):
    """Render a premium location display card with interactive PyDeck map preview."""
    source_badge = {
        "gps": ("📍 High-Accuracy Device GPS (5-10m)", "#2E7D32"),
        "ip-api": ("🌐 Auto-detected via IP", "#2563A8"),
        "manual": ("✏️ Manually entered", "#B45309"),
        "default": ("📍 Default (override recommended)", "#B91C1C"),
    }
    label, color = source_badge.get(loc.get("source", "default"), ("📍", "#6B776E"))

    lat, lon = loc["latitude"], loc["longitude"]

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {color};position:relative;overflow:hidden;margin-bottom:1rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
            <span style="font-size:1.2rem;font-weight:700;color:#1E2A22;">📍 Farm Geolocation & Field Center</span>
            <span style="font-size:0.75rem;font-weight:600;color:{color};background:{color}18;padding:5px 14px;border-radius:12px;border:1px solid {color}40;">{label}</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:0.8rem;">
            <div style="padding:0.7rem;background:rgba(30,42,34,0.03);border-radius:10px;">
                <div style="color:#6B776E;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">City / Village</div>
                <div style="color:#1E2A22;font-weight:700;font-size:1.05rem;margin-top:0.2rem;">{loc['city']}</div>
            </div>
            <div style="padding:0.7rem;background:rgba(30,42,34,0.03);border-radius:10px;">
                <div style="color:#6B776E;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">State / Province</div>
                <div style="color:#1E2A22;font-weight:700;font-size:1.05rem;margin-top:0.2rem;">{loc['state']}</div>
            </div>
            <div style="padding:0.7rem;background:rgba(30,42,34,0.03);border-radius:10px;">
                <div style="color:#6B776E;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">Country</div>
                <div style="color:#1E2A22;font-weight:700;font-size:1.05rem;margin-top:0.2rem;">{loc['country']}</div>
            </div>
            <div style="padding:0.7rem;background:rgba(30,42,34,0.03);border-radius:10px;">
                <div style="color:#6B776E;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.05em;">GPS Coordinates</div>
                <div style="color:#2E7D32;font-weight:700;font-size:1.05rem;margin-top:0.2rem;">{lat:.4f}°, {lon:.4f}°</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive PyDeck map
    map_df = pd.DataFrame([{"lat": lat, "lon": lon, "label": f"{loc['city']} ({lat:.4f}, {lon:.4f})"}])
    
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=11,
        pitch=35,
        bearing=0
    )

    pin_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_color=[46, 125, 50, 230],
        get_radius=800,
        pickable=True,
        stroked=True,
        filled=True,
        get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=2,
    )

    pulse_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_color=[46, 125, 50, 60],
        get_radius=2500,
        stroked=False,
        filled=True,
    )

    deck = pdk.Deck(
        layers=[pulse_layer, pin_layer],
        initial_view_state=view_state,
        map_provider="carto",          # free basemap — Mapbox styles need an API token
        map_style=pdk.map_styles.CARTO_LIGHT,
        tooltip={"text": "🌾 Field Center: {label}"}
    )
    st.pydeck_chart(deck)


def render_location_details(loc: dict):
    """Render elevation and dual-depth soil data for the location."""
    lat, lon = loc["latitude"], loc["longitude"]

    col_elev, col_soil = st.columns([1, 2])

    with col_elev:
        elev = loc.get("elevation")
        if elev is None:
            with st.spinner("Fetching elevation..."):
                elev = get_elevation(lat, lon)
                loc["elevation"] = elev
        if elev is not None:
            elev_color = "#B91C1C" if elev > 2000 else "#B45309" if elev > 500 else "#2E7D32"
            elev_label = "Highland / Montane" if elev > 2000 else "Mid-altitude Plateau" if elev > 500 else "Lowland / Plain"
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;border-top:3px solid {elev_color};padding:1.2rem;">
                <div style="font-size:2.4rem;margin-bottom:0.2rem;">⛰️</div>
                <div style="color:#6B776E;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;">Elevation & Topography</div>
                <div style="font-size:1.8rem;font-weight:800;color:{elev_color};margin:0.3rem 0;">{elev:.0f} m</div>
                <div style="font-size:0.8rem;color:#47554B;">{elev_label} · {elev * 3.281:.0f} ft</div>
                <div style="margin-top:0.8rem;padding:0.4rem;background:rgba(30,42,34,0.03);border-radius:8px;font-size:0.75rem;color:#6B776E;">
                    Thermal lapse adjustment active
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Elevation data unavailable.")

    with col_soil:
        soil = loc.get("soil_data")
        if soil is None or "agro_region" not in loc:
            with st.spinner("Fetching soil profile..."):
                _attach_soil(loc)
                soil = loc["soil_data"]
        if soil:
            live = soil.get("source") == "soilgrids"
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:0.6rem;">'
                f'<span class="chip chip-{"green" if live else "amber"}">'
                f'{"🛰️ Measured soil data (ISRIC SoilGrids)" if live else "📘 Typical soil for " + loc.get("agro_region", "your region")}</span>'
                f'<span class="chip chip-gray">🪨 {loc.get("soil_group_label", "")} soil</span>'
                f'<span class="chip chip-gray">🧭 {loc.get("agro_zone", "")}</span></div>'
                f'<div style="color:var(--text-3);font-size:0.85rem;margin-bottom:0.6rem;">{loc.get("soil_note", "")}'
                f'{"" if live else " Live soil data was unavailable, so these are typical values for the area — a soil test will be more accurate."}</div>',
                unsafe_allow_html=True)
            soil_type = loc.get("soil_type", _classify_soil(soil))
            ph = soil.get("ph", {}).get("value", 6.5)
            ph_color = "#2E7D32" if 6.0 <= ph <= 7.5 else "#B45309" if 5.5 <= ph <= 8.0 else "#B91C1C"
            ph_label = "Optimal" if 6.0 <= ph <= 7.5 else "Moderate" if 5.5 <= ph <= 8.0 else "Extreme"

            clay = soil.get("clay", {}).get("value", 25)
            sand = soil.get("sand", {}).get("value", 40)
            silt = soil.get("silt", {}).get("value", 35)
            soc = soil.get("organic_carbon", {}).get("value", 10)
            awhc = soil.get("awhc_topsoil", 140)

            # Subsoil comparison
            sub = soil.get("subsoil", {})
            c_sub = sub.get("clay", {}).get("value", clay)
            soc_sub = sub.get("organic_carbon", {}).get("value", max(2, soc * 0.5))

            st.markdown(f"""
            <div class="glass-card" style="border-top:3px solid #8B4513;padding:1.2rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
                    <div>
                        <div style="font-size:0.75rem;color:#6B776E;text-transform:uppercase;">Dominant Texture</div>
                        <div style="font-size:1.4rem;font-weight:800;color:#8B5E34;">🌱 {soil_type}</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:0.75rem;color:#6B776E;text-transform:uppercase;">Water Capacity (AWHC)</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#2563A8;">💧 {awhc} mm/m</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.75rem;color:#6B776E;text-transform:uppercase;">Soil pH</div>
                        <div style="font-size:1.4rem;font-weight:800;color:{ph_color};">{ph:.1f} <span style="font-size:0.7rem;color:#47554B;">{ph_label}</span></div>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:0.5rem;margin-bottom:0.8rem;">
                    <div style="text-align:center;padding:0.5rem;background:rgba(30,42,34,0.03);border-radius:8px;">
                        <div style="font-size:0.65rem;color:#6B776E;">CLAY (0-5cm)</div>
                        <div style="font-size:1rem;font-weight:700;color:#8B5E34;">{clay:.0f}%</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(30,42,34,0.03);border-radius:8px;">
                        <div style="font-size:0.65rem;color:#6B776E;">SAND (0-5cm)</div>
                        <div style="font-size:1rem;font-weight:700;color:#F4A460;">{sand:.0f}%</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(30,42,34,0.03);border-radius:8px;">
                        <div style="font-size:0.65rem;color:#6B776E;">SILT (0-5cm)</div>
                        <div style="font-size:1rem;font-weight:700;color:#DEB887;">{silt:.0f}%</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(30,42,34,0.03);border-radius:8px;">
                        <div style="font-size:0.65rem;color:#6B776E;">ORG. CARBON</div>
                        <div style="font-size:1rem;font-weight:700;color:#2E7D32;">{soc:.1f} g/kg</div>
                    </div>
                </div>
                <div style="font-size:0.78rem;color:#6B776E;background:rgba(30,42,34,0.02);padding:0.5rem;border-radius:8px;border:1px solid rgba(30,42,34,0.05);">
                    🧪 <b>Subsoil Profile (15–30 cm):</b> Clay: {c_sub:.0f}% · SOC: {soc_sub:.1f} g/kg · Root-Zone Water Retention: {"Good" if awhc >= 120 else "Moderate" if awhc >= 80 else "Low"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Soil data unavailable for this location.")


def _apply_gps_coordinates(lat: float, lon: float):
    """Resolve GPS coordinates to a place name and make them the active farm location."""
    lat, lon = round(float(lat), 4), round(float(lon), 4)
    rev = reverse_geocode(lat, lon) or {}
    set_manual_location(
        lat, lon,
        rev.get("city") or "My Field",
        rev.get("state") or "",
        rev.get("country") or "India",
        source="gps",
        district=rev.get("district") or "",
    )
    st.session_state["weather_raw"] = None
    st.session_state["weather_current"] = {}


def render_browser_gps_button():
    """
    One-click device GPS.

    Uses `streamlit-js-eval` to read navigator.geolocation and send the coordinates
    straight back to Python (no page reload, session is kept). If that package is not
    installed, falls back to a plain HTML button.
    """
    try:
        from streamlit_js_eval import get_geolocation
    except ImportError:
        _render_gps_fallback_html()
        return

    if st.button("📍 Detect My Precise Device GPS (1-Click)", key="btn_device_gps",
                 use_container_width=True):
        st.session_state["_gps_pending"] = True
        st.session_state["_gps_attempt"] = st.session_state.get("_gps_attempt", 0) + 1

    if not st.session_state.get("_gps_pending"):
        return

    st.caption("⏳ Waiting for your browser — please allow location access in the popup.")
    # A fresh key per attempt so a retry asks the browser again instead of reusing the last answer
    result = get_geolocation(component_key=f"gps_geolocation_{st.session_state['_gps_attempt']}")
    if not result:
        return  # component still waiting for the browser

    st.session_state["_gps_pending"] = False
    coords = result.get("coords") if isinstance(result, dict) else None
    if coords and coords.get("latitude") is not None:
        _apply_gps_coordinates(coords["latitude"], coords["longitude"])
        st.rerun()
    else:
        code = (result.get("error") or {}).get("code") if isinstance(result, dict) else None
        if code == 1:
            st.warning("⚠️ Location permission denied by the browser. Use the search below instead.")
        else:
            st.warning("⚠️ GPS signal unavailable. Use the search below instead.")


def _render_gps_fallback_html():
    """
    GPS button for installs without streamlit-js-eval.

    Streamlit sandboxes components in an iframe that may not navigate the main page, so we
    try that first and otherwise open the app in a new tab with the coordinates attached
    (detect_location() picks up ?gps_lat=&gps_lon=). The coordinates are also shown so they
    can be typed into the "Enter Exact Coordinates" tab.
    """
    gps_html = """
    <div style="margin: 0.5rem 0 1rem 0;">
        <button id="gps-btn" onclick="getLocation()" style="
            background: linear-gradient(135deg, #2E7D32, #1B5E20); color: white; border: none;
            padding: 10px 18px; font-size: 0.95rem; font-weight: 700; border-radius: 10px;
            cursor: pointer; width: 100%; box-shadow: 0 4px 14px rgba(46,125,50,0.3);">
            📍 Detect My Precise Device GPS (1-Click)
        </button>
        <p id="gps-status" style="font-size: 0.78rem; color: #6B776E; margin-top: 6px; text-align: center;"></p>
    </div>
    <script>
    function getLocation() {
        const btn = document.getElementById("gps-btn");
        const status = document.getElementById("gps-status");
        if (!navigator.geolocation) {
            status.innerText = "Geolocation is not supported by this browser.";
            return;
        }
        btn.innerHTML = "⏳ Acquiring GPS...";
        status.innerText = "Please allow location access in your browser popup.";
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude.toFixed(4);
                const lon = position.coords.longitude.toFixed(4);
                btn.innerHTML = "✅ GPS Found (±" + Math.round(position.coords.accuracy) + "m)";
                status.innerText = "Coordinates: " + lat + ", " + lon +
                    " — enter these under 'Enter Exact Coordinates' if the page does not update.";
                let target;
                try { target = new URL(window.top.location.href); }
                catch (e) { target = new URL(document.referrer || window.location.href); }
                target.searchParams.set("gps_lat", lat);
                target.searchParams.set("gps_lon", lon);
                try { window.top.location.href = target.toString(); }
                catch (e) { window.open(target.toString(), "_blank"); }
            },
            function(error) {
                btn.innerHTML = "📍 Retry Device GPS";
                status.innerText = error.code == error.PERMISSION_DENIED
                    ? "⚠️ Location permission denied by browser. Use search below."
                    : "⚠️ GPS signal unavailable. Use search below.";
            },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
        );
    }
    </script>
    """
    components.html(gps_html, height=90)


def render_location_override_form():
    """Render manual location override form with 1-click GPS, city-name search, and coordinate entry."""
    flash = st.session_state.pop("_location_flash", None)
    if flash:
        st.success(flash)
    with st.expander("✏️ Change Location & Set Farm Coordinates",
                     expanded=bool(st.session_state.get("_city_matches"))):
        render_browser_gps_button()

        tab_city, tab_coords = st.tabs(["🏙️ Search by City / Village", "🌐 Enter Exact Coordinates"])

        # ── Tab 1: City Name Search ──
        with tab_city:
            city_query = st.text_input(
                "Enter city, district or village name",
                placeholder="e.g. Nashik, Ludhiana, Guntur, Anand...",
                key="city_search_query",
            )
            if st.button("🔍 Search Place", key="btn_city_search", use_container_width=True):
                if city_query.strip():
                    with st.spinner(f"Searching for '{city_query}'..."):
                        matches = forward_geocode(city_query.strip())
                    if matches:
                        st.session_state["_city_matches"] = matches
                        st.success(f"Found {len(matches)} match(es).")
                    else:
                        st.session_state["_city_matches"] = []
                        st.warning("No results found. Try adding state or country.")
                else:
                    st.warning("Please enter a location query.")

            matches = st.session_state.get("_city_matches", [])
            if matches:
                st.caption("Pick the right place — one click sets it as your farm location.")
                for idx, m in enumerate(matches):
                    where = ", ".join(x for x in (m.get("district"), m.get("state"), m.get("country"))
                                      if x and x != m["city"])
                    card, action = st.columns([5, 1.4], vertical_alignment="center")
                    with card:
                        st.markdown(f"""
                        <div style="padding:0.65rem 0.9rem;background:var(--surface);border:1px solid var(--border);border-radius:12px;">
                            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                                <span style="font-weight:700;color:var(--text);font-size:0.95rem;">📍 {m['city']}</span>
                                <span class="chip chip-gray" style="font-size:0.7rem;padding:1px 8px;">{m.get('kind', 'Place')}</span>
                            </div>
                            <div style="color:var(--text-3);font-size:0.8rem;margin-top:0.2rem;">{where} · {m['latitude']:.4f}°, {m['longitude']:.4f}°</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with action:
                        if st.button("Use this", key=f"use_city_{idx}", type="primary" if idx == 0 else "secondary",
                                     use_container_width=True):
                            set_manual_location(m["latitude"], m["longitude"], m["city"], m["state"],
                                                m["country"], source="manual", district=m.get("district", ""))
                            st.session_state["weather_raw"] = None
                            st.session_state["weather_current"] = {}
                            st.session_state["_city_matches"] = []
                            st.session_state["_location_flash"] = f"✅ Location set to {m['city']}, {m['state']}"
                            st.rerun()

        # ── Tab 2: Coordinates (advanced) ──
        with tab_coords:
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("Latitude", value=28.6139, min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f", key="loc_lat")
            with col2:
                lon = st.number_input("Longitude", value=77.2090, min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f", key="loc_lon")

            if st.button("🔍 Resolve Place Name", key="btn_reverse_geo", use_container_width=True):
                with st.spinner("Resolving coordinates..."):
                    result = reverse_geocode(lat, lon)
                if result:
                    st.session_state["_geo_city"] = result["city"]
                    st.session_state["_geo_state"] = result["state"]
                    st.session_state["_geo_country"] = result["country"]
                    st.session_state["_geo_district"] = result.get("district", result["city"])
                    st.success(f"📍 Resolved: **{result['city']}**, {result['state']}, {result['country']}")
                else:
                    st.warning("Could not resolve reverse geocoding.")

            default_city = st.session_state.get("_geo_city", "New Delhi")
            default_state = st.session_state.get("_geo_state", "Delhi")
            default_country = st.session_state.get("_geo_country", "India")

            col3, col4 = st.columns(2)
            with col3:
                city = st.text_input("City / Village", value=default_city, key="loc_city")
                country = st.text_input("Country", value=default_country, key="loc_country")
            with col4:
                state = st.text_input("State / Province", value=default_state, key="loc_state")

            if st.button("📍 Set Exact Coordinates", type="primary", key="btn_set_loc", use_container_width=True):
                set_manual_location(lat, lon, city, state, country, source="manual",
                                    district=st.session_state.get("_geo_district", ""))
                st.session_state["weather_raw"] = None
                st.session_state["weather_current"] = {}
                st.success(f"✅ Location set to **{city}** ({lat:.4f}°, {lon:.4f}°)")
                st.rerun()
