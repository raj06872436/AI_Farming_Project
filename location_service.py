"""
AGRI-X AI — Location Service (Module 1) v2.0
Automatic geolocation via IP + manual override + reverse geocoding + map display.
"""
import requests
import streamlit as st

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
    Returns dict with city, state, country or None.
    """
    # --- Nominatim (OpenStreetMap) ---
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1, "zoom": 10},
            headers={"User-Agent": "AGRI-X-AI/2.0 (Agricultural Research)"},
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
    Returns a list of matching results (up to 5), each with
    lat, lon, display_name, city, state, country.
    """
    results = []
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "addressdetails": 1, "limit": 5},
            headers={"User-Agent": "AGRI-X-AI/2.0 (Agricultural Research)"},
            timeout=8,
        )
        data = resp.json()
        for item in data:
            addr = item.get("address", {})
            city = (
                addr.get("city")
                or addr.get("town")
                or addr.get("village")
                or addr.get("county")
                or addr.get("state_district")
                or query.title()
            )
            state = addr.get("state", addr.get("region", ""))
            country = addr.get("country", "")
            results.append({
                "latitude": float(item["lat"]),
                "longitude": float(item["lon"]),
                "display_name": item.get("display_name", ""),
                "city": city,
                "state": state,
                "country": country,
            })
    except Exception:
        pass
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


@st.cache_data(ttl=86400, show_spinner=False)
def get_soil_data(lat: float, lon: float) -> dict:
    """Get soil type/properties from OpenLandMap / ISRIC (free)."""
    try:
        # SoilGrids REST API — top-layer clay/sand/silt percentages + pH + SOC
        resp = requests.get(
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
            params={
                "lon": lon, "lat": lat,
                "property": "clay,sand,silt,phh2o,soc",
                "depth": "0-5cm",
                "value": "mean",
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
        data = resp.json()
        layers = data.get("properties", {}).get("layers", [])
        result = {}
        for layer in layers:
            name = layer.get("name", "")
            depths = layer.get("depths", [])
            if depths:
                val = depths[0].get("values", {}).get("mean", None)
                unit = layer.get("unit_measure", {}).get("mapped_units", "")
                if val is not None:
                    # SoilGrids returns g/kg for clay/sand/silt, convert to %
                    if name in ("clay", "sand", "silt"):
                        result[name] = {"value": val / 10.0, "unit": "%"}
                    elif name == "phh2o":
                        result["ph"] = {"value": val / 10.0, "unit": ""}
                    elif name == "soc":
                        result["organic_carbon"] = {"value": val / 10.0, "unit": "g/kg"}
        return result if result else None
    except Exception:
        return None


def _classify_soil(soil: dict) -> str:
    """Simple soil texture classification from clay/sand/silt percentages."""
    clay = soil.get("clay", {}).get("value", 0)
    sand = soil.get("sand", {}).get("value", 0)
    silt = soil.get("silt", {}).get("value", 0)
    if clay > 40:
        return "Clay"
    elif sand > 70:
        return "Sandy"
    elif silt > 60:
        return "Silty"
    elif clay > 25 and sand > 25:
        return "Clay Loam"
    elif sand > 50:
        return "Sandy Loam"
    elif silt > 40:
        return "Silt Loam"
    else:
        return "Loam"



def set_manual_location(lat, lon, city, state, country):
    """Store a manually entered location in session state."""
    elev = get_elevation(lat, lon)
    soil = get_soil_data(lat, lon)
    soil_type = _classify_soil(soil) if soil else "Loamy"
    
    st.session_state["location"] = {
        "latitude": lat,
        "longitude": lon,
        "city": city,
        "state": state,
        "country": country,
        "source": "manual",
        "elevation": elev,
        "soil_data": soil,
        "soil_type": soil_type,
    }

    # Clear stale crop/yield results from previous location
    for key in ["crop_results", "crop_params", "yield_result", "selected_yield_crop"]:
        st.session_state.pop(key, None)


def detect_location():
    """
    Return location dict. Priority:
    1. Session-state manual override
    2. IP-based geolocation
    3. Default fallback
    """
    if "location" in st.session_state and st.session_state["location"]:
        loc = st.session_state["location"]
        if "soil_type" not in loc:
            lat, lon = loc["latitude"], loc["longitude"]
            loc["elevation"] = get_elevation(lat, lon)
            loc["soil_data"] = get_soil_data(lat, lon)
            loc["soil_type"] = _classify_soil(loc["soil_data"]) if loc["soil_data"] else "Loamy"
        return loc

    loc = _ip_geolocation()
    if loc:
        lat, lon = loc["latitude"], loc["longitude"]
        loc["elevation"] = get_elevation(lat, lon)
        loc["soil_data"] = get_soil_data(lat, lon)
        loc["soil_type"] = _classify_soil(loc["soil_data"]) if loc["soil_data"] else "Loamy"
        st.session_state["location"] = loc
        return loc

    loc = DEFAULT_LOCATION.copy()
    lat, lon = loc["latitude"], loc["longitude"]
    loc["elevation"] = get_elevation(lat, lon)
    loc["soil_data"] = get_soil_data(lat, lon)
    loc["soil_type"] = _classify_soil(loc["soil_data"]) if loc["soil_data"] else "Loamy"
    st.session_state["location"] = loc
    return loc


def render_location_card(loc: dict):
    """Render a premium location display card with map preview."""
    source_badge = {
        "ip-api": ("🌐 Auto-detected via IP", "#3498db"),
        "manual": ("✏️ Manually entered", "#f39c12"),
        "default": ("📍 Default (override recommended)", "#e74c3c"),
    }
    label, color = source_badge.get(loc.get("source", "default"), ("📍", "#888"))

    lat, lon = loc["latitude"], loc["longitude"]

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {color};position:relative;overflow:hidden;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
            <span style="font-size:1.2rem;font-weight:700;color:#fff;">📍 Your Location</span>
            <span style="font-size:0.72rem;color:{color};background:{color}15;padding:4px 12px;border-radius:12px;border:1px solid {color}30;">{label}</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.8rem;">
            <div style="padding:0.6rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="color:#8899aa;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;">City</div>
                <div style="color:#fff;font-weight:700;font-size:1.05rem;margin-top:0.2rem;">{loc['city']}</div>
            </div>
            <div style="padding:0.6rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="color:#8899aa;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;">State</div>
                <div style="color:#fff;font-weight:700;font-size:1.05rem;margin-top:0.2rem;">{loc['state']}</div>
            </div>
            <div style="padding:0.6rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="color:#8899aa;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;">Country</div>
                <div style="color:#fff;font-weight:700;font-size:1.05rem;margin-top:0.2rem;">{loc['country']}</div>
            </div>
            <div style="padding:0.6rem;background:rgba(255,255,255,0.03);border-radius:10px;">
                <div style="color:#8899aa;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;">Coordinates</div>
                <div style="color:#2ecc71;font-weight:700;font-size:1.05rem;margin-top:0.2rem;">{lat:.4f}°, {lon:.4f}°</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive map
    import pandas as pd
    map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(map_df, zoom=10, use_container_width=True)


def render_location_details(loc: dict):
    """Render elevation and soil data for the location."""
    lat, lon = loc["latitude"], loc["longitude"]

    col_elev, col_soil = st.columns(2)

    with col_elev:
        elev = loc.get("elevation")
        if elev is None:
            with st.spinner("Fetching elevation..."):
                elev = get_elevation(lat, lon)
                loc["elevation"] = elev
        if elev is not None:
            elev_color = "#e74c3c" if elev > 2000 else "#f39c12" if elev > 500 else "#2ecc71"
            elev_label = "Highland" if elev > 2000 else "Mid-altitude" if elev > 500 else "Lowland"
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;border-top:3px solid {elev_color};">
                <div style="font-size:2.5rem;margin-bottom:0.3rem;">⛰️</div>
                <div style="color:#8899aa;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.05em;">Elevation</div>
                <div style="font-size:1.8rem;font-weight:800;color:{elev_color};margin:0.3rem 0;">{elev:.0f} m</div>
                <div style="font-size:0.8rem;color:#aabbcc;">{elev_label} · {elev * 3.281:.0f} ft</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Elevation data unavailable.")

    with col_soil:
        soil = loc.get("soil_data")
        if soil is None and "soil_data" not in loc:
            with st.spinner("Fetching soil data..."):
                soil = get_soil_data(lat, lon)
                loc["soil_data"] = soil
                loc["soil_type"] = _classify_soil(soil) if soil else "Loamy"
        if soil:
            soil_type = loc.get("soil_type", _classify_soil(soil))
            ph = soil.get("ph", {}).get("value", 0)
            ph_color = "#2ecc71" if 6.0 <= ph <= 7.5 else "#f39c12" if 5.5 <= ph <= 8.0 else "#e74c3c"
            ph_label = "Optimal" if 6.0 <= ph <= 7.5 else "Acceptable" if 5.5 <= ph <= 8.0 else "Extreme"
            clay = soil.get("clay", {}).get("value", 0)
            sand = soil.get("sand", {}).get("value", 0)
            silt = soil.get("silt", {}).get("value", 0)
            soc = soil.get("organic_carbon", {}).get("value", 0)
            st.markdown(f"""
            <div class="glass-card" style="border-top:3px solid #8B4513;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
                    <div>
                        <div style="font-size:0.75rem;color:#8899aa;text-transform:uppercase;">Soil Type</div>
                        <div style="font-size:1.4rem;font-weight:800;color:#D2B48C;">🌱 {soil_type}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.75rem;color:#8899aa;text-transform:uppercase;">pH Level</div>
                        <div style="font-size:1.4rem;font-weight:800;color:{ph_color};">{ph:.1f} <span style="font-size:0.7rem;color:#aab;">{ph_label}</span></div>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.5rem;">
                    <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:8px;">
                        <div style="font-size:0.65rem;color:#8899aa;">CLAY</div>
                        <div style="font-size:1rem;font-weight:700;color:#CD853F;">{clay:.0f}%</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:8px;">
                        <div style="font-size:0.65rem;color:#8899aa;">SAND</div>
                        <div style="font-size:1rem;font-weight:700;color:#F4A460;">{sand:.0f}%</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:8px;">
                        <div style="font-size:0.65rem;color:#8899aa;">SILT</div>
                        <div style="font-size:1rem;font-weight:700;color:#DEB887;">{silt:.0f}%</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(255,255,255,0.03);border-radius:8px;">
                        <div style="font-size:0.65rem;color:#8899aa;">ORG. C</div>
                        <div style="font-size:1rem;font-weight:700;color:#2ecc71;">{soc:.1f} g/kg</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Soil data unavailable for this location.")


def render_location_override_form():
    """Render manual location override form with city-name search and coordinate entry."""
    with st.expander("✏️ Change Location", expanded=False):
        tab_city, tab_coords = st.tabs(["🏙️ Search by City Name", "🌐 Enter Coordinates"])

        # ── Tab 1: City Name Search ──
        with tab_city:
            city_query = st.text_input(
                "Enter city or place name",
                placeholder="e.g. Mumbai, Bangalore, Pune...",
                key="city_search_query",
            )
            if st.button("🔍 Search", key="btn_city_search", use_container_width=True):
                if city_query.strip():
                    with st.spinner(f"Searching for '{city_query}'..."):
                        matches = forward_geocode(city_query.strip())
                    if matches:
                        st.session_state["_city_matches"] = matches
                        st.success(f"Found {len(matches)} result(s).")
                    else:
                        st.session_state["_city_matches"] = []
                        st.warning("No results found. Try a different spelling or add state/country.")
                else:
                    st.warning("Please enter a city name.")

            # Show search results as selectable cards
            matches = st.session_state.get("_city_matches", [])
            if matches:
                for idx, m in enumerate(matches):
                    selected = st.session_state.get("_selected_city_idx", 0) == idx
                    border_col = "#2ecc71" if selected else "rgba(255,255,255,0.06)"
                    bg = "rgba(46,204,113,0.06)" if selected else "rgba(255,255,255,0.02)"
                    st.markdown(f"""
                    <div style="padding:0.7rem 1rem;margin:0.3rem 0;background:{bg};border:1px solid {border_col};border-radius:12px;cursor:pointer;">
                        <div style="font-weight:700;color:#f0f2f5;font-size:0.95rem;">📍 {m['city']}, {m['state']}</div>
                        <div style="color:#8899aa;font-size:0.78rem;margin-top:0.2rem;">{m['country']} · {m['latitude']:.4f}°, {m['longitude']:.4f}°</div>
                    </div>
                    """, unsafe_allow_html=True)

                options = [f"{m['city']}, {m['state']}, {m['country']}" for m in matches]
                selected_idx = st.radio(
                    "Select your location:",
                    range(len(options)),
                    format_func=lambda i: options[i],
                    key="city_select_radio",
                    label_visibility="collapsed",
                )

                if st.button("📍 Use This Location", type="primary", key="btn_use_city", use_container_width=True):
                    m = matches[selected_idx]
                    set_manual_location(m["latitude"], m["longitude"], m["city"], m["state"], m["country"])
                    st.session_state["weather_raw"] = None
                    st.session_state["weather_current"] = {}
                    st.session_state["_city_matches"] = []
                    st.success(f"✅ Location set to **{m['city']}**, {m['state']}, {m['country']}")
                    st.rerun()

        # ── Tab 2: Coordinates (advanced) ──
        with tab_coords:
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("Latitude", value=28.6139, min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f", key="loc_lat")
            with col2:
                lon = st.number_input("Longitude", value=77.2090, min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f", key="loc_lon")

            if st.button("🔍 Lookup Place Name from Coordinates", key="btn_reverse_geo", use_container_width=True):
                with st.spinner("Looking up location..."):
                    result = reverse_geocode(lat, lon)
                if result:
                    st.session_state["_geo_city"] = result["city"]
                    st.session_state["_geo_state"] = result["state"]
                    st.session_state["_geo_country"] = result["country"]
                    st.session_state["_geo_district"] = result.get("district", result["city"])
                    st.success(f"📍 Found: **{result['city']}**, {result.get('district','')}, {result['state']}, {result['country']}")
                else:
                    st.warning("Could not resolve location. Please enter details manually.")

            default_city = st.session_state.get("_geo_city", "New Delhi")
            default_state = st.session_state.get("_geo_state", "Delhi")
            default_country = st.session_state.get("_geo_country", "India")
            default_district = st.session_state.get("_geo_district", "")

            col3, col4 = st.columns(2)
            with col3:
                city = st.text_input("City / Town", value=default_city, key="loc_city")
                country = st.text_input("Country", value=default_country, key="loc_country")
            with col4:
                state = st.text_input("State / Region", value=default_state, key="loc_state")
                district = st.text_input("District (optional)", value=default_district, key="loc_district")

            if st.button("📍 Set Location", type="primary", key="btn_set_loc", use_container_width=True):
                set_manual_location(lat, lon, city, state, country)
                if district:
                    st.session_state["location"]["district"] = district
                st.session_state["weather_raw"] = None
                st.session_state["weather_current"] = {}
                st.success(f"✅ Location set to **{city}**, {state}, {country}")
                st.rerun()
