"""
AGRI-X AI — Location Service (Module 1)
Automatic geolocation via IP + manual override + reverse geocoding.
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


def detect_location():
    """
    Return location dict. Priority:
    1. Session-state manual override
    2. IP-based geolocation
    3. Default fallback
    """
    if "location" in st.session_state and st.session_state["location"]:
        return st.session_state["location"]

    loc = _ip_geolocation()
    if loc:
        st.session_state["location"] = loc
        return loc

    st.session_state["location"] = DEFAULT_LOCATION.copy()
    return st.session_state["location"]


def set_manual_location(lat, lon, city, state, country):
    """Store a manually entered location in session state."""
    st.session_state["location"] = {
        "latitude": lat,
        "longitude": lon,
        "city": city,
        "state": state,
        "country": country,
        "source": "manual",
    }


def render_location_card(loc: dict):
    """Render a beautiful location display card."""
    source_badge = {
        "ip-api": ("🌐 Auto-detected via IP", "#3498db"),
        "manual": ("✏️ Manually entered", "#f39c12"),
        "default": ("📍 Default (override recommended)", "#e74c3c"),
    }
    label, color = source_badge.get(loc.get("source", "default"), ("📍", "#888"))

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {color};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
            <span style="font-size:1.1rem;font-weight:700;color:#fff;">📍 Your Location</span>
            <span style="font-size:0.7rem;color:{color};background:rgba(255,255,255,0.06);padding:3px 10px;border-radius:12px;">{label}</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;">
            <div><span style="color:#8899aa;font-size:0.75rem;">CITY</span><br><span style="color:#fff;font-weight:600;">{loc['city']}</span></div>
            <div><span style="color:#8899aa;font-size:0.75rem;">STATE</span><br><span style="color:#fff;font-weight:600;">{loc['state']}</span></div>
            <div><span style="color:#8899aa;font-size:0.75rem;">COUNTRY</span><br><span style="color:#fff;font-weight:600;">{loc['country']}</span></div>
            <div><span style="color:#8899aa;font-size:0.75rem;">COORDINATES</span><br><span style="color:#2ecc71;font-weight:600;">{loc['latitude']:.4f}°, {loc['longitude']:.4f}°</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


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
                options = [f"{m['city']}, {m['state']}, {m['country']}  ({m['latitude']:.4f}°, {m['longitude']:.4f}°)" for m in matches]
                selected_idx = st.radio(
                    "Select your location:",
                    range(len(options)),
                    format_func=lambda i: options[i],
                    key="city_select_radio",
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
