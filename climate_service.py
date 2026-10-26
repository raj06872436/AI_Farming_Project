"""
AGRI-X AI — Climate normals for a farm location.

Crop choice depends on the *season's* weather, not today's. This module gets the
average temperature and rainfall for every month at the exact farm location from
the last five years of ERA5 weather records (Open-Meteo archive API — free, no key),
and falls back to the regional normals in india_agro_data when offline.

Analogy: today's weather tells you whether to carry an umbrella; the climate normal
tells you whether to plant rice.
"""
from datetime import date

import requests
import streamlit as st

from india_agro_data import find_profile, monthly_climate

SEASON_START = {"Kharif": 6, "Rabi": 11, "Zaid": 3, "Perennial": 1}
SEASON_LABEL = {"Kharif": "June–October", "Rabi": "November–March", "Zaid": "March–June", "Perennial": "all year"}


@st.cache_data(ttl=30 * 86400, show_spinner=False)
def _fetch_normals(lat: float, lon: float):
    """Monthly normals from the last 5 full years of ERA5 data, or None if unavailable."""
    end_year = date.today().year - 1
    start_year = end_year - 4
    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": round(lat, 3), "longitude": round(lon, 3),
                "start_date": f"{start_year}-01-01", "end_date": f"{end_year}-12-31",
                "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
            },
            timeout=20,
        )
        daily = resp.json().get("daily", {})
        days = daily.get("time") or []
        if len(days) < 300:
            return None
        sums = {k: [0.0] * 12 for k in ("t", "tmax", "tmin", "rain")}
        counts = [0] * 12
        for i, d in enumerate(days):
            m = int(d[5:7]) - 1
            t, tx, tn, r = (daily["temperature_2m_mean"][i], daily["temperature_2m_max"][i],
                            daily["temperature_2m_min"][i], daily["precipitation_sum"][i])
            if t is None:
                continue
            counts[m] += 1
            sums["t"][m] += t
            sums["tmax"][m] += tx if tx is not None else t
            sums["tmin"][m] += tn if tn is not None else t
            sums["rain"][m] += r or 0.0
        n_years = end_year - start_year + 1
        if min(counts) == 0:
            return None
        rain = [round(sums["rain"][m] / n_years) for m in range(12)]
        return {
            "temp": [round(sums["t"][m] / counts[m], 1) for m in range(12)],
            "tmax": [round(sums["tmax"][m] / counts[m], 1) for m in range(12)],
            "tmin": [round(sums["tmin"][m] / counts[m], 1) for m in range(12)],
            "rain": rain,
            "annual_rain": int(sum(rain)),
            "source": f"Local weather records {start_year}–{end_year} (ERA5 via Open-Meteo)",
            "live": True,
        }
    except Exception:
        return None


def get_climate_normals(location):
    """Monthly climate normals for a location dict; never fails (regional fallback)."""
    if location and location.get("latitude") is not None:
        live = _fetch_normals(float(location["latitude"]), float(location["longitude"]))
        if live:
            return live
    profile = find_profile(location)
    fallback = monthly_climate(profile)
    # Regional normals describe typical farmland (plains ~300 m, hill regions ~1200 m).
    # Correct for the farm's own altitude: air cools ~6.5 °C per 1000 m.
    elev = (location or {}).get("elevation")
    if elev is not None:
        ref = 1200 if profile["soil_group"] == "Mountain" else 300
        shift = -0.0065 * (float(elev) - ref)
        if abs(shift) >= 0.5:
            fallback["temp"] = [round(x + shift, 1) for x in fallback["temp"]]
            fallback["source"] += f" (adjusted for {elev:.0f} m altitude)"
    fallback["live"] = False
    return fallback


def season_window(normals, season, duration_days):
    """
    Climate during a crop's growing period: starts in the season's sowing month and
    lasts `duration_days`. Returns mean / hottest / coldest temperature and total rain.
    """
    start = SEASON_START.get(season, 6) - 1
    n = max(1, min(12, round(duration_days / 30)))
    months = [(start + i) % 12 for i in range(n)]
    temps = [normals["temp"][m] for m in months]
    tmax = [normals.get("tmax", normals["temp"])[m] for m in months]
    tmin = [normals.get("tmin", normals["temp"])[m] for m in months]
    rain = sum(normals["rain"][m] for m in months)
    return {
        "months": months,
        "n_months": n,
        "mean_temp": round(sum(temps) / n, 1),
        "hottest": max(tmax) if "tmax" in normals else max(temps) + 5,
        "coldest": min(tmin) if "tmin" in normals else min(temps) - 6,
        "rain": round(rain),
        "monthly_rain": round(rain / n),
    }
