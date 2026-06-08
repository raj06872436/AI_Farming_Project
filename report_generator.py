"""
AGRI-X AI — Field Report Generator (Module 11)
Generates downloadable CSV and HTML reports containing prediction results,
weather data, risk analysis, treatment recommendations, and crop suggestions.
"""
import csv
import io
import os
import base64
from datetime import datetime

import streamlit as st

# ── Helpers ──

def _safe(val, default="N/A"):
    """Return val or default if None/empty."""
    if val is None or val == "":
        return default
    return val


def _fmt_currency(val):
    """Format a number as INR currency string."""
    try:
        return f"₹{int(val):,}"
    except (TypeError, ValueError):
        return "N/A"


# ═══════════════════════════════════════════
#  CSV REPORT
# ═══════════════════════════════════════════

def generate_csv_report(prediction=None, weather=None, location=None,
                        risk=None, treatment=None, severity=None,
                        crop_results=None, yield_result=None):
    """
    Build a CSV byte-string containing all available analysis data.
    Returns (filename, csv_bytes).
    """
    buf = io.StringIO()
    w = csv.writer(buf)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = f"AgriXAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Header
    w.writerow(["AGRI-X AI — Field Analysis Report"])
    w.writerow(["Generated", ts])
    w.writerow([])

    # Prediction
    if prediction:
        w.writerow(["=== PREDICTION ==="])
        w.writerow(["Disease Class", _safe(prediction.get("class"))])
        w.writerow(["Confidence", f"{prediction.get('confidence', 0):.1%}"])
        w.writerow(["Model", _safe(prediction.get("model"))])
        w.writerow(["Inference Time", f"{prediction.get('inference_ms', 0):.0f} ms"])
        if severity:
            w.writerow(["Severity", _safe(severity.get("level"))])
            w.writerow(["Severity Score", f"{severity.get('score', 0):.0%}"])
        w.writerow([])

    # Location
    if location:
        w.writerow(["=== LOCATION ==="])
        for k in ["city", "state", "country", "latitude", "longitude"]:
            w.writerow([k.title(), _safe(location.get(k))])
        w.writerow([])

    # Weather
    if weather:
        w.writerow(["=== WEATHER ==="])
        for k, label in [("temperature", "Temperature (°C)"),
                          ("humidity", "Humidity (%)"),
                          ("precipitation", "Precipitation (mm)"),
                          ("wind_speed", "Wind Speed (km/h)"),
                          ("pressure", "Pressure (hPa)"),
                          ("uv_index", "UV Index"),
                          ("cloud_cover", "Cloud Cover (%)")]:
            w.writerow([label, _safe(weather.get(k))])
        w.writerow([])

    # Risk
    if risk:
        w.writerow(["=== DISEASE RISK ==="])
        w.writerow(["Risk Level", _safe(risk.get("risk_level"))])
        w.writerow(["Risk Score", f"{risk.get('risk_score', 0)}%"])
        w.writerow(["Reason", _safe(risk.get("reason"))])
        w.writerow([])

    # Treatment
    if treatment:
        w.writerow(["=== TREATMENT ==="])
        w.writerow(["Disease", _safe(treatment.get("disease"))])
        w.writerow(["Plant", _safe(treatment.get("plant"))])
        w.writerow(["Severity", _safe(treatment.get("severity"))])
        w.writerow(["Description", _safe(treatment.get("description"))])
        w.writerow(["Cause", _safe(treatment.get("cause"))])
        organic = treatment.get("organic_treatment", [])
        w.writerow(["Organic Treatment", "; ".join(organic) if organic else "N/A"])
        chemical = treatment.get("chemical_treatment", [])
        w.writerow(["Chemical Treatment", "; ".join(chemical) if chemical else "N/A"])
        prevention = treatment.get("prevention", [])
        w.writerow(["Prevention", "; ".join(prevention) if prevention else "N/A"])
        w.writerow(["Irrigation", _safe(treatment.get("irrigation"))])
        w.writerow(["Fertilizer", _safe(treatment.get("fertilizer"))])
        w.writerow([])

    # Crop recommendations
    if crop_results:
        w.writerow(["=== CROP RECOMMENDATIONS ==="])
        w.writerow(["Rank", "Crop", "Score", "Season", "Duration (days)",
                     "Water Need", "Est. Yield (t)", "Revenue Min (₹)", "Revenue Max (₹)"])
        for i, c in enumerate(crop_results[:10], 1):
            w.writerow([i, c["name"], f"{c['score']}%", c["season"],
                        c["duration"], c["water_need"],
                        f"{c['yield_min']:.1f}-{c['yield_max']:.1f}",
                        _fmt_currency(c["revenue_min"]),
                        _fmt_currency(c["revenue_max"])])
        w.writerow([])

    # Yield
    if yield_result:
        w.writerow(["=== YIELD FORECAST ==="])
        w.writerow(["Crop", _safe(yield_result.get("crop_name"))])
        w.writerow(["Area (acres)", _safe(yield_result.get("area_acres"))])
        w.writerow(["Expected Yield (t)", _safe(yield_result.get("expected_yield"))])
        w.writerow(["Yield / Acre (t)", _safe(yield_result.get("yield_per_acre"))])
        w.writerow(["Revenue", _fmt_currency(yield_result.get("expected_revenue"))])
        w.writerow(["Harvest By", _safe(yield_result.get("harvest_date"))])
        w.writerow(["Confidence", f"{yield_result.get('confidence', 0):.0%}"])
        w.writerow([])

    csv_bytes = buf.getvalue().encode("utf-8-sig")  # BOM for Excel compat
    return fname, csv_bytes


# ═══════════════════════════════════════════
#  HTML REPORT  (printable / save-as-PDF)
# ═══════════════════════════════════════════

def _html_section(title, rows):
    """Build an HTML table section."""
    trs = "".join(
        f"<tr><td style='padding:6px 12px;color:#666;width:200px;'>{k}</td>"
        f"<td style='padding:6px 12px;font-weight:500;'>{v}</td></tr>"
        for k, v in rows
    )
    return f"""
    <h3 style="color:#27ae60;border-bottom:2px solid #eee;padding-bottom:4px;margin-top:24px;">{title}</h3>
    <table style="width:100%;border-collapse:collapse;font-size:14px;">{trs}</table>
    """


def generate_html_report(prediction=None, weather=None, location=None,
                         risk=None, treatment=None, severity=None,
                         crop_results=None, yield_result=None):
    """
    Build a self-contained HTML report.
    Users can open it in a browser and print / save-as-PDF.
    Returns (filename, html_string).
    """
    ts = datetime.now().strftime("%B %d, %Y  %I:%M %p")
    fname = f"AgriXAI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    sections = ""

    # Prediction
    if prediction:
        rows = [
            ("Disease", _safe(prediction.get("class", "").replace("_", " "))),
            ("Confidence", f"{prediction.get('confidence', 0):.1%}"),
            ("Model", _safe(prediction.get("model"))),
            ("Inference", f"{prediction.get('inference_ms', 0):.0f} ms"),
        ]
        if severity:
            rows.append(("Severity", f"{_safe(severity.get('level'))} ({severity.get('score', 0):.0%})"))
        sections += _html_section("🔬 Disease Prediction", rows)

    # Location
    if location:
        sections += _html_section("📍 Location", [
            ("City", _safe(location.get("city"))),
            ("State", _safe(location.get("state"))),
            ("Country", _safe(location.get("country"))),
            ("Coordinates", f"{location.get('latitude', 0):.4f}°, {location.get('longitude', 0):.4f}°"),
        ])

    # Weather
    if weather:
        sections += _html_section("🌦️ Weather Conditions", [
            ("Temperature", f"{weather.get('temperature', 0):.1f} °C"),
            ("Humidity", f"{weather.get('humidity', 0)}%"),
            ("Precipitation", f"{weather.get('precipitation', 0):.1f} mm"),
            ("Wind Speed", f"{weather.get('wind_speed', 0):.1f} km/h"),
            ("UV Index", f"{weather.get('uv_index', 0):.1f}"),
            ("Pressure", f"{weather.get('pressure', 0):.0f} hPa"),
        ])

    # Risk
    if risk:
        sections += _html_section("⚠️ Disease Risk Analysis", [
            ("Risk Level", _safe(risk.get("risk_level"))),
            ("Risk Score", f"{risk.get('risk_score', 0)}%"),
            ("Analysis", _safe(risk.get("reason"))),
        ])

    # Treatment
    if treatment:
        organic = treatment.get("organic_treatment", [])
        chemical = treatment.get("chemical_treatment", [])
        prevention = treatment.get("prevention", [])
        sections += _html_section("💊 Treatment Recommendations", [
            ("Disease", _safe(treatment.get("disease"))),
            ("Plant", _safe(treatment.get("plant"))),
            ("Severity", _safe(treatment.get("severity"))),
            ("Cause", _safe(treatment.get("cause"))),
            ("Organic Treatment", "<br>".join(f"• {t}" for t in organic) if organic else "N/A"),
            ("Chemical Treatment", "<br>".join(f"• {t}" for t in chemical) if chemical else "N/A"),
            ("Prevention", "<br>".join(f"• {p}" for p in prevention) if prevention else "N/A"),
            ("Irrigation", _safe(treatment.get("irrigation"))),
            ("Fertilizer", _safe(treatment.get("fertilizer"))),
        ])

    # Crop recommendations
    if crop_results:
        crop_rows = "".join(
            f"<tr><td style='padding:6px;'>{i}</td><td style='padding:6px;font-weight:600;'>{c['name']}</td>"
            f"<td style='padding:6px;'>{c['score']}%</td><td style='padding:6px;'>{c['season']}</td>"
            f"<td style='padding:6px;'>{c['yield_min']:.1f}-{c['yield_max']:.1f} t</td>"
            f"<td style='padding:6px;'>{_fmt_currency(c['revenue_min'])} - {_fmt_currency(c['revenue_max'])}</td></tr>"
            for i, c in enumerate(crop_results[:8], 1)
        )
        sections += f"""
        <h3 style="color:#27ae60;border-bottom:2px solid #eee;padding-bottom:4px;margin-top:24px;">🌾 Crop Recommendations</h3>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr style="background:#f8f8f8;">
                <th style="padding:8px;text-align:left;">Rank</th>
                <th style="padding:8px;text-align:left;">Crop</th>
                <th style="padding:8px;text-align:left;">Score</th>
                <th style="padding:8px;text-align:left;">Season</th>
                <th style="padding:8px;text-align:left;">Yield</th>
                <th style="padding:8px;text-align:left;">Revenue</th>
            </tr>
            {crop_rows}
        </table>
        """

    # Yield
    if yield_result:
        sections += _html_section("📊 Yield Forecast", [
            ("Crop", _safe(yield_result.get("crop_name"))),
            ("Area", f"{yield_result.get('area_acres', 0)} acres"),
            ("Expected Yield", f"{yield_result.get('expected_yield', 0):.1f} tonnes"),
            ("Revenue", _fmt_currency(yield_result.get("expected_revenue"))),
            ("Harvest By", _safe(yield_result.get("harvest_date"))),
            ("Confidence", f"{yield_result.get('confidence', 0):.0%}"),
        ])

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>AGRI-X AI Field Report — {ts}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 30px; color: #333; }}
  h1 {{ color: #27ae60; margin-bottom: 0; }}
  .subtitle {{ color: #888; font-size: 14px; margin-bottom: 30px; }}
  table {{ margin-bottom: 8px; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd; color: #aaa; font-size: 12px; text-align: center; }}
  @media print {{ body {{ padding: 15px; }} }}
</style>
</head><body>
<h1>🌿 AGRI-X AI — Field Analysis Report</h1>
<div class="subtitle">Generated: {ts}</div>
{sections}
<div class="footer">
    AGRI-X AI • Explainable Multi-Model Deep Learning Framework for Plant Disease Detection<br>
    This report is auto-generated. For research and educational use.
</div>
</body></html>"""

    return fname, html


# ═══════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════

def render_report_section(prediction=None, weather=None, location=None,
                          risk=None, treatment=None, severity=None,
                          crop_results=None, yield_result=None):
    """Render the report download section inside the app."""

    st.markdown("### 📄 Download Field Report")
    st.caption("Generate a downloadable report with all your analysis results.")

    has_data = any([prediction, weather, location, risk, treatment, crop_results, yield_result])

    if not has_data:
        st.info("No analysis data available yet. Use the Disease Detection, Crop Advisor, or Weather pages first, then come back here to download your report.")
        return

    # Show what will be included
    items = []
    if prediction: items.append("🔬 Disease Prediction")
    if severity: items.append("📊 Severity Analysis")
    if location: items.append("📍 Location Data")
    if weather: items.append("🌦️ Weather Conditions")
    if risk: items.append("⚠️ Risk Assessment")
    if treatment: items.append("💊 Treatment Recommendations")
    if crop_results: items.append("🌾 Crop Recommendations")
    if yield_result: items.append("📊 Yield Forecast")

    st.markdown("**Report contents:**")
    cols = st.columns(4)
    for i, item in enumerate(items):
        cols[i % 4].markdown(f"✅ {item}")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        csv_fname, csv_bytes = generate_csv_report(
            prediction, weather, location, risk, treatment, severity,
            crop_results, yield_result
        )
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_bytes,
            file_name=csv_fname,
            mime="text/csv",
            use_container_width=True,
            type="primary",
            key="dl_csv",
        )

    with c2:
        html_fname, html_str = generate_html_report(
            prediction, weather, location, risk, treatment, severity,
            crop_results, yield_result
        )
        st.download_button(
            label="📥 Download HTML Report",
            data=html_str.encode("utf-8"),
            file_name=html_fname,
            mime="text/html",
            use_container_width=True,
            key="dl_html",
        )

    st.caption("💡 **Tip:** Open the HTML report in a browser and use *Print → Save as PDF* for a PDF version.")
