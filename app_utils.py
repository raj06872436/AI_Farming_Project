"""
Utility functions and CSS for the Streamlit app.
Separated to keep app.py focused on layout/logic.
Phase 2: Enhanced with premium CSS, theme support, research data loading.
"""
import os, json, glob
import numpy as np

# ── Path Constants ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
GRAPHS_DIR = os.path.join(REPORTS_DIR, "graphs")
GRADCAM_DIR = os.path.join(REPORTS_DIR, "gradcam")
SUMMARY_DIR = os.path.join(REPORTS_DIR, "summary")
CONFUSION_DIR = os.path.join(REPORTS_DIR, "confusion_matrix")
ROC_DIR = os.path.join(REPORTS_DIR, "roc_curves")
METRICS_DIR = os.path.join(REPORTS_DIR, "metrics")
RESEARCH_DIR = os.path.join(BASE_DIR, "research_bundle")
RESEARCH_FIGURES = os.path.join(RESEARCH_DIR, "reports", "figures")
REGISTRY_PATH = os.path.join(RESEARCH_DIR, "model_registry.json")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")
METRICS_CSV = os.path.join(METRICS_DIR, "model_comparison.csv")

# ── Data Loaders ──
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            return [l.strip() for l in f if l.strip()]
    pv = os.path.join(BASE_DIR, "PlantVillage")
    if os.path.isdir(pv):
        return sorted([d for d in os.listdir(pv) if os.path.isdir(os.path.join(pv, d))])
    return []

def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}

def load_dataset_summary():
    p = os.path.join(SUMMARY_DIR, "dataset_summary.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None

def load_metrics_csv():
    import pandas as pd
    if os.path.exists(METRICS_CSV):
        return pd.read_csv(METRICS_CSV)
    return None

def find_model_path(model_name):
    for ext in ["_best.keras", "_final.keras", "_best.h5", "_final.h5"]:
        p = os.path.join(SAVED_MODELS_DIR, f"{model_name}{ext}")
        if os.path.exists(p):
            return p
    return None

def get_available_models(registry):
    available = []
    for name in registry:
        if find_model_path(name):
            available.append(name)
    return available

def find_training_graph(model_name):
    for d in [GRAPHS_DIR, RESEARCH_FIGURES]:
        p = os.path.join(d, f"{model_name}_training_history.png")
        if os.path.exists(p):
            return p
    return None

def find_gradcam_images(model_name):
    imgs = []
    seen = set()
    for d in [GRADCAM_DIR, RESEARCH_FIGURES]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith(model_name) and "sample" in f and f.endswith(".png"):
                    if f not in seen:
                        imgs.append(os.path.join(d, f))
                        seen.add(f)
    return imgs

def get_all_graph_images():
    imgs = {}
    for d in [GRAPHS_DIR, RESEARCH_FIGURES]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".png") and f not in imgs:
                    imgs[f] = os.path.join(d, f)
    return imgs

# ── Research Data Loaders ──
def load_json_report(filename):
    """Load a JSON report from multiple possible locations."""
    for d in [METRICS_DIR, SUMMARY_DIR, REPORTS_DIR, RESEARCH_DIR]:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    return None

def load_text_report(filename):
    """Load a text report from multiple possible locations."""
    for d in [METRICS_DIR, SUMMARY_DIR, REPORTS_DIR, RESEARCH_DIR]:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            with open(p, "r") as f:
                return f.read()
    return None

def load_csv_report(filename):
    """Load a CSV report from multiple possible locations."""
    import pandas as pd
    for d in [METRICS_DIR, SUMMARY_DIR, REPORTS_DIR, RESEARCH_DIR]:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

def find_all_reports(pattern="*.json"):
    """Find all report files matching a glob pattern."""
    results = {}
    for d in [METRICS_DIR, SUMMARY_DIR, REPORTS_DIR, RESEARCH_DIR]:
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, pattern)):
                basename = os.path.basename(f)
                if basename not in results:
                    results[basename] = f
    return results

# ── HTML Component Builders ──
# Colours come from theme.py so the whole UI stays on one palette.
from theme import GREEN, GREEN_SOFT, RED, AMBER

_CHIP_TONES = {"green", "red", "amber", "sky", "purple", "gray"}


def chip_html(text, tone="gray"):
    """Small rounded status label. tone: green | red | amber | sky | purple | gray."""
    tone = tone if tone in _CHIP_TONES else "gray"
    return f'<span class="chip chip-{tone}">{text}</span>'


def page_header_html(eyebrow, title, subtitle=""):
    """Consistent page header: small coloured label, big title, one-line subtitle."""
    sub = f'<p class="page-sub">{subtitle}</p>' if subtitle else ""
    return f"""
    <div class="page-header fade-in">
        <div class="page-eyebrow">{eyebrow}</div>
        <h1 class="page-title">{title}</h1>
        {sub}
    </div>"""


def section_head_html(title, note=""):
    note_html = f"<span>{note}</span>" if note else ""
    return f'<div class="section-head"><div class="sh-title">{title}</div>{note_html}</div>'


def kv_html(label, value, color=None):
    style = f' style="color:{color};"' if color else ""
    return f'<div class="kv"><div class="kv-label">{label}</div><div class="kv-value"{style}>{value}</div></div>'


def meter_html(fraction, color=GREEN):
    pct = max(0.0, min(1.0, float(fraction))) * 100
    return f'<div class="meter"><div style="width:{pct:.1f}%;background:{color};"></div></div>'


def confidence_bar_html(confidence, label=""):
    pct = confidence * 100
    color = GREEN if pct >= 80 else AMBER if pct >= 50 else RED
    return f"""
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color};">
            {label} {pct:.1f}%
        </div>
    </div>"""


def severity_badge_html(severity, color=AMBER):
    return f'<span class="severity-badge" style="background:{color}14;color:{color};border:1px solid {color}55;">{severity}</span>'


def metric_card_html(value, label, icon=""):
    return f"""
    <div class="metric-card">
        <div class="stat-icon" style="background:{GREEN_SOFT};">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def recommendation_card_html(title, icon, text):
    return f"""
    <div class="rec-card">
        <h4>{icon} {title}</h4>
        <p>{text}</p>
    </div>"""


def feature_card_html(icon, title, description, index=0):
    delay = index * 0.05
    return f"""
    <div class="feature-card" style="animation-delay:{delay}s;">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{description}</div>
    </div>"""


def stat_card_html(value, label, icon="", accent_color=GREEN):
    return f"""
    <div class="stat-card">
        <div class="stat-icon" style="background:{accent_color}18;color:{accent_color};">{icon}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>"""


def weather_strip_html(temp, humidity, rain, wind, icon="", desc=""):
    items = [
        (icon or "🌡️", f"{temp:.1f}°C", desc or "Temp"),
        ("💧", f"{humidity:.0f}%", "Humidity"),
        ("🌧️", f"{rain:.1f}mm", "Rainfall"),
        ("💨", f"{wind:.1f}km/h", "Wind"),
    ]
    parts = []
    for i, (ic, val, lbl) in enumerate(items):
        if i:
            parts.append('<div class="weather-strip-divider"></div>')
        parts.append(f'<div class="weather-strip-item"><span class="weather-strip-icon">{ic}</span>'
                     f'<span class="weather-strip-val">{val}</span><span class="weather-strip-lbl">{lbl}</span></div>')
    return f'<div class="weather-strip">{"".join(parts)}</div>'


def how_it_works_html():
    steps = [
        ("Upload", "Take a clear photo of one leaf and upload it."),
        ("AI analysis", "A trained model finds the disease and highlights the affected area."),
        ("Act", "Get severity, weather risk, treatment steps and crop advice."),
    ]
    cells = "".join(
        f'<div class="hiw-step"><div class="hiw-circle">{i}</div><div class="hiw-title">{t}</div>'
        f'<div class="hiw-desc">{d}</div></div>'
        for i, (t, d) in enumerate(steps, 1)
    )
    return f'<div class="hiw-container">{cells}</div>'


def badge_pill_html(text, color=GREEN):
    return f'<span class="badge-pill" style="color:{color};border-color:{color}40;background:{color}12;">{text}</span>'


def context_bar_html(items):
    """items: list of (icon, label, value) → a single row summarising the farm context."""
    cells = "".join(
        f'<div class="ctx"><div class="ctx-icon">{ic}</div>'
        f'<div><div class="kv-label">{lbl}</div><div class="kv-value" style="font-size:0.95rem;">{val}</div></div></div>'
        for ic, lbl, val in items
    )
    return f'<div class="context-bar">{cells}</div>'





# ── CSS ──
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

/* ══════════════════════════════════════════════════════════════
   AGRI-X AI — "Fresh Field" light theme
   Colours mirror theme.py. Change them here + in theme.py.
   ══════════════════════════════════════════════════════════════ */
:root {
    --bg: #F6F7F2;
    --surface: #FFFFFF;
    --surface-2: #F2F5EE;
    --surface-3: #E7ECE1;
    --border: #E0E6D9;
    --border-strong: #C9D2BF;

    --text: #1E2A22;
    --text-2: #47554B;
    --text-3: #6B776E;

    --green: #2E7D32;
    --green-strong: #1B5E20;
    --green-soft: #E5F1E3;
    --teal: #0F766E;
    --sky: #2563A8;
    --sky-soft: #E4EDF8;
    --amber: #B45309;
    --amber-soft: #FBF0DD;
    --orange: #C2410C;
    --red: #B91C1C;
    --red-soft: #FBE9E7;
    --purple: #6D28D9;
    --purple-soft: #EFE9FB;
    --soil: #8B5E34;

    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 22px;
    --shadow-sm: 0 1px 2px rgba(30,42,34,0.06);
    --shadow-md: 0 2px 6px rgba(30,42,34,0.05), 0 8px 24px rgba(30,42,34,0.06);
    --shadow-lg: 0 10px 40px rgba(30,42,34,0.10);
    --ease: 0.2s cubic-bezier(0.4, 0, 0.2, 1);

    /* legacy aliases (older inline styles use these names) */
    --text-primary: var(--text);
    --text-secondary: var(--text-3);
    --text-muted: var(--text-3);
    --accent-green: var(--green);
    --accent-blue: var(--sky);
    --accent-orange: var(--amber);
    --accent-red: var(--red);
    --accent-purple: var(--purple);
    --accent-teal: var(--teal);
    --glass-border: var(--border);
}

/* ── Base ── */
html, body, .stApp, .stMarkdown, .stText, button, input, textarea, select, label {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.stApp { background: var(--bg); color: var(--text); }
h1, h2, h3, h4,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}
.stMarkdown h3 { font-size: 1.25rem !important; font-weight: 700 !important; margin-top: 0.6rem; }
.stMarkdown h4 { font-size: 1.05rem !important; font-weight: 700 !important; }
.stMarkdown p, .stMarkdown li { color: var(--text-2); line-height: 1.6; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--text-3) !important; }
a { color: var(--green); }

[data-testid="stHeader"] { background: rgba(246,247,242,0.85); backdrop-filter: blur(8px); }
.main .block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 3.6rem;
    padding-bottom: 3rem;
    max-width: 1240px;
}
hr, [data-testid="stDivider"] hr { border-color: var(--border) !important; }

/* ── Page header ── */
.page-header { margin: 0 0 1.4rem 0; }
.page-eyebrow {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--green); background: var(--green-soft);
    padding: 4px 10px; border-radius: 999px; margin-bottom: 0.6rem;
}
.hero-title, .page-title {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    font-size: 2.1rem; font-weight: 800; color: var(--text);
    letter-spacing: -0.02em; line-height: 1.2; margin: 0;
}
.hero-sub, .page-sub {
    color: var(--text-3); font-size: 1rem; margin: 0.35rem 0 0 0; line-height: 1.55; max-width: 760px;
}
.section-head {
    display: flex; align-items: baseline; justify-content: space-between; gap: 1rem;
    margin: 1.6rem 0 0.8rem 0;
}
.section-head .sh-title { font-family: 'Plus Jakarta Sans', 'Inter', sans-serif; font-size: 1.2rem; font-weight: 700; color: var(--text); letter-spacing: -0.01em; }
.section-head span { color: var(--text-3); font-size: 0.85rem; }

/* ── Cards ── */
.glass-card, .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.3rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow var(--ease), border-color var(--ease);
    color: var(--text-2);
}
.glass-card:hover, .card:hover { box-shadow: var(--shadow-md); border-color: var(--border-strong); }
.card-soft { background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 0.9rem 1rem; }

/* ── Chips ── */
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 600; line-height: 1.5;
    border: 1px solid transparent; white-space: nowrap;
}
.chip-green { color: var(--green-strong); background: var(--green-soft); border-color: #CFE4CB; }
.chip-red { color: var(--red); background: var(--red-soft); border-color: #F3CFCA; }
.chip-amber { color: var(--amber); background: var(--amber-soft); border-color: #F1DDB8; }
.chip-sky { color: var(--sky); background: var(--sky-soft); border-color: #C9DAF0; }
.chip-purple { color: var(--purple); background: var(--purple-soft); border-color: #DCD0F6; }
.chip-gray { color: var(--text-2); background: var(--surface-2); border-color: var(--border); }
.badge-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    border: 1px solid; white-space: nowrap;
}
.severity-badge {
    display: inline-flex; align-items: center; gap: 6px; padding: 5px 14px;
    border-radius: 999px; font-size: 0.85rem; font-weight: 600;
}

/* ── Home: hero ── */
.hero-banner, .home-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #FFFFFF 0%, #F1F7EC 60%, #E6F1E1 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 2.2rem 2.4rem;
    box-shadow: var(--shadow-md);
    margin-bottom: 1.2rem;
    min-height: 260px;
}
.home-hero::after {
    content: ""; position: absolute; right: -60px; bottom: -80px; width: 280px; height: 280px;
    border-radius: 50%; background: radial-gradient(circle, rgba(46,125,50,0.14), transparent 70%);
}
.home-hero h1 {
    font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.5rem; font-weight: 800;
    color: var(--text); letter-spacing: -0.03em; line-height: 1.12; margin: 0.4rem 0 0.6rem 0;
}
.home-hero h1 .accent { color: var(--green); }
.home-hero p.lead { color: var(--text-2); font-size: 1.05rem; line-height: 1.6; margin: 0 0 1.1rem 0; max-width: 560px; }
.hero-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.shimmer-text { color: var(--green); font-weight: 700; }

/* Farm snapshot card (home, right column) */
.snapshot {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-xl);
    padding: 1.4rem 1.5rem; box-shadow: var(--shadow-md); min-height: 260px; margin-bottom: 1.2rem;
}
.snapshot-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.9rem; }
.snapshot-title { font-weight: 700; color: var(--text); font-size: 0.95rem; }
.snapshot-place { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.35rem; font-weight: 800; color: var(--text); }
.snapshot-temp { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 2.6rem; font-weight: 800; color: var(--text); line-height: 1; }
.snapshot-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-top: 1rem; }
.snapshot-grid .kv { background: var(--surface-2); border-radius: var(--radius-md); padding: 0.6rem 0.7rem; }

/* key/value blocks */
.kv-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-3); }
.kv-value { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-top: 2px; }

/* Quick actions = Streamlit buttons inside st.container(key="quick_actions") */
.st-key-quick_actions button {
    min-height: 92px; border-radius: var(--radius-lg) !important;
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm); text-align: left; justify-content: flex-start !important;
    padding: 0.9rem 1.1rem !important;
}
.st-key-quick_actions button p { font-size: 1rem !important; font-weight: 700 !important; color: var(--text) !important; white-space: pre-line; text-align: left; }
.st-key-quick_actions button:hover { border-color: var(--green) !important; box-shadow: var(--shadow-md); transform: translateY(-1px); }

/* ── Stat / metric cards ── */
.stat-card, .metric-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 1.1rem 1rem; text-align: left; box-shadow: var(--shadow-sm); position: relative; overflow: hidden;
    transition: box-shadow var(--ease);
}
.stat-card:hover, .metric-card:hover { box-shadow: var(--shadow-md); }
.stat-card-glow, .metric-card-accent { display: none; }
.stat-icon { font-size: 1.25rem; width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 0.6rem; }
.stat-value, .metric-value { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.6rem; font-weight: 800; color: var(--text); letter-spacing: -0.02em; line-height: 1.15; }
.stat-label, .metric-label { font-size: 0.75rem; color: var(--text-3); font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin-top: 0.25rem; }
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-card { flex: 1; min-width: 140px; }

[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
    padding: 12px 14px; box-shadow: var(--shadow-sm);
}
[data-testid="stMetricLabel"] p { color: var(--text-3) !important; font-weight: 600 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 800 !important; font-family: 'Plus Jakarta Sans', sans-serif; }

/* ── Feature cards ── */
.feature-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 1.2rem 1.2rem 1.1rem; min-height: 188px; margin-bottom: 1rem;
    box-shadow: var(--shadow-sm); transition: box-shadow var(--ease), border-color var(--ease);
    animation: fadeInUp 0.45s ease-out both;
}
.feature-card:hover { box-shadow: var(--shadow-md); border-color: #CFE4CB; }
.feature-card-accent { display: none; }
.feature-icon { font-size: 1.4rem; width: 44px; height: 44px; border-radius: 12px; background: var(--green-soft); display: flex; align-items: center; justify-content: center; margin-bottom: 0.75rem; }
.feature-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 0.3rem; }
.feature-desc { font-size: 0.86rem; color: var(--text-3); line-height: 1.5; }

/* ── How it works ── */
.hiw-container { display: flex; align-items: stretch; gap: 0.8rem; margin: 0.4rem 0 1rem 0; }
.hiw-step { flex: 1; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 1.1rem 1.2rem; box-shadow: var(--shadow-sm); }
.hiw-circle { width: 32px; height: 32px; border-radius: 50%; background: var(--green); color: #fff; font-weight: 800; display: flex; align-items: center; justify-content: center; margin-bottom: 0.6rem; font-size: 0.9rem; }
.hiw-title { font-weight: 700; color: var(--text); margin-bottom: 0.2rem; }
.hiw-desc { color: var(--text-3); font-size: 0.86rem; line-height: 1.5; }
.hiw-connector { display: none; }

/* ── Weather strip ── */
.weather-strip { display: flex; align-items: center; justify-content: space-around; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 1rem; box-shadow: var(--shadow-sm); }
.weather-strip-item { display: flex; flex-direction: column; align-items: center; }
.weather-strip-icon { font-size: 1.3rem; margin-bottom: 4px; }
.weather-strip-val { font-size: 1.1rem; font-weight: 700; color: var(--text); }
.weather-strip-lbl { font-size: 0.68rem; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }
.weather-strip-divider { width: 1px; height: 40px; background: var(--border); }

/* ── Diagnosis (Disease Detection) ── */
.diag-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-xl);
    padding: 1.4rem 1.5rem; box-shadow: var(--shadow-md); margin-bottom: 1rem; border-top: 5px solid var(--green);
}
.diag-card.is-disease { border-top-color: var(--red); }
.diag-plant { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-3); }
.diag-name { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.75rem; font-weight: 800; color: var(--text); line-height: 1.2; margin: 0.2rem 0 0.7rem 0; }
.diag-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1rem; }
.diag-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; }
.diag-grid .kv { background: var(--surface-2); border-radius: var(--radius-md); padding: 0.65rem 0.8rem; }
.meter { height: 10px; border-radius: 999px; background: var(--surface-3); overflow: hidden; margin: 0.35rem 0 1rem 0; }
.meter > div { height: 100%; border-radius: 999px; }
.diag-advice { margin-top: 1rem; padding: 0.8rem 1rem; border-radius: var(--radius-md); font-size: 0.92rem; line-height: 1.5; }
.empty-state { text-align: center; padding: 2.6rem 1.5rem; border: 2px dashed var(--border-strong); border-radius: var(--radius-xl); background: var(--surface); color: var(--text-3); }
.empty-state .big { font-size: 2.4rem; margin-bottom: 0.4rem; }
.empty-state b { color: var(--text); }

/* Farm context bar (Crop Advisor) */
.context-bar { display: flex; flex-wrap: wrap; gap: 0.6rem; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 0.8rem 1rem; box-shadow: var(--shadow-sm); margin-bottom: 1rem; align-items: center; }
.context-bar .ctx { display: flex; align-items: center; gap: 8px; padding: 0.3rem 0.8rem 0.3rem 0.3rem; border-right: 1px solid var(--border); }
.context-bar .ctx:last-child { border-right: none; }
.context-bar .ctx-icon { width: 34px; height: 34px; border-radius: 10px; background: var(--surface-2); display: flex; align-items: center; justify-content: center; font-size: 1.05rem; }

/* ── Confidence bar ── */
.conf-bar-bg { background: var(--surface-3); border-radius: 999px; height: 26px; overflow: hidden; margin: 0.5rem 0; }
.conf-bar-fill { height: 100%; border-radius: 999px; display: flex; align-items: center; padding-left: 12px; font-size: 0.8rem; font-weight: 700; color: #fff; }

/* ── Recommendation / info cards ── */
.rec-card { background: var(--surface); border: 1px solid var(--border); border-left: 4px solid var(--green); border-radius: var(--radius-md); padding: 1rem 1.2rem; margin: 0.5rem 0; }
.rec-card h4 { margin: 0 0 0.3rem 0; color: var(--green-strong) !important; font-size: 0.95rem; }
.rec-card p { margin: 0; color: var(--text-2); font-size: 0.9rem; line-height: 1.55; }
.info-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 1.2rem; }
.tech-badge { display: inline-block; padding: 5px 12px; margin: 4px; border-radius: 999px; background: var(--surface); border: 1px solid var(--border); color: var(--text-2); font-size: 0.82rem; font-weight: 600; }
.ref-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 0.8rem 1rem; margin: 0.45rem 0; color: var(--text-2); font-size: 0.88rem; line-height: 1.5; }
.ref-number { display: inline-flex; width: 24px; height: 24px; border-radius: 50%; background: var(--green-soft); color: var(--green-strong); font-weight: 700; font-size: 0.78rem; align-items: center; justify-content: center; margin-right: 8px; }
.method-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 1rem 1.2rem; margin: 0.5rem 0; box-shadow: var(--shadow-sm); }
.method-number { display: inline-flex; width: 28px; height: 28px; border-radius: 8px; background: var(--green); color: #fff; font-weight: 800; font-size: 0.85rem; align-items: center; justify-content: center; margin-right: 10px; }
.method-title { font-weight: 700; color: var(--text); }
.method-desc { margin-top: 0.4rem; font-size: 0.88rem; color: var(--text-3); line-height: 1.55; }
.topk-item { display: flex; justify-content: space-between; padding: 0.5rem 0.8rem; border-radius: var(--radius-sm); background: var(--surface-2); margin: 0.25rem 0; }
.topk-name { color: var(--text-2); font-size: 0.85rem; }
.topk-conf { color: var(--green); font-weight: 700; font-size: 0.85rem; }

/* ══════════════ Streamlit widgets ══════════════ */
/* Sidebar */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 0.6rem; }
.sb-brand { display: flex; align-items: center; gap: 10px; padding: 0.2rem 0 0.4rem 0; }
.sb-logo { width: 40px; height: 40px; border-radius: 12px; background: linear-gradient(135deg, #2E7D32, #0F766E); display: flex; align-items: center; justify-content: center; font-size: 1.3rem; box-shadow: 0 4px 12px rgba(46,125,50,0.25); }
.sb-name { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 1.15rem; color: var(--text); line-height: 1.1; }
.sb-tag { font-size: 0.75rem; color: var(--text-3); }
.sb-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-3); margin: 0.6rem 0 0.4rem 0.2rem; }
.sb-card { background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 0.6rem 0.75rem; margin: 0.3rem 0; font-size: 0.84rem; color: var(--text-2); }
.sb-model { display: flex; align-items: center; gap: 8px; padding: 5px 4px; font-size: 0.82rem; color: var(--text-2); }
.sb-model .bar { width: 54px; height: 6px; border-radius: 99px; background: var(--surface-3); overflow: hidden; }
.sb-model .bar > div { height: 100%; border-radius: 99px; }

section[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 0.55rem 0.75rem; border-radius: 10px; margin: 0; width: 100%;
    transition: background var(--ease); cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: var(--surface-2); }
section[data-testid="stSidebar"] .stRadio label:has(input:checked) { background: var(--green-soft); }
section[data-testid="stSidebar"] .stRadio label:has(input:checked) p { color: var(--green-strong) !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] .stRadio label > div:first-child { display: none; } /* hide radio dot */
section[data-testid="stSidebar"] .stRadio label p { font-size: 0.93rem; font-weight: 500; color: var(--text-2); }

/* Buttons */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: 10px; font-weight: 600; transition: all var(--ease);
    border: 1px solid var(--border-strong); background: var(--surface); color: var(--text);
    box-shadow: var(--shadow-sm);
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--green); color: var(--green-strong); background: var(--surface); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: var(--green) !important; border: 1px solid var(--green) !important; color: #fff !important;
    box-shadow: 0 2px 8px rgba(46,125,50,0.25);
}
.stButton > button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p { color: #fff !important; }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background: var(--green-strong) !important; border-color: var(--green-strong) !important;
    box-shadow: 0 4px 14px rgba(46,125,50,0.3);
}

button:disabled, .stButton > button:disabled, button[data-testid="stBaseButton-primary"]:disabled {
    background: var(--surface-3) !important; border-color: var(--surface-3) !important;
    color: var(--text-3) !important; box-shadow: none !important; cursor: not-allowed;
}
button[data-testid="stBaseButton-primary"]:disabled p { color: var(--text-3) !important; }

/* Bordered input panels: st.container(border=True, key="panel_...") */
[class*="st-key-panel_"] { background: var(--surface); box-shadow: var(--shadow-sm); border-radius: var(--radius-lg) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--surface-2); padding: 4px; border-radius: 12px; border: 1px solid var(--border); overflow-x: auto; }
.stTabs [data-baseweb="tab"] { border-radius: 9px; padding: 8px 14px; height: auto; color: var(--text-2); font-weight: 600; background: transparent; white-space: nowrap; }
.stTabs [data-baseweb="tab"]:hover { color: var(--text); background: rgba(255,255,255,0.7); }
.stTabs [aria-selected="true"] { background: var(--surface) !important; color: var(--green-strong) !important; box-shadow: var(--shadow-sm); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }

/* Expanders */
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--border) !important; border-radius: var(--radius-md) !important; box-shadow: var(--shadow-sm); }
[data-testid="stExpander"] summary { font-weight: 600; color: var(--text); }
[data-testid="stExpander"] summary:hover { color: var(--green-strong); }

/* Inputs */
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stTextArea textarea, [data-testid="stChatInput"] textarea {
    background: var(--surface) !important; border-color: var(--border-strong) !important; border-radius: 10px !important; color: var(--text) !important;
}
[data-baseweb="select"] > div:hover { border-color: var(--green) !important; }
[data-testid="stWidgetLabel"] p { color: var(--text-2) !important; font-weight: 600 !important; font-size: 0.88rem !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploader"] > section {
    background: var(--surface) !important; border: 2px dashed var(--border-strong) !important; border-radius: var(--radius-lg) !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--green) !important; background: #FAFCF8 !important; }

/* Progress */
.stProgress > div > div { background: var(--surface-3) !important; border-radius: 99px; }
.stProgress > div > div > div > div { background: var(--green) !important; border-radius: 99px; }

/* Alerts, dataframes, images */
[data-testid="stAlert"] { border-radius: var(--radius-md); }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: var(--radius-md); overflow: hidden; }
[data-testid="stImage"] img, [data-testid="stImage"] { border-radius: var(--radius-md); }
[data-testid="stPlotlyChart"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 6px; box-shadow: var(--shadow-sm); }
[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); }

/* Material icons (Streamlit's own icons) */
[data-testid="stIconMaterial"], .material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important; font-weight: normal; font-style: normal;
    line-height: 1; letter-spacing: normal; text-transform: none; white-space: nowrap; direction: ltr;
    -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #AEB9A3; }

/* Animations */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.fade-in { animation: fadeInUp 0.45s ease-out; }

/* Responsive */
@media (max-width: 900px) {
    .home-hero h1 { font-size: 1.9rem; }
    .hero-title, .page-title { font-size: 1.6rem; }
    .snapshot-grid, .diag-grid { grid-template-columns: repeat(2, 1fr); }
    .hiw-container { flex-direction: column; }
    .weather-strip { flex-wrap: wrap; gap: 0.6rem; }
    .weather-strip-divider { display: none; }
    .context-bar .ctx { border-right: none; }
}
</style>
"""
