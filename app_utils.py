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
def confidence_bar_html(confidence, label=""):
    pct = confidence * 100
    if pct >= 80:
        color = "linear-gradient(90deg, #2ecc71, #1abc9c)"
    elif pct >= 50:
        color = "linear-gradient(90deg, #f39c12, #e67e22)"
    else:
        color = "linear-gradient(90deg, #e74c3c, #c0392b)"
    return f"""
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color};">
            {label} {pct:.1f}%
        </div>
    </div>"""

def severity_badge_html(severity, color="#FFA500"):
    return f'<span class="severity-badge" style="background:rgba(255,255,255,0.08);color:{color};border:1px solid {color};">{severity}</span>'

def metric_card_html(value, label, icon=""):
    return f"""
    <div class="metric-card">
        <div class="metric-card-accent"></div>
        <div style="font-size:1.4rem;margin-bottom:6px;filter:drop-shadow(0 0 6px rgba(46,204,113,0.3));">{icon}</div>
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
    delay = index * 0.1
    return f"""
    <div class="feature-card" style="animation-delay:{delay}s;">
        <div class="feature-card-accent"></div>
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{description}</div>
    </div>"""

def stat_card_html(value, label, icon="", accent_color="#2ecc71"):
    return f"""
    <div class="stat-card">
        <div class="stat-card-glow" style="background:radial-gradient(circle at 50% 0%, {accent_color}22, transparent 70%);"></div>
        <div class="stat-icon" style="color:{accent_color};filter:drop-shadow(0 0 8px {accent_color}44);">{icon}</div>
        <div class="stat-value" style="background:linear-gradient(135deg, {accent_color}, #3498db);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{value}</div>
        <div class="stat-label">{label}</div>
    </div>"""

def weather_strip_html(temp, humidity, rain, wind, icon="", desc=""):
    return f"""
    <div class="weather-strip">
        <div class="weather-strip-item">
            <span class="weather-strip-icon">{icon if icon else '🌡️'}</span>
            <span class="weather-strip-val">{temp:.1f}°C</span>
            <span class="weather-strip-lbl">{desc if desc else 'Temp'}</span>
        </div>
        <div class="weather-strip-divider"></div>
        <div class="weather-strip-item">
            <span class="weather-strip-icon">💧</span>
            <span class="weather-strip-val">{humidity:.0f}%</span>
            <span class="weather-strip-lbl">Humidity</span>
        </div>
        <div class="weather-strip-divider"></div>
        <div class="weather-strip-item">
            <span class="weather-strip-icon">🌧️</span>
            <span class="weather-strip-val">{rain:.1f}mm</span>
            <span class="weather-strip-lbl">Rainfall</span>
        </div>
        <div class="weather-strip-divider"></div>
        <div class="weather-strip-item">
            <span class="weather-strip-icon">💨</span>
            <span class="weather-strip-val">{wind:.1f}km/h</span>
            <span class="weather-strip-lbl">Wind</span>
        </div>
    </div>"""

def how_it_works_html():
    return """
    <div class="hiw-container">
        <div class="hiw-step">
            <div class="hiw-circle">1</div>
            <div class="hiw-title">Upload</div>
            <div class="hiw-desc">Take a photo of your crop leaf and upload it</div>
        </div>
        <div class="hiw-connector"></div>
        <div class="hiw-step">
            <div class="hiw-circle">2</div>
            <div class="hiw-title">AI Analysis</div>
            <div class="hiw-desc">4 deep learning models analyze the image</div>
        </div>
        <div class="hiw-connector"></div>
        <div class="hiw-step">
            <div class="hiw-circle">3</div>
            <div class="hiw-title">Get Results</div>
            <div class="hiw-desc">Receive diagnosis, treatment & crop advice</div>
        </div>
    </div>"""

def badge_pill_html(text, color="#2ecc71"):
    return f'<span class="badge-pill" style="color:{color};border-color:{color};background:{color}18;">{text}</span>'


# ── CSS ──
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── CSS Variables ── */
:root {
    --bg-primary: #0a0f1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.75);
    --bg-card-hover: rgba(26, 35, 55, 0.9);
    --border-subtle: rgba(255,255,255,0.06);
    --border-glow: rgba(46, 204, 113, 0.35);
    --text-primary: #f0f2f5;
    --text-secondary: #8899aa;
    --text-muted: #5a6a7a;
    --accent-green: #2ecc71;
    --accent-blue: #3498db;
    --accent-purple: #9b59b6;
    --accent-orange: #f39c12;
    --accent-red: #e74c3c;
    --accent-teal: #1abc9c;
    --gradient-primary: linear-gradient(135deg, #2ecc71 0%, #27ae60 50%, #1abc9c 100%);
    --gradient-secondary: linear-gradient(135deg, #3498db 0%, #2980b9 50%, #9b59b6 100%);
    --gradient-warm: linear-gradient(135deg, #f39c12 0%, #e74c3c 100%);
    --glass-bg: rgba(17, 24, 39, 0.6);
    --glass-border: rgba(255,255,255,0.08);
    --glass-blur: 20px;
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-xl: 28px;
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.2);
    --shadow-md: 0 4px 20px rgba(0,0,0,0.25);
    --shadow-lg: 0 8px 40px rgba(0,0,0,0.3);
    --shadow-glow: 0 0 30px rgba(46, 204, 113, 0.15);
    --transition-fast: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-smooth: 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    max-width: 1280px;
}

/* ── Hero Section ── */
.hero-title {
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -1px;
    margin-bottom: 0;
    line-height: 1.15;
}
.hero-sub {
    color: var(--text-secondary);
    font-size: 1rem;
    margin-top: 0.25rem;
    font-weight: 400;
    line-height: 1.6;
}

/* ── Section Headers ── */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(46,204,113,0.2);
}

/* ── Glass Card v2 ── */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all var(--transition-smooth);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}
.glass-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: var(--radius-lg);
    background: linear-gradient(135deg, rgba(46,204,113,0.03), transparent 50%);
    pointer-events: none;
    opacity: 0;
    transition: opacity var(--transition-smooth);
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(46,204,113,0.12), var(--shadow-glow);
    border-color: var(--border-glow);
}
.glass-card:hover::after { opacity: 1; }

/* ── Feature Cards (Home Page) ── */
.feature-card {
    background: var(--glass-bg);
    backdrop-filter: blur(14px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    padding: 1.8rem 1.5rem;
    text-align: center;
    transition: all var(--transition-smooth);
    position: relative;
    overflow: hidden;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    animation: fadeInUp 0.5s ease-out both;
}
.feature-card-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-primary);
    transform: scaleX(0);
    transition: transform 0.4s ease;
    transform-origin: left;
}
.feature-card:hover .feature-card-accent {
    transform: scaleX(1);
}
.feature-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 40px rgba(46,204,113,0.12);
    border-color: var(--border-glow);
}
.feature-icon {
    font-size: 2.6rem;
    margin-bottom: 0.8rem;
    filter: drop-shadow(0 0 8px rgba(46,204,113,0.25));
    transition: transform 0.3s ease;
}
.feature-card:hover .feature-icon {
    transform: scale(1.15);
}
.feature-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.feature-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── Stat Cards (Home Page) ── */
.stat-card {
    background: linear-gradient(145deg, rgba(17,24,39,0.85), rgba(26,35,55,0.95));
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    text-align: center;
    transition: all var(--transition-smooth);
    position: relative;
    overflow: hidden;
}
.stat-card-glow {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.4s ease;
}
.stat-card:hover .stat-card-glow { opacity: 1; }
.stat-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 4px 24px rgba(46,204,113,0.1);
    transform: translateY(-3px);
}
.stat-icon {
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── Metric Cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(145deg, rgba(17,24,39,0.85), rgba(26,35,55,0.95));
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 1.2rem;
    text-align: center;
    transition: all var(--transition-fast);
    position: relative;
    overflow: hidden;
}
.metric-card-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--gradient-primary);
}
.metric-card:hover {
    border-color: rgba(46,204,113,0.35);
    box-shadow: 0 4px 20px rgba(46,204,113,0.08);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2ecc71, #3498db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── Confidence Bar ── */
.conf-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    height: 30px;
    overflow: hidden;
    margin: 0.5rem 0;
    border: 1px solid rgba(255,255,255,0.04);
}
.conf-bar-fill {
    height: 100%;
    border-radius: 12px;
    display: flex;
    align-items: center;
    padding-left: 14px;
    font-size: 0.8rem;
    font-weight: 600;
    color: white;
    transition: width 1s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    box-shadow: 0 0 16px rgba(46,204,113,0.2);
}

/* ── Severity Badge ── */
.severity-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 18px;
    border-radius: var(--radius-xl);
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all var(--transition-fast);
}

/* ── Recommendation Cards ── */
.rec-card {
    background: rgba(17,24,39,0.5);
    border-left: 4px solid #2ecc71;
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    padding: 1.2rem 1.4rem;
    margin: 0.5rem 0;
    transition: all var(--transition-fast);
}
.rec-card:hover {
    background: rgba(26,35,55,0.6);
    border-left-color: #1abc9c;
    transform: translateX(4px);
}
.rec-card h4 {
    margin: 0 0 0.4rem 0;
    color: #2ecc71;
    font-size: 0.92rem;
    font-weight: 600;
}
.rec-card p {
    margin: 0;
    color: #bbc;
    font-size: 0.85rem;
    line-height: 1.6;
}

/* ══════════════════════════════════════════
   SIDEBAR — Premium Navigation
   ══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060a14 0%, #0c1220 50%, #111827 100%);
    border-right: 1px solid rgba(46,204,113,0.08);
}
section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-primary);
    z-index: 10;
}
/* Nav radio items as pill buttons */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    padding: 10px 14px !important;
    border-radius: 12px !important;
    border: 1px solid transparent !important;
    transition: all 0.25s ease !important;
    margin: 1px 0 !important;
    font-size: 0.88rem !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(46,204,113,0.06) !important;
    border-color: rgba(46,204,113,0.12) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div [data-testid="stMarkdownContainer"] {
    font-weight: 500 !important;
}

/* ══════════════════════════════════════════
   BUTTONS — Gradient Primary
   ══════════════════════════════════════════ */
button[kind="primary"], .stButton > button[kind="primary"] {
    background: var(--gradient-primary) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(46,204,113,0.25) !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(46,204,113,0.35) !important;
    filter: brightness(1.08) !important;
}
button[kind="primary"]:active {
    transform: translateY(0) !important;
}
/* Secondary/default buttons */
.stButton > button:not([kind="primary"]) {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: rgba(46,204,113,0.3) !important;
    background: rgba(46,204,113,0.06) !important;
}

/* ══════════════════════════════════════════
   TABS — Gradient Underline
   ══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 10px 20px;
    font-weight: 500;
    transition: all 0.25s ease;
    color: var(--text-secondary);
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(46,204,113,0.06);
    color: var(--text-primary);
}
.stTabs [aria-selected="true"] {
    color: var(--accent-green) !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: var(--gradient-primary) !important;
    height: 3px !important;
    border-radius: 3px 3px 0 0 !important;
}

/* ══════════════════════════════════════════
   PROGRESS BAR — Glowing Gradient
   ══════════════════════════════════════════ */
.stProgress > div > div > div > div {
    background: var(--gradient-primary) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 12px rgba(46,204,113,0.3);
}
.stProgress > div > div {
    border-radius: 8px !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ══════════════════════════════════════════
   EXPANDER — Rounded + Accent
   ══════════════════════════════════════════ */
.streamlit-expanderHeader {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
}
.streamlit-expanderHeader:hover {
    color: var(--accent-green) !important;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    overflow: hidden;
    transition: border-color 0.3s ease;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(46,204,113,0.15) !important;
}
[data-testid="stExpander"] details[open] {
    border-color: rgba(46,204,113,0.2) !important;
}

/* ══════════════════════════════════════════
   FILE UPLOADER — Animated Border
   ══════════════════════════════════════════ */
[data-testid="stFileUploader"] > section {
    border: 2px dashed rgba(46,204,113,0.25) !important;
    border-radius: 16px !important;
    background: rgba(46,204,113,0.02) !important;
    transition: all 0.3s ease !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"] > section:hover {
    border-color: rgba(46,204,113,0.45) !important;
    background: rgba(46,204,113,0.04) !important;
    box-shadow: 0 0 20px rgba(46,204,113,0.08) !important;
}

/* ══════════════════════════════════════════
   SELECT BOX & INPUTS — Polish
   ══════════════════════════════════════════ */
[data-baseweb="select"] > div {
    border-radius: 12px !important;
    border-color: rgba(255,255,255,0.08) !important;
    transition: all 0.25s ease !important;
}
[data-baseweb="select"] > div:hover {
    border-color: rgba(46,204,113,0.25) !important;
}
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    border-radius: 12px !important;
}

/* ══════════════════════════════════════════
   DATAFRAMES — Premium Table
   ══════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── About / Info Cards ── */
.info-card {
    background: var(--glass-bg);
    backdrop-filter: blur(14px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    padding: 1.8rem;
    margin-bottom: 1rem;
    transition: all var(--transition-smooth);
}
.info-card:hover {
    border-color: rgba(52,152,219,0.3);
    box-shadow: 0 4px 24px rgba(52,152,219,0.1);
}

/* ── Tech Badge ── */
.tech-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: rgba(46,204,113,0.1);
    border: 1px solid rgba(46,204,113,0.2);
    border-radius: var(--radius-xl);
    font-size: 0.78rem;
    font-weight: 600;
    color: #2ecc71;
    margin: 3px;
    transition: all var(--transition-fast);
}
.tech-badge:hover {
    background: rgba(46,204,113,0.18);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(46,204,113,0.15);
}

/* ── Badge Pill (floating labels) ── */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 14px;
    border: 1px solid;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0 4px;
    letter-spacing: 0.3px;
}

/* ── Reference Card ── */
.ref-card {
    background: rgba(17,24,39,0.5);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    transition: all var(--transition-fast);
    font-size: 0.85rem;
    line-height: 1.5;
    color: var(--text-secondary);
}
.ref-card:hover {
    border-color: rgba(155,89,182,0.3);
    background: rgba(26,35,55,0.5);
    transform: translateX(4px);
}
.ref-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    background: var(--gradient-secondary);
    border-radius: 50%;
    font-size: 0.7rem;
    font-weight: 700;
    color: white;
    margin-right: 8px;
}

/* ── Research Table ── */
.research-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--radius-md);
    overflow: hidden;
    font-size: 0.85rem;
}
.research-table th {
    background: rgba(46,204,113,0.12);
    color: var(--accent-green);
    padding: 12px 16px;
    text-align: left;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    border-bottom: 2px solid rgba(46,204,113,0.2);
}
.research-table td {
    padding: 11px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    color: var(--text-primary);
}
.research-table tr:hover td {
    background: rgba(46,204,113,0.04);
}
.research-table tr:last-child td {
    border-bottom: none;
}

/* ── Status Badges ── */
.status-verified {
    background: rgba(46,204,113,0.12);
    color: #2ecc71;
    padding: 4px 12px;
    border-radius: var(--radius-xl);
    font-size: 0.75rem;
    font-weight: 600;
}
.status-pending {
    background: rgba(243,156,18,0.12);
    color: #f39c12;
    padding: 4px 12px;
    border-radius: var(--radius-xl);
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Image styling ── */
.report-img {
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-md);
    margin-bottom: 1rem;
}

/* ── Divider ── */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(46,204,113,0.3), rgba(52,152,219,0.3), transparent);
    margin: 2rem 0;
    border: none;
}

/* ── Comparison Table ── */
.comparison-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--radius-md);
    overflow: hidden;
}
.comparison-table th {
    background: rgba(46,204,113,0.12);
    color: var(--accent-green);
    padding: 12px 16px;
    text-align: left;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}
.comparison-table td {
    padding: 10px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem;
    color: var(--text-primary);
}
.comparison-table tr:hover td {
    background: rgba(46,204,113,0.05);
}

/* ── Top-K List ── */
.topk-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border-radius: var(--radius-sm);
    margin: 4px 0;
    background: rgba(255,255,255,0.02);
    border: 1px solid transparent;
    transition: all var(--transition-fast);
}
.topk-item:hover {
    background: rgba(46,204,113,0.05);
    border-color: rgba(46,204,113,0.1);
}
.topk-name { color: #ccd; font-size: 0.85rem; }
.topk-conf { color: #2ecc71; font-weight: 600; font-size: 0.85rem; }

/* ══════════════════════════════════════════
   WEATHER STRIP — Glass Bar
   ══════════════════════════════════════════ */
.weather-strip {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    background: linear-gradient(135deg, rgba(17,24,39,0.7), rgba(26,35,55,0.8));
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
}
.weather-strip-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0 1.5rem;
}
.weather-strip-icon { font-size: 1.4rem; margin-bottom: 4px; }
.weather-strip-val { font-size: 1.1rem; font-weight: 700; color: #f0f2f5; }
.weather-strip-lbl { font-size: 0.65rem; color: #8899aa; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }
.weather-strip-divider {
    width: 1px;
    height: 40px;
    background: rgba(255,255,255,0.08);
}

/* ══════════════════════════════════════════
   HOW IT WORKS — Step Flow
   ══════════════════════════════════════════ */
.hiw-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 2rem 1rem;
}
.hiw-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    flex: 1;
    max-width: 200px;
}
.hiw-circle {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: var(--gradient-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 20px rgba(46,204,113,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.hiw-step:hover .hiw-circle {
    transform: scale(1.12);
    box-shadow: 0 6px 28px rgba(46,204,113,0.4);
}
.hiw-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.hiw-desc {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.4;
}
.hiw-connector {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, rgba(46,204,113,0.4), rgba(52,152,219,0.4));
    margin: 0 0.5rem;
    margin-bottom: 2rem;
    border-radius: 1px;
}

/* ══════════════════════════════════════════
   ANIMATIONS
   ══════════════════════════════════════════ */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.92); }
    to { opacity: 1; transform: scale(1); }
}
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(46,204,113,0); }
    50% { box-shadow: 0 0 20px 4px rgba(46,204,113,0.15); }
}
@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
@keyframes floatUp {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(46,204,113,0.15); }
    50% { border-color: rgba(46,204,113,0.35); }
}

.fade-in { animation: fadeInUp 0.6s ease-out; }
.fade-in-left { animation: fadeInLeft 0.5s ease-out; }
.scale-in { animation: scaleIn 0.4s ease-out; }
.pulse { animation: pulse 2s infinite; }
.float-up { animation: floatUp 3s ease-in-out infinite; }

.shimmer-text {
    background: linear-gradient(90deg, #2ecc71 25%, #1abc9c 50%, #3498db 75%, #2ecc71 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
}

/* ── Hero Banner (Home) — Animated ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(10,15,26,0.96), rgba(17,24,39,0.88));
    border: 1px solid rgba(46,204,113,0.15);
    border-radius: var(--radius-xl);
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    animation: borderGlow 4s ease-in-out infinite;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -30%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(46,204,113,0.08) 0%, transparent 70%);
    pointer-events: none;
    animation: floatUp 6s ease-in-out infinite;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -50%; left: -20%;
    width: 50%; height: 200%;
    background: radial-gradient(ellipse, rgba(52,152,219,0.06) 0%, transparent 70%);
    pointer-events: none;
    animation: floatUp 8s ease-in-out infinite reverse;
}

/* ── Methodology Card ── */
.method-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 0.8rem;
    transition: all var(--transition-smooth);
    position: relative;
}
.method-card:hover {
    border-color: rgba(26,188,156,0.3);
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(26,188,156,0.08);
}
.method-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--gradient-primary);
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 700;
    color: white;
    margin-right: 10px;
}
.method-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    display: inline;
}
.method-desc {
    margin-top: 0.6rem;
    font-size: 0.85rem;
    color: var(--text-secondary);
    line-height: 1.6;
}

/* ── Streamlit metric polish ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(17,24,39,0.7), rgba(26,35,55,0.8));
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 14px 16px;
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(46,204,113,0.2);
    transform: translateY(-2px);
}
[data-testid="stMetricValue"] {
    font-weight: 800 !important;
}

/* ── Global Material Icons Fix ── */
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

[data-testid="stIconMaterial"],
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important;
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
    overflow: hidden;
    width: 20px;
    height: 20px;
}

/* ── Scrollbar Styling ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(17,24,39,0.5);
}
::-webkit-scrollbar-thumb {
    background: rgba(46,204,113,0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(46,204,113,0.5);
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero-banner { padding: 2rem 1.5rem; }
    .hero-title { font-size: 1.8rem !important; }
    .feature-card { min-height: 160px; padding: 1.2rem; }
    .weather-strip { flex-wrap: wrap; gap: 0.5rem; }
    .weather-strip-divider { display: none; }
    .hiw-container { flex-direction: column; }
    .hiw-connector { width: 2px; height: 30px; margin: 0.3rem 0; }
    .stat-value { font-size: 1.5rem; }
}

</style>
"""
