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
    for d in [GRADCAM_DIR, RESEARCH_FIGURES]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith(model_name) and "sample" in f and f.endswith(".png"):
                    imgs.append(os.path.join(d, f))
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
        color = "linear-gradient(90deg, #2ecc71, #27ae60)"
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
        <div style="font-size:1.2rem;margin-bottom:4px;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""

def recommendation_card_html(title, icon, text):
    return f"""
    <div class="rec-card">
        <h4>{icon} {title}</h4>
        <p>{text}</p>
    </div>"""

def feature_card_html(icon, title, description):
    return f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{description}</div>
    </div>"""

def stat_card_html(value, label, icon="", accent_color="#2ecc71"):
    return f"""
    <div class="stat-card">
        <div class="stat-icon" style="color:{accent_color};">{icon}</div>
        <div class="stat-value" style="background:linear-gradient(135deg, {accent_color}, #3498db);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{value}</div>
        <div class="stat-label">{label}</div>
    </div>"""


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
    --border-glow: rgba(46, 204, 113, 0.3);
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
    --glass-bg: rgba(17, 24, 39, 0.65);
    --glass-border: rgba(255,255,255,0.08);
    --glass-blur: 16px;
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

/* ── Glass Card ── */
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
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-glow);
    border-color: var(--border-glow);
}

/* ── Feature Cards (Home Page) ── */
.feature-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
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
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-primary);
    opacity: 0;
    transition: opacity var(--transition-smooth);
}
.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-glow);
    border-color: var(--border-glow);
}
.feature-card:hover::before {
    opacity: 1;
}
.feature-icon {
    font-size: 2.4rem;
    margin-bottom: 0.8rem;
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
    background: linear-gradient(145deg, rgba(17,24,39,0.8), rgba(26,35,55,0.9));
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    text-align: center;
    transition: all var(--transition-smooth);
    position: relative;
}
.stat-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 4px 24px rgba(46,204,113,0.1);
    transform: translateY(-2px);
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
    background: linear-gradient(145deg, rgba(17,24,39,0.8), rgba(26,35,55,0.9));
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 1.2rem;
    text-align: center;
    transition: all var(--transition-fast);
}
.metric-card:hover {
    border-color: rgba(46,204,113,0.35);
    box-shadow: 0 4px 20px rgba(46,204,113,0.08);
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
    box-shadow: 0 0 12px rgba(0,0,0,0.2);
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

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c16 0%, #0e1525 50%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.04);
}
section[data-testid="stSidebar"] .stRadio > label {
    font-size: 0.85rem;
}

/* ── About / Info Cards ── */
.info-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
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
    transform: translateY(-1px);
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

/* ── Animations ── */
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

.fade-in { animation: fadeInUp 0.6s ease-out; }
.fade-in-left { animation: fadeInLeft 0.5s ease-out; }
.scale-in { animation: scaleIn 0.4s ease-out; }
.pulse { animation: pulse 2s infinite; }

.shimmer-text {
    background: linear-gradient(90deg, #2ecc71 25%, #1abc9c 50%, #3498db 75%, #2ecc71 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
}

/* ── Hero Banner (Home) ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(10,15,26,0.95), rgba(17,24,39,0.85));
    border: 1px solid rgba(46,204,113,0.15);
    border-radius: var(--radius-xl);
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -30%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(46,204,113,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -50%;
    left: -20%;
    width: 50%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(52,152,219,0.05) 0%, transparent 70%);
    pointer-events: none;
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

</style>
"""
