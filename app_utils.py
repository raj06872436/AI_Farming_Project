"""
Utility functions and CSS for the Streamlit app.
Separated to keep app.py focused on layout/logic.
"""
import os, json, time
import numpy as np
import tensorflow as tf
from PIL import Image

# ── Path Constants ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
GRAPHS_DIR = os.path.join(REPORTS_DIR, "graphs")
GRADCAM_DIR = os.path.join(REPORTS_DIR, "gradcam")
SUMMARY_DIR = os.path.join(REPORTS_DIR, "summary")
RESEARCH_DIR = os.path.join(BASE_DIR, "research_bundle")
RESEARCH_FIGURES = os.path.join(RESEARCH_DIR, "reports", "figures")
REGISTRY_PATH = os.path.join(RESEARCH_DIR, "model_registry.json")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")

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

# ── CSS ──
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Hero header */
.hero-title {
    background: linear-gradient(135deg, #2ecc71 0%, #27ae60 50%, #1abc9c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.hero-sub {
    color: #8899aa;
    font-size: 0.95rem;
    margin-top: 0;
}

/* Glass card */
.glass-card {
    background: rgba(26, 31, 46, 0.65);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(46, 204, 113, 0.15);
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: linear-gradient(135deg, rgba(26,31,46,0.8), rgba(30,40,60,0.9));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.25s ease;
}
.metric-card:hover {
    border-color: rgba(46,204,113,0.4);
    box-shadow: 0 4px 20px rgba(46,204,113,0.1);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #2ecc71, #3498db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.75rem;
    color: #8899aa;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* Confidence bar */
.conf-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    height: 28px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 10px;
    display: flex;
    align-items: center;
    padding-left: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: white;
    transition: width 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

/* Severity badge */
.severity-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Recommendation cards */
.rec-card {
    background: rgba(26,31,46,0.5);
    border-left: 4px solid #2ecc71;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.rec-card h4 {
    margin: 0 0 0.4rem 0;
    color: #2ecc71;
    font-size: 0.9rem;
}
.rec-card p {
    margin: 0;
    color: #ccd;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1a 0%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] .stRadio > label {
    font-size: 0.85rem;
}

/* Top-K list */
.topk-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-radius: 8px;
    margin: 4px 0;
    background: rgba(255,255,255,0.03);
}
.topk-name { color: #ccd; font-size: 0.85rem; }
.topk-conf { color: #2ecc71; font-weight: 600; font-size: 0.85rem; }

/* Model comparison table */
.comparison-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
}
.comparison-table th {
    background: rgba(46,204,113,0.15);
    color: #2ecc71;
    padding: 12px 16px;
    text-align: left;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.comparison-table td {
    padding: 10px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.85rem;
    color: #dde;
}
.comparison-table tr:hover td {
    background: rgba(46,204,113,0.05);
}

/* Status badges */
.status-verified {
    background: rgba(46,204,113,0.15);
    color: #2ecc71;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-pending {
    background: rgba(243,156,18,0.15);
    color: #f39c12;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Image styling */
.report-img {
    border-radius: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
}

/* Animate on load */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeInUp 0.6s ease-out;
}

/* Pulse for live indicators */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.pulse { animation: pulse 2s infinite; }

/* ── Global Material Icons Fix ── */
/* Load Material Symbols font directly */
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

/* Force all icon elements to use the correct font */
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

</style>
"""

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

def metric_card_html(value, label):
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""

def recommendation_card_html(title, icon, text):
    return f"""
    <div class="rec-card">
        <h4>{icon} {title}</h4>
        <p>{text}</p>
    </div>"""
