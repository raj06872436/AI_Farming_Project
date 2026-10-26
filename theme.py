"""
AGRI-X AI — Design system ("Fresh Field" light theme).

One place for every colour, font and chart style used by the dashboard:
  * PALETTE / Python constants  → use in Plotly figures and inline HTML
  * CSS variables (see app_utils.CUSTOM_CSS) mirror the same names
  * A Plotly template "agrix" is registered and made the default, so every
    chart (even ones that don't pass a template) picks up the light styling.
"""
import plotly.graph_objects as go
import plotly.io as pio

# ── Core palette ──────────────────────────────────────────────────
BG = "#F6F7F2"            # page background (warm off-white)
SURFACE = "#FFFFFF"       # cards
SURFACE_2 = "#F2F5EE"     # subtle tinted panels
SURFACE_3 = "#E7ECE1"     # tracks, dividers
BORDER = "#E0E6D9"
BORDER_STRONG = "#C9D2BF"

TEXT = "#1E2A22"          # headings / primary text
TEXT_2 = "#47554B"        # body text
TEXT_3 = "#6B776E"        # captions, labels

# Accents — all ≥ 4.5:1 contrast on white so they work as text colours too
GREEN = "#2E7D32"         # primary (leaf)
GREEN_STRONG = "#1B5E20"
GREEN_SOFT = "#E5F1E3"
TEAL = "#0F766E"
SKY = "#2563A8"
SKY_STRONG = "#1E4E8C"
AMBER = "#B45309"         # harvest / money
ORANGE = "#C2410C"
RED = "#B91C1C"
RED_SOFT = "#FBE9E7"
PURPLE = "#6D28D9"
SOIL = "#8B5E34"
GOLD = "#A16207"

# Ordered colours for multi-series charts
CHART_COLORWAY = [GREEN, SKY, AMBER, PURPLE, TEAL, ORANGE, SOIL, RED]

# Fixed colour per model, used on every comparison chart
MODEL_COLORS = {
    "MobileNetV2": GREEN,
    "ResNet50": SKY,
    "EfficientNetB0": AMBER,
    "DenseNet121": PURPLE,
    "LesionNet": "#0F766E",
}

FONT_FAMILY = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
HEADING_FONT = "'Plus Jakarta Sans', Inter, 'Segoe UI', sans-serif"


def rgba(hex_color: str, alpha: float) -> str:
    """'#2E7D32', 0.15 → 'rgba(46,125,50,0.15)' (for soft fills in Plotly/HTML)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ── Plotly template ───────────────────────────────────────────────
_template = go.layout.Template(pio.templates["plotly_white"])
_template.layout.update(
    font=dict(family=FONT_FAMILY, color=TEXT_2, size=13),
    title=dict(font=dict(family=HEADING_FONT, color=TEXT, size=16), x=0.01, xanchor="left"),
    colorway=CHART_COLORWAY,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER_STRONG, font=dict(color=TEXT, family=FONT_FAMILY)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2)),
    xaxis=dict(gridcolor=SURFACE_3, linecolor=BORDER_STRONG, zerolinecolor=BORDER_STRONG,
               tickfont=dict(color=TEXT_3), title=dict(font=dict(color=TEXT_2))),
    yaxis=dict(gridcolor=SURFACE_3, linecolor=BORDER_STRONG, zerolinecolor=BORDER_STRONG,
               tickfont=dict(color=TEXT_3), title=dict(font=dict(color=TEXT_2))),
    polar=dict(bgcolor="rgba(0,0,0,0)",
               radialaxis=dict(gridcolor=SURFACE_3, linecolor=BORDER, tickfont=dict(color=TEXT_3)),
               angularaxis=dict(gridcolor=SURFACE_3, linecolor=BORDER_STRONG, tickfont=dict(color=TEXT_2))),
)
pio.templates["agrix"] = _template
pio.templates.default = "agrix"

PLOTLY_TEMPLATE = "agrix"
