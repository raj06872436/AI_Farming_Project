# ==============================================================================
# src/utils/recommendations.py
# Rule-based intelligent recommendation engine for agricultural decision support.
# Provides pesticide, fungicide, irrigation, and fertilizer recommendations.
# ==============================================================================

from typing import Dict, List, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Recommendation:
    """Structured recommendation output."""
    disease_name: str
    severity: str  # mild, moderate, severe
    severity_score: float  # 0.0 - 1.0
    pesticide: str
    fungicide: str
    irrigation: str
    fertilizer: str
    additional_notes: str
    urgency: str  # low, medium, high, critical


# ==============================================================================
# Disease Knowledge Base
# ==============================================================================

DISEASE_KNOWLEDGE_BASE: Dict[str, Dict] = {
    # ── Pepper Bell ──
    "Pepper__bell___Bacterial_spot": {
        "disease": "Bacterial Spot",
        "plant": "Bell Pepper",
        "pesticide": "Apply copper-based bactericides (copper hydroxide or copper sulfate) at "
                     "7-day intervals. Consider streptomycin sulfate for severe infections.",
        "fungicide": "Copper oxychloride 50% WP at 2.5g/L. Mancozeb 75% WP as preventive.",
        "irrigation": "Switch to drip irrigation. Avoid overhead watering completely. "
                      "Water early morning to allow foliage drying.",
        "fertilizer": "Apply balanced NPK (10-10-10). Increase potassium (K) to boost "
                      "disease resistance. Foliar spray of calcium chloride 0.5%.",
        "notes": "Remove and destroy infected plant debris. Practice 2-3 year crop rotation. "
                 "Use disease-free seeds and resistant varieties.",
    },
    "Pepper__bell___healthy": {
        "disease": "Healthy",
        "plant": "Bell Pepper",
        "pesticide": "No pesticide needed. Continue preventive monitoring.",
        "fungicide": "No fungicide needed. Optional preventive neem oil spray biweekly.",
        "irrigation": "Maintain consistent moisture. Water 1-2 inches per week via drip.",
        "fertilizer": "Apply balanced NPK (10-10-10) every 2-3 weeks. Side-dress with "
                      "compost during fruiting stage.",
        "notes": "Plant is healthy. Continue regular monitoring for early disease signs. "
                 "Maintain good air circulation and sanitation.",
    },

    # ── Potato ──
    "Potato___Early_blight": {
        "disease": "Early Blight",
        "plant": "Potato",
        "pesticide": "Apply chlorothalonil or mancozeb at first sign of symptoms. "
                     "Spray every 7-10 days during wet conditions.",
        "fungicide": "Azoxystrobin (Amistar) 23% SC at 1ml/L. Alternaria-specific "
                     "fungicides: difenoconazole + azoxystrobin.",
        "irrigation": "Use furrow or drip irrigation. Avoid wetting foliage. "
                      "Reduce irrigation frequency in humid conditions.",
        "fertilizer": "Ensure adequate nitrogen without excess. Apply potash (K₂O) "
                      "to strengthen cell walls. Foliar micronutrient spray (Mn, Zn).",
        "notes": "Remove lower infected leaves immediately. Mulch to prevent soil "
                 "splash. Harvest tubers promptly when mature.",
    },
    "Potato___Late_blight": {
        "disease": "Late Blight",
        "plant": "Potato",
        "pesticide": "URGENT: Apply metalaxyl + mancozeb (Ridomil Gold) immediately. "
                     "Spray every 5-7 days in outbreak conditions.",
        "fungicide": "Cymoxanil + mancozeb for curative action. Dimethomorph for "
                     "resistant strains. Rotate fungicide groups to prevent resistance.",
        "irrigation": "STOP overhead irrigation immediately. Use only drip irrigation. "
                      "Improve field drainage urgently.",
        "fertilizer": "Reduce nitrogen application. Increase phosphorus and potassium. "
                      "Apply calcium ammonium nitrate instead of urea.",
        "notes": "Late blight is HIGHLY DESTRUCTIVE — act immediately! Destroy all "
                 "infected plants. Do NOT compost infected material. Alert neighboring farms.",
    },
    "Potato___healthy": {
        "disease": "Healthy",
        "plant": "Potato",
        "pesticide": "No treatment needed. Apply preventive neem-based spray monthly.",
        "fungicide": "Optional preventive mancozeb spray before monsoon season.",
        "irrigation": "Maintain consistent moisture at 1-2 inches/week. Mulch to retain moisture.",
        "fertilizer": "NPK 12-12-17 at planting. Top-dress with nitrogen at hilling stage. "
                      "Apply boron and zinc as foliar spray.",
        "notes": "Healthy crop. Continue monitoring weekly. Hill soil around stems for "
                 "better tuber development.",
    },

    # ── Tomato ──
    "Tomato_Bacterial_spot": {
        "disease": "Bacterial Spot",
        "plant": "Tomato",
        "pesticide": "Copper hydroxide 77% WP at 2g/L every 7 days. Add mancozeb "
                     "for enhanced protection.",
        "fungicide": "Copper-based fungicides (Bordeaux mixture 1%). Streptocycline "
                     "100ppm as foliar spray for severe cases.",
        "irrigation": "Drip irrigation only. Avoid any leaf wetting. Water in morning.",
        "fertilizer": "Balanced NPK with extra calcium. Apply gypsum to soil. "
                      "Foliar calcium spray to strengthen cell walls.",
        "notes": "Highly contagious — isolate infected plants. Remove affected leaves. "
                 "Disinfect tools between plants. Use resistant varieties for next season.",
    },
    "Tomato_Early_blight": {
        "disease": "Early Blight",
        "plant": "Tomato",
        "pesticide": "Chlorothalonil 75% WP at 2g/L. Spray at first symptom appearance.",
        "fungicide": "Mancozeb 75% WP at 2.5g/L preventive. Azoxystrobin + difenoconazole "
                     "for curative treatment.",
        "irrigation": "Mulch soil surface. Drip irrigation preferred. Maintain consistent "
                      "but not excessive moisture.",
        "fertilizer": "Avoid excess nitrogen. Apply potash (K₂O 60). Micronutrient "
                      "spray with manganese and zinc.",
        "notes": "Prune lower branches for air circulation. Stake plants upright. "
                 "Remove and destroy fallen debris.",
    },
    "Tomato_Late_blight": {
        "disease": "Late Blight",
        "plant": "Tomato",
        "pesticide": "EMERGENCY: Apply metalaxyl-mancozeb immediately. Repeat every 5 days.",
        "fungicide": "Cymoxanil + mancozeb curative spray. Mandipropamid for resistance "
                     "management. Alternate fungicide modes of action.",
        "irrigation": "STOP all overhead watering. Drip only. Improve drainage immediately.",
        "fertilizer": "Reduce nitrogen drastically. Maximize potassium. Apply phosphorus "
                      "to boost root defense.",
        "notes": "CRITICAL DISEASE — can destroy entire crop in days! Remove ALL infected "
                 "plants. Burn or deep-bury infected material. Do not compost!",
    },
    "Tomato_Leaf_Mold": {
        "disease": "Leaf Mold",
        "plant": "Tomato",
        "pesticide": "Apply chlorothalonil preventively in humid conditions.",
        "fungicide": "Mancozeb 2.5g/L or copper oxychloride 3g/L. For greenhouses: "
                     "improve ventilation before chemical treatment.",
        "irrigation": "Reduce humidity: increase plant spacing, improve ventilation. "
                      "Water base of plants only.",
        "fertilizer": "Balanced NPK. Avoid excess nitrogen which promotes lush "
                      "growth susceptible to mold.",
        "notes": "Common in greenhouses/high humidity. Increase air flow. Remove "
                 "affected leaves. Resistant varieties available (Cf gene).",
    },
    "Tomato_Septoria_leaf_spot": {
        "disease": "Septoria Leaf Spot",
        "plant": "Tomato",
        "pesticide": "Chlorothalonil or copper-based spray every 7-10 days after symptoms.",
        "fungicide": "Mancozeb preventive + azoxystrobin curative. Rotate chemical "
                     "groups to prevent resistance.",
        "irrigation": "Mulch heavily to prevent soil splash. Drip irrigation. "
                      "Never water from above.",
        "fertilizer": "Standard NPK. Extra potassium for disease resistance. "
                      "Avoid overhead fertilizer application.",
        "notes": "Spreads rapidly in wet conditions. Remove infected lower leaves. "
                 "3-year crop rotation. Destroy plant debris at season end.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "disease": "Spider Mites (Two-spotted)",
        "plant": "Tomato",
        "pesticide": "Abamectin 1.8% EC at 0.5ml/L. Spiromesifen for resistance management. "
                     "Release predatory mites (Phytoseiulus persimilis) for biocontrol.",
        "fungicide": "Not applicable (pest, not fungal). Use miticides specifically.",
        "irrigation": "Increase humidity around plants — mites thrive in dry conditions. "
                      "Overhead misting can deter mites.",
        "fertilizer": "Standard nutrition. Stressed plants are more susceptible — "
                      "ensure adequate watering and nutrition.",
        "notes": "Check undersides of leaves with magnifying glass. Webbing indicates "
                 "heavy infestation. Neem oil spray as organic alternative.",
    },
    "Tomato__Target_Spot": {
        "disease": "Target Spot",
        "plant": "Tomato",
        "pesticide": "Chlorothalonil 75% WP every 7-10 days. Start at first symptoms.",
        "fungicide": "Azoxystrobin + difenoconazole combination. Boscalid for "
                     "resistant isolates.",
        "irrigation": "Improve air circulation. Stake and prune plants. "
                      "Drip irrigation only.",
        "fertilizer": "Balanced NPK. Extra potassium. Calcium foliar spray.",
        "notes": "Often confused with early blight. Concentric ring pattern on leaves. "
                 "Remove infected leaves promptly.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "plant": "Tomato",
        "pesticide": "Control whitefly vector: Imidacloprid 17.8% SL at 0.3ml/L. "
                     "Thiamethoxam 25% WG at 0.3g/L. Yellow sticky traps.",
        "fungicide": "Not applicable (viral disease). Focus on vector control.",
        "irrigation": "Normal irrigation. Use reflective mulch to repel whiteflies.",
        "fertilizer": "Boost plant immunity with balanced NPK + micronutrients. "
                      "Foliar salicylic acid spray (0.1mM) to induce systemic resistance.",
        "notes": "VIRAL — no cure exists! Remove and destroy infected plants immediately. "
                 "Use resistant/tolerant varieties (Ty genes). Control whitefly population. "
                 "Install insect-proof nets in nurseries.",
    },
    "Tomato__Tomato_mosaic_virus": {
        "disease": "Tomato Mosaic Virus (ToMV)",
        "plant": "Tomato",
        "pesticide": "No effective chemical treatment for virus. Control aphid vectors "
                     "with imidacloprid or thiamethoxam.",
        "fungicide": "Not applicable (viral disease).",
        "irrigation": "Normal irrigation. Avoid handling wet plants to prevent "
                      "mechanical transmission.",
        "fertilizer": "Balanced nutrition to maintain plant vigor. Micronutrient "
                      "supplementation (Zn, B, Mn).",
        "notes": "VIRAL — no cure! Extremely contagious via contact. Disinfect all tools "
                 "with 10% bleach. Wash hands before handling plants. Remove and destroy "
                 "infected plants. Use TMV-resistant varieties.",
    },
    "Tomato_healthy": {
        "disease": "Healthy",
        "plant": "Tomato",
        "pesticide": "No treatment needed. Optional preventive neem oil biweekly.",
        "fungicide": "Optional preventive copper spray before rainy season.",
        "irrigation": "Consistent watering 1-1.5 inches/week. Mulch to retain moisture "
                      "and prevent soil splash.",
        "fertilizer": "NPK 10-10-10 every 2 weeks. Calcium supplement during fruiting "
                      "to prevent blossom end rot.",
        "notes": "Healthy plant! Continue regular monitoring. Prune suckers for "
                 "better air circulation. Rotate crops annually.",
    },
}


# ==============================================================================
# Severity Estimation
# ==============================================================================

def estimate_severity(
    confidence: float,
    gradcam_activation_pct: Optional[float] = None
) -> Dict[str, any]:
    """
    Estimate disease severity based on model confidence and Grad-CAM activation.

    The severity is determined by a weighted combination of:
    - Model prediction confidence (higher = more certain the disease is present)
    - Grad-CAM activation percentage (higher = larger infected area)

    Args:
        confidence: Model prediction confidence (0.0 - 1.0).
        gradcam_activation_pct: Percentage of image area with high Grad-CAM
            activation (0.0 - 100.0). None if not available.

    Returns:
        Dictionary with severity classification and score.
    """
    # Base severity from confidence
    severity_score = confidence

    # Adjust with Grad-CAM activation area if available
    if gradcam_activation_pct is not None:
        # Normalize activation percentage to 0-1 range
        activation_normalized = min(gradcam_activation_pct / 100.0, 1.0)
        # Weighted combination: 40% confidence, 60% activation area
        severity_score = 0.4 * confidence + 0.6 * activation_normalized

    # Classify severity
    if severity_score < 0.35:
        severity = "Mild"
        description = "Early-stage infection detected. Minimal spread observed."
        color = "#FFA500"  # orange
    elif severity_score < 0.65:
        severity = "Moderate"
        description = "Moderate infection with noticeable spread. Treatment recommended."
        color = "#FF6347"  # tomato red
    else:
        severity = "Severe"
        description = "Severe infection with extensive spread. Immediate action required."
        color = "#DC143C"  # crimson

    return {
        "severity": severity,
        "severity_score": round(severity_score, 3),
        "infected_area_pct": round(gradcam_activation_pct, 1) if gradcam_activation_pct else None,
        "description": description,
        "color": color,
    }


# ==============================================================================
# Recommendation Engine
# ==============================================================================

def get_recommendation(
    predicted_class: str,
    confidence: float,
    gradcam_activation_pct: Optional[float] = None,
) -> Recommendation:
    """
    Generate comprehensive agricultural recommendation for a given prediction.

    Args:
        predicted_class: The predicted disease class name.
        confidence: Prediction confidence (0.0 - 1.0).
        gradcam_activation_pct: Grad-CAM activation area percentage.

    Returns:
        Recommendation dataclass with all treatment details.
    """
    # Look up disease info
    knowledge = DISEASE_KNOWLEDGE_BASE.get(predicted_class)

    if knowledge is None:
        logger.warning(f"No knowledge base entry for class: {predicted_class}")
        knowledge = {
            "disease": predicted_class.replace("_", " "),
            "plant": "Unknown",
            "pesticide": "Consult a local agricultural extension officer.",
            "fungicide": "Broad-spectrum fungicide may be applied as precaution.",
            "irrigation": "Maintain regular irrigation schedule.",
            "fertilizer": "Continue standard fertilization program.",
            "notes": "Unknown disease class. Seek professional diagnosis.",
        }

    # Determine severity
    is_healthy = "healthy" in predicted_class.lower()
    if is_healthy:
        severity = "Healthy"
        severity_score = 0.0
        urgency = "low"
    else:
        sev_info = estimate_severity(confidence, gradcam_activation_pct)
        severity = sev_info["severity"]
        severity_score = sev_info["severity_score"]
        # Map severity to urgency
        urgency_map = {"Mild": "medium", "Moderate": "high", "Severe": "critical"}
        urgency = urgency_map.get(severity, "medium")

    return Recommendation(
        disease_name=knowledge["disease"],
        severity=severity,
        severity_score=severity_score,
        pesticide=knowledge["pesticide"],
        fungicide=knowledge["fungicide"],
        irrigation=knowledge["irrigation"],
        fertilizer=knowledge["fertilizer"],
        additional_notes=knowledge["notes"],
        urgency=urgency,
    )


def get_all_disease_names() -> List[str]:
    """Return a list of all disease names in the knowledge base."""
    return list(DISEASE_KNOWLEDGE_BASE.keys())
