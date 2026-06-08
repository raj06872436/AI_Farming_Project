"""
AGRI-X AI — Treatment Recommendation Engine (Module 4)
Expanded knowledge base with structured treatment data.
"""
import streamlit as st

TREATMENT_DB = {
    "Pepper__bell___Bacterial_spot": {
        "disease": "Bacterial Spot", "plant": "Bell Pepper",
        "description": "A bacterial disease caused by Xanthomonas campestris pv. vesicatoria that creates water-soaked lesions on leaves, stems, and fruits.",
        "cause": "Xanthomonas bacteria, spread via rain splash, contaminated seeds, and infected transplants.",
        "symptoms": ["Water-soaked spots on leaves", "Brown/black lesions with yellow halos", "Fruit scabbing", "Premature leaf drop"],
        "severity": "Moderate to High",
        "organic_treatment": ["Copper-based sprays (Bordeaux mixture)", "Neem oil application every 7 days", "Bacillus subtilis-based biocontrol", "Remove and destroy infected plant parts"],
        "chemical_treatment": ["Copper hydroxide 77% WP at 2g/L", "Streptomycin sulfate for severe outbreaks", "Mancozeb 75% WP as preventive"],
        "prevention": ["Use certified disease-free seeds", "Crop rotation (2-3 years)", "Avoid overhead irrigation", "Sanitize tools between plants"],
        "irrigation": "Switch to drip irrigation. Avoid wetting foliage. Water early morning to allow leaves to dry.",
        "fertilizer": "Balanced NPK (10-10-10). Increase potassium for disease resistance. Apply calcium to strengthen cell walls.",
    },
    "Pepper__bell___healthy": {
        "disease": "Healthy", "plant": "Bell Pepper",
        "description": "No disease detected. The plant appears healthy with normal growth patterns.",
        "cause": "N/A — Plant is healthy.",
        "symptoms": ["No symptoms — healthy foliage"],
        "severity": "None",
        "organic_treatment": ["No treatment needed", "Continue preventive neem oil biweekly"],
        "chemical_treatment": ["No chemical treatment required"],
        "prevention": ["Maintain good air circulation", "Regular monitoring", "Balanced nutrition", "Proper spacing"],
        "irrigation": "Consistent moisture via drip irrigation. 1-1.5 inches per week.",
        "fertilizer": "Balanced NPK (10-10-10) every 2-3 weeks. Side-dress with compost.",
    },
    "Potato___Early_blight": {
        "disease": "Early Blight", "plant": "Potato",
        "description": "Fungal disease caused by Alternaria solani. Produces characteristic concentric ring 'target' patterns on lower leaves first.",
        "cause": "Alternaria solani fungus. Overwinters in infected debris and soil. Favored by warm, humid conditions.",
        "symptoms": ["Dark concentric ring lesions (target spots)", "Lower leaves affected first", "Yellowing around lesions", "Premature defoliation"],
        "severity": "Moderate",
        "organic_treatment": ["Copper fungicide sprays", "Trichoderma-based biocontrol", "Remove infected lower leaves", "Mulch to prevent soil splash"],
        "chemical_treatment": ["Chlorothalonil 75% WP at 2g/L every 7-10 days", "Azoxystrobin 23% SC at 1ml/L", "Mancozeb 75% WP as preventive"],
        "prevention": ["Crop rotation (3 years)", "Resistant varieties", "Remove infected debris", "Adequate plant spacing"],
        "irrigation": "Use furrow or drip irrigation. Avoid wetting foliage. Water early morning.",
        "fertilizer": "Adequate nitrogen, apply potash (K₂O) to strengthen cell walls. Avoid excess N.",
    },
    "Potato___Late_blight": {
        "disease": "Late Blight", "plant": "Potato",
        "description": "CRITICAL disease caused by Phytophthora infestans. Can destroy entire crop in days. The pathogen that caused the Irish Potato Famine.",
        "cause": "Phytophthora infestans (oomycete). Thrives in cool (10-24°C), humid (>90% RH) conditions with rainfall.",
        "symptoms": ["Water-soaked dark patches on leaves", "White fuzzy growth on leaf undersides", "Rapid browning and wilting", "Tuber rot (dark, firm, granular)"],
        "severity": "Critical",
        "organic_treatment": ["Copper-based fungicides immediately", "Remove and destroy ALL infected plants", "Improve drainage urgently", "Do NOT compost infected material"],
        "chemical_treatment": ["URGENT: Metalaxyl + mancozeb immediately", "Cymoxanil + mancozeb (curative)", "Dimethomorph for resistant strains", "Spray every 5-7 days"],
        "prevention": ["Use certified disease-free seed potatoes", "Plant resistant varieties", "Destroy volunteer plants", "Monitor weather — act at first sign"],
        "irrigation": "STOP overhead irrigation immediately. Drip only. Improve field drainage.",
        "fertilizer": "Reduce nitrogen. Increase phosphorus and potassium. Apply calcium.",
    },
    "Potato___healthy": {
        "disease": "Healthy", "plant": "Potato",
        "description": "No disease detected. Potato crop appears healthy.",
        "cause": "N/A", "symptoms": ["No disease symptoms"],
        "severity": "None",
        "organic_treatment": ["No treatment needed", "Preventive neem oil optional"],
        "chemical_treatment": ["Optional preventive mancozeb before monsoon season"],
        "prevention": ["Weekly monitoring", "Proper hilling", "Crop rotation"],
        "irrigation": "1-2 inches per week. Mulch to retain moisture and reduce soil splash.",
        "fertilizer": "NPK 12-12-17 at planting. Side-dress with nitrogen at tuber initiation.",
    },
    "Tomato_Bacterial_spot": {
        "disease": "Bacterial Spot", "plant": "Tomato",
        "description": "Bacterial disease caused by Xanthomonas species. Highly contagious in warm, wet conditions.",
        "cause": "Xanthomonas vesicatoria. Spread by rain, wind, contaminated seeds, and tools.",
        "symptoms": ["Small, water-soaked spots on leaves", "Dark, raised lesions on fruit", "Defoliation in severe cases", "Reduced yield"],
        "severity": "Moderate to High",
        "organic_treatment": ["Copper sprays (Bordeaux mixture 1%)", "Bacillus-based biocontrol agents", "Remove infected leaves promptly"],
        "chemical_treatment": ["Copper hydroxide 77% WP at 2g/L every 7 days", "Streptocycline 100ppm for severe cases", "Mancozeb as preventive"],
        "prevention": ["Hot water seed treatment (50°C, 25 min)", "Avoid overhead irrigation", "Crop rotation", "Resistant varieties"],
        "irrigation": "Drip only. No overhead watering. Water at base of plants.",
        "fertilizer": "Balanced NPK with extra calcium (Ca) to strengthen cell walls.",
    },
    "Tomato_Early_blight": {
        "disease": "Early Blight", "plant": "Tomato",
        "description": "Common fungal disease caused by Alternaria solani. Characteristic concentric ring patterns.",
        "cause": "Alternaria solani. Survives in soil and plant debris. Warm, humid conditions accelerate.",
        "symptoms": ["Target-shaped lesions on older leaves", "Progressive yellowing", "Lower leaves affected first", "Stem cankers possible"],
        "severity": "Moderate",
        "organic_treatment": ["Copper fungicide sprays", "Trichoderma-based products", "Neem oil", "Prune lower branches"],
        "chemical_treatment": ["Chlorothalonil 75% WP at 2g/L", "Mancozeb 75% WP preventive", "Azoxystrobin curative"],
        "prevention": ["Mulch to prevent splash", "Remove lower leaves", "Adequate spacing", "Crop rotation"],
        "irrigation": "Mulch soil surface. Drip irrigation preferred. Avoid leaf wetting.",
        "fertilizer": "Avoid excess nitrogen. Apply potash for disease resistance.",
    },
    "Tomato_Late_blight": {
        "disease": "Late Blight", "plant": "Tomato",
        "description": "CRITICAL fungal-like disease by Phytophthora infestans. Can destroy crops within days.",
        "cause": "Phytophthora infestans. Cool (10-24°C), humid (>90%) with rain = explosive spread.",
        "symptoms": ["Large, irregular water-soaked lesions", "White mold on leaf undersides", "Rapid plant collapse", "Fruit with firm, dark rot"],
        "severity": "Critical",
        "organic_treatment": ["Copper sprays immediately", "Remove ALL infected material", "Burn or bag — do NOT compost"],
        "chemical_treatment": ["EMERGENCY: Metalaxyl-mancozeb every 5 days", "Cymoxanil + mancozeb curative", "Mandipropamid for resistance management"],
        "prevention": ["Resistant varieties", "Destroy volunteers", "Fungicide before rainy season", "Monitor adjacent fields"],
        "irrigation": "STOP overhead watering. Drip only. Improve drainage immediately.",
        "fertilizer": "Reduce nitrogen. Maximize potassium and phosphorus.",
    },
    "Tomato_Leaf_Mold": {
        "disease": "Leaf Mold", "plant": "Tomato",
        "description": "Fungal disease caused by Passalora fulva. Common in greenhouses with poor ventilation.",
        "cause": "Passalora fulva (Cladosporium fulvum). Thrives in high humidity (>85%), moderate temperature.",
        "symptoms": ["Pale green/yellow spots on upper leaves", "Olive-green to brown velvety growth underneath", "Leaf curling and wilting", "Severe defoliation"],
        "severity": "Moderate",
        "organic_treatment": ["Improve ventilation immediately", "Reduce humidity below 85%", "Remove infected leaves", "Neem oil sprays"],
        "chemical_treatment": ["Chlorothalonil preventively", "Mancozeb 2.5g/L", "Copper oxychloride 3g/L"],
        "prevention": ["Greenhouse ventilation", "Wider plant spacing", "Resistant varieties", "Avoid leaf wetting"],
        "irrigation": "Reduce humidity. Use drip irrigation. Increase air circulation.",
        "fertilizer": "Balanced NPK. Avoid excess nitrogen which promotes dense foliage.",
    },
    "Tomato_Septoria_leaf_spot": {
        "disease": "Septoria Leaf Spot", "plant": "Tomato",
        "description": "Fungal disease by Septoria lycopersici. Spreads rapidly in wet conditions.",
        "cause": "Septoria lycopersici. Spread by rain splash and wind. Survives in debris.",
        "symptoms": ["Small circular spots with dark borders", "Gray-tan centers with dark specks (pycnidia)", "Lower leaves affected first", "Severe defoliation"],
        "severity": "Moderate to High",
        "organic_treatment": ["Copper-based sprays", "Remove lower infected leaves", "Mulch heavily", "Biofungicides"],
        "chemical_treatment": ["Chlorothalonil every 7-10 days", "Mancozeb preventive", "Azoxystrobin curative"],
        "prevention": ["Crop rotation (3 years)", "Remove plant debris", "Adequate spacing", "Stake plants"],
        "irrigation": "Mulch heavily. Drip irrigation only. Avoid overhead watering.",
        "fertilizer": "Standard NPK. Extra potassium for disease resistance.",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "disease": "Two-Spotted Spider Mites", "plant": "Tomato",
        "description": "Pest infestation by Tetranychus urticae. Thrives in hot, dry conditions.",
        "cause": "Tetranychus urticae (arachnid pest). Favored by heat, drought stress, and dusty conditions.",
        "symptoms": ["Fine stippling on leaf surface", "Webbing on leaf undersides", "Yellowing and bronzing", "Leaf drop in severe cases"],
        "severity": "Moderate",
        "organic_treatment": ["Neem oil sprays", "Insecticidal soap", "Release predatory mites (Phytoseiulus persimilis)", "Strong water spray to dislodge"],
        "chemical_treatment": ["Abamectin 1.8% EC at 0.5ml/L", "Spiromesifen", "Avoid broad-spectrum insecticides (kills beneficials)"],
        "prevention": ["Maintain adequate humidity", "Avoid plant stress", "Monitor undersides of leaves", "Remove weeds"],
        "irrigation": "Increase humidity. Mist plants in dry weather. Regular watering to reduce stress.",
        "fertilizer": "Standard nutrition. Avoid plant stress from nutrient deficiency.",
    },
    "Tomato__Target_Spot": {
        "disease": "Target Spot", "plant": "Tomato",
        "description": "Fungal disease by Corynespora cassiicola. Produces concentric ring patterns.",
        "cause": "Corynespora cassiicola. Warm, humid conditions with leaf wetness.",
        "symptoms": ["Small, dark brown spots with concentric rings", "Lesions on leaves, stems, fruit", "Premature defoliation", "Fruit blemishes"],
        "severity": "Moderate",
        "organic_treatment": ["Copper fungicides", "Remove infected leaves", "Improve air circulation", "Biofungicides"],
        "chemical_treatment": ["Chlorothalonil 75% WP every 7-10 days", "Azoxystrobin + difenoconazole combination"],
        "prevention": ["Improve air circulation", "Drip irrigation", "Remove plant debris", "Crop rotation"],
        "irrigation": "Improve air circulation. Drip irrigation only. Avoid leaf wetness.",
        "fertilizer": "Balanced NPK. Extra potassium and calcium.",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)", "plant": "Tomato",
        "description": "VIRAL disease transmitted by whiteflies (Bemisia tabaci). NO CURE — prevention is key.",
        "cause": "Begomovirus transmitted by silverleaf whitefly. Warm conditions increase vector activity.",
        "symptoms": ["Upward leaf curling and cupping", "Yellow leaf margins", "Stunted growth", "Reduced fruit set", "Plant remains alive but unproductive"],
        "severity": "High (incurable)",
        "organic_treatment": ["Remove infected plants immediately", "Yellow sticky traps for whiteflies", "Reflective mulch to repel vectors", "Encourage natural predators"],
        "chemical_treatment": ["Imidacloprid 17.8% SL for whitefly control", "Thiamethoxam", "Insect-proof netting"],
        "prevention": ["Resistant/tolerant varieties", "Whitefly management", "Remove weed hosts", "Crop-free period"],
        "irrigation": "Normal irrigation. Use reflective/silver mulch to repel whiteflies.",
        "fertilizer": "Boost immunity with balanced NPK + micronutrients (Zinc, Boron).",
    },
    "Tomato__Tomato_mosaic_virus": {
        "disease": "Tomato Mosaic Virus (ToMV)", "plant": "Tomato",
        "description": "Extremely contagious VIRAL disease spread by mechanical contact. Very stable virus.",
        "cause": "Tobamovirus. Transmitted by touch, tools, contaminated hands. Survives on surfaces for years.",
        "symptoms": ["Mosaic light/dark green pattern on leaves", "Leaf distortion and fern-leaf", "Stunted growth", "Fruit with yellow blotches"],
        "severity": "High (incurable)",
        "organic_treatment": ["Remove and destroy infected plants", "Wash hands with milk solution before handling", "Disinfect all tools with 10% bleach"],
        "chemical_treatment": ["No chemical cure exists", "Control aphid vectors with insecticides"],
        "prevention": ["Resistant varieties (Tm-2 gene)", "Seed treatment", "Disinfect tools", "Don't smoke near plants (TMV in tobacco)"],
        "irrigation": "Normal irrigation. Avoid handling wet plants to reduce spread.",
        "fertilizer": "Balanced nutrition for plant vigor. Micronutrients help resistance.",
    },
    "Tomato_healthy": {
        "disease": "Healthy", "plant": "Tomato",
        "description": "No disease detected. Tomato plant is healthy and showing normal growth.",
        "cause": "N/A", "symptoms": ["No disease symptoms — healthy plant"],
        "severity": "None",
        "organic_treatment": ["No treatment needed", "Preventive neem oil biweekly optional"],
        "chemical_treatment": ["Optional preventive copper spray before rainy season"],
        "prevention": ["Weekly monitoring", "Crop rotation annually", "Proper staking", "Balanced nutrition"],
        "irrigation": "1-1.5 inches per week. Mulch to retain moisture. Morning watering.",
        "fertilizer": "NPK 10-10-10 every 2 weeks. Extra calcium during fruiting to prevent blossom end rot.",
    },
}


def get_treatment(disease_class: str) -> dict:
    """Get full treatment data for a disease class."""
    return TREATMENT_DB.get(disease_class, {})


def render_treatment_cards(treatment: dict):
    """Render beautiful treatment recommendation cards."""
    if not treatment:
        st.info("No treatment data available for this condition.")
        return

    # Disease info header
    severity = treatment.get("severity", "Unknown")
    sev_colors = {"None": "#2ecc71", "Moderate": "#f39c12", "Moderate to High": "#e67e22",
                  "High": "#e74c3c", "High (incurable)": "#DC143C", "Critical": "#DC143C"}
    sev_color = sev_colors.get(severity, "#f39c12")

    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid {sev_color};">
        <h3 style="margin:0 0 0.5rem 0;color:#fff;">🔬 {treatment['disease']} — {treatment['plant']}</h3>
        <p style="color:#ccd;font-size:0.9rem;line-height:1.6;margin:0 0 0.5rem 0;">{treatment['description']}</p>
        <span style="background:rgba(255,255,255,0.08);color:{sev_color};padding:4px 12px;border-radius:12px;font-size:0.8rem;font-weight:600;">
            Severity: {severity}</span>
    </div>
    """, unsafe_allow_html=True)

    # Cause & Symptoms
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🦠 Cause", expanded=True):
            st.write(treatment.get("cause", "Unknown"))
    with col2:
        with st.expander("🔍 Symptoms", expanded=True):
            for s in treatment.get("symptoms", []):
                st.markdown(f"• {s}")

    # Treatments
    col3, col4 = st.columns(2)
    with col3:
        with st.expander("🌿 Organic Treatment", expanded=True):
            for t in treatment.get("organic_treatment", []):
                st.markdown(f"✅ {t}")
    with col4:
        with st.expander("🧪 Chemical Treatment", expanded=True):
            for t in treatment.get("chemical_treatment", []):
                st.markdown(f"💊 {t}")

    # Prevention, Irrigation, Fertilizer
    with st.expander("🛡️ Prevention Methods", expanded=False):
        for p in treatment.get("prevention", []):
            st.markdown(f"🔒 {p}")

    col5, col6 = st.columns(2)
    with col5:
        with st.expander("💧 Irrigation Advice"):
            st.write(treatment.get("irrigation", "No specific advice."))
    with col6:
        with st.expander("🌱 Fertilizer Advice"):
            st.write(treatment.get("fertilizer", "No specific advice."))
