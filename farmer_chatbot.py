"""
AGRI-X AI — Farmer Assistant Chatbot (Module 9)
Rule-based agricultural AI assistant using prediction, weather, crop, and disease knowledge bases.
"""
import streamlit as st
import re
from datetime import datetime


# ── Knowledge patterns ──
GREETING_PATTERNS = [
    r"\b(hi|hello|hey|good morning|good evening|namaste)\b",
]

DISEASE_PATTERNS = [
    (r"\b(yellow|yellowing)\b.*\b(leaf|leaves)\b", "yellow_leaves"),
    (r"\b(brown|dark)\b.*\b(spot|spots|lesion)\b", "brown_spots"),
    (r"\b(wilting|wilt|drooping)\b", "wilting"),
    (r"\b(curl|curling|curled)\b.*\b(leaf|leaves)\b", "leaf_curl"),
    (r"\b(mold|mould|fuzzy|fungus)\b", "mold"),
    (r"\b(rot|rotting)\b", "rot"),
    (r"\b(web|webbing|mite|mites|spider)\b", "spider_mites"),
    (r"\b(mosaic|mottled|mottle)\b", "mosaic"),
    (r"\b(blight)\b", "blight"),
    (r"\b(insect|pest|bug|worm|caterpillar)\b", "pest"),
]

CROP_PATTERNS = [
    r"\b(what|which)\b.*\b(crop|crops|plant|grow|sow)\b",
    r"\b(recommend|suggest|best)\b.*\b(crop|crops)\b",
    r"\b(what should i)\b.*\b(plant|grow|sow)\b",
]

YIELD_PATTERNS = [
    r"\b(yield|harvest|production|output)\b",
    r"\b(how much|expected|estimate)\b.*\b(yield|harvest|produce)\b",
    r"\b(increase|improve|boost)\b.*\b(yield|production)\b",
]

WEATHER_PATTERNS = [
    r"\b(weather|temperature|rain|rainfall|humidity|forecast)\b",
    r"\b(will it rain|is it going to rain)\b",
]

DISEASE_RESPONSES = {
    "yellow_leaves": {
        "title": "Yellow Leaves",
        "response": """**Yellowing leaves can indicate several issues:**

🦠 **Possible Diseases:**
- **Tomato Yellow Leaf Curl Virus (TYLCV)** — transmitted by whiteflies
- **Nitrogen deficiency** — common in poor soil
- **Early Blight** — yellowing around brown lesions

🔍 **Diagnosis Steps:**
1. Check for whiteflies on leaf undersides
2. Look for curling + stunted growth (viral)
3. Check if lower leaves yellow first (nitrogen deficiency)
4. Look for concentric ring patterns (early blight)

💊 **Immediate Actions:**
- Upload a leaf photo for AI diagnosis
- If whiteflies present: apply imidacloprid
- If nutrient issue: apply nitrogen-rich fertilizer
- Improve drainage and air circulation""",
    },
    "brown_spots": {
        "title": "Brown Spots",
        "response": """**Brown spots on leaves are commonly caused by:**

🦠 **Likely Diseases:**
- **Bacterial Spot** (Xanthomonas) — water-soaked spots with halos
- **Early Blight** (Alternaria) — concentric ring target patterns
- **Septoria Leaf Spot** — small spots with gray centers

💊 **Treatment:**
- Apply copper-based fungicide immediately
- Remove heavily infected leaves
- Switch to drip irrigation (avoid wetting foliage)
- Apply mancozeb as preventive

📸 **Tip:** Upload a photo for accurate AI diagnosis!""",
    },
    "wilting": {
        "title": "Wilting Plants",
        "response": """**Plant wilting can be caused by:**

💧 **Water Issues:**
- Underwatering — check soil moisture
- Overwatering — check for root rot
- Poor drainage — waterlogged soil

🦠 **Disease Causes:**
- **Late Blight** — rapid wilting with dark patches
- **Fusarium Wilt** — progressive wilting from base
- **Bacterial Wilt** — sudden collapse

💊 **Quick Check:**
1. Feel the soil — dry = water immediately
2. Check stems for dark discoloration
3. If disease suspected, upload photo for AI diagnosis""",
    },
    "leaf_curl": {
        "title": "Leaf Curling",
        "response": """**Leaf curling is a serious symptom. Common causes:**

🦠 **Primary Cause:**
- **Tomato Yellow Leaf Curl Virus (TYLCV)** — NO cure!
- Transmitted by whiteflies (Bemisia tabaci)

💊 **Actions:**
- Remove infected plants immediately
- Control whiteflies with imidacloprid or yellow sticky traps
- Use resistant varieties for next planting
- Install insect-proof netting

⚠️ **This is a viral disease — prevention is the only option!**""",
    },
    "mold": {
        "title": "Mold/Fungus",
        "response": """**Mold or fungal growth indicates high humidity conditions:**

🦠 **Likely Disease:**
- **Leaf Mold** (Passalora fulva) — olive-green fuzzy growth underneath
- **Late Blight** — white fuzzy growth on undersides

💊 **Treatment:**
- Improve ventilation immediately
- Reduce humidity below 85%
- Apply chlorothalonil or copper oxychloride
- Remove infected leaves
- Space plants wider for air circulation""",
    },
    "rot": {
        "title": "Rot",
        "response": """**Rotting can affect roots, stems, or fruit:**

🦠 **Causes:**
- **Phytophthora** (Late Blight) — tuber/fruit rot
- **Blossom End Rot** — calcium deficiency
- **Root Rot** — overwatering/poor drainage

💊 **Treatment:**
- Improve drainage immediately
- Stop overhead irrigation
- For blossom end rot: apply calcium
- For fungal rot: metalaxyl + mancozeb""",
    },
    "spider_mites": {
        "title": "Spider Mites",
        "response": """**Spider mites thrive in hot, dry conditions:**

🐛 **Identification:**
- Fine stippling/dots on leaf surface
- Webbing on leaf undersides
- Yellowing and bronzing

💊 **Treatment:**
- Spray neem oil (organic option)
- Abamectin 1.8% EC (chemical)
- Increase humidity around plants
- Strong water spray to dislodge
- Release predatory mites (biological control)""",
    },
    "mosaic": {
        "title": "Mosaic Virus",
        "response": """**Mosaic pattern indicates viral infection:**

🦠 **Disease:**
- **Tomato Mosaic Virus (ToMV)** — extremely contagious!
- Spread by touch, tools, and contaminated hands

💊 **Actions:**
- Remove and destroy infected plants
- Disinfect ALL tools with 10% bleach
- Wash hands thoroughly between plants
- Use resistant varieties (Tm-2 gene)
- Do NOT smoke near plants (TMV in tobacco)""",
    },
    "blight": {
        "title": "Blight",
        "response": """**Blight is one of the most destructive plant diseases:**

🦠 **Types:**
- **Early Blight** — target-shaped spots, lower leaves first
- **Late Blight** — CRITICAL! Can destroy entire crop in days

💊 **Treatment:**
- **Early Blight:** Chlorothalonil + azoxystrobin
- **Late Blight:** EMERGENCY metalaxyl-mancozeb every 5 days
- Remove ALL infected material
- Improve drainage, stop overhead irrigation

⚠️ **Late Blight is an EMERGENCY — act immediately!**""",
    },
    "pest": {
        "title": "Pest Infestation",
        "response": """**Common agricultural pests and treatments:**

🐛 **Identification & Treatment:**
- **Aphids** — neem oil or imidacloprid
- **Whiteflies** — yellow sticky traps + insecticides
- **Caterpillars** — Bt (Bacillus thuringiensis) spray
- **Fruit borers** — spinosad or chlorantraniliprole
- **Spider mites** — abamectin or predatory mites

💡 **IPM Approach:**
1. Scout fields regularly
2. Use biological control first
3. Chemical as last resort
4. Rotate pesticide classes""",
    },
}


def _match_intent(user_input):
    """Match user input to an intent category."""
    text = user_input.lower().strip()

    # Greetings
    for pattern in GREETING_PATTERNS:
        if re.search(pattern, text):
            return "greeting", None

    # Disease queries
    for pattern, key in DISEASE_PATTERNS:
        if re.search(pattern, text):
            return "disease", key

    # Crop recommendations
    for pattern in CROP_PATTERNS:
        if re.search(pattern, text):
            return "crop", None

    # Yield queries
    for pattern in YIELD_PATTERNS:
        if re.search(pattern, text):
            return "yield", None

    # Weather queries
    for pattern in WEATHER_PATTERNS:
        if re.search(pattern, text):
            return "weather", None

    return "general", None


def get_response(user_input, weather_current=None, prediction_result=None, location=None):
    """Generate a contextual response based on user input."""
    intent, key = _match_intent(user_input)

    if intent == "greeting":
        time_greeting = "Good morning" if datetime.now().hour < 12 else "Good afternoon" if datetime.now().hour < 17 else "Good evening"
        loc_str = f" from {location['city']}" if location else ""
        return f"""👋 {time_greeting}{loc_str}! I'm your **AGRI-X AI Farming Assistant**.

I can help you with:
- 🔬 **Disease diagnosis** — describe symptoms or upload a leaf photo
- 🌾 **Crop recommendations** — ask what to plant
- 📊 **Yield estimation** — predict expected harvest
- 🌦️ **Weather guidance** — get farming weather advice
- 💊 **Treatment advice** — how to treat plant diseases

**How can I help you today?**"""

    elif intent == "disease":
        resp_data = DISEASE_RESPONSES.get(key, {})
        return resp_data.get("response", "Please describe the symptoms in more detail, or upload a leaf photo for AI diagnosis.")

    elif intent == "crop":
        season = "Kharif" if datetime.now().month in [6,7,8,9,10] else "Rabi" if datetime.now().month in [11,12,1,2,3] else "Zaid"
        temp = weather_current.get("temperature", 25) if weather_current else 25
        response = f"""🌾 **Crop Recommendation for {season} Season**

Based on current conditions:
- 🌡️ Temperature: {temp:.0f}°C
- 📍 Location: {location.get('city', 'Unknown') if location else 'Unknown'}

"""
        if season == "Kharif":
            response += """**Top Recommended Crops:**
1. 🌾 **Rice** — ideal for monsoon
2. 🌽 **Maize** — fast growing
3. 🏵️ **Cotton** — good for warm areas
4. 🫘 **Soybean** — nitrogen-fixing
5. 🌶️ **Chilli** — high market value"""
        elif season == "Rabi":
            response += """**Top Recommended Crops:**
1. 🌾 **Wheat** — staple rabi crop
2. 🌻 **Mustard** — low water requirement
3. 🥔 **Potato** — good for cool weather
4. 🧅 **Onion** — high demand
5. 🫛 **Pulses** — soil enriching"""
        else:
            response += """**Top Recommended Crops:**
1. 🍅 **Tomato** — short duration
2. 🫑 **Bell Pepper** — high value
3. 🧅 **Onion** — summer variety"""

        response += "\n\n👉 Go to the **🌾 Crop Advisor** page for detailed analysis with revenue forecasts!"
        return response

    elif intent == "yield":
        return """📊 **Yield Prediction Guide**

To estimate your crop yield, I need:
1. **Crop type** — which crop you're growing
2. **Land area** — in acres or hectares
3. **Current weather** — temperature & humidity
4. **Soil quality** — poor, average, or good
5. **Irrigation** — available or rainfed

**Tips to increase yield:**
- 🌱 Use certified high-yield seed varieties
- 🧪 Apply balanced NPK fertilizer per soil test
- 💧 Maintain optimal irrigation schedule
- 🐛 Implement Integrated Pest Management (IPM)
- 🔄 Practice crop rotation to maintain soil health

👉 Go to the **🌾 Crop Advisor** page and use the Yield Prediction tool!"""

    elif intent == "weather":
        if weather_current:
            temp = weather_current.get("temperature", 0)
            humidity = weather_current.get("humidity", 0)
            desc = weather_current.get("description", "Unknown")
            icon = weather_current.get("icon", "🌡️")
            return f"""🌦️ **Current Weather**

{icon} **{desc}**
- 🌡️ Temperature: {temp:.1f}°C
- 💧 Humidity: {humidity}%
- 🌧️ Precipitation: {weather_current.get('precipitation', 0):.1f} mm
- 💨 Wind: {weather_current.get('wind_speed', 0):.1f} km/h

**Farming Impact:**
{"⚠️ High temperature — increase watering frequency." if temp >= 35 else ""}
{"⚠️ High humidity — watch for fungal diseases." if humidity >= 80 else ""}
{"✅ Favorable conditions for farming." if 20 <= temp <= 30 and 50 <= humidity <= 75 else ""}

👉 Go to **🌾 Crop Advisor** (Weather tab) for full forecast!"""
        else:
            return "Weather data is not available. Please check the **🌾 Crop Advisor** (Location tab)."

    else:
        # General response
        if prediction_result:
            pred_class = prediction_result.get("class", "Unknown")
            conf = prediction_result.get("confidence", 0)
            return f"""Based on your recent analysis:

🔬 **Detected: {pred_class.replace('_', ' ')}**
📊 Confidence: {conf:.0%}

I can help you with:
- 💊 Treatment options — ask "how to treat {pred_class.split('_')[-1]}"
- ⚠️ Risk assessment — ask "what's the risk level"
- 🌾 Crop alternatives — ask "what crop should I plant"

Or describe any symptoms you're seeing!"""

        return """I'm your **AGRI-X AI Farming Assistant**. Here's what I can help with:

🔬 **Disease Help** — "My leaves have brown spots", "What causes leaf curl?"
🌾 **Crop Advice** — "What crop should I plant?", "Best crop for this season"
📊 **Yield Info** — "How to increase yield?", "Expected harvest"
🌦️ **Weather** — "What's the weather?", "Will it rain?"
💊 **Treatment** — "How to treat blight?", "What pesticide for mites?"

**Try asking me something!** 🌿"""


def render_chatbot(weather_current=None, prediction_result=None, location=None):
    """Render the farmer assistant chatbot interface."""
    st.markdown('<h1 class="hero-title fade-in">🤖 Farmer Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub fade-in">AI-powered agricultural assistant — ask anything about farming, diseases, crops, or weather</p>', unsafe_allow_html=True)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Quick action buttons
    st.markdown("#### 💡 Quick Questions")
    qcols = st.columns(4)
    quick_questions = [
        ("🌿 Yellowing leaves", "My leaves are turning yellow, what should I do?"),
        ("🌾 Crop suggestion", "What crop should I plant this season?"),
        ("📊 Increase yield", "How can I increase my crop yield?"),
        ("🌦️ Weather impact", "What is the current weather and its impact on farming?"),
    ]
    for i, (label, question) in enumerate(quick_questions):
        with qcols[i]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state["chat_input_override"] = question

    # Chat display
    chat_container = st.container()
    with chat_container:
        if not st.session_state["chat_history"]:
            # Welcome message
            welcome = get_response("hello", weather_current, prediction_result, location)
            st.markdown(f"""
            <div style="background:rgba(46,204,113,0.08);border:1px solid rgba(46,204,113,0.2);border-radius:12px;padding:1rem;margin:0.5rem 0;">
                <div style="font-size:0.75rem;color:#2ecc71;font-weight:600;margin-bottom:0.3rem;">🤖 AGRI-X AI</div>
                <div style="color:#dde;font-size:0.9rem;line-height:1.6;">{welcome.replace(chr(10), '<br>')}</div>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="background:rgba(52,152,219,0.1);border:1px solid rgba(52,152,219,0.2);border-radius:12px;padding:0.8rem;margin:0.5rem 0;margin-left:20%;">
                    <div style="font-size:0.75rem;color:#3498db;font-weight:600;margin-bottom:0.3rem;">👤 You</div>
                    <div style="color:#dde;font-size:0.9rem;">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:rgba(46,204,113,0.08);border:1px solid rgba(46,204,113,0.2);border-radius:12px;padding:1rem;margin:0.5rem 0;margin-right:10%;">
                    <div style="font-size:0.75rem;color:#2ecc71;font-weight:600;margin-bottom:0.3rem;">🤖 AGRI-X AI</div>
                    <div style="color:#dde;font-size:0.9rem;line-height:1.6;">{msg['content'].replace(chr(10), '<br>').replace('**', '<b>').replace('*', '')}</div>
                </div>
                """, unsafe_allow_html=True)

    # Input
    override = st.session_state.pop("chat_input_override", None)
    user_input = st.chat_input("Ask me anything about farming...", key="chat_input_main")

    if override:
        user_input = override

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        response = get_response(user_input, weather_current, prediction_result, location)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        st.rerun()

    # Clear chat button
    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()
