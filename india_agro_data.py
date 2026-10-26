"""
AGRI-X AI — India agro-climatic reference data.

Used whenever live soil / climate services can't answer (they are often down, and
SoilGrids has no data for built-up city centres). Every Indian state and union
territory has a profile, and large, varied states are split into agro-climatic
regions matched by district name — so Nashik gets "North Maharashtra, medium black
soil, onion / grapes / tomato country" instead of a generic "Loamy, 60 %".

Values are typical (median) conditions compiled from general Indian soil maps
(ICAR-NBSS&LUP soil groups), IMD rainfall normals and state agriculture profiles.
They describe a region, not a single field — a soil test always beats them.

Plain-language analogy: this is the "regional almanac" a local agronomist carries
in their head; live satellite/soil data, when available, is the field visit.
"""
import re

# Indian soil groups → (display name, typical texture, short description)
SOIL_GROUPS = {
    "Alluvial": ("Alluvial", "Loam", "River-deposited, fertile, easy to work; usually low in nitrogen."),
    "Black": ("Black (Regur)", "Clay", "Deep clayey 'cotton soil'; holds water well, cracks when dry, rich in lime and potash."),
    "Red": ("Red & yellow", "Sandy Loam", "Light, well-drained, low in nitrogen and phosphorus; responds well to manure."),
    "Laterite": ("Laterite", "Sandy Clay Loam", "Acidic, leached by heavy rain; needs lime and organic matter."),
    "Arid": ("Desert / arid", "Sandy", "Sandy, low organic matter, often saline; crops need irrigation."),
    "Mountain": ("Mountain / forest", "Loam", "Rich in organic matter, often acidic; terraced fields."),
}

# Monthly share of annual rainfall (Jan→Dec) for different rainfall regimes
RAIN_PATTERNS = {
    "SW": [0.01, 0.01, 0.01, 0.02, 0.03, 0.14, 0.27, 0.25, 0.16, 0.06, 0.02, 0.02],   # south-west monsoon
    "NE": [0.03, 0.02, 0.02, 0.04, 0.06, 0.05, 0.07, 0.09, 0.11, 0.19, 0.20, 0.12],   # Tamil Nadu type
    "NEI": [0.01, 0.02, 0.04, 0.09, 0.13, 0.17, 0.17, 0.14, 0.11, 0.07, 0.03, 0.02],  # North-east India
    "KER": [0.01, 0.01, 0.02, 0.05, 0.08, 0.22, 0.21, 0.14, 0.08, 0.10, 0.06, 0.02],  # Kerala / coast
    "WH": [0.10, 0.10, 0.12, 0.09, 0.06, 0.05, 0.14, 0.13, 0.06, 0.03, 0.03, 0.09],   # Western Himalaya
}


def _p(name, zone, soil, ph, oc, npk, rain, temps, pattern, crops, texture=None):
    """
    Build a profile.
      oc    – organic carbon, g/kg (typical)
      npk   – "LMH" style status string for N, P, K (L=low, M=medium, H=high)
      rain  – annual normal rainfall, mm
      temps – mean °C for (Dec–Feb winter, Mar–May summer, Jun–Sep monsoon, Oct–Nov post-monsoon)
    """
    status = {"L": "Low", "M": "Medium", "H": "High"}
    return {
        "name": name, "zone": zone, "soil_group": soil,
        "soil_label": SOIL_GROUPS[soil][0],
        "texture": texture or SOIL_GROUPS[soil][1],
        "soil_note": SOIL_GROUPS[soil][2],
        "ph": ph, "oc": oc,
        "n": status[npk[0]], "p": status[npk[1]], "k": status[npk[2]],
        "rain": rain, "temps": temps, "pattern": pattern, "crops": crops,
    }


# ── State / UT defaults ──────────────────────────────────────────────
STATES = {
    "Andhra Pradesh": _p("Andhra Pradesh", "East Coast Plains", "Red", 7.2, 5, "LMH", 940, (24, 31, 29, 27), "SW",
                         ["Rice", "Chilli", "Cotton", "Maize", "Groundnut", "Tur (Arhar)", "Urad", "Banana", "Mango", "Tomato"]),
    "Arunachal Pradesh": _p("Arunachal Pradesh", "Eastern Himalaya", "Mountain", 5.0, 25, "MLM", 2780, (12, 19, 23, 17), "NEI",
                            ["Rice", "Maize", "Ginger", "Potato", "Mustard"]),
    "Assam": _p("Assam", "Brahmaputra Valley", "Alluvial", 5.3, 11, "MLM", 2820, (17, 25, 29, 23), "NEI",
                ["Rice", "Mustard", "Potato", "Sugarcane", "Banana", "Ginger", "Lentil (Masur)"]),
    "Bihar": _p("Bihar", "Middle Gangetic Plains", "Alluvial", 7.4, 5, "LMM", 1180, (17, 29, 30, 23), "SW",
                ["Rice", "Wheat", "Maize", "Lentil (Masur)", "Chickpea", "Potato", "Mustard", "Onion", "Banana"]),
    "Chhattisgarh": _p("Chhattisgarh", "Eastern Plateau & Hills", "Red", 6.1, 6, "LLM", 1290, (21, 31, 28, 24), "SW",
                       ["Rice", "Maize", "Chickpea", "Tur (Arhar)", "Soybean", "Urad", "Mustard", "Tomato"]),
    "Goa": _p("Goa", "West Coast Plains & Ghats", "Laterite", 5.5, 15, "MLL", 3000, (26, 29, 27, 28), "KER",
              ["Rice", "Mango", "Banana", "Groundnut", "Okra (Bhindi)"]),
    "Gujarat": _p("Gujarat", "Gujarat Plains & Hills", "Black", 7.9, 5, "LMH", 830, (21, 31, 29, 27), "SW",
                  ["Cotton", "Groundnut", "Wheat", "Castor", "Cumin", "Bajra", "Chickpea", "Potato", "Banana"],
                  texture="Clay Loam"),
    "Haryana": _p("Haryana", "Trans-Gangetic Plains", "Alluvial", 8.0, 4, "LMM", 570, (14, 30, 31, 23), "SW",
                  ["Wheat", "Rice", "Mustard", "Cotton", "Bajra", "Sugarcane", "Barley"], texture="Sandy Loam"),
    "Himachal Pradesh": _p("Himachal Pradesh", "Western Himalaya", "Mountain", 6.3, 16, "MMM", 1250, (8, 18, 22, 15), "WH",
                           ["Wheat", "Maize", "Potato", "Barley", "Ginger", "Tomato", "Cabbage", "Cauliflower", "Garlic"]),
    "Jharkhand": _p("Jharkhand", "Eastern Plateau & Hills", "Red", 5.6, 6, "LLM", 1300, (18, 30, 28, 22), "SW",
                    ["Rice", "Maize", "Tur (Arhar)", "Urad", "Chickpea", "Mustard", "Tomato", "Cauliflower"]),
    "Karnataka": _p("Karnataka", "Southern Plateau & Hills", "Red", 6.8, 6, "LMM", 1150, (23, 28, 25, 24), "SW",
                    ["Ragi", "Rice", "Maize", "Jowar", "Tur (Arhar)", "Groundnut", "Sugarcane", "Tomato", "Cotton"]),
    "Kerala": _p("Kerala", "West Coast Plains & Ghats", "Laterite", 5.2, 15, "MLL", 3000, (26, 29, 27, 27), "KER",
                 ["Rice", "Banana", "Ginger", "Turmeric", "Mango"]),
    "Madhya Pradesh": _p("Madhya Pradesh", "Central Plateau & Hills", "Black", 7.6, 5, "LLH", 1050, (18, 30, 28, 23), "SW",
                         ["Soybean", "Wheat", "Chickpea", "Maize", "Rice", "Tur (Arhar)", "Mustard", "Lentil (Masur)", "Garlic"],
                         texture="Clay"),
    "Maharashtra": _p("Maharashtra", "Western Plateau & Hills", "Black", 7.8, 5, "LMH", 1100, (22, 30, 27, 25), "SW",
                      ["Soybean", "Cotton", "Sugarcane", "Tur (Arhar)", "Jowar", "Onion", "Chickpea", "Rice", "Grapes"],
                      texture="Clay"),
    "Manipur": _p("Manipur", "Eastern Himalaya", "Mountain", 5.2, 20, "MLM", 1480, (14, 22, 25, 20), "NEI",
                  ["Rice", "Maize", "Potato", "Ginger", "Mustard"]),
    "Meghalaya": _p("Meghalaya", "Eastern Himalaya", "Laterite", 5.0, 22, "MLL", 2800, (13, 20, 23, 18), "NEI",
                    ["Rice", "Potato", "Ginger", "Turmeric", "Maize"]),
    "Mizoram": _p("Mizoram", "Eastern Himalaya", "Mountain", 5.0, 22, "MLL", 2500, (16, 23, 24, 21), "NEI",
                  ["Rice", "Maize", "Ginger", "Turmeric", "Banana"]),
    "Nagaland": _p("Nagaland", "Eastern Himalaya", "Mountain", 5.0, 22, "MLM", 1900, (14, 22, 25, 20), "NEI",
                   ["Rice", "Maize", "Potato", "Ginger", "Soybean"]),
    "Odisha": _p("Odisha", "East Coast Plains & Plateau", "Red", 6.0, 6, "LLM", 1450, (21, 30, 28, 25), "SW",
                 ["Rice", "Urad", "Moong", "Groundnut", "Maize", "Mustard", "Tomato", "Brinjal"]),
    "Punjab": _p("Punjab", "Trans-Gangetic Plains", "Alluvial", 8.0, 4, "LMM", 590, (13, 29, 31, 22), "SW",
                 ["Wheat", "Rice", "Cotton", "Maize", "Potato", "Sugarcane", "Mustard"], texture="Loam"),
    "Rajasthan": _p("Rajasthan", "Western Dry Region", "Arid", 8.2, 3, "LLM", 530, (17, 31, 31, 25), "SW",
                    ["Bajra", "Mustard", "Wheat", "Chickpea", "Moong", "Cumin", "Groundnut", "Coriander"]),
    "Sikkim": _p("Sikkim", "Eastern Himalaya", "Mountain", 5.2, 25, "MLM", 2740, (8, 15, 19, 14), "NEI",
                 ["Maize", "Ginger", "Potato", "Rice", "Cabbage"]),
    "Tamil Nadu": _p("Tamil Nadu", "East Coast Plains & Hills", "Red", 7.3, 5, "LMH", 950, (25, 30, 29, 27), "NE",
                     ["Rice", "Groundnut", "Sugarcane", "Banana", "Maize", "Urad", "Cotton", "Turmeric", "Tomato"]),
    "Telangana": _p("Telangana", "Southern Plateau & Hills", "Red", 7.2, 5, "LMH", 905, (23, 32, 28, 25), "SW",
                    ["Rice", "Cotton", "Maize", "Tur (Arhar)", "Chilli", "Soybean", "Turmeric", "Groundnut"]),
    "Tripura": _p("Tripura", "Eastern Himalaya", "Laterite", 5.0, 15, "MLL", 2200, (19, 27, 28, 25), "NEI",
                  ["Rice", "Potato", "Mustard", "Ginger", "Banana"]),
    "Uttar Pradesh": _p("Uttar Pradesh", "Upper & Middle Gangetic Plains", "Alluvial", 7.8, 4, "LMM", 950, (16, 30, 30, 23), "SW",
                        ["Wheat", "Rice", "Sugarcane", "Potato", "Mustard", "Chickpea", "Lentil (Masur)", "Maize"]),
    "Uttarakhand": _p("Uttarakhand", "Western Himalaya", "Mountain", 6.0, 15, "MMM", 1550, (10, 20, 23, 16), "WH",
                      ["Rice", "Wheat", "Ragi", "Potato", "Ginger", "Garlic", "Cabbage"]),
    "West Bengal": _p("West Bengal", "Lower Gangetic Plains", "Alluvial", 6.4, 7, "MLM", 1750, (19, 29, 29, 25), "SW",
                      ["Rice", "Potato", "Mustard", "Lentil (Masur)", "Wheat", "Sesame", "Cauliflower", "Brinjal"]),
    "Andaman and Nicobar Islands": _p("Andaman & Nicobar", "Islands", "Laterite", 5.5, 15, "MLL", 3000, (26, 28, 27, 27), "KER",
                                      ["Rice", "Banana", "Ginger", "Okra (Bhindi)"]),
    "Chandigarh": _p("Chandigarh", "Trans-Gangetic Plains", "Alluvial", 7.8, 5, "LMM", 1100, (14, 28, 29, 22), "SW",
                     ["Wheat", "Rice", "Maize", "Mustard"]),
    "Dadra and Nagar Haveli and Daman and Diu": _p("Dadra & Nagar Haveli, Daman & Diu", "Gujarat Plains & Hills", "Black", 7.0, 7,
                                                   "LMH", 2000, (23, 29, 27, 27), "SW", ["Rice", "Ragi", "Sugarcane", "Mango"]),
    "Delhi": _p("Delhi", "Trans-Gangetic Plains", "Alluvial", 7.9, 4, "LMM", 790, (15, 30, 31, 23), "SW",
                ["Wheat", "Mustard", "Bajra", "Cauliflower", "Tomato"]),
    "Jammu and Kashmir": _p("Jammu & Kashmir", "Western Himalaya", "Mountain", 6.8, 12, "MMM", 1100, (5, 15, 23, 13), "WH",
                            ["Rice", "Maize", "Wheat", "Mustard", "Potato"]),
    "Ladakh": _p("Ladakh", "Cold Desert", "Arid", 8.0, 4, "LLM", 100, (-8, 6, 15, 3), "WH",
                 ["Barley", "Wheat", "Potato", "Cabbage"]),
    "Lakshadweep": _p("Lakshadweep", "Islands", "Arid", 8.0, 6, "LLM", 1600, (27, 29, 28, 28), "KER", ["Banana"],
                      texture="Sandy"),
    "Puducherry": _p("Puducherry", "East Coast Plains & Hills", "Alluvial", 7.0, 6, "LMM", 1300, (25, 30, 30, 27), "NE",
                     ["Rice", "Urad", "Groundnut", "Sugarcane"]),
}

# ── Agro-climatic regions inside large, varied states (matched by district) ──
REGIONS = [
    # Maharashtra
    (_p("Konkan (Maharashtra)", "West Coast Plains & Ghats", "Laterite", 5.6, 12, "MLL", 3000, (24, 29, 27, 27), "KER",
        ["Rice", "Ragi", "Mango", "Urad", "Groundnut"]),
     "Maharashtra", ["mumbai", "mumbai suburban", "thane", "palghar", "raigad", "ratnagiri", "sindhudurg",
                     "navi mumbai", "kalyan", "vasai", "alibag"]),
    (_p("North Maharashtra (Khandesh & Nashik)", "Western Plateau & Hills", "Black", 7.8, 5, "LMH", 760, (21, 30, 26, 24), "SW",
        ["Onion", "Grapes", "Tomato", "Pomegranate", "Soybean", "Maize", "Bajra", "Wheat", "Banana", "Cotton", "Sugarcane"],
        texture="Clay Loam"),
     "Maharashtra", ["nashik", "nasik", "dhule", "nandurbar", "jalgaon", "ahmednagar", "ahilyanagar", "malegaon",
                     "sangamner", "shirdi", "niphad", "igatpuri", "sinnar"]),
    (_p("Western Maharashtra", "Western Plateau & Hills", "Black", 7.9, 5, "LMH", 720, (22, 30, 25, 24), "SW",
        ["Sugarcane", "Jowar", "Soybean", "Pomegranate", "Grapes", "Onion", "Turmeric", "Rice", "Groundnut"], texture="Clay"),
     "Maharashtra", ["pune", "satara", "sangli", "kolhapur", "solapur", "baramati", "pimpri chinchwad", "karad"]),
    (_p("Marathwada", "Western Plateau & Hills", "Black", 8.0, 4, "LMH", 760, (22, 31, 27, 25), "SW",
        ["Soybean", "Cotton", "Tur (Arhar)", "Jowar", "Chickpea", "Sugarcane", "Bajra", "Sunflower"], texture="Clay"),
     "Maharashtra", ["aurangabad", "chhatrapati sambhajinagar", "jalna", "beed", "bid", "latur", "osmanabad", "dharashiv",
                     "nanded", "parbhani", "hingoli"]),
    (_p("Vidarbha", "Central Plateau & Hills", "Black", 7.6, 5, "LMH", 1050, (21, 32, 28, 24), "SW",
        ["Cotton", "Soybean", "Tur (Arhar)", "Rice", "Wheat", "Chickpea", "Chilli"], texture="Clay"),
     "Maharashtra", ["nagpur", "wardha", "amravati", "akola", "yavatmal", "buldhana", "washim", "chandrapur",
                     "gadchiroli", "bhandara", "gondia"]),
    # Gujarat
    (_p("Saurashtra", "Gujarat Plains & Hills", "Black", 8.0, 5, "LMH", 650, (21, 30, 29, 27), "SW",
        ["Groundnut", "Cotton", "Wheat", "Chickpea", "Cumin", "Sesame", "Onion", "Coriander"], texture="Clay Loam"),
     "Gujarat", ["rajkot", "jamnagar", "junagadh", "amreli", "bhavnagar", "porbandar", "surendranagar", "gir somnath",
                 "morbi", "botad", "devbhumi dwarka", "dwarka"]),
    (_p("North Gujarat", "Gujarat Plains & Hills", "Alluvial", 8.0, 3, "LMM", 720, (20, 31, 29, 26), "SW",
        ["Castor", "Cumin", "Potato", "Mustard", "Bajra", "Wheat", "Groundnut"], texture="Sandy Loam"),
     "Gujarat", ["banaskantha", "mehsana", "patan", "sabarkantha", "gandhinagar", "aravalli", "palanpur", "deesa"]),
    (_p("Central Gujarat", "Gujarat Plains & Hills", "Alluvial", 7.8, 5, "LMM", 850, (21, 31, 29, 27), "SW",
        ["Rice", "Wheat", "Maize", "Banana", "Cotton", "Potato", "Tur (Arhar)"], texture="Loam"),
     "Gujarat", ["ahmedabad", "anand", "kheda", "vadodara", "baroda", "panchmahal", "dahod", "mahisagar",
                 "chhota udaipur", "nadiad"]),
    (_p("South Gujarat", "Gujarat Plains & Hills", "Black", 7.4, 7, "LMH", 1500, (23, 30, 28, 28), "SW",
        ["Sugarcane", "Rice", "Banana", "Mango", "Cotton", "Jowar", "Okra (Bhindi)"], texture="Clay"),
     "Gujarat", ["surat", "bharuch", "navsari", "valsad", "tapi", "narmada", "dang", "the dangs", "vapi"]),
    (_p("Kutch", "Gujarat Plains & Hills", "Arid", 8.4, 2, "LLM", 380, (20, 30, 30, 27), "SW",
        ["Cotton", "Castor", "Bajra", "Groundnut", "Cumin", "Pomegranate", "Mango"], texture="Sandy Loam"),
     "Gujarat", ["kutch", "kachchh", "bhuj", "gandhidham"]),
    # Karnataka
    (_p("Coastal Karnataka", "West Coast Plains & Ghats", "Laterite", 5.4, 14, "MLL", 3800, (26, 29, 26, 27), "KER",
        ["Rice", "Banana", "Ginger", "Turmeric", "Mango"]),
     "Karnataka", ["dakshina kannada", "udupi", "uttara kannada", "mangaluru", "mangalore", "karwar"]),
    (_p("Malnad (Karnataka)", "Southern Plateau & Hills", "Laterite", 5.6, 14, "MLM", 1900, (22, 26, 23, 23), "KER",
        ["Rice", "Maize", "Ginger", "Banana", "Turmeric"]),
     "Karnataka", ["shivamogga", "shimoga", "chikkamagaluru", "chikmagalur", "kodagu", "coorg"]),
    (_p("North Interior Karnataka", "Southern Plateau & Hills", "Black", 8.0, 4, "LMH", 650, (23, 31, 27, 25), "SW",
        ["Jowar", "Tur (Arhar)", "Chickpea", "Cotton", "Sugarcane", "Sunflower", "Maize", "Onion", "Groundnut", "Bajra"],
        texture="Clay"),
     "Karnataka", ["belagavi", "belgaum", "bagalkot", "vijayapura", "bijapur", "kalaburagi", "gulbarga", "bidar",
                   "raichur", "koppal", "ballari", "bellary", "vijayanagara", "yadgir", "gadag", "dharwad", "haveri",
                   "hubli", "hubballi"]),
    (_p("South Interior Karnataka", "Southern Plateau & Hills", "Red", 6.6, 6, "LMM", 780, (22, 27, 24, 23), "SW",
        ["Ragi", "Rice", "Maize", "Groundnut", "Tomato", "Sugarcane", "Mango", "Tur (Arhar)"], texture="Sandy Loam"),
     "Karnataka", ["bengaluru", "bangalore", "bengaluru urban", "bengaluru rural", "ramanagara", "kolar",
                   "chikkaballapur", "tumakuru", "tumkur", "chitradurga", "davanagere", "mandya", "mysuru", "mysore",
                   "chamarajanagar", "hassan"]),
    # Madhya Pradesh
    (_p("Malwa Plateau", "Central Plateau & Hills", "Black", 7.8, 5, "LLH", 900, (18, 30, 27, 23), "SW",
        ["Soybean", "Wheat", "Chickpea", "Maize", "Garlic", "Onion", "Coriander"], texture="Clay"),
     "Madhya Pradesh", ["indore", "ujjain", "dewas", "ratlam", "mandsaur", "neemuch", "shajapur", "agar malwa",
                        "dhar", "rajgarh", "sehore", "bhopal", "jhabua", "alirajpur"]),
    (_p("Gird (Gwalior–Chambal)", "Central Plateau & Hills", "Alluvial", 8.0, 4, "LMM", 750, (16, 31, 30, 23), "SW",
        ["Mustard", "Wheat", "Bajra", "Chickpea", "Sesame"], texture="Sandy Loam"),
     "Madhya Pradesh", ["gwalior", "bhind", "morena", "shivpuri", "datia", "sheopur", "guna", "ashoknagar"]),
    (_p("Bundelkhand (MP)", "Central Plateau & Hills", "Red", 7.3, 4, "LLM", 950, (18, 31, 28, 23), "SW",
        ["Chickpea", "Lentil (Masur)", "Wheat", "Urad", "Sesame", "Soybean"]),
     "Madhya Pradesh", ["sagar", "chhatarpur", "tikamgarh", "panna", "damoh", "niwari"]),
    (_p("Mahakoshal / Satpura", "Central Plateau & Hills", "Black", 7.0, 6, "LMM", 1300, (18, 29, 27, 22), "SW",
        ["Rice", "Wheat", "Chickpea", "Maize", "Sugarcane", "Soybean", "Tur (Arhar)"]),
     "Madhya Pradesh", ["jabalpur", "katni", "mandla", "seoni", "balaghat", "narsinghpur", "chhindwara", "dindori",
                        "betul", "hoshangabad", "narmadapuram", "harda"]),
    # Rajasthan
    (_p("Western Rajasthan (Thar)", "Western Dry Region", "Arid", 8.5, 2, "LLM", 280, (17, 32, 32, 25), "SW",
        ["Bajra", "Moong", "Cumin", "Mustard", "Chickpea", "Wheat", "Cotton"], texture="Sandy"),
     "Rajasthan", ["jaisalmer", "barmer", "bikaner", "jodhpur", "phalodi", "churu", "nagaur", "jalore", "sri ganganagar",
                   "ganganagar", "hanumangarh", "jhunjhunu", "sikar", "balotra"]),
    (_p("Hadoti (South-east Rajasthan)", "Central Plateau & Hills", "Black", 7.9, 5, "LMH", 850, (18, 31, 29, 24), "SW",
        ["Soybean", "Coriander", "Wheat", "Garlic", "Mustard", "Maize", "Chickpea"], texture="Clay Loam"),
     "Rajasthan", ["kota", "bundi", "baran", "jhalawar"]),
    # Uttar Pradesh
    (_p("Western Uttar Pradesh", "Upper Gangetic Plains", "Alluvial", 7.9, 4, "LMM", 800, (15, 30, 30, 23), "SW",
        ["Sugarcane", "Wheat", "Rice", "Potato", "Mustard", "Maize"], texture="Loam"),
     "Uttar Pradesh", ["saharanpur", "muzaffarnagar", "shamli", "meerut", "baghpat", "ghaziabad", "gautam buddha nagar",
                       "noida", "greater noida", "bulandshahr", "hapur", "bijnor", "moradabad", "rampur", "amroha",
                       "sambhal", "aligarh", "hathras", "mathura", "agra", "firozabad", "etah", "kasganj", "bareilly",
                       "budaun", "pilibhit", "shahjahanpur"]),
    (_p("Eastern Uttar Pradesh", "Middle Gangetic Plains", "Alluvial", 7.6, 5, "LMM", 1050, (16, 30, 30, 23), "SW",
        ["Rice", "Wheat", "Sugarcane", "Lentil (Masur)", "Potato", "Maize", "Mustard"], texture="Silt Loam"),
     "Uttar Pradesh", ["varanasi", "gorakhpur", "azamgarh", "ballia", "deoria", "kushinagar", "basti", "gonda",
                       "bahraich", "ayodhya", "faizabad", "sultanpur", "jaunpur", "ghazipur", "mau", "mirzapur",
                       "prayagraj", "allahabad", "pratapgarh", "siddharthnagar", "maharajganj", "sant kabir nagar",
                       "ambedkar nagar", "balrampur", "shravasti", "chandauli", "sonbhadra", "bhadohi", "kaushambi"]),
    (_p("Bundelkhand (UP)", "Central Plateau & Hills", "Red", 7.4, 4, "LLM", 850, (17, 32, 29, 24), "SW",
        ["Chickpea", "Lentil (Masur)", "Wheat", "Urad", "Sesame", "Groundnut"]),
     "Uttar Pradesh", ["jhansi", "lalitpur", "jalaun", "hamirpur", "mahoba", "banda", "chitrakoot"]),
    # Tamil Nadu
    (_p("Cauvery Delta", "East Coast Plains & Hills", "Alluvial", 7.5, 6, "LMH", 1100, (25, 31, 30, 27), "NE",
        ["Rice", "Urad", "Moong", "Cotton", "Sugarcane", "Banana"], texture="Clay Loam"),
     "Tamil Nadu", ["thanjavur", "tiruvarur", "nagapattinam", "mayiladuthurai", "kumbakonam"]),
    (_p("Western Tamil Nadu", "East Coast Plains & Hills", "Red", 7.6, 5, "LMH", 700, (24, 29, 27, 25), "NE",
        ["Maize", "Turmeric", "Banana", "Sugarcane", "Tomato", "Groundnut", "Cotton", "Onion"], texture="Sandy Loam"),
     "Tamil Nadu", ["coimbatore", "tiruppur", "erode", "salem", "namakkal", "karur", "dindigul", "dharmapuri",
                    "krishnagiri"]),
    (_p("Southern Tamil Nadu", "East Coast Plains & Hills", "Black", 7.8, 4, "LMH", 850, (26, 31, 30, 27), "NE",
        ["Rice", "Chilli", "Cotton", "Banana", "Urad", "Moong", "Maize"], texture="Clay Loam"),
     "Tamil Nadu", ["madurai", "theni", "virudhunagar", "sivaganga", "ramanathapuram", "tirunelveli", "thoothukudi",
                    "tuticorin", "tenkasi", "kanyakumari", "pudukkottai"]),
    (_p("Nilgiris hills", "Western Ghats", "Mountain", 5.2, 25, "MLM", 1900, (13, 17, 16, 15), "KER",
        ["Potato", "Cabbage", "Garlic", "Ginger"]),
     "Tamil Nadu", ["nilgiris", "the nilgiris", "ooty", "udhagamandalam", "kodaikanal"]),
    # Andhra Pradesh
    (_p("Rayalaseema", "Southern Plateau & Hills", "Red", 7.6, 4, "LMH", 650, (24, 32, 29, 26), "SW",
        ["Groundnut", "Tur (Arhar)", "Sunflower", "Tomato", "Mango", "Chickpea", "Banana", "Cotton"], texture="Sandy Loam"),
     "Andhra Pradesh", ["kurnool", "anantapur", "anantapuramu", "kadapa", "ysr", "chittoor", "nandyal",
                        "sri sathya sai", "annamayya", "tirupati"]),
    (_p("Coastal Andhra (Godavari–Krishna delta)", "East Coast Plains", "Alluvial", 7.4, 6, "LMH", 1050, (24, 30, 29, 27), "SW",
        ["Rice", "Chilli", "Maize", "Urad", "Banana", "Cotton", "Sugarcane", "Moong"],
        texture="Clay Loam"),
     "Andhra Pradesh", ["east godavari", "west godavari", "krishna", "guntur", "ntr", "eluru", "konaseema",
                        "kakinada", "bapatla", "palnadu", "vijayawada", "rajahmundry", "rajamahendravaram"]),
    # Odisha
    (_p("Coastal Odisha", "East Coast Plains", "Alluvial", 6.5, 7, "LLM", 1500, (21, 29, 29, 26), "SW",
        ["Rice", "Urad", "Moong", "Groundnut", "Brinjal", "Tomato"], texture="Clay Loam"),
     "Odisha", ["balasore", "baleswar", "bhadrak", "kendrapara", "jagatsinghpur", "cuttack", "puri", "khordha",
                "bhubaneswar", "ganjam", "jajpur", "nayagarh"]),
    # West Bengal
    (_p("Darjeeling & Kalimpong hills", "Eastern Himalaya", "Mountain", 5.0, 25, "MLM", 2800, (8, 15, 18, 13), "NEI",
        ["Ginger", "Potato", "Maize", "Cabbage"]),
     "West Bengal", ["darjeeling", "kalimpong"]),
    (_p("Red & laterite zone (West Bengal)", "Eastern Plateau & Hills", "Laterite", 5.6, 5, "LLM", 1350, (19, 30, 29, 24), "SW",
        ["Rice", "Mustard", "Tomato", "Sesame", "Potato"]),
     "West Bengal", ["purulia", "bankura", "paschim medinipur", "west midnapore", "jhargram", "birbhum", "paschim bardhaman"]),
    # Uttarakhand
    (_p("Uttarakhand Tarai & plains", "Upper Gangetic Plains", "Alluvial", 7.2, 8, "LMM", 1400, (14, 27, 29, 21), "SW",
        ["Rice", "Wheat", "Sugarcane", "Mustard", "Potato"]),
     "Uttarakhand", ["haridwar", "udham singh nagar", "rudrapur", "dehradun"]),
    # Jammu & Kashmir
    (_p("Jammu plains", "Western Himalaya", "Alluvial", 7.4, 7, "LMM", 1150, (14, 28, 29, 21), "WH",
        ["Rice", "Wheat", "Maize", "Mustard", "Bajra"]),
     "Jammu and Kashmir", ["jammu", "kathua", "samba"]),
    # Bihar
    (_p("South Bihar plains", "Middle Gangetic Plains", "Alluvial", 7.3, 5, "LMM", 1050, (17, 30, 30, 23), "SW",
        ["Rice", "Wheat", "Lentil (Masur)", "Chickpea", "Onion", "Potato"], texture="Clay Loam"),
     "Bihar", ["patna", "nalanda", "gaya", "jehanabad", "arwal", "aurangabad bihar", "rohtas", "kaimur", "buxar",
               "bhojpur", "nawada", "sheikhpura", "lakhisarai", "munger", "jamui", "banka", "bhagalpur"]),
]

# Common alternative spellings / renamed districts → canonical name used above
_ALIASES = {
    "ahilyanagar": "ahmednagar", "chhatrapati sambhajinagar": "aurangabad", "dharashiv": "osmanabad",
    "gurugram": "gurgaon", "bengaluru": "bangalore", "mysuru": "mysore", "belagavi": "belgaum",
    "kalaburagi": "gulbarga", "vijayapura": "bijapur", "ballari": "bellary", "shivamogga": "shimoga",
    "tumakuru": "tumkur", "prayagraj": "allahabad", "ayodhya": "faizabad",
}

# Normalise names like "Nashik District", "Pune Rural", "Nasik Taluka"
_NOISE = re.compile(r"\b(district|taluka|tehsil|subdistrict|sub-district|division|city|rural|urban|municipal corporation|mandal|block)\b")


def _norm(text):
    t = (text or "").lower().strip()
    t = _NOISE.sub("", t)
    t = re.sub(r"[^a-z ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _match_state(state):
    s = _norm(state)
    if not s:
        return None
    for name in STATES:
        n = _norm(name)
        if n == s or n in s or (len(s) > 4 and s in n):
            return name
    # Older / informal names
    extra = {"orissa": "Odisha", "pondicherry": "Puducherry", "nct of delhi": "Delhi", "j k": "Jammu and Kashmir",
             "daman": "Dadra and Nagar Haveli and Daman and Diu", "dadra": "Dadra and Nagar Haveli and Daman and Diu",
             "andaman": "Andaman and Nicobar Islands", "uttaranchal": "Uttarakhand"}
    for k, v in extra.items():
        if k in s:
            return v
    return None


def find_profile(location):
    """
    Best regional profile for a location dict (keys: state, district, city).
    Returns a profile dict (with 'match' = 'region' | 'state' | 'default').
    """
    location = location or {}
    state = _match_state(location.get("state", ""))
    names = [_norm(location.get(k, "")) for k in ("district", "city")]
    names = [n for n in names if n]
    names += [_ALIASES[n] for n in names if n in _ALIASES]

    for prof, reg_state, districts in REGIONS:
        if state and reg_state != state:
            continue
        for n in names:
            for d in districts:
                if n == d or (len(d) > 4 and (n.startswith(d) or d.startswith(n) and len(n) > 4)):
                    return dict(prof, match="region", state=reg_state)
    if state:
        return dict(STATES[state], match="state", state=state)
    # Not in India (or unknown) → a neutral, clearly-labelled default
    return dict(_p("Generic tropical lowland", "—", "Alluvial", 7.0, 6, "MMM", 1000, (22, 30, 28, 25), "SW",
                   ["Rice", "Maize", "Wheat", "Tomato", "Groundnut"]), match="default", state="")


def monthly_climate(profile):
    """12-month temperature (°C) and rainfall (mm) normals from a profile's seasonal values."""
    w, s, m, p = profile["temps"]
    temp = [w, w + 0.25 * (s - w), w + 0.6 * (s - w), s, s + 1.0, (s + m) / 2, m, m, m,
            (m + p) / 2, p, w + 0.3 * (p - w)]
    pattern = RAIN_PATTERNS[profile["pattern"]]
    total = sum(pattern)
    rain = [profile["rain"] * f / total for f in pattern]
    return {"temp": [round(t, 1) for t in temp], "rain": [round(r) for r in rain],
            "annual_rain": profile["rain"], "source": f"Regional normals — {profile['name']}"}


def regional_soil_data(profile):
    """Soil properties in the same shape as live SoilGrids data (so the rest of the app can use it)."""
    tex = profile["texture"]
    sand, silt, clay = {
        "Clay": (22, 30, 48), "Clay Loam": (32, 34, 34), "Loam": (42, 38, 20), "Silt Loam": (22, 58, 20),
        "Sandy Loam": (62, 24, 14), "Sandy": (85, 8, 7), "Sandy Clay Loam": (55, 18, 27),
    }.get(tex, (40, 35, 25))
    n_val = {"Low": 0.6, "Medium": 1.0, "High": 1.6}[profile["n"]]
    layer = {
        "clay": {"value": clay, "unit": "%"}, "sand": {"value": sand, "unit": "%"}, "silt": {"value": silt, "unit": "%"},
        "ph": {"value": profile["ph"], "unit": ""}, "organic_carbon": {"value": profile["oc"], "unit": "g/kg"},
        "nitrogen": {"value": n_val, "unit": "g/kg"},
    }
    data = {"topsoil": dict(layer), "subsoil": dict(layer), "source": "regional",
            "region": profile["name"], "soil_group": profile["soil_group"]}
    data.update(layer)
    return data
