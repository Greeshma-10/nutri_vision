import os
import io
import json
import time
import requests
import traceback
from typing import List, Set, Dict
from PIL import Image
from dotenv import load_dotenv
import streamlit as st
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from difflib import get_close_matches

# ----------------- Load .env and API key -----------------
load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")

# ----------------- Color Scheme (greens removed) -----------------
CANVAS_BG = "#0f1720"      # deep charcoal background for main canvas
CARD_BG = "#0b1412"        # slightly lighter than canvas for cards
TEXT_ON_DARK = "#E8F3EC"   # soft off-white for text on dark

# ----------------- App config -----------------
st.set_page_config(page_title="Smart Recipe Recommender", layout="wide")

# ---------- Custom CSS: dark unified scheme ----------
st.markdown(
    f"""
    <style>
    :root {{
        --canvas: {CANVAS_BG};
        --card: {CARD_BG};
        --muted-text: {TEXT_ON_DARK};
    }}

    /* main page background and text */
    .stApp, .block-container {{
        background-color: var(--canvas) !important;
        color: var(--muted-text) !important;
    }}

    /* page header (top bar) - unified with canvas, subtle border */
    header[data-testid="stHeader"] {{
        background: var(--canvas) !important;
        color: var(--muted-text) !important;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        box-shadow: none;
    }}

    /* sidebar styling - match canvas with slight separation */
    section[data-testid="stSidebar"] {{
        background: var(--canvas) !important;
        color: var(--muted-text) !important;
        padding-top: 1.2rem;
        border-left: 1px solid rgba(255,255,255,0.1);
        box-shadow: -2px 0 6px rgba(0,0,0,0.3);
    }}
    section[data-testid="stSidebar"] .css-1d391kg {{
        color: var(--muted-text) !important;
    }}

    /* rest of your CSS stays the same... */
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- App Header -----------------
st.title("🍽️ Smart Recipe Recommender")
st.subheader("Upload your food images and get recipe suggestions based on detected ingredients")
st.markdown("---")  # optional horizontal line separator

# ----------------- Utility: resilient requests session -----------------
def requests_session_with_retries(total_retries=3, backoff_factor=0.3, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist, allowed_methods=["GET", "POST"])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))
    return session

http = requests_session_with_retries()

# ----------------- Ingredient normalization & synonyms -----------------
SYNONYMS = {
    "tomato": "tomatoes",
    "tomatoes": "tomatoes",
    "potato": "potatoes",
    "potatoes": "potatoes",
    "onion": "onions",
    "chili": "chili pepper",
    "green chili": "chili pepper",
    "capsicum": "bell pepper",
    "bell pepper": "bell pepper",
    "eggplant": "eggplant",
    "brinjal": "eggplant",
    "coriander": "cilantro",
    "cilantro": "cilantro",
    "garlic clove": "garlic",
    "garlic": "garlic",
    "ginger root": "ginger",
    "ginger": "ginger",
}

try:
    from rapidfuzz import process as rf_process
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

if 'normalize_cache' not in st.session_state:
    st.session_state.normalize_cache = {}

def normalize_ingredient(ing: str) -> str:
    if not ing:
        return ing
    key = ing.strip().lower()
    if key in st.session_state.normalize_cache:
        return st.session_state.normalize_cache[key]
    if key in SYNONYMS:
        st.session_state.normalize_cache[key] = SYNONYMS[key]
        return SYNONYMS[key]
    tokens = key.replace('-', ' ').split()
    candidates = [" ".join(tokens[:i+1]) for i in range(len(tokens))]
    for c in candidates[::-1]:
        if c in SYNONYMS:
            st.session_state.normalize_cache[key] = SYNONYMS[c]
            return SYNONYMS[c]
    choices = list(SYNONYMS.keys())
    if RAPIDFUZZ_AVAILABLE:
        match = rf_process.extractOne(key, choices, score_cutoff=70)
        if match:
            mapped = SYNONYMS[match[0]]
            st.session_state.normalize_cache[key] = mapped
            return mapped
    else:
        close = get_close_matches(key, choices, n=1, cutoff=0.7)
        if close:
            mapped = SYNONYMS[close[0]]
            st.session_state.normalize_cache[key] = mapped
            return mapped
    st.session_state.normalize_cache[key] = key
    return key

def normalize_ingredients(ings: List[str]) -> List[str]:
    return sorted(list({normalize_ingredient(i) for i in ings if i}))

# ----------------- API helpers (cached for performance) -----------------
@st.cache_data(ttl=60*60)
def get_recipes_from_spoonacular(ingredients: List[str], number: int = 5):
    if not SPOONACULAR_API_KEY:
        return {"error": "missing_api_key"}
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "apiKey": SPOONACULAR_API_KEY,
        "ingredients": ",".join(ingredients),
        "number": number,
        "ranking": 1,
        "ignorePantry": True
    }
    try:
        resp = http.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"status_{resp.status_code}", "message": resp.text}
    except Exception as e:
        return {"error": "exception", "message": str(e)}

@st.cache_data(ttl=60*60)
def get_recipe_info_spoonacular(recipe_id: int):
    if not SPOONACULAR_API_KEY:
        return {"error": "missing_api_key"}
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information"
    params = {"apiKey": SPOONACULAR_API_KEY, "includeNutrition": True}
    try:
        resp = http.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"error": f"status_{resp.status_code}", "message": resp.text}
    except Exception as e:
        return {"error": "exception", "message": str(e)}

# ----------------- Model loading with robust error handling -----------------
@st.cache_resource
def load_yolov5_model(path: str = 'nutrition_best_windows.pt'):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=False)
        return model
    except Exception as e:
        st.error("Failed to load YOLOv5 model. Check that:\n1) torch & deps installed,\n2) the path is correct,\n3) you have internet for torch.hub (first run).\n\nError: " + str(e))
        st.write(traceback.format_exc())
        return None

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ----------------- Sidebar (settings) -----------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence = st.slider("Detection confidence threshold", 0.0, 1.0, 0.35, 0.05)
    top_k = st.slider("Number of recipes to fetch", 1, 20, 8)
    show_full_nutrition = st.checkbox("Show full nutrition details", value=False)
    batch_infer = st.checkbox("Load model now (may take time)", value=False)

    if batch_infer and not st.session_state.model_loaded:
        with st.spinner("Loading YOLOv5 model (this may take ~20-40s first run)..."):
            model_attempt = load_yolov5_model()
            if model_attempt is not None:
                st.session_state.model_loaded = True
                st.success("Model loaded and ready.")

uploaded_files = st.file_uploader("📸 Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Local fallback recipes (simple examples)
FALLBACK_RECIPES = [
    {"title": "Tomato Omelette", "usedIngredients": ["tomatoes","eggs"], "missedIngredients": ["salt","pepper"], "id": 999001},
    {"title": "Spicy Potato Stir-fry", "usedIngredients": ["potatoes","chili pepper"], "missedIngredients": ["oil","salt"], "id": 999002}
]

# ----------------- Main processing -----------------
if uploaded_files:
    if not st.session_state.model_loaded:
        with st.spinner("Loading YOLOv5 model..."):
            model = load_yolov5_model()
            if model:
                st.session_state.model_loaded = True
            else:
                st.warning("Model not available — detections will be skipped. You can still edit ingredients manually.")
    else:
        model = load_yolov5_model()

    combined_detected: Set[str] = set()

    st.subheader("Uploaded images & detections")
    for uploaded_file in uploaded_files:
        try:
            with st.expander(f"📷 {uploaded_file.name}"):
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption=uploaded_file.name, use_container_width=True)

                if st.session_state.model_loaded and model is not None:
                    with st.spinner(f"Detecting ingredients in {uploaded_file.name}..."):
                        results = model(image)
                        detections = results.pandas().xyxy[0]
                        items = detections[detections['confidence'] >= confidence]['name'].tolist()
                        items = list(dict.fromkeys(items))  # preserve order, remove duplicates
                        if items:
                            st.success("Detected: " + ", ".join(items))
                            combined_detected.update(items)
                        else:
                            st.info("No confident detections in this image.")
                else:
                    st.info("Model not loaded — skipping automatic detection for this image.")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            st.write(traceback.format_exc())

    # Ingredients & recipes section (row-wise below images)
    st.subheader("🥘 Ingredients & Recipes")
    st.markdown("You can edit detected ingredients before fetching recipes (one per line).")

    combined_list = sorted(list(combined_detected))
    detected_text = "\n".join(combined_list)

    with st.expander("👀 Click to view detected ingredients"):
        st.text_area("Detected ingredients (editable)", value=detected_text, height=200, key="detected_area")

    manual_input = st.session_state.get("detected_area", detected_text)
    manual_ings = [l.strip() for l in manual_input.splitlines() if l.strip()]

    normalized = normalize_ingredients(manual_ings)
    st.markdown("**Normalized ingredients (for recipe matching):**")
    st.write(", ".join(normalized) if normalized else "(none)")

    if st.button("🔎 Fetch recipes"):
        if not normalized:
            st.warning("No ingredients provided.")
        else:
            with st.spinner("Querying Spoonacular..."):
                recipes = get_recipes_from_spoonacular(normalized, number=top_k)
            
            if isinstance(recipes, dict) and recipes.get('error'):
                st.error("Spoonacular API error: " + recipes.get('error') + (f" — {recipes.get('message')}" if recipes.get('message') else ""))
                st.info("Using fallback local recipes.")
                recipes = FALLBACK_RECIPES

            filtered = []
            for r in recipes:
                used = []
                if isinstance(r.get('usedIngredients', []), list) and r['usedIngredients'] and isinstance(r['usedIngredients'][0], dict):
                    used = [u.get('name', '').lower() for u in r['usedIngredients']]
                else:
                    used = [u.lower() for u in r.get('usedIngredients', [])]
                norm_used = set(normalize_ingredients(used))
                if norm_used.intersection(set(normalized)):
                    filtered.append(r)

            if not filtered:
                st.warning("No strong matches found — showing all returned recipes.")
                filtered = recipes

            for r in filtered:
                st.markdown("---")
                st.markdown(f"<div class='recipe-card'><h3>🍴 {r.get('title', 'Unknown')}</h3></div>", unsafe_allow_html=True)
                if r.get('image'):
                    st.image(r.get('image'), use_column_width=False)

                used = []
                missed = []
                if isinstance(r.get('usedIngredients', []), list) and r['usedIngredients'] and isinstance(r['usedIngredients'][0], dict):
                    used = [u.get('name', '') for u in r['usedIngredients']]
                else:
                    used = r.get('usedIngredients', [])
                if isinstance(r.get('missedIngredients', []), list) and r['missedIngredients'] and isinstance(r['missedIngredients'][0], dict):
                    missed = [m.get('name', '') for m in r['missedIngredients']]
                else:
                    missed = r.get('missedIngredients', [])

                if used:
                    st.success("✅ Matched: " + ", ".join(used))
                if missed:
                    st.warning("❌ Missing: " + ", ".join(missed))

                if r.get('id') and isinstance(r.get('id'), int) and r.get('id') < 900000:
                    info = get_recipe_info_spoonacular(r['id'])
                    if info and not info.get('error') and info.get('nutrition'):
                        nutr = info['nutrition']
                        with st.expander("🥗 Nutrition (click to expand)"):
                            if show_full_nutrition:
                                st.markdown("**Nutrition (full):**")
                                for n in nutr.get('nutrients', []):
                                    st.write(f"- {n['name']}: {n['amount']} {n['unit']}")
                            else:
                                st.markdown("**Top nutrients:**")
                                for n in nutr.get('nutrients', [])[:5]:
                                    st.write(f"- {n['name']}: {n['amount']} {n['unit']}")
                    else:
                        st.info("Nutrition details not available for this recipe.")
                else:
                    st.info("(Fallback recipe — no nutrition available)")

            st.success("✅ Done — recipes displayed above.")
else:
    st.info("📥 Upload images of food items to start. You can also type ingredients manually in the sidebar.")

# Manual ingredient entry
st.markdown("---")
st.subheader("✍️ Manual Input")
manual = st.text_input("Enter ingredients separated by commas (e.g. tomato, onion, egg)")
if manual:
    manual_list = [m.strip() for m in manual.split(',') if m.strip()]
    st.write("Normalized:", normalize_ingredients(manual_list))
