import os
import streamlit as st
import requests
from PIL import Image
from dotenv import load_dotenv
import torch

# ----------------- Load .env and API key -----------------
load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")
if not SPOONACULAR_API_KEY:
    st.error("⚠️ Spoonacular API key not found. Please check your .env file.")

# ----------------- Helper Functions -----------------
def normalize_ingredients(ingredients):
    mapping = {
        "tomato": "tomatoes",
        "potato": "potatoes",
        "onion": "onions",
        "chili": "chili pepper",
        "capsicum": "bell pepper",
        "eggplant": "eggplant",
        "brinjal": "eggplant"
    }
    return [mapping.get(ing.lower(), ing.lower()) for ing in ingredients]

def get_recipes(ingredients, number=5):
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "apiKey": SPOONACULAR_API_KEY,
        "ingredients": ",".join(ingredients),
        "number": number,
        "ranking": 1,
        "ignorePantry": True
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"⚠️ Error {response.status_code}: {response.text}")
        return []

def get_recipe_info(recipe_id):
    url = f"https://api.spoonacular.com/recipes/{recipe_id}/information"
    params = {"apiKey": SPOONACULAR_API_KEY, "includeNutrition": True}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return {}

# ----------------- Load YOLOv5 Model -----------------
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='nutrition_best_windows.pt', force_reload=True)
    return model

model = load_model()

# ----------------- Streamlit UI -----------------
st.title("🥗 Smart Recipe Recommender")
st.write("Upload one or more images of food items to get recipe suggestions with nutrition info!")

uploaded_files = st.file_uploader(
    "📸 Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    combined_detected = set()

    for uploaded_file in uploaded_files:
        st.markdown(f"---\n### 📷 {uploaded_file.name}")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # YOLOv5 detection
        with st.spinner(f"🔍 Detecting ingredients in {uploaded_file.name}..."):
            results = model(image)
            detections = results.pandas().xyxy[0]
            detected_items = detections[detections['confidence'] > 0.3]['name'].tolist()
            detected_items = list(set(detected_items))  # remove duplicates

        if detected_items:
            st.success(f"✅ Detected in this image: {', '.join(detected_items)}")
            combined_detected.update(detected_items)
        else:
            st.warning(f"⚠️ No ingredients detected in {uploaded_file.name}.")

    if combined_detected:
        st.markdown("---")
        st.header("🍛 Combined Detected Ingredients Across All Images")
        st.write(", ".join(sorted(combined_detected)))

        # Normalize ingredients
        normalized_detected = normalize_ingredients(combined_detected)

        # Fetch recipes
        recipes = get_recipes(normalized_detected, number=10)

        if recipes:
            st.markdown("---")
            st.subheader("🍽️ Recommended Recipes")
            for recipe in recipes:
                title = recipe.get("title", "Unknown Dish")
                image_url = recipe.get("image", "")
                used = [ing.get("name", "").lower() for ing in recipe.get("usedIngredients", [])]
                missed = [ing.get("name", "").lower() for ing in recipe.get("missedIngredients", [])]
                normalized_used = set(normalize_ingredients(used))
                normalized_missed = set(normalize_ingredients(missed))

                if not set(normalized_detected).intersection(normalized_used):
                    continue

                st.markdown(f"### 🍴 {title}")
                if image_url:
                    st.image(image_url, use_container_width=True)
                if used:
                    st.success("✅ Matched ingredients: " + ", ".join(used))
                if missed:
                    st.warning("❌ Missing ingredients: " + ", ".join(missed))

                info = get_recipe_info(recipe["id"])
                if info.get("nutrition") and info["nutrition"].get("nutrients"):
                    st.markdown("**🥗 Nutrition Info (Top 5):**")
                    for n in info["nutrition"]["nutrients"][:5]:
                        st.write(f"- {n['name']}: {n['amount']} {n['unit']}")
                st.markdown("---")
        else:
            st.warning("⚠️ No recipes found. Try with more common ingredients.")
    else:
        st.warning("⚠️ No ingredients detected in any uploaded images.")
