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

# Ingredient normalization
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

# Fetch recipes by ingredients
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

# Fetch full recipe info with nutrition
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
    # Using torch.hub to load custom trained YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='nutrition_best_windows.pt', force_reload=True)
    return model

model = load_model()

# ----------------- Streamlit UI -----------------
st.title("🥗 Smart Recipe Recommender")
st.write("Upload an image of food items and get recipe suggestions with nutrition info!")

uploaded_file = st.file_uploader("📸 Upload a fridge/ingredient photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ----------------- Run YOLOv5 Detection -----------------
    with st.spinner("🔍 Detecting ingredients..."):
        results = model(image)  # pass PIL image directly
        detections = results.pandas().xyxy[0]  # pandas DataFrame
        detected_items = detections[detections['confidence'] > 0.3]['name'].tolist()
        detected_items = list(set(detected_items))  # remove duplicates

    if not detected_items:
        st.warning("⚠️ No ingredients detected. Please upload a clearer image.")
    else:
        st.markdown("### ✅ Detected Ingredients:")
        st.write(", ".join(detected_items))

        # Normalize for Spoonacular API
        normalized_detected = normalize_ingredients(detected_items)

        # ----------------- Fetch Recipes -----------------
        recipes = get_recipes(normalized_detected, number=10)

        if recipes:
            st.markdown("---")
            st.subheader("🍽️ Recommended Recipes")
            for recipe in recipes:
                title = recipe.get("title", "Unknown Dish")
                image_url = recipe.get("image", "")

                # Ingredients info
                used = [ing.get("name", "").lower() for ing in recipe.get("usedIngredients", [])]
                missed = [ing.get("name", "").lower() for ing in recipe.get("missedIngredients", [])]
                normalized_used = set(normalize_ingredients(used))
                normalized_missed = set(normalize_ingredients(missed))

                # Only show if detected ingredients are part of the recipe
                if not set(normalized_detected).intersection(normalized_used):
                    continue

                st.markdown(f"### 🍴 {title}")
                if image_url:
                    st.image(image_url, use_container_width=True)

                if used:
                    st.success("✅ Matched ingredients: " + ", ".join(used))
                if missed:
                    st.warning("❌ Missing ingredients: " + ", ".join(missed))

                # Nutrition info
                info = get_recipe_info(recipe["id"])
                if info.get("nutrition") and info["nutrition"].get("nutrients"):
                    st.markdown("**🥗 Nutrition Info (Top 5):**")
                    for n in info["nutrition"]["nutrients"][:5]:
                        st.write(f"- {n['name']}: {n['amount']} {n['unit']}")

                st.markdown("---")
        else:
            st.warning("⚠️ No recipes found. Try with more common ingredients.")
