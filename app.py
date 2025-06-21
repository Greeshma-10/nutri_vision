import streamlit as st
import torch
from PIL import Image
import numpy as np
import sys
import os

# Set path to yolov5
sys.path.append('./yolov5')

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# 🔖 Recipe Ingredients Map
recipe_map = {
    "Tomato Onion Curry": {"tomato", "onion"},
    "Vegetable Pulao": {"peas", "carrot", "potato"},
    "Baingan Bharta": {"eggplant", "onion", "garlic"},
    "Cabbage Stir Fry": {"cabbage", "garlic"},
    "Pumpkin Soup": {"pumpkin", "onion"},
    "Mixed Veg Sabzi": {"tomato", "potato", "carrot", "peas", "onion"},
    "Avocado Salad": {"avocado", "cucumber", "corn"},
    "Broccoli Stir Fry": {"broccoli", "garlic"},
    "Peas Curry": {"peas", "onion", "tomato"},
}

# 🥗 Nutritional Info for Each Dish
recipe_nutrition = {
    "Tomato Onion Curry": {"Calories": 120, "Protein": "2g", "Carbs": "12g", "Fats": "6g"},
    "Vegetable Pulao": {"Calories": 250, "Protein": "5g", "Carbs": "35g", "Fats": "8g"},
    "Baingan Bharta": {"Calories": 150, "Protein": "3g", "Carbs": "14g", "Fats": "7g"},
    "Cabbage Stir Fry": {"Calories": 110, "Protein": "2g", "Carbs": "10g", "Fats": "5g"},
    "Pumpkin Soup": {"Calories": 100, "Protein": "2g", "Carbs": "15g", "Fats": "3g"},
    "Mixed Veg Sabzi": {"Calories": 180, "Protein": "4g", "Carbs": "20g", "Fats": "7g"},
    "Avocado Salad": {"Calories": 200, "Protein": "3g", "Carbs": "10g", "Fats": "16g"},
    "Broccoli Stir Fry": {"Calories": 130, "Protein": "4g", "Carbs": "9g", "Fats": "6g"},
    "Peas Curry": {"Calories": 160, "Protein": "6g", "Carbs": "18g", "Fats": "5g"},
}

# 🍽️ Recommend dishes from detected ingredients
def recommend_dishes(detected_items, recipe_map):
    detected_set = set(item.strip().lower() for item in detected_items)
    matches = []
    for dish, ingredients in recipe_map.items():
        if ingredients & detected_set:
            matched = ingredients & detected_set
            missing = ingredients - detected_set
            matches.append((dish, matched, missing))
    return matches

# 📦 Load YOLOv5 model
@st.cache_resource
def load_model():
    device = select_device('cpu')  # or 'cuda'
    model = DetectMultiBackend('nutrition_best_windows.pt', device=device, dnn=False)
    return model, device

# 🔁 Main App Logic
model, device = load_model()

st.title("🥦 nutriVision — YOLOv5 Food Detector")

uploaded_files = st.file_uploader("📷 Upload one or more food/fridge images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    combined_detected = set()

    for uploaded_file in uploaded_files:
        st.markdown(f"---\n### 📸 {uploaded_file.name}")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = np.array(image)
        img = letterbox(img, 640, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with st.spinner("🔍 Detecting items..."):
            pred = model(img_tensor, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

            names = model.names
            detected = set()
            for det in pred:
                if len(det):
                    for *xyxy, conf, cls in det:
                        detected.add(names[int(cls)])

        if detected:
            st.success("✅ Items detected in this image:")
            st.write(", ".join(sorted(detected)))
            combined_detected.update(detected)
        else:
            st.warning("😕 No items detected in this image.")

    # 🧠 Final Suggestions
    if combined_detected:
        st.markdown("---")
        st.header("🍛 Combined Recommendation")
        st.write("Detected across all images:", ", ".join(sorted(combined_detected)))

        recipes = recommend_dishes(combined_detected, recipe_map)

        if recipes:
            st.markdown("---")
            st.subheader("🌟 Final Recipe Suggestions")

            for dish, matched, missing in recipes:
                nutrition = recipe_nutrition.get(dish, {})
                nutrient_html = "".join(
                    f"<li><b>{key}:</b> {value}</li>" for key, value in nutrition.items()
                ) if nutrition else "<li>No data available</li>"

                missing_html = (
                    f"<p>❌ <b>Missing:</b> <span style='color:#a33;'>{', '.join(sorted(missing))}</span></p>"
                    if len(missing) > 0 else ""
                )

                st.markdown(f"""
                    <div style="border: 1px solid #e4d0ff; border-radius: 10px; padding: 1rem; background-color: #fdf6ff; margin-bottom: 1rem; color:#5b0066;">
                        <h4 style="color:#a23dcf;">🍽️ <b>{dish}</b></h4>
                        <p>✅ <b style="color:black;">Matched:</b> <span style="color:#2c7a7b;">{', '.join(sorted(matched))}</span></p>
                        {missing_html}
                        <h5 style="margin-top:10px;">🥗 Nutritional Info:</h5>
                        <ul style="line-height: 1.5;">{nutrient_html}</ul>
                    </div>
                """, unsafe_allow_html=True)

        else:
            st.info("No recipe suggestions based on current detections.")
