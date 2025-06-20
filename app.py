import streamlit as st

st.markdown("""
    <style>
    .main .block-container {
        max-width: 100% !important;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

import torch
from PIL import Image
import numpy as np
import sys
import os

# Set path to yolov5
sys.path.append('./yolov5')

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device


# ✅ Ingredient-based recipe suggestions
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


# ✅ Recommend dishes if all required ingredients are present
def recommend_dishes(detected_items, recipe_map):
    detected_set = set(detected_items)
    matches = []

    for dish, ingredients in recipe_map.items():
        if ingredients & detected_set:  # At least one match
            matched = ingredients & detected_set
            missing = ingredients - detected_set
            matches.append((dish, matched, missing))
    return matches


# Load YOLOv5 model
@st.cache_resource
def load_model():
    device = select_device('cpu')  # or 'cuda' if GPU available
    model = DetectMultiBackend('nutrition_best_windows.pt', device=device, dnn=False)
    return model, device

model, device = load_model()

st.title("🥦 nutriVision — YOLOv5 Food Detector")

uploaded_files = st.file_uploader("📷 Upload one or more food/fridge images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    combined_detected = set()  # To hold all ingredients from all images

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
            st.write(", ".join(detected))
            combined_detected.update(detected)  # ⬅️ Add to overall set
        else:
            st.warning("😕 No items detected in this image.")

    # 🧠 Recipe suggestion using all detected ingredients
    if combined_detected:
        st.markdown("---")
        st.header("🍛 Combined Recommendation")
        st.write("Detected across all images:", ", ".join(combined_detected))

        recipes = recommend_dishes(combined_detected, recipe_map)
        if recipes:
            st.markdown("---")
            st.subheader("🌟 Final Recipe Suggestions")

            for dish, matched, missing in recipes:
                st.markdown(f"""
                <div style="border: 1px solid #e4d0ff; border-radius: 10px; padding: 1rem; background-color: #fdf6ff; margin-bottom: 1rem;">
            <h4 style="color:#a23dcf;">🍽️ <b>{dish}</b></h4>
            <p>✅ <b>Matched:</b> <code>{', '.join(sorted(matched))}</code></p>
            {"<p>❌ <b>Missing:</b> <code>" + ', '.join(sorted(missing)) + "</code></p>" if missing else ""}
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info("No recipe suggestions based on current detections.")
