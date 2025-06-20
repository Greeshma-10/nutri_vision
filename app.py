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
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load YOLOv5 model
@st.cache_resource
def load_model():
    device = select_device('cpu')  # or 'cuda' if GPU available
    model = DetectMultiBackend('nutrition_best_windows.pt', device=device, dnn=False)
    return model, device

model, device = load_model()

st.title("🥦 nutriVision — YOLOv5 Food Detector")

uploaded_file = st.file_uploader("📷 Upload an image of fridge or food items", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to BGR format for YOLO
    img = np.array(image)  # ✅ Keep it RGB if trained on RGB
    img = letterbox(img, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)


    with st.spinner("🔍 Detecting items..."):
        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)


        names = model.names
        detected = set()
        print("Raw predictions:", pred)
        print("Model classes:", model.names)

        for det in pred:
            if len(det):
                for *xyxy, conf, cls in det:
                    detected.add(names[int(cls)])

    if detected:
        st.success("✅ Detected items:")
        st.write(", ".join(detected))
    else:
        st.warning("😕 No items detected. Try a clearer image or lower threshold.")
