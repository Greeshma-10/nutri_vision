# 🥦 nutriVision — Smart Food Detection & Recipe Recommendation

**nutriVision** is a Streamlit-powered web app that uses a custom-trained YOLOv5 model to detect vegetables from fridge or kitchen images and recommends healthy Indian recipes based on available ingredients. It also displays the nutritional content of each dish to help users make informed choices.


## 🚀 Features

- Upload one or more fridge/food images
- YOLOv5-based real-time food item detection
- Smart recipe recommendations based on matched ingredients
- Displays nutritional values (Calories, Protein, Carbs, Fats)
- Handles multiple images and suggests combined recipes
- Clean pastel UI with compact cards for dish details


## 🧠 Model Training

The custom object detection model was trained using **YOLOv5** with annotated fridge/kitchen images.

📍 Model training notebook: 

- Classes trained: avocado, cabbage, onion, tomato, potato, garlic, peas, eggplant, carrot, etc.
- Training platform: Google Colab with custom dataset
- Inference model: `nutrition_best_windows.pt`


## 🔧 Tech Stack

| Component     | Stack                            |
|--------------|-----------------------------------|
| Frontend     | Streamlit, HTML, CSS              |
| Backend      | Python, Torch, NumPy              |
| Detection    | YOLOv5 with DetectMultiBackend    |
| Data Format  | Custom annotated dataset (YOLO format) |
| Hosting      | Run locally using `streamlit run app.py` |


## 🛠️ Installation & Run Locally

```bash
git clone https://github.com/Greeshma-10/nutriVision
cd nutriVision
pip install -r requirements.txt
streamlit run app.py
