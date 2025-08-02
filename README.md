# 🥦 nutriVision — Smart Food Detection & Recipe Recommendation

**nutriVision** is an intelligent Streamlit web app that transforms a photo of your fridge into a list of ingredients, their nutritional information, and delicious Indian recipe suggestions. Using a custom-trained **YOLOv5** model, it bridges the gap between what you have and what you can create.

---

## 📸 Demo

| Main Page | Ingredient Detection | Recipe Recommendations | Nutritional Analysis |
| :---: | :---: | :---: | :---: |
| ![Main Page](https://github.com/user-attachments/assets/68a25f6a-f7b0-4bec-8ff1-c68107385e5f) | ![Ingredient Detection](https://github.com/user-attachments/assets/b61a9e3e-da78-40e0-a5a4-8073a5396f56) | ![Recipe Recommendations](https://github.com/user-attachments/assets/486fd131-a137-4190-9232-0eff2fcb3a74) | ![Nutritional Analysis](https://github.com/user-attachments/assets/dc6be91e-f258-428f-88c8-bde1a9134993) |

---

## ✨ Features

-   **📤 Multi-Image Upload**: Analyze one or more photos of your ingredients at once.
-   **🤖 Smart Detection**: Uses a fine-tuned **YOLOv5** model to accurately identify vegetables and other food items.
-   **🍲 Intelligent Recipes**: Recommends healthy Indian recipes based on the combined list of detected ingredients.
-   **📊 Nutritional Insights**: Displays key nutritional values (Calories, Protein, Carbs, Fats) for each recommended dish.
-   **🎨 Clean UI**: Built with a modern, pastel-themed interface and compact cards for easy Browse.

---

## 🛠️ Tech Stack

| Component | Stack & Libraries |
| :--- | :--- |
| **Web Framework** | Streamlit |
| **Object Detection** | YOLOv5, PyTorch, OpenCV |
| **Data Handling** | Python, NumPy, Pandas, YAML |
| **Training Platform** | Google Colab with a custom annotated dataset |

---

## 🧠 Model Training

The object detection model was custom-trained on a dataset of common kitchen and fridge ingredients.

-   **Classes Trained**: `avocado`, `cabbage`, `onion`, `tomato`, `potato`, `garlic`, `peas`, `eggplant`, `carrot`, etc.
-   **Training Notebook**: Available at `https://github.com/Greeshma-10/nutriVision`
-   **Inference Model**: `nutrition_best.pt` or `nutrition_best_windows.pt`

---

## 🚀 Getting Started (Run the Web App)

Follow these steps to run the Streamlit application on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Greeshma-10/nutriVision.git](https://github.com/Greeshma-10/nutriVision.git)
    cd nutriVision
    ```
2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
The application will open in your web browser.

<br>

<details>
  <summary>💻 For Developers: Running Detection via Command Line</summary>
  
  If you want to test the YOLOv5 model directly without the Streamlit interface, you can run the `detect.py` script.

  1.  **Navigate to the YOLOv5 directory:**
      ```bash
      cd yolov5
      ```
  2.  **Install its specific dependencies:**
      ```bash
      pip install -r requirements.txt
      ```
  3.  **Run detection from the root directory:**
      Make sure you are in the main `nutriVision/` folder.
      ```bash
      python yolov5/detect.py --weights nutrition_best.pt --conf 0.4 --source test_images/your_image.jpg
      ```
      - `--weights`: Path to your custom model.
      - `--conf`: The confidence threshold for detection.
      - `--source`: The directory or image to analyze.
  
  Annotated images will be saved in the `yolov5/runs/detect/` directory.

</details>

---

## 📁 Project Structure
```text
nutriVision/
│
├── yolov5/              # Cloned YOLOv5 repository
├── test_images/         # Sample images for testing
│
├── app.py               # The main Streamlit application script
├── nutrition_best.pt    # Trained model weights
├── requirements.txt     # Python packages for the Streamlit app
└── README.md            # This file
```
