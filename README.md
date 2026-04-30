# 🥦 nutriVision — Smart Food Detection & Recipe Recommendation

**nutriVision** is an intelligent Streamlit web app that transforms a photo of your fridge into a list of ingredients, their nutritional information, and delicious Indian recipe suggestions. Using a custom-trained **YOLOv5** model, it bridges the gap between what you have and what you can create.

---

## 📸 Demo

## Main page 
![Main Page](https://github.com/user-attachments/assets/6fa1b0d8-eef1-4e6d-99fd-a78ee9964783)
## Ingredient Detection 
![Ingredient Detection](https://github.com/user-attachments/assets/93f0af19-1129-4727-a8f7-c13db0673e3d)
## Recipe Recommendation
![Recipe Recommendations](https://github.com/user-attachments/assets/cdf712ad-4e99-4102-acbf-5a1b3ef7e9b1) 
## Nutritional Analysis 
![Nutritional Analysis](https://github.com/user-attachments/assets/34dbb8c0-4d30-40c1-9aa8-855876da27c0) 
## Youtube videos 
![Youtube videos](https://github.com/user-attachments/assets/bacbc8d9-ed70-4fb8-ae49-05eb9e29ff90) |

---

## ✨ Features  

- 📤 **Multi-Image Upload**: Analyze one or more photos of your ingredients at once.  
- 🤖 **Smart Detection**: Uses a fine-tuned YOLOv5 model to accurately identify vegetables and other food items.  
- 🍲 **Intelligent Recipes**: Integrated with **Spoonacular API** to recommend healthy Indian & global recipes based on detected ingredients.  
- 📊 **Nutritional Insights**: Displays key nutritional values (Calories, Protein, Carbs, Fats) for each recommended dish.  
- 🎥 **YouTube Video Integration**: Fetches related cooking videos via **YouTube Data API**, making it easy to follow step-by-step instructions.  
- 🎨 **Clean UI**: Built with a modern, pastel-themed interface and compact cards for easy browsing.  

---

## 🛠️ Tech Stack  

| Component         | Stack & Libraries |  
|-------------------|-------------------|  
| Web Framework     | Streamlit         |  
| Object Detection  | YOLOv5, PyTorch, OpenCV |  
| Recipe API        | Spoonacular API   |  
| Video Integration | YouTube Data API v3 |  
| Data Handling     | Python, NumPy, Pandas, YAML |  
| Training Platform | Google Colab with a custom annotated dataset |  

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
