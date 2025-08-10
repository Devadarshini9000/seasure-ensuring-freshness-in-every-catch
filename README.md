# 🐟 seasure: ensuring freshness in every catch (Fish Freshness Detection System) 

A **deep learning and computer vision application** that predicts fish freshness by analyzing eye and gill images, combined with temperature estimation, to deliver a confidence score, freshness score and level.
Built with **TensorFlow/Keras** for model training and **Streamlit** for an interactive web interface.

## 🚀 Features

- 🧠 CNN-powered classification into 4 categories:
Eye-Fresh
Eye-Non-Fresh
Gill-Fresh
Gill-Non-Fresh
- 🌡 Temperature estimation based on classification
- 📊 Freshness score (0–100) with High, Medium, Low indicators
- 🔄 Data augmentation for robust performance
- 📷 Real-time predictions via Streamlit
- 🎯 Accuracy above 91% on validation data

## 📁 Project Structure

- ├── fish_train.py          # Basic CNN training script
- ├── temp_train.py          # CNN training with temperature metadata & callbacks
- ├── fish_test.py           # Streamlit app for basic image-based prediction
- ├── temp_test.py           # Streamlit app with temperature & freshness scoring
- ├── prep.py                # Dataset preprocessing & augmentation setup
- ├── temp.py                # Metadata generator with temperature assignment
- ├── fish_freshness_model.h5 # Trained CNN model (generated after training)
- ├── training_history.png   # Accuracy/Loss curves (generated after training)
- └── README.md              # Project documentation

## 🛠 Tech Stack

- Backend / Model Training: Python, TensorFlow, Keras
- Data Handling: Pandas, NumPy
- Image Processing: Keras ImageDataGenerator
- Visualization: Matplotlib
- UI: Streamlit

## 🧪 How It Works

1) Dataset Preparation: 
Dataset sourced from Roboflow.
Four classes: eye-fresh, eye-non-fresh, gill-fresh, gill-non-fresh.

2) Temperature metadata assigned:
Fresh: 0°C – 4°C
Non-fresh: 5°C – 10°C
(temp.py script creates fish_freshness_with_temperature.csv)

3) Preprocessing & Augmentation (prep.py):
Image resizing: 128×128 px
Normalization: pixel values scaled to [0,1]
Augmentations: rotation, shift, shear, zoom, horizontal flip

4) Model Training:
CNN architecture: 3× Conv2D + MaxPooling2D, Flatten, Dense layers
Optimizer: Adam (LR=0.001), Loss: categorical cross entropy
Early stopping & model checkpointing (temp_train.py)
Final model saved as fish_freshness_model.h5

5) Prediction & UI:
Users upload an image via Streamlit (fish_test.py or temp_test.py)
Model predicts freshness category & confidence
temp_test.py also:
Estimates storage temperature
Calculates freshness score (0–100)
Displays freshness level: High, Medium, Low (color-coded)

## 💻 How to Run Locally

### 1️⃣ Clone the Repository
git clone https://github.com/yourusername/fish-freshness-detection.git
cd fish-freshness-detection

### 2️⃣ Install Dependencies
pip install tensorflow streamlit pandas numpy matplotlib

### 3️⃣ Prepare the Dataset
Place train, valid, and test folders in the project directory.
Run temp.py to generate temperature metadata:
python temp.py

### 4️⃣ Train the Model
python temp_train.py
This will save:
fish_freshness_model.h5 — final trained model
training_history.png — accuracy/loss plot

### 5️⃣ Run the Web App
streamlit run temp_test.py
The app will be live at: http://localhost:8501

## 🎯 Use Cases

- 🐟 Seafood industry — Quality control and freshness verification
- 🧑‍🍳 Restaurants & markets — Real-time quality check before cooking/selling
- 📱 Mobile integration — Potential for consumer-facing freshness detection apps
- 🎓 Educational & research — Computer vision applications in food safety

## 👩‍💻 Author
**Devadarshini P**  
[🔗 LinkedIn](https://www.linkedin.com/in/devadarshini-p-707b15202/)  
[💻 GitHub](https://github.com/Devadarshini9000)

“Ensuring freshness in every catch — powered by AI.”
