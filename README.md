# 🌍 GeoAI Challenge for Cropland Mapping in Dry Regions (Fergana & Orenburg)

This project presents a robust and cost-effective deep learning pipeline for cropland mapping in arid and semi-arid regions using **multi-temporal satellite imagery**. The solution was developed for the **AI for Good - Zindi Challenge**, focused on distinguishing cropland from non-cropland areas (such as pastures and steppe land) in **Fergana, Uzbekistan** and **Orenburg, Russia**.

---

## 📌 Challenge Overview

The task is to predict cropland presence using remote sensing data:
- 🌿 **Input**: Time-series Sentinel-1 (SAR) and Sentinel-2 (optical) satellite data
- 🎯 **Output**: Binary classification — `1` = cropland, `0` = non-cropland
- 📊 **Metric**: Accuracy
- 📁 **Submission Format**:
```csv
ID,Target
ID_ABC123,1
ID_DEF456,0

This contributes to SDG 2 (Zero Hunger), SDG 13 (Climate Action), and SDG 15 (Life on Land).

📂 Project Structure

├── data/
│   ├── Sentinel1.csv              # SAR data (VH, VV)
│   ├── Sentinel2.csv              # Optical data (B2–B12)
│   ├── Train/                     # Ground truth shapefiles
│   └── Test.csv                   # Unlabeled test data
├── notebooks/
│   ├── 1_EDA.ipynb                # Data exploration
│   ├── 2_FeatureEngineering.ipynb
│   ├── 3_ModelTraining.ipynb
│   ├── 4_Inference.ipynb
├── models/
│   └── best_cnn_model.h5          # Trained 1D CNN model
├── outputs/
│   └── submission.csv             # Final predictions
├── requirements.txt
├── environment.yml
└── README.md                      # This file

⚙️ Environment Setup
📦 Dependencies (pip)
pip install --upgrade scikit-learn==1.6.1 imbalanced-learn category-encoders numpy==1.26.4 pandas scipy tensorflow keras geopandas matplotlib seaborn pyproj
🐍 Conda
conda env create -f environment.yml
conda activate cropland-env
🧠 Hardware Requirements
Minimum: 8GB RAM

Recommended: GPU-enabled environment (Google Colab, local CUDA)

Training runtime: ~25 minutes on GPU, ~1hr+ on CPU

📥 Dataset Description
Sentinel-1: Synthetic Aperture Radar (SAR)

Polarizations: VH and VV

Cloud-independent, sensitive to soil moisture and structure

Sentinel-2: Multispectral optical data

Bands used: B2 to B12

Sensitive to vegetation, chlorophyll, and land use

Labels:

Shapefiles with cropland polygons for Fergana and Orenburg

Converted and matched via spatial KDTree to tabular data

🔧 Data Processing Pipeline
1. Multi-source Data Integration
Joined Sentinel-1 and Sentinel-2 by coordinate

Matched labels to raster features via KDTree

2. Coordinate Handling
Converted UTM to WGS84 using pyproj

Supports EPSG:32642 (Fergana) and EPSG:32640 (Orenburg)

3. Vegetation Indices
NDVI = (B8 - B4) / (B8 + B4)

NDWI = (B3 - B8) / (B3 + B8)

4. Temporal Aggregation
Mean, std for each band and index across time

Feature scaling using StandardScaler

5. Data Cleaning
Removed missing values and invalid geometries

Harmonized data types and column order

🧠 Model Architecture: 1D Convolutional Neural Network

model = Sequential([
    Input(shape=(features, 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    Dropout(0.2),
    Conv1D(64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])
Training Strategy
Train/val split: Stratified 80/20

Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

Callbacks: EarlyStopping, ModelCheckpoint

🧪 Model Evaluation
Accuracy: Primary metric for leaderboard

Confusion Matrix: Analyzed precision/recall

Feature Importance: Evaluated for temporal patterns

Validation Plots:

Loss and Accuracy Curves

Label Distribution

Prediction Map Visualization

🚀 Inference & Submission
Predict & Save

test_preds = model.predict(X_test)
pred_labels = np.argmax(test_preds, axis=1)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Target': pred_labels
})
submission.to_csv('submission.csv', index=False)
Ensure your submission.csv strictly matches the required format.

🔁 Reproducibility Checklist
✅ random.seed, np.random.seed, tf.random.set_seed fixed

✅ Model checkpoints saved

✅ Dependencies fully listed (requirements.txt, environment.yml)

✅ Submission reproducible with provided notebooks

✅ Final run: Colab + GPU

📌 How to Run the Code
1. Clone Repository

git clone https://github.com/AI-Moguls/Week8_AI_For_SoftwareEngineering-

2. Set Up Environment

conda env create -f environment.yml
conda activate cropland-env

3. Run in Order
notebooks/1_EDA.ipynb – Explore the data

notebooks/2_FeatureEngineering.ipynb – Generate features

notebooks/3_ModelTraining.ipynb – Train CNN model

notebooks/4_Inference.ipynb – Run predictions & generate submission

📈 Applications
🛰️ Agricultural Monitoring – Cropland detection and area estimation

📊 Land Use Planning – Crop rotation and expansion analysis

🛡️ Food Security Policy – Agricultural forecasting and drought planning

🌱 Remote Sensing Research – Fusion of SAR and optical for land cover mapping

🙌 Acknowledgments
This project was developed as part of the Zindi AI for Good Challenge, supported by the International Telecommunication Union (ITU) and 40 UN Sister Agencies.

📜 License
All datasets and code used in this challenge are covered under the CC-BY 1.0 License. 

👥 Team
Team Name: AI MOGULS
Members:

## ✍️ Contributors
- [Brian Ouko](https://github.com/WellBrian)
- [Mmabath Naseba](https://github.com/Mmabatho)
- [Holuwercheyy Hisserhah](https://github.com/holuwercheyy)
- [Letshego Sephiri](https://github.com/CaramelF)
- [Christopher Obegi](https://github.com/mechriz)
