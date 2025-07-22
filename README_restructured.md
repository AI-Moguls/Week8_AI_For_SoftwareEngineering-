# 🌍 GeoAI Challenge for Cropland Mapping in Dry Regions (Fergana & Orenburg)

This project presents a robust and cost-effective deep learning pipeline for cropland mapping in arid and semi-arid regions using **multi-temporal satellite imagery**. It was developed for the **AI for Good - Zindi Challenge**, focused on distinguishing cropland from non-cropland (e.g., pastures and steppe) in **Fergana, Uzbekistan** and **Orenburg, Russia**.

---

## 📌 Challenge Overview

- 🌿 **Input**: Time-series Sentinel-1 (SAR) and Sentinel-2 (optical) satellite data  
- 🎯 **Output**: Binary classification — `1` = cropland, `0` = non-cropland  
- 📊 **Metric**: Accuracy  
- 📁 **Submission Format**:
```csv
ID,Target
ID_ABC123,1
ID_DEF456,0
```

✅ This challenge aligns with:
- **SDG 2**: Zero Hunger
- **SDG 13**: Climate Action
- **SDG 15**: Life on Land

---

## 📂 Project Structure

```
├── data/
│   ├── Sentinel1.csv
│   ├── Sentinel2.csv
│   ├── Train/
│   └── Test.csv
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_FeatureEngineering.ipynb
│   ├── 3_ModelTraining.ipynb
│   ├── 4_Inference.ipynb
├── models/
│   └── best_cnn_model.h5
├── outputs/
│   └── submission.csv
├── requirements.txt
├── environment.yml
└── README.md
```

---

## ⚙️ Environment Setup

### 📦 Install Dependencies
```bash
pip install --upgrade scikit-learn==1.6.1 imbalanced-learn category-encoders numpy==1.26.4 pandas scipy tensorflow keras geopandas matplotlib seaborn pyproj
```

### 🐍 Conda (optional)
```bash
conda env create -f environment.yml
conda activate cropland-env
```

### 🧠 Hardware Requirements
- **Minimum**: 8GB RAM  
- **Recommended**: GPU-enabled environment (Google Colab or local CUDA)  
- **Runtime**: ~25 mins on GPU, ~1hr+ on CPU

---

## 📥 Dataset Description

### 🛰️ Sentinel-1 (SAR)
- Bands: VH and VV
- Cloud-independent; sensitive to soil moisture

### 🌈 Sentinel-2 (Optical)
- Bands: B2–B12
- Sensitive to vegetation and chlorophyll

### 🗂️ Labels
- Ground truth shapefiles from Fergana and Orenburg
- Mapped to raster data using KDTree matching

---

## 🔧 Data Processing Pipeline

1. **Multi-source Integration**: Merge Sentinel-1 and Sentinel-2 data by coordinates  
2. **Coordinate Handling**: Convert UTM → WGS84 (EPSG:32642, EPSG:32640)  
3. **Vegetation Indices**:
   - NDVI = (B8 - B4) / (B8 + B4)
   - NDWI = (B3 - B8) / (B3 + B8)
4. **Temporal Aggregation**: Mean, std for each band/index  
5. **Data Cleaning**: Drop missing values, harmonize columns  
6. **Feature Scaling**: StandardScaler normalization

---

## 🧠 Model Architecture

A 1D Convolutional Neural Network:

```python
model = Sequential([
    Input(shape=(features, 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    Dropout(0.2),
    Conv1D(64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])
```

### 🔄 Training Strategy
- **Split**: Stratified 80/20 train-validation  
- **Loss**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Callbacks**: EarlyStopping, ModelCheckpoint

---

## 🧪 Evaluation

- **Primary Metric**: Accuracy  
- **Tools**:
  - Confusion Matrix
  - Loss & Accuracy Curves
  - Label Distribution
  - Prediction Map Visualization

---

## 🚀 Inference & Submission

```python
test_preds = model.predict(X_test)
pred_labels = np.argmax(test_preds, axis=1)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Target': pred_labels
})
submission.to_csv('submission.csv', index=False)
```

⚠️ Ensure strict adherence to the required format.

---

## 🔁 Reproducibility Checklist

- ✅ Random seeds (`random.seed`, `np.random.seed`, `tf.random.set_seed`)  
- ✅ Checkpoints saved  
- ✅ `requirements.txt` and `environment.yml` provided  
- ✅ Notebooks fully reproducible  
- ✅ Final run tested on Google Colab + GPU

---

## 🧪 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/AI-Moguls/Week8_AI_For_SoftwareEngineering-

# 2. Set up environment
conda env create -f environment.yml
conda activate cropland-env
```

Then run in order:
1. `1_EDA.ipynb`
2. `2_FeatureEngineering.ipynb`
3. `3_ModelTraining.ipynb`
4. `4_Inference.ipynb`

---

## 📈 Applications

- 🛰️ Agricultural Monitoring  
- 📊 Land Use Planning  
- 🛡️ Food Security Policy  
- 🌱 Remote Sensing Research

---

## 🙌 Acknowledgments

Developed for the **Zindi AI for Good Challenge**, supported by the **International Telecommunication Union (ITU)** and **40 UN Sister Agencies**.

---

## 📜 License

This project and dataset are released under the **CC-BY 1.0 License**.

---

## 👥 Team

**Team Name:** AI MOGULS

### ✍️ Contributors
- [Brian Ouko](https://github.com/WellBrian)
- [Mmabath Naseba](https://github.com/Mmabatho)
- [Holuwercheyy Hisserhah](https://github.com/holuwercheyy)
- [Letshego Sephiri](https://github.com/CaramelF)
- [Christopher Obegi](https://github.com/mechriz)