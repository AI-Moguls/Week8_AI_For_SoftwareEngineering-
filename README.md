# GeoAI Challenge for Cropland Mapping - Dry Dataset

A comprehensive machine learning project for cropland classification using satellite imagery from Sentinel-1 and Sentinel-2 satellites. This project implements a CNN-based approach for mapping agricultural land in dry/arid regions using geospatial data from Fergana and Orenburg regions.

## Project Overview

This project uses multi-temporal satellite data to classify cropland vs non-cropland areas in dry regions. The implementation combines Sentinel-1 SAR data and Sentinel-2 optical data with advanced feature engineering and deep learning techniques.

## Dataset

The dataset used for this project was obtained from **Kaggle**:  
📂 [GeoAI Challenge for Cropland Mapping (Dry Dataset)](https://www.kaggle.com/datasets/noob786/geoai-challenge-for-cropland-mapping-dry-dataset)

This dataset provides labeled data for cropland mapping tasks, including geospatial features and vegetation indices for cropland classification challenges.

---
The project uses the "GeoAI Challenge for Cropland Mapping - Dry Dataset" which includes:
- **Sentinel-1 SAR data**: VH and VV polarization bands
- **Sentinel-2 optical data**: Multi-spectral bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
- **Training labels**: Ground truth data from Fergana and Orenburg regions
- **Test data**: Unlabeled samples for prediction

---

## 🚀 Table of Contents

- [GeoAI Challenge for Cropland Mapping - Dry Dataset](#geoai-challenge-for-cropland-mapping---dry-dataset)
  - [Project Overview](#project-overview)
  - [Dataset](#dataset)
  - [🚀 Table of Contents](#-table-of-contents)
  - [Project Overview](#project-overview-1)
  - [Development Workflow](#development-workflow)
    - [1. Data Collection \& Preprocessing](#1-data-collection--preprocessing)
    - [2. Feature Engineering](#2-feature-engineering)
    - [3. Model Training \& Validation](#3-model-training--validation)
    - [4. Model Export](#4-model-export)
    - [5. App Development (Streamlit)](#5-app-development-streamlit)
  - [Pitch Deck](#pitch-deck)
  - [Contributing](#contributing)
  - [License](#license)

---

## Project Overview

This project aims to classify satellite imagery pixels as **cropland** or **non-cropland** using vegetation index data and machine learning. The final deliverable is an interactive web app built with **Streamlit**, allowing users to explore predictions interactively.

---

## Development Workflow

### 1. Data Collection & Preprocessing
- Download dataset from Kaggle (link above).
- Load using `pandas` and inspect the structure.
- Handle missing values, normalize numeric features.
- Convert categorical attributes into numeric (if any).

### 2. Feature Engineering
- Compute relevant indices (NDVI, EVI) if required.
- Aggregate time-series data into statistical summaries (mean, std).
- Encode features into a model-friendly format.

### 3. Model Training & Validation
- Split data into **training (80%)** and **testing (20%)**.
- Train model -> **Random Forest**
- Use **GridSearchCV** for hyperparameter tuning.
- Evaluate performance using **accuracy**.

### 4. Model Export
- Save trained model as `random_forest_cropland.pkl` using `joblib`.
- Store any preprocessing transformers if needed.

### 5. App Development (Streamlit)
- Build `app.py` to:
  - Load `random_forest_cropland.pkl`.
  - Accept user inputs (coordinates or uploaded data).
  - Display predictions and classification results.
- Example snippet:
  ```python
  st.title("GeoAI Cropland Classifier")
  user_input = st.file_uploader("Upload your data file", type=["csv"])
  if user_input:
      data = pd.read_csv(user_input)
      prediction = rf_model.predict(data)
      st.write("Prediction:", prediction)

## Requirements
```
streamlit
numpy
pandas
scikit-learn
imbalanced-learn
category-encoders
scipy
joblib
Pillow
chardet
requests
```
Push repository to GitHub.

Deploy on Streamlit Cloud by linking repo and selecting `app.py.`

## Usage

1. Clone this repo:
  ```
  git clone <repo-url>
  ```
2. Install dependencies:
  ```
  pip install -r requirements.txt
  ```
3. streamlit run app.py
  ```
  streamlit run app.py
  ```

👉 **Live Demo:** [https://geoai-cropland-classifier.streamlit.app/](https://geoai-cropland-classifier.streamlit.app/)

## Pitch Deck
 Please use the link below to access our pitch deck:
 
 https://gamma.app/docs/GeoAI-Challenge-Cropland-Mapping-qcvvga4s9gwiqo9

## Contributing

This project is part of the AI-Moguls Week 8 AI for Software Engineering challenge. Contributions are as follows:
- [Brian Ouko](https://github.com/WellBrian)
- [Mmabath Naseba](https://github.com/Mmabatho)
- [Holuwercheyy Hisserhah](https://github.com/holuwercheyy)
- [Letshego Sephiri](https://github.com/CaramelF)
- [Christopher Obegi](https://github.com/mechriz)

## License

Please refer to the original dataset page on Kaggle for licensing information and usage rights.
