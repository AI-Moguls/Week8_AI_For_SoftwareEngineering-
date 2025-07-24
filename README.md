# GeoAI Challenge for Cropland Mapping - Dry Dataset

A comprehensive machine learning project for cropland classification using satellite imagery from Sentinel-1 and Sentinel-2 satellites. This project implements a CNN-based approach for mapping agricultural land in dry/arid regions using geospatial data from Fergana and Orenburg regions.

## Project Overview

This project uses multi-temporal satellite data to classify cropland vs non-cropland areas in dry regions. The implementation combines Sentinel-1 SAR data and Sentinel-2 optical data with advanced feature engineering and deep learning techniques.

## Dataset

The project uses the "GeoAI Challenge for Cropland Mapping - Dry Dataset" which includes:
- **Sentinel-1 SAR data**: VH and VV polarization bands
- **Sentinel-2 optical data**: Multi-spectral bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
- **Training labels**: Ground truth data from Fergana and Orenburg regions
- **Test data**: Unlabeled samples for prediction

## Installation and Setup

### Required Dependencies

```bash
pip install --upgrade scikit-learn==1.6.1 imbalanced-learn category-encoders numpy==1.26.4 pandas scipy tensorflow keras geopandas matplotlib seaborn pyproj
```

### Dataset Download

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("noob786/geoai-challenge-for-cropland-mapping-dry-dataset")
print("Path to dataset files:", path)
```

## Key Features

### 1. Data Processing Pipeline
- **Multi-source data integration**: Combines Sentinel-1 and Sentinel-2 data
- **Temporal aggregation**: Uses mean and standard deviation across time series
- **Coordinate system handling**: Automatic conversion between UTM and WGS84 projections
- **Spatial matching**: KDTree-based nearest neighbor matching for label assignment

### 2. Feature Engineering
- **Vegetation indices calculation**:
  - NDVI (Normalized Difference Vegetation Index)
  - NDWI (Normalized Difference Water Index)
- **Statistical aggregation**: Mean and standard deviation for all temporal features
- **Data cleaning**: Robust handling of missing values and coordinate transformations

### 3. Machine Learning Model
- **Architecture**: 1D Convolutional Neural Network (CNN)
- **Input preprocessing**: StandardScaler normalization
- **Model structure**:
  - Conv1D layers (32, 64 filters)
  - Dropout for regularization
  - Dense layers with ReLU activation
  - Softmax output for multi-class classification

### 4. Training Strategy
- **Data splitting**: Stratified train-validation split (80/20)
- **Callbacks**: Model checkpointing and early stopping
- **Optimization**: Adam optimizer with categorical crossentropy loss
- **Evaluation**: Accuracy metrics and confusion matrix analysis

## Project Structure

```
├── data/
│   ├── Sentinel1.csv
│   ├── Sentinel2.csv
│   ├── Train/
│   │   ├── Fergana_training_samples.shp
│   │   └── Orenburg_training_samples.shp
│   └── Test.csv
├── models/
│   └── best_cnn_model.h5
├── outputs/
│   └── submission.csv
└── index.ipynb
```

## Usage

### 1. Data Loading and Preprocessing
```python
# Load satellite data
s1 = pd.read_csv("Sentinel1.csv").drop(columns=['date'])
s2 = pd.read_csv("Sentinel2.csv").drop(columns=['date'])

# Load training labels
train_gdf = load_training_data()
```

### 2. Feature Engineering
```python
# Calculate vegetation indices
s2 = calculate_vegetation_indices(s2)

# Aggregate temporal features
s2_agg = aggregate_features(s2)
s1_agg = aggregate_features(s1)
```

### 3. Model Training
```python
# Build and compile CNN model
model = Sequential([
    Input(shape=(features, 1)),
    Conv1D(32, kernel_size=3, activation='relu'),
    Dropout(0.2),
    Conv1D(64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Train with callbacks
model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[checkpoint])
```

### 4. Prediction and Evaluation
```python
# Generate predictions
test_preds = model.predict(X_test_cnn)
predicted_labels = np.argmax(test_preds, axis=1)

# Save submission
submission = pd.DataFrame({'ID': test_df['ID'], 'label': predicted_labels})
submission.to_csv('submission.csv', index=False)
```

## Key Implementation Details

### Coordinate System Handling
- Automatic detection and conversion of UTM coordinates to WGS84
- Support for multiple UTM zones (Fergana: EPSG:32642, Orenburg: EPSG:32640)
- Spatial matching with configurable distance threshold (0.5 degrees)

### Feature Engineering Pipeline
- Temporal aggregation using robust statistical measures
- Vegetation index calculation for enhanced crop detection
- Handling of missing values and data type conversions
- Feature scaling and normalization for neural network training

### Model Architecture
- 1D CNN designed for multi-temporal satellite data
- Dropout regularization to prevent overfitting
- Model checkpointing to save best performing weights
- Categorical output for multi-class land cover classification

## Results and Visualization

The project includes comprehensive visualization and analysis:
- Training/validation accuracy and loss curves
- Confusion matrix for model evaluation
- Predicted label distribution analysis
- Feature importance and data quality checks

## Applications

- **Agricultural monitoring**: Automated cropland mapping
- **Land use planning**: Regional agricultural assessment
- **Food security**: Crop area estimation and monitoring
- **Environmental studies**: Land cover change detection
- **Remote sensing research**: Multi-sensor data fusion techniques

## Technical Requirements

- Python 3.7+
- TensorFlow/Keras for deep learning
- GeoPandas for geospatial data processing
- Scikit-learn for preprocessing and evaluation
- Sufficient computational resources for CNN training


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
