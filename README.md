# 🛰️ GeoAI Cropland Mapping Pipeline

A complete Machine Learning pipeline for cropland mapping using Sentinel-1 and Sentinel-2 satellite data. This project implements a comprehensive ML workflow from data preprocessing to model deployment for agricultural land classification.

## 🌟 Features

- **Multi-sensor Data Integration**: Combines Sentinel-1 SAR and Sentinel-2 optical satellite data
- **Complete ML Pipeline**: 9-step end-to-end machine learning workflow
- **Multiple Algorithms**: Random Forest, Gradient Boosting, Logistic Regression, and SVM
- **Advanced Preprocessing**: Feature scaling, missing value imputation, and categorical encoding
- **Comprehensive Evaluation**: Confusion matrices, classification reports, and feature importance analysis
- **Model Persistence**: Save and load trained models with all components
- **Visualization**: EDA plots, confusion matrices, and feature importance charts

## 📊 Dataset

The pipeline is designed to work with the **GeoAI Challenge for Cropland Mapping** dataset containing:

### Sentinel-1 SAR Features:
- VV and VH polarization backscatter coefficients
- Statistical measures (mean, standard deviation)
- Texture features (contrast, homogeneity)
- Coherence measurements
- VV/VH ratio

### Sentinel-2 Optical Features:
- Spectral bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
- Vegetation indices (NDVI, NDWI, EVI, SAVI)
- Enhanced vegetation metrics

## 🚀 Quick Start

### Prerequisites

\`\`\`bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
\`\`\`

### Running the Pipeline

```python
from cropland_mapping_pipeline import CroplandMappingPipeline

# Initialize and run complete pipeline
pipeline = CroplandMappingPipeline()
pipeline.run_complete_pipeline()
