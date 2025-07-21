# Cropland Mapping using Deep Learning and Tabular Data

This project addresses the classification of cropland using geospatial features and deep learning techniques. The model utilizes remote sensing data and tabular attributes to classify cropland areas effectively.

## 📦 Dependencies

To run this notebook, install the following packages (some may already be available on Google Colab or Kaggle):

```bash
pip install --upgrade scikit-learn==1.6.1 imbalanced-learn category-encoders numpy==1.26.4 pandas scipy

Other required libraries:

geopandas

tensorflow

matplotlib, seaborn

imbalanced-learn

scikit-learn

# Data Description
The dataset includes geospatial shapefiles and tabular data. Preprocessing steps include:

Loading data with geopandas

Standardizing features

Handling class imbalance with SMOTE

Train-test split with train_test_split

Creating a K-D tree for spatial operations

# Model Architecture
The model is a 1D Convolutional Neural Network built using TensorFlow/Keras. It includes:

Input Layer

Conv1D Layer

Dropout Layers

Dense Layers with ReLU activation

Output Layer with Softmax activation for multi-class classification

python
Copy
Edit
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(64, 3, activation='relu'),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
# Evaluation
Model performance is evaluated using:

Accuracy

F1 Score (macro)

# Confusion Matrix

Visualization is done with matplotlib and seaborn.

# How to Run
Clone this repo or upload the notebook to your environment.

Ensure required libraries are installed.

Run the cells sequentially.

Output metrics and plots will be displayed at the end.

# Results
The notebook outputs:

Confusion matrix for validation set

Training/validation loss and accuracy plots

Best model checkpoint saved using ModelCheckpoint