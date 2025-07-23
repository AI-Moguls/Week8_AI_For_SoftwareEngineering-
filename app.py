import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('random_forest_cropland.pkl')

# Calculate vegetation indices
def calculate_vegetation_indices(df):
    eps = 1e-8
    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + eps)
    df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + eps)
    df['EVI'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 6*df['B4'] - 7.5*df['B2'] + 1 + eps)
    df['NDMI'] = (df['B8'] - df['B11']) / (df['B8'] + df['B11'] + eps)
    df['NDVIre'] = (df['B8A'] - df['B5']) / (df['B8A'] + df['B5'] + eps)
    df['NBR'] = (df['B8'] - df['B12']) / (df['B8'] + df['B12'] + eps)
    df['MSAVI'] = (2*df['B8'] + 1 - np.sqrt((2*df['B8']+1)**2 - 8*(df['B8']-df['B4']))) / 2
    df['ARVI'] = (df['B8'] - (2*df['B4'] - df['B2'])) / (df['B8'] + (2*df['B4'] - df['B2']) + eps)
    return df

# Main app
def main():
    st.set_page_config(page_title="Cropland Classifier", layout="wide")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("", ["Home", "Classify", "About"])
    
    # Load model
    model = load_model()
    
    if app_mode == "Home":
        st.title("🌾 Cropland Classification System")
        st.image("cropland.jpg", use_column_width=True)
        st.markdown("""
        **Upload Sentinel-2 satellite data to classify cropland types**
        
        This system uses machine learning to predict cropland types based on:
        - 10 Sentinel-2 spectral bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
        - 8 calculated vegetation indices (NDVI, NDWI, EVI, NDMI, NDVIre, NBR, MSAVI, ARVI)
        
        ### How to use:
        1. Go to the **Classify** page
        2. Upload a CSV file with Sentinel-2 band data
        3. View predictions and download results
        """)
        
    elif app_mode == "Classify":
        st.title("📊 Data Classification")
        uploaded_file = st.file_uploader("Upload Sentinel-2 CSV", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {df.shape[0]} records")
            
            with st.expander("View raw data"):
                st.dataframe(df.head())
            
            # Process data
            with st.spinner("Calculating vegetation indices..."):
                processed_df = calculate_vegetation_indices(df.copy())
                
            # Make predictions
            with st.spinner("Classifying cropland types..."):
                features = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
                            'NDVI','NDWI','EVI','NDMI','NDVIre','NBR','MSAVI','ARVI']
                X = processed_df[features]
                predictions = model.predict(X)
                processed_df['Predicted_Class'] = predictions
            
            # Results
            st.subheader("Classification Results")
            st.dataframe(processed_df[['ID', 'Predicted_Class']].head())
            
            # Download
            csv = processed_df[['ID', 'Predicted_Class']].to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="cropland_predictions.csv",
                mime="text/csv"
            )
            
            # Class distribution
            st.subheader("Class Distribution")
            class_counts = processed_df['Predicted_Class'].value_counts()
            st.bar_chart(class_counts)
            
    elif app_mode == "About":
        st.title("ℹ️ About")
        st.markdown("""
        ### Cropland Classification System
        This application uses a Random Forest classifier trained on Sentinel-2 satellite data 
        to identify different types of agricultural land cover.
        
        **Technical Specifications:**
        - Model: Random Forest (100 estimators)
        - Features: 10 spectral bands + 8 vegetation indices
        - Processing: SMOTE for handling class imbalance
        
        **Vegetation Indices Calculated:**
        - NDVI, NDWI, EVI, NDMI, NDVIre, NBR, MSAVI, ARVI
        
        Developed for precision agriculture applications using machine learning.
        """)

if __name__ == "__main__":
    main()