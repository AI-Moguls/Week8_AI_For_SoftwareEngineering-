import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import io
from datetime import datetime

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Cropland Classifier", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('random_forest_cropland.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure 'random_forest_cropland.pkl' is in the same directory")
        return None

# Extract vital information from Sentinel-2 data
def process_sentinel2(df):
    """Process Sentinel-2 data and calculate vegetation indices"""
    eps = 1e-8
    required_bands = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12', 'B8A']
    
    # Check for required bands
    missing_bands = [band for band in required_bands if band not in df.columns]
    if missing_bands:
        st.error(f"Missing required Sentinel-2 bands: {', '.join(missing_bands)}")
        return None
        
    # Convert to numeric
    for band in required_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')
    
    # Calculate vegetation indices
    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + eps)
    df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + eps)
    df['EVI'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 6*df['B4'] - 7.5*df['B2'] + 1 + eps)
    df['NDMI'] = (df['B8'] - df['B11']) / (df['B8'] + df['B11'] + eps)
    df['NDVIre'] = (df['B8A'] - df['B5']) / (df['B8A'] + df['B5'] + eps)
    df['NBR'] = (df['B8'] - df['B12']) / (df['B8'] + df['B12'] + eps)
    
    # Return only essential features
    return df[['ID', 'NDVI', 'NDWI', 'EVI', 'NDMI', 'NDVIre', 'NBR']]

# Extract vital information from Sentinel-1 data
def process_sentinel1(df):
    """Process Sentinel-1 SAR data and calculate ratios"""
    eps = 1e-8
    required_bands = ['VV', 'VH']
    
    # Check for required bands
    missing_bands = [band for band in required_bands if band not in df.columns]
    if missing_bands:
        st.error(f"Missing required Sentinel-1 bands: {', '.join(missing_bands)}")
        return None
        
    # Convert to numeric
    for band in required_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')
    
    # Calculate SAR ratios
    df['VV/VH'] = df['VV'] / (df['VH'] + eps)
    df['VH/VV'] = df['VH'] / (df['VV'] + eps)
    
    # Return only essential features
    return df[['ID', 'VV', 'VH', 'VV/VH', 'VH/VV']]

# Main app
def main():
    # Sidebar with accessibility fix
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Page", ["Home", "Classify", "About"], label_visibility="collapsed")
    
    model = load_model()
    
    if app_mode == "Home":
        st.title("🌾 Multi-Sensor Cropland Classification")
        
        # Using markdown for visual elements
        st.markdown("""
        <div style="background-color:#e8f5e9;padding:20px;border-radius:10px">
            <h2 style="color:#2e7d32">🌱 Satellite-based Agricultural Analysis</h2>
            <p>Upload Sentinel-1 and Sentinel-2 data separately to classify cropland types</p>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **This system uses machine learning to predict cropland types based on:**
        - **Sentinel-2 Optical Data**: Spectral bands and vegetation indices
        - **Sentinel-1 SAR Data**: Polarization values and band ratios
        
        ### How to use:
        1. Go to the **Classify** page
        2. Upload separate CSV files for Sentinel-1 and Sentinel-2 data
        3. View predictions and download results
        
        #### Required CSV columns:
        - **Sentinel-2**: ID, B2, B3, B4, B5, B8, B11, B12, B8A
        - **Sentinel-1**: ID, VV, VH
        """)
        
        if st.checkbox("Show data sources diagram"):
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Sentinel-1-2.jpg/800px-Sentinel-1-2.jpg", 
                     caption="Sentinel-1 (SAR) and Sentinel-2 (Optical) Satellites", use_column_width=True)
        
    elif app_mode == "Classify":
        st.title("📊 Multi-Sensor Data Classification")
        
        # Create two columns for separate uploaders
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentinel-2 Data (Optical)")
            sentinel2_file = st.file_uploader("Upload Sentinel-2 CSV", type="csv", key="s2")
            
        with col2:
            st.subheader("Sentinel-1 Data (SAR)")
            sentinel1_file = st.file_uploader("Upload Sentinel-1 CSV", type="csv", key="s1")
        
        # Check if both files are uploaded
        if sentinel2_file and sentinel1_file:
            try:
                # Check file sizes
                max_size = 2 * 1024 * 1024 * 1024  # 2GB
                if sentinel2_file.size > max_size or sentinel1_file.size > max_size:
                    st.error("File size exceeds 2GB limit. Please upload smaller files.")
                    return
                
                # Read and process Sentinel-2 data
                s2_df = pd.read_csv(sentinel2_file)
                st.success(f"Sentinel-2: {s2_df.shape[0]} records")
                s2_processed = process_sentinel2(s2_df)
                
                # Read and process Sentinel-1 data
                s1_df = pd.read_csv(sentinel1_file)
                st.success(f"Sentinel-1: {s1_df.shape[0]} records")
                s1_processed = process_sentinel1(s1_df)
                
                # Check if processing succeeded
                if s2_processed is None or s1_processed is None:
                    st.error("Failed to process one or more datasets")
                    return
                
                # Merge datasets on ID
                merged_df = pd.merge(s2_processed, s1_processed, on='ID', how='inner')
                
                if merged_df.empty:
                    st.error("No matching IDs found between Sentinel-1 and Sentinel-2 datasets")
                    return
                    
                st.success(f"Merged dataset: {merged_df.shape[0]} records with matching IDs")
                
                if model:
                    # Define all possible features
                    all_features = [
                        'NDVI', 'NDWI', 'EVI', 'NDMI', 'NDVIre', 'NBR',
                        'VV', 'VH', 'VV/VH', 'VH/VV'
                    ]
                    
                    # Select only features present in both data and model
                    available_features = [f for f in all_features if f in merged_df.columns]
                    
                    # Handle missing features
                    missing_features = [f for f in all_features if f not in available_features]
                    if missing_features:
                        st.warning(f"⚠️ Missing features: {', '.join(missing_features)}")
                    
                    X = merged_df[available_features]
                    
                    # Check for NaN values
                    if X.isna().any().any():
                        st.warning("⚠️ Some features contain missing values. Filling with zeros.")
                        X = X.fillna(0)
                    
                    # Make predictions
                    with st.spinner("Classifying cropland types..."):
                        predictions = model.predict(X)
                        merged_df['Predicted_Class'] = predictions
                    
                    # Results
                    st.subheader("📋 Classification Results")
                    st.dataframe(merged_df[['ID', 'Predicted_Class']].head())
                    
                    # Download
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv = merged_df[['ID', 'Predicted_Class']].to_csv(index=False)
                    st.download_button(
                        label="💾 Download Predictions",
                        data=csv,
                        file_name=f"cropland_predictions_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                    # Class distribution
                    st.subheader("📊 Class Distribution")
                    class_counts = merged_df['Predicted_Class'].value_counts()
                    st.bar_chart(class_counts)
                    
                    # Show sample features
                    with st.expander("🔍 View processed features"):
                        st.dataframe(merged_df[available_features].head())
            except Exception as e:
                st.error(f"❌ Error processing files: {e}")
                st.error("Please check the CSV formats and try again")
        elif sentinel2_file or sentinel1_file:
            st.warning("⚠️ Please upload both Sentinel-1 and Sentinel-2 datasets")
            
    elif app_mode == "About":
        st.title("ℹ️ About")
        st.markdown("""
        ### Multi-Sensor Cropland Classification System
        
        **New Feature: Separate Data Upload**
        - Upload Sentinel-1 (SAR) and Sentinel-2 (optical) data separately
        - Intelligent feature extraction:
          - Sentinel-2: Calculates 6 vegetation indices (NDVI, NDWI, EVI, NDMI, NDVIre, NBR)
          - Sentinel-1: Calculates polarization ratios (VV/VH, VH/VV)
        - Automatic merging on ID column
        
        **Technical Specifications:**
        - Model: Random Forest (100 estimators)
        - Features: 6 vegetation indices + 4 SAR features
        - Processing: Automatic handling of missing values
        
        **Data Processing Workflow:**
        1. Upload separate CSV files for each satellite
        2. System extracts essential features:
           - Optical → Vegetation indices
           - SAR → Polarization values and ratios
        3. Datasets merged using location ID
        4. Machine learning model makes predictions
        
        Developed for precision agriculture applications using multi-sensor fusion.
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()