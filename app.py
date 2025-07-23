import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys

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

# Calculate vegetation indices and handle SAR data
def preprocess_features(df):
    eps = 1e-8
    
    # Sentinel-2 bands processing
    sentinel2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    for band in sentinel2_bands:
        if band in df.columns:
            df[band] = pd.to_numeric(df[band], errors='coerce')
    
    # Calculate vegetation indices if Sentinel-2 bands are present
    if all(band in df.columns for band in ['B4', 'B8']):
        df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + eps)
        df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + eps)
        df['EVI'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 6*df['B4'] - 7.5*df['B2'] + 1 + eps)
        df['NDMI'] = (df['B8'] - df['B11']) / (df['B8'] + df['B11'] + eps)
        df['NDVIre'] = (df['B8A'] - df['B5']) / (df['B8A'] + df['B5'] + eps)
        df['NBR'] = (df['B8'] - df['B12']) / (df['B8'] + df['B12'] + eps)
        df['MSAVI'] = (2*df['B8'] + 1 - np.sqrt((2*df['B8']+1)**2 - 8*(df['B8']-df['B4']))) / 2
        df['ARVI'] = (df['B8'] - (2*df['B4'] - df['B2'])) / (df['B8'] + (2*df['B4'] - df['B2']) + eps)
    
    # Sentinel-1 SAR data processing
    sar_bands = ['VV', 'VH', 'VV/VH', 'VH/VV']
    for band in sar_bands:
        if band in df.columns:
            df[band] = pd.to_numeric(df[band], errors='coerce')
    
    # Calculate SAR ratios if bands are present
    if 'VV' in df.columns and 'VH' in df.columns:
        df['VV/VH'] = df['VV'] / (df['VH'] + eps)
        df['VH/VV'] = df['VH'] / (df['VV'] + eps)
    
    return df

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
            <p>Upload combined Sentinel-1 and Sentinel-2 data to classify cropland types</p>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **This system uses machine learning to predict cropland types based on:**
        - **Sentinel-2 Optical Data**: 10 spectral bands (B2-B8A, B11-B12)
        - **Vegetation Indices**: NDVI, NDWI, EVI, NDMI, NDVIre, NBR, MSAVI, ARVI
        - **Sentinel-1 SAR Data**: VV, VH polarizations and VV/VH ratios
        
        ### How to use:
        1. Go to the **Classify** page
        2. Upload a CSV file with combined Sentinel-1 and Sentinel-2 data
        3. View predictions and download results
        
        #### Required CSV columns:
        - **ID**: Unique identifier
        - **Sentinel-2**: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
        - **Sentinel-1**: VV, VH
        """)
        
        if st.checkbox("Show data sources diagram"):
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Sentinel-1-2.jpg/800px-Sentinel-1-2.jpg", 
                     caption="Sentinel-1 (SAR) and Sentinel-2 (Optical) Satellites", use_column_width=True)
        
    elif app_mode == "Classify":
        st.title("📊 Multi-Sensor Data Classification")
        uploaded_file = st.file_uploader("Upload Combined Sentinel-1 & Sentinel-2 CSV", type="csv")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Uploaded {df.shape[0]} records")
                
                # Check for required columns
                required_columns = ['ID', 'VV', 'VH']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    return
                    
                # Process data
                processed_df = preprocess_features(df.copy())
                    
                if model:
                    # Define all possible features
                    all_features = [
                        'B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
                        'NDVI','NDWI','EVI','NDMI','NDVIre','NBR','MSAVI','ARVI',
                        'VV', 'VH', 'VV/VH', 'VH/VV'
                    ]
                    
                    # Select only features present in both data and model
                    available_features = [f for f in all_features if f in processed_df.columns]
                    
                    # Handle missing features
                    missing_features = [f for f in all_features if f not in available_features]
                    if missing_features:
                        st.warning(f"⚠️ Missing optional features: {', '.join(missing_features)}")
                    
                    X = processed_df[available_features]
                    
                    # Check for NaN values
                    if X.isna().any().any():
                        st.warning("⚠️ Some features contain missing values. Filling with zeros.")
                        X = X.fillna(0)
                    
                    # Make predictions
                    with st.spinner("Classifying cropland types..."):
                        predictions = model.predict(X)
                        processed_df['Predicted_Class'] = predictions
                    
                    # Results
                    st.subheader("📋 Classification Results")
                    st.dataframe(processed_df[['ID', 'Predicted_Class']].head())
                    
                    # Download
                    csv = processed_df[['ID', 'Predicted_Class']].to_csv(index=False)
                    st.download_button(
                        label="💾 Download Predictions",
                        data=csv,
                        file_name="cropland_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Class distribution
                    st.subheader("📊 Class Distribution")
                    class_counts = processed_df['Predicted_Class'].value_counts()
                    st.bar_chart(class_counts)
                    
                    # Show sample features
                    with st.expander("🔍 View processed features"):
                        st.dataframe(processed_df[available_features].head())
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.error("Please check the CSV format and try again")
            
    elif app_mode == "About":
        st.title("ℹ️ About")
        st.markdown("""
        ### Multi-Sensor Cropland Classification System
        
        **Technical Specifications:**
        - Model: Random Forest (100 estimators)
        - Features: 
          - Sentinel-2 optical bands (10 bands)
          - 8 vegetation indices
          - Sentinel-1 SAR data (VV, VH polarizations)
          - SAR band ratios (VV/VH, VH/VV)
        - Processing: SMOTE for handling class imbalance
        
        **Data Fusion Approach:**
        This application combines data from both optical (Sentinel-2) and radar (Sentinel-1) satellites:
        
        <div style="display:flex; justify-content:space-between; margin:20px 0">
            <div style="text-align:center">
                <h4>Sentinel-2</h4>
                <p>Optical Data</p>
                <p>🌿 Vegetation Indices</p>
                <p>🌈 Spectral Signatures</p>
            </div>
            <div style="text-align:center">
                <h4>Sentinel-1</h4>
                <p>SAR Data</p>
                <p>📡 All-Weather Capability</p>
                <p>🔄 Surface Texture</p>
            </div>
        </div>
        
        **Advantages of Multi-Sensor Fusion:**
        - Works in cloudy conditions (SAR penetrates clouds)
        - Captures both spectral and structural information
        - Improves accuracy for difficult terrain
        - Provides more consistent monitoring
        
        Developed for precision agriculture applications using machine learning.
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()