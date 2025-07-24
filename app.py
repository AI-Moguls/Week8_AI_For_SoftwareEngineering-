import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime
from io import StringIO

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Cropland Classifier", layout="wide")

# Custom CSS to override upload limit text
st.markdown("""
<style>
/* Hide default 200MB text */
.st-emotion-cache-1gulkj5, 
.st-emotion-cache-7ym5gk, 
.uploadedFile { 
    display: none !important; 
}

/* Style for our custom message */
.custom-upload-limit {
    font-size: 0.8rem;
    color: #6c757d;
    margin-top: -15px;
    margin-bottom: 10px;
    text-align: center;
}

/* Style for URL input */
.url-input {
    width: 100%;
    padding: 8px;
    margin-top: 5px;
    border: 1px solid #ccc;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load('random_forest_cropland.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure 'random_forest_cropland.pkl' is in the same directory")
        return None

# Download file from URL
def download_from_url(url, max_size=2*1024*1024*1024):
    try:
        # Get file size from headers
        response = requests.head(url)
        file_size = int(response.headers.get('Content-Length', 0))
        
        if file_size > max_size:
            st.error(f"File size exceeds 2GB limit ({file_size/(1024**3):.2f} GB)")
            return None
            
        # Download with progress
        st.info(f"Downloading file from URL... (Size: {file_size/(1024**2):.2f} MB)")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        response = requests.get(url, stream=True)
        content = bytearray()
        downloaded = 0
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content.extend(chunk)
                downloaded += len(chunk)
                progress = min(downloaded / file_size, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Downloaded: {downloaded/(1024**2):.2f} MB / {file_size/(1024**2):.2f} MB")
        
        progress_bar.empty()
        status_text.empty()
        st.success("Download complete!")
        return content
    except Exception as e:
        st.error(f"Error downloading file: {e}")
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
            <p>Upload Sentinel-1 and Sentinel-2 data or provide direct URLs to classify cropland types</p>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **This system uses machine learning to predict cropland types based on:**
        - **Sentinel-2 Optical Data**: Spectral bands and vegetation indices
        - **Sentinel-1 SAR Data**: Polarization values and band ratios
        
        ### How to use:
        1. Go to the **Classify** page
        2. For each dataset, choose to either:
           - Upload a CSV file (max 2GB)
           - Provide a direct download URL
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
        st.info("💡 You can upload files directly or provide URLs to large files hosted elsewhere", icon="ℹ️")
        
        # Create two columns for separate uploaders
        col1, col2 = st.columns(2)
        
        # Initialize session state
        if 's2_data' not in st.session_state:
            st.session_state.s2_data = None
        if 's1_data' not in st.session_state:
            st.session_state.s1_data = None
        
        with col1:
            st.subheader("Sentinel-2 Data (Optical)")
            s2_option = st.radio("Choose input method:", ["Upload File", "Provide URL"], key="s2_option")
            
            if s2_option == "Upload File":
                sentinel2_file = st.file_uploader("Upload Sentinel-2 CSV", type="csv", key="s2_upload")
                st.markdown('<p class="custom-upload-limit">Max file size: 2GB</p>', unsafe_allow_html=True)
                
                if sentinel2_file:
                    # Check file size
                    if sentinel2_file.size > 2 * 1024 * 1024 * 1024:
                        st.error("File size exceeds 2GB limit")
                        st.session_state.s2_data = None
                    else:
                        try:
                            st.session_state.s2_data = pd.read_csv(sentinel2_file)
                            st.success("Sentinel-2 data loaded successfully!")
                        except Exception as e:
                            st.error(f"Error reading Sentinel-2 CSV: {e}")
                            st.session_state.s2_data = None
            else:
                s2_url = st.text_input("Enter direct download URL for Sentinel-2 CSV:", key="s2_url", 
                                      placeholder="https://example.com/sentinel2.csv")
                if st.button("Load from URL", key="s2_load"):
                    if s2_url:
                        content = download_from_url(s2_url)
                        if content:
                            try:
                                # Convert bytes to string and read CSV
                                csv_string = StringIO(content.decode('utf-8'))
                                st.session_state.s2_data = pd.read_csv(csv_string)
                                st.success("Sentinel-2 data loaded from URL!")
                            except Exception as e:
                                st.error(f"Error processing downloaded file: {e}")
                                st.session_state.s2_data = None
                    else:
                        st.warning("Please enter a valid URL")
        
        with col2:
            st.subheader("Sentinel-1 Data (SAR)")
            s1_option = st.radio("Choose input method:", ["Upload File", "Provide URL"], key="s1_option")
            
            if s1_option == "Upload File":
                sentinel1_file = st.file_uploader("Upload Sentinel-1 CSV", type="csv", key="s1_upload")
                st.markdown('<p class="custom-upload-limit">Max file size: 2GB</p>', unsafe_allow_html=True)
                
                if sentinel1_file:
                    # Check file size
                    if sentinel1_file.size > 2 * 1024 * 1024 * 1024:
                        st.error("File size exceeds 2GB limit")
                        st.session_state.s1_data = None
                    else:
                        try:
                            st.session_state.s1_data = pd.read_csv(sentinel1_file)
                            st.success("Sentinel-1 data loaded successfully!")
                        except Exception as e:
                            st.error(f"Error reading Sentinel-1 CSV: {e}")
                            st.session_state.s1_data = None
            else:
                s1_url = st.text_input("Enter direct download URL for Sentinel-1 CSV:", key="s1_url", 
                                      placeholder="https://example.com/sentinel1.csv")
                if st.button("Load from URL", key="s1_load"):
                    if s1_url:
                        content = download_from_url(s1_url)
                        if content:
                            try:
                                # Convert bytes to string and read CSV
                                csv_string = StringIO(content.decode('utf-8'))
                                st.session_state.s1_data = pd.read_csv(csv_string)
                                st.success("Sentinel-1 data loaded from URL!")
                            except Exception as e:
                                st.error(f"Error processing downloaded file: {e}")
                                st.session_state.s1_data = None
                    else:
                        st.warning("Please enter a valid URL")
        
        # Check if both datasets are loaded
        if st.session_state.s2_data is not None and st.session_state.s1_data is not None:
            try:
                # Process data
                s2_processed = process_sentinel2(st.session_state.s2_data)
                s1_processed = process_sentinel1(st.session_state.s1_data)
                
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
        elif st.session_state.s2_data is not None or st.session_state.s1_data is not None:
            st.warning("⚠️ Please load both Sentinel-1 and Sentinel-2 datasets")
            
    elif app_mode == "About":
        st.title("ℹ️ About")
        st.markdown("""
        ### Multi-Sensor Cropland Classification System
        
        **New Feature: Flexible Data Input**
        - Upload CSV files directly (max 2GB)
        - Provide URLs to large files hosted elsewhere
        - Direct download capability with progress tracking
        
        **Technical Specifications:**
        - Model: Random Forest (100 estimators)
        - Features: 6 vegetation indices + 4 SAR features
        - Processing: Automatic handling of missing values
        - Memory Management: Optimized for large datasets
        
        **Data Input Options:**
        - **Direct Upload**: Best for files under 2GB
        - **URL Download**: Ideal for large files hosted on:
          - Cloud storage (AWS S3, Google Cloud Storage)
          - Institutional servers
          - Public data repositories
        
        Developed for precision agriculture applications using multi-sensor fusion.
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()