import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import re
import os
import tempfile
import chardet
import csv
from datetime import datetime
from io import StringIO, BytesIO

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
    color: #0E1117;
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

/* Warning boxes */
.warning-box {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
}

/* Error boxes */
.error-box {
    background-color: #f8d7da;
    border-left: 4px solid #dc3545;
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Try to load model from current directory
        return joblib.load('random_forest_cropland.pkl')
    except FileNotFoundError:
        try:
            # Try to load from absolute path
            return joblib.load('/app/random_forest_cropland.pkl')
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.error("Please make sure 'random_forest_cropland.pkl' is in the correct directory")
            return None

# ==============================
# DOWNLOAD HANDLER (200MB LIMIT)
# ==============================
def download_from_url(url, max_size=200*1024*1024):  # 200MB limit
    """Download file from any source with 200MB limit"""
    try:
        # Create session to handle redirects
        session = requests.Session()
        response = session.head(url, allow_redirects=True)
        
        # Get file size
        file_size = int(response.headers.get('Content-Length', 0))
        
        # Check file size limit
        if file_size > max_size:
            st.error(f"File size exceeds 200MB limit ({file_size/(1024**2):.2f} MB)")
            return None
        
        # Download content
        st.info(f"Downloading file... ({file_size/(1024**2):.2f} MB)")
        response = session.get(url)
        st.success("Download complete!")
        return response.content
    except Exception as e:
        st.error(f"Error downloading file: {str(e)}")
        return None

# ===========================
# ADVANCED CSV PARSER
# ===========================
def robust_read_csv(file_path):
    """Read CSV with advanced error handling and format detection"""
    try:
        # First try standard pandas read
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            st.warning(f"Standard CSV read failed: {str(e)}. Trying advanced methods...")
        
        # Detect encoding
        with open(file_path, 'rb') as f:
            rawdata = f.read(10000)
            encoding = chardet.detect(rawdata)['encoding'] or 'utf-8'
        
        # Detect delimiter
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            first_lines = [f.readline() for _ in range(5)]
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff("\n".join(first_lines))
            except:
                dialect = None
        
        # Read with error-tolerant method
        data = []
        bad_lines = []
        total_lines = 0
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            reader = csv.reader(f, dialect) if dialect else csv.reader(f)
            
            # Get header
            try:
                header = next(reader)
            except StopIteration:
                st.error("CSV file is empty")
                return None
                
            # Process rows
            for i, row in enumerate(reader):
                total_lines += 1
                if len(row) != len(header):
                    bad_lines.append(i+2)  # +1 for header, +1 for 0-index
                    continue
                data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=header)
        
        # Report issues
        if bad_lines:
            st.warning(f"Skipped {len(bad_lines)} malformed lines (rows: {', '.join(map(str, bad_lines[:10]))}{'...' if len(bad_lines) > 10 else ''})")
        
        st.info(f"Successfully read {len(df)} of {total_lines} rows")
        return df
        
    except Exception as e:
        st.error(f"Advanced CSV reading failed: {str(e)}")
        return None

# =============================
# FILE PROCESSING PIPELINE
# =============================
def handle_uploaded_file(uploaded_file, file_type):
    """Process uploaded file with temp storage"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            bytes_data = uploaded_file.read()
            tmp_file.write(bytes_data)
            tmp_file_path = tmp_file.name
        
        # Process with robust reader
        df = robust_read_csv(tmp_file_path)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return df
    except Exception as e:
        st.error(f"Error processing {file_type} file: {str(e)}")
        return None

# =============================
# EXACT FEATURE ENGINEERING (MATCHING NOTEBOOK)
# =============================
def calculate_vegetation_indices(df):
    """Calculate all vegetation indices as in the notebook"""
    eps = 1e-8
    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + eps)
    df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + eps)
    df['EVI'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 6*df['B4'] - 7.5*df['B2'] + 1 + eps)
    df['NDMI'] = (df['B8'] - df['B11']) / (df['B8'] + df['B11'] + eps)
    df['ARVL'] = (df['B8A'] - df['B5']) / (df['B8A'] + df['B5'] + eps)
    df['NBR'] = (df['B8'] - df['B12']) / (df['B8'] + df['B12'] + eps)
    return df

def calculate_sar_ratios(df):
    """Calculate SAR ratios as in the notebook"""
    eps = 1e-8
    df['VV/VH'] = df['VV'] / (df['VH'] + eps)
    df['VH/VV'] = df['VH'] / (df['VV'] + eps)
    return df

def extract_stats(df, group_col, feature_columns):
    """Extract statistics including percentiles as in notebook"""
    # Define statistics including lambda functions for percentiles
    stats = ['min', 'max', 'mean', 'median', 'std', 
             lambda x: np.percentile(x, 25), 
             lambda x: np.percentile(x, 75)]
    
    # Group by ID and compute statistics
    df_agg = df.groupby(group_col)[feature_columns].agg(stats)
    
    # Flatten multi-index columns EXACTLY as in notebook
    df_agg.columns = [f'{col[0]}_{col[1]}' for col in df_agg.columns]
    
    # Replace lambda names with notebook's exact format
    df_agg.columns = [col.replace('_<lambda_0>', '_25percentile') for col in df_agg.columns]
    df_agg.columns = [col.replace('_<lambda_1>', '_75percentile') for col in df_agg.columns]
    
    return df_agg.reset_index()

# =============================
# OPTICAL DATA PROCESSING (FIXED TO MATCH MODEL)
# =============================
def process_sentinel2(df):
    """Process Sentinel-2 data to match model features"""
    required_bands = ['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12', 'B8A']
    
    if df is None:
        return None
    
    # Check for required bands
    missing_bands = [band for band in required_bands if band not in df.columns]
    if missing_bands:
        st.error(f"Missing required Sentinel-2 bands: {', '.join(missing_bands)}")
        return None
        
    # Convert to numeric
    for band in required_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')
    
    # Calculate vegetation indices
    df = calculate_vegetation_indices(df)
    
    # Define features to aggregate (as in notebook)
    optical_features = ['NDVI', 'NDWI', 'EVI', 'NDMI', 'ARVL', 'NBR']
    
    # Extract statistics
    optical_agg = extract_stats(df, 'ID', optical_features)
    
    return optical_agg

# =============================
# SAR DATA PROCESSING (FIXED TO MATCH MODEL)
# =============================
def process_sentinel1(df):
    """Process Sentinel-1 SAR data to match model features"""
    required_bands = ['VV', 'VH']
    
    if df is None:
        return None
    
    # Check for required bands
    missing_bands = [band for band in required_bands if band not in df.columns]
    if missing_bands:
        st.error(f"Missing required Sentinel-1 bands: {', '.join(missing_bands)}")
        return None
        
    # Convert to numeric
    for band in required_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')
    
    # Calculate SAR ratios
    df = calculate_sar_ratios(df)
    
    # Define features to aggregate (as in notebook)
    sar_features = ['VV', 'VH', 'VV/VH', 'VH/VV']
    
    # Extract statistics
    sar_agg = extract_stats(df, 'ID', sar_features)
    
    return sar_agg

# =============================
# OPTIMIZED MERGING FUNCTION
# =============================
def safe_merge(s2_df, s1_df):
    """Merge datasets safely"""
    try:
        # Merge using pandas merge
        merged_df = pd.merge(s2_df, s1_df, on='ID', how='inner')
        return merged_df
    except Exception as e:
        st.error(f"Merge error: {str(e)}")
        return None

# =============================
# MAIN APP WITH FIXED SESSION STATE
# =============================
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
            <p style="color:#2e7d32">Upload Sentinel-1 and Sentinel-2 data or provide direct URLs to classify cropland types</p>
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
           - Upload a CSV file (max 200MB)
           - Provide a direct download URL
        3. View predictions and download results
        
        #### Required CSV columns:
        - **Sentinel-2**: ID, B2, B3, B4, B5, B8, B11, B12, B8A
        - **Sentinel-1**: ID, VV, VH
        
        #### Supported URL Sources:
        - Google Drive (shared links)
        - Dropbox (shared links)
        - OneDrive (shared links)
        - Any direct download URL
        """)
        
        if st.checkbox("Show data sources diagram"):
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Sentinel-1-2.jpg/800px-Sentinel-1-2.jpg", 
                     caption="Sentinel-1 (SAR) and Sentinel-2 (Optical) Satellites", use_column_width=True)
        
    elif app_mode == "Classify":
        st.title("📊 Multi-Sensor Data Classification")
        st.info("💡 You can upload files directly or provide URLs to files hosted elsewhere", icon="ℹ️")
        
        # Warning about file limits
        st.markdown("""
        <div class="warning-box" style="color:#2e7d32">
            <strong>Important:</strong> All files are limited to 200MB maximum size.
            For optimal performance, use datasets with &lt; 100,000 records.
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for separate uploaders
        col1, col2 = st.columns(2)
        
        # Initialize session state for file uploaders
        if 's2_uploaded' not in st.session_state:
            st.session_state.s2_uploaded = None
        if 's1_uploaded' not in st.session_state:
            st.session_state.s1_uploaded = None
        if 's2_processed' not in st.session_state:
            st.session_state.s2_processed = None
        if 's1_processed' not in st.session_state:
            st.session_state.s1_processed = None
        
        with col1:
            st.subheader("Sentinel-2 Data (Optical)")
            s2_option = st.radio("Choose input method:", ["Upload File", "Provide URL"], key="s2_option")
            
            if s2_option == "Upload File":
                sentinel2_file = st.file_uploader("Upload Sentinel-2 CSV", type="csv", key="s2_upload")
                st.markdown('<p class="custom-upload-limit">Max file size: 200MB</p>', unsafe_allow_html=True)
                
                if sentinel2_file:
                    # Only process if it's a new file
                    if st.session_state.s2_uploaded != sentinel2_file.name:
                        with st.spinner("Processing Sentinel-2 data..."):
                            df = handle_uploaded_file(sentinel2_file, "Sentinel-2")
                            if df is not None:
                                st.session_state.s2_uploaded = sentinel2_file.name
                                st.session_state.s2_processed = process_sentinel2(df)
                                st.success(f"Loaded {len(df)} rows from file: {sentinel2_file.name}")
                    else:
                        st.success(f"Using previously loaded file: {sentinel2_file.name}")
                elif st.session_state.s2_uploaded:
                    st.info(f"Previously loaded file: {st.session_state.s2_uploaded}")
            else:
                s2_url = st.text_input("Enter URL for Sentinel-2 CSV:", key="s2_url", 
                                      placeholder="https://drive.google.com/... or https://dropbox.com/...")
                if st.button("Load from URL", key="s2_load"):
                    if s2_url:
                        with st.spinner("Downloading Sentinel-2 data..."):
                            content = download_from_url(s2_url)
                        if content:
                            try:
                                # Save to temp file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                                    tmp_file.write(content)
                                    tmp_file_path = tmp_file.name
                                
                                # Process with robust reader
                                df = robust_read_csv(tmp_file_path)
                                if df is not None:
                                    st.session_state.s2_uploaded = s2_url
                                    st.session_state.s2_processed = process_sentinel2(df)
                                    st.success(f"Loaded {len(df)} rows from URL")
                                
                                # Clean up temp file
                                os.unlink(tmp_file_path)
                            except Exception as e:
                                st.error(f"Error processing downloaded file: {e}")
                    else:
                        st.warning("Please enter a valid URL")
        
        with col2:
            st.subheader("Sentinel-1 Data (SAR)")
            s1_option = st.radio("Choose input method:", ["Upload File", "Provide URL"], key="s1_option")
            
            if s1_option == "Upload File":
                sentinel1_file = st.file_uploader("Upload Sentinel-1 CSV", type="csv", key="s1_upload")
                st.markdown('<p class="custom-upload-limit">Max file size: 200MB</p>', unsafe_allow_html=True)
                
                if sentinel1_file:
                    # Only process if it's a new file
                    if st.session_state.s1_uploaded != sentinel1_file.name:
                        with st.spinner("Processing Sentinel-1 data..."):
                            df = handle_uploaded_file(sentinel1_file, "Sentinel-1")
                            if df is not None:
                                st.session_state.s1_uploaded = sentinel1_file.name
                                st.session_state.s1_processed = process_sentinel1(df)
                                st.success(f"Loaded {len(df)} rows from file: {sentinel1_file.name}")
                    else:
                        st.success(f"Using previously loaded file: {sentinel1_file.name}")
                elif st.session_state.s1_uploaded:
                    st.info(f"Previously loaded file: {st.session_state.s1_uploaded}")
            else:
                s1_url = st.text_input("Enter URL for Sentinel-1 CSV:", key="s1_url", 
                                      placeholder="https://drive.google.com/... or https://dropbox.com/...")
                if st.button("Load from URL", key="s1_load"):
                    if s1_url:
                        with st.spinner("Downloading Sentinel-1 data..."):
                            content = download_from_url(s1_url)
                        if content:
                            try:
                                # Save to temp file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                                    tmp_file.write(content)
                                    tmp_file_path = tmp_file.name
                                
                                # Process with robust reader
                                df = robust_read_csv(tmp_file_path)
                                if df is not None:
                                    st.session_state.s1_uploaded = s1_url
                                    st.session_state.s1_processed = process_sentinel1(df)
                                    st.success(f"Loaded {len(df)} rows from URL")
                                
                                # Clean up temp file
                                os.unlink(tmp_file_path)
                            except Exception as e:
                                st.error(f"Error processing downloaded file: {e}")
                    else:
                        st.warning("Please enter a valid URL")
        
        # Classification section
        st.divider()
        st.subheader("Classification")
        
        # Add reset button
        if st.button("Clear All Data"):
            st.session_state.s2_uploaded = None
            st.session_state.s1_uploaded = None
            st.session_state.s2_processed = None
            st.session_state.s1_processed = None
            st.experimental_rerun()
        
        # Check if both datasets are processed
        if st.session_state.s2_processed is not None and st.session_state.s1_processed is not None:
            try:
                # Show processed stats
                st.info(f"Processed Sentinel-2: {len(st.session_state.s2_processed)} unique IDs")
                st.info(f"Processed Sentinel-1: {len(st.session_state.s1_processed)} unique IDs")
                
                # Merge datasets on ID
                with st.spinner("Matching IDs..."):
                    merged_df = safe_merge(st.session_state.s2_processed, st.session_state.s1_processed)
                
                if merged_df is None or merged_df.empty:
                    st.error("No matching IDs found between Sentinel-1 and Sentinel-2 datasets")
                    return
                    
                st.success(f"Merged dataset: {len(merged_df)} records with matching IDs")
                
                if model:
                    # Get all available features (except ID)
                    available_features = [col for col in merged_df.columns if col != 'ID']
                    
                    # Handle NaN values
                    merged_df_filled = merged_df.fillna(0)
                    
                    # Make predictions in chunks
                    with st.spinner("Classifying cropland types..."):
                        chunk_size = 10000
                        predictions = []
                        
                        for i in range(0, len(merged_df_filled), chunk_size):
                            chunk = merged_df_filled[available_features].iloc[i:i+chunk_size]
                            try:
                                chunk_pred = model.predict(chunk)
                                predictions.extend(chunk_pred)
                            except Exception as e:
                                # Enhanced error reporting
                                st.error(f"Prediction error: {e}")
                                if hasattr(model, 'feature_names_in_'):
                                    st.error(f"Model expects features: {list(model.feature_names_in_)}")
                                st.error(f"We provided features: {available_features}")
                                return
                        
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
                st.error(f"❌ Processing error: {e}")
        elif st.session_state.s2_processed is not None or st.session_state.s1_processed is not None:
            st.warning("⚠️ Please load both Sentinel-1 and Sentinel-2 datasets")
            
    elif app_mode == "About":
        st.title("ℹ️ About")
        st.markdown("""
        ### Multi-Sensor Cropland Classification System
        
        **Enhanced File Handling:**
        - Advanced CSV parser with automatic delimiter detection
        - Malformed row skipping with detailed diagnostics
        - Character encoding auto-detection
        - Universal URL support (Google Drive, Dropbox, OneDrive, etc.)
        
        **Technical Specifications:**
        - Model: Random Forest (100 estimators)
        - Features: Exact feature engineering from training notebook
          - Vegetation indices: NDVI, NDWI, EVI, NDMI, ARVL, NBR
          - SAR features: VV, VH, VV/VH, VH/VV
          - Statistical aggregates: min, max, mean, median, std, 25th/75th percentiles
        - Processing: Matches notebook's workflow exactly
        - Error Handling: Detailed diagnostics for malformed CSVs
        
        **File Size Limits:**
        - Max 200MB per file for both upload and URL methods
        - Optimized for datasets with &lt; 100,000 records
        
        **Supported URL Formats:**
        - Google Drive: `https://drive.google.com/file/d/...`
        - Dropbox: `https://www.dropbox.com/s/...?dl=0`
        - OneDrive: `https://1drv.ms/...` or `https://onedrive.live.com/...`
        - Direct links: `https://example.com/data.csv`
        
        Developed for precision agriculture applications using multi-sensor fusion.
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()