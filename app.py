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
        return joblib.load('random_forest_cropland.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure 'random_forest_cropland.pkl' is in the same directory")
        return None

# ==============================
# IMPROVED DOWNLOAD HANDLER
# ==============================
def download_from_url(url, max_size=2*1024*1024*1024):
    """Download file from any source with special handling for cloud storage"""
    try:
        # Handle Google Drive URLs
        if "drive.google.com" in url:
            pattern = r'https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)/.*'
            match = re.match(pattern, url)
            if match:
                file_id = match.group(1)
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Handle Dropbox URLs
        if "dropbox.com" in url:
            if "?dl=0" in url:
                url = url.replace("?dl=0", "?dl=1")
        
        # Handle OneDrive URLs
        if "onedrive.live.com" in url or "1drv.ms" in url:
            if "1drv.ms" in url:
                url = requests.head(url, allow_redirects=True).url
            if "?resid=" in url and "!download" not in url:
                url = url.split('?')[0] + "?download=1"
        
        # Create session to handle redirects
        session = requests.Session()
        response = session.head(url, allow_redirects=True)
        
        # Get final URL after redirects
        final_url = response.url
        
        # Get file size
        file_size = int(response.headers.get('Content-Length', 0))
        
        # Handle unknown file size
        if file_size == 0:
            st.warning("Could not determine file size. Downloading anyway...")
            response = session.get(url, stream=True)
            content = bytearray()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content.extend(chunk)
            return content
        
        # Check file size limit
        if file_size > max_size:
            st.error(f"File size exceeds 2GB limit ({file_size/(1024**3):.2f} GB)")
            return None
        
        # Download with progress
        st.info(f"Downloading file from URL... (Size: {file_size/(1024**2):.2f} MB)")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        response = session.get(url, stream=True)
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
            dialect = sniffer.sniff("\n".join(first_lines))
        
        # Read with error-tolerant method
        data = []
        bad_lines = []
        total_lines = 0
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            reader = csv.reader(f, dialect)
            
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
    """Process uploaded file with temp storage for large files"""
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
# OPTICAL DATA PROCESSING
# =============================
def process_sentinel2(df):
    """Process Sentinel-2 data and calculate vegetation indices"""
    eps = 1e-8
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
    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + eps)
    df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + eps)
    df['EVI'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 6*df['B4'] - 7.5*df['B2'] + 1 + eps)
    df['NDMI'] = (df['B8'] - df['B11']) / (df['B8'] + df['B11'] + eps)
    df['NDVIre'] = (df['B8A'] - df['B5']) / (df['B8A'] + df['B5'] + eps)
    df['NBR'] = (df['B8'] - df['B12']) / (df['B8'] + df['B12'] + eps)
    
    # Return only essential features
    return df[['ID', 'NDVI', 'NDWI', 'EVI', 'NDMI', 'NDVIre', 'NBR']]

# =============================
# SAR DATA PROCESSING
# =============================
def process_sentinel1(df):
    """Process Sentinel-1 SAR data and calculate ratios"""
    eps = 1e-8
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
    df['VV/VH'] = df['VV'] / (df['VH'] + eps)
    df['VH/VV'] = df['VH'] / (df['VV'] + eps)
    
    # Return only essential features
    return df[['ID', 'VV', 'VH', 'VV/VH', 'VH/VV']]

# =============================
# MAIN APP
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
        st.info("💡 You can upload files directly or provide URLs to large files hosted elsewhere", icon="ℹ️")
        
        # Warning about large files
        st.markdown("""
        <div class="warning-box">
            <strong>Important:</strong> For files over 200MB, use the URL method. 
            Streamlit limits direct uploads to 200MB, but our URL downloader supports files up to 2GB.
        </div>
        """, unsafe_allow_html=True)
        
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
                st.markdown('<p class="custom-upload-limit">Max file size: 200MB (for direct upload). For larger files use URL method.</p>', unsafe_allow_html=True)
                
                if sentinel2_file:
                    # Process file with memory-efficient method
                    with st.spinner("Processing Sentinel-2 data..."):
                        st.session_state.s2_data = handle_uploaded_file(sentinel2_file, "Sentinel-2")
                    if st.session_state.s2_data is not None:
                        st.success(f"Loaded {len(st.session_state.s2_data)} rows")
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
                                st.session_state.s2_data = robust_read_csv(tmp_file_path)
                                if st.session_state.s2_data is not None:
                                    st.success(f"Loaded {len(st.session_state.s2_data)} rows from URL")
                                
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
                st.markdown('<p class="custom-upload-limit">Max file size: 200MB (for direct upload). For larger files use URL method.</p>', unsafe_allow_html=True)
                
                if sentinel1_file:
                    # Process file with memory-efficient method
                    with st.spinner("Processing Sentinel-1 data..."):
                        st.session_state.s1_data = handle_uploaded_file(sentinel1_file, "Sentinel-1")
                    if st.session_state.s1_data is not None:
                        st.success(f"Loaded {len(st.session_state.s1_data)} rows")
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
                                st.session_state.s1_data = robust_read_csv(tmp_file_path)
                                if st.session_state.s1_data is not None:
                                    st.success(f"Loaded {len(st.session_state.s1_data)} rows from URL")
                                
                                # Clean up temp file
                                os.unlink(tmp_file_path)
                            except Exception as e:
                                st.error(f"Error processing downloaded file: {e}")
                    else:
                        st.warning("Please enter a valid URL")
        
        # Classification section
        st.divider()
        st.subheader("Classification")
        
        # Check if both datasets are loaded
        if st.session_state.s2_data is not None and st.session_state.s1_data is not None:
            try:
                # Process data
                with st.spinner("Processing optical data..."):
                    s2_processed = process_sentinel2(st.session_state.s2_data)
                with st.spinner("Processing SAR data..."):
                    s1_processed = process_sentinel1(st.session_state.s1_data)
                
                # Check if processing succeeded
                if s2_processed is None or s1_processed is None:
                    st.error("Failed to process one or more datasets")
                    return
                
                # Merge datasets on ID
                with st.spinner("Matching IDs..."):
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
                        # Process in chunks for large datasets
                        chunk_size = 10000
                        predictions = []
                        
                        for i in range(0, len(X), chunk_size):
                            chunk = X[i:i+chunk_size]
                            chunk_pred = model.predict(chunk)
                            predictions.extend(chunk_pred)
                        
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
        
        **Enhanced File Handling:**
        - Advanced CSV parser with automatic delimiter detection
        - Malformed row skipping with detailed diagnostics
        - Character encoding auto-detection
        - Universal URL support (Google Drive, Dropbox, OneDrive, etc.)
        
        **Technical Specifications:**
        - Model: Random Forest (100 estimators)
        - Features: 6 vegetation indices + 4 SAR features
        - Processing: Chunk-based processing for large files
        - Error Handling: Detailed diagnostics for malformed CSVs
        
        **Supported URL Formats:**
        - Google Drive: `https://drive.google.com/file/d/...`
        - Dropbox: `https://www.dropbox.com/s/...?dl=0`
        - OneDrive: `https://1drv.ms/...` or `https://onedrive.live.com/...`
        - Direct links: `https://example.com/data.csv`
        
        **Direct Upload Limits:**
        - Max 200MB per file (Streamlit limitation)
        - Use URL method for larger files (up to 2GB)
        
        Developed for precision agriculture applications using multi-sensor fusion.
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()