# app.py
try:
    import pydeck
    import geopandas
    GEOSPATIAL_ENABLED = True
except ImportError:
    GEOSPATIAL_ENABLED = False
    st.warning("Some geospatial features disabled due to missing dependencies")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deployment_utils import load_models, preprocess_new_data
import geopandas as gpd
import leafmap.foliumap as leafmap

# App title and description
st.title("GeoAI Cropland Mapping")
st.markdown("""
This application predicts cropland areas using Sentinel-1 and Sentinel-2 satellite data.
Upload your data or use the demo data to get predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Home", "Data Upload", "Model Prediction", "Results Visualization"])

# Load models once (cached for performance)
@st.cache_resource
def load_ml_models():
    return load_models()

cnn_model, rf_model = load_ml_models()

if options == "Home":
    st.header("About the Project")
    st.write("""
    This tool implements a machine learning pipeline for cropland mapping using:
    - Sentinel-1 SAR data
    - Sentinel-2 optical data
    """)
    
    st.subheader("Model Architecture")
    st.image("model_architecture.png", use_column_width=True)
    
elif options == "Data Upload":
    st.header("Upload Satellite Data")
    
    # File uploaders
    st.subheader("Sentinel-1 Data")
    s1_file = st.file_uploader("Upload Sentinel-1 CSV", type=['csv'])
    
    st.subheader("Sentinel-2 Data")
    s2_file = st.file_uploader("Upload Sentinel-2 CSV", type=['csv'])
    
    if s1_file and s2_file:
        try:
            s1_df = pd.read_csv(s1_file)
            s2_df = pd.read_csv(s2_file)
            
            st.success("Files uploaded successfully!")
            st.subheader("Data Preview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Sentinel-1 Data (First 5 rows)")
                st.dataframe(s1_df.head())
            
            with col2:
                st.write("Sentinel-2 Data (First 5 rows)")
                st.dataframe(s2_df.head())
                
        except Exception as e:
            st.error(f"Error reading files: {str(e)}")

elif options == "Model Prediction":
    st.header("Run Predictions")
    
    # Demo data or custom upload
    prediction_option = st.radio("Choose prediction data:",
                               ["Use demo data", "Use uploaded data"])
    
    if prediction_option == "Use demo data":
        # Load sample data
        try:
            s1_sample = pd.read_csv("sample_sentinel1.csv")
            s2_sample = pd.read_csv("sample_sentinel2.csv")
            
            st.write("Using demo data for prediction")
            
            if st.button("Run Prediction"):
                with st.spinner("Processing data and making predictions..."):
                    # Preprocess data
                    processed_data = preprocess_new_data(s1_sample, s2_sample)
                    
                    # Make predictions
                    cnn_pred = cnn_model.predict(processed_data)
                    rf_pred = rf_model.predict(processed_data)
                    
                    # Store results in session state
                    st.session_state['predictions'] = {
                        'cnn': cnn_pred,
                        'rf': rf_pred,
                        'data': processed_data
                    }
                    
                    st.success("Predictions completed!")
                    
        except Exception as e:
            st.error(f"Error loading demo data: {str(e)}")
    
    elif prediction_option == "Use uploaded data":
        if 's1_df' in st.session_state and 's2_df' in st.session_state:
            if st.button("Run Prediction"):
                with st.spinner("Processing data and making predictions..."):
                    try:
                        # Preprocess data
                        processed_data = preprocess_new_data(
                            st.session_state.s1_df, 
                            st.session_state.s2_df
                        )
                        
                        # Make predictions
                        cnn_pred = cnn_model.predict(processed_data)
                        rf_pred = rf_model.predict(processed_data)
                        
                        # Store results in session state
                        st.session_state['predictions'] = {
                            'cnn': cnn_pred,
                            'rf': rf_pred,
                            'data': processed_data
                        }
                        
                        st.success("Predictions completed!")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
        else:
            st.warning("Please upload data first on the Data Upload page")

elif options == "Results Visualization":
    st.header("Prediction Results")
    
    if 'predictions' not in st.session_state:
        st.warning("Please run predictions first on the Model Prediction page")
    else:
        predictions = st.session_state.predictions
        
        st.subheader("Model Agreement")
        agreement = np.mean(predictions['cnn'] == predictions['rf'])
        st.write(f"CNN and Random Forest agree on {agreement*100:.2f}% of predictions")
        
        # Show sample predictions
        st.subheader("Sample Predictions")
        sample_results = pd.DataFrame({
            'CNN Prediction': predictions['cnn'][:10].flatten(),
            'RF Prediction': predictions['rf'][:10]
        })
        st.dataframe(sample_results)
        
        # Plot feature importance (example)
        st.subheader("Feature Importance")
        fig, ax = plt.subplots()
        # Add your feature importance plotting code here
        ax.barh(range(10), np.random.rand(10))  # Replace with actual feature importance
        ax.set_yticks(range(10))
        ax.set_yticklabels([f"Feature {i}" for i in range(10)])
        st.pyplot(fig)
        
        # Map visualization
        st.subheader("Geospatial Visualization")
        st.write("Cropland prediction map")
        
        # Create a simple map (replace with your actual geospatial data)
        m = leafmap.Map(center=[40, 70], zoom=6)
        m.add_basemap("SATELLITE")
        
        # Add sample points (replace with your actual points)
        sample_points = pd.DataFrame({
            'lat': np.random.uniform(39, 42, 50),
            'lon': np.random.uniform(69, 73, 50),
            'prediction': np.random.randint(0, 2, 50)
        })
        
        m.add_points_from_xy(
            sample_points,
            x="lon",
            y="lat",
            color_column="prediction",
            colors=["red", "green"],
            color_map=None,
            icon_names=["times", "check"],
            spin=True,
            add_legend=True,
        )
        
        st.pydeck_chart(m.to_streamlit())
