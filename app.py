import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

# Hide TF logs and force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Streamlit page config
st.set_page_config(
    page_title="Cropland Classifier 🌾",
    page_icon="🌾",
    layout="centered"
)

# Inject working custom CSS
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1619026186627-1f9f2b3d066d");
        background-size: cover;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.90);
        padding: 2rem 2rem;
        border-radius: 1rem;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# App title and logo
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/65/UN_Logo.svg/320px-UN_Logo.svg.png", width=100)
st.title("🌾 Cropland Mapping Classifier")
st.markdown("""
Welcome to the Cropland Classifier!  
This tool uses a trained AI model to classify **cropland vs non-cropland** based on satellite time-series data.

### 📥 Instructions:
- Upload a `.csv` file with an **ID** column and feature columns.
- The model will predict whether each row is cropland (`1`) or not (`0`).
- Download a ready-to-submit `submission.csv`.

---
""")

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("best_cnn_model.h5")

model = load_trained_model()

# Upload section
uploaded_file = st.file_uploader("Upload your test CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if "ID" not in df.columns:
            st.error("❌ Your CSV must include an 'ID' column.")
        else:
            # Run prediction
            features = df.drop(columns=["ID"])
            preds = model.predict(features)
            predicted_classes = np.argmax(preds, axis=1)

            # Format output
            output = pd.DataFrame({
                "ID": df["ID"],
                "Target": predicted_classes
            })

            st.success("✅ Prediction complete!")
            st.dataframe(output.head())

            # Download button
            csv = output.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download submission.csv", csv, "submission.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
else:
    st.info("Please upload a test CSV file to begin.")


