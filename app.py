import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

# Hide TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load model
model = load_model("best_cnn_model.h5")

# App title
st.set_page_config(page_title="Cropland Classifier", layout="centered")
st.title("🌾 Cropland Mapping Classifier")
st.markdown("""
This AI model helps identify cropland in dry regions (Fergana & Orenburg) using satellite data.

📂 Upload your **test CSV file** with time-series imagery features to get cropland predictions.

""")

# Upload CSV
uploaded_file = st.file_uploader("Upload your .csv test file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "ID" not in df.columns:
            st.error("⚠️ CSV must contain an 'ID' column.")
        else:
            features = df.drop(columns=["ID"])
            predictions = model.predict(features)
            predicted_classes = np.argmax(predictions, axis=1)

            output = pd.DataFrame({
                "ID": df["ID"],
                "Target": predicted_classes
            })

            st.success("✅ Prediction completed!")
            st.dataframe(output.head())

            # Download link
            csv = output.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Submission CSV", csv, "submission.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("Please upload a CSV file to begin.")

