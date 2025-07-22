import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("best_cnn_model.h5")

# Page config
st.set_page_config(page_title="Cropland Mapping AI", layout="centered")

st.title("🌾 Cropland Mapping in Dry Regions")
st.write("Upload a preprocessed CSV file and get cropland predictions (0 = non-cropland, 1 = cropland).")

# File uploader
uploaded_file = st.file_uploader("Upload your feature CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load uploaded CSV
        data = pd.read_csv(uploaded_file)

        # Assume first column is ID (optional, remove if not needed)
        if 'ID' in data.columns:
            ids = data['ID']
            X = data.drop(columns=['ID'])
        else:
            ids = [f"ID_{i}" for i in range(len(data))]
            X = data

        # Reshape for Conv1D input: (samples, timesteps, features)
        X_reshaped = np.expand_dims(X, axis=2) if len(X.shape) == 2 else X

        # Make predictions
        predictions = model.predict(X_reshaped)
        binary_preds = (predictions > 0.5).astype(int).flatten()

        # Create submission file
        submission_df = pd.DataFrame({
            "ID": ids,
            "Target": binary_preds
        })

        st.subheader("🧾 Predictions Preview")
        st.dataframe(submission_df.head())

        # Download button
        csv = submission_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Submission CSV", csv, "submission.csv", "text/csv")

    except Exception as e:
        st.error(f"Something went wrong while processing the file: {e}")
else:
    st.info("Please upload a valid CSV file.")
