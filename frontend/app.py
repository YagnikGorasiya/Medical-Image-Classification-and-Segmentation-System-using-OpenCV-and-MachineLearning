import os
import streamlit as st
import requests

API = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Medical CV System", layout="wide")

st.title("🩺 Medical Image Classification & Segmentation System")
st.markdown("### OpenCV + Deep Learning (Chest, Brain, Skin)")

# -------------------------------------------------------
# MODULE SELECTION
# -------------------------------------------------------
module = st.sidebar.selectbox(
    "Select Medical Module",
    ["chest", "brain", "skin"]
)

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Classification (CNN)", "Image Processing", "Segmentation", "Feature Detection"]
)

uploaded = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

# -------------------------------------------------------
# CLASSIFICATION
# -------------------------------------------------------
if uploaded and mode == "Classification (CNN)":
    col1.image(uploaded, caption="Input Image", use_column_width=True)

    if st.button("Predict"):
        r = requests.post(
            f"{API}/predict/cnn",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            data={"module": module}
        )

        if r.status_code == 200:
            result = r.json()
            col2.success(f"Prediction: {result['predicted_class']}")
            col2.write(f"Confidence: {result['confidence']:.4f}")
        else:
            col2.error("Prediction failed")
            col2.write(r.text)

# -------------------------------------------------------
# IMAGE PROCESSING
# -------------------------------------------------------
elif uploaded and mode == "Image Processing":
    col1.image(uploaded, caption="Input Image", use_column_width=True)

    task = st.selectbox(
        "Select Processing",
        ["hist_eq", "gaussian", "canny"]
    )

    if st.button("Run Processing"):
        r = requests.post(
            f"{API}/process/image",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            data={"task": task}
        )

        if r.status_code == 200:
            col2.image(r.content, caption="Processed Output", use_column_width=True)
        else:
            col2.error("Processing failed")
            col2.write(r.text)

# -------------------------------------------------------
# SEGMENTATION
# -------------------------------------------------------
elif uploaded and mode == "Segmentation":
    col1.image(uploaded, caption="Input Image", use_column_width=True)

    task = st.selectbox(
        "Select Segmentation",
        ["threshold", "watershed"]
    )

    if st.button("Run Segmentation"):
        r = requests.post(
            f"{API}/process/image",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            data={"task": task}
        )

        if r.status_code == 200:
            col2.image(r.content, caption="Segmented Output", use_column_width=True)
        else:
            col2.error("Segmentation failed")
            col2.write(r.text)

# -------------------------------------------------------
# FEATURE DETECTION
# -------------------------------------------------------
elif uploaded and mode == "Feature Detection":
    col1.image(uploaded, caption="Input Image", use_column_width=True)

    task = st.selectbox(
        "Select Feature Detection",
        ["harris"]
    )

    if st.button("Run Feature Detection"):
        r = requests.post(
            f"{API}/process/image",
            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            data={"task": task}
        )

        if r.status_code == 200:
            col2.image(r.content, caption="Feature Output", use_column_width=True)
        else:
            col2.error("Feature detection failed")
            col2.write(r.text)
