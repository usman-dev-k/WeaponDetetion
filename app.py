import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import subprocess
from pathlib import Path

# Download YOLOv5 code if not already present
YOLO_DIR = Path("yolov5")
if not YOLO_DIR.exists():
    with st.spinner("🔄 Cloning YOLOv5..."):
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5"], check=True)
        subprocess.run(["pip", "install", "-r", "yolov5/requirements.txt"], check=True)

# Load the model
@st.cache_resource
def load_model():
    model = torch.hub.load("yolov5", "custom", path="yolov5.pt", source="local")
    return model

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Weapon Detection", layout="centered")
st.title("🔍 Weapon Detection with YOLOv5")
st.write("Upload an image to check if it contains a weapon using your custom YOLOv5 model.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running detection..."):
        results = model(np.array(image))

    results.render()
    st.image(results.ims[0], caption="Detection Result", use_column_width=True)

    detected = results.pandas().xyxy[0]
    detected_classes = detected["name"].tolist()

    if detected.empty:
        st.success("✅ No objects detected.")
    else:
        st.info(f"Detected: {', '.join(set(detected_classes))}")

        weapon_keywords = ["weapon", "gun", "knife", "pistol", "rifle"]
        if any(cls.lower() in weapon_keywords for cls in detected_classes):
            st.error("🚨 Weapon detected!")
        else:
            st.success("✅ No weapon detected.")
