import streamlit as st
import torch
import numpy as np
from PIL import Image

st.set_page_config(page_title="Weapon Detection", layout="centered")
st.title("🔍 Weapon Detection with YOLOv5")
st.write("Upload an image to detect weapons using your custom YOLOv5 model.")

# Load YOLOv5 model from torch.hub (automatically downloads repo)
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5.pt')

model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running detection..."):
        results = model(np.array(image))

    results.render()
    st.image(results.ims[0], caption="Detection Results", use_column_width=True)

    detected = results.pandas().xyxy[0]
    classes = detected['name'].tolist()

    if detected.empty:
        st.success("✅ No objects detected.")
    else:
        st.info(f"Detected: {', '.join(set(classes))}")
        if any(c.lower() in ['gun', 'knife', 'weapon', 'pistol', 'rifle'] for c in classes):
            st.error("🚨 Weapon Detected!")
        else:
            st.success("✅ No weapon detected.")
