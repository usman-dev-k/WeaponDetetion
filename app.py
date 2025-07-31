import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov5.pt")  # Or "yolov8.pt" if you're using YOLOv8

model = load_model()

st.set_page_config(page_title="Weapon Detection")
st.title("🔍 Weapon Detection App")
st.write("Upload an image to detect weapons using a YOLO model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    with st.spinner("Detecting..."):
        results = model.predict(source=np.array(image), save=False)

    # Draw results
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Detection Result", use_column_width=True)

    # Get detected class names
    names = model.names
    detections = results[0].boxes.cls.tolist()
    detected_labels = [names[int(i)] for i in detections]

    if not detected_labels:
        st.success("✅ No objects detected.")
    else:
        st.info(f"Detected: {', '.join(set(detected_labels))}")
        if any(lbl.lower() in ["weapon", "gun", "knife", "pistol", "rifle"] for lbl in detected_labels):
            st.error("🚨 Weapon Detected!")
        else:
            st.success("✅ No weapon detected.")
