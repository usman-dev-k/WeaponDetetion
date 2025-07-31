import streamlit as st
import torch
from PIL import Image
import numpy as np

# Set page title and layout
st.set_page_config(page_title="Weapon Detection", layout="centered")
st.title("🔍 Weapon Detection with YOLOv5")
st.write("Upload an image to check if it contains a weapon using your custom YOLOv5 model.")

# Load the YOLOv5 model (cached so it doesn't reload every time)
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5.pt', force_reload=False)
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run detection
    with st.spinner("Running detection..."):
        results = model(np.array(image))

    # Draw boxes on image
    results.render()
    st.image(results.ims[0], caption="Detection Result", use_column_width=True)

    # Extract detected class names
    detected = results.pandas().xyxy[0]
    detected_classes = detected['name'].tolist()

    # Show results
    if detected.empty:
        st.success("✅ No objects detected.")
    else:
        st.info(f"Detected objects: {', '.join(set(detected_classes))}")

        # Weapon check
        weapon_keywords = ['weapon', 'gun', 'knife', 'pistol', 'rifle']
        if any(cls.lower() in weapon_keywords for cls in detected_classes):
            st.error("🚨 Weapon detected in the image!")
        else:
            st.success("✅ No weapon detected.")
