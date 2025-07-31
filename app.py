import streamlit as st
import torch

@st.cache_resource
def load_model():
    model_path = 'yolov5s.pt'

    try:
        # Try loading with default (safe) config first
        model = torch.load(model_path)
    except Exception as e:
        st.warning("Safe loading failed. Attempting full load (trusted only)...")
        # Only do this if you trust the source
        model = torch.load(model_path, weights_only=False)

    if "model" in model and hasattr(model["model"], "names"):
        st.success("Model loaded successfully.")
        st.write("Classes:", model["model"].names)
        return model
    else:
        st.error("Model does not contain 'names'. It may not be trained correctly.")
        return None

model = load_model()
