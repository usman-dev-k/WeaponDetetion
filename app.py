import torch

# Load your model file
model_path = "yolov5s.pt"
model = torch.load(model_path, map_location=torch.device("cpu"))

# Check the class names
if "model" in model and hasattr(model["model"], "names"):
    print("✅ Model loaded successfully.")
    print("Classes:", model["model"].names)
else:
    print("❌ Model does not appear to have class names (not trained or corrupted).")
