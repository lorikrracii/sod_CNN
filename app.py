import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import sys
import time

# ------------------------------------------------------
# Custom CSS for better fonts
# ------------------------------------------------------
st.markdown("""
    <style>
        /* Sidebar section title */
        .sidebar-title {
            font-size: 1.2rem !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            padding-top: 10px !important;
            padding-bottom: 5px !important;
        }

        /* Metric label */
        .metric-label {
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: #d0d0d0 !important;
        }

        /* Metric value style */
        .metric-value {
            font-size: 1.4rem !important;
            font-weight: 800 !important;
            color: #00ff88 !important;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------------
# Add src/ to PYTHONPATH
# ------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# ------------------------------------------------------
# Import project modules
# ------------------------------------------------------
import config
from sod_model import SODModel

# ------------------------------------------------------
# Device
# ------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------
# Load model
# ------------------------------------------------------
@st.cache_resource
def load_model():
    model = SODModel().to(device)
    checkpoint = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# ------------------------------------------------------
# Preprocessing
# ------------------------------------------------------
IMAGE_SIZE = config.IMAGE_SIZE

def preprocess(img_pil):
    img = np.array(img_pil)
    original = img.copy()

    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))

    tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    return original, tensor

# ------------------------------------------------------
# Postprocessing
# ------------------------------------------------------
def postprocess(original, prob, threshold):
    H, W = original.shape[:2]

    prob_resized = cv2.resize(prob, (W, H))
    binary = (prob_resized >= threshold).astype(np.float32)

    overlay = original.astype(np.float32) / 255.0
    green = np.zeros_like(overlay)
    green[..., 1] = binary

    blended = 0.5 * overlay + 0.5 * green
    blended = (blended * 255).astype(np.uint8)

    return prob_resized, binary, blended

# ------------------------------------------------------
# Streamlit UI Setup
# ------------------------------------------------------
st.set_page_config(
    page_title="Salient Object Detection",

    layout="wide"
)

st.title("Salient Object Detection")
st.markdown("Upload an image and view **all results side-by-side** below.")

# ------------------------------------------------------
# Sidebar
# ------------------------------------------------------
st.sidebar.header("⚙️ Controls")

uploaded = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

threshold = st.sidebar.slider(
    "Mask Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

st.sidebar.markdown('<p class="sidebar-title">📊 Model Performance</p>', unsafe_allow_html=True)

st.sidebar.markdown("""
    <p class="metric-label">IoU</p>
    <p class="metric-value">0.6423</p>

    <p class="metric-label">Precision</p>
    <p class="metric-value">0.7863</p>

    <p class="metric-label">Recall</p>
    <p class="metric-value">0.7866</p>

    <p class="metric-label">F1-Score</p>
    <p class="metric-value">0.7596</p>

    <p class="metric-label">MAE</p>
    <p class="metric-value">0.1077</p>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------
if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")

    st.subheader("📸 Uploaded Image")
    st.image(img_pil, use_container_width=True)

    with st.spinner("Running model..."):
        original, tensor = preprocess(img_pil)

        start = time.time()
        with torch.no_grad():
            pred = model(tensor)
        end = time.time()

        inference_time = (end - start) * 1000
        prob = pred.squeeze().cpu().numpy()

        prob_map, binary_mask, overlay = postprocess(original, prob, threshold)

    st.success(f"Inference completed in **{inference_time:.2f} ms**")

    # --------------------------------------------------
    # SHOW ALL RESULTS SIDE-BY-SIDE IN ONE ROW
    # --------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.caption("📷 Original")
        st.image(original, use_container_width=True)

    with col2:
        st.caption("🧠 Saliency Map")
        st.image(prob_map, clamp=True, use_container_width=True)

    with col3:
        st.caption("🟩 Binary Mask")
        st.image(binary_mask, clamp=True, use_container_width=True)

    with col4:
        st.caption("🎨 Overlay")
        st.image(overlay, use_container_width=True)

else:
    st.info("📂 Upload an image from the sidebar to begin.")
