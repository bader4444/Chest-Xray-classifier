import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import urllib.request
import os

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# OPTION 1: Use a URL for the icon (Browser Tab)
# Replace the URL below with your image path (e.g., "logo.png") if you have a local file
PAGE_ICON_URL = "https://img.icons8.com/color/96/000000/lungs.png" 

st.set_page_config(
    page_title="MediScan AI | Chest X-Ray Diagnosis",
    page_icon=PAGE_ICON_URL, 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical interface
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    
    div.stButton > button {
        background-image: linear-gradient(to right, #00c6ff, #0072ff);
        border: none;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0, 198, 255, 0.4);
    }
    
    .stProgress > div > div > div > div {
        background-color: #00c6ff;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00c6ff;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0b0c10;
        border-right: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# DOWNLOAD & LOAD MODEL FROM HUGGING FACE
# ══════════════════════════════════════════════════════════════════

MODEL_URL = "https://huggingface.co/BADER4444/chest-xray-model/resolve/main/best_model.h5"
MODEL_PATH = "best_model.h5"

@st.cache_resource
def download_and_load_model():
    """Download model from Hugging Face and load it"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading AI model... (first time only, ~31MB)"):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                st.success(" ")
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                st.info("Please check your internet connection and try again.")
                return None
    
    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = download_and_load_model()

if model is None:
    st.error("❌ Failed to load model. Please refresh the page.")
    st.stop()

# Class names
CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']

# Find last conv layer for Grad-CAM
last_conv_layer = None
for layer in reversed(model.layers):
    if 'conv' in layer.name.lower():
        last_conv_layer = layer.name
        break

# ══════════════════════════════════════════════════════════════════
# GRAD-CAM FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generate Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        if isinstance(predictions, list):
            predictions = predictions[0]
        
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def create_gradcam_overlay(img, heatmap, alpha):
    """Overlay heatmap on image"""
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint, cv2.COLORMAP_JET)
    
    img_uint = np.array(img).astype('uint8')
    
    if img_uint.shape[-1] == 3:
        img_bgr = cv2.cvtColor(img_uint, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_uint
    
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════

# Sidebar
with st.sidebar:
    #
