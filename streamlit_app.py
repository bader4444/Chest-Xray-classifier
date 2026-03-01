import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import gdown
import os

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MediScan AI | Chest X-Ray Diagnosis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
# DOWNLOAD & LOAD MODEL
# ══════════════════════════════════════════════════════════════════

MODEL_URL = "https://drive.google.com/uc?export=download&id=13CTLYdVj0fmyTgDndSHfudKFW4SkwFfz"
MODEL_PATH = "best_model.h5"

@st.cache_resource
def download_and_load_model():
    """Download model from Google Drive and load it"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading AI model... (first time only, ~31MB)"):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None
    
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = download_and_load_model()

if model is None:
    st.error("❌ Failed to load model. Please refresh the page.")
    st.stop()

CLASS_NAMES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral_Pneumonia']

last_conv_layer = None
for layer in reversed(model.layers):
    if 'conv' in layer.name.lower():
        last_conv_layer = layer.name
        break

# ══════════════════════════════════════════════════════════════════
# GRAD-CAM FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
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

with st.sidebar:
    st.title("🫁 MediScan AI")
    st.markdown("---")
    
    st.header("Project Information")
    st.markdown("**Model:** DenseNet121")
    st.markdown(f"**Classes:** {len(CLASS_NAMES)}")
    st.markdown("**Accuracy:** ~84%")
    st.markdown("**AUC:** 0.97")
    
    st.markdown("---")
    st.header("Diagnostic Classes")
    for cls in CLASS_NAMES:
        st.markdown(f"- {cls}")
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** Educational tool only. Always consult medical professionals.")

st.title("☣️ AI-Powered Chest X-Ray Classification with Grad-CAM ☣️")
st.markdown("Upload a chest X-ray to generate an AI diagnostic report with visual explanation.")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("1. Upload Your Image")
    uploaded_file = st.file_uploader("Drag and drop X-ray here", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Input X-Ray", use_container_width=True)
        analyze_btn = st.button("🔬 Analyze Image", use_container_width=True)
    else:
        st.info("Please upload an image to start analysis.")
        analyze_btn = False

if 'heatmap' not in st.session_state:
    st.session_state['heatmap'] = None
    st.session_state['predictions'] = None
    st.session_state['original_img'] = None

with col2:
    st.subheader("2. Diagnostic Results")
    
    if analyze_btn and uploaded_file:
        with st.spinner("🔬 Analyzing..."):
            try:
                img_resized = image_pil.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                preds = model.predict(img_array, verbose=0)
                pred_class = np.argmax(preds[0])
                confidence = preds[0][pred_class]
                
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
                
                st.session_state['predictions'] = preds
                st.session_state['pred_class'] = pred_class
                st.session_state['confidence'] = confidence
                st.session_state['heatmap'] = heatmap
                st.session_state['original_img'] = np.array(img_resized)
            
            except Exception as e:
                st.error(f"Error during analysis: {e}")

    if st.session_state['predictions'] is not None:
        
        tab1, tab2, tab3 = st.tabs(["Diagnosis", "Visual Analysis", "Statistics"])
        
        with tab1:
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                st.markdown(f"### Prediction: **{CLASS_NAMES[st.session_state['pred_class']]}**")
                
                if st.session_state['confidence'] > 0.85:
                    conf_msg = "High Confidence"
                    color_text = "green"
                elif st.session_state['confidence'] > 0.60:
                    conf_msg = "Moderate Confidence"
                    color_text = "orange"
                else:
                    conf_msg = "Low Confidence"
                    color_text = "red"
                
                st.markdown(f"Confidence Level: :{color_text}[{conf_msg}]")
            
            with res_col2:
                st.metric("Score", f"{st.session_state['confidence']*100:.1f}%")
            
            st.progress(float(st.session_state['confidence']))
            
            st.markdown("#### Probability Distribution")
            for cls, prob in zip(CLASS_NAMES, st.session_state['predictions'][0]):
                st.markdown(f"**{cls}**")
                st.progress(float(prob))

        with tab2:
            st.markdown("#### Interactive Grad-CAM")
            st.caption("Adjust slider to see areas where AI focused.")
            
            alpha_slider = st.slider("Overlay Intensity", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
            
            overlay_img = create_gradcam_overlay(
                st.session_state['original_img'], 
                st.session_state['heatmap'], 
                alpha_slider
            )
            st.image(overlay_img, caption="AI Attention Map (Grad-CAM)", use_container_width=True)
            
            st.markdown("""
            - 🔴 **Red/Yellow**: High attention areas
            - 🔵 **Blue**: Low attention areas
            """)

        with tab3:
            st.markdown("#### Detailed Probabilities")
            st.dataframe(
                data={
                    "Class": CLASS_NAMES,
                    "Probability": [f"{p*100:.2f}%" for p in st.session_state['predictions'][0]]
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.markdown("""
        <div style="text-align: center; color: #555; padding: 50px; border: 2px dashed #333; border-radius: 10px;">
            <h3>Waiting for input...</h3>
            <p>Upload an image and click Analyze to see AI results.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>🎓 Graduation Project | Developed by Badreddine AABANE</div>",
    unsafe_allow_html=True
)
