pythonimport streamlit as st
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
    page_title="Chest X-Ray Diagnosis",
    page_icon="🫁",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════════
# DOWNLOAD & LOAD MODEL FROM GOOGLE DRIVE
# ══════════════════════════════════════════════════════════════════

MODEL_URL = "https://drive.google.com/uc?export=download&id=13CTLYdVj0fmyTgDndSHfudKFW4SkwFfz"
MODEL_PATH = "best_model.h5"

@st.cache_resource
def download_and_load_model():
    """تحميل النموذج من Google Drive (مرة واحدة فقط)"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading AI model... (first time only, ~30 MB)"):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Error downloading model: {e}")
                st.stop()
    
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = download_and_load_model()

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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model(
        model.input,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_gradcam_overlay(img, heatmap, alpha=0.4):
    """Create Grad-CAM overlay on image"""
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = heatmap * alpha + img * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype('uint8')
    return overlay

# ══════════════════════════════════════════════════════════════════
# UI DESIGN
# ══════════════════════════════════════════════════════════════════

st.title("🫁 Chest X-Ray Disease Classification")
st.markdown("### AI-Powered Medical Diagnosis Assistant")
st.markdown("---")

with st.sidebar:
    st.header("📋 About")
    st.markdown("""
    **Model:** DenseNet121  
    **Training Data:** 21,165 X-rays  
    **Accuracy:** 83.6%  
    **AUC:** 0.968  
    """)
    st.warning("⚠️ **Important:** This is an educational tool. Always consult qualified medical professionals for diagnosis.")
    
    st.markdown("---")
    
    st.header("🎯 Detectable Conditions")
    for i, cls in enumerate(CLASS_NAMES):
        st.markdown(f"**{i+1}.** {cls}")
    
    st.markdown("---")
    st.markdown("🎓 **Biomedical Engineering Project**")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload X-Ray Image")
    
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a frontal chest X-ray image (PNG, JPG, or JPEG)"
    )
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-Ray Image", use_container_width=True)
        
        analyze_btn = st.button("🔬 Analyze Image", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🔄 Analyzing image... Please wait..."):
                # Preprocess image
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                
                # Prediction
                predictions = model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]
                
                # Grad-CAM
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, predicted_class)
                img_np = np.array(img_resized)
                overlay = create_gradcam_overlay(img_np, heatmap)
                
                # Display results
                with col2:
                    st.header("📊 Analysis Results")
                    
                    st.subheader("🎯 Diagnosis")
                    
                    # Color based on confidence
                    if confidence > 0.9:
                        conf_color = "green"
                        conf_label = "High Confidence"
                    elif confidence > 0.7:
                        conf_color = "orange"
                        conf_label = "Medium Confidence"
                    else:
                        conf_color = "red"
                        conf_label = "Low Confidence"
                    
                    st.markdown(f"### :{conf_color}[{CLASS_NAMES[predicted_class]}]")
                    st.progress(float(confidence))
                    st.markdown(f"**Confidence:** {confidence*100:.2f}% ({conf_label})")
                    
                    st.markdown("---")
                    
                    st.subheader("📈 Probability Distribution")
                    for cls, prob in zip(CLASS_NAMES, predictions[0]):
                        st.markdown(f"**{cls}:** {prob*100:.2f}%")
                        st.progress(float(prob))
                    
                    st.markdown("---")
                    
                    st.subheader("🔬 Grad-CAM Visualization")
                    st.markdown("*Areas the AI model focused on for diagnosis:*")
                    st.image(overlay, caption="Model Attention Heatmap", use_container_width=True)
                    
                    st.info("""
                    **How to read Grad-CAM:**
                    - 🔴 **Red areas:** High attention (important for diagnosis)
                    - 🟡 **Yellow/Green:** Medium attention
                    - 🔵 **Blue areas:** Low attention
                    """)
                    
                    st.warning("""
                    ⚠️ **Medical Disclaimer:**  
                    This AI system is designed for educational and research purposes only. 
                    It should NOT be used as a substitute for professional medical diagnosis. 
                    Always consult qualified healthcare professionals for medical advice.
                    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎓 <b>Biomedical Engineering Graduation Project</b></p>
    <p>Built with Streamlit & TensorFlow | DenseNet121 Architecture</p>
</div>
""", unsafe_allow_html=True)
