import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import json
import os

# --- CONFIGURATION & STYLISATION ---

# 1. Définissez le chemin de votre logo ici
LOGO_PATH = r"C:\Users\HP PRO\Desktop\ChestXRay_Project\logo2"

# Aide pour trouver le logo (vérifie si c'est un fichier ou essaie les extensions courantes)
logo_image = None
if os.path.exists(LOGO_PATH):
    logo_image = LOGO_PATH
else:
    # Essaie les extensions d'image courantes
    for ext in ['.png', '.jpg', '.jpeg', '.ico']:
        if os.path.exists(LOGO_PATH + ext):
            logo_image = LOGO_PATH + ext
            break

# Configuration de la page - Utilise le logo comme icône s'il est trouvé
icon_to_use = logo_image if logo_image else "🫁"

st.set_page_config(
    page_title="MediScan AI | Diagnostic Thoracique",
    page_icon=icon_to_use,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour une interface médicale "Super"
st.markdown("""
    <style>
    /* Arrière-plan principal et polices */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* En-têtes personnalisés */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    
    /* Style de carte pour les conteneurs */
    .css-1r6slb0 {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #3b3b3b;
    }
    
    /* Style du téléchargeur de fichiers */
    .stFileUploader {
        background-color: #1f2937;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Style personnalisé des boutons */
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
    
    /* Barres de progression */
    .stProgress > div > div > div > div {
        background-color: #00c6ff;
    }
    
    /* Métriques */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00c6ff;
    }
    
    /* Style de la barre latérale */
    [data-testid="stSidebar"] {
        background-color: #0b0c10;
        border-right: 1px solid #333;
    }
    
    /* Style du logo dans la barre latérale */
    .sidebar-logo {
        display: flex;
        justify-content: center;
        padding: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- FONCTIONS ---

@st.cache_resource
def load_model_and_classes():
    """Charger le modèle entraîné et les mappings des classes"""
    try:
        model = load_model('best_model.h5')
        with open('class_mapping.json', 'r') as f:
            class_info = json.load(f)
            class_names = class_info['class_names']
        return model, class_names
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        st.stop()

def get_last_conv_layer(model):
    """Trouver la dernière couche convolutionnelle pour Grad-CAM"""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Générer une carte de chaleur Grad-CAM"""
    grad_model = tf.keras.models.Model(
        model.inputs, 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        # Gérer la sortie potentielle en liste des versions récentes de TF
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
    """Superposer la carte de chaleur sur l'image avec alpha dynamique"""
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint, cv2.COLORMAP_JET)
    
    img_uint = np.array(img).astype('uint8')
    
    # RVB vers BGR pour OpenCV
    if img_uint.shape[-1] == 3:
        img_bgr = cv2.cvtColor(img_uint, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_uint

    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- APPLICATION PRINCIPALE ---

# Charger les données
model, CLASS_NAMES = load_model_and_classes()
last_conv_layer = get_last_conv_layer(model)

# Barre latérale
with st.sidebar:
    # SECTION LOGO - Affiche l'image au lieu du texte
    if logo_image:
        try:
            st.image(logo_image, caption=None, use_container_width=True, clamp=True)
        except Exception as e:
            st.title("🫁 MediScan AI")  # Solution de secours si l'image échoue
    else:
        st.title("🫁 MediScan AI")  # Solution de secours si le fichier n'est pas trouvé
        
    st.markdown("---")
    st.header("Informations du Projet")
    st.markdown(f"**Modèle :** DenseNet121")
    st.markdown(f"**Classes :** {len(CLASS_NAMES)}")
    st.markdown(f"**Précision :** ~84%")
    
    st.markdown("---")
    st.header("Classes de Diagnostic")
    for cls in CLASS_NAMES:
        st.markdown(f"- {cls}")
    
    st.markdown("---")
    st.warning("⚠️ **Avertissement** : Cet outil est à but éducatif uniquement. Consultez toujours un spécialiste.")

# Titre principal
st.title("☣️ Classification d’images radiographiques par Deep Learning avec Grad-CAM ☣️")
st.markdown("Importez une radiographie pulmonaire pour générer un rapport diagnostique avec visualisation IA.")

# Mise en page : Entrée | Sortie
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("1. Importez votre image")
    uploaded_file = st.file_uploader("Glissez-déposez la radiographie ici", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Correction : conversion en RGB pour éviter le problème des 4 canaux
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="Radiographie d'entrée", use_container_width=True, clamp=True)
        
        analyze_btn = st.button("Analyser l'Image", use_container_width=True)
    else:
        st.info("Veuillez télécharger une image pour commencer l'analyse.")
        analyze_btn = False

# État de session pour l'interactivité
if 'heatmap' not in st.session_state:
    st.session_state['heatmap'] = None
    st.session_state['predictions'] = None
    st.session_state['original_img'] = None

with col2:
    st.subheader("2. Résultats du Diagnostic")
    
    if analyze_btn and uploaded_file:
        with st.spinner("Analyse en cours..."):
            # Prétraitement
            img_resized = image_pil.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # Prédiction
            preds = model.predict(img_array, verbose=0)
            pred_class = np.argmax(preds[0])
            confidence = preds[0][pred_class]
            
            # Grad-CAM
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
            
            # Stocker dans l'état de session
            st.session_state['predictions'] = preds
            st.session_state['pred_class'] = pred_class
            st.session_state['confidence'] = confidence
            st.session_state['heatmap'] = heatmap
            st.session_state['original_img'] = np.array(img_resized)

    # Afficher la zone des résultats
    if st.session_state['predictions'] is not None:
        
        # ONGLETS pour l'organisation
        tab1, tab2, tab3 = st.tabs(["Diagnostic", "Analyse Visuelle", "Statistiques"])
        
        with tab1:
            # Carte de diagnostic
            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                st.markdown(f"### Prédiction : **{CLASS_NAMES[st.session_state['pred_class']]}**")
                
                # Couleur dynamique pour la confiance
                if st.session_state['confidence'] > 0.85:
                    conf_msg = "Haute Confiance"
                    color_code = "#00ff88"
                    color_text = "green"
                elif st.session_state['confidence'] > 0.60:
                    conf_msg = "Confiance Modérée"
                    color_code = "#ffaa00"
                    color_text = "orange"
                else:
                    conf_msg = "Faible Confiance"
                    color_code = "#ff4444"
                    color_text = "red"
                
                st.markdown(f"Niveau de confiance : :{color_text}[{conf_msg}]")
                
            with res_col2:
                st.metric("Score", f"{st.session_state['confidence']*100:.1f}%")
            
            st.progress(float(st.session_state['confidence']))
            
            # Barres de probabilité
            st.markdown("#### Distribution des Probabilités")
            for cls, prob in zip(CLASS_NAMES, st.session_state['predictions'][0]):
                st.markdown(f"**{cls}**")
                st.progress(float(prob))

        with tab2:
            st.markdown("#### Grad-CAM Interactif")
            st.caption("Ajustez le curseur pour voir les zones sur lesquelles l'IA s'est concentrée.")
            
            # CURSEUR DYNAMIQUE
            alpha_slider = st.slider("Intensité de superposition", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
            
            overlay_img = create_gradcam_overlay(st.session_state['original_img'], st.session_state['heatmap'], alpha_slider)
            st.image(overlay_img, caption="Carte d'attention IA (Grad-CAM)", use_container_width=True, clamp=True)
            
            st.markdown("""
            - 🔴 **Rouge/Jaune** : Zones de haute attention.
            - 🔵 **Bleu** : Zones de faible attention.
            """)

        with tab3:
            st.markdown("#### Probabilités Détaillées")
            st.dataframe(
                data={
                    "Classe": CLASS_NAMES,
                    "Probabilité": [f"{p*100:.2f}%" for p in st.session_state['predictions'][0]]
                },
                hide_index=True,
                use_container_width=True
            )
            
    else:
        # État vide
        st.markdown("""
        <div style="text-align: center; color: #555; padding: 50px; border: 2px dashed #333; border-radius: 10px;">
            <h3>En attente d'entrée...</h3>
            <p>Importez une image et cliquez sur Analyser pour voir les résultats de l'IA.</p>
        </div>
        """, unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>🎓 Projet de Fin d'Études | Développé Badreddine AABANE", unsafe_allow_html=True)
