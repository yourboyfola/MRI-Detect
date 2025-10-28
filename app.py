import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
import numpy as np
from PIL import Image

MODEL_FILENAME = 'brain_tumor_final.keras'

# Page config
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("""
This AI model classifies brain MRI scans into four categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **No Tumor**
- **Pituitary Tumor**

Upload an MRI scan to get started!
""")

# Tumor type labels
tumor_types = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

@st.cache_resource
def load_keras_model():
    """Loads the Keras model from the .keras file."""
    try:
        model = load_model(MODEL_FILENAME)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_FILENAME}: {e}")
        st.error("⚠️ Model not found! Ensure 'brain_tumor_final.keras' is in the same directory.")
        return None

def predict_tumor(image, model):
    """Make prediction on uploaded image"""
    img = image.resize((224, 224))
    img = img.convert('RGB')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) 
    
    prediction = model.predict(img_array, verbose=0) 
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    
    return predicted_class, confidence, prediction[0]

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Model Details:**
    - Architecture: **MobileNetV2** (Transfer Learning)
    - Classification: 4-class (Glioma, Meningioma, No Tumor, Pituitary)
    - Validation Accuracy: **90+%**
    
    **Developer:** Folarin Eribake
    **GitHub:** https://github.com/yourboyfola
    """)
    
    st.header("⚠️ Disclaimer")
    st.warning("""
    This is a demo model for **educational purposes only**. 
    NOT intended for actual medical diagnosis.
    Always consult qualified medical professionals for any health concerns.
    """)

# Main app
st.header("📤 Upload MRI Scan")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an MRI image...",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a brain MRI scan image"
)

# Load model 
model = load_keras_model()

if model is None:
    st.stop() 

# Process uploaded image
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file) 
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        with st.spinner("Analyzing MRI scan..."):
            predicted_class, confidence, all_probs = predict_tumor(image, model)
        
        tumor_name = tumor_types[predicted_class].replace('_', ' ').title()
        
        if tumor_types[predicted_class] == "no_tumor":
            st.success(f"### ✅ {tumor_name}")
        else:
            st.error(f"### ⚠️ {tumor_name}")
        
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(float(confidence))
    
    # Show all probabilities
    st.subheader("📊 Detailed Probabilities")
    
    prob_data = []
    for i, prob in enumerate(all_probs):
        tumor_name = tumor_types[i].replace('_', ' ').title()
        prob_data.append({
            "Tumor Type": tumor_name,
            "Probability": f"{prob:.2%}",
            "Confidence": prob
        })
    
    prob_data = sorted(prob_data, key=lambda x: x['Confidence'], reverse=True)
    
    for item in prob_data:
        col1, col2, col3 = st.columns([3, 2, 5])
        with col1:
            st.write(item["Tumor Type"])
        with col2:
            st.write(item["Probability"])
        with col3:
            st.progress(float(item["Confidence"])) 
    
    st.divider()
    st.info("""
    **How to interpret results:**
    - **High confidence (>90%)**: Model is very certain about its prediction
    - **Medium confidence (70-90%)**: Model is fairly confident
    - **Low confidence (<70%)**: Results should be interpreted with caution
    
    ⚠️ **Remember:** This is a demo model. Real medical decisions require professional evaluation!
    """)

else:
    st.info("👆 Please upload a brain MRI scan image to begin analysis.")
    
    with st.expander("📋 Need help getting started?"):
        st.markdown("""
        **Supported formats:** PNG, JPG, JPEG
        
        **Tips:**
        1. Use clear, high-quality MRI scans
        2. Ensure the image shows a brain cross-section
        3. Avoid images with too much text or annotations
        
        **Testing the model:**
        - You can use images from the validation/test dataset
        - Try different tumor types to see how the model performs
        """)

# Footer
st.divider()
st.caption("Built with ❤️ using Streamlit and TensorFlow | Brain Tumor MRI Classifier v1.0")