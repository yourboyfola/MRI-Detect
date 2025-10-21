import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

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

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('brain_tumor_classifier.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Predict function
def predict_tumor(image, model):
    """Make prediction on uploaded image"""
    # Resize image to 224x224
    img = image.resize((224, 224))
    
    # Convert to array
    img_array = np.array(img)
    
    # Handle grayscale images (convert to RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel
    
    # Add batch dimension and preprocess
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    
    return predicted_class, confidence, prediction[0]

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Model Details:**
    - Architecture: EfficientNetB0 (Transfer Learning)
    - Training Data: Brain MRI Scans
    - Validation Accuracy: ~87%
    
    **Developer:** Your Name
    **GitHub:** [Your GitHub Link]
    """)
    
    st.header("⚠️ Disclaimer")
    st.warning("""
    This is a demo model for educational purposes only. 
    NOT intended for actual medical diagnosis.
    Always consult qualified medical professionals.
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
model = load_model()

if model is None:
    st.error("⚠️ Model not found! Make sure 'brain_tumor_classifier.keras' is in the same directory.")
    st.stop()

# Process uploaded image
if uploaded_file is not None:
    # Display columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        # Make prediction
        with st.spinner("Analyzing MRI scan..."):
            predicted_class, confidence, all_probs = predict_tumor(image, model)
        
        # Display results
        tumor_name = tumor_types[predicted_class].replace('_', ' ').title()
        
        # Color code based on prediction
        if tumor_types[predicted_class] == "no_tumor":
            st.success(f"### ✅ {tumor_name}")
        else:
            st.error(f"### ⚠️ {tumor_name}")
        
        # Confidence
        st.metric("Confidence", f"{confidence:.1%}")
        
        # Progress bar for confidence
        st.progress(float(confidence))
    
    # Show all probabilities
    st.subheader("📊 Detailed Probabilities")
    
    # Create a nice table
    prob_data = []
    for i, prob in enumerate(all_probs):
        tumor_name = tumor_types[i].replace('_', ' ').title()
        prob_data.append({
            "Tumor Type": tumor_name,
            "Probability": f"{prob:.2%}",
            "Confidence": prob
        })
    
    # Sort by probability
    prob_data = sorted(prob_data, key=lambda x: x['Confidence'], reverse=True)
    
    # Display as columns with bars
    for item in prob_data:
        col1, col2, col3 = st.columns([3, 2, 5])
        with col1:
            st.write(item["Tumor Type"])
        with col2:
            st.write(item["Probability"])
        with col3:
            st.progress(float(item["Confidence"]))
    
    # Additional info
    st.divider()
    st.info("""
    **How to interpret results:**
    - **High confidence (>90%)**: Model is very certain about its prediction
    - **Medium confidence (70-90%)**: Model is fairly confident
    - **Low confidence (<70%)**: Results should be interpreted with caution
    
    ⚠️ **Remember:** This is a demo model. Real medical decisions require professional evaluation!
    """)

else:
    # Show instructions when no file uploaded
    st.info("👆 Please upload a brain MRI scan image to begin analysis.")
    
    # Optional: Show example images or instructions
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