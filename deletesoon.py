import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('brain_tumor_classifier.keras')

print("✅ Model loaded!")

# Define your classes (same order as training!)
tumor_types = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

# Prediction function
def predict_tumor(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    
    # Show results
    print(f"\n🧠 Prediction: {tumor_types[predicted_class]}")
    print(f"📊 Confidence: {confidence:.2%}")
    print("\nAll probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"  {tumor_types[i]}: {prob:.2%}")
    
    return tumor_types[predicted_class], confidence

# Test on a new image!
img_path = r"C:\Users\USER\MRI-Detect\data\Brain_Tumor\Practice\randomglioma.jpg"
predict_tumor(img_path)