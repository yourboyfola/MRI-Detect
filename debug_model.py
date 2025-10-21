import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. SETUP: Must match your training script ---

val_dir = 'data/Brain_Tumors/Testing'
tumor_types = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
image_size = (224, 224)
batch_size = 16 # Keep batch size small for analysis if needed

# --- 2. LOAD THE SAVED MODEL ---

print("💾 Loading saved model...")
model = tf.keras.models.load_model('brain_tumor_classifier.keras')
print("✅ Model loaded successfully!")

# --- 3. RECREATE THE VALIDATION DATASET ---
# This part MUST be identical to your training script's validation loader

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int',
    shuffle=False # No need to shuffle for analysis
)

# Apply the same preprocessing
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# --- 4. PASTE THE ANALYSIS CODE HERE ---

print("\n🔍 Analyzing model's worst validation errors...")

# Unbatch the dataset to process one image at a time
val_ds = val_ds.unbatch() 

errors = []
for i, (image, label) in enumerate(val_ds):
    image_for_pred = np.expand_dims(image, axis=0)
    prediction = model.predict(image_for_pred, verbose=0)
    
    predicted_class = np.argmax(prediction[0])
    true_class = label.numpy()
    
    if predicted_class != true_class:
        confidence = prediction[0][predicted_class]
        errors.append((image.numpy(), true_class, predicted_class, confidence))

# Sort errors by the model's confidence in its wrong answer
errors.sort(key=lambda x: x[3], reverse=True) 

# Display the top N worst errors
print(f"Found {len(errors)} errors. Displaying the top 5 most confident mistakes:")
plt.figure(figsize=(15, 10))
for i in range(min(5, len(errors))):
    image, true_label, pred_label, conf = errors[i]
    
    # De-process the image for correct visualization
    img_to_show = (image - image.min()) / (image.max() - image.min())

    plt.subplot(2, 3, i + 1)
    plt.imshow(img_to_show)
    plt.title(f"True: {tumor_types[true_label]}\nPred: {tumor_types[pred_label]} ({conf:.2%})")
    plt.axis('off')

plt.tight_layout()
plt.savefig('worst_errors.png')
plt.show()

# You can also add the confusion matrix code here...
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("\n📊 Generating confusion matrix...")

# Get all true labels and predictions from the validation set
# NOTE: We can reuse the unbatched val_ds from the error analysis
y_true = []
y_pred = []

# This loop collects predictions for ALL validation images
for image, label in val_ds:
    image_for_pred = np.expand_dims(image, axis=0)
    prediction = model.predict(image_for_pred, verbose=0)
    
    y_true.append(label.numpy())
    y_pred.append(np.argmax(prediction[0]))

# Create the confusion matrix using scikit-learn
cm = confusion_matrix(y_true, y_pred)

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True,          # Show the numbers in each cell
    fmt='d',             # Format the numbers as integers
    cmap='Blues',        # Use a blue color scheme
    xticklabels=tumor_types, 
    yticklabels=tumor_types
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)
plt.savefig('confusion_matrix.png')
plt.show()