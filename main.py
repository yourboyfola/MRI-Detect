import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

val_dir = 'data/Brain_Tumor/Testing'
train_dir = 'data/Brain_Tumor/Training'

tumor_types = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

# Image parameters
image_size = (224, 224)  # ← MobileNetV2 works better with 224x224
batch_size = 16  # ← Smaller batch = more updates per epoch = better learning

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int',
    shuffle=True  # ← ADDED: Always shuffle training data!
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int'
)

# Data augmentation (applied BEFORE preprocessing)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),  # ← CHANGED: Medical images shouldn't flip vertically
    layers.RandomRotation(0.15),      # ← CHANGED: Reduced rotation
    layers.RandomZoom(0.15),          # ← CHANGED: Slightly more zoom
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.2)      # ← ADDED: Helps with different MRI contrasts
])

# ← REMOVED the manual rescaling! We'll use MobileNetV2's preprocessor instead

# Apply augmentation to training data only
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Apply MobileNetV2 preprocessing to BOTH train and val
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("Training batches:", len(train_ds))
print("Validation batches:", len(val_ds))

# Build the model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),  # ← CHANGED: Match the image_size
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),      # ← ADDED: Helps stabilize training
    layers.Dense(256, activation='relu'),  # ← CHANGED: More neurons
    layers.Dropout(0.5),              # ← CHANGED: Actually using dropout now!
    layers.Dense(128, activation='relu'),  # ← ADDED: Extra layer for more capacity
    layers.Dropout(0.3),              # ← ADDED: Lighter dropout on second layer
    layers.Dense(len(tumor_types), activation='softmax')
])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # ← CHANGED: Monitor accuracy instead of loss
    patience=5,              # ← CHANGED: More patience
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,              # ← Reduce LR by half when plateauing
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ← CHANGED: Higher learning rate to start
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Starting training (Phase 1: Frozen base)...")
model.summary()

# Train with frozen base
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,  # ← CHANGED: More epochs
    callbacks=[early_stopping, reduce_lr]
)

# PHASE 2: Fine-tuning (unfreeze some layers)
print("\n🔥 Phase 2: Fine-tuning top layers...")
base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # ← Much lower LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stopping, reduce_lr]
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Final Validation Accuracy: {history_fine.history['val_accuracy'][-1]:.2%}")

# Save the model
model.save('brain_tumor_classifier.keras')
print("💾 Model saved!")

# Test prediction function
def predict_tumor(img_path):
    """Predict tumor type from image path"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ← Use proper preprocessing!
    
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    
    print(f"\n🧠 Prediction: {tumor_types[predicted_class]}")
    print(f"📊 Confidence: {confidence:.2%}")
    print("\nAll probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"  {tumor_types[i]}: {prob:.2%}")
    
    return tumor_types[predicted_class], confidence

# Example usage
img_path = r"C:\Users\USER\MRI-Detect\data\Brain_Tumor\Practice\test_pituitary_tumor_image.jpg"
predict_tumor(img_path)