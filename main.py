import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

val_dir = 'data/Brain_Tumors/Testing'
train_dir = 'data/Brain_Tumors/Training'

tumor_types = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]

# Image parameters
image_size = (224, 224) 
batch_size = 16  

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int',
    shuffle=True  
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int'
)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),        
    layers.RandomContrast(0.3),
    layers.RandomZoom(0.15),           
    layers.RandomTranslation(0.1, 0.1), 
    layers.RandomBrightness(0.2)      
])

# Applying augmentation to training data only
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Applying preprocessing to BOTH train and val
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("Training batches:", len(train_ds))
print("Validation batches:", len(val_ds))

# Using MobileNetV2 as the base model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),      
    layers.Dense(256, activation='relu'), 
    layers.Dropout(0.5),              
    layers.Dense(128, activation='relu'),  
    layers.Dropout(0.3),              
    layers.Dense(len(tumor_types), activation='softmax')
])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  
    patience=5,             
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,             
    patience=3,
    min_lr=1e-7,
    verbose=1
)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Starting training (Phase 1: Frozen base)...")
model.summary()


# Get all labels from training data
train_labels = np.concatenate([y for x, y in train_ds], axis=0)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model_backup.keras',  # ← Change to .keras
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Use in training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)
# PHASE 2: Fine-tuning (unfreeze some layers)
print("\n🔥 Phase 2: Fine-tuning top layers...")
base_model.trainable = True

# Freeze all layers except the last 50
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompiling with lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-tune
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[early_stopping, reduce_lr, checkpoint],
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

print("\n💾 Saving model...")
model.save('brain_tumor_final.keras')
print("✅ Model saved as brain_tumor_final.keras")

def predict_tumor(img_path):
    """Predict tumor type from image path"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  
    
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    
    print(f"\n🧠 Prediction: {tumor_types[predicted_class]}")
    print(f"📊 Confidence: {confidence:.2%}")
    print("\nAll probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"  {tumor_types[i]}: {prob:.2%}")
    
    return tumor_types[predicted_class], confidence
