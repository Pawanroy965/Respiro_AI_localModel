import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

# 1. Load Data
X = np.load("X_data_aug.npy")[..., np.newaxis]
y = np.load("y_labels_aug.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Downloading MobileNetV2 Brain...")

# 2. Pre-trained Model
base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# 3. BULLETPROOF ARCHITECTURE (Bina Lambda layer ke)
inputs = layers.Input(shape=(13, 40, 1))

# Resize
x = layers.Resizing(96, 96)(inputs)

# NAYA MAGIC TRICK: 1 channel ki 3 copies aapas mein jodh do (No Lambda needed!)
x = layers.Concatenate(axis=-1)([x, x, x])

# Transfer Learning Model
x = base_model(x)

# Custom Head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

# Model Pack Karo
model = models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining Advanced Model (Final and Bulletproof)...")
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Naye .keras format me save
model.save("respiro_mobilenet.keras")
print("\nModel 'respiro_mobilenet.keras' successfully saved!")