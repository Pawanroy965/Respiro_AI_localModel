import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1. Load Data
X = np.load("X_data.npy")
y = np.load("y_labels.npy")

# CNN ko input me ek 'channel' dimension chahiye (jaise image me RGB ke liye 3, grayscale ke liye 1)
# Humari current shape: (174, 13, 40). Hum isko (174, 13, 40, 1) banayenge.
X = X[..., np.newaxis] 

# 2. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("AI Model ka structure ban raha hai...")

# 3. Model Architecture (Mobile ke liye Lightweight CNN)
model = models.Sequential([
    # Pehla Convolution Layer patterns pakadne ke liye
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    layers.MaxPooling2D((2, 2)), 
    
    # Dusra Convolution Layer deeper patterns ke liye
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    
    # 2D data ko 1D me convert karo taaki final output nikal sake
    layers.Flatten(),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), # Model ko 'ratta' (overfitting) maarne se rokne ke liye
    
    # Final Layer: Binary output - 0 (Healthy) ya 1 (Unhealthy)
    layers.Dense(1, activation='sigmoid') 
])

# 4. Model Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\nModel Training Start ho rahi hai...")

# 5. Model Train Karo (20 Epochs)
history = model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# 6. Evaluation
print("\n--- Final Results ---")
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=0)
print(f"✅ Testing Accuracy: {test_acc * 100:.2f}%")

# 7. Model Save Karo
model.save("respiro_model.h5")
print("\nModel 'respiro_model.h5' ke naam se save ho gaya hai!")