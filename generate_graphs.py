import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

print("Loading Data and Models... (Takes a few seconds)")

# 1. Load Old Data
X_old = np.load("X_data.npy")[..., np.newaxis]
y_old = np.load("y_labels.npy")
_, X_test_old, _, y_test_old = train_test_split(X_old, y_old, test_size=0.2, random_state=42)

# 2. Load Augmented Data
X_new = np.load("X_data_aug.npy")[..., np.newaxis]
y_new = np.load("y_labels_aug.npy")
_, X_test_new, _, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# 3. Load Both Models Safely (Ab koi error nahi aayega)
try:
    model_old = tf.keras.models.load_model('respiro_model.h5')
    model_new = tf.keras.models.load_model('respiro_mobilenet.keras')
except Exception as e:
    print(f"\n⚠️ Asli Error: {e}")
    exit()

print("Generating Predictions...")
y_pred_old = (model_old.predict(X_test_old, verbose=0) > 0.5).astype(int)
y_pred_new = (model_new.predict(X_test_new, verbose=0) > 0.5).astype(int)

acc_old = accuracy_score(y_test_old, y_pred_old) * 100
acc_new = accuracy_score(y_test_new, y_pred_new) * 100

# ==========================================
# GRAPH 1: ACCURACY COMPARISON
# ==========================================
plt.figure(figsize=(8, 6))
bars = plt.bar(['Custom CNN (174 files)', 'MobileNetV2 + Augmentation (696 files)'], 
               [acc_old, acc_new], color=['#ff9999', '#66b3ff'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', fontsize=12, fontweight='bold')

plt.title('Model Accuracy Comparison: Scratch vs Transfer Learning', fontsize=14, fontweight='bold')
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.ylim(0, 100)
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: accuracy_comparison.png")

# ==========================================
# GRAPH 2: CONFUSION MATRIX
# ==========================================
cm_old = confusion_matrix(y_test_old, y_pred_old)
cm_new = confusion_matrix(y_test_new, y_pred_new)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = ['Healthy', 'Unhealthy']

sns.heatmap(cm_old, annot=True, fmt='d', cmap='Reds', ax=axes[0], xticklabels=labels, yticklabels=labels)
axes[0].set_title(f'Old Custom CNN\n(Accuracy: {acc_old:.1f}%)', fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues', ax=axes[1], xticklabels=labels, yticklabels=labels)
axes[1].set_title(f'MobileNetV2 (Transfer Learning)\n(Accuracy: {acc_new:.1f}%)', fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrix_comparison.png")
print("\nDone! Dono graphs tumhare folder mein save ho gaye hain.")