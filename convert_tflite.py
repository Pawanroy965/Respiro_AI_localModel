import tensorflow as tf

print("Loading saved .h5 model...")
# 1. Purana model load karo
model = tf.keras.models.load_model('respiro_model.h5')

print("Converting to TFLite format...")
# 2. Converter initialize karo
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Nayi compressed file save karo
with open('respiro_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Success! 'respiro_model.tflite' ban gayi hai. Ab yeh React Native app me jaane ke liye ready hai!")