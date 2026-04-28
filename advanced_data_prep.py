import os
import librosa
import numpy as np

DATASET_PATH = "Dataset"
LABELS = ["Healthy", "Unhealthy"]
MAX_PAD_LENGTH = 40

X, y = [], []

def process_and_add(audio, sr, label_idx):
    """Helper function to extract MFCC and add to our dataset"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    # Pad or crop to 13x40
    if mfcc.shape[1] > MAX_PAD_LENGTH:
        mfcc = mfcc[:, :MAX_PAD_LENGTH]
    else:
        pad_width = MAX_PAD_LENGTH - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    X.append(mfcc)
    y.append(label_idx)

print("Starting Advanced Data Augmentation... Grab a coffee, this will take a minute.")

for label_idx, label_name in enumerate(LABELS):
    folder_path = os.path.join(DATASET_PATH, label_name)
    if not os.path.exists(folder_path): continue

    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.wav', '.mp3', '.m4a', '.flac')):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Load and trim base audio
                audio, sr = librosa.load(file_path, sr=16000)
                audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
                
                # 1. Original
                process_and_add(audio_trimmed, sr, label_idx)
                
                # 2. Add White Noise
                noise = np.random.randn(len(audio_trimmed))
                audio_noisy = audio_trimmed + 0.005 * noise
                process_and_add(audio_noisy, sr, label_idx)
                
                # 3. Fast (Speed up by 10%)
                audio_fast = librosa.effects.time_stretch(y=audio_trimmed, rate=1.1)
                process_and_add(audio_fast, sr, label_idx)
                
                # 4. Slow (Slow down by 10%)
                audio_slow = librosa.effects.time_stretch(y=audio_trimmed, rate=0.9)
                process_and_add(audio_slow, sr, label_idx)

            except Exception as e:
                pass # Skip corrupted files silently

X = np.array(X)
y = np.array(y)

print(f"\n✅ Augmentation Complete! Data expanded to {len(X)} files.")
np.save("X_data_aug.npy", X)
np.save("y_labels_aug.npy", y)