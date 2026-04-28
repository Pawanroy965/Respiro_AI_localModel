import os
import librosa
import numpy as np

# Tumhare VS Code structure ke exact paths
DATASET_PATH = "Dataset"
LABELS = ["Healthy", "Unhealthy"] # Dhyan rakhna H aur U capital hain

# AI model ko har audio ek hi length ki chahiye. 
# 40 frames ka matlab roughly 1-1.5 seconds ki cough sound hai.
MAX_PAD_LENGTH = 40 

X = [] # Isme humari MFCC 'images' (data) aayengi
y = [] # Isme labels aayenge (0 for Healthy, 1 for Unhealthy)

print("Audio preprocessing start ho rahi hai... System ko thoda time lag sakta hai.")

for label_idx, label_name in enumerate(LABELS):
    folder_path = os.path.join(DATASET_PATH, label_name)
    
    # Agar folder nahi mila toh script break na ho
    if not os.path.exists(folder_path):
        print(f"⚠️ Error: Folder nahi mila -> {folder_path}")
        continue

    print(f"Processing folder: {label_name}...")

    for file_name in os.listdir(folder_path):
        # Multiple formats allow kar diye hain in case .wav na ho
        if file_name.endswith(('.wav', '.mp3', '.m4a', '.flac')):
            file_path = os.path.join(folder_path, file_name)
            
            try:
                # 1. Load Audio (strictly at 16kHz)
                audio, sr = librosa.load(file_path, sr=16000)
                
                # 2. Trim Silence (Khali awaaz ko start/end se kaatna)
                audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
                
                # 3. Extract MFCCs
                mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=sr, n_mfcc=13)
                
                # 4. PADDING: Sabko exactly 13x40 size ka banana
                if mfcc.shape[1] > MAX_PAD_LENGTH:
                    mfcc = mfcc[:, :MAX_PAD_LENGTH] # Lambi file ko cut karo
                else:
                    pad_width = MAX_PAD_LENGTH - mfcc.shape[1]
                    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant') # Choti file mein zeros lagao
                
                X.append(mfcc)
                y.append(label_idx) # 0 ya 1
                
            except Exception as e:
                print(f"Error in file {file_name}: {e}")

# Lists ko Numpy arrays mein convert karna zaroori hai AI training ke liye
X = np.array(X)
y = np.array(y)

print("\n--- Processing Done! ---")
print(f"Total valid audio files processed: {len(X)}")
print(f"Data shape: {X.shape}") # (Total_Files, 13, 40) aana chahiye

# 5. Processed data ko save kar lo
np.save("X_data.npy", X)
np.save("y_labels.npy", y)
print("✅ Success! Data 'X_data.npy' aur 'y_labels.npy' mein save ho gaya hai.")