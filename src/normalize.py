import os
import soundfile as sf
import numpy as np
import librosa


input_folder = "data/raw/finger_snap"  
output_folder = "data/normalized/finger_snap"  
os.makedirs(output_folder, exist_ok=True)


SUPPORTED_EXTENSIONS = (".wav", ".mp3")


def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio

for filename in os.listdir(input_folder):
    if filename.lower().endswith(SUPPORTED_EXTENSIONS):
        in_path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".wav")
        if filename.lower().endswith(".wav"):
            audio, sr = sf.read(in_path)
        else:  # mp3
            audio, sr = librosa.load(in_path, sr=None, mono=True)
        audio_norm = normalize_audio(audio)
        sf.write(out_path, audio_norm, sr)
        print(f"Normalized: {filename} -> {os.path.basename(out_path)}")


print("All files normalized.")
