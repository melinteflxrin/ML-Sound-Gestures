import librosa
import numpy as np
import os

def extract_mfcc(file_path, n_mfcc=13):
    """
    extract mean MFCC features from an audio file
    """
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def process_folder(folder_path, label):
    """
    process all audio files in a folder and return features and labels
    """
    X = [] # features
    y = [] # labels
    for file in os.listdir(folder_path):
        if file.endswith(".mp3") or file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features = extract_mfcc(file_path)
            X.append(features)
            y.append(label)
    return X, y

def main():
    double_clap_folder = "data/raw/double_clap"
    negative_folder = "data/raw/negative"

    X_clap, y_clap = process_folder(double_clap_folder, 1)
    X_neg, y_neg = process_folder(negative_folder, 0)

    # combine features and labels
    X = np.array(X_clap + X_neg)
    y = np.array(y_clap + y_neg)

    print("Feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)

    # save processed data
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)

if __name__ == "__main__":
    main()
